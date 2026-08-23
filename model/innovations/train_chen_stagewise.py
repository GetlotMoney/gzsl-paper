from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.train_chen_style import (
    INPUT_KEYS,
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.innovations.unified_seen import UnifiedSeenPrototypeModel
from model.tg_vpr_h1 import train as h1
from tools.cub_data import load_cub_split
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
    require_finite_gradients,
    require_finite_model,
)
from tools.runtime import sha256_file


EVALUATION_PROTOCOL = "chen_shiming_code_aligned_stagewise_test_selected_gzsl"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "condition_id",
    "framework_id",
    "dataset",
    "evaluation_protocol",
    "test_used_for_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "training_strategy",
    "selection_scope",
    "nested_official_test_selection",
    "feature_backbone",
    "feature_provenance_complete",
    "device",
    "random_seed",
    "batch_size",
    "epochs",
    "niters",
    "report_interval",
    "optimizer",
    "weight_decay",
    "dropout",
    "inner_ratio",
    "outer_ratio",
    "topology_weight",
    "temperature",
    "transport_hidden_dim",
    "generator_hidden_dim",
    "max_transport_step",
    "max_generator_magnitude",
    "stages",
    "inputs",
    "expected_sha256",
    "class_order_sha256",
}


def load_config(path: Path) -> tuple[dict, str]:
    path = h1.repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Chen-style分阶段配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"分阶段配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if config["schema_version"] != "gzsl-paper.chen-stagewise.v1":
        raise ValueError("分阶段配置schema错误。")
    if config["experiment_id"] != "V2-CONFIRM-005" or config["condition_id"] != "NO-EXPERT":
        raise ValueError("首次分阶段实验固定V2-CONFIRM-005/NO-EXPERT。")
    if config["framework_id"] != "FRAMEWORK-V2" or config["dataset"] != "CUB":
        raise ValueError("分阶段训练只接受FRAMEWORK-V2/CUB。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("分阶段评估协议身份错误。")
    required = {
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
        "nested_official_test_selection": False,
    }
    for key, expected in required.items():
        if config[key] is not expected:
            raise ValueError(f"分阶段边界错误：{key}必须为{expected}。")
    if config["training_strategy"] != "stagewise_fixed_boundaries_then_joint":
        raise ValueError("分阶段训练策略身份错误。")
    if config["selection_scope"] != "whole_run_whole_model_only":
        raise ValueError("分阶段只允许整次RUN的整模型全局best。")
    if int(config["batch_size"]) != 50 or int(config["epochs"]) != 200:
        raise ValueError("分阶段固定batch 50和200名义epoch。")
    if int(config["niters"]) != 28228 or int(config["report_interval"]) != 141:
        raise ValueError("分阶段niters/report_interval错误。")
    if config["optimizer"] != "Adam":
        raise ValueError("分阶段固定Adam。")
    expected_stages = [
        {"name": "TG_ONLY", "start": 0, "end": 7050, "lr": 0.0001},
        {"name": "TRANSFER_CCGR", "start": 7050, "end": 21150, "lr": 0.0001},
        {"name": "JOINT_FINETUNE", "start": 21150, "end": 28228, "lr": 0.00001},
    ]
    if config["stages"] != expected_stages:
        raise ValueError("分阶段边界必须固定50/100/50名义epoch。")
    if set(config["inputs"]) != set(INPUT_KEYS) or set(config["expected_sha256"]) != set(INPUT_KEYS):
        raise ValueError("分阶段输入或SHA不完整。")
    return config, sha256_file(path)


def stage_for_iteration(config: dict, iteration: int) -> dict:
    for stage in config["stages"]:
        if int(stage["start"]) <= iteration < int(stage["end"]):
            return stage
    raise ValueError(f"iteration {iteration}不属于任何阶段。")


def set_trainable_stage(model: UnifiedSeenPrototypeModel, stage_name: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage_name == "TG_ONLY":
        active = list(model.tg_vpr.parameters())
    elif stage_name == "TRANSFER_CCGR":
        active = (
            list(model.transport_trunk.parameters())
            + list(model.transport_head.parameters())
            + list(model.generator_trunk.parameters())
            + list(model.generator_weight_head.parameters())
            + list(model.generator_magnitude_head.parameters())
        )
    elif stage_name == "JOINT_FINETUNE":
        active = list(model.parameters())
    else:
        raise ValueError(f"未知训练阶段：{stage_name}")
    for parameter in active:
        parameter.requires_grad_(True)
    return active


def gradient_group_norms(model: UnifiedSeenPrototypeModel) -> dict[str, float]:
    groups = {
        "tg_vpr": model.tg_vpr.parameters(),
        "transport": list(model.transport_trunk.parameters()) + list(model.transport_head.parameters()),
        "generator": list(model.generator_trunk.parameters())
        + list(model.generator_weight_head.parameters())
        + list(model.generator_magnitude_head.parameters()),
    }
    result = {}
    for name, parameters in groups.items():
        values = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
        result[name] = float(torch.stack(values).norm()) if values else 0.0
    return result


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config)
    input_sha = verify_inputs(config, paths)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("分阶段训练要求可见CUDA。")
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = h1.TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(
            seed, strict_determinism=True, deterministic_warn_only=False
        )
        sentence_embeds = torch.load(paths["sentence_embeds"], map_location="cpu", weights_only=True)
        train_features = torch.load(paths["train_features"], map_location="cpu", weights_only=True)
        train_labels = torch.load(paths["train_labels"], map_location="cpu", weights_only=True).long()
        official = {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in OFFICIAL_KEYS
        }
        seenclasses = torch.unique(train_labels, sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], train_labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seenclasses) or not torch.equal(checked_unseen, unseenclasses):
            raise RuntimeError("分阶段official split不一致。")
        centroids = h1.visual_centroids(train_features, train_labels, seenclasses)
        model = UnifiedSeenPrototypeModel(
            sentence_embeds, seenclasses, centroids, active_classes=allclasses,
            dropout=float(config["dropout"]), inner_ratio=float(config["inner_ratio"]),
            outer_ratio=float(config["outer_ratio"]), temperature=float(config["temperature"]),
            transport_hidden_dim=int(config["transport_hidden_dim"]),
            generator_hidden_dim=int(config["generator_hidden_dim"]),
            max_transport_step=float(config["max_transport_step"]),
            max_generator_magnitude=float(config["max_generator_magnitude"]),
        ).to(device)
        global_to_seen = torch.full((200,), -1, dtype=torch.long)
        global_to_seen[seenclasses] = torch.arange(150)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        report_interval = int(config["report_interval"])
        best_h = float("-inf")
        best_metrics = best_state = best_iteration = best_nominal_epoch = best_stage = None
        best_zs_observation = float("-inf")
        history = []
        stage_gradient_norms = {}
        current_stage = None
        optimizer = None
        print(f"实验：{config['experiment_id']} RUN={run_id} stagewise=true")
        print(f"代码commit：{code_commit} 配置SHA：{config_sha}")

        for iteration in range(int(config["niters"])):
            stage = stage_for_iteration(config, iteration)
            if stage["name"] != current_stage:
                current_stage = stage["name"]
                active_parameters = set_trainable_stage(model, current_stage)
                optimizer = torch.optim.Adam(
                    active_parameters,
                    lr=float(stage["lr"]),
                    weight_decay=float(config["weight_decay"]),
                )
                print(f"stage={current_stage} start={iteration} lr={stage['lr']}")
            model.train()
            indices = random_batch_indices(train_labels.numel(), int(config["batch_size"]), generator)
            images = train_features.index_select(0, indices).to(device).float()
            targets = global_to_seen[train_labels.index_select(0, indices)].to(device)
            optimizer.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model.logits(images, seenclasses), targets)
            topology = model.topology_loss()
            loss = ce + float(config["topology_weight"]) * topology
            if not torch.isfinite(loss):
                raise FloatingPointError("分阶段loss包含NaN/Inf。")
            loss.backward()
            require_finite_gradients(model)
            if iteration == int(stage["start"]):
                norms = gradient_group_norms(model)
                stage_gradient_norms[current_stage] = norms
                expected_active = {
                    "TG_ONLY": ("tg_vpr",),
                    "TRANSFER_CCGR": ("transport", "generator"),
                    "JOINT_FINETUNE": ("tg_vpr", "transport", "generator"),
                }[current_stage]
                if any(norms[name] <= 0.0 for name in expected_active):
                    raise RuntimeError(f"阶段{current_stage}活动模块梯度必须非零。")
            optimizer.step()

            if iteration % report_interval == 0:
                metrics = h1.evaluate(model, official, seenclasses, unseenclasses, device)
                nominal_epoch = iteration // report_interval
                row = {
                    "iteration": iteration,
                    "nominal_epoch": nominal_epoch,
                    "stage": current_stage,
                    "train_loss": float(loss.detach()),
                    "official_metrics_percent": metrics,
                    "diagnostics": model.diagnostics(),
                }
                history.append(row)
                best_zs_observation = max(best_zs_observation, metrics["ZS"])
                if metrics["H"] > best_h:
                    best_h = metrics["H"]
                    best_metrics = metrics
                    best_iteration = iteration
                    best_nominal_epoch = nominal_epoch
                    best_stage = current_stage
                    best_state = copy.deepcopy(model.state_dict())
                    atomic_torch_save(
                        output_dir / "model_best.pth",
                        {
                            "experiment_id": config["experiment_id"],
                            "run_id": run_id,
                            "code_commit": code_commit,
                            "config": config,
                            "config_sha256": config_sha,
                            "selected_iteration": best_iteration,
                            "selected_nominal_epoch": best_nominal_epoch,
                            "selected_stage": best_stage,
                            "best_metrics_percent": best_metrics,
                            "model_state_dict": best_state,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} epoch={nominal_epoch} stage={current_stage} "
                    f"U={metrics['U']:.6f} S={metrics['S']:.6f} H={metrics['H']:.6f} best_H={best_h:.6f}"
                )

        require_finite_model(model)
        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "experiment_id": config["experiment_id"],
                "run_id": run_id,
                "code_commit": code_commit,
                "config": config,
                "config_sha256": config_sha,
                "last_iteration": int(config["niters"]) - 1,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_model_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "selected_nominal_epoch": best_nominal_epoch,
                "selected_stage": best_stage,
                "history": history,
                "stage_gradient_norms": stage_gradient_norms,
                "reproducibility": reproducibility,
            },
        )
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        payload = {
            "experiment_id": config["experiment_id"],
            "run_id": run_id,
            "framework_id": config["framework_id"],
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "training_strategy": config["training_strategy"],
            "selection_scope": config["selection_scope"],
            "nested_official_test_selection": False,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "seed": seed,
            "niters": int(config["niters"]),
            "report_interval": report_interval,
            "official_test_evaluation_count": len(history),
            "selected_iteration": best_iteration,
            "selected_nominal_epoch": best_nominal_epoch,
            "selected_stage": best_stage,
            "best_metrics_percent": best_metrics,
            "best_zs_observation_percent": best_zs_observation,
            "stage_gradient_norms": stage_gradient_norms,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", payload)
        print({"best": best_metrics, "iteration": best_iteration, "stage": best_stage})
        return best_metrics
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.run_id)


if __name__ == "__main__":
    main()
