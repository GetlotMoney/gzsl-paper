"""Unified three-dataset Chen-style runner for the final FRAMEWORK-V2 paper matrix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.paper_v2 import CCGR_MODES, TG_MODES, TRANSPORT_MODES, PaperV2ThreeModuleModel
from model.tg_vpr_h1 import train as h1
from tools.gzsl_data import evaluate_prototypes
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


EVALUATION_PROTOCOL = "chen_shiming_code_aligned_multidataset_test_selected_gzsl"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "condition_id",
    "framework_id",
    "dataset",
    "asset_manifest",
    "asset_manifest_sha256",
    "evaluation_protocol",
    "test_used_for_selection",
    "test_used_for_hyperparameter_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "training_strategy",
    "selection_scope",
    "nested_official_test_selection",
    "device",
    "random_seed",
    "batch_size",
    "nominal_epochs",
    "optimizer",
    "weight_decay",
    "end_to_end_learning_rate",
    "stage1_learning_rate",
    "stage2_learning_rate",
    "stage3_learning_rate",
    "tg_vpr_mode",
    "transport_mode",
    "ccgr_mode",
    "dropout",
    "inner_ratio",
    "outer_ratio",
    "topology_weight",
    "temperature",
    "transport_hidden_dim",
    "generator_hidden_dim",
    "max_transport_step",
    "max_ntr_delta",
    "max_generator_magnitude",
}
ASSET_FILES = (
    "train_features.pt",
    "train_labels.pt",
    "test_seen_features.pt",
    "test_seen_labels.pt",
    "test_unseen_features.pt",
    "test_unseen_labels.pt",
    "class_name_embeds.pt",
    "role_sentence_embeds.pt",
)


def load_config(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"最终论文RUN配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"最终论文RUN配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if config["schema_version"] != "gzsl-paper.paper-v2-run.v1":
        raise ValueError("最终论文RUN schema错误。")
    if config["framework_id"] != "FRAMEWORK-V2" or config["dataset"] not in ("CUB", "AWA2", "SUN"):
        raise ValueError("最终论文RUN只接受FRAMEWORK-V2和CUB/AWA2/SUN。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("最终论文RUN评估协议身份错误。")
    if config["unseen_images_used_for_gradient"] is not False or config["strict_blind_claim"] is not False:
        raise ValueError("必须披露true-unseen无梯度且不作blind-test声明。")
    if config["nested_official_test_selection"] is not False:
        raise ValueError("分阶段RUN禁止嵌套official-test选模。")
    if config["selection_scope"] != "whole_run_whole_model_only":
        raise ValueError("只允许整次RUN的整模型全局H选模。")
    if config["training_strategy"] not in ("no_training", "end_to_end_joint", "stagewise_50_100_50"):
        raise ValueError("未知训练策略。")
    no_training = config["training_strategy"] == "no_training"
    if no_training:
        if config["test_used_for_selection"] is not False:
            raise ValueError("no_training条件没有checkpoint选择。")
    elif config["test_used_for_selection"] is not True:
        raise ValueError("训练RUN固定使用official test H选checkpoint。")
    if int(config["batch_size"]) != 50 or int(config["nominal_epochs"]) != 200:
        raise ValueError("Chen-style固定batch=50、200名义epoch。")
    if config["optimizer"] != "Adam" or float(config["weight_decay"]) != 1e-4:
        raise ValueError("最终论文RUN固定Adam和weight_decay=1e-4。")
    if config["tg_vpr_mode"] not in TG_MODES:
        raise ValueError("tg_vpr_mode错误。")
    if config["transport_mode"] not in TRANSPORT_MODES:
        raise ValueError("transport_mode错误。")
    if config["ccgr_mode"] not in CCGR_MODES:
        raise ValueError("ccgr_mode错误。")
    return config, sha256_file(path)


def load_assets(config: dict) -> tuple[dict, dict, Path]:
    manifest_path = Path(config["asset_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"资产manifest不存在：{manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["asset_manifest_sha256"]:
        raise ValueError("资产manifest SHA不匹配。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "gzsl-paper.clip-assets.v1" or manifest.get("dataset") != config["dataset"]:
        raise ValueError("资产manifest身份错误。")
    expected_outputs = manifest.get("outputs_sha256", {})
    if not set(ASSET_FILES).issubset(expected_outputs):
        raise ValueError("资产manifest缺少训练或评估缓存。")
    tensors = {}
    for filename in ASSET_FILES:
        path = manifest_path.parent / filename
        if not path.is_file() or sha256_file(path) != expected_outputs[filename]:
            raise ValueError(f"资产文件缺失或SHA错误：{filename}")
        tensors[filename.removesuffix(".pt")] = torch.load(path, map_location="cpu", weights_only=True)
    class_count = int(manifest["class_count"])
    expected_shapes = {
        "train_features": (int(manifest["train_count"]), 768),
        "train_labels": (int(manifest["train_count"]),),
        "test_seen_features": (int(manifest["test_seen_count"]), 768),
        "test_seen_labels": (int(manifest["test_seen_count"]),),
        "test_unseen_features": (int(manifest["test_unseen_count"]), 768),
        "test_unseen_labels": (int(manifest["test_unseen_count"]),),
        "class_name_embeds": (class_count, 768),
        "role_sentence_embeds": (class_count, 8, 768),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"资产{name}形状错误：{tuple(tensors[name].shape)} != {shape}")
    return tensors, manifest, manifest_path


def random_batch_indices(count: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(int(count), generator=generator)[: int(batch_size)]


def build_three_module_model(
    config: dict,
    tensors: dict,
    manifest: dict,
    device: torch.device,
    *,
    dropout_override: float | None = None,
) -> PaperV2ThreeModuleModel:
    seen_classes = torch.tensor(manifest["seen_classes"], dtype=torch.long)
    centroids = h1.visual_centroids(
        tensors["train_features"], tensors["train_labels"].long(), seen_classes
    )
    return PaperV2ThreeModuleModel(
        tensors["role_sentence_embeds"],
        seen_classes,
        centroids,
        tg_vpr_mode=config["tg_vpr_mode"],
        transport_mode=config["transport_mode"],
        ccgr_mode=config["ccgr_mode"],
        dropout=(float(config["dropout"]) if dropout_override is None else float(dropout_override)),
        inner_ratio=float(config["inner_ratio"]),
        outer_ratio=float(config["outer_ratio"]),
        temperature=float(config["temperature"]),
        transport_hidden_dim=int(config["transport_hidden_dim"]),
        generator_hidden_dim=int(config["generator_hidden_dim"]),
        max_transport_step=float(config["max_transport_step"]),
        max_ntr_delta=float(config["max_ntr_delta"]),
        max_generator_magnitude=float(config["max_generator_magnitude"]),
    ).to(device)


def stage_for_iteration(ntrain: int, iteration: int) -> tuple[str, float]:
    if 0 <= iteration < ntrain:
        return "TG_ONLY", 0.25
    if ntrain <= iteration < 3 * ntrain:
        return "TRANSFER_CCGR", 0.50
    if 3 * ntrain <= iteration < 4 * ntrain:
        return "JOINT_FINETUNE", 0.25
    raise ValueError("iteration不属于50/100/50阶段。")


def _active_groups(model: PaperV2ThreeModuleModel, strategy: str, stage: str) -> list[str]:
    groups = model.parameter_groups()
    nonempty = {name for name, parameters in groups.items() if parameters}
    if strategy == "end_to_end_joint":
        return sorted(nonempty)
    if stage == "TG_ONLY":
        selected = {"tg_vpr"}
    elif stage == "TRANSFER_CCGR":
        selected = {"transport", "ntr", "ccgr_class", "ccgr_shared"}
        if not (selected & nonempty):
            selected = {"tg_vpr"}
    elif stage == "JOINT_FINETUNE":
        selected = nonempty
    else:
        raise ValueError(f"未知阶段：{stage}")
    return sorted(selected & nonempty)


def set_trainable(model: PaperV2ThreeModuleModel, group_names: list[str]) -> list[torch.nn.Parameter]:
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = model.parameter_groups()
    active = []
    seen_ids = set()
    for name in group_names:
        for parameter in groups[name]:
            if id(parameter) not in seen_ids:
                parameter.requires_grad_(True)
                active.append(parameter)
                seen_ids.add(id(parameter))
    if not active:
        raise ValueError("当前训练阶段没有可训练参数。")
    return active


def _gradient_norms(model: PaperV2ThreeModuleModel) -> dict[str, float]:
    result = {}
    for name, parameters in model.parameter_groups().items():
        values = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
        result[name] = float(torch.stack(values).norm()) if values else 0.0
    return result


def _evaluate_model(model, tensors, manifest, device) -> dict[str, float]:
    if isinstance(model, PaperV2ThreeModuleModel):
        model.eval()
        prototypes = model.prototypes()
        scale = model.scale()
    else:
        prototypes, scale = model
    return evaluate_prototypes(
        prototypes,
        scale,
        tensors["test_seen_features"],
        tensors["test_seen_labels"],
        tensors["test_unseen_features"],
        tensors["test_unseen_labels"],
        torch.tensor(manifest["seen_classes"]),
        torch.tensor(manifest["unseen_classes"]),
        device=device,
    )


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    config, config_sha = load_config(config_path)
    tensors, manifest, manifest_path = load_assets(config)
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = h1.TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(seed, strict_determinism=True, deterministic_warn_only=False)
        device = torch.device(config["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("正式论文RUN要求CUDA。")
        seen_classes = torch.tensor(manifest["seen_classes"], dtype=torch.long)
        train_labels = tensors["train_labels"].long()
        if not torch.equal(torch.unique(train_labels, sorted=True), seen_classes):
            raise ValueError("训练缓存seen类别与资产manifest不一致。")

        if config["training_strategy"] == "no_training":
            if config["condition_id"] == "B0_PURE_CLIP":
                prototypes = tensors["class_name_embeds"].to(device)
                scale = torch.tensor(1.0 / float(config["temperature"]), device=device)
            elif config["condition_id"] == "B1_MEAN8":
                prototypes = F.normalize(tensors["role_sentence_embeds"].mean(dim=1), dim=-1).to(device)
                scale = torch.tensor(1.0 / float(config["temperature"]), device=device)
            else:
                frozen_model = build_three_module_model(
                    config, tensors, manifest, device, dropout_override=0.0
                ).eval()
                if any(frozen_model.parameter_groups().values()):
                    raise ValueError("no_training内部消融仍含活动模块参数。")
                prototypes = frozen_model.prototypes()
                scale = frozen_model.scale()
            metrics = _evaluate_model((prototypes, scale), tensors, manifest, device)
            state = {
                "experiment_id": config["experiment_id"],
                "condition_id": config["condition_id"],
                "run_id": run_id,
                "code_commit": code_commit,
                "config": config,
                "config_sha256": config_sha,
                "best_metrics_percent": metrics,
                "prototypes": prototypes.cpu(),
                "scale": scale.cpu(),
            }
            atomic_torch_save(output_dir / "model_best.pth", state)
            atomic_torch_save(output_dir / "checkpoint_last.pth", state)
            history = [{"iteration": None, "nominal_epoch": None, "stage": "NO_TRAINING", "official_metrics_percent": metrics}]
            selected = {"iteration": None, "epoch": None, "stage": "NO_TRAINING", "metrics": metrics, "diagnostics": {}}
            stage_gradient_norms = {}
        else:
            model = build_three_module_model(config, tensors, manifest, device)
            ntrain = int(train_labels.numel())
            niters = ntrain * int(config["nominal_epochs"]) // int(config["batch_size"])
            if niters != 4 * ntrain:
                raise ValueError("200名义epoch/batch50应严格得到4*ntrain次更新。")
            report_interval = niters // 200
            if report_interval <= 0:
                raise ValueError("report_interval必须为正数。")
            global_to_seen = torch.full((int(manifest["class_count"]),), -1, dtype=torch.long)
            global_to_seen[seen_classes] = torch.arange(seen_classes.numel())
            generator = torch.Generator(device="cpu").manual_seed(seed)
            current_stage = None
            optimizer = None
            history = []
            best_h = float("-inf")
            best_state = None
            selected = None
            best_zs = float("-inf")
            stage_gradient_norms = {}
            for iteration in range(niters):
                if config["training_strategy"] == "end_to_end_joint":
                    stage = "END_TO_END"
                    learning_rate = float(config["end_to_end_learning_rate"])
                else:
                    stage, _ = stage_for_iteration(ntrain, iteration)
                    learning_rate = {
                        "TG_ONLY": float(config["stage1_learning_rate"]),
                        "TRANSFER_CCGR": float(config["stage2_learning_rate"]),
                        "JOINT_FINETUNE": float(config["stage3_learning_rate"]),
                    }[stage]
                if stage != current_stage:
                    current_stage = stage
                    names = _active_groups(model, config["training_strategy"], stage)
                    active = set_trainable(model, names)
                    optimizer = torch.optim.Adam(active, lr=learning_rate, weight_decay=float(config["weight_decay"]))
                    print(f"stage={stage} start={iteration} lr={learning_rate} groups={names}")
                model.train()
                indices = random_batch_indices(ntrain, int(config["batch_size"]), generator)
                images = tensors["train_features"].index_select(0, indices).to(device).float()
                targets = global_to_seen[train_labels.index_select(0, indices)].to(device)
                optimizer.zero_grad(set_to_none=True)
                prototypes = model.prototypes()
                seen_prototypes = prototypes.index_select(0, seen_classes.to(device))
                logits = F.normalize(images, dim=-1) @ seen_prototypes.T * model.scale()
                ce = F.cross_entropy(logits, targets)
                topology = model.topology_loss(prototypes)
                loss = ce + float(config["topology_weight"]) * topology
                if not torch.isfinite(loss):
                    raise FloatingPointError("训练loss包含NaN/Inf。")
                loss.backward()
                require_finite_gradients(model)
                if iteration == 0 or (
                    config["training_strategy"] == "stagewise_50_100_50"
                    and iteration in (ntrain, 3 * ntrain)
                ):
                    stage_gradient_norms[stage] = _gradient_norms(model)
                optimizer.step()
                if iteration % report_interval == 0:
                    metrics = _evaluate_model(model, tensors, manifest, device)
                    diagnostics = model.diagnostics()
                    epoch = iteration // report_interval
                    row = {
                        "iteration": iteration,
                        "nominal_epoch": epoch,
                        "stage": stage,
                        "train_loss": float(loss.detach()),
                        "train_ce": float(ce.detach()),
                        "train_topology": float(topology.detach()),
                        "official_metrics_percent": metrics,
                        "diagnostics": diagnostics,
                    }
                    history.append(row)
                    best_zs = max(best_zs, metrics["ZS"])
                    if metrics["H"] > best_h:
                        best_h = metrics["H"]
                        best_state = copy.deepcopy(model.state_dict())
                        selected = {
                            "iteration": iteration,
                            "epoch": epoch,
                            "stage": stage,
                            "metrics": metrics,
                            "diagnostics": diagnostics,
                        }
                        atomic_torch_save(
                            output_dir / "model_best.pth",
                            {
                                "experiment_id": config["experiment_id"],
                                "condition_id": config["condition_id"],
                                "run_id": run_id,
                                "code_commit": code_commit,
                                "config": config,
                                "config_sha256": config_sha,
                                "selected": selected,
                                "model_state_dict": best_state,
                                "reproducibility": reproducibility,
                            },
                        )
                    print(
                        f"iter={iteration} epoch={epoch} stage={stage} "
                        f"U={metrics['U']:.6f} S={metrics['S']:.6f} H={metrics['H']:.6f} best_H={best_h:.6f}"
                    )
            require_finite_model(model)
            atomic_torch_save(
                output_dir / "checkpoint_last.pth",
                {
                    "experiment_id": config["experiment_id"],
                    "condition_id": config["condition_id"],
                    "run_id": run_id,
                    "code_commit": code_commit,
                    "config": config,
                    "config_sha256": config_sha,
                    "last_iteration": niters - 1,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_model_state_dict": best_state,
                    "selected": selected,
                    "history": history,
                    "stage_gradient_norms": stage_gradient_norms,
                    "reproducibility": reproducibility,
                },
            )
            selected["best_zs_observation_percent"] = best_zs

        atomic_write_json(output_dir / "evaluation_history.json", {"history": history})
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "asset_manifest": str(manifest_path),
                "asset_manifest_sha256": config["asset_manifest_sha256"],
                "asset_id": manifest["asset_id"],
                "files": manifest["outputs_sha256"],
            },
        )
        metrics_payload = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "run_id": run_id,
            "framework_id": config["framework_id"],
            "dataset": config["dataset"],
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "training_strategy": config["training_strategy"],
            "selection_scope": config["selection_scope"],
            "nested_official_test_selection": False,
            "test_used_for_selection": bool(config["test_used_for_selection"]),
            "test_used_for_hyperparameter_selection": bool(config["test_used_for_hyperparameter_selection"]),
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "asset_id": manifest["asset_id"],
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "seed": seed,
            "official_test_evaluation_count": len(history),
            "selected_iteration": selected["iteration"],
            "selected_nominal_epoch": selected["epoch"],
            "selected_stage": selected["stage"],
            "best_metrics_percent": selected["metrics"],
            "selected_diagnostics": selected["diagnostics"],
            "stage_gradient_norms": stage_gradient_norms,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
            "evaluation_history_sha256": sha256_file(output_dir / "evaluation_history.json"),
        }
        if "best_zs_observation_percent" in selected:
            metrics_payload["best_zs_observation_percent"] = selected["best_zs_observation_percent"]
        atomic_write_json(output_dir / "metrics.json", metrics_payload)
        print(json.dumps(metrics_payload, ensure_ascii=False))
        return metrics_payload
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
