from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml

from model.innovations.unified_expert import ExpertAttributeUnifiedModel
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


EVALUATION_PROTOCOL = "chen_shiming_code_aligned_test_selected_gzsl"
TRAINING_KEYS = ("sentence_embeds", "train_features", "train_labels", "res101", "att_splits")
OFFICIAL_KEYS = ("seen_features", "seen_labels", "unseen_features", "unseen_labels")
INPUT_KEYS = TRAINING_KEYS + OFFICIAL_KEYS
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
    "expert_attributes_used",
    "feature_backbone",
    "feature_provenance_complete",
    "device",
    "random_seed",
    "batch_size",
    "epochs",
    "niters",
    "report_interval",
    "optimizer",
    "learning_rate",
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
    "max_attribute_residual",
    "inputs",
    "expected_sha256",
    "class_order_sha256",
}


def load_config(path: Path) -> tuple[dict, str]:
    path = h1.repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Chen-style配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"Chen-style配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if config["schema_version"] != "gzsl-paper.chen-style.v1":
        raise ValueError("Chen-style配置schema错误。")
    if config["experiment_id"] != "V2-CONFIRM-004":
        raise ValueError("Chen-style实验身份错误。")
    if config["condition_id"] not in ("NO-EXPERT", "EXPERT"):
        raise ValueError("Chen-style condition只允许NO-EXPERT或EXPERT。")
    if config["framework_id"] != "FRAMEWORK-V2" or config["dataset"] != "CUB":
        raise ValueError("Chen-style训练只接受FRAMEWORK-V2/CUB。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("Chen-style评估协议身份错误。")
    required = {
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
    }
    for key, expected in required.items():
        if config[key] is not expected:
            raise ValueError(f"Chen-style边界错误：{key}必须为{expected}。")
    if config["training_strategy"] != "end_to_end_joint":
        raise ValueError("首次Chen-style实验固定端到端联合训练。")
    if config["selection_scope"] != "whole_model_only":
        raise ValueError("Chen-style只允许整模型H选模。")
    expected_expert = config["condition_id"] == "EXPERT"
    if config["expert_attributes_used"] is not expected_expert:
        raise ValueError("专家属性开关与condition不一致。")
    if config["feature_provenance_complete"] is not False:
        raise ValueError("必须披露遗留CLIP缓存来源不完整。")
    if int(config["batch_size"]) != 50 or int(config["epochs"]) != 200:
        raise ValueError("Chen-style固定batch 50和200名义epoch。")
    expected_niters = 7057 * int(config["epochs"]) // int(config["batch_size"])
    if int(config["niters"]) != expected_niters or int(config["report_interval"]) != expected_niters // 200:
        raise ValueError("Chen-style niters/report_interval与公开代码公式不一致。")
    if config["optimizer"] != "Adam" or float(config["learning_rate"]) != 1e-4:
        raise ValueError("Chen-style固定公开代码Adam lr=1e-4。")
    expected_topology = 0.2 if expected_expert else 0.1
    if float(config["topology_weight"]) != expected_topology:
        raise ValueError("Chen-style首RUN必须沿用已登记路线topology值。")
    if set(config["inputs"]) != set(INPUT_KEYS) or set(config["expected_sha256"]) != set(INPUT_KEYS):
        raise ValueError("Chen-style输入或SHA字段不完整。")
    return config, sha256_file(path)


def resolve_paths(config: dict) -> dict[str, Path]:
    paths = {name: h1.repo_path(config["inputs"][name]) for name in INPUT_KEYS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少Chen-style输入：" + ", ".join(missing))
    return paths


def verify_inputs(config: dict, paths: dict[str, Path]) -> dict[str, str]:
    actual = {name: sha256_file(paths[name]) for name in INPUT_KEYS}
    mismatch = [name for name in INPUT_KEYS if actual[name] != config["expected_sha256"][name]]
    if mismatch:
        raise ValueError("Chen-style输入SHA不匹配：" + ", ".join(mismatch))
    names = sio.loadmat(paths["att_splits"], variable_names=["allclasses_names"])["allclasses_names"]
    serialized = json.dumps(
        [str(item[0][0]) for item in names], ensure_ascii=False, separators=(",", ":")
    )
    if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != config["class_order_sha256"]:
        raise ValueError("CUB类别顺序不匹配。")
    return actual


def random_batch_indices(count: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """严格对齐TransZero next_batch：每步独立randperm后截取batch。"""
    return torch.randperm(count, generator=generator)[:batch_size]


def _gradient_group_norms(model) -> dict[str, float]:
    text_model = model.text_model if isinstance(model, ExpertAttributeUnifiedModel) else model
    groups = {
        "tg_vpr": text_model.tg_vpr.parameters(),
        "transport": list(text_model.transport_trunk.parameters()) + list(text_model.transport_head.parameters()),
        "generator": list(text_model.generator_trunk.parameters())
        + list(text_model.generator_weight_head.parameters())
        + list(text_model.generator_magnitude_head.parameters()),
    }
    if isinstance(model, ExpertAttributeUnifiedModel):
        groups["expert_attribute"] = [
            model.raw_attribute_residual,
            *model.attribute_projection.parameters(),
        ]
    result = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
        result[name] = float(torch.stack(gradients).norm()) if gradients else 0.0
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
        raise RuntimeError("Chen-style训练要求可见CUDA。")
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
            paths["res101"],
            paths["att_splits"],
            train_labels,
            official["seen_labels"],
            official["unseen_labels"],
            "cpu",
        )
        if not torch.equal(checked_seen, seenclasses) or not torch.equal(checked_unseen, unseenclasses):
            raise RuntimeError("Chen-style official split不一致。")
        centroids = h1.visual_centroids(train_features, train_labels, seenclasses)
        text_model = UnifiedSeenPrototypeModel(
            sentence_embeds,
            seenclasses,
            centroids,
            active_classes=allclasses,
            dropout=float(config["dropout"]),
            inner_ratio=float(config["inner_ratio"]),
            outer_ratio=float(config["outer_ratio"]),
            temperature=float(config["temperature"]),
            transport_hidden_dim=int(config["transport_hidden_dim"]),
            generator_hidden_dim=int(config["generator_hidden_dim"]),
            max_transport_step=float(config["max_transport_step"]),
            max_generator_magnitude=float(config["max_generator_magnitude"]),
        )
        if config["condition_id"] == "EXPERT":
            attribute_mat = sio.loadmat(paths["att_splits"], variable_names=["att"])["att"]
            model = ExpertAttributeUnifiedModel(
                text_model,
                torch.from_numpy(attribute_mat.T).float(),
                max_attribute_residual=float(config["max_attribute_residual"]),
            )
        else:
            model = text_model
        model = model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        global_to_seen = torch.full((200,), -1, dtype=torch.long)
        global_to_seen[seenclasses] = torch.arange(150)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        niters = int(config["niters"])
        report_interval = int(config["report_interval"])
        best_h = float("-inf")
        best_metrics = None
        best_state = None
        best_iteration = None
        best_nominal_epoch = None
        best_zs_observation = float("-inf")
        history = []
        first_batch_gradient_norms = None
        print(f"实验：{config['experiment_id']} 条件：{config['condition_id']}")
        print(f"代码commit：{code_commit} 配置SHA：{config_sha}")
        print(f"niters={niters} report_interval={report_interval} test_used_for_selection=true")

        for iteration in range(niters):
            model.train()
            indices = random_batch_indices(train_labels.numel(), int(config["batch_size"]), generator)
            images = train_features.index_select(0, indices).to(device).float()
            targets = global_to_seen[train_labels.index_select(0, indices)].to(device)
            optimizer.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model.logits(images, seenclasses), targets)
            topology = model.topology_loss()
            loss = ce + float(config["topology_weight"]) * topology
            if not torch.isfinite(loss):
                raise FloatingPointError("Chen-style训练loss包含NaN/Inf。")
            loss.backward()
            require_finite_gradients(model)
            if iteration == 0:
                first_batch_gradient_norms = _gradient_group_norms(model)
                if any(value <= 0.0 for value in first_batch_gradient_norms.values()):
                    raise RuntimeError("端到端各模块必须在首批获得非零梯度。")
            optimizer.step()

            if iteration % report_interval == 0:
                metrics = h1.evaluate(model, official, seenclasses, unseenclasses, device)
                nominal_epoch = iteration // report_interval
                row = {
                    "iteration": iteration,
                    "nominal_epoch": nominal_epoch,
                    "train_loss": float(loss.detach()),
                    "train_ce": float(ce.detach()),
                    "train_topology": float(topology.detach()),
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
                    best_state = copy.deepcopy(model.state_dict())
                    atomic_torch_save(
                        output_dir / "model_best.pth",
                        {
                            "experiment_id": config["experiment_id"],
                            "condition_id": config["condition_id"],
                            "run_id": run_id,
                            "code_commit": code_commit,
                            "config": config,
                            "config_sha256": config_sha,
                            "selected_iteration": best_iteration,
                            "selected_nominal_epoch": best_nominal_epoch,
                            "best_metrics_percent": best_metrics,
                            "model_state_dict": best_state,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} epoch={nominal_epoch} loss={float(loss.detach()):.6f} "
                    f"U={metrics['U']:.6f} S={metrics['S']:.6f} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f}"
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
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "selected_nominal_epoch": best_nominal_epoch,
                "history": history,
                "reproducibility": reproducibility,
            },
        )
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        metrics_payload = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "run_id": run_id,
            "framework_id": config["framework_id"],
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "training_strategy": "end_to_end_joint",
            "selection_scope": "whole_model_only",
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "expert_attributes_used": bool(config["expert_attributes_used"]),
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "seed": seed,
            "niters": niters,
            "report_interval": report_interval,
            "official_test_evaluation_count": len(history),
            "selected_iteration": best_iteration,
            "selected_nominal_epoch": best_nominal_epoch,
            "best_metrics_percent": best_metrics,
            "best_zs_observation_percent": best_zs_observation,
            "first_batch_gradient_norms": first_batch_gradient_norms,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", metrics_payload)
        print({"best": best_metrics, "iteration": best_iteration, "epoch": best_nominal_epoch})
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
