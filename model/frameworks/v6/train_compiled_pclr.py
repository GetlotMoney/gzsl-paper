"""Train and evaluate the V6 compiled-PCLR Gate-B head."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v4.train import build_model, load_assets, load_config
from model.frameworks.v5.model import v5_logits
from model.frameworks.v6.compiled_pclr import CompiledPCLRHead
from tools.gzsl_data import per_class_accuracy
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v6-compiled-pclr-gate-b.v1"
EXPERIMENT_ID = "V6-TRY-006"
BASE_COMMIT = "52b511d77b4ad048f35b40dc3cbd9afd092167e9"
SOURCE_CODE_COMMIT = "b0a756dd624e883eb50d19a2455ba06bdc73f118"
SOURCE_CONFIG_SHA = "0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49"
SOURCE_CHECKPOINT_SHA = "16b5071f21a3217e58a72315029c28b8cfd97b68f812641bd0145d3f5e0702ab"
RELATION_MANIFEST_SHA = "0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4"
ASSET_MANIFEST_SHA = "3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6"
PARENT_METRICS = {
    "U": 80.69409728050232,
    "S": 81.44695162773132,
    "H": 81.06877662507551,
    "ZS": 88.78527283668518,
}

CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "idea_id",
    "base_commit",
    "dataset",
    "source_config",
    "source_config_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "source_code_commit",
    "asset_manifest_sha256",
    "relation_asset_manifest_sha256",
    "device",
    "random_seed",
    "batch_size",
    "nominal_epochs",
    "total_updates",
    "eval_interval_steps",
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "relation_loss_weight",
    "ridge_lambda",
    "relation_temperature",
    "direction_temperature",
    "seen_logit_gamma",
    "alpha_max",
    "initial_alpha",
    "role_weight_max",
    "initial_role_weights",
    "candidate_top_k",
    "required_module_delta_h",
    "max_us_gap",
    "parent_metrics_percent",
    "test_used_for_selection",
    "test_used_for_hyperparameter_selection",
    "nested_official_test_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "human_annotations_used",
    "expert_attributes_used",
    "llm_world_knowledge_used",
}


def load_compiled_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    invalid = (
        not isinstance(config, dict)
        or actual != CONFIG_KEYS
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("idea_id") != "IDEA-201"
        or config.get("base_commit") != BASE_COMMIT
        or config.get("dataset") != "CUB"
        or config.get("source_config_sha256") != SOURCE_CONFIG_SHA
        or config.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA
        or config.get("source_code_commit") != SOURCE_CODE_COMMIT
        or config.get("asset_manifest_sha256") != ASSET_MANIFEST_SHA
        or config.get("relation_asset_manifest_sha256") != RELATION_MANIFEST_SHA
        or config.get("device") != "cuda:0"
        or int(config.get("random_seed", -1)) != 7
        or int(config.get("batch_size", -1)) != 50
        or int(config.get("nominal_epochs", -1)) != 150
        or int(config.get("total_updates", -1)) != 21171
        or int(config.get("eval_interval_steps", -1)) != 141
        or float(config.get("learning_rate", -1)) != 1e-4
        or float(config.get("min_learning_rate", -1)) != 1e-5
        or float(config.get("weight_decay", -1)) != 0.0
        or float(config.get("relation_loss_weight", -1)) != 1.0
        or float(config.get("ridge_lambda", -1)) != 0.3
        or float(config.get("relation_temperature", -1)) != 0.2
        or float(config.get("direction_temperature", -1)) != 0.07
        or float(config.get("seen_logit_gamma", -1)) != 0.91
        or float(config.get("alpha_max", -1)) != 2.0
        or abs(float(config.get("initial_alpha", -1)) - 0.7258594751358033) > 1e-12
        or float(config.get("role_weight_max", -1)) != 1.0
        or config.get("initial_role_weights") != [0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.36, 0.0]
        or config.get("candidate_top_k") is not None
        or float(config.get("required_module_delta_h", -1)) != 1.0
        or float(config.get("max_us_gap", -1)) != 8.0
        or config.get("parent_metrics_percent") != PARENT_METRICS
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not True
        or config.get("nested_official_test_selection") is not True
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
        or config.get("expert_attributes_used") is not False
        or config.get("llm_world_knowledge_used") is not True
    )
    if invalid:
        raise ValueError("C-PCLR Gate-B配置身份或预注册合同错误。")
    return config, sha256_file(path)


def _validate_source_path(config: dict, key: str) -> Path:
    path = Path(config[key])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"C-PCLR {key}必须是存在的绝对文件。")
    if sha256_file(path) != config[f"{key}_sha256"]:
        raise ValueError(f"C-PCLR {key} SHA不匹配。")
    return path


def load_source(config: dict, device: torch.device):
    source_config_path = _validate_source_path(config, "source_config")
    source_checkpoint_path = _validate_source_path(config, "source_checkpoint")
    source_config, source_sha = load_config(source_config_path)
    if source_sha != SOURCE_CONFIG_SHA:
        raise ValueError("C-PCLR source config loader SHA不匹配。")
    tensors = load_assets(source_config)
    source = build_model(source_config, tensors, device)
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("code_commit") != SOURCE_CODE_COMMIT
        or checkpoint.get("config_sha256") != SOURCE_CONFIG_SHA
    ):
        raise ValueError("C-PCLR source checkpoint身份错误。")
    source.load_state_dict(checkpoint["model_state_dict"], strict=True)
    source.eval()
    source.requires_grad_(False)
    return source, tensors, checkpoint


def build_head(source, config: dict, device: torch.device) -> CompiledPCLRHead:
    head = CompiledPCLRHead.from_source_model(
        source,
        ridge_lambda=float(config["ridge_lambda"]),
        relation_temperature=float(config["relation_temperature"]),
        direction_temperature=float(config["direction_temperature"]),
        seen_logit_gamma=float(config["seen_logit_gamma"]),
        alpha_max=float(config["alpha_max"]),
        initial_alpha=float(config["initial_alpha"]),
        role_weight_max=float(config["role_weight_max"]),
        initial_role_weights=torch.tensor(config["initial_role_weights"]),
    ).to(device)
    return head


def _condition_logits(head: CompiledPCLRHead, images: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "full": head(images),
        "s_off": head(images, semantic_enabled=False),
        "v_off": head(images, visual_enabled=False),
        "i_off": head(images, interaction_enabled=False),
    }


def _metrics(predictions: dict[str, torch.Tensor], tensors: dict, head: CompiledPCLRHead) -> dict[str, float]:
    seen = head.seen_classes.cpu()
    all_classes = torch.arange(200)
    unseen = all_classes[~torch.isin(all_classes, seen)]
    labels_seen = tensors["test_seen_labels"].long()
    labels_unseen = tensors["test_unseen_labels"].long()
    s = 100.0 * per_class_accuracy(labels_seen, predictions["seen"], seen)
    u = 100.0 * per_class_accuracy(labels_unseen, predictions["unseen"], unseen)
    zs = 100.0 * per_class_accuracy(labels_unseen, predictions["zs"], unseen)
    h = 2.0 * s * u / (s + u) if s + u else 0.0
    return {"U": float(u), "S": float(s), "H": float(h), "ZS": float(zs)}


def _transitions(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    old = before.eq(labels.cpu())
    new = after.eq(labels.cpu())
    return {
        "corrected_wrong_to_right": int((~old & new).sum()),
        "damaged_right_to_wrong": int((old & ~new).sum()),
        "net_correct": int(new.sum() - old.sum()),
        "prediction_changed": int(before.ne(after).sum()),
    }


@torch.no_grad()
def evaluate_head(
    head: CompiledPCLRHead,
    tensors: dict,
    device: torch.device,
    *,
    source=None,
) -> dict:
    head.eval()
    names = ("full", "s_off", "v_off", "i_off")
    outputs = {name: {"seen": [], "unseen": [], "zs": []} for name in names}
    parent = {"seen": [], "unseen": [], "zs": []} if source is not None else None
    unseen = torch.arange(200)[
        ~torch.isin(torch.arange(200), head.seen_classes.cpu())
    ].to(device)
    for split, features in (
        ("seen", tensors["test_seen_features"]),
        ("unseen", tensors["test_unseen_features"]),
    ):
        for start in range(0, len(features), 256):
            images = features[start : start + 256].to(device).float()
            logits_by_name = _condition_logits(head, images)
            for name, logits in logits_by_name.items():
                if tuple(logits.shape) != (len(images), 200) or not torch.isfinite(logits).all():
                    raise RuntimeError(f"C-PCLR {name}评估logits错误。")
                outputs[name][split].append(logits.argmax(dim=1).cpu())
                if split == "unseen":
                    outputs[name]["zs"].append(
                        unseen[logits.index_select(1, unseen).argmax(dim=1)].cpu()
                    )
            if source is not None:
                parent_logits = v5_logits(source, images)
                parent[split].append(parent_logits.argmax(dim=1).cpu())
                if split == "unseen":
                    parent["zs"].append(
                        unseen[parent_logits.index_select(1, unseen).argmax(dim=1)].cpu()
                    )
    for group in outputs.values():
        for split in group:
            group[split] = torch.cat(group[split])
    scores = {name: _metrics(outputs[name], tensors, head) for name in names}
    transitions = {
        name: {
            "seen": _transitions(
                outputs[name]["seen"], outputs["full"]["seen"], tensors["test_seen_labels"]
            ),
            "unseen": _transitions(
                outputs[name]["unseen"], outputs["full"]["unseen"], tensors["test_unseen_labels"]
            ),
            "zs": _transitions(
                outputs[name]["zs"], outputs["full"]["zs"], tensors["test_unseen_labels"]
            ),
        }
        for name in ("s_off", "v_off", "i_off")
    }
    result = {"metrics": scores, "transitions": transitions}
    if parent is not None:
        for split in parent:
            parent[split] = torch.cat(parent[split])
        parent_scores = _metrics(parent, tensors, head)
        for metric, expected in PARENT_METRICS.items():
            if abs(parent_scores[metric] - expected) > 1e-6:
                raise RuntimeError(f"C-PCLR V5 Parent {metric}复现失败。")
        result["parent_metrics"] = parent_scores
        result["parent_predictions"] = parent
    return result


def _learning_rate(config: dict, update: int) -> float:
    total = int(config["total_updates"])
    start = float(config["learning_rate"])
    end = float(config["min_learning_rate"])
    progress = (int(update) - 1) / max(total - 1, 1)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def _gradient_receipt(head: CompiledPCLRHead) -> dict[str, float]:
    values = {}
    for name, parameter in head.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"C-PCLR trainable参数缺少梯度：{name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"C-PCLR梯度包含NaN/Inf：{name}")
        values[name] = float(parameter.grad.detach().norm().cpu())
    return values


def micro_batch(config_path: Path) -> dict:
    config, config_sha = load_compiled_config(config_path)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C-PCLR GPU micro-batch要求CUDA。")
    torch.manual_seed(int(config["random_seed"]))
    torch.cuda.manual_seed_all(int(config["random_seed"]))
    source, tensors, _ = load_source(config, device)
    head = build_head(source, config, device)
    images = tensors["train_features"][: int(config["batch_size"])].to(device).float()
    labels = tensors["train_labels"][: int(config["batch_size"])].to(device).long()
    losses = head.training_losses(
        images,
        labels,
        relation_loss_weight=float(config["relation_loss_weight"]),
    )
    losses["total"].backward()
    gradients = _gradient_receipt(head)
    export = head.export()
    result = {
        "schema_version": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config_sha,
        "batch_size": len(images),
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "gradient_norms": gradients,
        "alpha": float(head.alpha().detach().cpu()),
        "role_weights": [float(value) for value in head.role_weights().detach().cpu()],
        "export_q_shape": list(export.q.shape),
        "export_bias_shape": list(export.bias.shape),
        "finite": all(math.isfinite(value) for value in gradients.values()),
        "persistent_writes": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("C-PCLR expected commit与当前干净HEAD不一致。")
    config, config_sha = load_compiled_config(config_path)
    if config_sha != expected_config_sha:
        raise ValueError("C-PCLR expected config SHA不匹配。")
    if output_dir.name != EXPERIMENT_ID:
        raise ValueError("C-PCLR output-dir末级必须是V6-TRY-006。")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C-PCLR正式RUN要求CUDA。")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(config["random_seed"]))
    torch.cuda.manual_seed_all(int(config["random_seed"]))
    source, tensors, source_checkpoint = load_source(config, device)
    head = build_head(source, config, device)
    train_features = tensors["train_features"].to(device).float()
    train_labels = tensors["train_labels"].to(device).long()
    if len(train_features) != 7057 or torch.unique(train_labels).numel() != 150:
        raise RuntimeError("C-PCLR训练split身份错误。")
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))

    output = prepare_output_dir(output_dir)
    (output / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    log = (output / "training.log").open("w", encoding="utf-8", buffering=1)

    def emit(value: dict) -> None:
        line = json.dumps(value, ensure_ascii=False, sort_keys=True)
        print(line)
        log.write(line + "\n")

    initial = evaluate_head(head, tensors, device, source=source)
    parent_predictions = initial.pop("parent_predictions")
    history = [{"update": 0, **initial}]
    best = copy.deepcopy(initial)
    best_update = 0
    best_state = copy.deepcopy(head.state_dict())
    emit({"event": "initial", "update": 0, **initial})

    interval = {"total": 0.0, "classification": 0.0, "relation": 0.0}
    interval_steps = 0
    for update in range(1, int(config["total_updates"]) + 1):
        head.train()
        lr = _learning_rate(config, update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = torch.randperm(len(train_features), generator=generator)[
            : int(config["batch_size"])
        ].to(device)
        images = train_features.index_select(0, indices)
        targets = train_labels.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        losses = head.training_losses(
            images,
            targets,
            relation_loss_weight=float(config["relation_loss_weight"]),
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("C-PCLR训练loss包含NaN/Inf。")
        losses["total"].backward()
        _gradient_receipt(head)
        optimizer.step()
        for key in interval:
            interval[key] += float(losses[key].detach().cpu())
        interval_steps += 1

        if update % int(config["eval_interval_steps"]) != 0 and update != int(config["total_updates"]):
            continue
        evaluation = evaluate_head(head, tensors, device)
        record = {
            "update": update,
            "learning_rate": lr,
            "train_mean": {key: value / interval_steps for key, value in interval.items()},
            "alpha": float(head.alpha().detach().cpu()),
            "role_weights": [float(value) for value in head.role_weights().detach().cpu()],
            **evaluation,
        }
        history.append(record)
        emit({"event": "evaluation", **record})
        interval = {key: 0.0 for key in interval}
        interval_steps = 0
        if float(evaluation["metrics"]["full"]["H"]) > float(best["metrics"]["full"]["H"]):
            best = copy.deepcopy(evaluation)
            best_update = update
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state, strict=True)
    final = evaluate_head(head, tensors, device)
    full = final["metrics"]["full"]
    deltas = {
        name: float(full["H"] - final["metrics"][name]["H"])
        for name in ("s_off", "v_off", "i_off")
    }
    parent_delta = float(full["H"] - PARENT_METRICS["H"])
    passed = (
        parent_delta > 0.0
        and all(value >= float(config["required_module_delta_h"]) for value in deltas.values())
        and abs(float(full["U"] - full["S"])) < float(config["max_us_gap"])
    )
    export = head.export()
    checkpoint = {
        "schema_version": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
        "source_checkpoint_code_commit": source_checkpoint.get("code_commit"),
        "best_update": best_update,
        "model_state_dict": best_state,
        "export": export.__dict__,
    }
    torch.save(checkpoint, output / "model_best.pth")
    atomic_write_json(output / "evaluation_history.json", history)
    result = {
        "schema_version": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "idea_id": "IDEA-201",
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "base_commit": BASE_COMMIT,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
        "best_update": best_update,
        "parent_metrics": PARENT_METRICS,
        "metrics": final["metrics"],
        "transitions": final["transitions"],
        "delta_H_vs_parent": parent_delta,
        "module_off_delta_H": deltas,
        "module_contract_passed": passed,
        "decision": "keep_gate_b" if passed else "drop_gate_b_contract_failed",
        "candidate_top_k": None,
        "test_used_for_selection": True,
        "test_used_for_hyperparameter_selection": True,
        "nested_official_test_selection": True,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
        "human_annotations_used": False,
        "expert_attributes_used": False,
        "llm_world_knowledge_used": True,
        "parent_predictions_reproduced": all(
            value.numel() > 0 for value in parent_predictions.values()
        ),
    }
    atomic_write_json(output / "metrics.json", result)
    emit({"event": "complete", **result})
    log.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-config-sha")
    parser.add_argument("--micro-batch-only", action="store_true")
    args = parser.parse_args()
    if args.micro_batch_only:
        micro_batch(args.config)
        return
    if args.output_dir is None or not args.expected_commit or not args.expected_config_sha:
        parser.error("正式RUN必须提供output-dir、expected-commit和expected-config-sha。")
    run(args.config, args.output_dir, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()
