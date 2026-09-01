"""Train and evaluate the V6 compiled-PCLR Gate-B head."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v2 import train as h1
from model.frameworks.v4.train import (
    GroupwiseSchedule,
    build_model,
    load_assets,
    load_config,
    rank_modulo_class_folds,
    refresh_oracle_targets,
    teacher_refresh_updates,
)
from model.frameworks.v5.model import v5_logits
from model.frameworks.v6.compiled_pclr import CompiledPCLRHead
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
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
    "source_checkpoint_usage",
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
        or config.get("source_checkpoint_usage") != "parent_control_only_not_training_initialization"
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


def load_training_source(config: dict, device: torch.device):
    """Build the same one-stage R2 source initialization without loading its result."""
    source_config_path = _validate_source_path(config, "source_config")
    _validate_source_path(config, "source_checkpoint")
    source_config, source_sha = load_config(source_config_path)
    if source_sha != SOURCE_CONFIG_SHA:
        raise ValueError("C-PCLR source config loader SHA不匹配。")
    tensors = load_assets(source_config)
    source = build_model(source_config, tensors, device)
    source.requires_grad_(False)
    for parameter in tuple(source.parent.parameters()) + tuple(source.gate.parameters()):
        parameter.requires_grad_(True)
    return source, tensors, source_config


def load_parent_control(config: dict, tensors: dict, device: torch.device):
    """Load the immutable formal V5 source only for read-only Parent parity."""
    source_config_path = _validate_source_path(config, "source_config")
    source_checkpoint_path = _validate_source_path(config, "source_checkpoint")
    source_config, source_sha = load_config(source_config_path)
    if source_sha != SOURCE_CONFIG_SHA:
        raise ValueError("C-PCLR Parent source config loader SHA不匹配。")
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
    return source, checkpoint


def build_head(source, config: dict, device: torch.device) -> CompiledPCLRHead:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
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
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    return head


def _parent_loss(
    source,
    images: torch.Tensor,
    global_targets: torch.Tensor,
    *,
    seen_device: torch.Tensor,
    global_to_seen: torch.Tensor,
    fold_package: dict[str, torch.Tensor],
    source_config: dict,
) -> dict[str, torch.Tensor]:
    targets = global_to_seen.index_select(0, global_targets)
    logits = source.parent.logits(images, seen_device)
    ce = F.cross_entropy(logits, targets)
    topology = source.parent.topology_loss()
    raw_ratio = source.gate.raw_ratio(fold_package["features"])
    gate = F.smooth_l1_loss(raw_ratio, fold_package["target_ratio"])
    total = (
        ce
        + float(source_config["topology_weight"]) * topology
        + float(source_config["gate_loss_weight"]) * gate
    )
    return {"total": total, "ce": ce, "topology": topology, "gate": gate}


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


def gate_b_contract_passed(
    metrics: dict[str, dict[str, float]],
    *,
    best_update: int,
    required_module_delta_h: float,
    max_us_gap: float,
) -> bool:
    """Apply the preregistered AND gate; update 0 can never prove training."""
    full = metrics["full"]
    return bool(
        int(best_update) > 0
        and float(full["H"]) > float(PARENT_METRICS["H"])
        and all(
            float(full["H"] - metrics[name]["H"]) >= float(required_module_delta_h)
            for name in ("s_off", "v_off", "i_off")
        )
        and abs(float(full["U"] - full["S"])) < float(max_us_gap)
    )


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


def _gradient_receipt(
    head: CompiledPCLRHead,
    *,
    require_nonzero: bool = False,
) -> dict[str, float]:
    values = {}
    for name, parameter in head.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"C-PCLR trainable参数缺少梯度：{name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"C-PCLR梯度包含NaN/Inf：{name}")
        norm = float(parameter.grad.detach().norm().cpu())
        if not math.isfinite(norm) or (require_nonzero and norm <= 0.0):
            requirement = "有限正数" if require_nonzero else "有限非负数"
            raise RuntimeError(f"C-PCLR trainable参数梯度范数必须为{requirement}：{name}")
        values[name] = norm
    return values


def _finite_source_gradients(source) -> dict[str, float]:
    values = {}
    active_groups = source.parent.parameter_groups()
    for group_name, parameters in active_groups.items():
        finite_count = 0
        for index, parameter in enumerate(parameters):
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"C-PCLR active parent参数梯度非有限：{group_name}[{index}]")
            values[f"parent.{group_name}[{index}]"] = float(parameter.grad.detach().norm().cpu())
            finite_count += 1
        if parameters and finite_count == 0:
            raise RuntimeError(f"C-PCLR active parent参数组没有任何实际梯度：{group_name}")
    if not values:
        raise RuntimeError("C-PCLR没有实际启用的Parent参数组。")
    for name, parameter in source.gate.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"C-PCLR gate参数缺少有限梯度：{name}")
        values[f"gate.{name}"] = float(parameter.grad.detach().norm().cpu())
    return values


def micro_batch(config_path: Path) -> dict:
    config, config_sha = load_compiled_config(config_path)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C-PCLR GPU micro-batch要求CUDA。")
    configure_reproducibility(
        int(config["random_seed"]), strict_determinism=True, deterministic_warn_only=False
    )
    source, tensors, source_config = load_training_source(config, device)
    head = build_head(source, config, device)
    labels_cpu = tensors["train_labels"].long()
    seen = torch.unique(labels_cpu, sorted=True)
    seen_device = seen.to(device)
    global_to_seen = torch.full((200,), -1, dtype=torch.long, device=device)
    global_to_seen[seen_device] = torch.arange(len(seen), device=device)
    visual_centroids = h1.visual_centroids(
        tensors["train_features"], labels_cpu, seen
    ).to(device)
    folds = rank_modulo_class_folds(seen)
    packages = refresh_oracle_targets(
        source, visual_centroids, folds, float(source_config["theta_penalty"])
    )
    images = tensors["train_features"][: int(config["batch_size"])].to(device).float()
    labels = labels_cpu[: int(config["batch_size"])].to(device)
    parent_losses = _parent_loss(
        source,
        images,
        labels,
        seen_device=seen_device,
        global_to_seen=global_to_seen,
        fold_package=packages[0],
        source_config=source_config,
    )
    head_losses = head.training_losses(
        images,
        labels,
        relation_loss_weight=float(config["relation_loss_weight"]),
    )
    total = parent_losses["total"] + head_losses["total"]
    total.backward()
    _gradient_receipt(head)
    _finite_source_gradients(source)
    parent_optimizer = torch.optim.Adam(
        [*source.parent.parameters(), *source.gate.parameters()], lr=1e-4
    )
    head_optimizer = torch.optim.Adam(head.parameters(), lr=float(config["learning_rate"]))
    parent_optimizer.step()
    head_optimizer.step()
    head.sync_source_prototypes(source)
    parent_optimizer.zero_grad(set_to_none=True)
    head_optimizer.zero_grad(set_to_none=True)
    parent_losses = _parent_loss(
        source,
        images,
        labels,
        seen_device=seen_device,
        global_to_seen=global_to_seen,
        fold_package=packages[0],
        source_config=source_config,
    )
    head_losses = head.training_losses(
        images,
        labels,
        relation_loss_weight=float(config["relation_loss_weight"]),
    )
    total = parent_losses["total"] + head_losses["total"]
    total.backward()
    head_gradients = _gradient_receipt(head, require_nonzero=True)
    source_gradients = _finite_source_gradients(source)
    export = head.export()
    result = {
        "schema_version": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config_sha,
        "batch_size": len(images),
        "losses": {
            "joint_total": float(total.detach().cpu()),
            **{f"parent_{key}": float(value.detach().cpu()) for key, value in parent_losses.items()},
            **{f"head_{key}": float(value.detach().cpu()) for key, value in head_losses.items()},
        },
        "head_gradient_norms": head_gradients,
        "source_gradient_norms": source_gradients,
        "alpha": float(head.alpha().detach().cpu()),
        "role_weights": [float(value) for value in head.role_weights().detach().cpu()],
        "export_q_shape": list(export.q.shape),
        "export_bias_shape": list(export.bias.shape),
        "finite": all(math.isfinite(value) for value in (*head_gradients.values(), *source_gradients.values())),
        "one_stage_joint_training": True,
        "micro_steps": 2,
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
    reproducibility = configure_reproducibility(
        int(config["random_seed"]), strict_determinism=True, deterministic_warn_only=False
    )
    source, tensors, source_config = load_training_source(config, device)
    head = build_head(source, config, device)
    train_features = tensors["train_features"].to(device).float()
    train_labels = tensors["train_labels"].to(device).long()
    if len(train_features) != 7057 or torch.unique(train_labels).numel() != 150:
        raise RuntimeError("C-PCLR训练split身份错误。")
    seen = torch.unique(train_labels.detach().cpu(), sorted=True)
    seen_device = seen.to(device)
    global_to_seen = torch.full((200,), -1, dtype=torch.long, device=device)
    global_to_seen[seen_device] = torch.arange(len(seen), device=device)
    visual_centroids = h1.visual_centroids(
        tensors["train_features"], tensors["train_labels"].long(), seen
    ).to(device)
    folds = rank_modulo_class_folds(seen)
    refresh_updates = teacher_refresh_updates(
        train_count=len(train_features),
        nominal_epochs=int(config["nominal_epochs"]),
        batch_size=int(config["batch_size"]),
    )
    refresh_set = set(refresh_updates)
    packages = refresh_oracle_targets(
        source, visual_centroids, folds, float(source_config["theta_penalty"])
    )
    teacher_refresh_count = 1

    parent_parameters = list(source.parent.parameters())
    gate_parameters = list(source.gate.parameters())
    if {id(value) for value in parent_parameters}.intersection(id(value) for value in gate_parameters):
        raise RuntimeError("C-PCLR Parent与Gate参数组不得重叠。")
    parent_optimizer = torch.optim.Adam(
        [
            {"params": parent_parameters, "lr": float(source_config["tg_learning_rate"])},
            {"params": gate_parameters, "lr": float(source_config["gate_learning_rate"])},
        ],
        weight_decay=float(source_config["weight_decay"]),
    )
    warmup_updates = (
        len(train_features) * int(source_config["gate_warmup_epochs"])
        // int(config["batch_size"])
    )
    parent_scheduler = GroupwiseSchedule(
        parent_optimizer,
        total_updates=int(config["total_updates"]),
        warmup_updates=warmup_updates,
        tg_min_multiplier=float(source_config["tg_min_learning_rate"])
        / float(source_config["tg_learning_rate"]),
        gate_min_multiplier=float(source_config["gate_min_learning_rate"])
        / float(source_config["gate_learning_rate"]),
    )
    head_optimizer = torch.optim.Adam(
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

    # Parent parity is read-only and RNG-neutral; it does not initialize training.
    parent_cpu_rng = torch.random.get_rng_state()
    parent_cuda_rng = torch.cuda.get_rng_state_all()
    try:
        parent_control, source_checkpoint = load_parent_control(config, tensors, device)
        initial = evaluate_head(head, tensors, device, source=parent_control)
    finally:
        torch.random.set_rng_state(parent_cpu_rng)
        torch.cuda.set_rng_state_all(parent_cuda_rng)
    del parent_control
    parent_predictions = initial.pop("parent_predictions")
    history = [{"update": 0, **initial}]
    best = None
    best_update = -1
    best_state = None
    best_source_state = None
    best_zs = {
        "update": 0,
        "ZS": float(initial["metrics"]["full"]["ZS"]),
        "metrics": copy.deepcopy(initial["metrics"]["full"]),
    }
    emit({"event": "initial", "update": 0, **initial})

    interval = {
        "joint_total": 0.0,
        "parent_total": 0.0,
        "parent_ce": 0.0,
        "parent_topology": 0.0,
        "parent_gate": 0.0,
        "head_total": 0.0,
        "head_classification": 0.0,
        "head_relation": 0.0,
    }
    interval_steps = 0
    for update in range(1, int(config["total_updates"]) + 1):
        if update in refresh_set and update != 1:
            packages = refresh_oracle_targets(
                source, visual_centroids, folds, float(source_config["theta_penalty"])
            )
            teacher_refresh_count += 1
        source.train()
        head.train()
        parent_scheduler.set_for_update(update)
        head_lr = _learning_rate(config, update)
        for group in head_optimizer.param_groups:
            group["lr"] = head_lr
        indices = torch.randperm(len(train_features), generator=generator)[
            : int(config["batch_size"])
        ].to(device)
        images = train_features.index_select(0, indices)
        targets = train_labels.index_select(0, indices)
        parent_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        parent_losses = _parent_loss(
            source,
            images,
            targets,
            seen_device=seen_device,
            global_to_seen=global_to_seen,
            fold_package=packages[(update - 1) % 3],
            source_config=source_config,
        )
        head_losses = head.training_losses(
            images,
            targets,
            relation_loss_weight=float(config["relation_loss_weight"]),
        )
        joint_total = parent_losses["total"] + head_losses["total"]
        if not torch.isfinite(joint_total):
            raise FloatingPointError("C-PCLR训练loss包含NaN/Inf。")
        joint_total.backward()
        _gradient_receipt(head)
        _finite_source_gradients(source)
        parent_optimizer.step()
        head_optimizer.step()
        head.sync_source_prototypes(source)
        values = {
            "joint_total": joint_total,
            "parent_total": parent_losses["total"],
            "parent_ce": parent_losses["ce"],
            "parent_topology": parent_losses["topology"],
            "parent_gate": parent_losses["gate"],
            "head_total": head_losses["total"],
            "head_classification": head_losses["classification"],
            "head_relation": head_losses["relation"],
        }
        for key, value in values.items():
            interval[key] += float(value.detach().cpu())
        interval_steps += 1

        if update % int(config["eval_interval_steps"]) != 0 and update != int(config["total_updates"]):
            continue
        evaluation = evaluate_head(head, tensors, device)
        record = {
            "update": update,
            "parent_learning_rates": [float(group["lr"]) for group in parent_optimizer.param_groups],
            "head_learning_rate": head_lr,
            "train_mean": {key: value / interval_steps for key, value in interval.items()},
            "alpha": float(head.alpha().detach().cpu()),
            "role_weights": [float(value) for value in head.role_weights().detach().cpu()],
            **evaluation,
        }
        history.append(record)
        emit({"event": "evaluation", **record})
        interval = {key: 0.0 for key in interval}
        interval_steps = 0
        if float(evaluation["metrics"]["full"]["ZS"]) > float(best_zs["ZS"]):
            best_zs = {
                "update": update,
                "ZS": float(evaluation["metrics"]["full"]["ZS"]),
                "metrics": copy.deepcopy(evaluation["metrics"]["full"]),
            }
        if best is None or float(evaluation["metrics"]["full"]["H"]) > float(best["metrics"]["full"]["H"]):
            best = copy.deepcopy(evaluation)
            best_update = update
            best_state = copy.deepcopy(head.state_dict())
            best_source_state = copy.deepcopy(source.state_dict())

    if best is None or best_state is None or best_source_state is None or best_update <= 0:
        raise RuntimeError("C-PCLR没有产生训练后的best-H checkpoint。")
    head.load_state_dict(best_state, strict=True)
    final = evaluate_head(head, tensors, device)
    full = final["metrics"]["full"]
    deltas = {
        name: float(full["H"] - final["metrics"][name]["H"])
        for name in ("s_off", "v_off", "i_off")
    }
    parent_delta = float(full["H"] - PARENT_METRICS["H"])
    passed = gate_b_contract_passed(
        final["metrics"],
        best_update=best_update,
        required_module_delta_h=float(config["required_module_delta_h"]),
        max_us_gap=float(config["max_us_gap"]),
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
        "source_model_state_dict": best_source_state,
        "best_zs_observation": best_zs,
        "export": export.__dict__,
    }
    atomic_torch_save(output / "model_best.pth", checkpoint)
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
        "best_zs_observation": best_zs,
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
        "one_stage_joint_training": True,
        "source_initialized_from_checkpoint": False,
        "total_updates": int(config["total_updates"]),
        "teacher_refresh_count": teacher_refresh_count,
        "reproducibility": reproducibility,
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
