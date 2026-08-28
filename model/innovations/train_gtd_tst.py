"""Fixed-150 TG+GTD training for CUB, AWA2, and SUN."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.gtd_tst import GTDTSTModel
from model.paper_v2 import PaperV2ThreeModuleModel
from model.tg_vpr_h1 import train as h1
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


SCHEMA = "gzsl-paper.v3-gtd-tst-train.v1"
SCRATCH_SCHEMA = "gzsl-paper.v3-gtd-scratch-confirm.v1"
MULTIDATASET_SCHEMA = "gzsl-paper.v3-gtd-multidataset.v1"
DATASET_SPECS = {
    "CUB": {
        "train_count": 7057,
        "test_seen_count": 1764,
        "test_unseen_count": 2967,
        "seen_count": 150,
        "class_count": 200,
    },
    "AWA2": {
        "train_count": 23527,
        "test_seen_count": 5882,
        "test_unseen_count": 7913,
        "seen_count": 40,
        "class_count": 50,
    },
    "SUN": {
        "train_count": 10320,
        "test_seen_count": 2580,
        "test_unseen_count": 1440,
        "seen_count": 645,
        "class_count": 717,
    },
}
TRAIN_COUNT = DATASET_SPECS["CUB"]["train_count"]
SEEN_COUNT = DATASET_SPECS["CUB"]["seen_count"]
CLASS_COUNT = DATASET_SPECS["CUB"]["class_count"]
NOMINAL_EPOCHS = 150
BATCH_SIZE = 50
EVAL_INTERVAL = TRAIN_COUNT // BATCH_SIZE
TOTAL_UPDATES = TRAIN_COUNT * NOMINAL_EPOCHS // BATCH_SIZE
TEACHER_REFRESH_UPDATES = tuple(1 + EVAL_INTERVAL * index for index in range(NOMINAL_EPOCHS))
MATCHED_CONTROL_ID = "V3-TRY-020"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "framework_id",
    "dataset",
    "condition_id",
    "asset_manifest",
    "asset_manifest_sha256",
    "asset_id",
    "tg_checkpoint",
    "tg_checkpoint_sha256",
    "parent_metrics_percent",
    "required_delta_h",
    "max_us_gap",
    "device",
    "random_seed",
    "batch_size",
    "nominal_epochs",
    "total_updates",
    "eval_interval_steps",
    "tg_learning_rate",
    "gate_learning_rate",
    "tg_min_learning_rate",
    "gate_min_learning_rate",
    "gate_warmup_epochs",
    "weight_decay",
    "topology_weight",
    "gate_loss_weight",
    "hidden_dim",
    "grid_points",
    "theta_penalty",
    "max_transport_step",
    "early_stopping_enabled",
    "human_annotations_used",
    "test_used_for_selection",
    "test_used_for_hyperparameter_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"GTD配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    parent = {
        "U": 78.40787768363953,
        "S": 74.98387098312378,
        "H": 76.65765903827264,
        "ZS": 86.14675998687744,
    }
    spec = DATASET_SPECS.get(config["dataset"])
    expected_interval = (
        int(spec["train_count"]) // BATCH_SIZE if spec is not None else -1
    )
    expected_updates = (
        int(spec["train_count"]) * NOMINAL_EPOCHS // BATCH_SIZE
        if spec is not None
        else -1
    )
    shared_invalid = (
        config["framework_id"] != "FRAMEWORK-V3-EXPLORATION"
        or spec is None
        or int(config["random_seed"]) != 7
        or int(config["batch_size"]) != BATCH_SIZE
        or int(config["nominal_epochs"]) != NOMINAL_EPOCHS
        or int(config["total_updates"]) != expected_updates
        or int(config["eval_interval_steps"]) != expected_interval
        or int(config["gate_warmup_epochs"]) != 5
        or float(config["weight_decay"]) != 1e-4
        or float(config["topology_weight"]) != 0.1
        or int(config["hidden_dim"]) != 16
        or int(config["grid_points"]) != 33
        or float(config["theta_penalty"]) != 0.1
        or float(config["max_transport_step"]) != 1.5
        or config["early_stopping_enabled"] is not False
        or float(config["required_delta_h"]) != 1.0
        or float(config["max_us_gap"]) != 8.0
        or config["human_annotations_used"] is not False
        or config["test_used_for_selection"] is not True
        or config["test_used_for_hyperparameter_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    )
    if shared_invalid:
        raise ValueError("GTD共享训练参数、预算或披露边界错误。")
    if config["schema_version"] == SCHEMA:
        invalid = (
            config["dataset"] != "CUB"
            or config["experiment_id"] != "V3-TRY-022"
            or config["condition_id"] != "TG_PLUS_GTD_TST_FIXED150"
            or config["parent_metrics_percent"] != parent
            or float(config["gate_loss_weight"]) != 1.0
            or float(config["tg_learning_rate"]) != 1e-5
            or float(config["gate_learning_rate"]) != 1e-4
            or float(config["tg_min_learning_rate"]) != 1e-6
            or float(config["gate_min_learning_rate"]) != 1e-5
            or not isinstance(config["tg_checkpoint"], str)
            or not isinstance(config["tg_checkpoint_sha256"], str)
        )
    elif config["schema_version"] == SCRATCH_SCHEMA:
        expected = {
            "V3-TRY-040": ("TG_SCRATCH_FIXED150", 0.0),
            "V3-TRY-041": ("TG_PLUS_GTD_SCRATCH_FIXED150", 1.0),
        }
        identity = expected.get(config["experiment_id"])
        invalid = (
            config["dataset"] != "CUB"
            or identity is None
            or config["condition_id"] != (identity[0] if identity else None)
            or float(config["gate_loss_weight"]) != (identity[1] if identity else -1.0)
            or config["tg_checkpoint"] is not None
            or config["tg_checkpoint_sha256"] is not None
            or config["parent_metrics_percent"] is not None
            or float(config["tg_learning_rate"]) != 1e-4
            or float(config["gate_learning_rate"]) != 1e-4
            or float(config["tg_min_learning_rate"]) != 1e-4
            or float(config["gate_min_learning_rate"]) != 1e-5
        )
    elif config["schema_version"] == MULTIDATASET_SCHEMA:
        expected = {
            "V3-TRY-046": "AWA2",
            "V3-TRY-047": "SUN",
        }
        invalid = (
            expected.get(config["experiment_id"]) != config["dataset"]
            or config["condition_id"] != "TG_PLUS_GTD_SCRATCH_FIXED150"
            or config["tg_checkpoint"] is not None
            or config["tg_checkpoint_sha256"] is not None
            or config["parent_metrics_percent"] is not None
            or float(config["gate_loss_weight"]) != 1.0
            or float(config["tg_learning_rate"]) != 1e-4
            or float(config["gate_learning_rate"]) != 1e-4
            or float(config["tg_min_learning_rate"]) != 1e-4
            or float(config["gate_min_learning_rate"]) != 1e-5
        )
    else:
        invalid = True
    if invalid:
        raise ValueError("GTD运行身份、初始化方式或条件开关错误。")
    return config, sha256_file(path)


def checkpoint_parent_metrics(
    checkpoint: dict,
    configured_parent: dict | None,
) -> dict[str, float]:
    """Restore the update-0 metric anchor required by resumed delta reporting."""
    candidate = checkpoint.get("parent_metrics_percent")
    if candidate is None:
        history = checkpoint.get("history")
        if not isinstance(history, list) or not history or history[0].get("update") != 0:
            raise ValueError("GTD resume checkpoint缺少update-0父指标。")
        candidate = {metric: history[0].get(metric) for metric in ("U", "S", "H", "ZS")}
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"U", "S", "H", "ZS"}
        or any(isinstance(value, bool) for value in candidate.values())
    ):
        raise ValueError("GTD resume父指标字段错误。")
    restored = {metric: float(candidate[metric]) for metric in ("U", "S", "H", "ZS")}
    if not all(math.isfinite(value) for value in restored.values()):
        raise ValueError("GTD resume父指标包含NaN/Inf。")
    if configured_parent is not None:
        configured = {metric: float(configured_parent[metric]) for metric in restored}
        if any(abs(restored[metric] - configured[metric]) > 1e-8 for metric in restored):
            raise ValueError("GTD resume父指标与配置父指标不一致。")
    return restored


def _verified_tensor(manifest_path: Path, manifest: dict, name: str) -> torch.Tensor:
    path = manifest_path.parent / name
    if not path.is_file() or sha256_file(path) != manifest["outputs_sha256"].get(name):
        raise ValueError(f"GTD资产文件缺失或SHA错误：{name}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_assets(config: dict) -> dict[str, torch.Tensor]:
    manifest_path = Path(config["asset_manifest"])
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != config["asset_manifest_sha256"]
    ):
        raise ValueError("GTD资产manifest SHA错误。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = DATASET_SPECS[config["dataset"]]
    manifest_counts = manifest.get("counts")
    if (
        manifest.get("schema_version") != "gzsl-paper.clip-assets.v1"
        or manifest.get("asset_id") != config["asset_id"]
        or manifest.get("dataset") != config["dataset"]
        or (
            manifest_counts is not None
            and manifest_counts
            != {
                "train": spec["train_count"],
                "test_seen": spec["test_seen_count"],
                "test_unseen": spec["test_unseen_count"],
            }
        )
    ):
        raise ValueError("GTD资产身份、数据集或数量错误。")
    forbidden = {
        "attributes",
        "class_attributes",
        "part_labels",
        "parts",
        "boxes",
        "bounding_boxes",
        "expert_residuals",
    }
    if forbidden.intersection(manifest.get("outputs_sha256", {})):
        raise ValueError("GTD资产包含人工属性、部位、框或专家残差。")
    names = (
        "train_features.pt",
        "train_labels.pt",
        "test_seen_features.pt",
        "test_seen_labels.pt",
        "test_unseen_features.pt",
        "test_unseen_labels.pt",
        "role_sentence_embeds.pt",
    )
    tensors = {name.removesuffix(".pt"): _verified_tensor(manifest_path, manifest, name) for name in names}
    expected_shapes = {
        "train_features": (spec["train_count"], 768),
        "train_labels": (spec["train_count"],),
        "test_seen_features": (spec["test_seen_count"], 768),
        "test_seen_labels": (spec["test_seen_count"],),
        "test_unseen_features": (spec["test_unseen_count"], 768),
        "test_unseen_labels": (spec["test_unseen_count"],),
        "role_sentence_embeds": (spec["class_count"], 8, 768),
    }
    for name, expected in expected_shapes.items():
        if tuple(tensors[name].shape) != expected:
            raise ValueError(f"GTD资产{name} shape错误：{tuple(tensors[name].shape)}")
    train_classes = torch.unique(tensors["train_labels"].long(), sorted=True)
    test_seen_classes = torch.unique(tensors["test_seen_labels"].long(), sorted=True)
    test_unseen_classes = torch.unique(tensors["test_unseen_labels"].long(), sorted=True)
    all_classes = torch.cat((train_classes, test_unseen_classes)).sort().values
    if (
        train_classes.numel() != spec["seen_count"]
        or not torch.equal(test_seen_classes, train_classes)
        or test_unseen_classes.numel() != spec["class_count"] - spec["seen_count"]
        or torch.isin(train_classes, test_unseen_classes).any()
        or not torch.equal(all_classes, torch.arange(spec["class_count"]))
    ):
        raise ValueError("GTD资产seen/unseen类别轴或split身份错误。")
    return tensors


def build_model(config: dict, tensors: dict[str, torch.Tensor], device: torch.device) -> GTDTSTModel:
    labels = tensors["train_labels"].long()
    seen = torch.unique(labels, sorted=True)
    centroids = h1.visual_centroids(tensors["train_features"], labels, seen)
    parent = PaperV2ThreeModuleModel(
        tensors["role_sentence_embeds"],
        seen,
        centroids,
        tg_vpr_mode="full",
        transport_mode="off",
        ccgr_mode="off",
        dropout=0.5,
        inner_ratio=0.35,
        outer_ratio=0.65,
        temperature=0.07,
    ).to(device)
    if config["tg_checkpoint"] is not None:
        checkpoint_path = Path(config["tg_checkpoint"])
        if (
            not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != config["tg_checkpoint_sha256"]
        ):
            raise ValueError("GTD TG checkpoint SHA错误。")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state = {
            key.removeprefix("parent."): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("parent.")
        }
        missing, unexpected = parent.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"GTD TG状态不完整：missing={missing}, unexpected={unexpected}")
    return GTDTSTModel(
        parent,
        seen,
        class_count=int(tensors["role_sentence_embeds"].size(0)),
        hidden_dim=int(config["hidden_dim"]),
        max_transport_step=float(config["max_transport_step"]),
        grid_points=int(config["grid_points"]),
    ).to(device)


class GroupwiseSchedule:
    """Set the exact LR used by each optimizer update."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_updates: int,
        warmup_updates: int,
        tg_min_multiplier: float,
        gate_min_multiplier: float,
    ):
        if len(optimizer.param_groups) != 2:
            raise ValueError("GTD调度器固定要求TG/Gate两个参数组。")
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.total_updates = int(total_updates)
        self.warmup_updates = int(warmup_updates)
        self.tg_min_multiplier = float(tg_min_multiplier)
        self.gate_min_multiplier = float(gate_min_multiplier)
        self.last_update = 0
        self._validate()

    def _validate(self):
        if (
            self.total_updates <= 0
            or not 1 < self.warmup_updates < self.total_updates
            or not 0.0 < self.tg_min_multiplier <= 1.0
            or not 0.0 < self.gate_min_multiplier < 1.0
        ):
            raise ValueError("GTD调度器边界错误。")

    def multipliers(self, update: int) -> tuple[float, float]:
        import math

        step = int(update)
        if not 1 <= step <= self.total_updates:
            raise ValueError("GTD scheduler update超出训练边界。")
        if step <= self.warmup_updates:
            tg = 1.0
            progress = (step - 1) / (self.warmup_updates - 1)
            gate = self.gate_min_multiplier + (1.0 - self.gate_min_multiplier) * progress
        else:
            progress = (step - self.warmup_updates) / (
                self.total_updates - self.warmup_updates
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            tg = self.tg_min_multiplier + (1.0 - self.tg_min_multiplier) * cosine
            gate = self.gate_min_multiplier + (1.0 - self.gate_min_multiplier) * cosine
        return tg, gate

    def set_for_update(self, update: int) -> None:
        values = self.multipliers(update)
        for group, base, value in zip(self.optimizer.param_groups, self.base_lrs, values):
            group["lr"] = base * value
        self.last_update = int(update)

    def state_dict(self) -> dict:
        return {
            "base_lrs": list(self.base_lrs),
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "tg_min_multiplier": self.tg_min_multiplier,
            "gate_min_multiplier": self.gate_min_multiplier,
            "last_update": self.last_update,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "base_lrs": list(self.base_lrs),
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "tg_min_multiplier": self.tg_min_multiplier,
            "gate_min_multiplier": self.gate_min_multiplier,
        }
        if not isinstance(state, dict) or any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("GTD scheduler checkpoint身份错误。")
        last_update = int(state.get("last_update", -1))
        if last_update == 0:
            self.last_update = 0
            return
        self.set_for_update(last_update)


def rank_modulo_class_folds(
    seenclasses: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic three-fold split for any supported dataset class axis."""
    classes = torch.as_tensor(seenclasses).detach().cpu().long().sort().values
    if classes.numel() < 6 or classes.unique().numel() != classes.numel():
        raise ValueError("GTD三折要求至少6个唯一seen类。")
    ranks = torch.arange(classes.numel())
    folds = []
    for fold_id in range(3):
        pseudo_unseen = classes[ranks.remainder(3) == fold_id]
        pseudo_seen = classes[ranks.remainder(3) != fold_id]
        folds.append((pseudo_seen, pseudo_unseen))
    return folds


def teacher_refresh_updates(
    *,
    train_count: int = TRAIN_COUNT,
    nominal_epochs: int = NOMINAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, ...]:
    interval = int(train_count) // int(batch_size)
    if interval <= 0 or int(nominal_epochs) <= 0:
        raise ValueError("GTD teacher refresh预算必须为正数。")
    return tuple(1 + interval * index for index in range(int(nominal_epochs)))


def evaluation_updates(
    *,
    train_count: int = TRAIN_COUNT,
    nominal_epochs: int = NOMINAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, ...]:
    interval = int(train_count) // int(batch_size)
    total = int(train_count) * int(nominal_epochs) // int(batch_size)
    values = sorted(
        set(
            [interval * index for index in range(1, int(nominal_epochs) + 1)]
            + [total]
        )
    )
    if len(values) != int(nominal_epochs) + 1 or values[-1] != total:
        raise RuntimeError("GTD评估点必须是每名义epoch加最终尾点。")
    return tuple(values)


def gtd_screen_decision(delta_h: float, gap: float) -> str:
    if float(gap) >= 8.0 or float(delta_h) < 0.8:
        return "drop_fixed_150"
    if float(delta_h) < 1.0:
        return "trigger_try020_static_below1"
    return "pending_matched_try020_comparison"


def gtd_screen_outcome(delta_h: float, gap: float) -> dict[str, str | bool | None]:
    decision = gtd_screen_decision(delta_h, gap)
    triggered = decision != "drop_fixed_150"
    return {
        "decision": decision,
        "matched_control_triggered": triggered,
        "static_support_passed": decision == "pending_matched_try020_comparison",
        "matched_comparison_required": MATCHED_CONTROL_ID if triggered else None,
    }


def tensor_mapping_sha256(mapping: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(mapping):
        tensor = mapping[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def teacher_packages_sha256(packages: list[dict[str, torch.Tensor]]) -> str:
    if len(packages) != 3:
        raise ValueError("GTD teacher package SHA固定要求三个fold。")
    flattened = {
        f"fold{fold}.{name}": tensor
        for fold, package in enumerate(packages)
        for name, tensor in package.items()
    }
    return tensor_mapping_sha256(flattened)


def teacher_packages_to_cpu(
    packages: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    return [
        {name: tensor.detach().cpu().clone() for name, tensor in package.items()}
        for package in packages
    ]


def teacher_packages_to_device(
    packages: list[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    if len(packages) != 3:
        raise ValueError("GTD checkpoint teacher package固定要求三个fold。")
    return [
        {name: tensor.to(device) for name, tensor in package.items()}
        for package in packages
    ]


def teacher_refresh_record(
    *,
    update: int,
    model: GTDTSTModel,
    packages: list[dict[str, torch.Tensor]],
    folds: list[tuple[torch.Tensor, torch.Tensor]],
    valid_updates: tuple[int, ...] = TEACHER_REFRESH_UPDATES,
) -> dict:
    if int(update) not in valid_updates or len(packages) != 3 or len(folds) != 3:
        raise ValueError("GTD teacher refresh update/fold边界错误。")
    class_ids = torch.cat([package["class_ids"].detach().cpu() for package in packages])
    order = class_ids.argsort(stable=True)
    target_ratio = torch.cat([package["target_ratio"].detach().cpu() for package in packages])[order]
    target_theta = torch.cat([package["target_theta"].detach().cpu() for package in packages])[order]
    move_mask = torch.cat([package["move_mask"].detach().cpu() for package in packages])[order]
    valid = torch.cat([package["valid"].detach().cpu() for package in packages])[order]
    gain = torch.cat([package["oracle_gain"].detach().cpu() for package in packages])[order]
    class_ids = class_ids[order]
    return {
        "update": int(update),
        "model_state_sha256": tensor_mapping_sha256(dict(model.state_dict())),
        "package_sha256": teacher_packages_sha256(packages),
        "fold_pseudo_unseen_class_ids": [
            [int(value) for value in pseudo_unseen.detach().cpu().sort().values]
            for _, pseudo_unseen in folds
        ],
        "class_ids": [int(value) for value in class_ids],
        "target_ratio": [float(value) for value in target_ratio],
        "target_theta_radians": [float(value) for value in target_theta],
        "move_mask": [bool(value) for value in move_mask],
        "valid_direction_mask": [bool(value) for value in valid],
        "oracle_gain": [float(value) for value in gain],
    }


def refresh_oracle_targets(
    model: GTDTSTModel,
    visual_centroids: torch.Tensor,
    folds: list[tuple[torch.Tensor, torch.Tensor]],
    theta_penalty: float,
) -> list[dict[str, torch.Tensor]]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        packages = [
            model.oracle_targets(
                visual_centroids,
                pseudo_seen,
                pseudo_unseen,
                theta_penalty=theta_penalty,
            )
            for pseudo_seen, pseudo_unseen in folds
        ]
    model.train(was_training)
    coverage = torch.cat([item["class_ids"].cpu() for item in packages]).sort().values
    if not torch.equal(coverage, model.seen_classes.cpu()):
        raise RuntimeError("GTD oracle target没有完整覆盖全部seen类。")
    return packages


@torch.no_grad()
def _predict(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    class_ids: torch.Tensor | None,
    batch_size: int = 256,
) -> torch.Tensor:
    axis = (
        torch.arange(prototypes.size(0), device=device)
        if class_ids is None
        else class_ids.to(device).long()
    )
    candidates = F.normalize(prototypes.index_select(0, axis).float(), dim=-1)
    predictions = []
    for start in range(0, features.size(0), int(batch_size)):
        image = F.normalize(features[start : start + int(batch_size)].to(device).float(), dim=-1)
        logits = image @ candidates.T * scale
        if not torch.isfinite(logits).all():
            raise ValueError("GTD评估logits包含NaN/Inf。")
        predictions.append(axis[logits.argmax(dim=1)].cpu())
    return torch.cat(predictions)


def _transitions(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    old = before.eq(labels.cpu())
    new = after.eq(labels.cpu())
    return {
        "corrected_wrong_to_right": int((~old & new).sum()),
        "damaged_right_to_wrong": int((old & ~new).sum()),
        "net_correct": int(new.sum() - old.sum()),
    }


@torch.no_grad()
def evaluate(
    model: GTDTSTModel,
    tensors: dict[str, torch.Tensor],
    packages: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    gtd_enabled: bool = True,
    frozen_baseline: dict[str, torch.Tensor] | None = None,
    return_predictions: bool = False,
) -> dict:
    model.eval()
    bundle = model.prototype_bundle()
    final = bundle["final"] if bool(gtd_enabled) else bundle["parent"]
    parent = bundle["parent"]
    scale = model.scale().detach()
    seenclasses = model.seen_classes.cpu()
    unseenclasses = model.unseen_classes.cpu()
    predictions = {
        "seen": _predict(tensors["test_seen_features"], final, scale, device, None),
        "unseen": _predict(tensors["test_unseen_features"], final, scale, device, None),
        "zs": _predict(
            tensors["test_unseen_features"], final, scale, device, unseenclasses
        ),
    }
    joint_parent = {
        "seen": _predict(tensors["test_seen_features"], parent, scale, device, None),
        "unseen": _predict(tensors["test_unseen_features"], parent, scale, device, None),
        "zs": _predict(
            tensors["test_unseen_features"], parent, scale, device, unseenclasses
        ),
    }
    seen_labels = tensors["test_seen_labels"].long()
    unseen_labels = tensors["test_unseen_labels"].long()
    s = 100.0 * per_class_accuracy(seen_labels, predictions["seen"], seenclasses)
    u = 100.0 * per_class_accuracy(unseen_labels, predictions["unseen"], unseenclasses)
    z = 100.0 * per_class_accuracy(unseen_labels, predictions["zs"], unseenclasses)
    h = 2.0 * s * u / (s + u) if s + u else 0.0
    parent_s = 100.0 * per_class_accuracy(seen_labels, joint_parent["seen"], seenclasses)
    parent_u = 100.0 * per_class_accuracy(unseen_labels, joint_parent["unseen"], unseenclasses)
    parent_z = 100.0 * per_class_accuracy(unseen_labels, joint_parent["zs"], unseenclasses)
    parent_h = (
        2.0 * parent_s * parent_u / (parent_s + parent_u)
        if parent_s + parent_u
        else 0.0
    )
    result = {
        "U": u,
        "S": s,
        "H": h,
        "ZS": z,
        "module_off_metrics": {
            "U": parent_u,
            "S": parent_s,
            "H": parent_h,
            "ZS": parent_z,
        },
        "full_minus_off_delta": {
            "U": u - parent_u,
            "S": s - parent_s,
            "H": h - parent_h,
            "ZS": z - parent_z,
        },
        "gtd_residual_transitions_vs_joint_tg": {
            "seen": _transitions(joint_parent["seen"], predictions["seen"], seen_labels),
            "unseen": _transitions(
                joint_parent["unseen"], predictions["unseen"], unseen_labels
            ),
            "zs": _transitions(joint_parent["zs"], predictions["zs"], unseen_labels),
        },
        "diagnostics": model.diagnostics(packages),
    }
    if frozen_baseline is not None:
        result["full_model_transitions_vs_frozen_tg"] = {
            "seen": _transitions(frozen_baseline["seen"], predictions["seen"], seen_labels),
            "unseen": _transitions(
                frozen_baseline["unseen"], predictions["unseen"], unseen_labels
            ),
            "zs": _transitions(frozen_baseline["zs"], predictions["zs"], unseen_labels),
        }
    if return_predictions:
        result["_predictions"] = predictions
    return result


def prepare_run_directory(
    output_dir: Path,
    resume_from: Path | None,
    config: dict,
) -> tuple[Path, str]:
    if resume_from is None:
        return prepare_output_dir(output_dir), "x"
    if not output_dir.is_absolute() or not resume_from.is_absolute():
        raise ValueError("GTD resume的output-dir和checkpoint必须是绝对路径。")
    resolved_output = output_dir.resolve()
    resolved_resume = resume_from.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if repo_root == resolved_output or repo_root in resolved_output.parents:
        raise ValueError("GTD output-dir必须位于Git仓库外。")
    if not resolved_output.is_dir() or resolved_resume != resolved_output / "checkpoint_last.pth":
        raise ValueError("GTD只允许从同一RUN目录的checkpoint_last.pth续训。")
    snapshot = resolved_output / "config.snapshot.yaml"
    if not snapshot.is_file() or yaml.safe_load(snapshot.read_text(encoding="utf-8")) != config:
        raise ValueError("GTD resume config snapshot与当前配置不一致。")
    return resolved_output, "a"


def next_teacher_refresh_after(
    update: int,
    refresh_updates: tuple[int, ...] = TEACHER_REFRESH_UPDATES,
) -> int | None:
    return next((value for value in refresh_updates if value > int(update)), None)


def restore_rng_states(checkpoint: dict, generator: torch.Generator) -> None:
    generator.set_state(checkpoint["batch_generator_state"])
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    cuda_states = checkpoint["cuda_rng_state_all"]
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError("GTD resume CUDA RNG设备数量不一致。")
    torch.cuda.set_rng_state_all(cuda_states)


def run(
    config_path: Path,
    output_dir: Path,
    expected_commit: str,
    resume_from: Path | None = None,
) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("GTD expected-commit与当前干净HEAD不一致。")
    config, config_sha = load_config(config_path)
    spec = DATASET_SPECS[config["dataset"]]
    train_count = int(spec["train_count"])
    seen_count = int(spec["seen_count"])
    class_count = int(spec["class_count"])
    total_updates = int(config["total_updates"])
    refresh_updates = teacher_refresh_updates(
        train_count=train_count,
        nominal_epochs=int(config["nominal_epochs"]),
        batch_size=int(config["batch_size"]),
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GTD正式训练要求CUDA。")
    tensors = load_assets(config)
    labels = tensors["train_labels"].long()
    seen = torch.unique(labels, sorted=True)
    if labels.numel() != train_count or seen.numel() != seen_count:
        raise ValueError("GTD trainval样本或seen类别数量不符合数据集身份。")
    output_dir, log_mode = prepare_run_directory(output_dir, resume_from, config)
    if resume_from is None:
        (output_dir / "config.snapshot.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    log_handle = (output_dir / "training.log").open(log_mode, encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        reproducibility = configure_reproducibility(
            int(config["random_seed"]), strict_determinism=True, deterministic_warn_only=False
        )
        print(f"GTD RUN={config['experiment_id']} commit={code_commit} config_sha={config_sha}")
        model = build_model(config, tensors, device)
        gtd_enabled = config["condition_id"] != "TG_SCRATCH_FIXED150"
        scratch_initialization = config["tg_checkpoint"] is None
        initial_tg_state_sha256 = tensor_mapping_sha256(dict(model.parent.state_dict()))
        train_features = tensors["train_features"].to(device).float()
        train_labels = labels.to(device)
        seen_device = seen.to(device)
        global_to_seen = torch.full((class_count,), -1, dtype=torch.long, device=device)
        global_to_seen[seen_device] = torch.arange(seen_count, device=device)
        visual_centroids = h1.visual_centroids(
            tensors["train_features"], labels, seen
        ).to(device)
        folds = rank_modulo_class_folds(seen)

        parent_parameters = list(model.parent.parameters())
        gate_parameters = list(model.gate.parameters())
        if {id(p) for p in parent_parameters}.intersection(id(p) for p in gate_parameters):
            raise RuntimeError("GTD TG/Gate参数组不得重叠。")
        optimizer = torch.optim.Adam(
            [
                {"params": parent_parameters, "lr": float(config["tg_learning_rate"])},
                {"params": gate_parameters, "lr": float(config["gate_learning_rate"])},
            ],
            weight_decay=float(config["weight_decay"]),
        )
        warmup_updates = (
            train_count
            * int(config["gate_warmup_epochs"])
            // int(config["batch_size"])
        )
        scheduler = GroupwiseSchedule(
            optimizer,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            tg_min_multiplier=float(config["tg_min_learning_rate"])
            / float(config["tg_learning_rate"]),
            gate_min_multiplier=float(config["gate_min_learning_rate"])
            / float(config["gate_learning_rate"]),
        )
        batch_generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
        parent_metrics = config["parent_metrics_percent"]
        if resume_from is None:
            packages = refresh_oracle_targets(
                model, visual_centroids, folds, float(config["theta_penalty"])
            )
            teacher_history = [
                teacher_refresh_record(
                    update=refresh_updates[0],
                    model=model,
                    packages=packages,
                    folds=folds,
                    valid_updates=refresh_updates,
                )
            ]
            next_teacher_refresh = refresh_updates[1]
            initial = evaluate(
                model,
                tensors,
                packages,
                device,
                gtd_enabled=gtd_enabled,
                return_predictions=True,
            )
            frozen_baseline = initial.pop("_predictions")
            zero_transitions = {
                split: _transitions(prediction, prediction, label)
                for split, prediction, label in (
                    ("seen", frozen_baseline["seen"], tensors["test_seen_labels"]),
                    ("unseen", frozen_baseline["unseen"], tensors["test_unseen_labels"]),
                    ("zs", frozen_baseline["zs"], tensors["test_unseen_labels"]),
                )
            }
            initial["full_model_transitions_vs_frozen_tg"] = zero_transitions
            if parent_metrics is None:
                parent_metrics = {
                    metric: float(initial[metric]) for metric in ("U", "S", "H", "ZS")
                }
            else:
                for metric in ("U", "S", "H", "ZS"):
                    if abs(float(initial[metric]) - float(parent_metrics[metric])) > 1e-6:
                        raise ValueError(f"GTD theta0未复现父TG {metric}。")
            initial.update(
                {
                    "evaluation_index": 0,
                    "update": 0,
                    "delta_U": 0.0,
                    "delta_S": 0.0,
                    "delta_H": 0.0,
                    "delta_ZS": 0.0,
                }
            )
            history = [initial]
            best_metrics = copy.deepcopy(initial)
            best_state = copy.deepcopy(model.state_dict())
            best_update = 0
            best_zs = {
                "ZS": float(initial["ZS"]),
                "update": 0,
                "metrics": copy.deepcopy(initial),
            }
            start_update = 1
        else:
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("experiment_id") != config["experiment_id"]
                or checkpoint.get("code_commit") != code_commit
                or checkpoint.get("config_sha256") != config_sha
                or checkpoint.get("initial_tg_state_sha256") != initial_tg_state_sha256
                or not 0 < int(checkpoint.get("update", 0)) < total_updates
            ):
                raise ValueError("GTD resume checkpoint RUN身份或update错误。")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            packages = teacher_packages_to_device(checkpoint["teacher_packages"], device)
            teacher_history = checkpoint["teacher_refresh_history"]
            next_teacher_refresh = checkpoint["next_teacher_refresh_update"]
            history = checkpoint["history"]
            parent_metrics = checkpoint_parent_metrics(
                checkpoint, config["parent_metrics_percent"]
            )
            frozen_baseline = checkpoint["frozen_baseline_predictions"]
            best_metrics = checkpoint["best_metrics"]
            best_state = checkpoint["best_model_state_dict"]
            best_update = int(checkpoint["best_update"])
            best_zs = checkpoint["best_zs_observation"]
            expected_next = next_teacher_refresh_after(
                int(checkpoint["update"]), refresh_updates
            )
            if (
                next_teacher_refresh != expected_next
                or len(teacher_history)
                != sum(value <= int(checkpoint["update"]) for value in refresh_updates)
                or teacher_packages_sha256(packages) != teacher_history[-1]["package_sha256"]
            ):
                raise ValueError("GTD resume teacher package/history/next refresh不一致。")
            start_update = int(checkpoint["update"]) + 1
            reproducibility = checkpoint["reproducibility"]
            restore_rng_states(checkpoint, batch_generator)
            print(f"GTD resume_from={resume_from} next_update={start_update}")
        eval_set = set(
            evaluation_updates(
                train_count=train_count,
                nominal_epochs=int(config["nominal_epochs"]),
                batch_size=int(config["batch_size"]),
            )
        )
        interval_sums: dict[str, float] = {}
        interval_steps = 0
        for update in range(start_update, total_updates + 1):
            if next_teacher_refresh is not None and update == int(next_teacher_refresh):
                packages = refresh_oracle_targets(
                    model, visual_centroids, folds, float(config["theta_penalty"])
                )
                teacher_history.append(
                    teacher_refresh_record(
                        update=update,
                        model=model,
                        packages=packages,
                        folds=folds,
                        valid_updates=refresh_updates,
                    )
                )
                next_teacher_refresh = next_teacher_refresh_after(
                    update, refresh_updates
                )
            model.train()
            scheduler.set_for_update(update)
            indices_cpu = torch.randperm(train_count, generator=batch_generator)[
                : int(config["batch_size"])
            ]
            indices = indices_cpu.to(device)
            images = train_features.index_select(0, indices)
            targets = global_to_seen.index_select(0, train_labels.index_select(0, indices))
            fold_package = packages[(update - 1) % 3]
            optimizer.zero_grad(set_to_none=True)
            parent_logits = model.parent.logits(images, seen_device)
            ce = F.cross_entropy(parent_logits, targets)
            topology = model.parent.topology_loss()
            raw_ratio = model.gate.raw_ratio(fold_package["features"])
            gate_loss = F.smooth_l1_loss(raw_ratio, fold_package["target_ratio"])
            total = (
                ce
                + float(config["topology_weight"]) * topology
                + float(config["gate_loss_weight"]) * gate_loss
            )
            if not torch.isfinite(total):
                raise FloatingPointError("GTD训练loss包含NaN/Inf。")
            total.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"GTD梯度包含NaN/Inf：{name}")
            optimizer.step()
            interval_steps += 1
            values = {
                "total": total,
                "ce": ce,
                "topology": topology,
                "gate_smooth_l1": gate_loss,
                "target_ratio_mean": fold_package["target_ratio"].mean(),
                "oracle_gain_mean": fold_package["oracle_gain"].mean(),
                "tg_lr": torch.tensor(optimizer.param_groups[0]["lr"], device=device),
                "gate_lr": torch.tensor(optimizer.param_groups[1]["lr"], device=device),
            }
            for name, value in values.items():
                interval_sums[name] = interval_sums.get(name, 0.0) + float(value.detach())
            if update not in eval_set:
                continue
            metrics = evaluate(
                model,
                tensors,
                packages,
                device,
                gtd_enabled=gtd_enabled,
                frozen_baseline=frozen_baseline,
            )
            metrics.update(
                {
                    "evaluation_index": len(history),
                    "update": update,
                    "delta_U": float(metrics["U"]) - float(parent_metrics["U"]),
                    "delta_S": float(metrics["S"]) - float(parent_metrics["S"]),
                    "delta_H": float(metrics["H"]) - float(parent_metrics["H"]),
                    "delta_ZS": float(metrics["ZS"]) - float(parent_metrics["ZS"]),
                    "train": {
                        name: value / interval_steps for name, value in interval_sums.items()
                    },
                }
            )
            history.append(metrics)
            print(
                f"eval={metrics['evaluation_index']} update={update} "
                f"U={metrics['U']:.6f} S={metrics['S']:.6f} H={metrics['H']:.6f} "
                f"ZS={metrics['ZS']:.6f} deltaH={metrics['delta_H']:.6f}"
            )
            interval_sums = {}
            interval_steps = 0
            if float(metrics["H"]) > float(best_metrics["H"]):
                best_metrics = copy.deepcopy(metrics)
                best_state = copy.deepcopy(model.state_dict())
                best_update = update
            if float(metrics["ZS"]) > float(best_zs["ZS"]):
                best_zs = {"ZS": float(metrics["ZS"]), "update": update, "metrics": copy.deepcopy(metrics)}
            checkpoint = {
                "experiment_id": config["experiment_id"],
                "code_commit": code_commit,
                "config": config,
                "config_sha256": config_sha,
                "parent_metrics_percent": parent_metrics,
                "initial_tg_state_sha256": initial_tg_state_sha256,
                "update": update,
                "evaluation_index": metrics["evaluation_index"],
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_update": best_update,
                "best_metrics": best_metrics,
                "best_model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
                "best_zs_observation": best_zs,
                "teacher_packages": teacher_packages_to_cpu(packages),
                "teacher_refresh_history": teacher_history,
                "next_teacher_refresh_update": next_teacher_refresh,
                "frozen_baseline_predictions": frozen_baseline,
                "batch_generator_state": batch_generator.get_state(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "history": history,
                "reproducibility": reproducibility,
            }
            atomic_torch_save(output_dir / "checkpoint_last.pth", checkpoint)
        expected_history_length = int(config["nominal_epochs"]) + 2
        if len(history) != expected_history_length or history[-1]["update"] != total_updates:
            raise RuntimeError("GTD完整训练必须保存初始点、逐epoch点和最终尾点。")
        if (
            len(teacher_history) != int(config["nominal_epochs"])
            or [row["update"] for row in teacher_history] != list(refresh_updates)
            or next_teacher_refresh is not None
        ):
            raise RuntimeError("GTD完整训练必须逐名义epoch保存确定性teacher refresh。")
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "experiment_id": config["experiment_id"],
                "code_commit": code_commit,
                "config_sha256": config_sha,
                "best_update": best_update,
                "best_metrics": best_metrics,
                "model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
            },
        )
        delta_h = float(best_metrics["H"]) - float(parent_metrics["H"])
        gap = abs(float(best_metrics["U"]) - float(best_metrics["S"]))
        if scratch_initialization:
            screen = {
                "matched_comparison_required": bool(gtd_enabled),
                "matched_control_triggered": bool(gtd_enabled),
                "static_support_passed": None,
            }
            decision = (
                "pending_matched_scratch_control"
                if gtd_enabled
                else "complete_scratch_tg_control"
            )
        else:
            screen = gtd_screen_outcome(delta_h, gap)
            decision = str(screen["decision"])
        atomic_write_json(output_dir / "evaluation_history.json", {"rows": history})
        atomic_write_json(
            output_dir / "teacher_refresh_history.json",
            {"count": len(teacher_history), "rows": teacher_history},
        )
        evaluation_history_sha = sha256_file(output_dir / "evaluation_history.json")
        teacher_history_sha = sha256_file(output_dir / "teacher_refresh_history.json")
        checkpoint_last_sha = sha256_file(output_dir / "checkpoint_last.pth")
        result = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "dataset": config["dataset"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "parent_metrics_percent": parent_metrics,
            "initialization_mode": (
                "random_tg_seed7" if scratch_initialization else "tg_checkpoint_warm_start"
            ),
            "gtd_enabled": bool(gtd_enabled),
            "initial_tg_state_sha256": initial_tg_state_sha256,
            "best_metrics": best_metrics,
            "best_update": best_update,
            "best_delta_H": delta_h,
            "module_off_metrics": best_metrics["module_off_metrics"],
            "best_full_minus_off_delta": best_metrics["full_minus_off_delta"],
            "best_gap_U_S": gap,
            "matched_comparison_required": screen["matched_comparison_required"],
            "matched_control_triggered": screen["matched_control_triggered"],
            "static_support_passed": screen["static_support_passed"],
            "old_tg_screen_threshold_H": 0.8,
            "independent_support_threshold_H": float(config["required_delta_h"]),
            "best_zs_observation": best_zs,
            "stop_reason": "completed_fixed_150",
            "decision": decision,
            "history_length": len(history),
            "target_refresh_count": len(teacher_history),
            "train_count": train_count,
            "seen_count": seen_count,
            "class_count": class_count,
            "total_updates": total_updates,
            "eval_interval_steps": int(config["eval_interval_steps"]),
            "test_used_for_selection": True,
            "test_used_for_hyperparameter_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "human_annotations_used": False,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "asset_id": config["asset_id"],
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "tg_checkpoint_sha256": config["tg_checkpoint_sha256"],
            "checkpoint_last_sha256": checkpoint_last_sha,
            "evaluation_history_sha256": evaluation_history_sha,
            "teacher_refresh_history_sha256": teacher_history_sha,
            "final_teacher_package_sha256": teacher_history[-1]["package_sha256"],
            "final_teacher_model_state_sha256": teacher_history[-1]["model_state_sha256"],
        }
        atomic_write_json(output_dir / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return result
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.resume_from)


if __name__ == "__main__":
    main()
