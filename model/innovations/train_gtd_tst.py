"""Fixed-150 one-stage GTD-TST training from the exact V3 TG checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.elpt import fixed_class_folds
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
TRAIN_COUNT = 7057
SEEN_COUNT = 150
CLASS_COUNT = 200
NOMINAL_EPOCHS = 150
BATCH_SIZE = 50
EVAL_INTERVAL = 141
TOTAL_UPDATES = TRAIN_COUNT * NOMINAL_EPOCHS // BATCH_SIZE
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
    if (
        config["schema_version"] != SCHEMA
        or config["experiment_id"] != "V3-TRY-022"
        or config["framework_id"] != "FRAMEWORK-V3-EXPLORATION"
        or config["dataset"] != "CUB"
        or config["condition_id"] != "TG_PLUS_GTD_TST_FIXED150"
        or int(config["random_seed"]) != 7
        or int(config["batch_size"]) != BATCH_SIZE
        or int(config["nominal_epochs"]) != NOMINAL_EPOCHS
        or int(config["total_updates"]) != TOTAL_UPDATES
        or int(config["eval_interval_steps"]) != EVAL_INTERVAL
        or float(config["tg_learning_rate"]) != 1e-5
        or float(config["gate_learning_rate"]) != 1e-4
        or float(config["tg_min_learning_rate"]) != 1e-6
        or float(config["gate_min_learning_rate"]) != 1e-5
        or int(config["gate_warmup_epochs"]) != 5
        or float(config["weight_decay"]) != 1e-4
        or float(config["topology_weight"]) != 0.1
        or float(config["gate_loss_weight"]) != 1.0
        or int(config["hidden_dim"]) != 16
        or int(config["grid_points"]) != 33
        or float(config["theta_penalty"]) != 0.1
        or float(config["max_transport_step"]) != 1.5
        or config["early_stopping_enabled"] is not False
        or config["parent_metrics_percent"] != parent
        or float(config["required_delta_h"]) != 1.0
        or float(config["max_us_gap"]) != 8.0
        or config["human_annotations_used"] is not False
        or config["test_used_for_selection"] is not True
        or config["test_used_for_hyperparameter_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("GTD首轮身份、训练参数、硬门槛或披露边界错误。")
    return config, sha256_file(path)


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
    if (
        manifest.get("schema_version") != "gzsl-paper.clip-assets.v1"
        or manifest.get("asset_id") != config["asset_id"]
        or manifest.get("dataset") != "CUB"
        or manifest.get("counts")
        != {"train": 7057, "test_seen": 1764, "test_unseen": 2967}
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
        "train_features": (7057, 768),
        "train_labels": (7057,),
        "test_seen_features": (1764, 768),
        "test_seen_labels": (1764,),
        "test_unseen_features": (2967, 768),
        "test_unseen_labels": (2967,),
        "role_sentence_embeds": (200, 8, 768),
    }
    for name, expected in expected_shapes.items():
        if tuple(tensors[name].shape) != expected:
            raise ValueError(f"GTD资产{name} shape错误：{tuple(tensors[name].shape)}")
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
            or not 0.0 < self.tg_min_multiplier < 1.0
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


def evaluation_updates() -> tuple[int, ...]:
    values = sorted(
        set([EVAL_INTERVAL * index for index in range(1, NOMINAL_EPOCHS + 1)] + [TOTAL_UPDATES])
    )
    if len(values) != 151 or values[-2:] != [21150, 21171]:
        raise RuntimeError("GTD评估点必须是141×1..150加21171。")
    return tuple(values)


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
        raise RuntimeError("GTD oracle target没有完整覆盖150个seen类。")
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
        torch.arange(CLASS_COUNT, device=device)
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
    frozen_baseline: dict[str, torch.Tensor] | None = None,
    return_predictions: bool = False,
) -> dict:
    model.eval()
    bundle = model.prototype_bundle()
    final = bundle["final"]
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
    result = {
        "U": u,
        "S": s,
        "H": h,
        "ZS": z,
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


def run(config_path: Path, output_dir: Path, expected_commit: str) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("GTD expected-commit与当前干净HEAD不一致。")
    config, config_sha = load_config(config_path)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GTD正式训练要求CUDA。")
    tensors = load_assets(config)
    labels = tensors["train_labels"].long()
    seen = torch.unique(labels, sorted=True)
    if labels.numel() != TRAIN_COUNT or seen.numel() != SEEN_COUNT:
        raise ValueError("GTD固定CUB 7057张trainval和150个seen类。")
    output_dir = prepare_output_dir(output_dir)
    (output_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        reproducibility = configure_reproducibility(
            int(config["random_seed"]), strict_determinism=True, deterministic_warn_only=False
        )
        print(f"GTD RUN={config['experiment_id']} commit={code_commit} config_sha={config_sha}")
        model = build_model(config, tensors, device)
        train_features = tensors["train_features"].to(device).float()
        train_labels = labels.to(device)
        seen_device = seen.to(device)
        global_to_seen = torch.full((CLASS_COUNT,), -1, dtype=torch.long, device=device)
        global_to_seen[seen_device] = torch.arange(SEEN_COUNT, device=device)
        visual_centroids = h1.visual_centroids(
            tensors["train_features"], labels, seen
        ).to(device)
        folds = fixed_class_folds(seen)

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
        warmup_updates = TRAIN_COUNT * int(config["gate_warmup_epochs"]) // BATCH_SIZE
        scheduler = GroupwiseSchedule(
            optimizer,
            total_updates=TOTAL_UPDATES,
            warmup_updates=warmup_updates,
            tg_min_multiplier=float(config["tg_min_learning_rate"])
            / float(config["tg_learning_rate"]),
            gate_min_multiplier=float(config["gate_min_learning_rate"])
            / float(config["gate_learning_rate"]),
        )
        batch_generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
        packages = refresh_oracle_targets(
            model, visual_centroids, folds, float(config["theta_penalty"])
        )
        target_refresh_count = 1
        initial = evaluate(model, tensors, packages, device, return_predictions=True)
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
        parent_metrics = config["parent_metrics_percent"]
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
        best_zs = {"ZS": float(initial["ZS"]), "update": 0, "metrics": copy.deepcopy(initial)}
        eval_set = set(evaluation_updates())
        interval_sums: dict[str, float] = {}
        interval_steps = 0
        for update in range(1, TOTAL_UPDATES + 1):
            # Refresh exactly 150 times: update 1 and starts of the next 149 report intervals.
            if update > 1 and (update - 1) % EVAL_INTERVAL == 0 and (update - 1) // EVAL_INTERVAL < 150:
                packages = refresh_oracle_targets(
                    model, visual_centroids, folds, float(config["theta_penalty"])
                )
                target_refresh_count += 1
            model.train()
            scheduler.set_for_update(update)
            indices_cpu = torch.randperm(TRAIN_COUNT, generator=batch_generator)[:BATCH_SIZE]
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
                "update": update,
                "evaluation_index": metrics["evaluation_index"],
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_update": best_update,
                "best_metrics": best_metrics,
                "best_model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
                "best_zs_observation": best_zs,
                "target_refresh_count": target_refresh_count,
                "history": history,
                "reproducibility": reproducibility,
            }
            atomic_torch_save(output_dir / "checkpoint_last.pth", checkpoint)
        if len(history) != 152 or history[-1]["update"] != TOTAL_UPDATES:
            raise RuntimeError("GTD完整150轮必须保存152个评估点并结束于21171。")
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
        decision = (
            "promote_to_fixed_200_and_matched_ablation"
            if delta_h >= float(config["required_delta_h"]) and gap < float(config["max_us_gap"])
            else "drop_fixed_150"
        )
        result = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "parent_metrics_percent": parent_metrics,
            "best_metrics": best_metrics,
            "best_update": best_update,
            "best_delta_H": delta_h,
            "best_gap_U_S": gap,
            "best_zs_observation": best_zs,
            "stop_reason": "completed_fixed_150",
            "decision": decision,
            "history_length": len(history),
            "target_refresh_count": target_refresh_count,
            "test_used_for_selection": True,
            "test_used_for_hyperparameter_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "human_annotations_used": False,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", result)
        atomic_write_json(output_dir / "evaluation_history.json", {"rows": history})
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
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit)


if __name__ == "__main__":
    main()
