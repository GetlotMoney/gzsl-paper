"""Formal one-stage fixed-200 TG+GTD+PECV training on CUB."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.pecv_gtd import PECVGTDModel, stable_topk_ids
from model.innovations.train_gtd_tst import (
    TeeStream,
    build_model as build_gtd_model,
    evaluation_updates,
    load_assets,
    next_teacher_refresh_after,
    rank_modulo_class_folds,
    refresh_oracle_targets,
    restore_rng_states,
    teacher_packages_sha256,
    teacher_packages_to_cpu,
    teacher_packages_to_device,
    teacher_refresh_record,
    teacher_refresh_updates,
    tensor_mapping_sha256,
)
from model.tg_vpr_h1 import train as h1
from tools.gzsl_data import per_class_accuracy
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


SCHEMA = "gzsl-paper.v4-pecv-gtd-formal.v1"
TRAIN_COUNT = 7057
SEEN_COUNT = 150
CLASS_COUNT = 200
BATCH_SIZE = 50
NOMINAL_EPOCHS = 200
EVAL_INTERVAL = TRAIN_COUNT // BATCH_SIZE
TOTAL_UPDATES = TRAIN_COUNT * NOMINAL_EPOCHS // BATCH_SIZE
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "idea_id",
    "framework_id",
    "dataset",
    "condition_id",
    "asset_manifest",
    "asset_manifest_sha256",
    "asset_id",
    "tg_checkpoint",
    "gtd_checkpoint",
    "pecv_checkpoint",
    "device",
    "random_seed",
    "batch_size",
    "nominal_epochs",
    "total_updates",
    "eval_interval_steps",
    "tg_learning_rate",
    "tg_min_learning_rate",
    "gtd_learning_rate",
    "gtd_min_learning_rate",
    "pecv_learning_rate",
    "pecv_min_learning_rate",
    "warmup_epochs",
    "weight_decay",
    "topology_weight",
    "gtd_loss_weight",
    "pecv_loss_weight",
    "gtd_hidden_dim",
    "gtd_grid_points",
    "theta_penalty",
    "max_transport_step",
    "pecv_hidden_dim",
    "pecv_max_correction",
    "pecv_top_k",
    "required_delta_h",
    "max_us_gap",
    "test_used_for_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "human_annotations_used",
}


def load_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"PECV formal config fields mismatch; missing={sorted(CONFIG_KEYS-actual)}, "
            f"extra={sorted(actual-CONFIG_KEYS)}"
        )
    condition = config["condition_id"]
    expected_weight = {
        "TG_GTD_PECV_FULL_FIXED200": 1.0,
        "TG_GTD_PECV_PARENT_FIXED200": 0.0,
    }.get(condition)
    expected_run = {
        "TG_GTD_PECV_FULL_FIXED200": "V4-TRY-020-FULL",
        "TG_GTD_PECV_PARENT_FIXED200": "V4-TRY-020-PARENT",
    }.get(condition)
    invalid = (
        config["schema_version"] != SCHEMA
        or config["idea_id"] != "IDEA-182"
        or config["framework_id"] != "FRAMEWORK-V4"
        or config["dataset"] != "CUB"
        or expected_weight is None
        or config["experiment_id"] != expected_run
        or config["tg_checkpoint"] is not None
        or config["gtd_checkpoint"] is not None
        or config["pecv_checkpoint"] is not None
        or int(config["random_seed"]) != 7
        or int(config["batch_size"]) != BATCH_SIZE
        or int(config["nominal_epochs"]) != NOMINAL_EPOCHS
        or int(config["total_updates"]) != TOTAL_UPDATES
        or int(config["eval_interval_steps"]) != EVAL_INTERVAL
        or float(config["tg_learning_rate"]) != 1e-4
        or float(config["tg_min_learning_rate"]) != 1e-4
        or float(config["gtd_learning_rate"]) != 1e-4
        or float(config["gtd_min_learning_rate"]) != 1e-5
        or float(config["pecv_learning_rate"]) != 1e-3
        or float(config["pecv_min_learning_rate"]) != 1e-4
        or int(config["warmup_epochs"]) != 5
        or float(config["weight_decay"]) != 1e-4
        or float(config["topology_weight"]) != 0.1
        or float(config["gtd_loss_weight"]) != 1.0
        or float(config["pecv_loss_weight"]) != expected_weight
        or int(config["gtd_hidden_dim"]) != 16
        or int(config["gtd_grid_points"]) != 33
        or float(config["theta_penalty"]) != 0.1
        or float(config["max_transport_step"]) != 1.5
        or int(config["pecv_hidden_dim"]) != 32
        or float(config["pecv_max_correction"]) != 4.0
        or int(config["pecv_top_k"]) != 5
        or float(config["required_delta_h"]) != 1.0
        or float(config["max_us_gap"]) != 8.0
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
        or config["human_annotations_used"] is not False
    )
    if invalid:
        raise ValueError("PECV formal preregistered identity or values changed.")
    return config, sha256_file(path)


def build_model(config: dict, tensors: dict[str, torch.Tensor], device: torch.device) -> PECVGTDModel:
    gtd_config = {
        "tg_checkpoint": None,
        "tg_checkpoint_sha256": None,
        "hidden_dim": config["gtd_hidden_dim"],
        "max_transport_step": config["max_transport_step"],
        "grid_points": config["gtd_grid_points"],
    }
    return PECVGTDModel(build_gtd_model(gtd_config, tensors, device)).to(device)


class ThreeGroupSchedule:
    """Exact LR schedule for TG, GTD, and PECV parameter groups."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: dict):
        if len(optimizer.param_groups) != 3:
            raise ValueError("PECV formal optimizer requires exactly three groups.")
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.min_multipliers = [
            float(config["tg_min_learning_rate"]) / float(config["tg_learning_rate"]),
            float(config["gtd_min_learning_rate"]) / float(config["gtd_learning_rate"]),
            float(config["pecv_min_learning_rate"]) / float(config["pecv_learning_rate"]),
        ]
        self.total_updates = int(config["total_updates"])
        self.warmup_updates = TRAIN_COUNT * int(config["warmup_epochs"]) // BATCH_SIZE
        self.last_update = 0
        if self.min_multipliers != [1.0, 0.1, 0.1] or not 1 < self.warmup_updates < self.total_updates:
            raise ValueError("PECV formal LR schedule contract changed.")

    def multipliers(self, update: int) -> list[float]:
        step = int(update)
        if not 1 <= step <= self.total_updates:
            raise ValueError("PECV scheduler update out of range.")
        if step <= self.warmup_updates:
            progress = (step - 1) / (self.warmup_updates - 1)
            return [
                1.0,
                self.min_multipliers[1] + 0.9 * progress,
                self.min_multipliers[2] + 0.9 * progress,
            ]
        progress = (step - self.warmup_updates) / (self.total_updates - self.warmup_updates)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [minimum + (1.0 - minimum) * cosine for minimum in self.min_multipliers]

    def set_for_update(self, update: int) -> None:
        for group, base, multiplier in zip(
            self.optimizer.param_groups, self.base_lrs, self.multipliers(update)
        ):
            group["lr"] = base * multiplier
        self.last_update = int(update)

    def state_dict(self) -> dict:
        return {
            "base_lrs": self.base_lrs,
            "min_multipliers": self.min_multipliers,
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "last_update": self.last_update,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("base_lrs", "min_multipliers", "total_updates", "warmup_updates"):
            if state.get(key) != self.state_dict()[key]:
                raise ValueError(f"PECV scheduler resume mismatch: {key}")
        last = int(state.get("last_update", -1))
        if last == 0:
            self.last_update = 0
        else:
            self.set_for_update(last)


def _predict(
    features: torch.Tensor,
    model: PECVGTDModel,
    prototypes: torch.Tensor,
    device: torch.device,
    class_ids: torch.Tensor,
    *,
    pecv_enabled: bool,
    batch_size: int = 256,
) -> torch.Tensor:
    axis = class_ids.to(device).long()
    axis_prototypes = F.normalize(prototypes.index_select(0, axis).float(), dim=-1)
    global_to_axis = torch.full(
        (prototypes.size(0),), -1, dtype=torch.long, device=device
    )
    global_to_axis[axis] = torch.arange(axis.numel(), device=device)
    predictions = []
    for start in range(0, features.size(0), batch_size):
        images = features[start : start + batch_size].to(device).float()
        logits = F.normalize(images, dim=-1) @ axis_prototypes.T * model.scale()
        candidate_ids = stable_topk_ids(logits, axis, 5)
        candidate_positions = global_to_axis.index_select(
            0, candidate_ids.reshape(-1)
        ).reshape_as(candidate_ids)
        parent_scores = logits.gather(1, candidate_positions)
        scores = model.corrected_scores(
            images,
            candidate_ids,
            parent_scores,
            enabled=pecv_enabled,
            prototypes=prototypes,
        )
        predictions.append(
            candidate_ids.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1).cpu()
        )
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
    model: PECVGTDModel,
    tensors: dict[str, torch.Tensor],
    packages: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    pecv_enabled: bool,
    return_predictions: bool = False,
) -> dict:
    model.eval()
    prototypes = model.backbone.prototypes()
    all_classes = torch.arange(CLASS_COUNT)
    seen = model.seen_classes.cpu()
    unseen = model.unseen_classes.cpu()
    off = {
        "seen": _predict(
            tensors["test_seen_features"], model, prototypes, device, all_classes, pecv_enabled=False
        ),
        "unseen": _predict(
            tensors["test_unseen_features"], model, prototypes, device, all_classes, pecv_enabled=False
        ),
        "zs": _predict(
            tensors["test_unseen_features"], model, prototypes, device, unseen, pecv_enabled=False
        ),
    }
    full = off if not pecv_enabled else {
        "seen": _predict(
            tensors["test_seen_features"], model, prototypes, device, all_classes, pecv_enabled=True
        ),
        "unseen": _predict(
            tensors["test_unseen_features"], model, prototypes, device, all_classes, pecv_enabled=True
        ),
        "zs": _predict(
            tensors["test_unseen_features"], model, prototypes, device, unseen, pecv_enabled=True
        ),
    }
    seen_labels = tensors["test_seen_labels"].long()
    unseen_labels = tensors["test_unseen_labels"].long()

    def metrics(predictions: dict[str, torch.Tensor]) -> dict[str, float]:
        s = 100.0 * per_class_accuracy(seen_labels, predictions["seen"], seen)
        u = 100.0 * per_class_accuracy(unseen_labels, predictions["unseen"], unseen)
        zs = 100.0 * per_class_accuracy(unseen_labels, predictions["zs"], unseen)
        h = 2.0 * s * u / (s + u) if s + u else 0.0
        return {"U": u, "S": s, "H": h, "ZS": zs}

    full_metrics = metrics(full)
    off_metrics = metrics(off)
    result = {
        **full_metrics,
        "module_off_metrics": off_metrics,
        "full_minus_off_delta": {
            name: full_metrics[name] - off_metrics[name] for name in ("U", "S", "H", "ZS")
        },
        "pecv_transitions_vs_same_checkpoint_off": {
            "seen": _transitions(off["seen"], full["seen"], seen_labels),
            "unseen": _transitions(off["unseen"], full["unseen"], unseen_labels),
            "zs": _transitions(off["zs"], full["zs"], unseen_labels),
        },
        "gtd_diagnostics": model.backbone.diagnostics(packages),
    }
    if return_predictions:
        result["_predictions"] = {"full": full, "off": off}
    return result


def _prepare_output(output_dir: Path, resume_from: Path | None, config: dict) -> tuple[Path, str]:
    if resume_from is None:
        return prepare_output_dir(output_dir), "x"
    resolved_output = output_dir.resolve()
    resolved_resume = resume_from.resolve()
    if not resolved_output.is_dir() or resolved_resume != resolved_output / "checkpoint_last.pth":
        raise ValueError("PECV formal resumes only its own checkpoint_last.pth.")
    snapshot = resolved_output / "config.snapshot.yaml"
    if not snapshot.is_file() or yaml.safe_load(snapshot.read_text(encoding="utf-8")) != config:
        raise ValueError("PECV formal resume config snapshot changed.")
    return resolved_output, "a"


def run(
    config_path: Path,
    output_dir: Path,
    expected_commit: str,
    expected_config_sha: str,
    resume_from: Path | None = None,
) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("PECV formal expected commit differs from clean HEAD.")
    config, config_sha = load_config(config_path)
    if config_sha != expected_config_sha:
        raise ValueError("PECV formal expected config SHA mismatch.")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PECV formal training requires CUDA.")
    tensors = load_assets(config)
    labels = tensors["train_labels"].long()
    seen = torch.unique(labels, sorted=True)
    if labels.numel() != TRAIN_COUNT or seen.numel() != SEEN_COUNT:
        raise ValueError("PECV formal trainval identity changed.")
    output_dir, log_mode = _prepare_output(output_dir, resume_from, config)
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
        print(
            f"PECV RUN={config['experiment_id']} commit={code_commit} "
            f"config_sha={config_sha} loaded_training_checkpoints=[]"
        )
        model = build_model(config, tensors, device)
        pecv_enabled = float(config["pecv_loss_weight"]) > 0.0
        initial_states = {
            "tg": tensor_mapping_sha256(dict(model.backbone.parent.state_dict())),
            "gtd": tensor_mapping_sha256(dict(model.backbone.gate.state_dict())),
            "pecv": tensor_mapping_sha256(dict(model.verifier.state_dict())),
        }
        train_features = tensors["train_features"].to(device).float()
        train_labels = labels.to(device)
        seen_device = seen.to(device)
        global_to_seen = torch.full((CLASS_COUNT,), -1, dtype=torch.long, device=device)
        global_to_seen[seen_device] = torch.arange(SEEN_COUNT, device=device)
        centroids = h1.visual_centroids(tensors["train_features"], labels, seen).to(device)
        folds = rank_modulo_class_folds(seen)
        groups = [
            list(model.backbone.parent.parameters()),
            list(model.backbone.gate.parameters()),
            list(model.verifier.parameters()),
        ]
        if len({id(parameter) for group in groups for parameter in group}) != sum(
            len(group) for group in groups
        ):
            raise RuntimeError("PECV formal optimizer parameter groups overlap.")
        optimizer = torch.optim.Adam(
            [
                {"params": groups[0], "lr": float(config["tg_learning_rate"])},
                {"params": groups[1], "lr": float(config["gtd_learning_rate"])},
                {"params": groups[2], "lr": float(config["pecv_learning_rate"])},
            ],
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = ThreeGroupSchedule(optimizer, config)
        batch_generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
        refresh_updates = teacher_refresh_updates(
            train_count=TRAIN_COUNT, nominal_epochs=NOMINAL_EPOCHS, batch_size=BATCH_SIZE
        )
        eval_updates = set(
            evaluation_updates(
                train_count=TRAIN_COUNT, nominal_epochs=NOMINAL_EPOCHS, batch_size=BATCH_SIZE
            )
        )
        if resume_from is None:
            packages = refresh_oracle_targets(
                model.backbone, centroids, folds, float(config["theta_penalty"])
            )
            teacher_history = [
                teacher_refresh_record(
                    update=refresh_updates[0],
                    model=model.backbone,
                    packages=packages,
                    folds=folds,
                    valid_updates=refresh_updates,
                )
            ]
            next_refresh = refresh_updates[1]
            initial = evaluate(
                model, tensors, packages, device, pecv_enabled=pecv_enabled
            )
            for name in ("U", "S", "H", "ZS"):
                if abs(initial[name] - initial["module_off_metrics"][name]) > 1e-8:
                    raise ValueError("PECV zero initialization does not reproduce TG+GTD off.")
            initial.update({"evaluation_index": 0, "update": 0})
            history = [initial]
            best_metrics = copy.deepcopy(initial)
            best_state = copy.deepcopy(model.state_dict())
            best_update = 0
            best_zs = {"ZS": float(initial["ZS"]), "update": 0, "metrics": copy.deepcopy(initial)}
            start_update = 1
        else:
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("experiment_id") != config["experiment_id"]
                or checkpoint.get("code_commit") != code_commit
                or checkpoint.get("config_sha256") != config_sha
                or checkpoint.get("initial_states") != initial_states
                or not 0 < int(checkpoint.get("update", 0)) <= TOTAL_UPDATES
            ):
                raise ValueError("PECV formal checkpoint identity/update mismatch.")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            packages = teacher_packages_to_device(checkpoint["teacher_packages"], device)
            teacher_history = checkpoint["teacher_refresh_history"]
            next_refresh = checkpoint["next_teacher_refresh_update"]
            history = checkpoint["history"]
            best_metrics = checkpoint["best_metrics"]
            best_state = checkpoint["best_model_state_dict"]
            best_update = int(checkpoint["best_update"])
            best_zs = checkpoint["best_zs_observation"]
            expected_refresh = next_teacher_refresh_after(int(checkpoint["update"]), refresh_updates)
            if (
                next_refresh != expected_refresh
                or teacher_packages_sha256(packages) != teacher_history[-1]["package_sha256"]
            ):
                raise ValueError("PECV formal resume teacher identity mismatch.")
            restore_rng_states(checkpoint, batch_generator)
            reproducibility = checkpoint["reproducibility"]
            start_update = int(checkpoint["update"]) + 1
            print(f"PECV resume_from={resume_from} next_update={start_update}")

        interval_sums: dict[str, float] = {}
        interval_steps = 0
        for update in range(start_update, TOTAL_UPDATES + 1):
            if next_refresh is not None and update == int(next_refresh):
                packages = refresh_oracle_targets(
                    model.backbone, centroids, folds, float(config["theta_penalty"])
                )
                teacher_history.append(
                    teacher_refresh_record(
                        update=update,
                        model=model.backbone,
                        packages=packages,
                        folds=folds,
                        valid_updates=refresh_updates,
                    )
                )
                next_refresh = next_teacher_refresh_after(update, refresh_updates)
            model.train()
            scheduler.set_for_update(update)
            rows_cpu = torch.randperm(TRAIN_COUNT, generator=batch_generator)[:BATCH_SIZE]
            rows = rows_cpu.to(device)
            images = train_features.index_select(0, rows)
            labels_global = train_labels.index_select(0, rows)
            labels_seen = global_to_seen.index_select(0, labels_global)
            package = packages[(update - 1) % 3]
            optimizer.zero_grad(set_to_none=True)
            parent_logits = model.backbone.parent.logits(images, seen_device)
            ce = F.cross_entropy(parent_logits, labels_seen)
            topology = model.backbone.parent.topology_loss()
            raw_ratio = model.backbone.gate.raw_ratio(package["features"])
            gtd_loss = F.smooth_l1_loss(raw_ratio, package["target_ratio"])
            if pecv_enabled:
                pecv_scores, _ = model.training_candidate_scores(
                    images, labels_global, seen_device, enabled=True
                )
                pecv_loss = F.cross_entropy(
                    pecv_scores, torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
                )
            else:
                pecv_loss = ce.new_zeros(())
            total = (
                ce
                + float(config["topology_weight"]) * topology
                + float(config["gtd_loss_weight"]) * gtd_loss
                + float(config["pecv_loss_weight"]) * pecv_loss
            )
            if not torch.isfinite(total):
                raise FloatingPointError("PECV formal loss contains NaN/Inf.")
            total.backward()
            require_finite_gradients(model)
            optimizer.step()
            require_finite_model(model)
            if update == start_update:
                print(
                    f"startup_confirm update={update} tg_lr={optimizer.param_groups[0]['lr']:.8g} "
                    f"gtd_lr={optimizer.param_groups[1]['lr']:.8g} "
                    f"pecv_lr={optimizer.param_groups[2]['lr']:.8g}"
                )
            interval_steps += 1
            values = {
                "total": total,
                "parent_ce": ce,
                "topology": topology,
                "gtd_smooth_l1": gtd_loss,
                "pecv_listwise_ce": pecv_loss,
                "tg_lr": torch.tensor(optimizer.param_groups[0]["lr"], device=device),
                "gtd_lr": torch.tensor(optimizer.param_groups[1]["lr"], device=device),
                "pecv_lr": torch.tensor(optimizer.param_groups[2]["lr"], device=device),
            }
            for name, value in values.items():
                interval_sums[name] = interval_sums.get(name, 0.0) + float(value.detach())
            if update not in eval_updates:
                continue
            metrics = evaluate(model, tensors, packages, device, pecv_enabled=pecv_enabled)
            metrics.update(
                {
                    "evaluation_index": len(history),
                    "update": update,
                    "train": {
                        name: value / interval_steps for name, value in interval_sums.items()
                    },
                }
            )
            history.append(metrics)
            print(
                f"eval={metrics['evaluation_index']} update={update} "
                f"U={metrics['U']:.6f} S={metrics['S']:.6f} H={metrics['H']:.6f} "
                f"ZS={metrics['ZS']:.6f} removeH={metrics['full_minus_off_delta']['H']:.6f}"
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
                "config_sha256": config_sha,
                "initial_states": initial_states,
                "update": update,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_update": best_update,
                "best_metrics": best_metrics,
                "best_model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
                "best_zs_observation": best_zs,
                "teacher_packages": teacher_packages_to_cpu(packages),
                "teacher_refresh_history": teacher_history,
                "next_teacher_refresh_update": next_refresh,
                "batch_generator_state": batch_generator.get_state(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "history": history,
                "reproducibility": reproducibility,
            }
            atomic_torch_save(output_dir / "checkpoint_last.pth", checkpoint)

        if len(history) != NOMINAL_EPOCHS + 2 or history[-1]["update"] != TOTAL_UPDATES:
            raise RuntimeError("PECV formal requires 202 evaluation points ending at 28228.")
        if len(teacher_history) != NOMINAL_EPOCHS or next_refresh is not None:
            raise RuntimeError("PECV formal requires 200 complete teacher refreshes.")
        model.load_state_dict(best_state, strict=True)
        recomputed_best = evaluate(model, tensors, packages, device, pecv_enabled=pecv_enabled)
        for name in ("U", "S", "H", "ZS"):
            if abs(float(recomputed_best[name]) - float(best_metrics[name])) > 1e-8:
                raise ValueError("PECV best checkpoint does not reproduce selected metrics.")
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
        atomic_write_json(output_dir / "evaluation_history.json", {"rows": history})
        atomic_write_json(
            output_dir / "teacher_refresh_history.json",
            {"count": len(teacher_history), "rows": teacher_history},
        )
        checkpoint_sha = sha256_file(output_dir / "checkpoint_last.pth")
        result = {
            "schema_version": SCHEMA,
            "experiment_id": config["experiment_id"],
            "idea_id": config["idea_id"],
            "condition_id": config["condition_id"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "initialization_mode": "all_modules_same_run_seed7_update1",
            "loaded_training_checkpoints": (
                [] if resume_from is None else [str(resume_from.resolve())]
            ),
            "resume_used": resume_from is not None,
            "pecv_enabled": pecv_enabled,
            "initial_states": initial_states,
            "best_metrics": best_metrics,
            "best_update": best_update,
            "module_off_metrics": best_metrics["module_off_metrics"],
            "best_full_minus_off_delta": best_metrics["full_minus_off_delta"],
            "best_gap_U_S": abs(float(best_metrics["U"]) - float(best_metrics["S"])),
            "best_zs_observation": best_zs,
            "stop_reason": "completed_fixed_200",
            "history_length": len(history),
            "target_refresh_count": len(teacher_history),
            "total_updates": TOTAL_UPDATES,
            "eval_interval_steps": EVAL_INTERVAL,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "human_annotations_used": False,
            "asset_id": config["asset_id"],
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": checkpoint_sha,
            "evaluation_history_sha256": sha256_file(output_dir / "evaluation_history.json"),
            "teacher_refresh_history_sha256": sha256_file(
                output_dir / "teacher_refresh_history.json"
            ),
            "reproducibility": reproducibility,
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "gpu": torch.cuda.get_device_name(device),
                "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            },
        }
        atomic_write_json(output_dir / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return result
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run(
        args.config,
        args.output_dir,
        args.expected_commit,
        args.expected_config_sha,
        args.resume_from,
    )


if __name__ == "__main__":
    main()
