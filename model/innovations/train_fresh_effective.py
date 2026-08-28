"""Fresh one-stage matched confirmation for TG, GTD, MMT, and BD."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.innovations.bd_tst import BalancedDecoupledTST
from model.innovations.elpt import fixed_class_folds
from model.innovations.gtd_tst import GTDTSTModel
from model.innovations.mmt_tst import (
    MMTTSTModel,
    MarginTargetTable,
    MinimumMarginGate,
    build_margin_target_table,
    mmt_losses,
)
from model.innovations.train_gtd_tst import (
    load_assets,
    refresh_oracle_targets,
    teacher_packages_to_cpu,
    teacher_packages_to_device,
    tensor_mapping_sha256,
)
from model.paper_v2 import PaperV2ThreeModuleModel
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


SCHEMA = "gzsl-paper.v3-fresh-effective-confirm.v1"
TRAIN_COUNT = 7057
SEEN_COUNT = 150
CLASS_COUNT = 200
BATCH_SIZE = 50
NOMINAL_EPOCHS = 150
TOTAL_UPDATES = 21171
EVAL_INTERVAL = 141
WARMUP_UPDATES = 705
AUXILIARY_HALF = 25
CONDITIONS = {
    "V3-TRY-042": ("TG_FRESH_FIXED150", "tg"),
    "V3-TRY-043": ("TG_PLUS_GTD_FRESH_FIXED150", "gtd"),
    "V3-TRY-044": ("TG_PLUS_MMT_FRESH_FIXED150", "mmt"),
    "V3-TRY-045": ("TG_PLUS_BD_FRESH_FIXED150", "bd"),
}
FORBIDDEN_NON_NULL = (
    "tg_checkpoint",
    "tg_checkpoint_sha256",
    "pretrained_module_checkpoint",
)
CONFIG_KEYS = {
    "schema_version", "experiment_id", "idea_id", "framework_id", "dataset",
    "condition_id", "module", "asset_manifest", "asset_manifest_sha256", "asset_id",
    "initialization_strategy", "training_strategy", "stagewise_training",
    "checkpoint_handoff", "module_pretraining", "tg_checkpoint",
    "tg_checkpoint_sha256", "pretrained_module_checkpoint", "parent_metrics_percent",
    "device", "random_seed", "module_initialization_seed", "auxiliary_random_seed",
    "batch_size", "auxiliary_batch_half", "nominal_epochs", "total_updates",
    "eval_interval_steps", "tg_learning_rate", "tg_min_learning_rate",
    "gate_learning_rate", "gate_min_learning_rate", "gate_warmup_epochs",
    "weight_decay", "topology_weight", "required_delta_h", "weak_delta_h",
    "max_us_gap", "gtd_gate_loss_weight", "gtd_hidden_dim", "gtd_grid_points",
    "gtd_theta_penalty", "gtd_max_transport_step", "mmt_weight", "mmt_hidden_dim",
    "mmt_soft_negative_topk", "mmt_soft_negative_temperature",
    "mmt_margin_quantile", "mmt_theta_max_degrees",
    "mmt_initial_move_probability", "mmt_initial_theta_degrees",
    "mmt_bisection_steps", "mmt_tangent_epsilon", "mmt_margin_scale",
    "mmt_leak_tolerance", "mmt_theta_loss_weight", "mmt_margin_loss_weight",
    "mmt_zero_loss_weight", "mmt_leak_loss_weight", "bd_fold_topology_weight",
    "bd_gate_hidden_dim", "bd_max_transport_step", "bd_initial_transport_step",
    "early_stopping_enabled", "human_annotations_used", "test_used_for_selection",
    "test_used_for_hyperparameter_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim",
}


@dataclass
class ModelBundle:
    model: nn.Module
    parent: PaperV2ThreeModuleModel
    module_name: str

    def module_parameters(self) -> list[nn.Parameter]:
        if self.module_name == "gtd":
            return list(self.model.gate.parameters())
        if self.module_name == "mmt":
            return list(self.model.gate.parameters())
        if self.module_name == "bd":
            return self.model.gate_parameters()
        return []


class FreshSchedule:
    """Constant fresh-TG LR; Gate warmup then cosine, with exact update state."""

    def __init__(self, tg_optimizer, gate_optimizer=None):
        self.tg_optimizer = tg_optimizer
        self.gate_optimizer = gate_optimizer
        self.last_update = 0

    def learning_rates(self, update: int) -> tuple[float, float | None]:
        update = int(update)
        if not 1 <= update <= TOTAL_UPDATES:
            raise ValueError("fresh scheduler update越界。")
        tg_lr = 1e-4
        if self.gate_optimizer is None:
            return tg_lr, None
        if update <= WARMUP_UPDATES:
            progress = (update - 1) / (WARMUP_UPDATES - 1)
            gate_lr = 1e-5 + (1e-4 - 1e-5) * progress
        else:
            progress = (update - WARMUP_UPDATES) / (TOTAL_UPDATES - WARMUP_UPDATES)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            gate_lr = 1e-5 + (1e-4 - 1e-5) * cosine
        return tg_lr, gate_lr

    def set_for_update(self, update: int) -> None:
        tg_lr, gate_lr = self.learning_rates(update)
        self.tg_optimizer.param_groups[0]["lr"] = tg_lr
        if self.gate_optimizer is not None:
            self.gate_optimizer.param_groups[0]["lr"] = gate_lr
        self.last_update = int(update)

    def state_dict(self) -> dict[str, Any]:
        return {"last_update": self.last_update, "has_gate": self.gate_optimizer is not None}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"last_update", "has_gate"} or bool(state["has_gate"]) != (
            self.gate_optimizer is not None
        ):
            raise ValueError("fresh scheduler checkpoint身份错误。")
        update = int(state["last_update"])
        if not 1 <= update < TOTAL_UPDATES:
            raise ValueError("fresh resume scheduler update错误。")
        self.set_for_update(update)


def load_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"fresh配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。"
        )
    identity = CONDITIONS.get(config["experiment_id"])
    checks = (
        config["schema_version"] == SCHEMA,
        identity is not None,
        config["condition_id"] == (identity[0] if identity else None),
        config["module"] == (identity[1] if identity else None),
        config["idea_id"] == "IDEA-155",
        config["framework_id"] == "FRAMEWORK-V3-EXPLORATION",
        config["dataset"] == "CUB",
        config["initialization_strategy"] == "fresh_seeded_tg",
        config["training_strategy"] == "one_stage_simultaneous",
        config["stagewise_training"] is False,
        config["checkpoint_handoff"] is False,
        config["module_pretraining"] is False,
        all(config[name] is None for name in FORBIDDEN_NON_NULL),
        config["parent_metrics_percent"] is None,
        int(config["random_seed"]) == 7,
        int(config["module_initialization_seed"]) == 1557,
        int(config["auxiliary_random_seed"]) == 7151,
        int(config["batch_size"]) == BATCH_SIZE,
        int(config["auxiliary_batch_half"]) == AUXILIARY_HALF,
        int(config["nominal_epochs"]) == NOMINAL_EPOCHS,
        int(config["total_updates"]) == TOTAL_UPDATES,
        int(config["eval_interval_steps"]) == EVAL_INTERVAL,
        float(config["tg_learning_rate"]) == 1e-4,
        float(config["tg_min_learning_rate"]) == 1e-4,
        float(config["gate_learning_rate"]) == 1e-4,
        float(config["gate_min_learning_rate"]) == 1e-5,
        int(config["gate_warmup_epochs"]) == 5,
        float(config["weight_decay"]) == 1e-4,
        float(config["topology_weight"]) == 0.1,
        float(config["required_delta_h"]) == 1.0,
        float(config["weak_delta_h"]) == 0.8,
        float(config["max_us_gap"]) == 8.0,
        float(config["gtd_gate_loss_weight"]) == 1.0,
        int(config["gtd_hidden_dim"]) == 16,
        int(config["gtd_grid_points"]) == 33,
        float(config["gtd_theta_penalty"]) == 0.1,
        float(config["gtd_max_transport_step"]) == 1.5,
        float(config["mmt_weight"]) == 0.1,
        int(config["mmt_hidden_dim"]) == 16,
        int(config["mmt_soft_negative_topk"]) == 5,
        float(config["mmt_soft_negative_temperature"]) == 0.07,
        float(config["mmt_margin_quantile"]) == 0.25,
        float(config["mmt_theta_max_degrees"]) == 30.0,
        float(config["mmt_initial_move_probability"]) == 0.1,
        float(config["mmt_initial_theta_degrees"]) == 5.0,
        int(config["mmt_bisection_steps"]) == 20,
        float(config["mmt_tangent_epsilon"]) == 1e-6,
        float(config["mmt_margin_scale"]) == 0.02,
        float(config["mmt_leak_tolerance"]) == 0.005,
        float(config["mmt_theta_loss_weight"]) == 1.0,
        float(config["mmt_margin_loss_weight"]) == 0.5,
        float(config["mmt_zero_loss_weight"]) == 0.1,
        float(config["mmt_leak_loss_weight"]) == 0.1,
        float(config["bd_fold_topology_weight"]) == 0.1,
        int(config["bd_gate_hidden_dim"]) == 16,
        float(config["bd_max_transport_step"]) == 1.5,
        float(config["bd_initial_transport_step"]) == 0.1,
        config["early_stopping_enabled"] is False,
        config["human_annotations_used"] is False,
        config["test_used_for_selection"] is True,
        config["test_used_for_hyperparameter_selection"] is True,
        config["unseen_images_used_for_gradient"] is False,
        config["strict_blind_claim"] is False,
    )
    if not all(checks):
        raise ValueError("fresh运行身份、初始化、预算、公式超参或披露边界错误。")
    return config, sha256_file(Path(path))


def build_parent(tensors: dict[str, torch.Tensor]) -> PaperV2ThreeModuleModel:
    labels = tensors["train_labels"].long()
    seen = torch.unique(labels, sorted=True)
    centroids = h1.visual_centroids(tensors["train_features"], labels, seen)
    return PaperV2ThreeModuleModel(
        tensors["role_sentence_embeds"], seen, centroids,
        tg_vpr_mode="full", transport_mode="off", ccgr_mode="off",
        dropout=0.5, inner_ratio=0.35, outer_ratio=0.65, temperature=0.07,
    )


def build_model(config: dict, tensors: dict[str, torch.Tensor], device: torch.device) -> ModelBundle:
    # The TG parent is always constructed first on CPU. Module initialization is
    # forked so it cannot advance the parent/dropout RNG stream.
    parent = build_parent(tensors)
    seen = parent.seen_classes.cpu()
    module_name = str(config["module"])
    if module_name == "tg":
        model: nn.Module = parent
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                int(config["module_initialization_seed"])
            )
            if module_name == "gtd":
                model = GTDTSTModel(
                    parent, seen, hidden_dim=int(config["gtd_hidden_dim"]),
                    max_transport_step=float(config["gtd_max_transport_step"]),
                    grid_points=int(config["gtd_grid_points"]),
                )
            elif module_name == "mmt":
                gate = MinimumMarginGate(
                    hidden_dim=int(config["mmt_hidden_dim"]),
                    initial_move_probability=float(config["mmt_initial_move_probability"]),
                    initial_theta_fraction=float(config["mmt_initial_theta_degrees"])
                    / float(config["mmt_theta_max_degrees"]),
                )
                model = MMTTSTModel(
                    parent.tg_vpr, gate,
                    global_theta_max=math.radians(float(config["mmt_theta_max_degrees"])),
                    tangent_epsilon=float(config["mmt_tangent_epsilon"]),
                    topk=int(config["mmt_soft_negative_topk"]),
                )
            elif module_name == "bd":
                model = BalancedDecoupledTST(
                    parent, seen, gate_enabled=True,
                    gate_hidden_dim=int(config["bd_gate_hidden_dim"]),
                    max_transport_step=float(config["bd_max_transport_step"]),
                    initial_transport_step=float(config["bd_initial_transport_step"]),
                    gate_initialization_seed=int(config["module_initialization_seed"]),
                )
            else:
                raise AssertionError(module_name)
    model = model.to(device)
    parent = parent.to(device)
    return ModelBundle(model=model, parent=parent, module_name=module_name)


def evaluation_updates() -> tuple[int, ...]:
    values = tuple([EVAL_INTERVAL * index for index in range(1, 151)] + [TOTAL_UPDATES])
    if len(values) != 151 or values[-2:] != (21150, 21171):
        raise RuntimeError("fresh评估计划必须是141×1..150加21171。")
    return values


def teacher_refresh_updates(current_update: int) -> tuple[int, ...]:
    """返回每个训练区间的起点，包括末尾21步的partial区间。"""
    if not 0 <= int(current_update) <= TOTAL_UPDATES:
        raise ValueError("fresh teacher current_update越界。")
    return tuple(range(1, int(current_update) + 1, EVAL_INTERVAL))


def canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"D")
            for key in sorted(item, key=lambda x: str(x)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"L")
            for child in item:
                visit(child)
        elif item is None:
            digest.update(b"N")
        elif isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))
        else:
            raise TypeError(f"canonical digest不支持{type(item).__name__}。")

    visit(value)
    return digest.hexdigest()


def primary_batch_prefix_sha256(generator_state: torch.Tensor, count: int = 142) -> str:
    generator = torch.Generator(device="cpu")
    generator.set_state(generator_state.clone())
    batches = [torch.randperm(TRAIN_COUNT, generator=generator)[:BATCH_SIZE] for _ in range(count)]
    return canonical_sha256(batches)


def balanced_auxiliary_indices(
    labels: torch.Tensor, pseudo_seen: torch.Tensor, pseudo_unseen: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    labels = labels.detach().cpu().long()
    seen_pool = torch.isin(labels, pseudo_seen.cpu()).nonzero(as_tuple=False).flatten()
    unseen_pool = torch.isin(labels, pseudo_unseen.cpu()).nonzero(as_tuple=False).flatten()
    if min(seen_pool.numel(), unseen_pool.numel()) < AUXILIARY_HALF:
        raise ValueError("BD辅助采样池不足。")
    left = seen_pool[torch.randperm(seen_pool.numel(), generator=generator)[:AUXILIARY_HALF]]
    right = unseen_pool[torch.randperm(unseen_pool.numel(), generator=generator)[:AUXILIARY_HALF]]
    return torch.cat((left, right))


@torch.no_grad()
def make_mmt_teacher(
    model: MMTTSTModel, visual_centroids: torch.Tensor, config: dict, device: torch.device,
) -> MarginTargetTable:
    was_training = model.training
    model.eval()
    all_classes = torch.arange(CLASS_COUNT, device=device)
    table = build_margin_target_table(
        mean8=model.parent.base_prototypes(),
        value=model.parent.value_candidate(all_classes),
        tg_prototypes=model.parent_prototypes(),
        visual_centroids=visual_centroids,
        seen_classes=model.seen_classes,
        topk=int(config["mmt_soft_negative_topk"]),
        negative_temperature=float(config["mmt_soft_negative_temperature"]),
        margin_quantile=float(config["mmt_margin_quantile"]),
        global_theta_max=math.radians(float(config["mmt_theta_max_degrees"])),
        tangent_epsilon=float(config["mmt_tangent_epsilon"]),
        bisection_steps=int(config["mmt_bisection_steps"]),
    ).to(device)
    model.train(was_training)
    return table


def mmt_teacher_to_cpu(table: MarginTargetTable) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in table.__dict__.items()}


def mmt_teacher_from_state(state: dict[str, torch.Tensor], device: torch.device) -> MarginTargetTable:
    return MarginTargetTable(**{name: tensor.to(device) for name, tensor in state.items()})


def full_and_off_prototypes(bundle: ModelBundle) -> tuple[torch.Tensor, torch.Tensor]:
    off = bundle.parent.prototypes()
    if bundle.module_name == "tg":
        return off, off
    if bundle.module_name == "gtd":
        return bundle.model.prototype_bundle()["final"], off
    if bundle.module_name == "mmt":
        return bundle.model.prototype_components()["final"], off
    return bundle.model.prototype_bundle()["final"], off


@torch.no_grad()
def evaluate(bundle: ModelBundle, tensors: dict, device: torch.device) -> dict:
    bundle.model.eval()
    full, off = full_and_off_prototypes(bundle)
    seen = bundle.parent.seen_classes.cpu()
    all_classes = torch.arange(CLASS_COUNT)
    unseen = all_classes[~torch.isin(all_classes, seen)]
    kwargs = dict(
        scale=bundle.parent.scale(),
        seen_features=tensors["test_seen_features"], seen_labels=tensors["test_seen_labels"],
        unseen_features=tensors["test_unseen_features"], unseen_labels=tensors["test_unseen_labels"],
        seen_classes=seen, unseen_classes=unseen, device=device,
    )
    full_metrics = evaluate_prototypes(full, **kwargs)
    off_metrics = evaluate_prototypes(off, **kwargs)
    return {
        **full_metrics,
        "module_off_metrics": off_metrics,
        "full_minus_off_delta": {
            key: float(full_metrics[key]) - float(off_metrics[key]) for key in ("U", "S", "H", "ZS")
        },
    }


def snapshot_model(model: nn.Module) -> dict[str, torch.Tensor]:
    require_finite_model(model)
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def gradient_report(parameters: list[nn.Parameter]) -> dict[str, Any]:
    norms = [float(parameter.grad.detach().norm()) if parameter.grad is not None else None for parameter in parameters]
    return {
        "parameter_count": len(parameters),
        "all_gradients_present": all(value is not None for value in norms),
        "any_nonzero_gradient": any(value is not None and value > 0.0 for value in norms),
        "gradient_norms": norms,
    }


def prepare_run_directory(output_dir: Path, resume_from: Path | None, config: dict) -> tuple[Path, str]:
    if resume_from is None:
        return prepare_output_dir(output_dir), "x"
    output = output_dir.resolve()
    resume = resume_from.resolve()
    repo = Path(__file__).resolve().parents[2]
    if not output.is_absolute() or repo == output or repo in output.parents:
        raise ValueError("fresh output必须是仓库外绝对路径。")
    if not output.is_dir() or resume != output / "checkpoint_last.pth":
        raise ValueError("fresh只允许同RUN checkpoint_last续训。")
    snapshot = output / "config.snapshot.yaml"
    if yaml.safe_load(snapshot.read_text(encoding="utf-8")) != config:
        raise ValueError("fresh resume config snapshot不一致。")
    return output, "a"


def _rng_state(primary: torch.Generator, auxiliary: torch.Generator) -> dict[str, Any]:
    return {
        "primary": primary.get_state(), "auxiliary": auxiliary.get_state(),
        "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def _require_finite_tree(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{name}包含NaN/Inf。")
    elif isinstance(value, dict):
        for key, child in value.items():
            _require_finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_tree(child, f"{name}[{index}]")
    elif isinstance(value, (float, complex)):
        parts = (value.real, value.imag) if isinstance(value, complex) else (value,)
        if not all(math.isfinite(float(part)) for part in parts):
            raise ValueError(f"{name}包含NaN/Inf。")


def _first_strict_max(rows: list[dict], metric: str) -> dict:
    best = rows[0]
    for row in rows[1:]:
        if float(row[metric]) > float(best[metric]):
            best = row
    return best


def validate_history(
    history: list[dict], *, current_update: int,
    current_model_sha256: str, best_update: int, best_metrics: dict,
    best_state: dict[str, torch.Tensor], best_zs: dict,
    best_zs_state: dict[str, torch.Tensor],
) -> None:
    expected_updates = [0] + [
        update for update in evaluation_updates() if update <= int(current_update)
    ]
    if not isinstance(history, list) or [row.get("update") for row in history] != expected_updates:
        raise ValueError("fresh resume history update序列不连续。")
    if [row.get("evaluation_index") for row in history] != list(range(len(history))):
        raise ValueError("fresh resume history evaluation_index不连续。")
    required = {"U", "S", "H", "ZS", "module_off_metrics", "full_minus_off_delta", "model_state_sha256"}
    for row in history:
        if not required.issubset(row) or len(str(row["model_state_sha256"])) != 64:
            raise ValueError("fresh resume history schema错误。")
        _require_finite_tree(row, "history")
    if history[-1]["model_state_sha256"] != current_model_sha256:
        raise ValueError("fresh resume current model与history末行SHA不一致。")
    expected_best = _first_strict_max(history, "H")
    if int(best_update) != int(expected_best["update"]) or best_metrics != expected_best:
        raise ValueError("fresh resume best-H不是history首次严格最大值。")
    if canonical_sha256(best_state) != expected_best["model_state_sha256"]:
        raise ValueError("fresh resume best-H state SHA错误。")
    expected_zs = _first_strict_max(history, "ZS")
    if (
        int(best_zs.get("update", -1)) != int(expected_zs["update"])
        or float(best_zs.get("ZS", float("nan"))) != float(expected_zs["ZS"])
        or best_zs.get("metrics") != expected_zs
    ):
        raise ValueError("fresh resume best-ZS不是history首次严格最大值。")
    if canonical_sha256(best_zs_state) != expected_zs["model_state_sha256"]:
        raise ValueError("fresh resume best-ZS state SHA错误。")


def validate_teacher_state(
    *, module_name: str, current_update: int,
    teacher_history: list[dict], teacher_state: Any,
) -> None:
    expected_updates = list(teacher_refresh_updates(current_update))
    if module_name not in {"gtd", "mmt"}:
        if teacher_history != [] or teacher_state is not None:
            raise ValueError("fresh非teacher模块包含teacher状态。")
        return
    if (
        not isinstance(teacher_history, list)
        or [row.get("update") for row in teacher_history] != expected_updates
        or any(set(row) != {"update", "sha256"} or len(str(row["sha256"])) != 64 for row in teacher_history)
        or teacher_state is None
    ):
        raise ValueError("fresh teacher update/schema错误。")
    if canonical_sha256(teacher_state) != teacher_history[-1]["sha256"]:
        raise ValueError("fresh teacher state SHA错误。")


def checkpoint_semantic_material(checkpoint: dict) -> dict[str, Any]:
    return {
        "scheduler_state_dict": checkpoint["scheduler_state_dict"],
        "history": checkpoint["history"],
        "best_metrics": checkpoint["best_metrics"],
        "best_update": checkpoint["best_update"],
        "best_model_state_dict": checkpoint["best_model_state_dict"],
        "best_zs_observation": checkpoint["best_zs_observation"],
        "best_zs_model_state_dict": checkpoint["best_zs_model_state_dict"],
        "teacher_history": checkpoint["teacher_history"],
        "teacher_state": checkpoint["teacher_state"],
        "first_update_gradients": checkpoint["first_update_gradients"],
        "initial_identity": checkpoint["initial_identity"],
        "model_state_dict": checkpoint["model_state_dict"],
        "tg_optimizer_state_dict": checkpoint["tg_optimizer_state_dict"],
        "gate_optimizer_state_dict": checkpoint["gate_optimizer_state_dict"],
        "rng_state": checkpoint["rng_state"],
    }


def seal_checkpoint(checkpoint: dict) -> dict:
    optimizer_payload = {
        "tg": checkpoint["tg_optimizer_state_dict"],
        "gate": checkpoint["gate_optimizer_state_dict"],
    }
    checkpoint["canonical_digests"] = {
        "model": canonical_sha256(checkpoint["model_state_dict"]),
        "optimizer": canonical_sha256(optimizer_payload),
        "rng": canonical_sha256(checkpoint["rng_state"]),
    }
    checkpoint["semantic_evidence_sha256"] = canonical_sha256(
        checkpoint_semantic_material(checkpoint)
    )
    return checkpoint


def validate_checkpoint(
    checkpoint: dict, *, module_name: str, experiment_id: str,
    code_commit: str, config_sha: str, initial_identity: dict,
) -> None:
    current_update = int(checkpoint.get("update", 0))
    if not all((
        checkpoint.get("experiment_id") == experiment_id,
        checkpoint.get("code_commit") == code_commit,
        checkpoint.get("config_sha256") == config_sha,
        checkpoint.get("initial_identity") == initial_identity,
        0 < current_update < TOTAL_UPDATES,
    )):
        raise ValueError("fresh resume RUN身份错误。")
    digests = checkpoint.get("canonical_digests", {})
    optimizer_payload = {
        "tg": checkpoint["tg_optimizer_state_dict"],
        "gate": checkpoint["gate_optimizer_state_dict"],
    }
    if (
        digests.get("model") != canonical_sha256(checkpoint["model_state_dict"])
        or digests.get("optimizer") != canonical_sha256(optimizer_payload)
        or digests.get("rng") != canonical_sha256(checkpoint["rng_state"])
    ):
        raise ValueError("fresh resume current model/optimizer/RNG digest错误。")
    if checkpoint.get("semantic_evidence_sha256") != canonical_sha256(
        checkpoint_semantic_material(checkpoint)
    ):
        raise ValueError("fresh resume semantic evidence digest错误。")
    scheduler = checkpoint.get("scheduler_state_dict")
    if (
        not isinstance(scheduler, dict)
        or set(scheduler) != {"last_update", "has_gate"}
        or int(scheduler.get("last_update", -1)) != current_update
        or bool(scheduler.get("has_gate")) != (module_name != "tg")
    ):
        raise ValueError("fresh resume scheduler update错误。")
    validate_history(
        checkpoint["history"], current_update=current_update,
        current_model_sha256=digests["model"],
        best_update=int(checkpoint["best_update"]),
        best_metrics=checkpoint["best_metrics"],
        best_state=checkpoint["best_model_state_dict"],
        best_zs=checkpoint["best_zs_observation"],
        best_zs_state=checkpoint["best_zs_model_state_dict"],
    )
    validate_teacher_state(
        module_name=module_name, current_update=current_update,
        teacher_history=checkpoint["teacher_history"],
        teacher_state=checkpoint["teacher_state"],
    )
    current_state = checkpoint["model_state_dict"]
    for name, state in (
        ("best-H", checkpoint["best_model_state_dict"]),
        ("best-ZS", checkpoint["best_zs_model_state_dict"]),
    ):
        if set(state) != set(current_state) or any(
            state[key].shape != current_state[key].shape for key in current_state
        ):
            raise ValueError(f"fresh resume {name} state schema错误。")
    first = checkpoint.get("first_update_gradients")
    if (
        not isinstance(first, dict)
        or set(first) != {"update1_pre_main_cuda_rng_sha256", "tg", "module"}
        or len(str(first["update1_pre_main_cuda_rng_sha256"])) != 64
        or not isinstance(first["tg"], dict)
        or first["tg"].get("any_nonzero_gradient") is not True
        or ((module_name == "tg") != (first["module"] is None))
    ):
        raise ValueError("fresh resume first-update gradient schema错误。")
    _require_finite_tree(checkpoint_semantic_material(checkpoint), "checkpoint")


def restore_checkpoint_objects(
    checkpoint: dict, *, model: nn.Module,
    tg_optimizer: torch.optim.Optimizer,
    gate_optimizer: torch.optim.Optimizer | None,
    scheduler: FreshSchedule,
    primary_generator: torch.Generator,
    auxiliary_generator: torch.Generator,
) -> None:
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    tg_optimizer.load_state_dict(checkpoint["tg_optimizer_state_dict"])
    if gate_optimizer is not None:
        if checkpoint["gate_optimizer_state_dict"] is None:
            raise ValueError("fresh candidate缺少Gate optimizer。")
        gate_optimizer.load_state_dict(checkpoint["gate_optimizer_state_dict"])
    elif checkpoint["gate_optimizer_state_dict"] is not None:
        raise ValueError("TG control不得恢复Gate optimizer。")
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    rng = checkpoint["rng_state"]
    primary_generator.set_state(rng["primary"])
    auxiliary_generator.set_state(rng["auxiliary"])
    torch.set_rng_state(rng["cpu"])
    if len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("fresh resume CUDA RNG设备数变化。")
    torch.cuda.set_rng_state_all(rng["cuda"])
    optimizer_payload = {
        "tg": tg_optimizer.state_dict(),
        "gate": gate_optimizer.state_dict() if gate_optimizer else None,
    }
    if (
        canonical_sha256(model.state_dict()) != checkpoint["canonical_digests"]["model"]
        or canonical_sha256(optimizer_payload) != checkpoint["canonical_digests"]["optimizer"]
        or canonical_sha256(_rng_state(primary_generator, auxiliary_generator))
        != checkpoint["canonical_digests"]["rng"]
    ):
        raise ValueError("fresh resume load后model/optimizer/RNG身份错误。")


def run(
    config_path: Path, output_dir: Path, expected_commit: str,
    resume_from: Path | None = None,
) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("fresh expected-commit与当前干净HEAD不一致。")
    config, config_sha = load_config(config_path)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fresh正式训练要求CUDA。")
    tensors = load_assets(config)
    labels_cpu = tensors["train_labels"].long()
    if labels_cpu.numel() != TRAIN_COUNT or torch.unique(labels_cpu).numel() != SEEN_COUNT:
        raise ValueError("fresh资产不是CUB trainval 7057/150。")
    output_dir, log_mode = prepare_run_directory(output_dir, resume_from, config)
    if resume_from is None:
        (output_dir / "config.snapshot.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    log_handle = (output_dir / "training.log").open(log_mode, encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = type("Tee", (), {
        "write": lambda self, value: [stream.write(value) for stream in (original_stdout, log_handle)][-1],
        "flush": lambda self: [stream.flush() for stream in (original_stdout, log_handle)],
    })()
    try:
        reproducibility = configure_reproducibility(
            int(config["random_seed"]), strict_determinism=True, deterministic_warn_only=False
        )
        bundle = build_model(config, tensors, device)
        post_build_cuda_rng_sha256 = canonical_sha256(
            torch.cuda.get_rng_state(device)
        )
        initial_tg_sha = tensor_mapping_sha256(dict(bundle.parent.tg_vpr.state_dict()))
        initial_parent_sha = tensor_mapping_sha256(dict(bundle.parent.state_dict()))
        primary_generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
        auxiliary_generator = torch.Generator(device="cpu").manual_seed(
            int(config["auxiliary_random_seed"])
        )
        initial_primary_state = primary_generator.get_state().clone()
        initial_identity = {
            "initialization_strategy": "fresh_seeded_tg",
            "initial_tg_state_sha256": initial_tg_sha,
            "initial_parent_state_sha256": initial_parent_sha,
            "primary_batch_generator_initial_sha256": canonical_sha256(initial_primary_state),
            "primary_batches_updates_1_142_sha256": primary_batch_prefix_sha256(initial_primary_state),
            "post_build_cuda_rng_sha256": post_build_cuda_rng_sha256,
            "loaded_training_checkpoints": [],
            "allowed_initialization_sources": ["frozen_clip_features", "frozen_text_embeddings"],
        }
        train_features = tensors["train_features"].to(device).float()
        train_labels = labels_cpu.to(device)
        seen = bundle.parent.seen_classes.to(device)
        global_to_seen = torch.full((CLASS_COUNT,), -1, dtype=torch.long, device=device)
        global_to_seen[seen] = torch.arange(SEEN_COUNT, device=device)
        visual_centroids = h1.visual_centroids(
            tensors["train_features"], labels_cpu, bundle.parent.seen_classes.cpu()
        ).to(device)
        folds = fixed_class_folds(bundle.parent.seen_classes.cpu())
        tg_parameters = list(bundle.parent.parameter_groups()["tg_vpr"])
        module_parameters = bundle.module_parameters()
        tg_optimizer = torch.optim.Adam(
            tg_parameters, lr=float(config["tg_learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        gate_optimizer = (
            torch.optim.Adam(
                module_parameters, lr=float(config["gate_learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            if module_parameters else None
        )
        scheduler = FreshSchedule(tg_optimizer, gate_optimizer)
        teacher = None
        teacher_history: list[dict[str, Any]] = []
        first_update_gradients = None
        if resume_from is None:
            initial = evaluate(bundle, tensors, device)
            initial_state = snapshot_model(bundle.model)
            initial.update({
                "evaluation_index": 0,
                "update": 0,
                "model_state_sha256": canonical_sha256(initial_state),
            })
            history = [initial]
            best_metrics = copy.deepcopy(initial)
            best_state = copy.deepcopy(initial_state)
            best_update = 0
            best_zs = {"ZS": float(initial["ZS"]), "update": 0, "metrics": copy.deepcopy(initial)}
            best_zs_state = copy.deepcopy(initial_state)
            start_update = 1
        else:
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
            validate_checkpoint(
                checkpoint, module_name=bundle.module_name,
                experiment_id=config["experiment_id"], code_commit=code_commit,
                config_sha=config_sha, initial_identity=initial_identity,
            )
            restore_checkpoint_objects(
                checkpoint, model=bundle.model, tg_optimizer=tg_optimizer,
                gate_optimizer=gate_optimizer, scheduler=scheduler,
                primary_generator=primary_generator,
                auxiliary_generator=auxiliary_generator,
            )
            history = checkpoint["history"]
            best_metrics = checkpoint["best_metrics"]
            best_state = checkpoint["best_model_state_dict"]
            best_update = int(checkpoint["best_update"])
            best_zs = checkpoint["best_zs_observation"]
            best_zs_state = checkpoint["best_zs_model_state_dict"]
            teacher_history = checkpoint["teacher_history"]
            first_update_gradients = checkpoint["first_update_gradients"]
            teacher_state = checkpoint["teacher_state"]
            if bundle.module_name == "gtd" and teacher_state is not None:
                teacher = teacher_packages_to_device(teacher_state, device)
            elif bundle.module_name == "mmt" and teacher_state is not None:
                teacher = mmt_teacher_from_state(teacher_state, device)
            reproducibility = checkpoint["reproducibility"]
            start_update = int(checkpoint["update"]) + 1
        eval_set = set(evaluation_updates())
        interval: dict[str, float] = {}
        interval_steps = 0
        for update in range(start_update, TOTAL_UPDATES + 1):
            if bundle.module_name in {"gtd", "mmt"} and (
                update == 1 or (update - 1) % EVAL_INTERVAL == 0
            ):
                if bundle.module_name == "gtd":
                    teacher = refresh_oracle_targets(
                        bundle.model, visual_centroids, folds, float(config["gtd_theta_penalty"])
                    )
                    teacher_sha = canonical_sha256(teacher_packages_to_cpu(teacher))
                else:
                    teacher = make_mmt_teacher(bundle.model, visual_centroids, config, device)
                    teacher_sha = canonical_sha256(mmt_teacher_to_cpu(teacher))
                teacher_history.append({"update": update, "sha256": teacher_sha})
            if update == 1:
                update1_pre_main_cuda_rng_sha256 = canonical_sha256(
                    torch.cuda.get_rng_state(device)
                )
            bundle.model.train()
            scheduler.set_for_update(update)
            primary_indices_cpu = torch.randperm(
                TRAIN_COUNT, generator=primary_generator
            )[:BATCH_SIZE]
            primary_indices = primary_indices_cpu.to(device)
            images = train_features.index_select(0, primary_indices)
            targets = global_to_seen.index_select(
                0, train_labels.index_select(0, primary_indices)
            )
            tg_optimizer.zero_grad(set_to_none=True)
            if gate_optimizer is not None:
                gate_optimizer.zero_grad(set_to_none=True)
            logits = bundle.parent.logits(images, seen)
            ce = F.cross_entropy(logits, targets)
            topology = bundle.parent.topology_loss()
            main_loss = ce + float(config["topology_weight"]) * topology
            module_loss = main_loss.new_zeros(())
            module_parts: dict[str, torch.Tensor] = {}
            if bundle.module_name == "gtd":
                package = teacher[(update - 1) % 3]
                raw = bundle.model.gate.raw_ratio(package["features"])
                module_loss = float(config["gtd_gate_loss_weight"]) * F.smooth_l1_loss(
                    raw, package["target_ratio"]
                )
                module_parts = {"gtd_gate": module_loss}
            elif bundle.module_name == "mmt":
                gate = bundle.model.gate(teacher.features, teacher.theta_cap)
                parts = mmt_losses(
                    gate, teacher, margin_scale=float(config["mmt_margin_scale"]),
                    leak_tolerance=float(config["mmt_leak_tolerance"]),
                )
                raw_module = (
                    parts["move"]
                    + float(config["mmt_theta_loss_weight"]) * parts["theta"]
                    + float(config["mmt_margin_loss_weight"]) * parts["margin"]
                    + float(config["mmt_zero_loss_weight"]) * parts["zero"]
                    + float(config["mmt_leak_loss_weight"]) * parts["leak"]
                )
                module_loss = float(config["mmt_weight"]) * raw_module
                module_parts = {f"mmt_{name}": value for name, value in parts.items() if name != "achieved_margin_mean"}
            elif bundle.module_name == "bd":
                fold_seen, fold_unseen = folds[(update - 1) % 3]
                aux_cpu = balanced_auxiliary_indices(
                    labels_cpu, fold_seen, fold_unseen, auxiliary_generator
                )
                aux = bundle.model.auxiliary_objective(
                    train_features.index_select(0, aux_cpu.to(device)),
                    train_labels.index_select(0, aux_cpu.to(device)),
                    fold_seen, fold_unseen, float(config["bd_fold_topology_weight"]),
                )
                module_loss = aux["loss"]
                module_parts = {"bd_ce": aux["ce"], "bd_topology": aux["topology"]}
                if update == 1:
                    main_to_gate = torch.autograd.grad(
                        main_loss, module_parameters, retain_graph=True, allow_unused=True
                    )
                    aux_to_tg = torch.autograd.grad(
                        module_loss, tg_parameters, retain_graph=True, allow_unused=True
                    )
                    if any(value is not None and float(value.norm()) != 0.0 for value in main_to_gate):
                        raise RuntimeError("BD main loss错误连接Gate。")
                    if any(value is not None and float(value.norm()) != 0.0 for value in aux_to_tg):
                        raise RuntimeError("BD auxiliary loss错误连接TG。")
            total = main_loss + module_loss
            if not torch.isfinite(total):
                raise FloatingPointError("fresh loss包含NaN/Inf。")
            total.backward()
            require_finite_gradients(bundle.model)
            if update == 1:
                first_update_gradients = {
                    "update1_pre_main_cuda_rng_sha256": update1_pre_main_cuda_rng_sha256,
                    "tg": gradient_report(tg_parameters),
                    "module": gradient_report(module_parameters) if module_parameters else None,
                }
                # TG keeps the historical fixed-equal semantic_group_logits parameter;
                # semantic_group_weights intentionally ignores it. All other TG weights
                # are covered by the shared main objective, so require a real TG signal
                # and disclose every dormant entry instead of pretending it is trained.
                if not first_update_gradients["tg"]["any_nonzero_gradient"]:
                    raise RuntimeError("fresh TG首步没有有效梯度。")
                if module_parameters and (
                    not first_update_gradients["module"]["all_gradients_present"]
                    or not first_update_gradients["module"]["any_nonzero_gradient"]
                ):
                    raise RuntimeError("fresh模块首步梯度不完整。")
            tg_optimizer.step()
            if gate_optimizer is not None:
                gate_optimizer.step()
            values = {"total": total, "ce": ce, "topology": topology, "module": module_loss, **module_parts}
            for name, value in values.items():
                interval[name] = interval.get(name, 0.0) + float(value.detach())
            interval_steps += 1
            if update not in eval_set:
                continue
            metrics = evaluate(bundle, tensors, device)
            model_state = snapshot_model(bundle.model)
            metrics.update({
                "evaluation_index": len(history), "update": update,
                "model_state_sha256": canonical_sha256(model_state),
                "train": {name: value / interval_steps for name, value in interval.items()},
                "tg_lr": float(tg_optimizer.param_groups[0]["lr"]),
                "gate_lr": float(gate_optimizer.param_groups[0]["lr"]) if gate_optimizer else None,
            })
            history.append(metrics)
            interval = {}
            interval_steps = 0
            if float(metrics["H"]) > float(best_metrics["H"]):
                best_metrics = copy.deepcopy(metrics)
                best_state = snapshot_model(bundle.model)
                best_update = update
            if float(metrics["ZS"]) > float(best_zs["ZS"]):
                best_zs = {"ZS": float(metrics["ZS"]), "update": update, "metrics": copy.deepcopy(metrics)}
                best_zs_state = copy.deepcopy(model_state)
            optimizer_payload = {
                "tg": tg_optimizer.state_dict(),
                "gate": gate_optimizer.state_dict() if gate_optimizer else None,
            }
            rng_state = _rng_state(primary_generator, auxiliary_generator)
            teacher_state = (
                teacher_packages_to_cpu(teacher) if bundle.module_name == "gtd" and teacher is not None
                else mmt_teacher_to_cpu(teacher) if bundle.module_name == "mmt" and teacher is not None
                else None
            )
            checkpoint = {
                "experiment_id": config["experiment_id"], "code_commit": code_commit,
                "config_sha256": config_sha, "initial_identity": initial_identity,
                "update": update, "model_state_dict": model_state,
                "tg_optimizer_state_dict": optimizer_payload["tg"],
                "gate_optimizer_state_dict": optimizer_payload["gate"],
                "scheduler_state_dict": scheduler.state_dict(),
                "best_update": best_update, "best_metrics": best_metrics,
                "best_model_state_dict": best_state, "best_zs_observation": best_zs,
                "best_zs_model_state_dict": best_zs_state,
                "teacher_state": teacher_state, "teacher_history": teacher_history,
                "first_update_gradients": first_update_gradients,
                "rng_state": rng_state, "history": history,
                "reproducibility": reproducibility,
            }
            seal_checkpoint(checkpoint)
            atomic_torch_save(output_dir / "checkpoint_last.pth", checkpoint)
        if len(history) != 152 or history[-1]["update"] != TOTAL_UPDATES:
            raise RuntimeError("fresh完整RUN必须有152个评估点并结束于21171。")
        if bundle.module_name in {"gtd", "mmt"}:
            expected_refreshes = teacher_refresh_updates(TOTAL_UPDATES)
            if [row["update"] for row in teacher_history] != list(expected_refreshes):
                raise RuntimeError(
                    "fresh teacher refresh必须覆盖150个完整区间和最终partial区间。"
                )
        atomic_torch_save(output_dir / "model_best.pth", {
            "experiment_id": config["experiment_id"], "code_commit": code_commit,
            "config_sha256": config_sha, "best_update": best_update,
            "best_metrics": best_metrics, "model_state_dict": best_state,
        })
        atomic_torch_save(output_dir / "model_best_zs.pth", {
            "experiment_id": config["experiment_id"], "code_commit": code_commit,
            "config_sha256": config_sha, "best_update": int(best_zs["update"]),
            "best_metrics": best_zs["metrics"],
            "model_state_dict": best_zs_state,
        })
        atomic_write_json(output_dir / "evaluation_history.json", {"rows": history})
        result = {
            "experiment_id": config["experiment_id"], "condition_id": config["condition_id"],
            "module": bundle.module_name, "code_commit": code_commit,
            "config_sha256": config_sha, **initial_identity,
            "best_metrics": best_metrics, "best_update": best_update,
            "best_full_minus_off_delta": best_metrics["full_minus_off_delta"],
            "best_zs_observation": best_zs, "first_update_gradients": first_update_gradients,
            "update1_pre_main_cuda_rng_sha256": first_update_gradients[
                "update1_pre_main_cuda_rng_sha256"
            ],
            "cross_run_add_delta_vs_try042": None,
            "gate_decision_pending_matched_try042": bundle.module_name != "tg",
            "stop_reason": "completed_fixed_150", "history_length": len(history),
            "test_used_for_selection": True, "test_used_for_hyperparameter_selection": True,
            "unseen_images_used_for_gradient": False, "strict_blind_claim": False,
            "human_annotations_used": False, "asset_id": config["asset_id"],
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "random_seed": int(config["random_seed"]),
            "batch_size": BATCH_SIZE,
            "total_updates": TOTAL_UPDATES,
            "eval_interval_steps": EVAL_INTERVAL,
            "tg_learning_rate": float(config["tg_learning_rate"]),
            "semantic_evidence_sha256": checkpoint["semantic_evidence_sha256"],
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "model_best_zs_sha256": sha256_file(output_dir / "model_best_zs.pth"),
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
            "evaluation_history_sha256": sha256_file(output_dir / "evaluation_history.json"),
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
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.resume_from)


if __name__ == "__main__":
    main()
