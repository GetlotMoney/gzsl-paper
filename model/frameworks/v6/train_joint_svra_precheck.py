"""IDEA-200 J-SVRA fixed-trace training precheck.

This runner trains three final-checkpoint conditions only:

1. Full joint objective: L_action + L_risk + L_joint.
2. No-joint control: L_action + L_risk on the same initialization/trace.
3. Sequential control: 500 policy updates then 500 frozen-policy risk updates.

It does not read official test features or labels.  Official evaluation is a
separate step over the saved checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from tools.reproducibility import configure_reproducibility
from tools.run_contract import current_code_commit, require_clean_code_tree

from .rwdg_data import (
    FORMAL_SVRA_CONFIG,
    ManifestContract,
    SVRAAssetConfig,
    SVRADataError,
    SVRAGateSubsetView,
    TensorContract,
    load_svra_gate_data,
    resolve_subset_output,
    sha256_file,
)
from .svra import (
    ACTION_COUNT,
    FEATURE_DIM,
    JSVRA_ACTION_POS_WEIGHT,
    JSVRA_RISK_POS_WEIGHT,
    JointSVRATargets,
    SemanticVisualRiskArbiter,
    joint_action_targets_from_logits,
    joint_svra_loss,
)

CHECKPOINT_SCHEMA = "gzsl-paper.v6-joint-svra-precheck-train.v1"
FULL_CONDITION = "JOINT_SVRA_FULL"
NO_JOINT_CONDITION = "JOINT_SVRA_NO_JOINT"
SEQUENTIAL_CONDITION = "JOINT_SVRA_SEQUENTIAL"
CONDITION_IDS: Mapping[str, str] = {
    "full_joint": FULL_CONDITION,
    "no_joint": NO_JOINT_CONDITION,
    "sequential": SEQUENTIAL_CONDITION,
}

EXPECTED_TARGET_CENSUS: Mapping[str, int] = {
    "rows": 7057,
    "abstain": 6065,
    "action": 992,
    "leader": 4485,
    "challenger": 1022,
    "outside": 1550,
    "conflict": 30,
}


@dataclass(frozen=True)
class JointPrecheckConfig:
    schema_version: str = "gzsl-paper.v6-joint-svra-precheck-train-config.v1"
    experiment_id: str = "V6-TRY-004"
    condition_id: str = "JOINT_SVRA_PRECHECK"
    output_dir: str = "/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/joint_svra/V6-TRY-004-PRECHECK"
    seed: int = 7
    batch_size: int = 50
    updates: int = 1000
    sequential_policy_updates: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    strict_sha: bool = True
    validate_tensor_values: bool = True
    verify_large_file_sha: bool = False
    require_clean_tree: bool = True
    strict_fixed_contract: bool = False
    expected_target_census: Mapping[str, int] | None = field(
        default_factory=lambda: dict(EXPECTED_TARGET_CENSUS)
    )


@dataclass(frozen=True)
class JointTrainTable:
    role_embeddings: Tensor
    name_embeddings: Tensor
    class_ids: Tensor
    full_cls: Tensor
    patch_tokens: Tensor
    all_crop_cls: Tensor
    target_class_ids: Tensor
    raw_indices: Tensor
    source_splits: tuple[str, ...]
    feature_positions: Tensor | None = None

    @property
    def rows(self) -> int:
        return int(self.full_cls.shape[0])

    def validate(self) -> None:
        if self.role_embeddings.shape != (
            self.name_embeddings.shape[0],
            8,
            FEATURE_DIM,
        ):
            raise ValueError("role_embeddings must be [C,8,768] and match name axis")
        if self.class_ids.shape != (self.name_embeddings.shape[0],):
            raise ValueError("class_ids must match the full text axis")
        if self.feature_positions is None:
            if self.full_cls.shape != (self.rows, FEATURE_DIM):
                raise ValueError("full_cls must be [N,768]")
            if self.patch_tokens.shape != (self.rows, 576, FEATURE_DIM):
                raise ValueError("patch_tokens must be [N,576,768]")
        else:
            if self.feature_positions.shape != (self.rows,):
                raise ValueError("feature_positions must be [N]")
            max_pos = int(self.feature_positions.max().item()) if self.rows else -1
            if self.full_cls.shape[0] <= max_pos or self.full_cls.shape[-1] != FEATURE_DIM:
                raise ValueError("full_cls must cover feature_positions and end in 768")
            if (
                self.patch_tokens.shape[0] <= max_pos
                or self.patch_tokens.shape[1:] != (576, FEATURE_DIM)
            ):
                raise ValueError("patch_tokens must cover feature_positions and be [*,576,768]")
        if self.all_crop_cls.shape != (self.rows, ACTION_COUNT, FEATURE_DIM):
            raise ValueError("all_crop_cls must be [N,25,768]")
        if self.target_class_ids.shape != (self.rows,):
            raise ValueError("target_class_ids must be [N]")
        if self.raw_indices.shape != (self.rows,):
            raise ValueError("raw_indices must be [N]")
        if len(self.source_splits) != self.rows:
            raise ValueError("source_splits length must match rows")

    def batch(self, rows: Tensor, *, device: torch.device) -> dict[str, Tensor]:
        rows = rows.to(dtype=torch.long, device="cpu")
        feature_rows = (
            self.feature_positions.index_select(0, rows).to(dtype=torch.long, device="cpu")
            if self.feature_positions is not None
            else rows
        )
        return {
            "rows": rows.to(device=device),
            "raw_indices": self.raw_indices.index_select(0, rows).to(device=device),
            "full_cls": _take_rows(self.full_cls, feature_rows, device=device),
            "patch_tokens": _take_rows(self.patch_tokens, feature_rows, device=device),
            "all_crop_cls": _take_rows(self.all_crop_cls, rows, device=device),
            "target_class_ids": self.target_class_ids.index_select(0, rows).to(device=device),
        }


def load_config(path: str | Path) -> tuple[JointPrecheckConfig, SVRAAssetConfig]:
    payload = _load_json(Path(path))
    train_cfg = JointPrecheckConfig(**payload.get("train", {}))
    asset_cfg = _asset_config_from_mapping(payload.get("assets"))
    return train_cfg, asset_cfg


def load_formal_joint_table(
    asset_config: SVRAAssetConfig = FORMAL_SVRA_CONFIG,
    *,
    strict_sha: bool = True,
    validate_tensor_values: bool = True,
    verify_large_file_sha: bool = False,
) -> tuple[JointTrainTable, Any]:
    assets, views = load_svra_gate_data(
        asset_config,
        strict_sha=strict_sha,
        validate_tensor_values=validate_tensor_values,
        strict_eval_boundary=False,
        verify_large_file_sha=verify_large_file_sha,
    )
    train = _load_subset_supervision(assets, views["dev_train"], "dev_train")
    oracle = _load_subset_supervision(assets, views["dev_eval"], "dev_eval_oracle")
    table = JointTrainTable(
        role_embeddings=assets.role_embeddings.float(),
        name_embeddings=assets.name_embeddings.float(),
        class_ids=torch.arange(assets.name_embeddings.shape[0], dtype=torch.long),
        full_cls=assets.cls_features.float(),
        patch_tokens=assets.patch_features,
        all_crop_cls=torch.cat([train["all_crop_cls"], oracle["all_crop_cls"]], dim=0),
        target_class_ids=torch.cat(
            [train["target_class_ids"], oracle["target_class_ids"]], dim=0
        ),
        raw_indices=torch.cat([train["raw_indices"], oracle["raw_indices"]], dim=0),
        source_splits=("dev_train",) * train["full_cls"].shape[0]
        + ("dev_eval_oracle",) * oracle["full_cls"].shape[0],
        feature_positions=torch.cat([train["feature_positions"], oracle["feature_positions"]], dim=0),
    )
    table.validate()
    return table, assets


def generate_batch_trace(
    *,
    rows: int,
    updates: int,
    batch_size: int,
    seed: int,
) -> list[Tensor]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if batch_size <= 0 or batch_size > rows:
        raise ValueError("batch_size must be in [1, rows]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return [
        torch.randperm(rows, generator=generator)[:batch_size].clone()
        for _ in range(updates)
    ]


def build_model(table: JointTrainTable, *, seed: int, device: torch.device) -> SemanticVisualRiskArbiter:
    return SemanticVisualRiskArbiter(
        table.role_embeddings,
        table.name_embeddings,
        table.class_ids,
        seed=seed,
    ).to(device)


def build_target_census(
    model: SemanticVisualRiskArbiter,
    table: JointTrainTable,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, int], list[Tensor]]:
    counts = {
        "rows": table.rows,
        "abstain": 0,
        "action": 0,
        "leader": 0,
        "challenger": 0,
        "outside": 0,
        "conflict": 0,
    }
    all_action_targets: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, table.rows, batch_size):
            rows = torch.arange(start, min(start + batch_size, table.rows))
            batch = table.batch(rows, device=device)
            pair = model.parent_state(batch["full_cls"])
            targets = joint_action_targets_from_logits(
                pair,
                batch["all_crop_cls"],
                model.name_embeddings,
                batch["target_class_ids"],
                model.class_ids,
            )
            all_action_targets.append(targets.action_targets26.detach().cpu())
            counts["abstain"] += int((targets.action_targets26 == 0).sum().item())
            counts["action"] += int((targets.action_targets26 > 0).sum().item())
            counts["leader"] += int((targets.top2_group == 0).sum().item())
            counts["challenger"] += int((targets.top2_group == 1).sum().item())
            counts["outside"] += int((targets.top2_group == 2).sum().item())
            counts["conflict"] += int(targets.conflict_mask.sum().item())
    return counts, all_action_targets


def train_precheck(
    table: JointTrainTable,
    config: JointPrecheckConfig,
    *,
    code_commit: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    table.validate()
    device = _resolve_device(config.device)
    if config.strict_fixed_contract:
        _assert_fixed_precheck_config(config)
    output_dir = _prepare_new_output_dir(config.output_dir)

    trace = generate_batch_trace(
        rows=table.rows,
        updates=config.updates,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    trace_sha = sha256_json([row.tolist() for row in trace])

    census_model = build_model(table, seed=config.seed, device=device)
    target_census, all_action_targets = build_target_census(
        census_model, table, batch_size=config.batch_size, device=device
    )
    _assert_expected_census(target_census, config.expected_target_census)
    target_census_path = output_dir / "target_census.json"
    action_histogram = _action_histogram(torch.cat(all_action_targets))
    _write_json(
        target_census_path,
        {
            "schema_version": "gzsl-paper.v6-joint-svra-target-census.v1",
            "experiment_id": config.experiment_id,
            "condition_id": config.condition_id,
            "census": target_census,
            "action_target_histogram26": action_histogram,
            "conflict_definition": "top2_group == challenger and action_targets26 == 0",
            "source_splits": {
                split: table.source_splits.count(split)
                for split in sorted(set(table.source_splits))
            },
            "full_axis_classes": int(table.class_ids.numel()),
        },
    )
    target_census_sha = sha256_file(target_census_path)

    base_model = build_model(table, seed=config.seed, device=device)
    base_state = copy.deepcopy(base_model.state_dict())
    base_sha = tensor_tree_sha256(base_state)
    receipts: dict[str, Any] = {
        "schema": "gzsl-paper.joint-svra.precheck-receipt.v1",
        "schema_version": "gzsl-paper.v6-joint-svra-precheck-train-receipt.v1",
        "experiment_id": config.experiment_id,
        "condition_id": config.condition_id,
        "code_commit": code_commit,
        "config_sha256": config_sha256,
        "official_test_loaded": False,
        "unseen_images_used_for_gradient": False,
        "test_used_for_selection": False,
        "full_axis_classes": int(table.class_ids.numel()),
        "train_rows": table.rows,
        "batch_trace_sha256": trace_sha,
        "target_census_path": str(target_census_path),
        "target_census_sha256": target_census_sha,
        "target_census": target_census,
        "action_target_histogram26": action_histogram,
        "target_action_sha256": tensor_sha256(torch.cat(all_action_targets)),
        "initialization_sha256": base_sha,
        "loss_scales": {
            "action_positive": JSVRA_ACTION_POS_WEIGHT,
            "risk_positive": JSVRA_RISK_POS_WEIGHT,
            "joint_positive": JSVRA_ACTION_POS_WEIGHT,
            "coefficients": {"action": 1.0, "risk": 1.0, "joint": 1.0},
        },
        "conditions": {},
    }
    checkpoint_specs: dict[str, dict[str, str]] = {}

    conditions = {
        "full_joint": _train_full_or_no_joint(
            copy.deepcopy(base_state), table, config, trace, device, joint=True
        ),
        "no_joint": _train_full_or_no_joint(
            copy.deepcopy(base_state), table, config, trace, device, joint=False
        ),
        "sequential": _train_sequential(
            copy.deepcopy(base_state), table, config, trace, device
        ),
    }
    _assert_gradient_contracts(conditions)
    for name, result in conditions.items():
        checkpoint_path = output_dir / f"{name}_final.pt"
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "experiment_id": config.experiment_id,
                "condition": name,
                "condition_id": CONDITION_IDS[name],
                "code_commit": code_commit,
                "config_sha256": config_sha256,
                "state_dict": _cpu_state_dict(result.pop("state_dict")),
                "target_census_sha256": target_census_sha,
                "target_census": target_census,
                "action_target_histogram26": action_histogram,
                "batch_trace_sha256": trace_sha,
                "full_axis_classes": int(table.class_ids.numel()),
            },
            checkpoint_path,
        )
        result["checkpoint_path"] = str(checkpoint_path)
        result["checkpoint_sha256"] = sha256_file(checkpoint_path)
        result["condition_id"] = CONDITION_IDS[name]
        checkpoint_specs[name] = {
            "path": str(checkpoint_path),
            "sha256": result["checkpoint_sha256"],
            "training_commit": str(code_commit),
            "train_config_sha256": str(config_sha256),
        }
        receipts["conditions"][name] = result
    receipts["checkpoint_specs"] = checkpoint_specs

    receipt_path = output_dir / "train_receipt.json"
    _write_json(receipt_path, receipts)
    receipts["receipt_path"] = str(receipt_path)
    receipts["receipt_sha256"] = sha256_file(receipt_path)
    return receipts


def _train_full_or_no_joint(
    initial_state: Mapping[str, Tensor],
    table: JointTrainTable,
    config: JointPrecheckConfig,
    trace: Sequence[Tensor],
    device: torch.device,
    *,
    joint: bool,
) -> dict[str, Any]:
    model = build_model(table, seed=config.seed, device=device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    grad_receipt: dict[str, Any] = {}
    last_loss: dict[str, float] = {}
    for step, rows in enumerate(trace, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss_parts, digest = _loss_for_rows(model, table, rows, device)
        loss = loss_parts.total if joint else (loss_parts.action + loss_parts.risk)
        loss.backward()
        _capture_grad_receipt(grad_receipt, step, len(trace), model)
        optimizer.step()
        last_loss = {
            "total": _float(loss),
            "action": _float(loss_parts.action),
            "risk": _float(loss_parts.risk),
            "joint": _float(loss_parts.joint),
            "soft_hard_trigger_equal": bool(loss_parts.soft_hard_trigger_equal),
            **{f"raw_{k}": _float(v) for k, v in loss_parts.raw_means.items()},
            **{f"weighted_{k}": _float(v) for k, v in loss_parts.weighted_means.items()},
        }
    final_digest = _condition_digest(model, table, trace[-1], device)
    final_digest["last_train_batch_digest"] = digest
    return {
        "state_dict": model.state_dict(),
        "objective": "L_action+L_risk+L_joint" if joint else "L_action+L_risk",
        "updates": len(trace),
        "last_loss": last_loss,
        "gradients": grad_receipt,
        "digests": final_digest,
    }


def _train_sequential(
    initial_state: Mapping[str, Tensor],
    table: JointTrainTable,
    config: JointPrecheckConfig,
    trace: Sequence[Tensor],
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(table, seed=config.seed, device=device)
    model.load_state_dict(initial_state)
    policy_params = list(model.semantic.parameters()) + list(model.visual.parameters())
    policy_optimizer = torch.optim.AdamW(policy_params, lr=config.lr, weight_decay=config.weight_decay)
    grad_receipt: dict[str, Any] = {}
    policy_updates = min(config.sequential_policy_updates, len(trace))
    for step, rows in enumerate(trace[:policy_updates], start=1):
        policy_optimizer.zero_grad(set_to_none=True)
        loss_parts, _ = _loss_for_rows(model, table, rows, device)
        loss_parts.action.backward()
        _capture_grad_receipt(grad_receipt, step, len(trace), model)
        policy_optimizer.step()

    final_trigger = _all_train_triggers(model, table, batch_size=config.batch_size, device=device)
    for parameter in list(model.semantic.parameters()) + list(model.visual.parameters()):
        parameter.requires_grad_(False)
    risk_optimizer = torch.optim.AdamW(model.interaction.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    last_loss: dict[str, float] = {}
    for offset, rows in enumerate(trace[policy_updates:], start=policy_updates + 1):
        risk_optimizer.zero_grad(set_to_none=True)
        batch = table.batch(rows, device=device)
        output = model.joint_forward(batch["full_cls"], batch["patch_tokens"])
        targets = joint_action_targets_from_logits(
            output.pair,
            batch["all_crop_cls"],
            model.name_embeddings,
            batch["target_class_ids"],
            model.class_ids,
        )
        batch_trigger = final_trigger.index_select(0, rows.to(dtype=torch.long)).to(device)
        masked_loss = _masked_risk_loss(
            output.risk_logits,
            targets.risk_targets,
            batch_trigger,
            model,
        )
        masked_loss.backward()
        _capture_grad_receipt(grad_receipt, offset, len(trace), model)
        risk_optimizer.step()
        last_loss = {"risk_on_final_trigger": _float(masked_loss)}
    for parameter in list(model.semantic.parameters()) + list(model.visual.parameters()):
        parameter.requires_grad_(True)
    final_digest = _condition_digest(model, table, trace[-1], device)
    return {
        "state_dict": model.state_dict(),
        "objective": "500_policy_then_500_frozen_risk_on_final_hard_trigger",
        "updates": len(trace),
        "policy_updates": policy_updates,
        "risk_updates": max(0, len(trace) - policy_updates),
        "final_trigger_count": int(final_trigger.sum().item()),
        "last_loss": last_loss,
        "gradients": grad_receipt,
        "digests": final_digest,
    }


def _loss_for_rows(
    model: SemanticVisualRiskArbiter,
    table: JointTrainTable,
    rows: Tensor,
    device: torch.device,
) -> tuple[Any, dict[str, str]]:
    batch = table.batch(rows, device=device)
    output = model.joint_forward(batch["full_cls"], batch["patch_tokens"])
    targets = joint_action_targets_from_logits(
        output.pair,
        batch["all_crop_cls"],
        model.name_embeddings,
        batch["target_class_ids"],
        model.class_ids,
    )
    loss = joint_svra_loss(output, targets)
    if not loss.soft_hard_trigger_equal:
        raise RuntimeError("soft/hard opportunity trigger equivalence failed")
    digest = {
        "batch_sha256": tensor_sha256(batch["rows"]),
        "logit_sha256": tensor_sha256(output.logits.detach()),
        "action_sha256": tensor_sha256(output.action_state.selected_action.detach()),
        "trigger_sha256": tensor_sha256(output.hard_trigger.detach()),
        "swap_sha256": tensor_sha256(output.swapped.detach()),
    }
    return loss, digest


def _condition_digest(
    model: SemanticVisualRiskArbiter,
    table: JointTrainTable,
    rows: Tensor,
    device: torch.device,
) -> dict[str, str | bool]:
    with torch.no_grad():
        batch = table.batch(rows, device=device)
        output = model.joint_forward(batch["full_cls"], batch["patch_tokens"])
        return {
            "batch_sha256": tensor_sha256(batch["rows"]),
            "logit_sha256": tensor_sha256(output.logits),
            "parent_logit_sha256": tensor_sha256(output.parent_logits),
            "action_sha256": tensor_sha256(output.action_state.selected_action),
            "trigger_sha256": tensor_sha256(output.hard_trigger),
            "swap_sha256": tensor_sha256(output.swapped),
            "soft_hard_trigger_equal": output.soft_hard_trigger_equal,
        }


def _all_train_triggers(
    model: SemanticVisualRiskArbiter,
    table: JointTrainTable,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, table.rows, batch_size):
            rows = torch.arange(start, min(start + batch_size, table.rows))
            batch = table.batch(rows, device=device)
            output = model.joint_forward(batch["full_cls"], batch["patch_tokens"])
            chunks.append(output.hard_trigger.detach().cpu())
    return torch.cat(chunks, dim=0)


def _masked_risk_loss(
    risk_logits: Tensor,
    risk_targets: Tensor,
    trigger: Tensor,
    model: SemanticVisualRiskArbiter,
) -> Tensor:
    if not bool(trigger.any()):
        return model.interaction.output.weight.sum() * 0.0
    target = risk_targets.to(device=risk_logits.device).float()
    raw = F.binary_cross_entropy_with_logits(risk_logits.float(), target, reduction="none")
    weight = torch.where(
        target > 0,
        risk_logits.new_full((), float(JSVRA_RISK_POS_WEIGHT)),
        risk_logits.new_ones(()),
    )
    mask = trigger.to(device=risk_logits.device).bool()
    return (raw[mask] * weight[mask]).mean()


def _capture_grad_receipt(
    out: dict[str, Any],
    step: int,
    final_step: int,
    model: SemanticVisualRiskArbiter,
) -> None:
    if step not in {1, 2, final_step}:
        return
    out[f"step{step}"] = {
        "S": _grad_norms(model.semantic.parameters()),
        "V": _grad_norms(model.visual.parameters()),
        "I": _grad_norms(model.interaction.parameters()),
    }


def _grad_norms(parameters: Sequence[torch.nn.Parameter] | Any) -> dict[str, Any]:
    total = 0.0
    finite = True
    nonzero_params = 0
    param_count = 0
    for param in parameters:
        param_count += 1
        grad = param.grad
        if grad is None:
            continue
        finite = finite and bool(torch.isfinite(grad).all())
        norm = float(grad.detach().float().norm().cpu().item())
        total += norm
        if norm > 0:
            nonzero_params += 1
    return {
        "finite": finite,
        "total_norm": total,
        "nonzero": total > 0,
        "nonzero_params": nonzero_params,
        "param_count": param_count,
    }


def _assert_expected_census(
    actual: Mapping[str, int],
    expected: Mapping[str, int] | None,
) -> None:
    if expected is None:
        return
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"target census mismatch: {mismatches}")


def _assert_fixed_precheck_config(config: JointPrecheckConfig) -> None:
    invalid = (
        config.schema_version != "gzsl-paper.v6-joint-svra-precheck-train-config.v1"
        or config.seed != 7
        or config.batch_size != 50
        or config.updates != 1000
        or config.sequential_policy_updates != 500
        or float(config.lr) != 1e-3
        or float(config.weight_decay) != 0.0
        or not str(config.device).startswith("cuda")
        or config.expected_target_census != EXPECTED_TARGET_CENSUS
    )
    if invalid:
        raise RuntimeError("J-SVRA formal precheck config violates the fixed IDEA-200 contract")


def _prepare_new_output_dir(path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        raise RuntimeError(f"output_dir already exists and would overwrite receipts: {output}")
    output.mkdir(parents=True)
    return output


def _action_histogram(action_targets26: Tensor) -> list[int]:
    return [int(x) for x in torch.bincount(action_targets26.cpu().long(), minlength=ACTION_COUNT + 1)]


def _cpu_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _assert_gradient_contracts(conditions: Mapping[str, Mapping[str, Any]]) -> None:
    for name in ("full_joint", "no_joint"):
        gradients = conditions[name]["gradients"]
        for step in ("step2", f"step{conditions[name]['updates']}"):
            if step not in gradients:
                raise RuntimeError(f"{name}: missing gradient receipt {step}")
            for module in ("S", "V", "I"):
                stats = gradients[step][module]
                if not stats["finite"] or not stats["nonzero"]:
                    raise RuntimeError(f"{name}: {step} {module} gradient must be finite and nonzero")
    sequential = conditions["sequential"]
    if sequential["policy_updates"] >= 2:
        policy_step = sequential["gradients"].get("step2")
        if policy_step is None:
            raise RuntimeError("sequential: missing policy-phase step2 gradient receipt")
        for module in ("S", "V"):
            stats = policy_step[module]
            if not stats["finite"] or not stats["nonzero"]:
                raise RuntimeError(f"sequential: policy step2 {module} gradient must be finite and nonzero")
    final_step = f"step{sequential['updates']}"
    if final_step not in sequential["gradients"]:
        raise RuntimeError("sequential: missing final gradient receipt")
    final_i = sequential["gradients"][final_step]["I"]
    if not final_i["finite"]:
        raise RuntimeError("sequential: final I gradient must be finite")
    if sequential["final_trigger_count"] > 0 and not final_i["nonzero"]:
        raise RuntimeError("sequential: final I gradient must be nonzero when trigger cohort is non-empty")


def tensor_sha256(value: Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def tensor_tree_sha256(state: Mapping[str, Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, Mapping):
        raise ValueError(f"config must be JSON object: {path}")
    return value


def _float(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _take_rows(value: Any, rows: Tensor, *, device: torch.device) -> Tensor:
    row_array = rows.detach().cpu().numpy().astype(np.int64, copy=False)
    if torch.is_tensor(value):
        out = value.index_select(0, rows)
    else:
        out = torch.as_tensor(np.asarray(value[row_array]))
    return out.float().to(device)


def _resolve_device(raw: str) -> torch.device:
    if raw.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("J-SVRA precheck training requires CUDA for formal runs")
    return torch.device(raw)


def _load_subset_supervision(
    assets: Any,
    view: SVRAGateSubsetView,
    subset_name: str,
) -> dict[str, Tensor]:
    rows = np.arange(view.size, dtype=np.int64)
    view_batch = view.batch(rows, include_patches=False, as_torch=True, device="cpu")
    labels = _load_first_available_tensor(
        assets,
        subset_name,
        ("labels.pt", "labels.npy", "targets.pt", "targets.npy"),
    ).long()
    crops = _load_first_available_tensor(
        assets,
        subset_name,
        ("all25_crop_features.npy", "crop_features.npy", "all25_crop_features.pt", "crop_features.pt"),
    ).float()
    if labels.shape != (view.size,):
        raise SVRADataError(f"{subset_name}: labels shape {tuple(labels.shape)} != ({view.size},)")
    if crops.shape != (view.size, ACTION_COUNT, FEATURE_DIM):
        raise SVRADataError(
            f"{subset_name}: crop features shape {tuple(crops.shape)} != ({view.size},25,768)"
        )
    return {
        "full_cls": view_batch["cls"].float(),
        "all_crop_cls": crops,
        "target_class_ids": labels,
        "raw_indices": view_batch["raw_global_indices"].long(),
        "feature_positions": view_batch["trainval_positions"].long(),
    }


def _load_first_available_tensor(
    assets: Any,
    subset_name: str,
    filenames: Sequence[str],
) -> Tensor:
    last_error: Exception | None = None
    for filename in filenames:
        try:
            path = resolve_subset_output(assets, subset_name, filename, verify_sha=True)
            return _load_tensor_path(path)
        except Exception as exc:  # keep trying alternate canonical names
            last_error = exc
    raise SVRADataError(
        f"{subset_name}: none of {', '.join(filenames)} could be loaded; last_error={last_error}"
    )


def _load_tensor_path(path: Path) -> Tensor:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, Mapping):
            for key in ("labels", "targets", "crop_features", "all25_crop_features"):
                candidate = value.get(key)
                if torch.is_tensor(candidate):
                    return candidate
            for candidate in value.values():
                if torch.is_tensor(candidate):
                    return candidate
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(value)
    if suffix == ".npy":
        return torch.as_tensor(np.asarray(np.load(path, mmap_mode="r")))
    raise SVRADataError(f"unsupported tensor path: {path}")


def _asset_config_from_mapping(raw: Any) -> SVRAAssetConfig:
    if raw is None:
        return FORMAL_SVRA_CONFIG
    if not isinstance(raw, Mapping):
        raise ValueError("assets config must be object")
    return SVRAAssetConfig(
        text_manifest=_manifest(raw["text_manifest"]),
        role_tensor=_tensor(raw["role_tensor"]),
        name_tensor=_tensor(raw["name_tensor"]),
        patch_manifest=_manifest(raw["patch_manifest"]),
        cls_tensor=_tensor(raw["cls_tensor"]),
        patch_tensor=_tensor(raw["patch_tensor"]),
        action_bundle_manifest=_manifest(raw["action_bundle_manifest"]),
        dev_train_manifest_sha256=str(raw["dev_train_manifest_sha256"]),
        dev_eval_manifest_sha256=str(raw["dev_eval_manifest_sha256"]),
        dev_eval_oracle_manifest_sha256=str(raw["dev_eval_oracle_manifest_sha256"]),
        att_splits_mat_path=raw.get("att_splits_mat_path"),
        trainval_count=int(raw.get("trainval_count", 7057)),
    )


def _manifest(raw: Mapping[str, Any]) -> ManifestContract:
    return ManifestContract(path=str(raw["path"]), sha256=raw.get("sha256"))


def _tensor(raw: Mapping[str, Any]) -> TensorContract:
    return TensorContract(
        path=str(raw["path"]),
        sha256=raw.get("sha256"),
        shape=tuple(int(x) for x in raw["shape"]),
        dtype=raw.get("dtype"),
    )


def run(config_path: str | Path, *, expected_commit: str, expected_config_sha: str) -> dict[str, Any]:
    config_path = Path(config_path)
    actual_config_sha = sha256_file(config_path)
    if actual_config_sha.lower() != expected_config_sha.lower():
        raise RuntimeError(
            f"J-SVRA train config SHA mismatch: got {actual_config_sha}, expected {expected_config_sha}"
        )
    train_cfg, asset_cfg = load_config(config_path)
    _assert_fixed_precheck_config(train_cfg)
    if train_cfg.require_clean_tree:
        require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise RuntimeError(
            f"J-SVRA train commit mismatch: got {code_commit}, expected {expected_commit}"
        )
    configure_reproducibility(
        train_cfg.seed,
        strict_determinism=True,
        deterministic_warn_only=False,
    )
    table, assets = load_formal_joint_table(
        asset_cfg,
        strict_sha=train_cfg.strict_sha,
        validate_tensor_values=train_cfg.validate_tensor_values,
        verify_large_file_sha=train_cfg.verify_large_file_sha,
    )
    try:
        return train_precheck(
            table,
            train_cfg,
            code_commit=code_commit,
            config_sha256=actual_config_sha,
        )
    finally:
        close = getattr(assets, "close", None)
        if close is not None:
            close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args(argv)
    receipt = run(
        args.config,
        expected_commit=args.expected_commit,
        expected_config_sha=args.expected_config_sha,
    )
    print(
        json.dumps(
            {
                "receipt_path": receipt["receipt_path"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
