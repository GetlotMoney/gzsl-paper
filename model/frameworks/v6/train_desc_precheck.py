"""IDEA-202 DESC fixed-trace training precheck.

DESC trains a direct evidence-conditioned keep/swap logit.  It does not read
official test features or labels; official evaluation is a separate frozen-logit
step over the saved checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from tools.reproducibility import configure_reproducibility
from tools.run_contract import current_code_commit, require_clean_code_tree

from .rwdg_data import (
    FORMAL_SVRA_CONFIG,
    ManifestContract,
    SVRAAssetConfig,
    SVRAAssets,
    SVRADataError,
    SVRAGateSubsetView,
    TensorContract,
    load_svra_gate_data,
    resolve_subset_output,
    sha256_file,
)
from .svra import (
    ACTION_COUNT,
    DirectSVRALoss,
    FEATURE_DIM,
    SemanticVisualRiskArbiter,
    direct_svra_loss,
    joint_action_targets_from_logits,
)


CHECKPOINT_SCHEMA = "gzsl-paper.v6-desc-precheck-train.v1"
EXPECTED_TARGET_CENSUS: Mapping[str, int] = {
    "rows": 7057,
    "abstain": 6065,
    "action": 992,
    "leader": 4485,
    "challenger": 1022,
    "outside": 1550,
    "conflict": 30,
}
FULL_CONDITION = "DESC_FULL"
NO_ACTION_AUX_CONDITION = "DESC_NO_ACTION_AUX"
PARENT_ONLY_CONDITION = "DESC_PARENT_ONLY"
CONDITION_IDS: Mapping[str, str] = {
    "full": FULL_CONDITION,
    "no_action_aux": NO_ACTION_AUX_CONDITION,
    "parent_only": PARENT_ONLY_CONDITION,
}


@dataclass(frozen=True)
class JointTrainTable:
    """Formal DESC train table: full trainval features plus train/dev-oracle all25 evidence."""

    role_embeddings: Tensor
    name_embeddings: Tensor
    class_ids: Tensor
    full_cls: Any
    patch_tokens: Any
    all_crop_cls: Any
    target_class_ids: Tensor
    raw_indices: Tensor
    source_splits: tuple[str, ...]
    feature_positions: Tensor | None = None

    @property
    def rows(self) -> int:
        return int(self.target_class_ids.shape[0])

    def validate(self) -> None:
        if self.role_embeddings.shape != (self.class_ids.numel(), 8, FEATURE_DIM):
            raise RuntimeError(
                "role_embeddings must have shape [class_count,8,768], "
                f"got {tuple(self.role_embeddings.shape)}"
            )
        if self.name_embeddings.shape != (self.class_ids.numel(), FEATURE_DIM):
            raise RuntimeError(
                "name_embeddings must have shape [class_count,768], "
                f"got {tuple(self.name_embeddings.shape)}"
            )
        if self.target_class_ids.shape != (self.rows,):
            raise RuntimeError("target_class_ids must be a 1-D train table axis")
        if self.raw_indices.shape != (self.rows,):
            raise RuntimeError("raw_indices must match target_class_ids")
        if len(self.source_splits) != self.rows:
            raise RuntimeError("source_splits must match target_class_ids")
        if self.feature_positions is not None and self.feature_positions.shape != (self.rows,):
            raise RuntimeError("feature_positions must match target_class_ids")
        _validate_first_axis("full_cls", self.full_cls, self.rows, self.feature_positions)
        _validate_first_axis("patch_tokens", self.patch_tokens, self.rows, self.feature_positions)
        _validate_first_axis("all_crop_cls", self.all_crop_cls, self.rows, None)

    def batch(self, rows: Tensor | Sequence[int], *, device: torch.device) -> dict[str, Tensor]:
        row_tensor = torch.as_tensor(rows, dtype=torch.long)
        if row_tensor.ndim != 1:
            raise RuntimeError("batch rows must be 1-D")
        if row_tensor.numel() and (int(row_tensor.min()) < 0 or int(row_tensor.max()) >= self.rows):
            raise RuntimeError("batch rows outside JointTrainTable axis")
        feature_rows = (
            self.feature_positions.index_select(0, row_tensor)
            if self.feature_positions is not None
            else row_tensor
        )
        return {
            "rows": row_tensor.to(device=device),
            "feature_positions": feature_rows.to(device=device),
            "raw_indices": self.raw_indices.index_select(0, row_tensor).to(device=device),
            "full_cls": _take_rows(self.full_cls, feature_rows, device=device),
            "patch_tokens": _take_rows(self.patch_tokens, feature_rows, device=device),
            "all_crop_cls": _take_rows(self.all_crop_cls, row_tensor, device=device),
            "target_class_ids": self.target_class_ids.index_select(0, row_tensor).to(device=device),
        }


def _validate_first_axis(
    name: str,
    value: Any,
    rows: int,
    feature_positions: Tensor | None,
) -> None:
    shape = tuple(int(x) for x in value.shape)
    if not shape:
        raise RuntimeError(f"{name} must have a batch axis")
    required_rows = rows
    if feature_positions is not None and feature_positions.numel():
        required_rows = int(feature_positions.max().item()) + 1
    if shape[0] < required_rows:
        raise RuntimeError(f"{name} first axis {shape[0]} cannot cover required rows {required_rows}")


def _take_rows(value: Any, rows: Tensor, *, device: torch.device) -> Tensor:
    rows_cpu = rows.detach().cpu().long()
    if torch.is_tensor(value):
        return value.index_select(0, rows_cpu).to(device=device).float()
    return torch.as_tensor(np.asarray(value)[rows_cpu.numpy()], dtype=torch.float32, device=device)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _prepare_new_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    if output_dir.exists():
        raise RuntimeError(f"output_dir already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    return output_dir


def _resolve_device(raw: str) -> torch.device:
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DESC precheck requires CUDA for cuda device config")
    return device


def _float(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def generate_batch_trace(*, rows: int, updates: int, batch_size: int, seed: int) -> list[Tensor]:
    if rows <= 0 or updates <= 0 or batch_size <= 0:
        raise RuntimeError("rows/updates/batch_size must be positive")
    if batch_size > rows:
        raise RuntimeError("batch_size cannot exceed training rows")
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(rows, generator=generator)[:batch_size].long() for _ in range(updates)]


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def tensor_tree_sha256(state: Mapping[str, Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        h.update(b"\0")
    return h.hexdigest()


def _cpu_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _action_histogram(targets26: Tensor) -> list[int]:
    return [int(x) for x in torch.bincount(targets26.detach().cpu().long(), minlength=ACTION_COUNT + 1)]


def _assert_expected_census(actual: Mapping[str, int], expected: Mapping[str, int] | None) -> None:
    if expected is None:
        return
    if dict(actual) != dict(expected):
        raise RuntimeError(f"DESC target census mismatch: got {dict(actual)}, expected {dict(expected)}")


def _asset_config_from_mapping(value: Any) -> SVRAAssetConfig:
    if value is None:
        return FORMAL_SVRA_CONFIG
    if not isinstance(value, Mapping):
        raise RuntimeError("assets config must be a mapping")
    return SVRAAssetConfig(
        text_manifest=_manifest(value["text_manifest"]),
        role_tensor=_tensor(value["role_tensor"]),
        name_tensor=_tensor(value["name_tensor"]),
        patch_manifest=_manifest(value["patch_manifest"]),
        cls_tensor=_tensor(value["cls_tensor"]),
        patch_tensor=_tensor(value["patch_tensor"]),
        action_bundle_manifest=_manifest(value["action_bundle_manifest"]),
        dev_train_manifest_sha256=str(value["dev_train_manifest_sha256"]),
        dev_eval_manifest_sha256=str(value["dev_eval_manifest_sha256"]),
        dev_eval_oracle_manifest_sha256=str(value["dev_eval_oracle_manifest_sha256"]),
        att_splits_mat_path=value.get("att_splits_mat_path"),
        trainval_count=int(value.get("trainval_count", 7057)),
    )


def _manifest(value: Any) -> ManifestContract:
    if not isinstance(value, Mapping):
        raise RuntimeError("manifest contract must be a mapping")
    return ManifestContract(path=str(value["path"]), sha256=value.get("sha256"))


def _tensor(value: Any) -> TensorContract:
    if not isinstance(value, Mapping):
        raise RuntimeError("tensor contract must be a mapping")
    return TensorContract(
        path=str(value["path"]),
        sha256=value.get("sha256"),
        shape=tuple(int(x) for x in value["shape"]),
        dtype=value.get("dtype"),
    )


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _first_tensor(value: Any, *, source: Path) -> Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        for item in value.values():
            if torch.is_tensor(item):
                return item.detach().cpu()
    raise SVRADataError(f"{source} contains no tensor")


def _load_tensor_path(path: Path) -> Tensor:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return _first_tensor(_torch_load_cpu(path), source=path)
    if suffix == ".npy":
        return torch.as_tensor(np.asarray(np.load(path, mmap_mode="r")))
    raise SVRADataError(f"unsupported tensor file type: {path}")


def _resolve_first_subset_output(
    assets: SVRAAssets,
    subset_name: str,
    filenames: Sequence[str],
    *,
    verify_sha: bool,
) -> Path:
    errors: list[str] = []
    for filename in filenames:
        try:
            return resolve_subset_output(assets, subset_name, filename, verify_sha=verify_sha)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    raise SVRADataError(f"{subset_name}: cannot resolve any of {filenames}; " + " | ".join(errors))


def _load_subset_supervision(
    assets: SVRAAssets,
    view: SVRAGateSubsetView,
    subset_name: str,
    *,
    verify_sha: bool,
) -> dict[str, Tensor]:
    rows = np.arange(view.size, dtype=np.int64)
    batch = view.batch(rows, include_patches=False, as_torch=True, device="cpu")
    labels_path = _resolve_first_subset_output(
        assets,
        subset_name,
        ("labels.pt", "labels.npy", "targets.pt", "targets.npy", "target_class_ids.pt", "target_class_ids.npy"),
        verify_sha=verify_sha,
    )
    crops_path = _resolve_first_subset_output(
        assets,
        subset_name,
        ("all25_crop_features.npy", "all25_crop_features.pt", "crop_features.npy", "crop_features.pt"),
        verify_sha=verify_sha,
    )
    target_class_ids = _load_tensor_path(labels_path).long().reshape(-1)
    all_crop_cls = _load_tensor_path(crops_path).float()
    if target_class_ids.shape != (view.size,):
        raise SVRADataError(f"{subset_name}: labels shape {tuple(target_class_ids.shape)} != ({view.size},)")
    if all_crop_cls.shape != (view.size, ACTION_COUNT, FEATURE_DIM):
        raise SVRADataError(
            f"{subset_name}: all25 crop shape {tuple(all_crop_cls.shape)} != "
            f"({view.size},{ACTION_COUNT},{FEATURE_DIM})"
        )
    return {
        "target_class_ids": target_class_ids,
        "all_crop_cls": all_crop_cls,
        "raw_indices": batch["raw_indices"].long().cpu(),
        "feature_positions": batch["trainval_positions"].long().cpu(),
    }


def _subset_view_from_manifest(
    assets: SVRAAssets,
    subset_name: str,
    *,
    verify_sha: bool,
) -> SVRAGateSubsetView:
    raw_path = _resolve_first_subset_output(
        assets,
        subset_name,
        ("raw_indices.pt", "raw_indices.npy", "raw_indices.json"),
        verify_sha=verify_sha,
    )
    if raw_path.suffix.lower() == ".json":
        raw_value = json.loads(raw_path.read_text(encoding="utf-8"))
        if isinstance(raw_value, Mapping):
            raw_value = raw_value.get("raw_indices", raw_value.get("indices"))
        raw_indices = np.asarray(raw_value, dtype=np.int64).reshape(-1)
    else:
        raw_indices = _load_tensor_path(raw_path).long().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    missing = [int(x) for x in raw_indices.tolist() if int(x) not in assets.raw_global_to_trainval_position]
    if missing:
        raise SVRADataError(f"{subset_name}: raw indices missing from trainval_loc, examples={missing[:5]}")
    positions = np.asarray(
        [assets.raw_global_to_trainval_position[int(x)] for x in raw_indices.tolist()],
        dtype=np.int64,
    )
    return SVRAGateSubsetView(subset_name, raw_indices, positions, assets)


def load_formal_joint_table(
    asset_config: SVRAAssetConfig = FORMAL_SVRA_CONFIG,
    *,
    strict_sha: bool = True,
    validate_tensor_values: bool = True,
    verify_large_file_sha: bool = False,
) -> tuple[JointTrainTable, SVRAAssets]:
    assets, views = load_svra_gate_data(
        asset_config,
        strict_sha=strict_sha,
        validate_tensor_values=validate_tensor_values,
        strict_eval_boundary=False,
        verify_large_file_sha=verify_large_file_sha,
    )
    train = _load_subset_supervision(assets, views["dev_train"], "dev_train", verify_sha=strict_sha)
    oracle_view = _subset_view_from_manifest(assets, "dev_eval_oracle", verify_sha=strict_sha)
    oracle = _load_subset_supervision(assets, oracle_view, "dev_eval_oracle", verify_sha=strict_sha)
    table = JointTrainTable(
        role_embeddings=assets.role_embeddings.float(),
        name_embeddings=assets.name_embeddings.float(),
        class_ids=torch.arange(assets.name_embeddings.shape[0], dtype=torch.long),
        full_cls=assets.cls_features.float(),
        patch_tokens=assets.patch_features,
        all_crop_cls=torch.cat((train["all_crop_cls"], oracle["all_crop_cls"]), dim=0),
        target_class_ids=torch.cat((train["target_class_ids"], oracle["target_class_ids"]), dim=0),
        raw_indices=torch.cat((train["raw_indices"], oracle["raw_indices"]), dim=0),
        source_splits=("dev_train",) * int(train["target_class_ids"].numel())
        + ("dev_eval_oracle",) * int(oracle["target_class_ids"].numel()),
        feature_positions=torch.cat((train["feature_positions"], oracle["feature_positions"]), dim=0),
    )
    table.validate()
    return table, assets


@dataclass(frozen=True)
class DESCPrecheckConfig:
    schema_version: str = "gzsl-paper.v6-desc-precheck-train-config.v1"
    experiment_id: str = "V6-TRY-005"
    condition_id: str = "DESC_PRECHECK"
    output_dir: str = "/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/desc/V6-TRY-005-PRECHECK"
    seed: int = 7
    batch_size: int = 50
    updates: int = 1000
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


def load_config(path: str | Path) -> tuple[DESCPrecheckConfig, SVRAAssetConfig]:
    payload = _load_json(Path(path))
    train_cfg = DESCPrecheckConfig(**payload.get("train", {}))
    asset_cfg = _asset_config_from_mapping(payload.get("assets"))
    return train_cfg, asset_cfg


def build_model(table: JointTrainTable, *, seed: int, device: torch.device) -> SemanticVisualRiskArbiter:
    return SemanticVisualRiskArbiter(
        table.role_embeddings,
        table.name_embeddings,
        table.class_ids,
        seed=seed,
    ).to(device)


def train_desc_precheck(
    table: JointTrainTable,
    config: DESCPrecheckConfig,
    *,
    code_commit: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    table.validate()
    if config.strict_fixed_contract:
        _assert_fixed_desc_config(config)
    device = _resolve_device(config.device)
    output_dir = _prepare_new_output_dir(config.output_dir)
    trace = generate_batch_trace(
        rows=table.rows,
        updates=config.updates,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    trace_sha = sha256_json([row.tolist() for row in trace])

    census_model = build_model(table, seed=config.seed, device=device)
    target_census, all_action_targets = _build_desc_target_census(
        census_model,
        table,
        batch_size=config.batch_size,
        device=device,
    )
    _assert_expected_census(target_census, config.expected_target_census)
    action_histogram = _action_histogram(torch.cat(all_action_targets))
    target_census_path = output_dir / "target_census.json"
    _write_json(
        target_census_path,
        {
            "schema_version": "gzsl-paper.v6-desc-target-census.v1",
            "experiment_id": config.experiment_id,
            "condition_id": config.condition_id,
            "census": target_census,
            "swap_positive": target_census["challenger"],
            "swap_negative": target_census["leader"] + target_census["outside"],
            "action_target_histogram26": action_histogram,
            "full_axis_classes": int(table.class_ids.numel()),
        },
    )
    target_census_sha = sha256_file(target_census_path)

    base_model = build_model(table, seed=config.seed, device=device)
    base_state = copy.deepcopy(base_model.state_dict())
    init_sha = tensor_tree_sha256(base_state)
    conditions = {
        "full": _train_condition(
            copy.deepcopy(base_state),
            table,
            config,
            trace,
            device,
            include_action=True,
            parent_only=False,
        ),
        "no_action_aux": _train_condition(
            copy.deepcopy(base_state),
            table,
            config,
            trace,
            device,
            include_action=False,
            parent_only=False,
        ),
        "parent_only": _train_condition(
            copy.deepcopy(base_state),
            table,
            config,
            trace,
            device,
            include_action=False,
            parent_only=True,
        ),
    }
    _assert_gradient_contracts(conditions)

    receipt: dict[str, Any] = {
        "schema_version": "gzsl-paper.v6-desc-precheck-train-receipt.v1",
        "experiment_id": config.experiment_id,
        "condition_id": config.condition_id,
        "code_commit": code_commit,
        "config_sha256": config_sha256,
        "official_test_loaded": False,
        "unseen_images_used_for_gradient": False,
        "full_axis_classes": int(table.class_ids.numel()),
        "train_rows": table.rows,
        "initialization_sha256": init_sha,
        "batch_trace_sha256": trace_sha,
        "target_census_path": str(target_census_path),
        "target_census_sha256": target_census_sha,
        "target_census": target_census,
        "action_target_histogram26": action_histogram,
        "loss_scales": {
            "swap": "unweighted_bce",
            "action": "unweighted_cross_entropy",
            "coefficients": {"swap": 1.0, "action": 1.0},
        },
        "conditions": {},
        "checkpoint_specs": {},
    }
    for name, result in conditions.items():
        checkpoint_path = output_dir / f"{name}_final.pt"
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "experiment_id": config.experiment_id,
                "condition": name,
                "condition_id": CONDITION_IDS[name],
                "model_class": "SemanticVisualRiskArbiter",
                "code_commit": code_commit,
                "config_sha256": config_sha256,
                "state_dict": _cpu_state_dict(result.pop("state_dict")),
                "target_census_sha256": target_census_sha,
                "target_census": target_census,
                "action_target_histogram26": action_histogram,
                "batch_trace_sha256": trace_sha,
                "initialization_sha256": init_sha,
                "full_axis_classes": int(table.class_ids.numel()),
            },
            checkpoint_path,
        )
        result["condition_id"] = CONDITION_IDS[name]
        result["checkpoint_path"] = str(checkpoint_path)
        result["checkpoint_sha256"] = sha256_file(checkpoint_path)
        receipt["conditions"][name] = result
        receipt["checkpoint_specs"][name] = {
            "path": str(checkpoint_path),
            "sha256": result["checkpoint_sha256"],
            "training_commit": str(code_commit),
            "train_config_sha256": str(config_sha256),
        }

    receipt_path = output_dir / "train_receipt.json"
    _write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = sha256_file(receipt_path)
    return receipt


def _train_condition(
    initial_state: Mapping[str, Tensor],
    table: JointTrainTable,
    config: DESCPrecheckConfig,
    trace: Sequence[Tensor],
    device: torch.device,
    *,
    include_action: bool,
    parent_only: bool,
) -> dict[str, Any]:
    model = build_model(table, seed=config.seed, device=device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    gradients: dict[str, Any] = {}
    last_loss: dict[str, float] = {}
    for step, rows in enumerate(trace, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss, digest = _loss_for_rows(
            model,
            table,
            rows,
            device,
            include_action=include_action,
            parent_only=parent_only,
        )
        loss.total.backward()
        _capture_grad_receipt(gradients, step, len(trace), model)
        optimizer.step()
        last_loss = {
            "total": _float(loss.total),
            "swap": _float(loss.swap),
            "action": _float(loss.action),
            "include_action": bool(loss.include_action),
        }
    final_digest = _condition_digest(model, table, trace[-1], device, parent_only=parent_only)
    final_digest["last_train_batch_digest"] = digest
    return {
        "state_dict": model.state_dict(),
        "objective": "L_swap+L_action" if include_action else "L_swap",
        "parent_only": parent_only,
        "updates": len(trace),
        "last_loss": last_loss,
        "gradients": gradients,
        "digests": final_digest,
    }


def _loss_for_rows(
    model: SemanticVisualRiskArbiter,
    table: JointTrainTable,
    rows: Tensor,
    device: torch.device,
    *,
    include_action: bool,
    parent_only: bool,
) -> tuple[DirectSVRALoss, dict[str, str]]:
    batch = table.batch(rows, device=device)
    output = model.direct_forward(
        batch["full_cls"],
        batch["patch_tokens"] if not parent_only else None,
        parent_only=parent_only,
    )
    targets = joint_action_targets_from_logits(
        output.pair,
        batch["all_crop_cls"],
        model.name_embeddings,
        batch["target_class_ids"],
        model.class_ids,
    )
    loss = direct_svra_loss(output, targets, include_action=include_action)
    digest = {
        "batch_sha256": tensor_sha256(batch["rows"]),
        "swap_logit_sha256": tensor_sha256(output.swap_logits.detach()),
        "action_logits_sha256": tensor_sha256(output.action_logits25.detach()),
        "evidence_pool_sha256": tensor_sha256(output.evidence_pool.detach()),
        "swap_sha256": tensor_sha256(output.swapped.detach()),
    }
    return loss, digest


def _condition_digest(
    model: SemanticVisualRiskArbiter,
    table: JointTrainTable,
    rows: Tensor,
    device: torch.device,
    *,
    parent_only: bool,
) -> dict[str, str]:
    with torch.no_grad():
        batch = table.batch(rows, device=device)
        output = model.direct_forward(
            batch["full_cls"],
            batch["patch_tokens"] if not parent_only else None,
            parent_only=parent_only,
        )
        return {
            "batch_sha256": tensor_sha256(batch["rows"]),
            "logit_sha256": tensor_sha256(output.logits),
            "parent_logit_sha256": tensor_sha256(output.parent_logits),
            "swap_logit_sha256": tensor_sha256(output.swap_logits),
            "action_logits_sha256": tensor_sha256(output.action_logits25),
            "evidence_pool_sha256": tensor_sha256(output.evidence_pool),
            "swap_sha256": tensor_sha256(output.swapped),
        }


def _build_desc_target_census(
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
    action_targets: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, table.rows, batch_size):
            rows = torch.arange(start, min(start + batch_size, table.rows))
            batch = table.batch(rows, device=device)
            output = model.direct_forward(batch["full_cls"], batch["patch_tokens"])
            targets = joint_action_targets_from_logits(
                output.pair,
                batch["all_crop_cls"],
                model.name_embeddings,
                batch["target_class_ids"],
                model.class_ids,
            )
            action_targets.append(targets.action_targets26.detach().cpu())
            counts["abstain"] += int((targets.action_targets26 == 0).sum().item())
            counts["action"] += int((targets.action_targets26 > 0).sum().item())
            counts["leader"] += int((targets.top2_group == 0).sum().item())
            counts["challenger"] += int((targets.top2_group == 1).sum().item())
            counts["outside"] += int((targets.top2_group == 2).sum().item())
            counts["conflict"] += int(targets.conflict_mask.sum().item())
    return counts, action_targets


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
        "I": _grad_norms(model.direct_interaction.parameters()),
    }


def _grad_norms(parameters: Any) -> dict[str, Any]:
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


def _assert_gradient_contracts(conditions: Mapping[str, Mapping[str, Any]]) -> None:
    for name in ("full", "no_action_aux"):
        gradients = conditions[name]["gradients"]
        for step in ("step2", f"step{conditions[name]['updates']}"):
            if step not in gradients:
                raise RuntimeError(f"{name}: missing gradient receipt {step}")
            for module in ("S", "V", "I"):
                stats = gradients[step][module]
                if not stats["finite"] or not stats["nonzero"]:
                    raise RuntimeError(f"{name}: {step} {module} gradient must be finite and nonzero")
    parent = conditions["parent_only"]
    for step in ("step2", f"step{parent['updates']}"):
        if step not in parent["gradients"]:
            raise RuntimeError(f"parent_only: missing gradient receipt {step}")
        stats = parent["gradients"][step]["I"]
        if not stats["finite"]:
            raise RuntimeError(f"parent_only: {step} I gradient must be finite")


def _assert_fixed_desc_config(config: DESCPrecheckConfig) -> None:
    invalid = (
        config.schema_version != "gzsl-paper.v6-desc-precheck-train-config.v1"
        or config.seed != 7
        or config.batch_size != 50
        or config.updates != 1000
        or float(config.lr) != 1e-3
        or float(config.weight_decay) != 0.0
        or not str(config.device).startswith("cuda")
        or config.expected_target_census != EXPECTED_TARGET_CENSUS
    )
    if invalid:
        raise RuntimeError("DESC formal precheck config violates the fixed IDEA-202 contract")


def run(config_path: str | Path, *, expected_commit: str, expected_config_sha: str) -> dict[str, Any]:
    config_path = Path(config_path)
    actual_config_sha = sha256_file(config_path)
    if actual_config_sha.lower() != expected_config_sha.lower():
        raise RuntimeError(
            f"DESC train config SHA mismatch: got {actual_config_sha}, expected {expected_config_sha}"
        )
    train_cfg, asset_cfg = load_config(config_path)
    _assert_fixed_desc_config(train_cfg)
    if train_cfg.require_clean_tree:
        require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise RuntimeError(f"DESC train commit mismatch: got {code_commit}, expected {expected_commit}")
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
        return train_desc_precheck(
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
            {"receipt_path": receipt["receipt_path"], "receipt_sha256": receipt["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
