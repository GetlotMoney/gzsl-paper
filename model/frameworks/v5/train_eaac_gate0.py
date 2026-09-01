"""Gate-0 Full-only trainer for IDEA-196 / EAAC.

The script is deliberately narrow:

* trains only the Full EAAC explicit-abstention action competition objective
  for the fixed Gate-0 budget;
* reads dev_train labels/class_ids/all25 crop features only from the subset
  manifest and validates each file against that manifest's output SHA;
* uses :mod:`rwdg_data` safe views for projected CLS/patch access;
* writes one non-overwriting checkpoint outside the repository;
* stores only cross-axis trainable model state, never class-axis asset buffers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from .rwdg import (
        ACTION_COUNT,
        ACTION_GEOMETRY_SHA256,
        FEATURE_DIM,
        PAIR_TEMPERATURE,
        RoleWindowDenseGlimpse,
    )
    from .rwdg_data import (
        ManifestContract,
        RWDGAssetConfig,
        RWDGDataError,
        RWDGGateSubsetView,
        TensorContract,
        load_rwdg_gate_data,
        resolve_subset_output,
        sha256_file,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from rwdg import (  # type: ignore[no-redef]
        ACTION_COUNT,
        ACTION_GEOMETRY_SHA256,
        FEATURE_DIM,
        PAIR_TEMPERATURE,
        RoleWindowDenseGlimpse,
    )
    from rwdg_data import (  # type: ignore[no-redef]
        ManifestContract,
        RWDGAssetConfig,
        RWDGDataError,
        RWDGGateSubsetView,
        TensorContract,
        load_rwdg_gate_data,
        resolve_subset_output,
        sha256_file,
    )
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_torch_save, atomic_write_json


TRAIN_SCHEMA = "gzsl-paper.v5-eaac-gate0-train.v1"
EAAC_CLASS_COUNT = ACTION_COUNT + 1


EAAC_PREREGISTERED_TARGET_HISTOGRAM = [
    4107,
    11,
    17,
    29,
    9,
    11,
    18,
    32,
    35,
    32,
    17,
    22,
    34,
    46,
    24,
    23,
    19,
    32,
    28,
    26,
    17,
    19,
    16,
    29,
    19,
    30,
]


EAAC_PREREGISTERED_TARGET_COUNTS = {
    "total_rows": 4702,
    "leader_rows": 3478,
    "challenger_rows": 634,
    "outside_rows": 590,
    "abstain_rows": 4107,
    "action_rows": 595,
    "uncorrectable_challenger_rows": 39,
}


STRICT_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "condition_id",
        "text_manifest",
        "text_manifest_sha256",
        "role_tensor",
        "role_tensor_sha256",
        "name_tensor",
        "name_tensor_sha256",
        "patch_manifest",
        "patch_manifest_sha256",
        "cls_tensor",
        "cls_tensor_sha256",
        "patch_tensor",
        "patch_tensor_sha256",
        "cuav_bundle_manifest",
        "cuav_bundle_manifest_sha256",
        "dev_train_manifest_sha256",
        "dev_eval_manifest_sha256",
        "dev_eval_oracle_manifest_sha256",
        "att_splits_mat_path",
        "trainval_count",
        "oracle_receipt",
        "oracle_receipt_sha256",
        "action_geometry_sha256",
        "output_dir",
        "device",
        "seed",
        "batch_size",
        "updates",
        "lr",
        "weight_decay",
        "strict_sha",
        "validate_tensor_values",
        "require_clean_tree",
        "allow_cpu",
        "official_test_loaded",
        "unseen_images_used_for_gradient",
        "pclr_online_inference",
    }
)


@dataclass(frozen=True)
class Gate0TrainConfig:
    schema_version: str
    experiment_id: str
    condition_id: str
    text_manifest: str
    text_manifest_sha256: str
    role_tensor: str
    role_tensor_sha256: str
    name_tensor: str
    name_tensor_sha256: str
    patch_manifest: str
    patch_manifest_sha256: str
    cls_tensor: str
    cls_tensor_sha256: str
    patch_tensor: str
    patch_tensor_sha256: str
    cuav_bundle_manifest: str
    cuav_bundle_manifest_sha256: str
    dev_train_manifest_sha256: str
    dev_eval_manifest_sha256: str
    dev_eval_oracle_manifest_sha256: str
    att_splits_mat_path: str
    trainval_count: int
    oracle_receipt: str
    oracle_receipt_sha256: str
    action_geometry_sha256: str
    output_dir: str
    device: str = "cuda"
    seed: int = 7
    batch_size: int = 8
    updates: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    strict_sha: bool = True
    validate_tensor_values: bool = True
    require_clean_tree: bool = True
    allow_cpu: bool = False
    official_test_loaded: bool = False
    unseen_images_used_for_gradient: bool = False
    pclr_online_inference: bool = False


@dataclass(frozen=True)
class TrainSubsetTargets:
    labels: Tensor
    class_ids: Tensor
    crop_features: Any
    labels_path: str
    class_ids_path: str
    crop_features_path: str
    labels_sha256: str
    class_ids_sha256: str
    crop_features_sha256: str


@dataclass(frozen=True)
class EAACTargetPlan:
    targets26: Tensor
    groups: Tensor
    stats: Mapping[str, Any]


@dataclass(frozen=True)
class GradientGateReport:
    finite: bool
    nonzero: bool
    grad_abs_sum: float
    grad_max_abs: float


def load_strict_config(path: str | os.PathLike[str]) -> tuple[Gate0TrainConfig, str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise RWDGDataError(f"config must be a JSON object: {p}")
    got = set(str(k) for k in raw.keys())
    missing = STRICT_CONFIG_FIELDS - got
    extra = got - STRICT_CONFIG_FIELDS
    if missing or extra:
        raise RWDGDataError(
            f"config fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    config_sha = sha256_file(p)
    return Gate0TrainConfig(**raw), config_sha


def validate_config(config: Gate0TrainConfig) -> None:
    if config.schema_version != TRAIN_SCHEMA:
        raise RWDGDataError(f"schema_version must be {TRAIN_SCHEMA}")
    if config.condition_id != "EAAC_FULL":
        raise RWDGDataError("Gate0 training condition_id must be EAAC_FULL")
    if not str(config.experiment_id):
        raise RWDGDataError("experiment_id must be non-empty")
    if config.seed != 7:
        raise RWDGDataError("Gate0 seed is fixed at 7")
    if config.batch_size != 8:
        raise RWDGDataError("Gate0 batch_size is fixed at 8")
    if config.updates != 1000:
        raise RWDGDataError("Gate0 updates is fixed at 1000")
    if float(config.lr) != 1e-3:
        raise RWDGDataError("Gate0 lr is fixed at 1e-3")
    if float(config.weight_decay) != 1e-4:
        raise RWDGDataError("Gate0 weight_decay is fixed at 1e-4")
    if not config.strict_sha:
        raise RWDGDataError("Gate0 formal training requires strict_sha=true")
    if not config.validate_tensor_values:
        raise RWDGDataError("Gate0 formal training requires validate_tensor_values=true")
    if not config.require_clean_tree:
        raise RWDGDataError("Gate0 formal training requires require_clean_tree=true")
    if config.allow_cpu:
        raise RWDGDataError("Gate0 formal training requires allow_cpu=false")
    if config.official_test_loaded is not False:
        raise RWDGDataError("official_test_loaded must be false for training")
    if config.unseen_images_used_for_gradient is not False:
        raise RWDGDataError("unseen_images_used_for_gradient must be false")
    if config.pclr_online_inference is not False:
        raise RWDGDataError("pclr_online_inference must be false")
    if config.action_geometry_sha256 != ACTION_GEOMETRY_SHA256:
        raise RWDGDataError("action_geometry_sha256 mismatch")
    if int(config.trainval_count) != 7057:
        raise RWDGDataError("trainval_count must be 7057")


def asset_config_from_train_config(config: Gate0TrainConfig) -> RWDGAssetConfig:
    count = int(config.trainval_count)
    return RWDGAssetConfig(
        text_manifest=ManifestContract(
            path=str(config.text_manifest),
            sha256=str(config.text_manifest_sha256),
        ),
        role_tensor=TensorContract(
            path=str(config.role_tensor),
            sha256=str(config.role_tensor_sha256),
            shape=(200, 8, FEATURE_DIM),
            dtype="float32",
        ),
        name_tensor=TensorContract(
            path=str(config.name_tensor),
            sha256=str(config.name_tensor_sha256),
            shape=(200, FEATURE_DIM),
            dtype="float32",
        ),
        patch_manifest=ManifestContract(
            path=str(config.patch_manifest),
            sha256=str(config.patch_manifest_sha256),
        ),
        cls_tensor=TensorContract(
            path=str(config.cls_tensor),
            sha256=str(config.cls_tensor_sha256),
            shape=(count, FEATURE_DIM),
            dtype="float32",
        ),
        patch_tensor=TensorContract(
            path=str(config.patch_tensor),
            sha256=str(config.patch_tensor_sha256),
            shape=(count, 576, FEATURE_DIM),
            dtype="float16",
        ),
        cuav_bundle_manifest=ManifestContract(
            path=str(config.cuav_bundle_manifest),
            sha256=str(config.cuav_bundle_manifest_sha256),
        ),
        dev_train_manifest_sha256=str(config.dev_train_manifest_sha256),
        dev_eval_manifest_sha256=str(config.dev_eval_manifest_sha256),
        dev_eval_oracle_manifest_sha256=str(config.dev_eval_oracle_manifest_sha256),
        att_splits_mat_path=str(config.att_splits_mat_path),
        trainval_count=count,
    )


def load_and_validate_oracle_receipt(config: Gate0TrainConfig) -> Mapping[str, Any]:
    path = Path(config.oracle_receipt)
    if not path.is_file():
        raise RWDGDataError(f"oracle_receipt missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha.lower() != config.oracle_receipt_sha256.lower():
        raise RWDGDataError(
            f"oracle_receipt SHA mismatch: got {actual_sha}, expected {config.oracle_receipt_sha256}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RWDGDataError("oracle_receipt must be a JSON object")
    if value.get("schema_version") != "gzsl-paper.v5-rwdg-projected-pair-oracle.v1":
        raise RWDGDataError("oracle_receipt schema_version mismatch")
    if not isinstance(value.get("gates"), Mapping):
        raise RWDGDataError("oracle_receipt missing gates object")
    if value.get("used_for_training") is not False:
        raise RWDGDataError("oracle_receipt must be diagnostic-only, not used_for_training")
    if (
        value.get("official_test_loaded") is not False
        or value.get("unseen_images_used_for_gradient") is not False
        or value.get("pclr_online_inference") is not False
        or value.get("oracle_all25_opened") is not True
    ):
        raise RWDGDataError("oracle_receipt protocol flags mismatch")
    if int(value.get("rows", -1)) != 2355 or int(value.get("active_classes", -1)) != 150:
        raise RWDGDataError("oracle_receipt row/axis mismatch")
    oracle_gate = oracle_gate_from_receipt(value, min_gain=1.0)
    if not oracle_gate["passed"]:
        raise RWDGDataError(f"oracle_receipt gate failed: {oracle_gate}")
    receipt_identity = oracle_identity_from_receipt(value)
    compare_oracle_identity(receipt_identity, config)
    return {"receipt": value, "gate": oracle_gate, "identity": receipt_identity}


def oracle_gate_from_receipt(
    receipt: Mapping[str, Any],
    *,
    min_gain: float,
) -> Mapping[str, Any]:
    gates = receipt.get("gates")
    if isinstance(gates, Mapping):
        failed = {str(k): v for k, v in gates.items() if v is not True}
        if failed:
            raise RWDGDataError(f"oracle_receipt gates contain failures: {failed}")
    required = (
        "parent_macro_top1_percent",
        "pair_crop_oracle25_macro_top1_percent",
        "oracle_gain_pp",
    )
    if any(not isinstance(receipt.get(key), (int, float)) for key in required):
        raise RWDGDataError("oracle_receipt missing exact Parent/Oracle/gain fields")
    parent = float(receipt["parent_macro_top1_percent"])
    oracle = float(receipt["pair_crop_oracle25_macro_top1_percent"])
    gain = float(receipt["oracle_gain_pp"])
    if abs((oracle - parent) - gain) > 1e-9:
        raise RWDGDataError("oracle_receipt gain is inconsistent with Parent/Oracle")
    return {
        "parent_macro_top1": parent,
        "oracle_macro_top1": oracle,
        "oracle_gain_pp": float(gain),
        "passed": float(gain) >= float(min_gain),
    }


def oracle_identity_from_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = receipt.get("asset_identity")
    if not isinstance(identity, Mapping):
        raise RWDGDataError("oracle_receipt missing asset_identity")

    def object_sha(name: str) -> str | None:
        value = identity.get(name)
        return str(value.get("sha256")) if isinstance(value, Mapping) and value.get("sha256") else None

    return {
        "text_manifest_sha256": object_sha("text_manifest"),
        "patch_manifest_sha256": object_sha("patch_manifest"),
        "cuav_bundle_manifest_sha256": object_sha("bundle_manifest"),
        "dev_eval_manifest_sha256": object_sha("eval_manifest"),
        "dev_eval_oracle_manifest_sha256": object_sha("oracle_manifest"),
        "action_geometry_sha256": identity.get("action_geometry_sha256"),
    }


def compare_oracle_identity(
    receipt_identity: Mapping[str, Any],
    config: Gate0TrainConfig,
) -> None:
    expected = {
        "text_manifest_sha256": config.text_manifest_sha256,
        "patch_manifest_sha256": config.patch_manifest_sha256,
        "cuav_bundle_manifest_sha256": config.cuav_bundle_manifest_sha256,
        "dev_eval_manifest_sha256": config.dev_eval_manifest_sha256,
        "dev_eval_oracle_manifest_sha256": config.dev_eval_oracle_manifest_sha256,
        "action_geometry_sha256": config.action_geometry_sha256,
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual = receipt_identity.get(key)
        if actual is None or str(actual).lower() != str(expected_value).lower():
            mismatches[key] = {"actual": actual, "expected": expected_value}
    if mismatches:
        raise RWDGDataError(f"oracle_receipt identity mismatch: {mismatches}")


def _nested_number(value: Any, keys: Sequence[str]) -> float | None:
    wanted = {key.lower() for key in keys}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in wanted and isinstance(item, (int, float)):
                return float(item)
            found = _nested_number(item, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _nested_number(item, keys)
            if found is not None:
                return found
    return None


def _nested_string(value: Any, key_name: str) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() == key_name.lower() and isinstance(item, str):
                return item
            found = _nested_string(item, key_name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _nested_string(item, key_name)
            if found is not None:
                return found
    return None


def git_commit_and_clean(repo_root: Path) -> tuple[str, bool, str]:
    commit = _git(repo_root, "rev-parse", "HEAD").strip()
    status = _git(repo_root, "status", "--porcelain").strip()
    return commit, status == "", status


def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def prepare_output_dir(output_dir: str | os.PathLike[str], repo_root: Path) -> Path:
    out = Path(output_dir).expanduser().resolve()
    repo = repo_root.resolve()
    try:
        out.relative_to(repo)
        inside_repo = True
    except ValueError:
        inside_repo = False
    if inside_repo:
        raise RWDGDataError(f"output_dir must be outside the repository: {out}")
    if out.exists():
        raise RWDGDataError(f"output_dir already exists; refusing overwrite: {out}")
    out.mkdir(parents=True)
    return out


def set_reproducibility(seed: int) -> torch.Generator:
    configure_reproducibility(
        seed,
        strict_determinism=True,
        deterministic_warn_only=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def resolve_device(config: Gate0TrainConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RWDGDataError("requested CUDA device, but torch.cuda.is_available() is false")
    if device.type == "cpu" and not config.allow_cpu:
        raise RWDGDataError("CPU training is disabled unless allow_cpu=true")
    return device


def load_dev_train_targets(
    assets: Any,
    *,
    strict_sha: bool,
) -> TrainSubsetTargets:
    labels_path = _resolve_first_subset_output(
        assets, "dev_train", ("labels.pt", "labels.npy", "targets.pt", "targets.npy")
    )
    class_ids_path = _resolve_first_subset_output(
        assets,
        "dev_train",
        (
            "class_ids.pt",
            "class_ids.npy",
            "classes.pt",
            "classes.npy",
            "train_class_ids.pt",
            "train_class_ids.npy",
            "seen_class_ids.pt",
            "seen_class_ids.npy",
        ),
    )
    crop_path = _resolve_first_subset_output(
        assets,
        "dev_train",
        (
            "crop_features.pt",
            "crop_features.npy",
            "all25_crop_features.pt",
            "all25_crop_features.npy",
        ),
    )
    labels_sha = sha256_file(labels_path)
    class_ids_sha = sha256_file(class_ids_path)
    crop_sha = sha256_file(crop_path)

    labels = _load_long_vector(labels_path, name="labels")
    class_ids = _load_long_vector(class_ids_path, name="class_ids")
    crop_features = _load_crop_feature_table(crop_path)
    return TrainSubsetTargets(
        labels=labels,
        class_ids=class_ids,
        crop_features=crop_features,
        labels_path=str(labels_path),
        class_ids_path=str(class_ids_path),
        crop_features_path=str(crop_path),
        labels_sha256=labels_sha,
        class_ids_sha256=class_ids_sha,
        crop_features_sha256=crop_sha,
    )


def _resolve_first_subset_output(
    assets: Any,
    subset_name: str,
    candidate_basenames: Sequence[str],
) -> Path:
    errors = []
    for filename in candidate_basenames:
        try:
            return resolve_subset_output(
                assets,
                subset_name,
                filename,
                verify_sha=True,
            )
        except RWDGDataError as exc:
            errors.append(f"{filename}: {exc}")
    raise RWDGDataError(
        f"{subset_name}: none of required outputs resolved: "
        + "; ".join(errors)
    )


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        return torch.load(str(path), map_location="cpu")


def _first_tensor(value: Any, *, source: Path) -> Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            if torch.is_tensor(item):
                return item
    raise RWDGDataError(f"no tensor found in {source}")


def _load_long_vector(path: Path, *, name: str) -> Tensor:
    if path.suffix.lower() == ".pt":
        value = _first_tensor(_torch_load_cpu(path), source=path)
        tensor = value.detach().cpu().long().reshape(-1)
    elif path.suffix.lower() == ".npy":
        tensor = torch.as_tensor(np.load(path), dtype=torch.long).reshape(-1)
    else:
        raise RWDGDataError(f"{name}: unsupported file type {path}")
    if tensor.numel() == 0:
        raise RWDGDataError(f"{name}: empty vector")
    if int(tensor.min()) < 0:
        raise RWDGDataError(f"{name}: negative class id detected")
    return tensor


def _load_crop_feature_table(path: Path) -> Any:
    if path.suffix.lower() == ".pt":
        value = _first_tensor(_torch_load_cpu(path), source=path).detach().cpu()
        _validate_crop_shape(value, path)
        return value
    if path.suffix.lower() == ".npy":
        value = np.load(path, mmap_mode="r")
        _validate_crop_shape(value, path)
        return value
    raise RWDGDataError(f"crop_features: unsupported file type {path}")


def _validate_crop_shape(value: Any, path: Path) -> None:
    shape = tuple(int(x) for x in value.shape)
    if len(shape) != 3 or shape[1:] != (ACTION_COUNT, FEATURE_DIM):
        raise RWDGDataError(f"{path}: crop feature shape must be [N,25,768], got {shape}")


def validate_training_axis(
    targets: TrainSubsetTargets,
    train_view: RWDGGateSubsetView,
    full_class_count: int,
) -> None:
    if targets.labels.shape != (train_view.size,):
        raise RWDGDataError(
            f"labels count {targets.labels.numel()} != dev_train rows {train_view.size}"
        )
    if tuple(int(x) for x in targets.crop_features.shape[:1]) != (train_view.size,):
        raise RWDGDataError(
            f"crop feature rows {targets.crop_features.shape[0]} != dev_train rows {train_view.size}"
        )
    if torch.unique(targets.class_ids).numel() != targets.class_ids.numel():
        raise RWDGDataError("class_ids contains duplicates")
    if targets.class_ids.numel() != 100:
        raise RWDGDataError(f"Gate0 train axis must contain 100 classes, got {targets.class_ids.numel()}")
    if int(targets.class_ids.max()) >= full_class_count:
        raise RWDGDataError(
            f"class_ids max {int(targets.class_ids.max())} outside asset axis {full_class_count}"
        )
    label_set = set(int(x) for x in torch.unique(targets.labels).tolist())
    class_set = set(int(x) for x in targets.class_ids.tolist())
    if not label_set.issubset(class_set):
        missing = sorted(label_set - class_set)[:10]
        raise RWDGDataError(f"labels contain classes outside class_ids, examples={missing}")


def build_train100_model(
    role_embeddings: Tensor,
    name_embeddings: Tensor,
    class_ids: Tensor,
    *,
    device: torch.device,
    seed: int,
) -> RoleWindowDenseGlimpse:
    role_slice = role_embeddings.index_select(0, class_ids.cpu()).float()
    name_slice = name_embeddings.index_select(0, class_ids.cpu()).float()
    return RoleWindowDenseGlimpse(
        role_embeddings=role_slice,
        name_embeddings=name_slice,
        class_ids=class_ids,
        seed=seed,
    ).to(device)


def batch_from_rows(
    train_view: RWDGGateSubsetView,
    targets: TrainSubsetTargets,
    rows: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    row_np = rows.detach().cpu().numpy().astype(np.int64, copy=False)
    batch = train_view.batch(row_np, include_patches=True, as_torch=True, device=device)
    labels = targets.labels.index_select(0, rows.cpu()).to(device=device)
    crops = _take_crop_rows(targets.crop_features, row_np).to(device=device, dtype=torch.float32)
    return batch["cls"], batch["patches"], crops, labels


def _take_crop_rows(crop_features: Any, rows: np.ndarray) -> Tensor:
    if torch.is_tensor(crop_features):
        return crop_features.index_select(0, torch.as_tensor(rows, dtype=torch.long)).float()
    return torch.as_tensor(np.asarray(crop_features[rows]), dtype=torch.float32)


def crop_leader_minus_challenger_margins(
    model: RoleWindowDenseGlimpse,
    all_crop_cls: Tensor,
    pair: Any,
) -> Tensor:
    if all_crop_cls.ndim != 3 or all_crop_cls.shape[1:] != (ACTION_COUNT, FEATURE_DIM):
        raise ValueError(
            "all_crop_cls must have shape [B,25,768], "
            f"got {tuple(all_crop_cls.shape)}"
        )
    names = model.name_embeddings.to(device=all_crop_cls.device)
    crop_logits = torch.einsum(
        "bad,cd->bac",
        F.normalize(all_crop_cls.float(), dim=-1),
        names,
    ) / PAIR_TEMPERATURE
    rows = torch.arange(all_crop_cls.shape[0], device=all_crop_cls.device)
    actions = torch.arange(ACTION_COUNT, device=all_crop_cls.device)
    leader = pair.top2[:, 0]
    challenger = pair.top2[:, 1]
    leader_score = crop_logits[rows[:, None], actions, leader[:, None]]
    challenger_score = crop_logits[rows[:, None], actions, challenger[:, None]]
    return leader_score - challenger_score


def eaac_action_targets(
    model: RoleWindowDenseGlimpse,
    full_cls: Tensor,
    all_crop_cls: Tensor,
    target_class_ids: Tensor,
    *,
    semantic_off: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Any, Tensor]:
    dense_targets, groups, pair = model.dense_utility_targets(
        full_cls,
        all_crop_cls,
        target_class_ids,
        semantic_off=semantic_off,
    )
    margins = crop_leader_minus_challenger_margins(model, all_crop_cls, pair)
    targets26 = torch.zeros(full_cls.shape[0], dtype=torch.long, device=full_cls.device)
    challenger_rows = groups == 1
    correctable = challenger_rows & dense_targets.bool().any(dim=1)
    if bool(correctable.any()):
        candidate_margins = margins[correctable].masked_fill(~dense_targets[correctable].bool(), float("inf"))
        targets26[correctable] = torch.argmin(candidate_margins, dim=1).long() + 1
    return targets26, groups, margins, pair, dense_targets


def eaac_strongest_action_targets(
    model: RoleWindowDenseGlimpse,
    full_cls: Tensor,
    all_crop_cls: Tensor,
    target_class_ids: Tensor,
) -> tuple[Tensor, Tensor, Mapping[str, Any]]:
    targets26, groups, margins, _, dense_targets = eaac_action_targets(
        model, full_cls, all_crop_cls, target_class_ids
    )
    return targets26, groups, {
        "target_policy": "strongest_margin_action_with_explicit_abstain",
        "margins": margins,
        "correcting_mask": dense_targets.bool(),
    }


# Public contract name used by tests and downstream controls.
eaac_targets = eaac_action_targets


def eaac_policy_logits(utility_logits: Tensor) -> Tensor:
    if utility_logits.ndim != 2 or utility_logits.shape[1] != ACTION_COUNT:
        raise ValueError(f"EAAC utility logits must have shape [B, {ACTION_COUNT}]")
    abstain = utility_logits.new_zeros((utility_logits.shape[0], 1))
    return torch.cat([abstain, utility_logits], dim=1)


def eaac_action_loss(utility_logits: Tensor, targets26: Tensor) -> Tensor:
    if targets26.shape != (utility_logits.shape[0],):
        raise ValueError(
            f"EAAC logits/target shape mismatch: logits={tuple(utility_logits.shape)} "
            f"targets26={tuple(targets26.shape)}"
        )
    if targets26.dtype != torch.long:
        targets26 = targets26.long()
    if bool((targets26 < 0).any()) or bool((targets26 >= EAAC_CLASS_COUNT).any()):
        raise ValueError(f"EAAC targets must be class ids in [0,{EAAC_CLASS_COUNT - 1}]")
    return F.cross_entropy(eaac_policy_logits(utility_logits), targets26)


eaac_policy_loss = eaac_action_loss


def _int_tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _class_histogram(targets26: Tensor) -> list[int]:
    return [int(x) for x in torch.bincount(targets26.detach().cpu().long(), minlength=EAAC_CLASS_COUNT)[:EAAC_CLASS_COUNT].tolist()]


def _group_target_counts(targets26: Tensor, groups: Tensor) -> dict[str, dict[str, int]]:
    names = ("leader", "challenger", "outside")
    out: dict[str, dict[str, int]] = {}
    for group_id, name in enumerate(names):
        mask = groups == group_id
        group_targets = targets26[mask]
        out[name] = {
            "rows": int(mask.sum().item()),
            "abstain_rows": int(group_targets.eq(0).sum().item()),
            "action_rows": int(group_targets.gt(0).sum().item()),
        }
    return out


def _challenger_target_diagnostics(
    dense_targets: Tensor,
    margins: Tensor,
    groups: Tensor,
) -> dict[str, Any]:
    challenger_mask = groups.eq(1)
    counts = dense_targets[challenger_mask].long().sum(dim=1).detach().cpu().long()
    choice_hist = torch.zeros(ACTION_COUNT, dtype=torch.long)
    gaps: list[float | None] = []
    for row_targets, row_margins in zip(
        dense_targets[challenger_mask].detach().cpu().bool(),
        margins[challenger_mask].detach().cpu().double(),
        strict=True,
    ):
        correcting = torch.nonzero(row_targets, as_tuple=False).flatten()
        if correcting.numel() == 0:
            gaps.append(None)
            continue
        correcting_margins = row_margins.index_select(0, correcting)
        order = torch.argsort(correcting_margins, stable=True)
        best_action = int(correcting.index_select(0, order[:1]).item())
        choice_hist[best_action] += 1
        if correcting.numel() >= 2:
            best = float(correcting_margins.index_select(0, order[:1]).item())
            second = float(correcting_margins.index_select(0, order[1:2]).item())
            gaps.append(float(second - best))
        else:
            gaps.append(None)
    finite_gaps = [float(x) for x in gaps if x is not None]
    return {
        "correcting_action_counts": [int(x) for x in counts.tolist()],
        "best_second_margin_gap": gaps,
        "best_second_margin_gap_summary": {
            "count": len(finite_gaps),
            "min": min(finite_gaps) if finite_gaps else None,
            "mean": sum(finite_gaps) / len(finite_gaps) if finite_gaps else None,
            "max": max(finite_gaps) if finite_gaps else None,
        },
        "chosen_action_histogram": [int(x) for x in choice_hist.tolist()],
    }


def precompute_eaac_target_plan(
    model: RoleWindowDenseGlimpse,
    train_view: RWDGGateSubsetView,
    targets: TrainSubsetTargets,
    *,
    batch_size: int,
    device: torch.device,
) -> EAACTargetPlan:
    model.eval()
    all_groups: list[Tensor] = []
    all_targets26: list[Tensor] = []
    all_dense_targets: list[Tensor] = []
    all_margins: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, train_view.size, batch_size):
            rows = torch.arange(start, min(start + batch_size, train_view.size), dtype=torch.long)
            full_cls, _, crops, labels = batch_from_rows(train_view, targets, rows, device=device)
            targets26, groups, margins, _, dense_targets = eaac_action_targets(
                model,
                full_cls,
                crops,
                labels,
            )
            targets26_cpu = targets26.detach().cpu().long()
            groups_cpu = groups.detach().cpu().long()
            if bool((targets26_cpu < 0).any()) or bool((targets26_cpu >= EAAC_CLASS_COUNT).any()):
                raise RWDGDataError(f"EAAC targets must be class ids in [0,{EAAC_CLASS_COUNT - 1}]")
            all_groups.append(groups_cpu)
            all_targets26.append(targets26_cpu)
            all_dense_targets.append(dense_targets.detach().cpu().float())
            all_margins.append(margins.detach().cpu().float())

    groups = torch.cat(all_groups).long()
    targets26 = torch.cat(all_targets26).long()
    dense_targets = torch.cat(all_dense_targets).float()
    margins = torch.cat(all_margins).float()
    group_counts = torch.bincount(groups, minlength=3)[:3].long()
    histogram = _class_histogram(targets26)
    group_target_counts = _group_target_counts(targets26, groups)
    challenger_diagnostics = _challenger_target_diagnostics(dense_targets, margins, groups)
    total_rows = int(groups.numel())
    challenger_rows = int(group_counts[1].item())
    stats = {
        "target_definition": "eaac_explicit_abstention_action_competition",
        "natural_distribution": {
            "total_rows": total_rows,
            "group_counts": {
                "leader": int(group_counts[0].item()),
                "challenger": challenger_rows,
                "outside": int(group_counts[2].item()),
            },
            "group_ids": [int(x) for x in groups.tolist()],
            "group_ids_sha256": _int_tensor_sha256(groups),
        },
        "class_histogram": histogram,
        "abstain_rows": int(targets26.eq(0).sum().item()),
        "action_rows": int(targets26.gt(0).sum().item()),
        "uncorrectable_challenger_rows": int((groups.eq(1) & targets26.eq(0)).sum().item()),
        "group_target_counts": group_target_counts,
        "action_label_density": float(targets26.gt(0).sum().item() / max(total_rows, 1)),
        "challenger_action_density": float(group_target_counts["challenger"]["action_rows"] / max(challenger_rows, 1)),
        "challenger_diagnostics": challenger_diagnostics,
    }
    assert_pre_registered_eaac_counts(stats)
    return EAACTargetPlan(targets26=targets26, groups=groups, stats=stats)


def assert_pre_registered_eaac_counts(stats: Mapping[str, Any]) -> None:
    natural = stats.get("natural_distribution")
    if not isinstance(natural, Mapping):
        raise RWDGDataError("EAAC target stats missing natural_distribution")
    groups = natural.get("group_counts")
    if not isinstance(groups, Mapping):
        raise RWDGDataError("EAAC target stats missing group_counts")
    histogram = stats.get("class_histogram")
    group_targets = stats.get("group_target_counts")
    challenger_diagnostics = stats.get("challenger_diagnostics")
    if not isinstance(histogram, Sequence) or isinstance(histogram, (str, bytes)):
        raise RWDGDataError("EAAC target stats missing class_histogram")
    if [int(x) for x in histogram] != EAAC_PREREGISTERED_TARGET_HISTOGRAM:
        raise RWDGDataError(
            "EAAC natural target histogram mismatch: "
            f"actual={[int(x) for x in histogram]}, expected={EAAC_PREREGISTERED_TARGET_HISTOGRAM}"
        )
    if not isinstance(group_targets, Mapping):
        raise RWDGDataError("EAAC target stats missing group_target_counts")
    if not isinstance(challenger_diagnostics, Mapping):
        raise RWDGDataError("EAAC target stats missing challenger_diagnostics")

    actual = {
        "total_rows": int(natural.get("total_rows", -1)),
        "leader_rows": int(groups.get("leader", -1)),
        "challenger_rows": int(groups.get("challenger", -1)),
        "outside_rows": int(groups.get("outside", -1)),
        "abstain_rows": int(stats.get("abstain_rows", -1)),
        "action_rows": int(stats.get("action_rows", -1)),
        "uncorrectable_challenger_rows": int(stats.get("uncorrectable_challenger_rows", -1)),
    }
    mismatches = {
        key: {"actual": actual[key], "expected": expected}
        for key, expected in EAAC_PREREGISTERED_TARGET_COUNTS.items()
        if actual[key] != expected
    }
    leader_counts = group_targets.get("leader", {})
    outside_counts = group_targets.get("outside", {})
    if int(leader_counts.get("action_rows", -1)) != 0 or int(outside_counts.get("action_rows", -1)) != 0:
        mismatches["action_rows_not_challenger"] = {
            "actual": {
                "leader": int(leader_counts.get("action_rows", -1)),
                "outside": int(outside_counts.get("action_rows", -1)),
            },
            "expected": {"leader": 0, "outside": 0},
        }
    correcting_counts = challenger_diagnostics.get("correcting_action_counts")
    if not isinstance(correcting_counts, Sequence) or isinstance(correcting_counts, (str, bytes)):
        mismatches["challenger_correcting_action_counts"] = "missing"
    elif len(correcting_counts) != EAAC_PREREGISTERED_TARGET_COUNTS["challenger_rows"]:
        mismatches["challenger_correcting_action_counts"] = {
            "actual": len(correcting_counts),
            "expected": EAAC_PREREGISTERED_TARGET_COUNTS["challenger_rows"],
        }
    if mismatches:
        raise RWDGDataError(f"EAAC preregistered target counts mismatch: {mismatches}")


class BalancedGroupSampler:
    """Seeded 4:4 challenger/non-challenger sampler with per-group reshuffle cycles."""

    def __init__(
        self,
        groups: Tensor,
        *,
        batch_size: int,
        challenger_per_batch: int = 4,
        seed: int = 7,
    ) -> None:
        if batch_size != 8 or challenger_per_batch != 4:
            raise RWDGDataError("EAAC sampler requires batch8 with 4 challenger rows")
        groups = groups.detach().cpu().long().reshape(-1)
        challenger = torch.nonzero(groups == 1, as_tuple=False).flatten().long()
        non_challenger = torch.nonzero(groups != 1, as_tuple=False).flatten().long()
        if challenger.numel() < challenger_per_batch:
            raise RWDGDataError("EAAC sampler has too few challenger rows")
        if non_challenger.numel() < batch_size - challenger_per_batch:
            raise RWDGDataError("EAAC sampler has too few non-challenger rows")
        self.batch_size = int(batch_size)
        self.challenger_per_batch = int(challenger_per_batch)
        self.non_challenger_per_batch = self.batch_size - self.challenger_per_batch
        self.seed = int(seed)
        self.challenger_indices = challenger
        self.non_challenger_indices = non_challenger
        self._challenger_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self._non_challenger_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self._batch_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self._challenger_buffer = torch.empty(0, dtype=torch.long)
        self._non_challenger_buffer = torch.empty(0, dtype=torch.long)

    def sample(self) -> Tensor:
        challenger_rows = self._take("challenger", self.challenger_per_batch)
        non_challenger_rows = self._take("non_challenger", self.non_challenger_per_batch)
        batch = torch.cat([challenger_rows, non_challenger_rows])
        order = torch.randperm(batch.numel(), generator=self._batch_gen)
        return batch.index_select(0, order).long()

    def next_batch(self) -> Tensor:
        return self.sample()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "name": "BalancedGroupSampler",
            "seed": self.seed,
            "batch_size": self.batch_size,
            "challenger_per_batch": self.challenger_per_batch,
            "non_challenger_per_batch": self.non_challenger_per_batch,
            "challenger_rows": int(self.challenger_indices.numel()),
            "non_challenger_rows": int(self.non_challenger_indices.numel()),
            "reshuffle_policy": "independent per-group randperm cycles; reshuffle when exhausted; shuffle within batch",
        }

    def _take(self, group_name: str, count: int) -> Tensor:
        if group_name == "challenger":
            pool = self.challenger_indices
            gen = self._challenger_gen
            buffer = self._challenger_buffer
        elif group_name == "non_challenger":
            pool = self.non_challenger_indices
            gen = self._non_challenger_gen
            buffer = self._non_challenger_buffer
        else:  # pragma: no cover - internal invariant.
            raise RWDGDataError(f"unknown sampler group: {group_name}")

        chunks: list[Tensor] = []
        remaining = int(count)
        while remaining > 0:
            if buffer.numel() == 0:
                buffer = pool.index_select(0, torch.randperm(pool.numel(), generator=gen))
            take = min(remaining, int(buffer.numel()))
            chunks.append(buffer[:take])
            buffer = buffer[take:]
            remaining -= take

        if group_name == "challenger":
            self._challenger_buffer = buffer
        else:
            self._non_challenger_buffer = buffer
        return torch.cat(chunks).long()


def sampled_class_histogram(
    targets26: Tensor,
    groups: Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    challenger_per_batch: int = 4,
) -> Mapping[str, Any]:
    sampler = BalancedGroupSampler(
        groups,
        batch_size=batch_size,
        challenger_per_batch=challenger_per_batch,
        seed=seed,
    )
    histogram = torch.zeros(EAAC_CLASS_COUNT, dtype=torch.long)
    for _ in range(int(updates)):
        rows = sampler.sample()
        batch_targets = targets26.index_select(0, rows).long()
        histogram += torch.bincount(batch_targets.cpu(), minlength=EAAC_CLASS_COUNT)[:EAAC_CLASS_COUNT].long()
    if int(histogram[0].item()) <= 0 or int(histogram[1:].sum().item()) <= 0:
        raise RWDGDataError(f"EAAC sampled histogram must include both abstain/action labels: {histogram.tolist()}")
    total_rows = max(int(updates) * int(batch_size), 1)
    return {
        "updates": int(updates),
        "batch_size": int(batch_size),
        "sampled_rows": total_rows,
        "class_histogram": [int(x) for x in histogram.tolist()],
        "abstain_rows": int(histogram[0].item()),
        "action_rows": int(histogram[1:].sum().item()),
        "class_density": [float(x / total_rows) for x in histogram.tolist()],
        "sampler": sampler.state_dict(),
    }


def train_step(
    model: RoleWindowDenseGlimpse,
    optimizer: torch.optim.Optimizer,
    train_view: RWDGGateSubsetView,
    targets: TrainSubsetTargets,
    rows: Tensor,
    *,
    device: torch.device,
    apply_update: bool = True,
) -> tuple[float, Mapping[str, GradientGateReport]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    full_cls, patches, crops, labels = batch_from_rows(train_view, targets, rows, device=device)
    target26, _, _, _, _ = eaac_action_targets(model, full_cls, crops, labels)
    utility = model.utility_state(full_cls, patches)
    loss = eaac_action_loss(utility.utility_logits, target26)
    loss.backward()
    report = collect_gradient_report(model)
    if apply_update:
        optimizer.step()
    return float(loss.detach().cpu()), report


def collect_gradient_report(model: RoleWindowDenseGlimpse) -> Mapping[str, GradientGateReport]:
    named = {
        "W_r": model.semantic.role_projection.weight,
        "W_n": model.semantic.name_projection.weight,
        "W_x": model.visual.window_key.weight,
        "W_vx": model.visual.window_value.weight,
        "W_vr": model.visual.role_value.weight,
        "W_h": model.visual.utility_hidden.weight,
        "w_u": model.visual.utility_output.weight,
    }
    return {name: gradient_gate_report(param.grad) for name, param in named.items()}


def gradient_gate_report(grad: Tensor | None) -> GradientGateReport:
    if grad is None:
        return GradientGateReport(False, False, 0.0, 0.0)
    finite = bool(torch.isfinite(grad).all().detach().cpu())
    abs_grad = grad.detach().abs()
    abs_sum = float(abs_grad.sum().cpu())
    max_abs = float(abs_grad.max().cpu()) if abs_grad.numel() else 0.0
    return GradientGateReport(finite, abs_sum > 0.0, abs_sum, max_abs)


def assert_step1_gradient_gate(report: Mapping[str, GradientGateReport]) -> None:
    wu = report["w_u"]
    if not (wu.finite and wu.nonzero):
        raise RWDGDataError(f"step1 w_u gradient gate failed: {wu}")


def assert_step2_gradient_gate(report: Mapping[str, GradientGateReport]) -> None:
    failed = {
        name: gate
        for name, gate in report.items()
        if not (gate.finite and gate.nonzero)
    }
    if failed:
        raise RWDGDataError(f"step2 projection gradient gate failed: {failed}")


def static_best_action_from_natural_targets(targets26: Tensor) -> tuple[int, Mapping[str, Any]]:
    targets26 = targets26.detach().cpu().long().reshape(-1)
    if bool((targets26 < 0).any()) or bool((targets26 >= EAAC_CLASS_COUNT).any()):
        raise RWDGDataError(f"natural EAAC targets must be class ids in [0,{EAAC_CLASS_COUNT - 1}]")
    histogram = torch.bincount(targets26, minlength=EAAC_CLASS_COUNT)[:EAAC_CLASS_COUNT].long()
    non_abstain = histogram[1:]
    if int(non_abstain.sum().item()) <= 0:
        raise RWDGDataError("EAAC StaticBest requires at least one non-abstain action")
    action = int(torch.argmax(non_abstain).item())  # torch.argmax picks smallest tie.
    return action, {
        "count": int(targets26.numel()),
        "class_histogram": [int(x) for x in histogram.tolist()],
        "abstain_rows": int(histogram[0].item()),
        "action_rows": int(non_abstain.sum().item()),
        "static_best_definition": "most frequent non-abstain action in natural 4702-row EAAC target",
        "static_best_rule": "most_frequent_non_abstain_action",
    }


def compute_static_best_action(
    model: RoleWindowDenseGlimpse,
    train_view: RWDGGateSubsetView,
    targets: TrainSubsetTargets,
    *,
    batch_size: int,
    device: torch.device,
) -> Mapping[str, Any]:
    model.eval()
    group_counts = torch.zeros(3, dtype=torch.long)
    all_targets26: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, train_view.size, batch_size):
            rows = torch.arange(start, min(start + batch_size, train_view.size), dtype=torch.long)
            full_cls, _, crops, labels = batch_from_rows(train_view, targets, rows, device=device)
            targets26, groups, _, _, _ = eaac_action_targets(model, full_cls, crops, labels)
            group_counts += torch.bincount(groups.detach().cpu(), minlength=3)[:3]
            all_targets26.append(targets26.detach().cpu().long())
    targets26_all = torch.cat(all_targets26).long()
    static_best, detail = static_best_action_from_natural_targets(targets26_all)
    return {
        "static_best_action": static_best,
        "static_best_rule": "most frequent non-abstain action in natural EAAC target",
        **dict(detail),
        "group_counts": {
            "leader": int(group_counts[0].item()),
            "challenger": int(group_counts[1].item()),
            "outside": int(group_counts[2].item()),
        },
    }


def train_full_gate0(
    config: Gate0TrainConfig,
    *,
    config_sha256: str,
    repo_root: Path,
    expected_commit: str,
) -> Path:
    validate_config(config)
    commit, clean, status = git_commit_and_clean(repo_root)
    if commit != expected_commit:
        raise RWDGDataError(
            f"expected commit mismatch: HEAD={commit}, expected={expected_commit}"
        )
    if config.require_clean_tree and not clean:
        raise RWDGDataError("repository must be clean before formal Gate0 training:\n" + status)
    device = resolve_device(config)
    set_reproducibility(config.seed)
    asset_config = asset_config_from_train_config(config)
    oracle_receipt = load_and_validate_oracle_receipt(config)

    assets, views = load_rwdg_gate_data(
        asset_config,
        strict_sha=config.strict_sha,
        validate_tensor_values=config.validate_tensor_values,
        strict_eval_boundary=True,
    )
    try:
        train_view = views["dev_train"]
        train_targets = load_dev_train_targets(
            assets,
            strict_sha=config.strict_sha,
        )
        validate_training_axis(
            train_targets,
            train_view,
            full_class_count=int(assets.name_embeddings.shape[0]),
        )
        model = build_train100_model(
            assets.role_embeddings,
            assets.name_embeddings,
            train_targets.class_ids,
            device=device,
            seed=config.seed,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            foreach=False,
            fused=False,
        )
        target_plan = precompute_eaac_target_plan(
            model,
            train_view,
            train_targets,
            batch_size=config.batch_size,
            device=device,
        )
        sampled_stats = sampled_class_histogram(
            target_plan.targets26,
            target_plan.groups,
            updates=config.updates,
            batch_size=config.batch_size,
            seed=config.seed,
            challenger_per_batch=4,
        )
        sampler = BalancedGroupSampler(
            target_plan.groups,
            batch_size=config.batch_size,
            challenger_per_batch=4,
            seed=config.seed,
        )

        losses: list[float] = []
        rows = sampler.sample()
        loss, step1_report = train_step(
            model, optimizer, train_view, train_targets, rows, device=device
        )
        losses.append(loss)
        assert_step1_gradient_gate(step1_report)

        rows = sampler.sample()
        loss, step2_report = train_step(
            model, optimizer, train_view, train_targets, rows, device=device
        )
        losses.append(loss)
        assert_step2_gradient_gate(step2_report)

        for _ in range(2, config.updates):
            rows = sampler.sample()
            loss, _ = train_step(
                model, optimizer, train_view, train_targets, rows, device=device
            )
            losses.append(loss)

        static_best_action, static_best_detail = static_best_action_from_natural_targets(target_plan.targets26)
        static_best = {
            "static_best_action": static_best_action,
            "static_best_rule": "most frequent non-abstain action in natural EAAC target",
            **dict(static_best_detail),
            "group_counts": dict(target_plan.stats["natural_distribution"]["group_counts"]),
            "action_class_histogram": target_plan.stats["class_histogram"][1:],
        }
        checkpoint = build_checkpoint_payload(
            model,
            config=config,
            config_sha256=config_sha256,
            commit=commit,
            assets=assets,
            oracle_receipt=oracle_receipt,
            train_targets=train_targets,
            losses=losses,
            step1_report=step1_report,
            step2_report=step2_report,
            static_best=static_best,
            target_plan=target_plan,
            sampler=sampler,
            sampled_stats=sampled_stats,
        )
        output_dir = prepare_output_dir(config.output_dir, repo_root)
        checkpoint_path = output_dir / "eaac_gate0_full.pt"
        if checkpoint_path.exists():
            raise RWDGDataError(f"checkpoint already exists: {checkpoint_path}")
        atomic_torch_save(checkpoint_path, checkpoint)
        checkpoint_sha = sha256_file(checkpoint_path)
        history_path = output_dir / "train_history.json"
        atomic_write_json(
            history_path,
            {
                "schema_version": TRAIN_SCHEMA,
                "experiment_id": config.experiment_id,
                "condition_id": config.condition_id,
                "code_commit": commit,
                "config_sha256": config_sha256,
                "oracle_receipt_sha256": config.oracle_receipt_sha256,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "history_path": str(history_path),
                "official_test_loaded": config.official_test_loaded,
                "unseen_images_used_for_gradient": config.unseen_images_used_for_gradient,
                "pclr_online_inference": config.pclr_online_inference,
                "loss": checkpoint["loss"],
                "gradient_gates": checkpoint["gradient_gates"],
                "static_best": checkpoint["static_best"],
                "target_stats": checkpoint["target_stats"],
                "sampler": checkpoint["sampler"],
                "sampled_class_stats": checkpoint["sampled_class_stats"],
            },
        )
        return checkpoint_path
    finally:
        assets.close()


def build_checkpoint_payload(
    model: RoleWindowDenseGlimpse,
    *,
    config: Gate0TrainConfig,
    config_sha256: str,
    commit: str,
    assets: Any,
    oracle_receipt: Mapping[str, Any],
    train_targets: TrainSubsetTargets,
    losses: Sequence[float],
    step1_report: Mapping[str, GradientGateReport],
    step2_report: Mapping[str, GradientGateReport],
    static_best: Mapping[str, Any],
    target_plan: EAACTargetPlan,
    sampler: BalancedGroupSampler,
    sampled_stats: Mapping[str, Any],
) -> Mapping[str, Any]:
    model_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    forbidden_asset_keys = [
        key for key in model_state if key.endswith(("role_embeddings", "name_embeddings", "class_ids"))
    ]
    if forbidden_asset_keys:
        raise RWDGDataError(
            "checkpoint state_dict includes class-axis asset buffers: "
            + ", ".join(forbidden_asset_keys)
        )
    return {
        "schema_version": TRAIN_SCHEMA,
        "experiment_id": config.experiment_id,
        "condition_id": config.condition_id,
        "code_commit": commit,
        "config_sha256": config_sha256,
        "cuav_bundle_manifest_sha256": config.cuav_bundle_manifest_sha256,
        "official_test_loaded": config.official_test_loaded,
        "unseen_images_used_for_gradient": config.unseen_images_used_for_gradient,
        "pclr_online_inference": config.pclr_online_inference,
        "method": "EAAC",
        "gate": "Gate0",
        "train_scope": "Full1000",
        "state_dict": model_state,
        "static_best": static_best,
        "config": asdict(config),
        "reproducibility_identity": {
            "code_commit": commit,
            "config_sha256": config_sha256,
            "seed": config.seed,
            "updates": config.updates,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "sampler": sampler.state_dict(),
            "sampled_class_stats": sampled_stats,
            "action_geometry_sha256": ACTION_GEOMETRY_SHA256,
            "oracle_receipt_sha256": config.oracle_receipt_sha256,
            "official_test_loaded": config.official_test_loaded,
            "unseen_images_used_for_gradient": config.unseen_images_used_for_gradient,
            "pclr_online_inference": config.pclr_online_inference,
        },
        "asset_receipt": {
            "text_manifest": {
                "path": assets.config.text_manifest.path,
                "sha256": assets.config.text_manifest.sha256,
            },
            "role_tensor": asdict(assets.config.role_tensor),
            "name_tensor": asdict(assets.config.name_tensor),
            "patch_manifest": {
                "path": assets.config.patch_manifest.path,
                "sha256": assets.config.patch_manifest.sha256,
            },
            "cls_tensor": asdict(assets.config.cls_tensor),
            "patch_tensor": asdict(assets.config.patch_tensor),
            "cuav_bundle_manifest": {
                "path": assets.config.cuav_bundle_manifest.path,
                "sha256": assets.config.cuav_bundle_manifest.sha256,
            },
            "dev_train_manifest_sha256": assets.config.dev_train_manifest_sha256,
            "dev_eval_manifest_sha256": assets.config.dev_eval_manifest_sha256,
            "dev_eval_oracle_manifest_sha256": assets.config.dev_eval_oracle_manifest_sha256,
            "train_targets": {
                "labels_path": train_targets.labels_path,
                "labels_sha256": train_targets.labels_sha256,
                "class_ids_path": train_targets.class_ids_path,
                "class_ids_sha256": train_targets.class_ids_sha256,
                "crop_features_path": train_targets.crop_features_path,
                "crop_features_sha256": train_targets.crop_features_sha256,
            },
        },
        "oracle_receipt": {
            "path": config.oracle_receipt,
            "sha256": config.oracle_receipt_sha256,
            "dev_eval_oracle_manifest_sha256": assets.config.dev_eval_oracle_manifest_sha256,
            "dev_eval_oracle_summary": dict(assets.subset_summaries["dev_eval_oracle"]),
            "gate": oracle_receipt["gate"],
            "identity": oracle_receipt["identity"],
        },
        "opened_keys": [
            "text_manifest",
            "role_tensor",
            "name_tensor",
            "patch_manifest",
            "cls_tensor",
            "patch_tensor_safe_view",
            "cuav_bundle_manifest",
            "dev_train_manifest",
            "dev_train.labels",
            "dev_train.class_ids",
            "dev_train.crop_features",
        ],
        "loss": {
            "first": float(losses[0]),
            "second": float(losses[1]),
            "last": float(losses[-1]),
            "num_updates": len(losses),
        },
        "gradient_gates": {
            "step1": _gradient_report_to_json(step1_report),
            "step2": _gradient_report_to_json(step2_report),
        },
        "target_stats": target_plan.stats,
        "sampler": sampler.state_dict(),
        "sampled_class_stats": sampled_stats,
    }


def _gradient_report_to_json(
    report: Mapping[str, GradientGateReport],
) -> Mapping[str, Mapping[str, float | bool]]:
    return {
        name: {
            "finite": gate.finite,
            "nonzero": gate.nonzero,
            "grad_abs_sum": gate.grad_abs_sum,
            "grad_max_abs": gate.grad_max_abs,
        }
        for name, gate in report.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Strict Gate0 JSON config path")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    config, config_sha = load_strict_config(args.config)
    if config_sha.lower() != args.expected_config_sha.lower():
        raise RWDGDataError(
            f"config SHA mismatch: got {config_sha}, expected {args.expected_config_sha}"
        )
    checkpoint_path = train_full_gate0(
        config,
        config_sha256=config_sha,
        repo_root=repo_root,
        expected_commit=args.expected_commit,
    )
    print(str(checkpoint_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Gate0TrainConfig",
    "GradientGateReport",
    "BalancedGroupSampler",
    "EAAC_CLASS_COUNT",
    "EAAC_PREREGISTERED_TARGET_COUNTS",
    "EAAC_PREREGISTERED_TARGET_HISTOGRAM",
    "EAACTargetPlan",
    "STRICT_CONFIG_FIELDS",
    "TrainSubsetTargets",
    "asset_config_from_train_config",
    "batch_from_rows",
    "build_checkpoint_payload",
    "build_train100_model",
    "collect_gradient_report",
    "compare_oracle_identity",
    "compute_static_best_action",
    "crop_leader_minus_challenger_margins",
    "eaac_action_loss",
    "eaac_action_targets",
    "eaac_policy_logits",
    "eaac_targets",
    "git_commit_and_clean",
    "gradient_gate_report",
    "load_and_validate_oracle_receipt",
    "load_dev_train_targets",
    "load_strict_config",
    "oracle_gate_from_receipt",
    "oracle_identity_from_receipt",
    "prepare_output_dir",
    "precompute_eaac_target_plan",
    "sampled_class_histogram",
    "set_reproducibility",
    "static_best_action_from_natural_targets",
    "train_full_gate0",
    "train_step",
    "validate_config",
    "validate_training_axis",
]
