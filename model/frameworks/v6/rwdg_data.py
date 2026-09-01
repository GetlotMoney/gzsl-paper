"""Data and asset boundary helpers for V6 / SVRA.

The file name stays ``rwdg_data.py`` so existing runner wiring can import one
data boundary module, but the public contract uses SVRA names and schema text.
Eval-time batches expose only row ids, raw ids, trainval positions, full CLS,
and optional patch tokens; crop/oracle features stay outside this pre-action
view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import os

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .svra import ROLE_ORDER


EXPECTED_ROLE_ORDER: tuple[str, ...] = ROLE_ORDER


class SVRADataError(RuntimeError):
    """Raised when an SVRA asset or subset violates the registered contract."""


@dataclass(frozen=True)
class TensorContract:
    """Expected tensor identity for formal assets or injected test assets."""

    path: str
    sha256: str | None
    shape: tuple[int, ...]
    dtype: str | None = None


@dataclass(frozen=True)
class ManifestContract:
    """Expected manifest path and optional SHA."""

    path: str
    sha256: str | None


@dataclass(frozen=True)
class SVRAAssetConfig:
    """All paths needed to build the SVRA gate data view."""

    text_manifest: ManifestContract
    role_tensor: TensorContract
    name_tensor: TensorContract
    patch_manifest: ManifestContract
    cls_tensor: TensorContract
    patch_tensor: TensorContract
    action_bundle_manifest: ManifestContract
    dev_train_manifest_sha256: str
    dev_eval_manifest_sha256: str
    dev_eval_oracle_manifest_sha256: str
    att_splits_mat_path: str | None = None
    trainval_count: int = 7057
    role_order: tuple[str, ...] = EXPECTED_ROLE_ORDER


FORMAL_SVRA_CONFIG = SVRAAssetConfig(
    text_manifest=ManifestContract(
        path="/data/lby/projects/cv_project/GZSL_Warehouse/assets/clip_vitl14_336/CUB/69c9c6d82a755fe8/asset_manifest.json",
        sha256="52c50c2f55250399bce360a218c30e70b66945953bec7e825e8dd8f20dddf91f",
    ),
    role_tensor=TensorContract(
        path="/data/lby/projects/cv_project/GZSL_Warehouse/assets/clip_vitl14_336/CUB/69c9c6d82a755fe8/role_sentence_embeds.pt",
        sha256="f614a06cd93b071a4d8c7355f78a10588a0b954e46bc99c64b76399c8af5a889",
        shape=(200, 8, 768),
        dtype="float32",
    ),
    name_tensor=TensorContract(
        path="/data/lby/projects/cv_project/GZSL_Warehouse/assets/clip_vitl14_336/CUB/69c9c6d82a755fe8/class_name_embeds.pt",
        sha256="c3a2f177f728621a56d1e972b91614346eee47a749e2902db0af33fac0543232",
        shape=(200, 768),
        dtype="float32",
    ),
    patch_manifest=ManifestContract(
        path="/data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1/asset_manifest.json",
        sha256="d096087c9bd37d90157688e21e79b8ba6a61f0ea9b1fa91f4f544f8bc1dd1ad0",
    ),
    cls_tensor=TensorContract(
        path="/data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1/train_features.pt",
        sha256="5c6e69fbfca4d41d73e133c6085e058e2c6f25237a34a7d00902c80b00b9db9a",
        shape=(7057, 768),
        dtype="float32",
    ),
    patch_tensor=TensorContract(
        path="/data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1/train_patch_features.npy",
        sha256="937a906d18cc7acc556e75fe8b9822e47be8cc6b3d21c89e181a80a257940537",
        shape=(7057, 576, 768),
        dtype="float16",
    ),
    action_bundle_manifest=ManifestContract(
        path="/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/cuav/assets/V5-TRY-005-PRELIM/asset_manifest.json",
        sha256="0b956bb4445033e14bb692dd725fbf894db1f8e2fc337d78cf4a1b3b63cd3450",
    ),
    dev_train_manifest_sha256="0fc6df6b5babd8c8c2822f33ade9caa1fdad9284a424f028dd20991e7b50d20c",
    dev_eval_manifest_sha256="2342392b5fb6f839c07a78922e2c3c59de63f68016e8d7cdbe9f6f11770d8af2",
    dev_eval_oracle_manifest_sha256="2e4db3b918b7ea915e272d4266fd6241e0aa7624e9d5483a7ffdcbf9348b9fea",
    att_splits_mat_path="/data/lby/projects/cv_project/GZSL_Warehouse/datasets/splits/xlsa17/data/CUB/att_splits.mat",
)


@dataclass(frozen=True)
class SVRAAssets:
    """Validated immutable assets backing one SVRA gate data view."""

    config: SVRAAssetConfig
    role_embeddings: Any
    name_embeddings: Any
    cls_features: Any
    patch_features: np.memmap
    trainval_loc_zero_based: np.ndarray
    raw_global_to_trainval_position: Mapping[int, int]
    text_manifest: Mapping[str, Any]
    patch_manifest: Mapping[str, Any]
    action_manifest: Mapping[str, Any]
    dev_train_manifest: Mapping[str, Any]
    dev_eval_manifest: Mapping[str, Any]
    dev_eval_oracle_manifest: Mapping[str, Any]
    subset_summaries: Mapping[str, Mapping[str, Any]]

    def close(self) -> None:
        mmap_obj = getattr(self.patch_features, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def __enter__(self) -> "SVRAAssets":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@dataclass(frozen=True)
class SVRAGateSubsetView:
    """Safe subset-indexed CLS/patch reader for SVRA gate rows."""

    name: str
    raw_global_indices: np.ndarray
    trainval_positions: np.ndarray
    assets: SVRAAssets

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_global_indices)
        pos = np.asarray(self.trainval_positions)
        if raw.ndim != 1:
            raise SVRADataError(f"{self.name}: raw_global_indices must be 1-D, got {raw.shape}")
        if pos.ndim != 1:
            raise SVRADataError(f"{self.name}: trainval_positions must be 1-D, got {pos.shape}")
        if raw.shape != pos.shape:
            raise SVRADataError(
                f"{self.name}: raw_global_indices shape {raw.shape} != trainval_positions shape {pos.shape}"
            )
        if not np.issubdtype(raw.dtype, np.integer):
            raise SVRADataError(f"{self.name}: raw_global_indices must be integer, got {raw.dtype}")
        if not np.issubdtype(pos.dtype, np.integer):
            raise SVRADataError(f"{self.name}: trainval_positions must be integer, got {pos.dtype}")
        if len(raw) == 0:
            raise SVRADataError(f"{self.name}: subset is empty")
        _validate_raw_global_indices(self.name, raw)
        _validate_trainval_positions(self.name, pos, self.assets.config.trainval_count)
        object.__setattr__(self, "raw_global_indices", raw.astype(np.int64, copy=False))
        object.__setattr__(self, "trainval_positions", pos.astype(np.int64, copy=False))

    @property
    def size(self) -> int:
        return int(self.raw_global_indices.shape[0])

    def raw_index_for_rows(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        row_array = np.asarray(rows, dtype=np.int64)
        self._validate_rows(row_array)
        return self.raw_global_indices[row_array]

    def trainval_position_for_rows(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        row_array = np.asarray(rows, dtype=np.int64)
        self._validate_rows(row_array)
        return self.trainval_positions[row_array]

    def batch(
        self,
        rows: Sequence[int] | np.ndarray,
        *,
        include_patches: bool = True,
        as_torch: bool = True,
        device: str | Any | None = None,
    ) -> dict[str, Any]:
        raw_global_indices = self.raw_index_for_rows(rows)
        trainval_positions = self.trainval_position_for_rows(rows)
        cls = _take_first_axis(self.assets.cls_features, trainval_positions)
        out: dict[str, Any] = {
            "subset": self.name,
            "rows": np.asarray(rows, dtype=np.int64),
            "raw_indices": raw_global_indices,
            "raw_global_indices": raw_global_indices,
            "trainval_positions": trainval_positions,
            "cls": cls,
        }
        if include_patches:
            out["patches"] = np.asarray(self.assets.patch_features[trainval_positions])
        if as_torch:
            _require_torch("SVRAGateSubsetView.batch(as_torch=True)")
            out["rows"] = torch.as_tensor(out["rows"], dtype=torch.long, device=device)
            out["raw_indices"] = torch.as_tensor(raw_global_indices, dtype=torch.long, device=device)
            out["raw_global_indices"] = torch.as_tensor(raw_global_indices, dtype=torch.long, device=device)
            out["trainval_positions"] = torch.as_tensor(trainval_positions, dtype=torch.long, device=device)
            out["cls"] = _to_torch_float(cls, device=device)
            if include_patches:
                out["patches"] = _to_torch_float(out["patches"], device=device)
        return out

    def _validate_rows(self, rows: np.ndarray) -> None:
        if rows.ndim != 1:
            raise SVRADataError(f"{self.name}: rows must be 1-D, got {rows.shape}")
        if rows.size and (rows.min() < 0 or rows.max() >= self.size):
            raise SVRADataError(f"{self.name}: row id outside [0,{self.size})")


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_and_load_assets(
    config: SVRAAssetConfig = FORMAL_SVRA_CONFIG,
    *,
    strict_sha: bool = True,
    validate_tensor_values: bool = True,
    verify_large_file_sha: bool = False,
) -> SVRAAssets:
    text_manifest = _load_manifest_contract(config.text_manifest, strict_sha=strict_sha)
    patch_manifest = _load_manifest_contract(config.patch_manifest, strict_sha=strict_sha)
    action_manifest = _load_manifest_contract(config.action_bundle_manifest, strict_sha=strict_sha)

    role_embeddings = _load_torch_tensor_contract(
        config.role_tensor,
        strict_sha=strict_sha,
        validate_values=validate_tensor_values,
        l2_axis=-1,
        l2_min=0.999,
        l2_max=1.001,
    )
    name_embeddings = _load_torch_tensor_contract(
        config.name_tensor,
        strict_sha=strict_sha,
        validate_values=validate_tensor_values,
        l2_axis=-1,
        l2_min=0.999,
        l2_max=1.001,
    )
    cls_features = _load_torch_tensor_contract(
        config.cls_tensor,
        strict_sha=strict_sha,
        validate_values=validate_tensor_values,
        l2_axis=-1,
        l2_min=0.999,
        l2_max=1.001,
    )
    _validate_manifest_output_contract(
        patch_manifest,
        Path(config.patch_tensor.path).name,
        config.patch_tensor.sha256,
        strict_sha=strict_sha,
    )
    patch_features = _load_npy_memmap_contract(
        config.patch_tensor,
        strict_sha=strict_sha,
        verify_file_sha=verify_large_file_sha,
    )

    _validate_role_order(config.role_order)
    _validate_patch_manifest_semantics(patch_manifest, config.patch_tensor)

    base_dir = Path(config.action_bundle_manifest.path).parent
    dev_train_manifest = _load_named_subset_manifest(
        action_manifest,
        "dev_train",
        expected_sha256=config.dev_train_manifest_sha256,
        strict_sha=strict_sha,
        base_dir=base_dir,
    )
    dev_eval_manifest = _load_named_subset_manifest(
        action_manifest,
        "dev_eval",
        expected_sha256=config.dev_eval_manifest_sha256,
        strict_sha=strict_sha,
        base_dir=base_dir,
    )
    dev_eval_oracle_manifest = _load_named_subset_manifest(
        action_manifest,
        "dev_eval_oracle",
        expected_sha256=config.dev_eval_oracle_manifest_sha256,
        strict_sha=strict_sha,
        base_dir=base_dir,
    )

    trainval_loc_zero_based = _load_trainval_loc_zero_based(
        config.att_splits_mat_path,
        expected_count=config.trainval_count,
    )
    raw_global_to_trainval_position = {
        int(raw_global): int(pos)
        for pos, raw_global in enumerate(trainval_loc_zero_based.tolist())
    }

    return SVRAAssets(
        config=config,
        role_embeddings=role_embeddings,
        name_embeddings=name_embeddings,
        cls_features=cls_features,
        patch_features=patch_features,
        trainval_loc_zero_based=trainval_loc_zero_based,
        raw_global_to_trainval_position=raw_global_to_trainval_position,
        text_manifest=text_manifest,
        patch_manifest=patch_manifest,
        action_manifest=action_manifest,
        dev_train_manifest=dev_train_manifest,
        dev_eval_manifest=dev_eval_manifest,
        dev_eval_oracle_manifest=dev_eval_oracle_manifest,
        subset_summaries={
            "dev_train": subset_manifest_summary(dev_train_manifest),
            "dev_eval": subset_manifest_summary(dev_eval_manifest),
            "dev_eval_oracle": subset_manifest_summary(dev_eval_oracle_manifest),
        },
    )


def build_gate_subset_views(
    assets: SVRAAssets,
    *,
    strict_eval_boundary: bool = True,
    strict_sha: bool = True,
) -> dict[str, SVRAGateSubsetView]:
    if strict_eval_boundary:
        _reject_eval_preaction_crop_handles("dev_eval", assets.dev_eval_manifest)
    dev_train_raw = _extract_raw_indices(
        assets.dev_train_manifest,
        "dev_train",
        strict_sha=strict_sha,
    )
    dev_eval_raw = _extract_raw_indices(
        assets.dev_eval_manifest,
        "dev_eval",
        strict_sha=strict_sha,
    )
    _validate_subset_summary("dev_train", assets.subset_summaries["dev_train"], dev_train_raw)
    _validate_subset_summary("dev_eval", assets.subset_summaries["dev_eval"], dev_eval_raw)
    dev_train_pos = _map_raw_global_to_trainval_positions(
        "dev_train",
        dev_train_raw,
        assets.raw_global_to_trainval_position,
    )
    dev_eval_pos = _map_raw_global_to_trainval_positions(
        "dev_eval",
        dev_eval_raw,
        assets.raw_global_to_trainval_position,
    )
    _assert_disjoint_and_complete(
        dev_train_pos,
        dev_eval_pos,
        expected_total=assets.config.trainval_count,
    )
    return {
        "dev_train": SVRAGateSubsetView("dev_train", dev_train_raw, dev_train_pos, assets),
        "dev_eval": SVRAGateSubsetView("dev_eval", dev_eval_raw, dev_eval_pos, assets),
    }


def load_svra_gate_data(
    config: SVRAAssetConfig = FORMAL_SVRA_CONFIG,
    *,
    strict_sha: bool = True,
    validate_tensor_values: bool = True,
    strict_eval_boundary: bool = True,
    verify_large_file_sha: bool = False,
) -> tuple[SVRAAssets, dict[str, SVRAGateSubsetView]]:
    assets = validate_and_load_assets(
        config,
        strict_sha=strict_sha,
        validate_tensor_values=validate_tensor_values,
        verify_large_file_sha=verify_large_file_sha,
    )
    return assets, build_gate_subset_views(
        assets,
        strict_eval_boundary=strict_eval_boundary,
        strict_sha=strict_sha,
    )


def resolve_subset_output(
    assets: SVRAAssets,
    subset_name: str,
    filename: str,
    *,
    verify_sha: bool = True,
) -> Path:
    subset_manifest = _subset_manifest_by_name(assets, subset_name)
    path = _resolve_manifest_output_path(subset_manifest, filename)
    if verify_sha:
        expected_sha = _find_output_sha256(subset_manifest, path.name)
        if expected_sha is None:
            raise SVRADataError(f"{subset_name}: missing outputs_sha256 entry for {path.name}")
        _validate_sha(path, expected_sha, strict_sha=True)
    return path


def _require_torch(context: str) -> None:
    if torch is None:
        raise SVRADataError(f"{context} requires torch, but torch is unavailable")


def _validate_role_order(role_order: Sequence[str]) -> None:
    got = tuple(role_order)
    if got != EXPECTED_ROLE_ORDER:
        raise SVRADataError(f"role order mismatch: got {got}, expected {EXPECTED_ROLE_ORDER}")


def _ensure_file(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if not p.is_file():
        raise SVRADataError(f"required file is missing: {p}")
    return p


def _validate_sha(path: Path, expected: str | None, *, strict_sha: bool) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        msg = f"sha256 mismatch for {path}: got {actual}, expected {expected}"
        if strict_sha:
            raise SVRADataError(msg)
        print(f"[svra_data warning] {msg}")


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, Mapping):
        raise SVRADataError(f"manifest must be a JSON object: {path}")
    return value


def _load_manifest_contract(contract: ManifestContract, *, strict_sha: bool) -> Mapping[str, Any]:
    p = _ensure_file(contract.path)
    _validate_sha(p, contract.sha256, strict_sha=strict_sha)
    loaded = dict(_load_json(p))
    loaded["__manifest_dir__"] = str(p.parent)
    return loaded


def _load_torch_tensor_contract(
    contract: TensorContract,
    *,
    strict_sha: bool,
    validate_values: bool,
    l2_axis: int | None = None,
    l2_min: float | None = None,
    l2_max: float | None = None,
) -> Any:
    _require_torch(f"loading {contract.path}")
    p = _ensure_file(contract.path)
    _validate_sha(p, contract.sha256, strict_sha=strict_sha)
    tensor = _torch_load_cpu(p)
    if isinstance(tensor, Mapping):
        tensor = _first_tensor_value(tensor, source=str(p))
    if not hasattr(tensor, "shape"):
        raise SVRADataError(f"expected tensor in {p}, got {type(tensor)!r}")
    if tuple(int(x) for x in tensor.shape) != contract.shape:
        raise SVRADataError(f"{p}: shape {tuple(tensor.shape)} != expected {contract.shape}")
    if contract.dtype and str(tensor.dtype).replace("torch.", "") != contract.dtype:
        raise SVRADataError(f"{p}: dtype {tensor.dtype} != expected {contract.dtype}")
    if validate_values:
        if not bool(torch.isfinite(tensor).all()):
            raise SVRADataError(f"{p}: tensor contains non-finite values")
        if l2_axis is not None:
            norm = tensor.float().norm(dim=l2_axis)
            if l2_min is not None and bool((norm < l2_min).any()):
                raise SVRADataError(f"{p}: L2 norm below {l2_min}")
            if l2_max is not None and bool((norm > l2_max).any()):
                raise SVRADataError(f"{p}: L2 norm above {l2_max}")
    return tensor


def _torch_load_cpu(path: Path) -> Any:
    _require_torch(f"loading {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _first_tensor_value(value: Mapping[str, Any], *, source: str) -> Any:
    for item in value.values():
        if torch is not None and torch.is_tensor(item):
            return item
    raise SVRADataError(f"no tensor value found in {source}")


def _load_npy_memmap_contract(
    contract: TensorContract,
    *,
    strict_sha: bool,
    verify_file_sha: bool,
) -> np.memmap:
    p = _ensure_file(contract.path)
    if verify_file_sha:
        _validate_sha(p, contract.sha256, strict_sha=strict_sha)
    array = np.load(p, mmap_mode="r")
    if tuple(int(x) for x in array.shape) != contract.shape:
        raise SVRADataError(f"{p}: shape {tuple(array.shape)} != expected {contract.shape}")
    if contract.dtype and str(array.dtype) != contract.dtype:
        raise SVRADataError(f"{p}: dtype {array.dtype} != expected {contract.dtype}")
    return array


def _validate_patch_manifest_semantics(
    manifest: Mapping[str, Any],
    patch_tensor: TensorContract,
) -> None:
    _validate_manifest_output_contract(
        manifest,
        Path(patch_tensor.path).name,
        patch_tensor.sha256,
        strict_sha=True,
    )
    text = json.dumps(manifest, sort_keys=True).lower()
    required_fragments = ("ln_post", "visual.proj", "l2", "576", "24")
    missing = [frag for frag in required_fragments if frag not in text]
    if missing:
        raise SVRADataError("patch manifest missing fragments: " + ", ".join(missing))


def _load_named_subset_manifest(
    bundle_manifest: Mapping[str, Any],
    subset_name: str,
    *,
    expected_sha256: str,
    strict_sha: bool,
    base_dir: Path,
) -> Mapping[str, Any]:
    path = _find_manifest_path(bundle_manifest, subset_name, base_dir=base_dir)
    _validate_sha(path, expected_sha256, strict_sha=strict_sha)
    loaded = dict(_load_json(path))
    loaded["__manifest_dir__"] = str(path.parent)
    return loaded


def _find_manifest_path(
    manifest: Mapping[str, Any],
    subset_name: str,
    *,
    base_dir: Path,
) -> Path:
    subset_lower = subset_name.lower()
    for raw in _walk_manifest_paths(manifest):
        s = str(raw)
        lower = s.lower().replace("\\", "/")
        parts = [part for part in lower.split("/") if part]
        basename = parts[-1] if parts else lower
        parent = parts[-2] if len(parts) >= 2 else ""
        if lower.endswith(".json") and (
            parent == subset_lower
            or basename
            in {
                f"{subset_lower}.json",
                f"{subset_lower}_asset_manifest.json",
                f"{subset_lower}_manifest.json",
            }
            or f"/{subset_lower}/" in f"/{lower}"
        ):
            p = Path(s)
            if not p.is_absolute():
                p = base_dir / p
            if p.is_file():
                return p
    for fallback in (
        base_dir / subset_name / "asset_manifest.json",
        base_dir / f"{subset_name}_asset_manifest.json",
        base_dir / f"{subset_name}_manifest.json",
    ):
        if fallback.is_file():
            return fallback
    raise SVRADataError(f"could not locate {subset_name} manifest under {base_dir}")


def _walk_manifest_paths(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for v in value.values():
            yield from _walk_manifest_paths(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_manifest_paths(v)
    elif isinstance(value, str):
        lower = value.lower()
        if lower.endswith((".json", ".pt", ".npy", ".mat")) or "/" in value or "\\" in value:
            yield value


def subset_manifest_summary(subset_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_path = _resolve_manifest_output_path_or_none(subset_manifest, "raw_indices.pt") or (
        _resolve_manifest_output_path_or_none(subset_manifest, "raw_indices.npy")
    )
    labels_path = (
        _resolve_manifest_output_path_or_none(subset_manifest, "labels.pt")
        or _resolve_manifest_output_path_or_none(subset_manifest, "labels.npy")
        or _resolve_manifest_output_path_or_none(subset_manifest, "targets.pt")
        or _resolve_manifest_output_path_or_none(subset_manifest, "targets.npy")
    )
    return {
        "declared_count": _first_int_value(
            subset_manifest,
            ("count", "num_rows", "n_rows", "num_images", "n_images", "row_count"),
        ),
        "raw_indices_sha256": (
            _find_output_sha256(subset_manifest, "raw_indices.pt")
            or _find_output_sha256(subset_manifest, "raw_indices.npy")
        ),
        "raw_indices_path": str(raw_path) if raw_path is not None else None,
        "labels_sha256": (
            _find_output_sha256(subset_manifest, "labels.pt")
            or _find_output_sha256(subset_manifest, "labels.npy")
            or _find_output_sha256(subset_manifest, "targets.pt")
            or _find_output_sha256(subset_manifest, "targets.npy")
        ),
        "labels_path": str(labels_path) if labels_path is not None else None,
        "manifest_dir": subset_manifest.get("__manifest_dir__"),
    }


def _validate_subset_summary(
    subset_name: str,
    summary: Mapping[str, Any],
    raw_indices: np.ndarray,
) -> None:
    declared_count = summary.get("declared_count")
    if declared_count is not None and int(declared_count) != int(raw_indices.shape[0]):
        raise SVRADataError(
            f"{subset_name}: declared_count {declared_count} != raw_indices count {raw_indices.shape[0]}"
        )


def _first_int_value(value: Any, keys: Sequence[str]) -> int | None:
    wanted = {k.lower() for k in keys}
    if isinstance(value, Mapping):
        for k, v in value.items():
            if str(k).lower() in wanted and isinstance(v, (int, np.integer)):
                return int(v)
            found = _first_int_value(v, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _first_int_value(item, keys)
            if found is not None:
                return found
    return None


def _subset_manifest_by_name(assets: SVRAAssets, subset_name: str) -> Mapping[str, Any]:
    if subset_name == "dev_train":
        return assets.dev_train_manifest
    if subset_name == "dev_eval":
        return assets.dev_eval_manifest
    if subset_name == "dev_eval_oracle":
        return assets.dev_eval_oracle_manifest
    raise SVRADataError(f"unknown subset: {subset_name}")


def _find_output_sha256(manifest: Mapping[str, Any], filename: str) -> str | None:
    target = filename.lower().replace("\\", "/").split("/")[-1]
    outputs = manifest.get("outputs_sha256")
    if isinstance(outputs, Mapping):
        for key, value in outputs.items():
            key_base = str(key).lower().replace("\\", "/").split("/")[-1]
            if key_base == target and isinstance(value, str):
                return value
    return _find_sha_by_filename(manifest, target)


def _validate_manifest_output_contract(
    manifest: Mapping[str, Any],
    filename: str,
    expected_sha256: str | None,
    *,
    strict_sha: bool,
) -> None:
    if expected_sha256 is None:
        return
    actual = _find_output_sha256(manifest, filename)
    if actual is None:
        msg = f"manifest missing outputs_sha256 entry for {filename}"
        if strict_sha:
            raise SVRADataError(msg)
        print(f"[svra_data warning] {msg}")
        return
    if actual.lower() != expected_sha256.lower():
        msg = f"manifest output SHA for {filename} is {actual}, expected {expected_sha256}"
        if strict_sha:
            raise SVRADataError(msg)
        print(f"[svra_data warning] {msg}")


def _resolve_manifest_output_path_or_none(manifest: Mapping[str, Any], filename: str) -> Path | None:
    try:
        return _resolve_manifest_output_path(manifest, filename)
    except SVRADataError:
        return None


def _resolve_manifest_output_path(manifest: Mapping[str, Any], filename: str) -> Path:
    target = filename.lower().replace("\\", "/").split("/")[-1]
    base_dir = Path(str(manifest.get("__manifest_dir__", ".")))
    for raw in _walk_manifest_paths(manifest):
        s = str(raw)
        base = s.lower().replace("\\", "/").split("/")[-1]
        if base == target:
            p = Path(s)
            if not p.is_absolute():
                p = base_dir / p
            if p.is_file():
                return p
    fallback = base_dir / target
    if fallback.is_file():
        return fallback
    raise SVRADataError(f"could not resolve subset output {filename} under {base_dir}")


def _find_sha_by_filename(value: Any, target_basename: str) -> str | None:
    if isinstance(value, Mapping):
        items = list(value.items())
        for k, v in items:
            key_base = str(k).lower().replace("\\", "/").split("/")[-1]
            if key_base == target_basename and isinstance(v, str) and _looks_like_sha256(v):
                return v
            if key_base == target_basename and isinstance(v, Mapping):
                for sha_key in ("sha256", "sha", "file_sha256"):
                    sha_value = v.get(sha_key)
                    if isinstance(sha_value, str) and _looks_like_sha256(sha_value):
                        return sha_value
        for _, v in items:
            found = _find_sha_by_filename(v, target_basename)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_sha_by_filename(item, target_basename)
            if found is not None:
                return found
    return None


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _extract_raw_indices(
    subset_manifest: Mapping[str, Any],
    subset_name: str,
    *,
    strict_sha: bool,
) -> np.ndarray:
    path = _find_raw_indices_path(subset_manifest, subset_name)
    expected_sha = _find_output_sha256(subset_manifest, path.name)
    _validate_sha(path, expected_sha, strict_sha=strict_sha)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        _require_torch(f"loading raw_indices {path}")
        value = _torch_load_cpu(path)
        if isinstance(value, Mapping):
            value = value.get("raw_indices", value.get("indices", _first_tensor_value(value, source=str(path))))
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().astype(np.int64, copy=False)
        return np.asarray(value, dtype=np.int64)
    if suffix == ".npy":
        return np.load(path).astype(np.int64, copy=False)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, Mapping):
            value = value.get("raw_indices", value.get("indices"))
        return np.asarray(value, dtype=np.int64)
    raise SVRADataError(f"{subset_name}: unsupported raw_indices file type: {path}")


def _find_raw_indices_path(subset_manifest: Mapping[str, Any], subset_name: str) -> Path:
    for filename in ("raw_indices.pt", "raw_indices.npy", "raw_indices.json"):
        path = _resolve_manifest_output_path_or_none(subset_manifest, filename)
        if path is not None:
            return path
    raise SVRADataError(f"{subset_name}: raw_indices file not found in subset manifest")


def _load_trainval_loc_zero_based(path: str | None, *, expected_count: int) -> np.ndarray:
    if not path:
        return np.arange(expected_count, dtype=np.int64)
    p = Path(path)
    if not p.is_file():
        raise SVRADataError(f"required att_splits.mat is missing: {p}")
    try:
        from scipy.io import loadmat
    except Exception as exc:  # pragma: no cover
        raise SVRADataError(f"scipy is required to read att_splits.mat: {exc}") from exc
    mat = loadmat(str(p))
    if "trainval_loc" not in mat:
        raise SVRADataError(f"{p}: missing trainval_loc")
    loc = np.asarray(mat["trainval_loc"]).reshape(-1).astype(np.int64) - 1
    if loc.shape[0] != expected_count:
        raise SVRADataError(f"{p}: trainval_loc count {loc.shape[0]} != expected {expected_count}")
    if loc.size and int(loc.min()) < 0:
        raise SVRADataError(f"{p}: trainval_loc must be 1-based positive before conversion")
    if np.unique(loc).shape[0] != loc.shape[0]:
        raise SVRADataError(f"{p}: trainval_loc contains duplicates")
    return loc


def _validate_raw_global_indices(name: str, raw_indices: np.ndarray) -> None:
    if raw_indices.ndim != 1:
        raise SVRADataError(f"{name}: raw_global_indices must be 1-D")
    if raw_indices.size == 0:
        raise SVRADataError(f"{name}: raw_global_indices is empty")
    if int(raw_indices.min()) < 0:
        raise SVRADataError(f"{name}: raw_global_indices contains negative values")
    if int(np.unique(raw_indices).shape[0]) != int(raw_indices.shape[0]):
        raise SVRADataError(f"{name}: raw_global_indices contains duplicates")


def _validate_trainval_positions(name: str, positions: np.ndarray, trainval_count: int) -> None:
    if positions.ndim != 1:
        raise SVRADataError(f"{name}: trainval_positions must be 1-D")
    if positions.size == 0:
        raise SVRADataError(f"{name}: trainval_positions is empty")
    if int(positions.min()) < 0:
        raise SVRADataError(f"{name}: trainval_positions contains negative values")
    if int(positions.max()) >= trainval_count:
        raise SVRADataError(
            f"{name}: trainval_positions max {int(positions.max())} >= trainval_count {trainval_count}"
        )
    if int(np.unique(positions).shape[0]) != int(positions.shape[0]):
        raise SVRADataError(f"{name}: trainval_positions contains duplicates")


def _map_raw_global_to_trainval_positions(
    name: str,
    raw_global_indices: np.ndarray,
    mapping: Mapping[int, int],
) -> np.ndarray:
    missing = [int(x) for x in raw_global_indices.tolist() if int(x) not in mapping]
    if missing:
        raise SVRADataError(
            f"{name}: raw global indices are not in att_splits trainval_loc, examples={missing[:5]}"
        )
    return np.asarray([mapping[int(x)] for x in raw_global_indices.tolist()], dtype=np.int64)


def _assert_disjoint_and_complete(
    dev_train_pos: np.ndarray,
    dev_eval_pos: np.ndarray,
    *,
    expected_total: int,
) -> None:
    train_set = set(int(x) for x in dev_train_pos.tolist())
    eval_set = set(int(x) for x in dev_eval_pos.tolist())
    overlap = train_set.intersection(eval_set)
    if overlap:
        raise SVRADataError(f"dev_train/dev_eval trainval positions overlap, examples={sorted(overlap)[:5]}")
    union_count = len(train_set.union(eval_set))
    if union_count != expected_total:
        raise SVRADataError(f"dev_train/dev_eval union size {union_count} != expected {expected_total}")


def _reject_eval_preaction_crop_handles(
    subset_name: str,
    subset_manifest: Mapping[str, Any],
) -> None:
    if _truthy_manifest_flag(subset_manifest, "crop_features_present"):
        raise SVRADataError(f"{subset_name}: crop_features_present must be false")
    found = sorted(_forbidden_eval_outputs(subset_manifest))
    if found:
        raise SVRADataError(
            f"{subset_name}: eval pre-action manifest exposes forbidden crop handles: "
            + ", ".join(found)
        )


def _truthy_manifest_flag(value: Any, key_name: str) -> bool:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if str(k).lower() == key_name.lower():
                return _manifest_bool(v)
            if _truthy_manifest_flag(v, key_name):
                return True
    elif isinstance(value, list):
        return any(_truthy_manifest_flag(item, key_name) for item in value)
    return False


def _manifest_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"false", "0", "no", "none", "null", ""}:
            return False
        if text in {"true", "1", "yes"}:
            return True
    return bool(value)


def _forbidden_eval_outputs(manifest: Mapping[str, Any]) -> set[str]:
    forbidden: set[str] = set()
    outputs = manifest.get("outputs_sha256")
    if isinstance(outputs, Mapping):
        for key in outputs:
            base = str(key).lower().replace("\\", "/").split("/")[-1]
            if base in {
                "crop_features.npy",
                "crop_features.pt",
                "all25_crop_features.npy",
                "all25_crop_features.pt",
            }:
                forbidden.add(str(key))
    for raw in _walk_manifest_paths(manifest):
        base = str(raw).lower().replace("\\", "/").split("/")[-1]
        if base in {
            "crop_features.npy",
            "crop_features.pt",
            "all25_crop_features.npy",
            "all25_crop_features.pt",
        }:
            forbidden.add(str(raw))
    return forbidden


def _take_first_axis(value: Any, indices: np.ndarray) -> Any:
    if torch is not None and torch.is_tensor(value):
        idx = torch.as_tensor(indices, dtype=torch.long)
        return value.index_select(0, idx)
    return np.asarray(value)[indices]


def _to_torch_float(value: Any, *, device: str | Any | None = None) -> Any:
    _require_torch("converting batch arrays to torch")
    if torch.is_tensor(value):
        tensor = value.float()
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


__all__ = [
    "EXPECTED_ROLE_ORDER",
    "FORMAL_SVRA_CONFIG",
    "ManifestContract",
    "SVRAAssetConfig",
    "SVRAAssets",
    "SVRADataError",
    "SVRAGateSubsetView",
    "TensorContract",
    "build_gate_subset_views",
    "load_svra_gate_data",
    "resolve_subset_output",
    "sha256_file",
    "subset_manifest_summary",
    "validate_and_load_assets",
]
