"""Asset loading helpers for CTPM."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from tools.runtime import sha256_file


DATASET_SPECS = {
    "CUB": {
        "train_count": 7057,
        "test_seen_count": 1764,
        "test_unseen_count": 2967,
        "seen_count": 150,
        "class_count": 200,
    }
}


@dataclass(frozen=True)
class CTPMAssets:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_seen_features: torch.Tensor
    test_seen_labels: torch.Tensor
    test_unseen_features: torch.Tensor
    test_unseen_labels: torch.Tensor
    class_name_embeds: torch.Tensor
    role_sentence_embeds: torch.Tensor
    train_patches: torch.Tensor
    test_seen_patches: torch.Tensor
    test_unseen_patches: torch.Tensor
    seen_classes: torch.Tensor
    unseen_classes: torch.Tensor
    identity: dict


def _manifest(path: str | Path, expected_sha256: str, name: str) -> tuple[dict, Path]:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        raise ValueError(f"{name} must be an absolute path.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {manifest_path}")
    actual = sha256_file(manifest_path)
    if actual != expected_sha256:
        raise ValueError(f"{name} SHA mismatch: {actual}")
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def _load_tensor(manifest_path: Path, manifest: dict, candidates: tuple[str, ...], shape: tuple[int, ...]) -> tuple[str, torch.Tensor]:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict):
        raise ValueError("asset manifest missing outputs_sha256.")
    for filename in candidates:
        if filename not in outputs:
            continue
        path = manifest_path.parent / filename
        if not path.is_file() or sha256_file(path) != outputs[filename]:
            raise ValueError(f"asset tensor missing or SHA mismatch: {filename}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{filename} shape mismatch: {tuple(tensor.shape)} != {shape}")
        if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
            raise ValueError(f"{filename} contains NaN/Inf.")
        return filename, tensor
    raise ValueError(f"asset manifest lacks any of {candidates!r}.")


def _check_base_manifest(manifest: dict, config: dict, spec: dict) -> None:
    counts = manifest.get("counts")
    expected_counts = {
        "train": spec["train_count"],
        "test_seen": spec["test_seen_count"],
        "test_unseen": spec["test_unseen_count"],
    }
    if (
        manifest.get("schema_version") != "gzsl-paper.clip-assets.v1"
        or manifest.get("dataset") != config["dataset"]
        or manifest.get("asset_id") != config["base_asset_id"]
        or (counts is not None and counts != expected_counts)
    ):
        raise ValueError("base asset identity mismatch.")
    forbidden = {"attributes", "class_attributes", "part_labels", "parts", "boxes", "bounding_boxes", "expert_residuals"}
    if forbidden.intersection(manifest.get("outputs_sha256", {})):
        raise ValueError("base asset contains forbidden human/expert annotations.")


def _check_visual_manifest(manifest: dict, config: dict) -> None:
    if manifest.get("dataset") != config["dataset"] or manifest.get("asset_id") != config["visual_asset_id"]:
        raise ValueError("visual asset identity mismatch.")
    if "coarse36" not in str(manifest.get("asset_id", "")).lower() and "coarse36" not in str(manifest.get("formula", "")).lower():
        raise ValueError("visual asset is not declared as coarse36.")


def load_ctpm_assets(config: dict) -> CTPMAssets:
    spec = DATASET_SPECS.get(config.get("dataset"))
    if spec is None:
        raise ValueError("CTPM currently supports CUB only.")

    base_manifest, base_path = _manifest(config["base_asset_manifest"], config["base_asset_manifest_sha256"], "base_asset_manifest")
    visual_manifest, visual_path = _manifest(config["visual_asset_manifest"], config["visual_asset_manifest_sha256"], "visual_asset_manifest")
    class_manifest, class_path = _manifest(config["class_name_asset_manifest"], config["class_name_asset_manifest_sha256"], "class_name_asset_manifest")
    _check_base_manifest(base_manifest, config, spec)
    _check_visual_manifest(visual_manifest, config)
    if class_manifest.get("dataset") != config["dataset"] or class_manifest.get("asset_id") != config["class_name_asset_id"]:
        raise ValueError("class-name asset identity mismatch.")

    tensors = {}
    for key, shape in {
        "train_features": (spec["train_count"], 768),
        "train_labels": (spec["train_count"],),
        "test_seen_features": (spec["test_seen_count"], 768),
        "test_seen_labels": (spec["test_seen_count"],),
        "test_unseen_features": (spec["test_unseen_count"], 768),
        "test_unseen_labels": (spec["test_unseen_count"],),
        "role_sentence_embeds": (spec["class_count"], 8, 768),
    }.items():
        _, tensors[key] = _load_tensor(base_path, base_manifest, (f"{key}.pt",), shape)
    _, class_name_embeds = _load_tensor(class_path, class_manifest, ("class_name_embeds.pt",), (spec["class_count"], 768))

    patch_names = {
        "train_patches": ("train_patch_features.pt", "train_coarse36_features.pt", "train_features.pt"),
        "test_seen_patches": ("test_seen_patch_features.pt", "test_seen_coarse36_features.pt", "test_seen_features.pt"),
        "test_unseen_patches": ("test_unseen_patch_features.pt", "test_unseen_coarse36_features.pt", "test_unseen_features.pt"),
    }
    patch_shapes = {
        "train_patches": (spec["train_count"], 36, 768),
        "test_seen_patches": (spec["test_seen_count"], 36, 768),
        "test_unseen_patches": (spec["test_unseen_count"], 36, 768),
    }
    patches = {key: _load_tensor(visual_path, visual_manifest, names, patch_shapes[key])[1] for key, names in patch_names.items()}

    seen = torch.unique(tensors["train_labels"].long(), sorted=True)
    unseen = torch.unique(tensors["test_unseen_labels"].long(), sorted=True)
    if (
        seen.numel() != spec["seen_count"]
        or unseen.numel() != spec["class_count"] - spec["seen_count"]
        or not torch.equal(torch.unique(tensors["test_seen_labels"].long(), sorted=True), seen)
        or torch.isin(seen, unseen).any()
        or not torch.equal(torch.cat((seen, unseen)).sort().values, torch.arange(spec["class_count"]))
    ):
        raise ValueError("seen/unseen split identity mismatch.")
    if not torch.equal(tensors["train_labels"].long(), tensors["train_labels"].long()):
        raise ValueError("train labels are invalid.")

    return CTPMAssets(
        train_features=tensors["train_features"],
        train_labels=tensors["train_labels"].long(),
        test_seen_features=tensors["test_seen_features"],
        test_seen_labels=tensors["test_seen_labels"].long(),
        test_unseen_features=tensors["test_unseen_features"],
        test_unseen_labels=tensors["test_unseen_labels"].long(),
        class_name_embeds=class_name_embeds,
        role_sentence_embeds=tensors["role_sentence_embeds"],
        train_patches=patches["train_patches"],
        test_seen_patches=patches["test_seen_patches"],
        test_unseen_patches=patches["test_unseen_patches"],
        seen_classes=seen,
        unseen_classes=unseen,
        identity={
            "base_asset_manifest_sha256": config["base_asset_manifest_sha256"],
            "visual_asset_manifest_sha256": config["visual_asset_manifest_sha256"],
            "class_name_asset_manifest_sha256": config["class_name_asset_manifest_sha256"],
        },
    )
