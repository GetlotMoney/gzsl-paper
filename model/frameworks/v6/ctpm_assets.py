"""Identity-bound train/eval assets for IDEA-208 CTPM."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from tools.runtime import sha256_file


EMBED_DIM = 768
CLASS_COUNT = 200
TRAIN_COUNT = 7057
TEST_SEEN_COUNT = 1764
TEST_UNSEEN_COUNT = 2967
SEEN_COUNT = 150
PATCH_COUNT = 36


@dataclass(frozen=True)
class CTPMTrainAssets:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    train_patches: np.memmap
    class_name_embeds: torch.Tensor
    role_sentence_embeds: torch.Tensor
    seen_classes: torch.Tensor
    unseen_classes: torch.Tensor
    identity: dict[str, Any]


@dataclass(frozen=True)
class CTPMEvalAssets:
    test_seen_features: torch.Tensor
    test_seen_labels: torch.Tensor
    test_seen_patches: np.memmap
    test_unseen_features: torch.Tensor
    test_unseen_labels: torch.Tensor
    test_unseen_patches: np.memmap
    class_name_embeds: torch.Tensor
    role_sentence_embeds: torch.Tensor
    seen_classes: torch.Tensor
    unseen_classes: torch.Tensor
    identity: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _manifest(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(config["asset_manifest"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("CTPM asset manifest path is invalid.")
    if sha256_file(path) != config["asset_manifest_sha256"]:
        raise ValueError("CTPM asset manifest SHA mismatch.")
    manifest = _json(path)
    expected_counts = {
        "train": TRAIN_COUNT,
        "test_seen": TEST_SEEN_COUNT,
        "test_unseen": TEST_UNSEEN_COUNT,
    }
    outputs = manifest.get("outputs_sha256")
    forbidden = {
        "attributes", "class_attributes", "part_labels", "parts",
        "boxes", "bounding_boxes", "expert_residuals",
    }
    if (
        manifest.get("schema_version") != "gzsl-paper.clip-assets.v1"
        or manifest.get("asset_id") != config["asset_id"]
        or manifest.get("dataset") != "CUB"
        or manifest.get("counts") != expected_counts
        or not isinstance(outputs, dict)
        or forbidden.intersection(outputs)
    ):
        raise ValueError("CTPM asset identity, counts, or data boundary mismatch.")
    configured = config["coarse_patch_files_sha256"]
    for name in (
        "train_coarse_patch_features.npy",
        "test_seen_coarse_patch_features.npy",
        "test_unseen_coarse_patch_features.npy",
    ):
        if configured.get(name) != outputs.get(name):
            raise ValueError("CTPM coarse patch config/manifest mismatch.")
    return path, manifest


def _output(path: Path, manifest: Mapping[str, Any], name: str) -> Path:
    candidate = path.parent / name
    expected = manifest["outputs_sha256"].get(name)
    if not candidate.is_file() or sha256_file(candidate) != expected:
        raise ValueError(f"CTPM asset output mismatch: {name}")
    return candidate


def _tensor(
    path: Path, manifest: Mapping[str, Any], name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    value = torch.load(_output(path, manifest, name), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"CTPM tensor shape mismatch: {name}")
    if value.is_floating_point() and not torch.isfinite(value.float()).all():
        raise ValueError(f"CTPM tensor non-finite: {name}")
    return value


def _patches(
    path: Path, manifest: Mapping[str, Any], name: str, count: int
) -> np.memmap:
    value = np.load(_output(path, manifest, name), mmap_mode="r")
    if not isinstance(value, np.memmap):
        raise ValueError("CTPM patches must remain memory-mapped.")
    if tuple(value.shape) != (count, PATCH_COUNT, EMBED_DIM) or value.dtype != np.float16:
        raise ValueError(f"CTPM patch shape/dtype mismatch: {name}")
    return value


def _axes(
    train_labels: torch.Tensor | None = None,
    seen_labels: torch.Tensor | None = None,
    unseen_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if train_labels is not None:
        seen = torch.unique(train_labels.long(), sorted=True)
        unseen = torch.arange(CLASS_COUNT)[~torch.isin(torch.arange(CLASS_COUNT), seen)]
    elif seen_labels is not None and unseen_labels is not None:
        seen = torch.unique(seen_labels.long(), sorted=True)
        unseen = torch.unique(unseen_labels.long(), sorted=True)
    else:
        raise ValueError("CTPM class axes require train or test labels.")
    if (
        seen.numel() != SEEN_COUNT
        or unseen.numel() != CLASS_COUNT - SEEN_COUNT
        or bool(torch.isin(seen, unseen).any())
        or not torch.equal(torch.cat((seen, unseen)).sort().values, torch.arange(CLASS_COUNT))
    ):
        raise ValueError("CTPM seen/unseen class axes mismatch.")
    return seen, unseen


def load_top2_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["top2_gate_result"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("CTPM Top2 gate path is invalid.")
    if sha256_file(path) != config["top2_gate_result_sha256"]:
        raise ValueError("CTPM Top2 gate SHA mismatch.")
    value = _json(path)
    gates = value.get("gates", {})
    parent = value.get("parent_metrics", {})
    if (
        value.get("schema_version") != "gzsl-paper.v6-ctpm-top2-gate.v1"
        or value.get("asset_manifest_sha256") != config["asset_manifest_sha256"]
        or value.get("passed") is not True
        or not all(gates.get(name) is True for name in (
            "train_coverage_ge_0p60", "train_truth_c2_ge_100", "oracle_H_gain_ge_1"
        ))
        or abs(float(parent.get("H", -1.0)) - float(config["parent_metrics_percent"]["H"])) > 1e-6
        or float(value.get("oracle_H_gain", -1.0)) < 1.0
        or value.get("unseen_images_used_for_gradient") is not False
    ):
        raise ValueError("CTPM Top2 gate identity or hard gates mismatch.")
    return {
        "path": str(path),
        "sha256": config["top2_gate_result_sha256"],
        "script_sha256": config["top2_gate_script_sha256"],
        "split_counts": value["split_counts"],
        "parent_metrics": parent,
        "oracle_metrics": value["oracle_metrics"],
        "oracle_H_gain": value["oracle_H_gain"],
    }


def load_ctpm_train_assets(config: Mapping[str, Any]) -> CTPMTrainAssets:
    gate = load_top2_gate(config)
    path, manifest = _manifest(config)
    features = _tensor(path, manifest, "train_features.pt", (TRAIN_COUNT, EMBED_DIM))
    labels = _tensor(path, manifest, "train_labels.pt", (TRAIN_COUNT,)).long()
    names = _tensor(path, manifest, "class_name_embeds.pt", (CLASS_COUNT, EMBED_DIM))
    roles = _tensor(path, manifest, "role_sentence_embeds.pt", (CLASS_COUNT, 8, EMBED_DIM))
    patches = _patches(path, manifest, "train_coarse_patch_features.npy", TRAIN_COUNT)
    seen, unseen = _axes(train_labels=labels)
    return CTPMTrainAssets(
        train_features=features,
        train_labels=labels,
        train_patches=patches,
        class_name_embeds=names,
        role_sentence_embeds=roles,
        seen_classes=seen,
        unseen_classes=unseen,
        identity={
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "class_name_sha256": manifest["outputs_sha256"]["class_name_embeds.pt"],
            "role_text_sha256": manifest["outputs_sha256"]["role_sentence_embeds.pt"],
            "train_patch_sha256": manifest["outputs_sha256"]["train_coarse_patch_features.npy"],
            "top2_gate": gate,
        },
    )


def load_ctpm_eval_assets(config: Mapping[str, Any]) -> CTPMEvalAssets:
    gate = load_top2_gate(config)
    path, manifest = _manifest(config)
    seen_features = _tensor(path, manifest, "test_seen_features.pt", (TEST_SEEN_COUNT, EMBED_DIM))
    seen_labels = _tensor(path, manifest, "test_seen_labels.pt", (TEST_SEEN_COUNT,)).long()
    unseen_features = _tensor(path, manifest, "test_unseen_features.pt", (TEST_UNSEEN_COUNT, EMBED_DIM))
    unseen_labels = _tensor(path, manifest, "test_unseen_labels.pt", (TEST_UNSEEN_COUNT,)).long()
    names = _tensor(path, manifest, "class_name_embeds.pt", (CLASS_COUNT, EMBED_DIM))
    roles = _tensor(path, manifest, "role_sentence_embeds.pt", (CLASS_COUNT, 8, EMBED_DIM))
    seen_patches = _patches(path, manifest, "test_seen_coarse_patch_features.npy", TEST_SEEN_COUNT)
    unseen_patches = _patches(path, manifest, "test_unseen_coarse_patch_features.npy", TEST_UNSEEN_COUNT)
    seen, unseen = _axes(seen_labels=seen_labels, unseen_labels=unseen_labels)
    return CTPMEvalAssets(
        test_seen_features=seen_features,
        test_seen_labels=seen_labels,
        test_seen_patches=seen_patches,
        test_unseen_features=unseen_features,
        test_unseen_labels=unseen_labels,
        test_unseen_patches=unseen_patches,
        class_name_embeds=names,
        role_sentence_embeds=roles,
        seen_classes=seen,
        unseen_classes=unseen,
        identity={
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "test_seen_patch_sha256": manifest["outputs_sha256"]["test_seen_coarse_patch_features.npy"],
            "test_unseen_patch_sha256": manifest["outputs_sha256"]["test_unseen_coarse_patch_features.npy"],
            "top2_gate": gate,
        },
    )
