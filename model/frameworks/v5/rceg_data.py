"""Strict manifest-backed RCEG asset loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools.runtime import sha256_file


SUBSET_SCHEMA = "gzsl-paper.rceg-masked-subset.v1"
BUNDLE_SCHEMA = "gzsl-paper.rceg-masked-bundle.v1"


def load_rceg_subset(
    manifest_path: Path,
    expected_sha256: str,
    *,
    expected_subset: str,
    include_target: bool,
):
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_sha256:
        raise ValueError("RCEG subset manifest路径或SHA错误。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SUBSET_SCHEMA or manifest.get("subset") != expected_subset:
        raise ValueError("RCEG subset身份错误。")
    outputs = manifest.get("outputs_sha256", {})
    tensor_names = [
        "image_cls.pt", "masked_cls.pt", "labels.pt", "raw_indices.pt",
        "class_ids.pt", "name_embeddings.pt", "role_embeddings.pt",
    ]
    if include_target:
        tensor_names.append("target.pt")
    values = {}
    for filename in tensor_names:
        path = manifest_path.parent / filename
        if not path.is_file() or sha256_file(path) != outputs.get(filename):
            raise ValueError(f"RCEG subset资产缺失或SHA错误：{filename}")
        values[filename.removesuffix(".pt")] = torch.load(
            path, map_location="cpu", weights_only=True
        )
    visible_path = manifest_path.parent / "visible_tokens.npy"
    if not visible_path.is_file() or sha256_file(visible_path) != outputs.get(visible_path.name):
        raise ValueError("RCEG visible token资产缺失或SHA错误。")
    visible = np.load(visible_path, mmap_mode="r")
    count = int(manifest["count"])
    class_count = len(manifest["class_ids"])
    if visible.shape != (count, 4, 432, 768) or visible.dtype != np.float16:
        raise ValueError("RCEG visible token shape/dtype错误。")
    expected = {
        "image_cls": (count, 768), "masked_cls": (count, 4, 768),
        "labels": (count,), "raw_indices": (count,), "class_ids": (class_count,),
        "name_embeddings": (class_count, 768),
        "role_embeddings": (class_count, 8, 768),
    }
    if include_target:
        expected["target"] = (count, 4, 1024)
    for key, shape in expected.items():
        if tuple(values[key].shape) != shape:
            raise ValueError(f"RCEG {key} shape错误：{tuple(values[key].shape)} != {shape}")
    if not torch.equal(values["class_ids"].long(), torch.tensor(manifest["class_ids"]).long()):
        raise ValueError("RCEG class_ids与manifest不一致。")
    return values, visible, manifest


def validate_bundle(
    bundle_path: Path,
    expected_sha256: str,
    *,
    subset_name: str,
    subset_sha256: str,
):
    if not bundle_path.is_file() or sha256_file(bundle_path) != expected_sha256:
        raise ValueError("RCEG bundle manifest路径或SHA错误。")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (
        bundle.get("schema_version") != BUNDLE_SCHEMA
        or bundle.get("mode") != "dev"
        or bundle.get("subsets", {}).get(subset_name, {}).get("sha256") != subset_sha256
    ):
        raise ValueError("RCEG bundle/subset身份不一致。")
    return bundle
