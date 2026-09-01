"""Strict OREF visible-token asset loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools.runtime import sha256_file


SUBSET_SCHEMA = "gzsl-paper.oref-visible-subset.v1"
BUNDLE_SCHEMA = "gzsl-paper.oref-visible-bundle.v1"


def load_subset(path: Path, sha: str, *, subset: str, open_patches: bool, open_roles: bool):
    if not path.is_file() or sha256_file(path) != sha:
        raise ValueError("OREF subset manifest路径/SHA错误。")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SUBSET_SCHEMA or meta.get("subset") != subset:
        raise ValueError("OREF subset身份错误。")
    outputs = meta["outputs_sha256"]
    files = ["image_cls.pt", "labels.pt", "raw_indices.pt", "class_ids.pt", "name_embeddings.pt"]
    if open_roles:
        files.append("role_embeddings.pt")
    values = {}
    for filename in files:
        tensor_path = path.parent / filename
        if not tensor_path.is_file() or sha256_file(tensor_path) != outputs.get(filename):
            raise ValueError(f"OREF资产错误：{filename}")
        values[filename.removesuffix(".pt")] = torch.load(tensor_path, map_location="cpu", weights_only=True)
    patches = None
    if open_patches:
        patch_path = path.parent / "patch_tokens.npy"
        if not patch_path.is_file() or sha256_file(patch_path) != outputs.get(patch_path.name):
            raise ValueError("OREF patch token资产错误。")
        patches = np.load(patch_path, mmap_mode="r")
        if patches.shape != (int(meta["count"]), 576, 768) or patches.dtype != np.float16:
            raise ValueError("OREF patch token shape/dtype错误。")
    count, classes = int(meta["count"]), len(meta["class_ids"])
    shapes = {
        "image_cls": (count, 768), "labels": (count,), "raw_indices": (count,),
        "class_ids": (classes,), "name_embeddings": (classes, 768),
    }
    if open_roles:
        shapes["role_embeddings"] = (classes, 8, 768)
    for key, shape in shapes.items():
        if tuple(values[key].shape) != shape:
            raise ValueError(f"OREF {key} shape错误。")
    return values, patches, meta


def validate_bundle(path: Path, sha: str, *, subset: str, subset_sha: str):
    if not path.is_file() or sha256_file(path) != sha:
        raise ValueError("OREF bundle路径/SHA错误。")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if (
        meta.get("schema_version") != BUNDLE_SCHEMA
        or meta.get("mode") != "dev"
        or meta.get("subsets", {}).get(subset, {}).get("sha256") != subset_sha
    ):
        raise ValueError("OREF bundle/subset身份错误。")
    return meta
