"""Strict CUAV crop-asset loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools.runtime import sha256_file


SUBSET_SCHEMA = "gzsl-paper.cuav-crop-subset.v1"
BUNDLE_SCHEMA = "gzsl-paper.cuav-crop-bundle.v1"


def load_subset(path, sha, *, subset, open_crops, open_paths, open_lowres=False):
    path = Path(path)
    if not path.is_file() or sha256_file(path) != sha:
        raise ValueError("CUAV subset manifest路径/SHA错误。")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SUBSET_SCHEMA or meta.get("subset") != subset:
        raise ValueError("CUAV subset身份错误。")
    outputs = meta["outputs_sha256"]
    values = {}
    for filename in ("full_cls.pt", "labels.pt", "raw_indices.pt", "class_ids.pt", "name_embeddings.pt", "crop_boxes.pt"):
        tensor_path = path.parent / filename
        if not tensor_path.is_file() or sha256_file(tensor_path) != outputs.get(filename):
            raise ValueError(f"CUAV资产错误：{filename}")
        values[filename.removesuffix(".pt")] = torch.load(tensor_path, map_location="cpu", weights_only=True)
    crops = None
    if open_crops:
        crop_path = path.parent / "crop_features.npy"
        if not crop_path.is_file() or sha256_file(crop_path) != outputs.get(crop_path.name):
            raise ValueError("CUAV crop feature资产错误。")
        crops = np.load(crop_path, mmap_mode="r")
        if crops.shape != (int(meta["count"]), 25, 768) or crops.dtype != np.float16:
            raise ValueError("CUAV crop feature shape/dtype错误。")
    lowres = None
    if open_lowres:
        lowres_path = path.parent / "lowres_crop_features.npy"
        if not lowres_path.is_file() or sha256_file(lowres_path) != outputs.get(lowres_path.name):
            raise ValueError("CUAV lowres crop feature资产错误。")
        lowres = np.load(lowres_path, mmap_mode="r")
        if lowres.shape != (int(meta["count"]), 25, 768) or lowres.dtype != np.float16:
            raise ValueError("CUAV lowres crop feature shape/dtype错误。")
    image_paths = None
    if open_paths:
        paths_file = path.parent / "image_paths.json"
        if not paths_file.is_file() or sha256_file(paths_file) != outputs.get(paths_file.name):
            raise ValueError("CUAV image path资产错误。")
        image_paths = json.loads(paths_file.read_text(encoding="utf-8"))
        if len(image_paths) != int(meta["count"]):
            raise ValueError("CUAV image path count错误。")
    if not torch.equal(values["class_ids"].long(), torch.tensor(meta["class_ids"]).long()):
        raise ValueError("CUAV class_ids与manifest不一致。")
    return values, crops, lowres, image_paths, meta


def validate_bundle(path, sha, *, subset, subset_sha):
    path = Path(path)
    if not path.is_file() or sha256_file(path) != sha:
        raise ValueError("CUAV bundle路径/SHA错误。")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if (
        meta.get("schema_version") != BUNDLE_SCHEMA or meta.get("mode") != "dev"
        or meta.get("subsets", {}).get(subset, {}).get("sha256") != subset_sha
    ):
        raise ValueError("CUAV bundle/subset身份错误。")
    return meta
