"""Build traceable four-view CLIP assets for LVER on CUB.

Each view is a fixed 75% corner crop of the original RGB image.  Cropping is
performed before the unmodified OpenAI CLIP preprocessing pipeline, so every
view reaches the same global image-encoder output used by the parent model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tools.gzsl_data import class_order_sha256, load_xlsa_split, resolve_xlsa_image_path
from tools.prepare_paper_clip_assets import (
    MODEL_NAME,
    OFFICIAL_CHECKPOINT_SHA256,
    _atomic_torch_save,
    _encode_images,
    _verify_file,
    load_source_config,
)
from tools.run_contract import current_code_commit
from tools.runtime import sha256_file


SCHEMA_VERSION = "gzsl-paper.lver-local-view-assets.v1"
CROP_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
NORMALIZED_CROP_BOXES = (
    (0.0, 0.0, 0.75, 0.75),
    (0.25, 0.0, 1.0, 0.75),
    (0.0, 0.25, 0.75, 1.0),
    (0.25, 0.25, 1.0, 1.0),
)
OUTPUT_FILES = {
    "train": "train_local_view_features.pt",
    "test_seen": "test_seen_local_view_features.pt",
    "test_unseen": "test_unseen_local_view_features.pt",
}
PARENT_FEATURE_FILES = {
    "train": "train_features.pt",
    "test_seen": "test_seen_features.pt",
    "test_unseen": "test_unseen_features.pt",
}
PREPROCESS_IDENTITY = {
    "resolution": 336,
    "resize": "bicubic_shorter_side",
    "crop": "center_336",
    "rgb": True,
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711],
}


def pixel_crop_boxes(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    """Convert the fixed normalized boxes to deterministic PIL crop boxes."""
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("图像宽高必须为正数。")
    boxes = []
    for left, top, right, bottom in NORMALIZED_CROP_BOXES:
        x0 = max(0, min(int(width) - 1, math.floor(left * int(width))))
        y0 = max(0, min(int(height) - 1, math.floor(top * int(height))))
        x1 = max(x0 + 1, min(int(width), math.ceil(right * int(width))))
        y1 = max(y0 + 1, min(int(height), math.ceil(bottom * int(height))))
        boxes.append((x0, y0, x1, y1))
    return tuple(boxes)


def crop_pil_views(image: Image.Image) -> tuple[Image.Image, ...]:
    rgb = image.convert("RGB")
    return tuple(rgb.crop(box) for box in pixel_crop_boxes(*rgb.size))


class FourCropDataset(Dataset):
    def __init__(self, paths: list[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return torch.stack([self.preprocess(view) for view in crop_pil_views(image)])


def encode_four_crops(
    model,
    preprocess,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    if not paths:
        raise ValueError("待编码图像列表不能为空。")
    loader = DataLoader(
        FourCropDataset(paths, preprocess),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
    )
    rows = []
    with torch.inference_mode():
        for views in loader:
            if views.ndim != 5 or views.size(1) != len(CROP_NAMES):
                raise RuntimeError("四裁剪预处理输出形状错误。")
            flat = views.flatten(0, 1).to(device, non_blocking=True)
            encoded = F.normalize(model.encode_image(flat).float(), dim=-1)
            rows.append(encoded.reshape(views.size(0), len(CROP_NAMES), -1).cpu())
    result = torch.cat(rows)
    expected = (len(paths), len(CROP_NAMES), 768)
    if tuple(result.shape) != expected or not torch.isfinite(result).all():
        raise RuntimeError(f"LVER局部视图缓存形状或有限性错误：{tuple(result.shape)}。")
    return result


def _repeatability_stats(
    repeat_a: torch.Tensor, repeat_b: torch.Tensor, configured_batch: torch.Tensor
) -> dict[str, object]:
    bitwise_equal = torch.equal(repeat_a, repeat_b)
    max_abs = float((repeat_a - configured_batch).abs().max())
    cosine = F.cosine_similarity(repeat_a.float(), configured_batch.float(), dim=-1)
    minimum_cosine = float(cosine.min())
    if not bitwise_equal or max_abs > 1e-3 or minimum_cosine < 0.99999:
        raise RuntimeError("LVER固定裁剪重复性或batch形状一致性检查失败。")
    return {
        "repeat_bitwise_equal": bitwise_equal,
        "batch1_vs_configured_batch_max_abs_difference": max_abs,
        "batch1_vs_configured_batch_minimum_cosine": minimum_cosine,
        "max_abs_tolerance": 0.001,
        "minimum_cosine_tolerance": 0.99999,
    }


def _parent_parity_stats(encoded: torch.Tensor, parent: torch.Tensor) -> dict[str, object]:
    if encoded.shape != parent.shape or encoded.ndim != 2 or encoded.size(1) != 768:
        raise ValueError("整图parity样本形状不一致。")
    encoded = F.normalize(encoded.float(), dim=-1)
    parent = F.normalize(parent.float(), dim=-1)
    max_abs = float((encoded - parent).abs().max())
    minimum_cosine = float(F.cosine_similarity(encoded, parent, dim=-1).min())
    if max_abs > 0.003 or minimum_cosine < 0.9998:
        raise RuntimeError("当前CLIP运行时与父资产整图出口不一致。")
    return {
        "sample_count": int(encoded.size(0)),
        "max_abs_difference": max_abs,
        "minimum_cosine": minimum_cosine,
        "max_abs_tolerance": 0.003,
        "minimum_cosine_tolerance": 0.9998,
        "bitwise_equal_required": False,
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    norms = tensor.float().norm(dim=-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "finite": bool(torch.isfinite(tensor).all()),
    }


def build_manifest(
    *,
    dataset: str,
    code_commit: str,
    script_sha256: str,
    source_config: Path,
    source_config_sha256: str,
    parent_manifest: Path,
    parent_manifest_sha256: str,
    parent_asset_id: str,
    clip_checkpoint: Path,
    clip_checkpoint_sha256: str,
    clip_runtime: dict[str, object],
    counts: dict[str, int],
    class_order_sha: str,
    raw_image_order_sha: str,
    inputs_sha256: dict[str, str],
    outputs_sha256: dict[str, str],
    output_stats: dict[str, dict[str, object]],
    determinism_check: dict[str, object],
    parent_parity: dict[str, object],
) -> dict[str, object]:
    identity = {
        "parent_manifest_sha256": parent_manifest_sha256,
        "inputs_sha256": inputs_sha256,
        "outputs_sha256": outputs_sha256,
        "crop_boxes": NORMALIZED_CROP_BOXES,
    }
    asset_id = "CUB_openai_vitl14_336_lver_4view_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_id": asset_id,
        "dataset": dataset,
        "generator": {
            "code_commit": code_commit,
            "script": str(Path(__file__).resolve()),
            "script_sha256": script_sha256,
        },
        "parent": {
            "manifest": str(parent_manifest.resolve()),
            "manifest_sha256": parent_manifest_sha256,
            "asset_id": parent_asset_id,
        },
        "source_config": {
            "uri": str(source_config.resolve()),
            "sha256": source_config_sha256,
        },
        "clip": {
            "model": MODEL_NAME,
            "checkpoint": str(clip_checkpoint.resolve()),
            "checkpoint_sha256": clip_checkpoint_sha256,
            **clip_runtime,
        },
        "preprocess": PREPROCESS_IDENTITY,
        "crop_semantics": {
            "crop_before_clip_preprocess": True,
            "order": list(CROP_NAMES),
            "normalized_boxes_xyxy": [list(box) for box in NORMALIZED_CROP_BOXES],
            "pixel_rounding": "floor_start_ceil_end_then_clamp",
            "human_annotations_used": False,
            "bounding_boxes_used": False,
            "part_annotations_used": False,
            "multiscale_used": False,
            "augmentation_used": False,
        },
        "counts": counts,
        "class_order_sha256": class_order_sha,
        "raw_image_order_and_size_sha256": raw_image_order_sha,
        "inputs_sha256": inputs_sha256,
        "outputs_sha256": outputs_sha256,
        "output_tensors": output_stats,
        "determinism_check": determinism_check,
        "full_view_parent_parity": parent_parity,
        "unseen_images_used_for_gradient": False,
    }


def _load_parent(
    manifest_path: Path, config: dict, config_sha: str, split
) -> tuple[dict, str, dict[str, torch.Tensor]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"父资产manifest不存在：{manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "gzsl-paper.clip-assets.v1":
        raise ValueError("父资产manifest schema错误。")
    if manifest.get("dataset") != config["dataset"]:
        raise ValueError("父资产dataset与源配置不一致。")
    if manifest.get("clip_checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("父资产未绑定官方ViT-L/14@336px checkpoint。")
    if manifest.get("source_config_sha256") not in (None, config_sha):
        raise ValueError("父资产source config与当前配置不一致。")
    if manifest.get("class_order_sha256") != class_order_sha256(split.class_names):
        raise ValueError("父资产类别顺序与xlsa17不一致。")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict):
        raise ValueError("父资产缺少outputs_sha256。")
    tensors = {}
    expected_counts = {
        "train": int(split.train_indices.numel()),
        "test_seen": int(split.test_seen_indices.numel()),
        "test_unseen": int(split.test_unseen_indices.numel()),
    }
    for split_name, filename in PARENT_FEATURE_FILES.items():
        path = manifest_path.parent / filename
        if outputs.get(filename) != sha256_file(path):
            raise ValueError(f"父资产{filename} SHA不一致。")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(value.shape) != (expected_counts[split_name], 768):
            raise ValueError(f"父资产{filename}形状错误。")
        tensors[split_name] = value.float()
    return manifest, manifest_sha, tensors


def run(
    config_path: Path,
    parent_manifest_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    workers: int,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"资产输出目录必须不存在：{output_dir}")
    config, config_sha = load_source_config(config_path)
    if config["dataset"] != "CUB":
        raise ValueError("LVER第一版只允许CUB。")
    paths = {
        "raw_root": Path(config["raw_root"]),
        "raw_archive": Path(config["raw_archive"]),
        "res101": Path(config["res101"]),
        "att_splits": Path(config["att_splits"]),
        "clip_checkpoint": Path(config["clip_checkpoint"]),
    }
    input_sha = {
        name: _verify_file(paths[name], config["expected_sha256"][name], name)
        for name in ("raw_archive", "res101", "att_splits", "clip_checkpoint")
    }
    if input_sha["clip_checkpoint"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("CLIP checkpoint不是OpenAI ViT-L/14@336px官方权重。")
    split = load_xlsa_split(paths["res101"], paths["att_splits"])
    image_paths = [
        resolve_xlsa_image_path(paths["raw_root"], value, config["image_path_anchors"])
        for value in split.image_files
    ]
    if len({str(path.resolve()) for path in image_paths}) != len(image_paths):
        raise ValueError("解析后的原始图像路径存在重复。")
    image_order_payload = [
        {"path": str(path.relative_to(paths["raw_root"])).replace("\\", "/"), "size": path.stat().st_size}
        for path in image_paths
    ]
    image_order_sha = hashlib.sha256(
        json.dumps(image_order_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    parent, parent_sha, parent_tensors = _load_parent(
        parent_manifest_path, config, config_sha, split
    )
    if parent.get("raw_image_order_and_size_sha256") != image_order_sha:
        raise ValueError("父资产原图顺序/大小与当前解析结果不一致。")

    import clip

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA生成缓存，但CUDA不可用。")
    model, preprocess = clip.load(str(paths["clip_checkpoint"]), device=device, jit=False)
    model.eval()
    all_features = encode_four_crops(
        model, preprocess, image_paths, device, batch_size, workers
    )
    repeat_a = encode_four_crops(model, preprocess, [image_paths[0]], device, 1, 0)[0]
    repeat_b = encode_four_crops(model, preprocess, [image_paths[0]], device, 1, 0)[0]
    determinism = _repeatability_stats(repeat_a, repeat_b, all_features[0])

    parity_indices = [
        int(split.train_indices[0]),
        int(split.test_seen_indices[0]),
        int(split.test_unseen_indices[0]),
    ]
    parity_encoded = _encode_images(
        model, preprocess, [image_paths[index] for index in parity_indices], device, 1, 0
    )
    parity_parent = torch.stack(
        [parent_tensors["train"][0], parent_tensors["test_seen"][0], parent_tensors["test_unseen"][0]]
    )
    parent_parity = _parent_parity_stats(parity_encoded, parity_parent)

    split_indices = {
        "train": split.train_indices,
        "test_seen": split.test_seen_indices,
        "test_unseen": split.test_unseen_indices,
    }
    output_tensors = {
        OUTPUT_FILES[name]: all_features.index_select(0, indices)
        for name, indices in split_indices.items()
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, tensor in output_tensors.items():
        _atomic_torch_save(output_dir / filename, tensor)
    output_sha = {name: sha256_file(output_dir / name) for name in output_tensors}

    clip_source = Path(clip.__file__).resolve().with_name("clip.py")
    distribution = importlib.metadata.distribution("clip")
    direct_url_text = distribution.read_text("direct_url.json")
    clip_runtime = {
        "python_source": str(clip_source),
        "python_source_sha256": sha256_file(clip_source),
        "distribution_version": distribution.version,
        "distribution_direct_url": json.loads(direct_url_text) if direct_url_text else None,
    }
    manifest = build_manifest(
        dataset=config["dataset"],
        code_commit=current_code_commit(),
        script_sha256=sha256_file(Path(__file__)),
        source_config=config_path,
        source_config_sha256=config_sha,
        parent_manifest=parent_manifest_path,
        parent_manifest_sha256=parent_sha,
        parent_asset_id=str(parent.get("asset_id")),
        clip_checkpoint=paths["clip_checkpoint"],
        clip_checkpoint_sha256=input_sha["clip_checkpoint"],
        clip_runtime=clip_runtime,
        counts={name: int(indices.numel()) for name, indices in split_indices.items()},
        class_order_sha=class_order_sha256(split.class_names),
        raw_image_order_sha=image_order_sha,
        inputs_sha256=input_sha,
        outputs_sha256=output_sha,
        output_stats={name: _tensor_stats(tensor) for name, tensor in output_tensors.items()},
        determinism_check=determinism,
        parent_parity=parent_parity,
    )
    manifest_path = output_dir / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["asset_manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.parent_manifest,
                args.output_dir,
                device_name=args.device,
                batch_size=args.batch_size,
                workers=args.workers,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
