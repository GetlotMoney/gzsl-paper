"""Build traceable OpenAI CLIP ViT-L/14@336px assets for CUB/AWA2/SUN."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tools.gzsl_data import (
    class_order_sha256,
    clean_class_name,
    load_xlsa_split,
    resolve_xlsa_image_path,
)
from tools.runtime import sha256_file


MODEL_NAME = "ViT-L/14@336px"
OFFICIAL_CHECKPOINT_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
SOURCE_KEYS = {
    "schema_version",
    "dataset",
    "raw_root",
    "raw_archive",
    "image_path_anchors",
    "res101",
    "att_splits",
    "role_texts",
    "clip_checkpoint",
    "expected_sha256",
}


def load_source_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != SOURCE_KEYS:
        actual = set(config) if isinstance(config, dict) else set()
        raise ValueError(
            f"资产配置字段错误；缺少={sorted(SOURCE_KEYS-actual)}，多出={sorted(actual-SOURCE_KEYS)}。"
        )
    if config["schema_version"] != "gzsl-paper.clip-asset-source.v1":
        raise ValueError("资产配置schema错误。")
    if config["dataset"] not in ("CUB", "AWA2", "SUN"):
        raise ValueError("资产配置dataset只允许CUB/AWA2/SUN。")
    if not isinstance(config["image_path_anchors"], list) or not config["image_path_anchors"]:
        raise ValueError("image_path_anchors必须是非空列表。")
    expected = config["expected_sha256"]
    if set(expected) != {"raw_archive", "res101", "att_splits", "role_texts", "clip_checkpoint"}:
        raise ValueError("资产输入SHA字段不完整。")
    return config, sha256_file(path)


def _verify_file(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"缺少{name}：{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA不匹配：{actual}")
    return actual


def load_role_texts(path: Path, dataset: str, class_names: tuple[str, ...]) -> tuple[list[str], list[list[str]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "dataset",
        "class_order_sha256",
        "role_names",
        "generator",
        "descriptions",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("八角色原文文件字段错误。")
    if payload["schema_version"] != "gzsl-paper.role-texts.v1" or payload["dataset"] != dataset:
        raise ValueError("八角色原文身份错误。")
    if payload["class_order_sha256"] != class_order_sha256(class_names):
        raise ValueError("八角色原文类别顺序SHA不一致。")
    roles = payload["role_names"]
    descriptions = payload["descriptions"]
    if len(roles) != 8 or len(descriptions) != len(class_names):
        raise ValueError("八角色原文必须逐类包含8句。")
    for class_id, rows in enumerate(descriptions):
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError(f"类别{class_id}没有8句描述。")
        for sentence in rows:
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError(f"类别{class_id}包含空描述。")
            if len(sentence.split()) > 35:
                raise ValueError(f"类别{class_id}描述超过35词。")
    return roles, descriptions, payload["generator"]


class _ImageDataset(Dataset):
    def __init__(self, paths: list[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


def _encode_images(model, preprocess, paths: list[Path], device: torch.device, batch_size: int, workers: int) -> torch.Tensor:
    loader = DataLoader(
        _ImageDataset(paths, preprocess),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
    )
    rows = []
    with torch.inference_mode():
        for images in loader:
            encoded = model.encode_image(images.to(device, non_blocking=True))
            rows.append(F.normalize(encoded.float(), dim=-1).cpu())
    result = torch.cat(rows)
    if tuple(result.shape) != (len(paths), 768) or not torch.isfinite(result).all():
        raise RuntimeError("图像CLIP缓存形状或有限性检查失败。")
    return result


def _encode_texts(model, clip_module, texts: list[str], device: torch.device, batch_size: int = 256) -> torch.Tensor:
    rows = []
    for start in range(0, len(texts), int(batch_size)):
        tokens = clip_module.tokenize(texts[start : start + int(batch_size)], truncate=False).to(device)
        with torch.inference_mode():
            encoded = model.encode_text(tokens)
        rows.append(F.normalize(encoded.float(), dim=-1).cpu())
    result = torch.cat(rows)
    if result.ndim != 2 or result.size(1) != 768 or not torch.isfinite(result).all():
        raise RuntimeError("文本CLIP缓存形状或有限性检查失败。")
    return result


def _atomic_torch_save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run(config_path: Path, output_dir: Path, *, device_name: str, batch_size: int, workers: int) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"资产输出目录必须不存在：{output_dir}")
    config, config_sha = load_source_config(config_path)
    paths = {
        "raw_root": Path(config["raw_root"]),
        "raw_archive": Path(config["raw_archive"]),
        "res101": Path(config["res101"]),
        "att_splits": Path(config["att_splits"]),
        "role_texts": Path(config["role_texts"]),
        "clip_checkpoint": Path(config["clip_checkpoint"]),
    }
    input_sha = {
        name: _verify_file(paths[name], config["expected_sha256"][name], name)
        for name in ("raw_archive", "res101", "att_splits", "role_texts", "clip_checkpoint")
    }
    if input_sha["clip_checkpoint"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("CLIP checkpoint不是OpenAI ViT-L/14@336px官方权重。")
    split = load_xlsa_split(paths["res101"], paths["att_splits"])
    roles, role_rows, generator_identity = load_role_texts(
        paths["role_texts"], config["dataset"], split.class_names
    )
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

    import clip

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA生成缓存，但CUDA不可用。")
    model, preprocess = clip.load(str(paths["clip_checkpoint"]), device=device, jit=False)
    model.eval()
    all_features = _encode_images(model, preprocess, image_paths, device, batch_size, workers)
    repeat_a = _encode_images(model, preprocess, [image_paths[0]], device, 1, 0)
    repeat_b = _encode_images(model, preprocess, [image_paths[0]], device, 1, 0)
    if not torch.equal(repeat_a, repeat_b) or not torch.allclose(
        repeat_a[0], all_features[0], atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("固定样本重复CLIP提取未逐值一致。")

    clean_names = [clean_class_name(name) for name in split.class_names]
    class_prompts = [f"a photo of a {name}." for name in clean_names]
    class_name_embeds = _encode_texts(model, clip, class_prompts, device)
    flattened_roles = [sentence for rows in role_rows for sentence in rows]
    role_embeds = _encode_texts(model, clip, flattened_roles, device).reshape(split.class_count, 8, 768)

    output_dir.mkdir(parents=True)
    tensors = {
        "train_features.pt": all_features.index_select(0, split.train_indices),
        "train_labels.pt": split.labels.index_select(0, split.train_indices),
        "test_seen_features.pt": all_features.index_select(0, split.test_seen_indices),
        "test_seen_labels.pt": split.labels.index_select(0, split.test_seen_indices),
        "test_unseen_features.pt": all_features.index_select(0, split.test_unseen_indices),
        "test_unseen_labels.pt": split.labels.index_select(0, split.test_unseen_indices),
        "class_name_embeds.pt": class_name_embeds,
        "role_sentence_embeds.pt": role_embeds,
    }
    for filename, tensor in tensors.items():
        _atomic_torch_save(output_dir / filename, tensor)
    (output_dir / "class_names.json").write_text(
        json.dumps(
            {"xlsa": list(split.class_names), "display": clean_names, "prompts": class_prompts},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = {filename: sha256_file(output_dir / filename) for filename in tensors}
    outputs["class_names.json"] = sha256_file(output_dir / "class_names.json")
    clip_source = Path(clip.__file__).resolve().with_name("clip.py")
    distribution = importlib.metadata.distribution("clip")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    manifest = {
        "schema_version": "gzsl-paper.clip-assets.v1",
        "dataset": config["dataset"],
        "asset_id": hashlib.sha256(
            json.dumps({"inputs": input_sha, "outputs": outputs}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
        "source_config_sha256": config_sha,
        "model": MODEL_NAME,
        "clip_checkpoint_sha256": input_sha["clip_checkpoint"],
        "clip_python_source": str(clip_source),
        "clip_python_source_sha256": sha256_file(clip_source),
        "clip_distribution_version": distribution.version,
        "clip_distribution_direct_url": direct_url,
        "preprocess": {
            "resolution": 336,
            "resize": "bicubic_shorter_side",
            "crop": "center_336",
            "rgb": True,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
        "class_count": split.class_count,
        "seen_class_count": int(split.seen_classes.numel()),
        "unseen_class_count": int(split.unseen_classes.numel()),
        "image_count": int(split.labels.numel()),
        "raw_image_order_and_size_sha256": image_order_sha,
        "train_count": int(split.train_indices.numel()),
        "test_seen_count": int(split.test_seen_indices.numel()),
        "test_unseen_count": int(split.test_unseen_indices.numel()),
        "seen_classes": [int(value) for value in split.seen_classes],
        "unseen_classes": [int(value) for value in split.unseen_classes],
        "class_order_sha256": class_order_sha256(split.class_names),
        "role_names": roles,
        "role_text_generator": generator_identity,
        "inputs_sha256": input_sha,
        "outputs_sha256": outputs,
    }
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["asset_manifest_sha256"] = sha256_file(output_dir / "asset_manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
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
