"""Build physically isolated CUB 100/50 masked assets for RCEG Gate 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.gzsl_data import load_xlsa_split, resolve_xlsa_image_path
from tools.prepare_cub_standard_validation import build_standard_validation_split
from tools.prepare_paper_clip_assets import (
    OFFICIAL_CHECKPOINT_SHA256,
    _ImageDataset,
    _verify_file,
    load_source_config,
)
from tools.reproducibility import configure_reproducibility
from tools.run_contract import current_code_commit, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.rceg-masked-bundle.v1"
SUBSET_SCHEMA = "gzsl-paper.rceg-masked-subset.v1"
GRID = 24
PATCH_SIZE = 14
MASK_COUNT = 4


def interleaved_masks(device: torch.device | str = "cpu") -> torch.Tensor:
    rows = torch.arange(GRID).view(GRID, 1)
    cols = torch.arange(GRID).view(1, GRID)
    ids = 2 * (rows.remainder(2)) + cols.remainder(2)
    return torch.stack([ids.eq(index).reshape(-1) for index in range(MASK_COUNT)]).to(device)


@torch.no_grad()
def encode_rceg_batch(model, images: torch.Tensor):
    visual = model.visual
    images = images.float()
    masks = interleaved_masks(images.device)
    patch = visual.conv1(images).reshape(images.size(0), visual.conv1.out_channels, -1).permute(0, 2, 1)
    patch = F.normalize(patch.float(), dim=-1)
    targets = torch.stack(
        [F.normalize(patch[:, mask].mean(dim=1), dim=-1) for mask in masks], dim=1
    )
    image_cls = F.normalize(model.encode_image(images).float(), dim=-1)
    masked_cls, visible_rows = [], []
    for mask in masks:
        pixel_mask = mask.view(GRID, GRID).repeat_interleave(PATCH_SIZE, 0).repeat_interleave(PATCH_SIZE, 1)
        masked = images.clone()
        masked[:, :, pixel_mask] = 0.0
        x = visual.conv1(masked)
        x = x.reshape(x.size(0), x.size(1), -1).permute(0, 2, 1)
        class_token = visual.class_embedding + torch.zeros(
            x.size(0), 1, x.size(-1), dtype=x.dtype, device=x.device
        )
        x = torch.cat((class_token, x), dim=1)
        x = visual.ln_pre(x + visual.positional_embedding)
        x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        x = F.normalize(x.float(), dim=-1)
        masked_cls.append(x[:, 0])
        visible_rows.append(x[:, 1:][:, ~mask])
    masked_cls = torch.stack(masked_cls, dim=1)
    visible = torch.stack(visible_rows, dim=1)
    expected = (
        (images.size(0), 768),
        (images.size(0), MASK_COUNT, 768),
        (images.size(0), MASK_COUNT, 432, 768),
        (images.size(0), MASK_COUNT, 1024),
    )
    actual = (tuple(image_cls.shape), tuple(masked_cls.shape), tuple(visible.shape), tuple(targets.shape))
    if actual != expected or not all(torch.isfinite(value).all() for value in (image_cls, masked_cls, visible, targets)):
        raise RuntimeError(f"RCEG CLIP输出错误：actual={actual}, expected={expected}")
    return image_cls, masked_cls, visible, targets


def _atomic_save(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_subset(
    *, name: str, indices: torch.Tensor, class_ids: torch.Tensor, output_root: Path,
    model, preprocess, image_paths, labels, name_embeddings, role_embeddings,
    device: torch.device, batch_size: int, workers: int, common_identity: dict,
):
    subset = output_root / name
    subset.mkdir(parents=True)
    ordered = indices.cpu().long().sort().values
    loader = DataLoader(
        _ImageDataset([image_paths[int(index)] for index in ordered], preprocess),
        batch_size=int(batch_size), shuffle=False, num_workers=int(workers),
        pin_memory=device.type == "cuda",
    )
    visible_path = subset / "visible_tokens.npy"
    visible_map = np.lib.format.open_memmap(
        visible_path, mode="w+", dtype=np.float16,
        shape=(ordered.numel(), MASK_COUNT, 432, 768),
    )
    cls_rows, masked_rows, target_rows = [], [], []
    cursor = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True).float()
        image_cls, masked_cls, visible, target = encode_rceg_batch(model, batch)
        count = batch.size(0)
        visible_map[cursor:cursor + count] = visible.cpu().numpy().astype(np.float16)
        cls_rows.append(image_cls.cpu())
        masked_rows.append(masked_cls.cpu())
        target_rows.append(target.cpu())
        cursor += count
    visible_map.flush()
    del visible_map
    if cursor != ordered.numel():
        raise RuntimeError(f"RCEG {name}写入计数错误。")
    tensors = {
        "image_cls.pt": torch.cat(cls_rows),
        "masked_cls.pt": torch.cat(masked_rows),
        "target.pt": torch.cat(target_rows),
        "labels.pt": labels.index_select(0, ordered),
        "raw_indices.pt": ordered,
        "class_ids.pt": class_ids.cpu().long(),
        "name_embeddings.pt": name_embeddings.index_select(0, class_ids.cpu().long()),
        "role_embeddings.pt": role_embeddings.index_select(0, class_ids.cpu().long()),
    }
    for filename, value in tensors.items():
        _atomic_save(subset / filename, value)
    outputs = {filename: sha256_file(subset / filename) for filename in tensors}
    outputs[visible_path.name] = sha256_file(visible_path)
    order_payload = [
        {"raw_index": int(index), "path": str(image_paths[int(index)]).replace("\\", "/")}
        for index in ordered
    ]
    manifest = {
        "schema_version": SUBSET_SCHEMA,
        "subset": name,
        "count": int(ordered.numel()),
        "class_ids": [int(value) for value in class_ids],
        "visible_shape": [MASK_COUNT, 432, 768],
        "visible_dtype": "float16",
        "target_shape": [MASK_COUNT, 1024],
        "target_dtype": "float32",
        "image_order_sha256": hashlib.sha256(
            json.dumps(order_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "common_identity": common_identity,
        "outputs_sha256": outputs,
    }
    manifest_path = subset / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path), "count": int(ordered.numel())}


def run(
    config_path: Path, output_dir: Path, *, device_name: str, batch_size: int,
    workers: int, text_manifest_path: Path, text_manifest_sha256: str,
):
    require_clean_code_tree()
    if output_dir.exists():
        raise FileExistsError(f"RCEG资产输出目录必须不存在：{output_dir}")
    config, config_sha = load_source_config(config_path)
    paths = {name: Path(config[name]) for name in (
        "raw_root", "raw_archive", "res101", "att_splits", "role_texts", "clip_checkpoint"
    )}
    input_sha = {
        name: _verify_file(paths[name], config["expected_sha256"][name], name)
        for name in ("raw_archive", "res101", "att_splits", "role_texts", "clip_checkpoint")
    }
    if input_sha["clip_checkpoint"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("RCEG只允许OpenAI官方ViT-L/14@336 checkpoint。")
    if not text_manifest_path.is_file() or sha256_file(text_manifest_path) != text_manifest_sha256:
        raise ValueError("RCEG text-v2 manifest路径或SHA错误。")
    text_manifest = json.loads(text_manifest_path.read_text(encoding="utf-8"))
    text_outputs = text_manifest.get("outputs_sha256", {})
    text_tensors = {}
    for key, filename in (("name", "class_name_embeds.pt"), ("role", "role_sentence_embeds.pt")):
        path = text_manifest_path.parent / filename
        if not path.is_file() or sha256_file(path) != text_outputs.get(filename):
            raise ValueError(f"RCEG文本资产错误：{filename}")
        text_tensors[key] = torch.load(path, map_location="cpu", weights_only=True).float()
    split = load_xlsa_split(paths["res101"], paths["att_splits"])
    if text_tensors["name"].shape != (200, 768) or text_tensors["role"].shape != (200, 8, 768):
        raise ValueError("RCEG文本张量形状错误。")
    validation = build_standard_validation_split(
        paths["res101"], paths["att_splits"], text_manifest_path.parent / "train_labels.pt"
    )
    train_indices = torch.cat((validation["fit_raw_indices"], validation["val_seen_raw_indices"])).sort().values
    eval_indices = validation["val_unseen_raw_indices"].sort().values
    if train_indices.numel() != 4702 or eval_indices.numel() != 2355:
        raise ValueError("RCEG开发划分必须是4702/2355。")
    train_classes = validation["dev_seen_classes"]
    active_classes = torch.cat((train_classes, validation["dev_unseen_classes"])).sort().values
    image_paths = [
        resolve_xlsa_image_path(paths["raw_root"], value, config["image_path_anchors"])
        for value in split.image_files
    ]
    configure_reproducibility(20260901, strict_determinism=True, deterministic_warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import clip
    device = torch.device(device_name)
    model, preprocess = clip.load(str(paths["clip_checkpoint"]), device=device, jit=False)
    model = model.float().eval()
    output_dir.mkdir(parents=True)
    common_identity = {
        "schema_version": SCHEMA,
        "code_commit": current_code_commit(),
        "script_sha256": sha256_file(Path(__file__)),
        "source_config_sha256": config_sha,
        "inputs_sha256": input_sha,
        "text_manifest_sha256": text_manifest_sha256,
        "checkpoint_sha256": input_sha["clip_checkpoint"],
        "mask_definition": "24x24 interleaved (row%2,column%2), four groups of 144",
        "target_definition": "L2-normalized original visual.conv1 patch mean, width1024",
        "visible_definition": "masked CLIP final projected non-target tokens, L2-normalized",
        "autocast": False, "tf32_matmul": False, "tf32_cudnn": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "batch_size": int(batch_size), "workers": int(workers),
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda or ""),
        "cudnn": str(torch.backends.cudnn.version()),
    }
    identity_seed = {
        "common_identity": common_identity,
        "train_classes": [int(value) for value in train_classes],
        "active_classes": [int(value) for value in active_classes],
    }
    common_identity["bundle_id"] = hashlib.sha256(
        json.dumps(identity_seed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    subsets = {
        "dev_train": _write_subset(
            name="dev_train", indices=train_indices, class_ids=train_classes,
            output_root=output_dir, model=model, preprocess=preprocess,
            image_paths=image_paths, labels=split.labels,
            name_embeddings=text_tensors["name"], role_embeddings=text_tensors["role"],
            device=device, batch_size=batch_size, workers=workers,
            common_identity=common_identity,
        ),
        "dev_eval": _write_subset(
            name="dev_eval", indices=eval_indices, class_ids=active_classes,
            output_root=output_dir, model=model, preprocess=preprocess,
            image_paths=image_paths, labels=split.labels,
            name_embeddings=text_tensors["name"], role_embeddings=text_tensors["role"],
            device=device, batch_size=batch_size, workers=workers,
            common_identity=common_identity,
        ),
    }
    root = {"schema_version": SCHEMA, "mode": "dev", "common_identity": common_identity, "subsets": subsets}
    root_path = output_dir / "asset_manifest.json"
    root_path.write_text(json.dumps(root, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**root, "asset_manifest_sha256": sha256_file(root_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--text-manifest", type=Path, required=True)
    parser.add_argument("--text-manifest-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.config, args.output_dir, device_name=args.device,
        batch_size=args.batch_size, workers=args.workers,
        text_manifest_path=args.text_manifest,
        text_manifest_sha256=args.text_manifest_sha,
    ), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
