"""Build physically isolated unmasked CLIP token assets for OREF Gate 0."""

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
    OFFICIAL_CHECKPOINT_SHA256, _ImageDataset, _verify_file, load_source_config,
)
from tools.reproducibility import configure_reproducibility
from tools.run_contract import current_code_commit, require_clean_code_tree
from tools.runtime import sha256_file


BUNDLE_SCHEMA = "gzsl-paper.oref-visible-bundle.v1"
SUBSET_SCHEMA = "gzsl-paper.oref-visible-subset.v1"


@torch.no_grad()
def encode_visible(model, images):
    visual = model.visual
    images = images.float()
    x = visual.conv1(images)
    x = x.reshape(x.size(0), x.size(1), -1).permute(0, 2, 1)
    cls = visual.class_embedding + torch.zeros(
        x.size(0), 1, x.size(-1), dtype=x.dtype, device=x.device
    )
    x = torch.cat((cls, x), 1)
    x = visual.ln_pre(x + visual.positional_embedding)
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    x = F.normalize(x.float(), dim=-1)
    reference = F.normalize(model.encode_image(images).float(), dim=-1)
    parity = (x[:, 0] * reference).sum(-1)
    if x.shape[1:] != (577, 768) or float(parity.min()) < 0.9998 or not torch.isfinite(x).all():
        raise RuntimeError("OREF CLIP token/CLS parity错误。")
    return x[:, 0], x[:, 1:], parity


def _save(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_subset(
    *, name, indices, class_ids, output_root, model, preprocess, image_paths,
    labels, names, roles, device, batch_size, workers, common,
):
    directory = output_root / name
    directory.mkdir(parents=True)
    ordered = indices.long().cpu().sort().values
    loader = DataLoader(
        _ImageDataset([image_paths[int(index)] for index in ordered], preprocess),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    token_path = directory / "patch_tokens.npy"
    token_map = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=np.float16,
        shape=(ordered.numel(), 576, 768),
    )
    cls_rows, parity_rows, cursor = [], [], 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True).float()
        cls, tokens, parity = encode_visible(model, batch)
        count = batch.size(0)
        token_map[cursor:cursor + count] = tokens.cpu().numpy().astype(np.float16)
        cls_rows.append(cls.cpu())
        parity_rows.append(parity.cpu())
        cursor += count
    token_map.flush()
    del token_map
    if cursor != ordered.numel():
        raise RuntimeError("OREF subset写入计数错误。")
    tensors = {
        "image_cls.pt": torch.cat(cls_rows),
        "labels.pt": labels.index_select(0, ordered),
        "raw_indices.pt": ordered,
        "class_ids.pt": class_ids.long().cpu(),
        "name_embeddings.pt": names.index_select(0, class_ids.long().cpu()),
        "role_embeddings.pt": roles.index_select(0, class_ids.long().cpu()),
    }
    for filename, value in tensors.items():
        _save(directory / filename, value)
    outputs = {filename: sha256_file(directory / filename) for filename in tensors}
    outputs[token_path.name] = sha256_file(token_path)
    order = [
        {"raw_index": int(index), "path": str(image_paths[int(index)]).replace("\\", "/")}
        for index in ordered
    ]
    manifest = {
        "schema_version": SUBSET_SCHEMA, "subset": name,
        "count": int(ordered.numel()), "class_ids": [int(x) for x in class_ids],
        "patch_shape": [576, 768], "patch_dtype": "float16",
        "cls_parity_min": float(torch.cat(parity_rows).min()),
        "image_order_sha256": hashlib.sha256(
            json.dumps(order, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "common_identity": common, "outputs_sha256": outputs,
    }
    path = directory / "asset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "count": int(ordered.numel())}


def run(config_path, output_dir, *, device_name, batch_size, workers, text_manifest, text_manifest_sha):
    require_clean_code_tree()
    if output_dir.exists():
        raise FileExistsError(f"OREF资产目录已存在：{output_dir}")
    config, config_sha = load_source_config(config_path)
    paths = {key: Path(config[key]) for key in (
        "raw_root", "raw_archive", "res101", "att_splits", "role_texts", "clip_checkpoint"
    )}
    inputs = {key: _verify_file(paths[key], config["expected_sha256"][key], key) for key in (
        "raw_archive", "res101", "att_splits", "role_texts", "clip_checkpoint"
    )}
    if inputs["clip_checkpoint"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("OREF checkpoint错误。")
    if not text_manifest.is_file() or sha256_file(text_manifest) != text_manifest_sha:
        raise ValueError("OREF text-v2 manifest错误。")
    text_meta = json.loads(text_manifest.read_text(encoding="utf-8"))
    text_values = {}
    for key, filename in (("names", "class_name_embeds.pt"), ("roles", "role_sentence_embeds.pt")):
        path = text_manifest.parent / filename
        if not path.is_file() or sha256_file(path) != text_meta["outputs_sha256"].get(filename):
            raise ValueError(f"OREF文本tensor错误：{filename}")
        text_values[key] = torch.load(path, map_location="cpu", weights_only=True).float()
    split = load_xlsa_split(paths["res101"], paths["att_splits"])
    validation = build_standard_validation_split(
        paths["res101"], paths["att_splits"], text_manifest.parent / "train_labels.pt"
    )
    train_indices = torch.cat((validation["fit_raw_indices"], validation["val_seen_raw_indices"])).sort().values
    eval_indices = validation["val_unseen_raw_indices"].sort().values
    train_classes = validation["dev_seen_classes"]
    active = torch.cat((train_classes, validation["dev_unseen_classes"])).sort().values
    if train_indices.numel() != 4702 or eval_indices.numel() != 2355:
        raise ValueError("OREF开发划分必须是4702/2355。")
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
    common = {
        "schema_version": BUNDLE_SCHEMA, "code_commit": current_code_commit(),
        "script_sha256": sha256_file(Path(__file__)), "source_config_sha256": config_sha,
        "inputs_sha256": inputs, "text_manifest_sha256": text_manifest_sha,
        "checkpoint_sha256": inputs["clip_checkpoint"],
        "token_definition": "unmasked final projected CLIP patch tokens, L2-normalized",
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "batch_size": batch_size, "workers": workers, "autocast": False,
        "tf32_matmul": False, "tf32_cudnn": False,
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda or ""),
        "cudnn": str(torch.backends.cudnn.version()),
    }
    common["bundle_id"] = hashlib.sha256(json.dumps({
        "common": common, "train_classes": [int(x) for x in train_classes],
        "active": [int(x) for x in active],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    subsets = {
        "dev_train": _write_subset(
            name="dev_train", indices=train_indices, class_ids=train_classes,
            output_root=output_dir, model=model, preprocess=preprocess,
            image_paths=image_paths, labels=split.labels, names=text_values["names"],
            roles=text_values["roles"], device=device, batch_size=batch_size,
            workers=workers, common=common,
        ),
        "dev_eval": _write_subset(
            name="dev_eval", indices=eval_indices, class_ids=active,
            output_root=output_dir, model=model, preprocess=preprocess,
            image_paths=image_paths, labels=split.labels, names=text_values["names"],
            roles=text_values["roles"], device=device, batch_size=batch_size,
            workers=workers, common=common,
        ),
    }
    root = {"schema_version": BUNDLE_SCHEMA, "mode": "dev", "common_identity": common, "subsets": subsets}
    root_path = output_dir / "asset_manifest.json"
    root_path.write_text(json.dumps(root, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**root, "asset_manifest_sha256": sha256_file(root_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--text-manifest", type=Path, required=True)
    parser.add_argument("--text-manifest-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.config, args.output_dir, device_name=args.device,
        batch_size=args.batch_size, workers=args.workers,
        text_manifest=args.text_manifest, text_manifest_sha=args.text_manifest_sha,
    ), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
