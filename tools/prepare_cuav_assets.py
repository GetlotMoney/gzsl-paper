"""Build CUAV full-view and fixed 25-crop assets on the CUB 100/50 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tools.gzsl_data import load_xlsa_split, resolve_xlsa_image_path
from tools.prepare_cub_standard_validation import build_standard_validation_split
from tools.prepare_paper_clip_assets import OFFICIAL_CHECKPOINT_SHA256, _verify_file, load_source_config
from tools.reproducibility import configure_reproducibility
from tools.run_contract import current_code_commit, require_clean_code_tree
from tools.runtime import sha256_file


STARTS = [0, 4, 9, 14, 18]
WINDOWS = [(r, c) for r in STARTS for c in STARTS]
WINDOW_SHA = "4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb"
SIDE = 6
PATCH_PIXELS = 14
BUNDLE_SCHEMA = "gzsl-paper.cuav-crop-bundle.v1"
SUBSET_SCHEMA = "gzsl-paper.cuav-crop-subset.v1"


def raw_crop_with_box(image: Image.Image, window):
    width, height = image.size
    scale = 336.0 / min(width, height)
    resized_width, resized_height = width * scale, height * scale
    offset_x, offset_y = (resized_width - 336.0) / 2.0, (resized_height - 336.0) / 2.0
    row, column = window
    left = (column * PATCH_PIXELS + offset_x) / scale
    top = (row * PATCH_PIXELS + offset_y) / scale
    right = ((column + SIDE) * PATCH_PIXELS + offset_x) / scale
    bottom = ((row + SIDE) * PATCH_PIXELS + offset_y) / scale
    box = (
        max(0, min(int(math.floor(left)), width - 1)),
        max(0, min(int(math.floor(top)), height - 1)),
        max(1, min(int(math.ceil(right)), width)),
        max(1, min(int(math.ceil(bottom)), height)),
    )
    box = (box[0], box[1], max(box[0] + 1, box[2]), max(box[1] + 1, box[3]))
    return image.crop(box), box


class CropDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as handle:
            image = handle.convert("RGB")
            tensors = [self.preprocess(image)]
            boxes = []
            for window in WINDOWS:
                crop, box = raw_crop_with_box(image, window)
                tensors.append(self.preprocess(crop))
                boxes.append(box)
        return torch.stack(tensors), torch.tensor(boxes, dtype=torch.int64)


def _save(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def encode_subset(model, preprocess, paths, *, device, batch_size, workers, include_lowres):
    loader = DataLoader(
        CropDataset(paths, preprocess), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )
    full_rows, crop_rows, box_rows, lowres_rows = [], [], [], []
    for views, boxes in loader:
        batch, count = views.shape[:2]
        features = F.normalize(
            model.encode_image(views.reshape(batch * count, 3, 336, 336).to(device).float()).float(),
            dim=-1,
        ).reshape(batch, count, 768).cpu()
        full_rows.append(features[:, 0])
        crop_rows.append(features[:, 1:])
        box_rows.append(boxes)
        if include_lowres:
            full_tensor = views[:, 0]
            lowres = []
            for row, column in WINDOWS:
                crop = full_tensor[:, :, row*PATCH_PIXELS:(row+SIDE)*PATCH_PIXELS, column*PATCH_PIXELS:(column+SIDE)*PATCH_PIXELS]
                lowres.append(F.interpolate(crop, size=(336, 336), mode="bicubic", align_corners=False))
            lowres = torch.stack(lowres, dim=1)
            encoded = F.normalize(
                model.encode_image(lowres.reshape(batch * 25, 3, 336, 336).to(device).float()).float(),
                dim=-1,
            ).reshape(batch, 25, 768).cpu()
            lowres_rows.append(encoded)
    return (
        torch.cat(full_rows), torch.cat(crop_rows), torch.cat(box_rows),
        torch.cat(lowres_rows) if lowres_rows else None,
    )


def _write_manifest(directory, *, subset, indices, class_ids, full_cls, crop_features, lowres_features, boxes, labels, names, paths, common, include_crops):
    directory.mkdir(parents=True)
    tensors = {
        "full_cls.pt": full_cls.float(), "labels.pt": labels.index_select(0, indices).long(),
        "raw_indices.pt": indices.long(), "class_ids.pt": class_ids.long(),
        "name_embeddings.pt": names.index_select(0, class_ids.long()).float(),
        "crop_boxes.pt": boxes.long(),
    }
    for filename, value in tensors.items():
        _save(directory / filename, value)
    outputs = {filename: sha256_file(directory / filename) for filename in tensors}
    if include_crops:
        crop_path = directory / "crop_features.npy"
        array = np.lib.format.open_memmap(
            crop_path, mode="w+", dtype=np.float16, shape=tuple(crop_features.shape)
        )
        array[:] = crop_features.numpy().astype(np.float16)
        array.flush(); del array
        outputs[crop_path.name] = sha256_file(crop_path)
    if lowres_features is not None:
        lowres_path = directory / "lowres_crop_features.npy"
        array = np.lib.format.open_memmap(
            lowres_path, mode="w+", dtype=np.float16, shape=tuple(lowres_features.shape)
        )
        array[:] = lowres_features.numpy().astype(np.float16)
        array.flush(); del array
        outputs[lowres_path.name] = sha256_file(lowres_path)
    relative = [str(path).replace("\\", "/") for path in paths]
    paths_file = directory / "image_paths.json"
    paths_file.write_text(json.dumps(relative, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs[paths_file.name] = sha256_file(paths_file)
    order_sha = hashlib.sha256(json.dumps([
        {"raw_index": int(index), "path": path}
        for index, path in zip(indices.tolist(), relative, strict=True)
    ], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "schema_version": SUBSET_SCHEMA, "subset": subset,
        "count": int(indices.numel()), "class_ids": [int(x) for x in class_ids],
        "crop_actions": WINDOWS, "crop_action_sha256": WINDOW_SHA,
        "crop_features_present": include_crops,
        "lowres_crop_features_present": lowres_features is not None,
        "image_order_sha256": order_sha, "common_identity": common,
        "outputs_sha256": outputs,
    }
    manifest_path = directory / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path), "count": int(indices.numel())}


def run(config_path, output_dir, *, device_name, batch_size, workers, text_manifest, text_manifest_sha):
    require_clean_code_tree()
    if output_dir.exists():
        raise FileExistsError(f"CUAV资产目录已存在：{output_dir}")
    config, config_sha = load_source_config(config_path)
    paths = {key: Path(config[key]) for key in (
        "raw_root", "raw_archive", "res101", "att_splits", "role_texts", "clip_checkpoint"
    )}
    inputs = {key: _verify_file(paths[key], config["expected_sha256"][key], key) for key in (
        "raw_archive", "res101", "att_splits", "role_texts", "clip_checkpoint"
    )}
    if inputs["clip_checkpoint"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("CUAV checkpoint错误。")
    if not text_manifest.is_file() or sha256_file(text_manifest) != text_manifest_sha:
        raise ValueError("CUAV text manifest错误。")
    text_meta = json.loads(text_manifest.read_text(encoding="utf-8"))
    name_path = text_manifest.parent / "class_name_embeds.pt"
    if sha256_file(name_path) != text_meta["outputs_sha256"].get(name_path.name):
        raise ValueError("CUAV name embeddings错误。")
    names = torch.load(name_path, map_location="cpu", weights_only=True).float()
    split = load_xlsa_split(paths["res101"], paths["att_splits"])
    validation = build_standard_validation_split(
        paths["res101"], paths["att_splits"], text_manifest.parent / "train_labels.pt"
    )
    train_indices = torch.cat((validation["fit_raw_indices"], validation["val_seen_raw_indices"])).sort().values
    eval_indices = validation["val_unseen_raw_indices"].sort().values
    train_classes = validation["dev_seen_classes"]
    active = torch.cat((train_classes, validation["dev_unseen_classes"])).sort().values
    image_paths = [
        resolve_xlsa_image_path(paths["raw_root"], value, config["image_path_anchors"])
        for value in split.image_files
    ]
    configure_reproducibility(20260901, strict_determinism=True, deterministic_warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import clip
    device = torch.device(device_name)
    model, preprocess = clip.load(str(paths["clip_checkpoint"]), device=device, jit=False)
    model = model.float().eval()
    output_dir.mkdir(parents=True)
    common = {
        "schema_version": BUNDLE_SCHEMA, "code_commit": current_code_commit(),
        "script_sha256": sha256_file(Path(__file__)), "source_config_sha256": config_sha,
        "inputs_sha256": inputs, "text_manifest_sha256": text_manifest_sha,
        "checkpoint_sha256": inputs["clip_checkpoint"], "crop_action_sha256": WINDOW_SHA,
        "crop_definition": "IDEA-172 6x6 patches; starts 0,4,9,14,18; row-major25",
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "batch_size": batch_size, "workers": workers,
    }
    common["bundle_id"] = hashlib.sha256(json.dumps({
        "common": common, "train": [int(x) for x in train_classes], "active": [int(x) for x in active]
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    train_paths = [image_paths[int(i)] for i in train_indices]
    eval_paths = [image_paths[int(i)] for i in eval_indices]
    train_full, train_crops, train_boxes, _ = encode_subset(
        model, preprocess, train_paths, device=device, batch_size=batch_size, workers=workers,
        include_lowres=False,
    )
    eval_full, eval_crops, eval_boxes, eval_lowres = encode_subset(
        model, preprocess, eval_paths, device=device, batch_size=batch_size, workers=workers,
        include_lowres=True,
    )
    subsets = {
        "dev_train": _write_manifest(
            output_dir / "dev_train", subset="dev_train", indices=train_indices,
            class_ids=train_classes, full_cls=train_full, crop_features=train_crops,
            boxes=train_boxes, labels=split.labels, names=names, paths=train_paths,
            common=common, include_crops=True, lowres_features=None,
        ),
        "dev_eval": _write_manifest(
            output_dir / "dev_eval", subset="dev_eval", indices=eval_indices,
            class_ids=active, full_cls=eval_full, crop_features=eval_crops,
            boxes=eval_boxes, labels=split.labels, names=names, paths=eval_paths,
            common=common, include_crops=False, lowres_features=eval_lowres,
        ),
        "dev_eval_oracle": _write_manifest(
            output_dir / "dev_eval_oracle", subset="dev_eval_oracle", indices=eval_indices,
            class_ids=active, full_cls=eval_full, crop_features=eval_crops,
            boxes=eval_boxes, labels=split.labels, names=names, paths=eval_paths,
            common=common, include_crops=True, lowres_features=None,
        ),
    }
    root = {"schema_version": BUNDLE_SCHEMA, "mode": "dev", "common_identity": common, "subsets": subsets}
    root_path = output_dir / "asset_manifest.json"
    root_path.write_text(json.dumps(root, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**root, "asset_manifest_sha256": sha256_file(root_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--text-manifest", type=Path, required=True)
    parser.add_argument("--text-manifest-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.config, args.output_dir, device_name=args.device, batch_size=args.batch_size,
        workers=args.workers, text_manifest=args.text_manifest,
        text_manifest_sha=args.text_manifest_sha,
    ), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
