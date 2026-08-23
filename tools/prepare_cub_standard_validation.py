"""从xlsa17 Proposed Split生成不含official test的CUB开发划分。"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import scipy.io as sio
import torch


SPLIT_SEED = 20260823
SEEN_HOLDOUT_FRACTION = 0.2


def _per_class_fit_and_seen_validation(
    raw_indices: torch.Tensor,
    labels: torch.Tensor,
    classes: torch.Tensor,
    *,
    seed: int = SPLIT_SEED,
    holdout_fraction: float = SEEN_HOLDOUT_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    fit_parts = []
    validation_parts = []
    for class_id in classes.tolist():
        class_indices = raw_indices[labels.index_select(0, raw_indices) == class_id]
        class_indices = class_indices.sort().values
        generator = torch.Generator(device="cpu").manual_seed(seed + int(class_id))
        order = torch.randperm(class_indices.numel(), generator=generator)
        validation_count = max(
            1, int(math.ceil(class_indices.numel() * float(holdout_fraction)))
        )
        validation_parts.append(class_indices[order[:validation_count]])
        fit_parts.append(class_indices[order[validation_count:]])
    return torch.cat(fit_parts).sort().values, torch.cat(validation_parts).sort().values


def build_standard_validation_split(
    res101_path: Path,
    att_splits_path: Path,
    train_labels_path: Path,
) -> dict:
    labels_mat = sio.loadmat(res101_path, variable_names=["labels"])
    split_mat = sio.loadmat(
        att_splits_path,
        variable_names=["train_loc", "val_loc", "trainval_loc"],
    )
    labels = torch.from_numpy(labels_mat["labels"].squeeze().astype(int) - 1).long()
    train_loc = torch.from_numpy(split_mat["train_loc"].squeeze() - 1).long()
    val_loc = torch.from_numpy(split_mat["val_loc"].squeeze() - 1).long()
    trainval_loc = torch.from_numpy(split_mat["trainval_loc"].squeeze() - 1).long()
    cached_labels = torch.load(train_labels_path, map_location="cpu", weights_only=True).long()
    if not torch.equal(cached_labels, labels.index_select(0, trainval_loc)):
        raise ValueError("train标签缓存与xlsa17 trainval_loc顺序不一致。")

    dev_seen_classes = torch.unique(labels.index_select(0, train_loc), sorted=True)
    dev_unseen_classes = torch.unique(labels.index_select(0, val_loc), sorted=True)
    if dev_seen_classes.numel() != 100 or dev_unseen_classes.numel() != 50:
        raise ValueError("CUB标准开发划分必须是100个训练类和50个验证unseen类。")
    if torch.isin(dev_seen_classes, dev_unseen_classes).any():
        raise ValueError("开发seen类与validation-unseen类必须不相交。")

    fit_raw, val_seen_raw = _per_class_fit_and_seen_validation(
        train_loc, labels, dev_seen_classes
    )
    raw_to_cache = {int(raw): position for position, raw in enumerate(trainval_loc.tolist())}

    def cache_positions(raw_values: torch.Tensor) -> torch.Tensor:
        try:
            return torch.tensor(
                [raw_to_cache[int(raw)] for raw in raw_values.tolist()], dtype=torch.long
            )
        except KeyError as error:
            raise ValueError("开发索引不属于trainval_loc缓存。") from error

    fit_positions = cache_positions(fit_raw)
    val_seen_positions = cache_positions(val_seen_raw)
    val_unseen_positions = cache_positions(val_loc)
    if torch.isin(fit_positions, val_seen_positions).any():
        raise ValueError("开发梯度图像与validation-seen图像重叠。")
    if torch.isin(
        torch.cat((fit_positions, val_seen_positions)), val_unseen_positions
    ).any():
        raise ValueError("开发seen图像与validation-unseen图像重叠。")
    if fit_positions.numel() + val_seen_positions.numel() != train_loc.numel():
        raise ValueError("开发fit/val-seen未完整覆盖train_loc。")

    return {
        "schema_version": "gzsl-paper.cub-standard-validation.v1",
        "split_seed": SPLIT_SEED,
        "seen_holdout_fraction": SEEN_HOLDOUT_FRACTION,
        "trainval_cache_count": int(trainval_loc.numel()),
        "fit_positions": fit_positions,
        "val_seen_positions": val_seen_positions,
        "val_unseen_positions": val_unseen_positions,
        "dev_seen_classes": dev_seen_classes,
        "dev_unseen_classes": dev_unseen_classes,
        "fit_raw_indices": fit_raw,
        "val_seen_raw_indices": val_seen_raw,
        "val_unseen_raw_indices": val_loc.sort().values,
    }


def atomic_torch_save(payload: dict, output_path: Path) -> None:
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有开发划分：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res101", type=Path, required=True)
    parser.add_argument("--att-splits", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = build_standard_validation_split(
        args.res101.resolve(), args.att_splits.resolve(), args.train_labels.resolve()
    )
    atomic_torch_save(split, args.output)
    print(
        {
            "fit_images": split["fit_positions"].numel(),
            "val_seen_images": split["val_seen_positions"].numel(),
            "val_unseen_images": split["val_unseen_positions"].numel(),
            "dev_seen_classes": split["dev_seen_classes"].numel(),
            "dev_unseen_classes": split["dev_unseen_classes"].numel(),
        }
    )


if __name__ == "__main__":
    main()
