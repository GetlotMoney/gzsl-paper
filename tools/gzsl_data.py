"""Dataset-agnostic xlsa17 loading and GZSL metric helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class XlsaSplit:
    labels: torch.Tensor
    image_files: tuple[str, ...]
    class_names: tuple[str, ...]
    train_indices: torch.Tensor
    test_seen_indices: torch.Tensor
    test_unseen_indices: torch.Tensor
    seen_classes: torch.Tensor
    unseen_classes: torch.Tensor

    @property
    def class_count(self) -> int:
        return len(self.class_names)


def _matlab_string(value) -> str:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _matlab_string_list(values) -> tuple[str, ...]:
    array = np.asarray(values).reshape(-1)
    return tuple(_matlab_string(value).strip() for value in array)


def _zero_based_indices(values, image_count: int, name: str) -> torch.Tensor:
    indices = torch.from_numpy(np.asarray(values).reshape(-1).astype(np.int64) - 1)
    if indices.numel() == 0 or indices.unique().numel() != indices.numel():
        raise ValueError(f"{name}必须是非空且无重复的索引。")
    if int(indices.min()) < 0 or int(indices.max()) >= int(image_count):
        raise ValueError(f"{name}超出图像索引范围。")
    return indices.long()


def load_xlsa_split(res101_path: Path | str, att_splits_path: Path | str) -> XlsaSplit:
    res101_path = Path(res101_path)
    att_splits_path = Path(att_splits_path)
    res101 = sio.loadmat(res101_path, variable_names=["labels", "image_files"])
    splits = sio.loadmat(
        att_splits_path,
        variable_names=[
            "trainval_loc",
            "test_seen_loc",
            "test_unseen_loc",
            "allclasses_names",
        ],
    )
    if "labels" not in res101 or "image_files" not in res101:
        raise ValueError("res101.mat缺少labels或image_files。")
    labels = torch.from_numpy(np.asarray(res101["labels"]).reshape(-1).astype(np.int64) - 1).long()
    image_files = _matlab_string_list(res101["image_files"])
    if labels.numel() != len(image_files) or labels.numel() == 0:
        raise ValueError("xlsa17标签数量与图像路径数量不一致。")
    if int(labels.min()) < 0:
        raise ValueError("xlsa17类别编号必须从1开始并可转换为0-based。")
    train = _zero_based_indices(splits["trainval_loc"], labels.numel(), "trainval_loc")
    test_seen = _zero_based_indices(splits["test_seen_loc"], labels.numel(), "test_seen_loc")
    test_unseen = _zero_based_indices(splits["test_unseen_loc"], labels.numel(), "test_unseen_loc")
    for left_name, left, right_name, right in (
        ("trainval", train, "test_seen", test_seen),
        ("trainval", train, "test_unseen", test_unseen),
        ("test_seen", test_seen, "test_unseen", test_unseen),
    ):
        if torch.isin(left, right).any():
            raise ValueError(f"{left_name}与{right_name}索引不得重叠。")
    coverage = torch.cat((train, test_seen, test_unseen)).sort().values
    if not torch.equal(coverage, torch.arange(labels.numel())):
        raise ValueError("trainval/test-seen/test-unseen必须完整覆盖xlsa17图像。")
    seen_classes = torch.unique(labels.index_select(0, train), sorted=True)
    test_seen_classes = torch.unique(labels.index_select(0, test_seen), sorted=True)
    unseen_classes = torch.unique(labels.index_select(0, test_unseen), sorted=True)
    if not torch.equal(seen_classes, test_seen_classes):
        raise ValueError("trainval和test-seen类别集合不一致。")
    if torch.isin(seen_classes, unseen_classes).any():
        raise ValueError("seen与unseen类别不得重叠。")
    class_names = _matlab_string_list(splits["allclasses_names"])
    class_count = int(labels.max()) + 1
    if len(class_names) != class_count:
        raise ValueError(
            f"类别名数量{len(class_names)}与标签类别轴{class_count}不一致。"
        )
    all_classes = torch.cat((seen_classes, unseen_classes)).sort().values
    if not torch.equal(all_classes, torch.arange(class_count)):
        raise ValueError("seen/unseen类别必须完整覆盖全局类别轴。")
    return XlsaSplit(
        labels=labels,
        image_files=image_files,
        class_names=class_names,
        train_indices=train,
        test_seen_indices=test_seen,
        test_unseen_indices=test_unseen,
        seen_classes=seen_classes,
        unseen_classes=unseen_classes,
    )


def class_order_sha256(class_names: Iterable[str]) -> str:
    serialized = json.dumps(list(class_names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def clean_class_name(name: str) -> str:
    value = re.sub(r"^\d+[._\s-]+", "", str(name).strip())
    value = value.replace("_", " ").replace("/", " ")
    return " ".join(value.split())


def resolve_xlsa_image_path(
    raw_root: Path | str,
    xlsa_image_path: str,
    anchors: Iterable[str],
) -> Path:
    root = Path(raw_root)
    normalized = re.sub(r"/+", "/", str(xlsa_image_path).replace("\\", "/").strip())
    lowered = normalized.lower()
    candidates: list[Path] = []
    for anchor in anchors:
        clean_anchor = str(anchor).replace("\\", "/").strip("/")
        position = lowered.find(clean_anchor.lower())
        if position >= 0:
            candidates.append(root / Path(normalized[position:]))
            suffix = normalized[position + len(clean_anchor) :].lstrip("/")
            candidates.append(root / Path(suffix))
    candidates.append(root / Path(normalized.lstrip("/")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = ", ".join(str(path) for path in candidates[:4])
    raise FileNotFoundError(f"无法解析xlsa17图像路径：{normalized}；尝试：{attempted}")


def per_class_accuracy(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    classes: torch.Tensor,
) -> float:
    labels = labels.detach().cpu().long()
    predictions = predictions.detach().cpu().long()
    values = []
    for class_id in classes.detach().cpu().long():
        mask = labels.eq(class_id)
        if not mask.any():
            raise ValueError(f"评估split缺少类别{int(class_id)}。")
        values.append(predictions[mask].eq(class_id).float().mean())
    return float(torch.stack(values).mean())


@torch.no_grad()
def evaluate_prototypes(
    prototypes: torch.Tensor,
    scale: torch.Tensor | float,
    seen_features: torch.Tensor,
    seen_labels: torch.Tensor,
    unseen_features: torch.Tensor,
    unseen_labels: torch.Tensor,
    seen_classes: torch.Tensor,
    unseen_classes: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    prototypes = F.normalize(prototypes.detach().to(device).float(), dim=-1)
    if prototypes.ndim != 2 or prototypes.size(1) != 768:
        raise ValueError("原型必须是[class_count,768]。")
    all_classes = torch.cat((seen_classes, unseen_classes)).sort().values
    if not torch.equal(all_classes.cpu(), torch.arange(prototypes.size(0))):
        raise ValueError("seen/unseen类别没有完整覆盖原型类别轴。")
    scale_value = torch.as_tensor(scale, device=device, dtype=torch.float32)

    def predict(features: torch.Tensor, candidates: torch.Tensor | None = None) -> torch.Tensor:
        candidate_ids = (
            torch.arange(prototypes.size(0), device=device)
            if candidates is None
            else candidates.to(device).long()
        )
        candidate_prototypes = prototypes.index_select(0, candidate_ids)
        predictions = []
        for start in range(0, features.size(0), int(batch_size)):
            images = F.normalize(features[start : start + int(batch_size)].to(device).float(), dim=-1)
            logits = images @ candidate_prototypes.T * scale_value
            if not torch.isfinite(logits).all():
                raise ValueError("评估logits包含NaN/Inf。")
            predictions.append(candidate_ids[logits.argmax(dim=1)].cpu())
        return torch.cat(predictions)

    seen_prediction = predict(seen_features)
    unseen_prediction = predict(unseen_features)
    zsl_prediction = predict(unseen_features, unseen_classes)
    seen = per_class_accuracy(seen_labels, seen_prediction, seen_classes)
    unseen = per_class_accuracy(unseen_labels, unseen_prediction, unseen_classes)
    zsl = per_class_accuracy(unseen_labels, zsl_prediction, unseen_classes)
    harmonic = 2.0 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    return {"U": unseen * 100.0, "S": seen * 100.0, "H": harmonic * 100.0, "ZS": zsl * 100.0}
