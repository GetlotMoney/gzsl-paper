"""只读取 V1 正式训练需要的 CUB/xlsa17 划分信息。"""

from pathlib import Path

import scipy.io as sio
import torch


def validate_standard_split_sizes(train_count, test_seen_count, test_unseen_count):
    actual = (int(train_count), int(test_seen_count), int(test_unseen_count))
    expected = (7057, 1764, 2967)
    if actual != expected:
        raise ValueError(
            "CUB V1 固定要求 trainval/test-seen/test-unseen 样本数为 "
            f"{expected}，实际为 {actual}。"
        )


def validate_standard_class_counts(seenclasses, unseenclasses):
    seen = torch.as_tensor(seenclasses).flatten()
    unseen = torch.as_tensor(unseenclasses).flatten()
    if seen.numel() != 150 or unseen.numel() != 50:
        raise ValueError(
            "CUB V1 固定要求 150 个 seen 类和 50 个 unseen 类；"
            f"实际为 {seen.numel()}/{unseen.numel()}。"
        )


def load_cub_split(
    res101_path,
    split_path,
    train_labels,
    test_seen_labels,
    test_unseen_labels,
    device,
):
    res101 = sio.loadmat(Path(res101_path))
    splits = sio.loadmat(Path(split_path))
    labels = torch.from_numpy(res101["labels"].astype(int).squeeze() - 1).long()
    train_indices = torch.from_numpy(splits["trainval_loc"].squeeze() - 1).long()
    seen_indices = torch.from_numpy(splits["test_seen_loc"].squeeze() - 1).long()
    unseen_indices = torch.from_numpy(splits["test_unseen_loc"].squeeze() - 1).long()

    expected = {
        "train": labels[train_indices],
        "test_seen": labels[seen_indices],
        "test_unseen": labels[unseen_indices],
    }
    validate_standard_split_sizes(
        expected["train"].numel(),
        expected["test_seen"].numel(),
        expected["test_unseen"].numel(),
    )
    actual = {
        "train": train_labels.detach().cpu().long(),
        "test_seen": test_seen_labels.detach().cpu().long(),
        "test_unseen": test_unseen_labels.detach().cpu().long(),
    }
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            raise ValueError(f"{name} 缓存标签与 xlsa17 划分不一致。")

    seenclasses = torch.unique(expected["train"], sorted=True)
    test_seenclasses = torch.unique(expected["test_seen"], sorted=True)
    unseenclasses = torch.unique(expected["test_unseen"], sorted=True)
    if not torch.equal(seenclasses, test_seenclasses):
        raise ValueError("训练 seen 类与 test_seen 类集合不一致。")
    validate_standard_class_counts(seenclasses, unseenclasses)
    return seenclasses.to(device), unseenclasses.to(device)
