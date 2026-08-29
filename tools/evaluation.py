"""gzsl-paper V1 的严格缓存评估入口。

正式评估必须同时使用真实 CLS 和 576 个局部图像块；缺文件时直接停止，
不得复制 CLS 冒充局部特征，也不得悄悄切换到在线提取路线。
"""

from pathlib import Path

import torch


_CACHE_FILES = {
    "seen_cls": "CUB_test_seen_features.pt",
    "seen_labels": "CUB_test_seen_labels.pt",
    "seen_patches": "CUB_test_seen_patch_features.pt",
    "unseen_cls": "CUB_test_unseen_features.pt",
    "unseen_labels": "CUB_test_unseen_labels.pt",
    "unseen_patches": "CUB_test_unseen_patch_features.pt",
}

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


def test_cache_paths(cache_dir=_DEFAULT_CACHE_DIR):
    # 保留仓库内逻辑挂载路径；resolve()会在Windows上展开Junction，
    # 使运行契约错误地把G盘物理目标当成代码入口身份。
    root = Path(cache_dir).absolute()
    return {name: root / filename for name, filename in _CACHE_FILES.items()}


def load_test_cache(cache_dir=_DEFAULT_CACHE_DIR):
    paths = test_cache_paths(cache_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "V1 正式评估缺少真实 CLS/局部块缓存：" + ", ".join(missing)
        )
    cache = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in paths.items()
    }
    _validate_cache_split(
        "seen", cache["seen_cls"], cache["seen_patches"], cache["seen_labels"]
    )
    _validate_cache_split(
        "unseen",
        cache["unseen_cls"],
        cache["unseen_patches"],
        cache["unseen_labels"],
    )
    return cache


def _validate_cache_split(name, cls_features, patches, labels):
    if cls_features.dim() != 2:
        raise ValueError(f"{name} CLS 缓存必须是 [N, D]，实际为 {tuple(cls_features.shape)}。")
    if patches.dim() != 3 or patches.size(1) != 576:
        raise ValueError(
            f"{name} 局部块缓存必须是 [N, 576, D]，实际为 {tuple(patches.shape)}。"
        )
    if labels.dim() != 1:
        raise ValueError(f"{name} 标签缓存必须是 [N]，实际为 {tuple(labels.shape)}。")
    if cls_features.size(0) != patches.size(0) or labels.size(0) != patches.size(0):
        raise ValueError(f"{name} 的 CLS、局部块和标签数量不一致。")
    if cls_features.size(1) != patches.size(2):
        raise ValueError(f"{name} 的 CLS 与局部块特征维度不一致。")


def _predict(model, cls_features, patches, device, batch_size):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, cls_features.size(0), batch_size):
            cls_batch = cls_features[start : start + batch_size].to(device).float()
            patch_batch = patches[start : start + batch_size].to(device).float()
            features = torch.cat([cls_batch.unsqueeze(1), patch_batch], dim=1)
            logits = model(features, is_train=False)["clip_S_pp"]
            if not torch.isfinite(logits).all():
                raise ValueError("评估 logits 包含 NaN/Inf，拒绝计算指标。")
            predictions.append(logits.cpu())
    return torch.cat(predictions, dim=0)


def _per_class_accuracy(labels, predictions, classes):
    values = []
    labels = labels.cpu().long()
    predictions = predictions.cpu().long()
    for class_id in classes.cpu().long():
        mask = labels == class_id
        if not mask.any():
            raise ValueError(f"评估缓存里缺少类别 {int(class_id)} 的样本。")
        values.append((predictions[mask] == labels[mask]).float().mean())
    return float(torch.stack(values).mean().item())


def evaluate_cached(model, device, cache, seenclasses, unseenclasses, batch_size=64):
    seenclasses = torch.as_tensor(seenclasses).detach().cpu().long()
    unseenclasses = torch.as_tensor(unseenclasses).detach().cpu().long()
    if seenclasses.dim() != 1 or unseenclasses.dim() != 1:
        raise ValueError("seenclasses 和 unseenclasses 必须是一维全局类别编号。")
    if seenclasses.unique().numel() != seenclasses.numel():
        raise ValueError("seenclasses 含有重复类别。")
    if unseenclasses.unique().numel() != unseenclasses.numel():
        raise ValueError("unseenclasses 含有重复类别。")
    if torch.isin(seenclasses, unseenclasses).any():
        raise ValueError("seenclasses 与 unseenclasses 不能重叠。")
    expected_classes = getattr(model, "nclass", None)
    if expected_classes is not None:
        combined = torch.cat([seenclasses, unseenclasses]).sort().values
        expected = torch.arange(int(expected_classes), dtype=torch.long)
        if not torch.equal(combined.cpu(), expected):
            raise ValueError("seen/unseen 类别没有完整覆盖模型的全局类别轴。")
    _validate_cache_split(
        "seen", cache["seen_cls"], cache["seen_patches"], cache["seen_labels"]
    )
    _validate_cache_split(
        "unseen",
        cache["unseen_cls"],
        cache["unseen_patches"],
        cache["unseen_labels"],
    )
    if hasattr(model, "seenclass") and not torch.equal(
        model.seenclass.detach().cpu().long(), seenclasses.cpu()
    ):
        raise ValueError("评估 seenclasses 顺序与模型不一致。")
    if hasattr(model, "unseenclass") and not torch.equal(
        model.unseenclass.detach().cpu().long(), unseenclasses.cpu()
    ):
        raise ValueError("评估 unseenclasses 顺序与模型不一致。")
    seen_logits = _predict(
        model, cache["seen_cls"], cache["seen_patches"], device, batch_size
    )
    unseen_logits = _predict(
        model, cache["unseen_cls"], cache["unseen_patches"], device, batch_size
    )

    seen_prediction = seen_logits.argmax(dim=1)
    unseen_prediction = unseen_logits.argmax(dim=1)
    unseen_only_prediction = unseenclasses[
        unseen_logits[:, unseenclasses].argmax(dim=1)
    ]

    seen_accuracy = _per_class_accuracy(
        cache["seen_labels"], seen_prediction, seenclasses
    )
    unseen_accuracy = _per_class_accuracy(
        cache["unseen_labels"], unseen_prediction, unseenclasses
    )
    zsl_accuracy = _per_class_accuracy(
        cache["unseen_labels"], unseen_only_prediction, unseenclasses
    )
    denominator = seen_accuracy + unseen_accuracy
    harmonic = 2.0 * seen_accuracy * unseen_accuracy / denominator if denominator else 0.0
    return seen_accuracy, unseen_accuracy, harmonic, zsl_accuracy
