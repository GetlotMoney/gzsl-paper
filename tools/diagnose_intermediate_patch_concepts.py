"""Diagnose whether intermediate frozen-CLIP local tokens retain role concepts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tools.gzsl_data import load_xlsa_split, resolve_xlsa_image_path


LAYERS = (12, 16, 20, 24)
ROLE_COUNT = 6
CLASS_COUNT = 200
CONCEPT_THRESHOLD = 0.85
NEIGHBOR_COUNT = 5


class SelectedImages(Dataset):
    def __init__(self, paths: list[Path], local_positions: np.ndarray, labels: np.ndarray, preprocess):
        self.paths = paths
        self.local_positions = local_positions
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as handle:
            image = self.preprocess(handle.convert("RGB"))
        return image, int(self.local_positions[index]), int(self.labels[index])


def deterministic_sample(labels: np.ndarray, sample_count: int, seed: int) -> np.ndarray:
    classes = np.unique(labels)
    base, remainder = divmod(sample_count, len(classes))
    rng = np.random.default_rng(seed)
    selected = []
    for class_rank, class_id in enumerate(classes):
        positions = np.flatnonzero(labels == class_id).copy()
        rng.shuffle(positions)
        take = base + int(class_rank < remainder)
        if len(positions) < take:
            raise ValueError(f"类别{class_id}只有{len(positions)}张图，无法抽取{take}张。")
        selected.extend(positions[:take].tolist())
    result = np.asarray(sorted(selected), dtype=np.int64)
    if len(result) != sample_count or len(np.unique(result)) != sample_count:
        raise RuntimeError("确定性seen抽样数量或唯一性错误。")
    return result


@torch.no_grad()
def phrase_embeddings(model, descriptions: list[list[str]], device: torch.device):
    import clip

    phrases = [
        [re.sub(r"^.*?, showing\s+", "", sentence, flags=re.I).rstrip(".") for sentence in row[:ROLE_COUNT]]
        for row in descriptions
    ]
    flat = [phrases[class_id][role] for role in range(ROLE_COUNT) for class_id in range(CLASS_COUNT)]
    outputs = []
    for start in range(0, len(flat), 128):
        tokens = clip.tokenize(flat[start : start + 128]).to(device)
        encoded = F.normalize(model.encode_text(tokens).float(), dim=-1)
        outputs.append(encoded.cpu())
    return torch.cat(outputs).reshape(ROLE_COUNT, CLASS_COUNT, 768), phrases


def build_concepts(
    embeddings: torch.Tensor,
    phrases: list[list[str]],
    seen_classes: set[int],
):
    unseen_classes = set(range(CLASS_COUNT)) - seen_classes
    clusters: list[tuple[int, list[int]]] = []
    for role in range(ROLE_COUNT):
        similarities = embeddings[role] @ embeddings[role].T
        similarities.fill_diagonal_(-9)
        neighbors = similarities.topk(NEIGHBOR_COUNT, dim=1).indices
        adjacency = [set() for _ in range(CLASS_COUNT)]
        for left in range(CLASS_COUNT):
            for right in neighbors[left].tolist():
                if (
                    similarities[left, right] >= CONCEPT_THRESHOLD
                    and bool((neighbors[right] == left).any())
                ):
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        visited: set[int] = set()
        for start in range(CLASS_COUNT):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if len(set(component) & seen_classes) >= 3 and len(set(component) & unseen_classes) >= 1:
                clusters.append((role, sorted(component)))
    if len(clusters) != 31:
        raise RuntimeError(f"概念构造没有复现31簇：{len(clusters)}")
    concepts = []
    names = []
    for role, component in clusters:
        concept = embeddings[role, component].mean(dim=0)
        concept = F.normalize(concept, dim=0)
        concepts.append(concept)
        medoid = component[int((embeddings[role, component] @ concept).argmax())]
        names.append(phrases[medoid][role])
    return torch.stack(concepts), clusters, names


@torch.no_grad()
def intermediate_scores(model, images: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
    visual = model.visual
    images = images.to(dtype=visual.conv1.weight.dtype)
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_token = visual.class_embedding.to(x.dtype)
    class_tokens = class_token + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat((class_tokens, x), dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)
    outputs = []
    for index, block in enumerate(visual.transformer.resblocks, start=1):
        x = block(x)
        if index in LAYERS:
            patches = visual.ln_post(x[1:].permute(1, 0, 2))
            if visual.proj is not None:
                patches = patches @ visual.proj
            patches = F.normalize(patches.float(), dim=-1)
            outputs.append(torch.matmul(patches, concepts.T).amax(dim=1))
    if len(outputs) != len(LAYERS):
        raise RuntimeError("CLIP中间层局部token提取不完整。")
    return torch.stack(outputs, dim=1)


def extract(args) -> None:
    source = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    split = load_xlsa_split(source["res101"], source["att_splits"])
    train_labels = torch.load(args.train_labels, map_location="cpu", weights_only=True).long().numpy()
    expected_labels = split.labels.index_select(0, split.train_indices).numpy()
    if not np.array_equal(train_labels, expected_labels):
        raise ValueError("正式patch资产与Xian trainval标签/行序不一致。")
    selected = deterministic_sample(train_labels, args.sample_count, args.seed)
    shard_positions = selected[args.shard_index :: args.shard_count]
    global_indices = split.train_indices.numpy()[shard_positions]
    paths = [
        resolve_xlsa_image_path(source["raw_root"], split.image_files[index], source["image_path_anchors"])
        for index in global_indices
    ]

    import clip

    device = torch.device(args.device)
    model, preprocess = clip.load(str(source["clip_checkpoint"]), device=device, jit=False)
    model.eval()
    text = json.loads(args.role_texts.read_text(encoding="utf-8"))
    embeddings, phrases = phrase_embeddings(model, text["descriptions"], device)
    seen_classes = set(np.unique(train_labels).tolist())
    concepts, clusters, names = build_concepts(embeddings, phrases, seen_classes)
    concepts_device = concepts.to(device)
    cached_patches = np.load(args.final_patches, mmap_mode="r")
    if cached_patches.shape != (len(train_labels), 576, 768) or cached_patches.dtype != np.float16:
        raise ValueError("正式576-patch资产shape/dtype错误。")

    loader = DataLoader(
        SelectedImages(paths, shard_positions, train_labels[shard_positions], preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    layer_rows = []
    cached_rows = []
    position_rows = []
    label_rows = []
    for images, local_positions, labels in loader:
        images = images.to(device, non_blocking=True)
        layer_rows.append(intermediate_scores(model, images, concepts_device).cpu())
        cached = torch.from_numpy(
            np.asarray(cached_patches[local_positions.numpy()], dtype=np.float16).copy()
        ).to(device)
        cached = F.normalize(cached.float(), dim=-1)
        cached_rows.append(torch.matmul(cached, concepts_device.T).amax(dim=1).cpu())
        position_rows.append(local_positions)
        label_rows.append(labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=torch.cat(position_rows).numpy(),
        labels=torch.cat(label_rows).numpy(),
        layer_scores=torch.cat(layer_rows).numpy(),
        cached_final_scores=torch.cat(cached_rows).numpy(),
        cluster_roles=np.asarray([role for role, _ in clusters], dtype=np.int64),
        cluster_members=np.asarray([json.dumps(component) for _, component in clusters]),
        concept_names=np.asarray(names),
    )
    print(json.dumps({"output": str(args.output), "count": len(shard_positions), "device": args.device}))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    return (ranks[y_true].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)


def aucs(labels, scores, cluster_members, seen_classes):
    values = []
    for concept_index, members in enumerate(cluster_members):
        positives = np.isin(labels, list(set(members) & seen_classes))
        values.append(roc_auc(positives, scores[:, concept_index]))
    return np.asarray(values)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "median_auc": float(np.median(values)),
        "mean_auc": float(values.mean()),
        "count_ge_0_60": int((values >= 0.60).sum()),
        "fraction_ge_0_60": float((values >= 0.60).mean()),
        "count_ge_0_70": int((values >= 0.70).sum()),
        "fraction_ge_0_70": float((values >= 0.70).mean()),
        "minimum_auc": float(values.min()),
        "maximum_auc": float(values.max()),
    }


def merge(args) -> None:
    shards = [np.load(path, allow_pickle=False) for path in args.shards]
    positions = np.concatenate([shard["positions"] for shard in shards])
    if len(np.unique(positions)) != args.sample_count:
        raise ValueError("双卡输出没有完整、唯一覆盖固定样本。")
    order = np.argsort(positions)
    labels = np.concatenate([shard["labels"] for shard in shards])[order]
    layer_scores = np.concatenate([shard["layer_scores"] for shard in shards])[order]
    cached_scores = np.concatenate([shard["cached_final_scores"] for shard in shards])[order]
    members = [json.loads(value) for value in shards[0]["cluster_members"].tolist()]
    names = shards[0]["concept_names"].tolist()
    seen_classes = set(np.unique(labels).tolist())
    layer_aucs = [aucs(labels, layer_scores[:, index], members, seen_classes) for index in range(len(LAYERS))]
    cached_aucs = aucs(labels, cached_scores, members, seen_classes)
    rng = np.random.default_rng(args.seed)
    permuted = labels[rng.permutation(len(labels))]
    permuted_aucs = aucs(permuted, cached_scores, members, seen_classes)
    result = {
        "sample_count": int(len(labels)),
        "class_count": int(len(seen_classes)),
        "layers": {str(layer): summarize(values) for layer, values in zip(LAYERS, layer_aucs)},
        "cached_final_576": summarize(cached_aucs),
        "layer24_cached_score_max_abs": float(np.max(np.abs(layer_scores[:, -1] - cached_scores))),
        "permuted_control": summarize(permuted_aucs),
        "pass_rule": "any intermediate layer: median_auc>=0.60 and fraction_ge_0_60>=0.60 and median gain over same-subset cached final >=0.03",
    }
    cached_median = result["cached_final_576"]["median_auc"]
    qualifying = []
    for layer, values in zip(LAYERS[:-1], layer_aucs[:-1]):
        summary = result["layers"][str(layer)]
        gain = summary["median_auc"] - cached_median
        summary["median_gain_vs_cached_final"] = gain
        if summary["median_auc"] >= 0.60 and summary["fraction_ge_0_60"] >= 0.60 and gain >= 0.03:
            qualifying.append(layer)
    result["qualifying_layers"] = qualifying
    result["decision"] = "intermediate_local_signal_supported" if qualifying else "intermediate_local_signal_not_supported"
    per_concept = []
    for index, name in enumerate(names):
        per_concept.append(
            {
                "name": name,
                "cached_final_auc": float(cached_aucs[index]),
                **{f"layer_{layer}_auc": float(values[index]) for layer, values in zip(LAYERS, layer_aucs)},
            }
        )
    result["per_concept"] = per_concept
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--source-config", type=Path, required=True)
    extract_parser.add_argument("--role-texts", type=Path, required=True)
    extract_parser.add_argument("--train-labels", type=Path, required=True)
    extract_parser.add_argument("--final-patches", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--device", required=True)
    extract_parser.add_argument("--shard-index", type=int, required=True)
    extract_parser.add_argument("--shard-count", type=int, default=2)
    extract_parser.add_argument("--sample-count", type=int, default=1000)
    extract_parser.add_argument("--seed", type=int, default=7)
    extract_parser.add_argument("--batch-size", type=int, default=8)
    extract_parser.add_argument("--workers", type=int, default=4)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--shards", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--sample-count", type=int, default=1000)
    merge_parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
