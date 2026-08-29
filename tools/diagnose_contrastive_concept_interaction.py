"""Gate contrastive concept-margin region interactions for IDEA-169."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
from collections import defaultdict
from pathlib import Path

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from tools.gzsl_data import load_xlsa_split, resolve_xlsa_image_path
from tools.run_contract import current_code_commit, require_clean_code_tree
from tools.runtime import sha256_file


ROLE_COUNT = 6
CLASS_COUNT = 200
PATCH_DIMENSION = 768
CONCEPT_THRESHOLD = 0.85
NEIGHBOR_COUNT = 5
PROMPT_TEMPLATES = (
    "a photo of a bird with {phrase}",
    "a close-up photo of a bird showing {phrase}",
    "a bird whose visible features include {phrase}",
)
PERTURBATIONS = ("mean_fill", "local_blur")
CONTROLS = ("hard", "random")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "idea_id", "rescue_of", "parent_result_uri",
        "parent_result_sha256", "dataset", "role_texts", "role_texts_sha256",
        "clip_checkpoint", "clip_checkpoint_sha256", "source_config",
        "source_config_sha256", "visual_asset_manifest", "visual_asset_manifest_sha256",
        "train_labels", "final_patches", "seed", "reader_rank", "reader_updates",
        "reader_batch_size", "reader_learning_rate", "reader_weight_decay",
        "reader_auc_gate", "evaluation_image_count", "max_concepts_per_image",
        "shard_count", "token_grid_side", "window_patch_side", "primary_readout",
        "cublas_workspace_config", "contrastive_minimum_neighbors",
        "contrastive_maximum_neighbors",
        "hard_attention_log_distance_max", "hard_frequency_log_distance_max",
        "blur_kernel_size", "blur_sigma", "minimum_interaction_pairs",
        "minimum_interaction_classes", "bootstrap_replicates",
        "standardized_effect_gate", "patch_identity_gate", "eligible_indices",
        "formal_unseen_images_used", "all_200_class_texts_used", "human_annotations_used",
    }
    actual = set(config) if isinstance(config, dict) else set()
    if actual != required:
        raise ValueError(
            f"IDEA-169配置错误：缺少={sorted(required-actual)}，多出={sorted(actual-required)}"
        )
    if (
        config["schema_version"] != "gzsl-paper.contrastive-concept-interaction.v1"
        or config["idea_id"] != "IDEA-169"
        or config["rescue_of"] != "IDEA-168"
        or config["dataset"] != "CUB"
        or config["formal_unseen_images_used"] is not False
        or config["all_200_class_texts_used"] is not True
        or config["human_annotations_used"] is not False
    ):
        raise ValueError("IDEA-169配置身份或数据边界错误。")
    if int(config["token_grid_side"]) != 24 or int(config["window_patch_side"]) != 4:
        raise ValueError("IDEA-169固定使用24x24 token与4x4-patch窗口。")
    if (
        int(config["shard_count"]) != 2
        or config["primary_readout"] != "fixed_original_attention_contrast_margin"
        or int(config["contrastive_minimum_neighbors"]) != 2
        or int(config["contrastive_maximum_neighbors"]) != 3
    ):
        raise ValueError("IDEA-169固定双卡、原图Attention和2至3个对比概念。")
    if config["cublas_workspace_config"] != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError("IDEA-169固定cuBLAS确定性工作区为:4096:8。")
    if int(config["blur_kernel_size"]) % 2 != 1:
        raise ValueError("局部模糊kernel必须为奇数。")
    return config


def validate_assets(config: dict) -> dict:
    parent_result_path = Path(config["parent_result_uri"])
    if sha256_file(parent_result_path) != config["parent_result_sha256"]:
        raise ValueError("IDEA-168父结果SHA错误。")
    manifest_path = Path(config["visual_asset_manifest"])
    if sha256_file(manifest_path) != config["visual_asset_manifest_sha256"]:
        raise ValueError("576-patch manifest SHA错误。")
    if sha256_file(Path(config["role_texts"])) != config["role_texts_sha256"]:
        raise ValueError("角色文本SHA错误。")
    if sha256_file(Path(config["source_config"])) != config["source_config_sha256"]:
        raise ValueError("source config SHA错误。")
    checkpoint_path = Path(config["clip_checkpoint"])
    if sha256_file(checkpoint_path) != config["clip_checkpoint_sha256"]:
        raise ValueError("冻结CLIP checkpoint SHA错误。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "gzsl-paper.projected-patch-assets.v1"
        or manifest.get("patch_shape") != [576, 768]
        or manifest.get("clip_checkpoint_sha256") != config["clip_checkpoint_sha256"]
    ):
        raise ValueError("576-patch资产schema或CLIP身份错误。")
    for key, filename in (("train_labels", "train_labels.pt"), ("final_patches", "train_patch_features.npy")):
        actual_path = Path(config[key])
        expected_path = manifest_path.parent / filename
        if actual_path.resolve() != expected_path.resolve() or not actual_path.is_file():
            raise ValueError(f"{key}没有绑定manifest目录。")
        if sha256_file(actual_path) != manifest.get("outputs_sha256", {}).get(filename):
            raise ValueError(f"{key}内容SHA与manifest不一致。")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": config["visual_asset_manifest_sha256"],
        "asset_id": manifest.get("asset_id"),
        "class_order_sha256": manifest.get("class_order_sha256"),
        "role_texts_sha256": config["role_texts_sha256"],
        "clip_checkpoint_sha256": config["clip_checkpoint_sha256"],
        "parent_result_sha256": config["parent_result_sha256"],
    }


def environment(device: torch.device) -> dict:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_total_memory": int(properties.total_memory),
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC需要同时包含正负样本。")
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
    return float((ranks[y_true].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def metric_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median_auc": float(np.median(values)),
        "mean_auc": float(values.mean()),
        "count_ge_0_60": int((values >= 0.60).sum()),
        "fraction_ge_0_60": float((values >= 0.60).mean()),
    }


@torch.no_grad()
def phrase_embeddings(model, descriptions: list[list[str]], device: torch.device):
    import clip

    phrases = [
        [re.sub(r"^.*?, showing\s+", "", sentence, flags=re.I).rstrip(".") for sentence in row[:ROLE_COUNT]]
        for row in descriptions
    ]
    flat = [phrases[class_id][role] for role in range(ROLE_COUNT) for class_id in range(CLASS_COUNT)]
    rows = []
    for start in range(0, len(flat), 128):
        tokens = clip.tokenize(flat[start:start + 128]).to(device)
        rows.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    return torch.cat(rows).reshape(ROLE_COUNT, CLASS_COUNT, PATCH_DIMENSION), phrases


def build_concepts(embeddings: torch.Tensor, phrases: list[list[str]], seen_classes: set[int]):
    unseen_classes = set(range(CLASS_COUNT)) - seen_classes
    clusters: list[tuple[int, list[int]]] = []
    for role in range(ROLE_COUNT):
        similarities = embeddings[role] @ embeddings[role].T
        similarities.fill_diagonal_(-9)
        neighbors = similarities.topk(NEIGHBOR_COUNT, dim=1).indices
        adjacency = [set() for _ in range(CLASS_COUNT)]
        for left in range(CLASS_COUNT):
            for right in neighbors[left].tolist():
                if similarities[left, right] >= CONCEPT_THRESHOLD and bool((neighbors[right] == left).any()):
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
    names = []
    for role, component in clusters:
        concept = F.normalize(embeddings[role, component].mean(dim=0), dim=0)
        medoid = component[int((embeddings[role, component] @ concept).argmax())]
        names.append(phrases[medoid][role])
    return clusters, names


@torch.no_grad()
def prompted_queries(model, phrases: list[list[str]], clusters, device: torch.device) -> torch.Tensor:
    import clip

    prompts = [
        PROMPT_TEMPLATES[template].format(phrase=phrases[class_id][role])
        for role in range(ROLE_COUNT)
        for class_id in range(CLASS_COUNT)
        for template in range(len(PROMPT_TEMPLATES))
    ]
    rows = []
    for start in range(0, len(prompts), 128):
        tokens = clip.tokenize(prompts[start:start + 128]).to(device)
        rows.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    values = torch.cat(rows).reshape(ROLE_COUNT, CLASS_COUNT, len(PROMPT_TEMPLATES), PATCH_DIMENSION)
    values = F.normalize(values.mean(dim=2), dim=-1)
    return torch.stack([F.normalize(values[role, members].mean(dim=0), dim=0) for role, members in clusters])


def setup_concepts(config: dict, clip_model, device: torch.device) -> dict:
    labels = torch.load(config["train_labels"], map_location="cpu", weights_only=True).long().numpy()
    seen_classes = np.unique(labels)
    if len(labels) != 7057 or len(seen_classes) != 150:
        raise ValueError("IDEA-169固定CUB 7,057张formal-seen图像与150类。")
    payload = json.loads(Path(config["role_texts"]).read_text(encoding="utf-8"))
    embeddings, phrases = phrase_embeddings(clip_model, payload["descriptions"], device)
    all_clusters, all_names = build_concepts(embeddings, phrases, set(seen_classes.tolist()))
    all_queries = prompted_queries(clip_model, phrases, all_clusters, device)

    rng = np.random.default_rng(int(config["seed"]))
    class_order = rng.permutation(seen_classes)
    train_classes = set(class_order[:100].tolist())
    evaluation_classes = set(class_order[100:].tolist())
    eligible = []
    for index, (_, members) in enumerate(all_clusters):
        if len(train_classes & set(members)) >= 2 and len(evaluation_classes & set(members)) >= 1:
            eligible.append(index)
    expected = [int(value) for value in config["eligible_indices"]]
    if eligible != expected or len(eligible) != 27:
        raise ValueError(f"IDEA-169没有复现固定27概念轴：{eligible}")
    clusters = [all_clusters[index] for index in eligible]
    names = [all_names[index] for index in eligible]
    queries = all_queries[eligible]
    positive_classes = [set(members) & set(seen_classes.tolist()) for _, members in clusters]
    targets = np.stack([np.isin(labels, list(classes)) for classes in positive_classes], axis=1).astype(np.float32)
    return {
        "labels": labels,
        "queries": queries,
        "clusters": clusters,
        "names": names,
        "targets": targets,
        "train_classes": sorted(train_classes),
        "evaluation_classes": sorted(evaluation_classes),
        "train_rows": np.flatnonzero(np.isin(labels, list(train_classes))),
        "evaluation_rows": np.flatnonzero(np.isin(labels, list(evaluation_classes))),
    }


class SharedConceptReadout(nn.Module):
    def __init__(self, rank: int = 64):
        super().__init__()
        self.visual_down = nn.Linear(PATCH_DIMENSION, rank, bias=False)
        self.visual_up = nn.Linear(rank, PATCH_DIMENSION, bias=False)
        self.text_down = nn.Linear(PATCH_DIMENSION, rank, bias=False)
        self.text_up = nn.Linear(rank, PATCH_DIMENSION, bias=False)
        nn.init.zeros_(self.visual_up.weight)
        nn.init.zeros_(self.text_up.weight)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def details(self, patches: torch.Tensor, concepts: torch.Tensor):
        patches = F.normalize(patches.float(), dim=-1)
        concepts = F.normalize(concepts.float(), dim=-1)
        visual = F.normalize(patches + self.visual_up(F.gelu(self.visual_down(patches))), dim=-1)
        text = F.normalize(concepts + self.text_up(F.gelu(self.text_down(concepts))), dim=-1)
        similarities = torch.matmul(visual, text.T)
        attention = torch.softmax(similarities / 0.07, dim=1)
        evidence = (attention * similarities).sum(dim=1)
        logits = self.logit_scale.exp().clamp(max=100.0) * evidence + self.bias
        return logits, similarities.permute(0, 2, 1), attention.permute(0, 2, 1)

    def forward(self, patches: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        return self.details(patches, concepts)[0]


@torch.no_grad()
def adapted_text_queries(reader: SharedConceptReadout, concepts: torch.Tensor) -> torch.Tensor:
    concepts = F.normalize(concepts.float(), dim=-1)
    return F.normalize(
        concepts + reader.text_up(F.gelu(reader.text_down(concepts))), dim=-1
    )


def train_reader(values: dict, patches, config: dict, device: torch.device):
    torch.manual_seed(int(config["seed"]))
    rng = np.random.default_rng(int(config["seed"]))
    reader = SharedConceptReadout(rank=int(config["reader_rank"])).to(device)
    concepts = values["queries"].to(device)
    rows = values["train_rows"]
    targets = values["targets"][rows]
    positives = targets.sum(axis=0)
    pos_weight = torch.from_numpy((len(targets) - positives) / np.maximum(positives, 1.0)).to(device)
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=float(config["reader_learning_rate"]),
        weight_decay=float(config["reader_weight_decay"]),
    )
    losses = []
    reader.train()
    for _ in range(int(config["reader_updates"])):
        local = rng.integers(0, len(rows), size=int(config["reader_batch_size"]))
        batch_rows = rows[local]
        batch = torch.from_numpy(np.asarray(patches[batch_rows], dtype=np.float16).copy()).to(device)
        target = torch.from_numpy(targets[local]).to(device)
        logits = reader(batch, concepts)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return reader.eval(), {
        "initial_loss_mean_20": float(np.mean(losses[:20])),
        "final_loss_mean_20": float(np.mean(losses[-20:])),
    }


@torch.no_grad()
def reader_scores(reader, patches, rows, concepts, device, batch_size=16):
    outputs = []
    concepts = concepts.to(device)
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        batch = torch.from_numpy(np.asarray(patches[selected], dtype=np.float16).copy()).to(device)
        outputs.append(reader(batch, concepts).cpu())
    return torch.cat(outputs).numpy()


def balanced_threshold(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("阈值校准需要正负样本。")
    values = np.concatenate((positive, negative))
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    order = np.argsort(values, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    tpr = np.cumsum(sorted_labels) / len(positive)
    tnr = 1.0 - np.cumsum(1 - sorted_labels) / len(negative)
    return float(values[order[int(np.argmax(0.5 * (tpr + tnr)))]])


def calibrate_thresholds(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.asarray(
        [balanced_threshold(scores[targets[:, index] > 0.5, index], scores[targets[:, index] <= 0.5, index]) for index in range(scores.shape[1])],
        dtype=np.float32,
    )


def stratified_rows(labels: np.ndarray, classes: list[int], count: int, seed: int) -> np.ndarray:
    if count % len(classes) != 0:
        raise ValueError("IDEA-169固定每个pseudo-unseen类抽取相同图像数。")
    rng = np.random.default_rng(seed)
    per_class = count // len(classes)
    selected = []
    for class_id in classes:
        candidates = np.flatnonzero(labels == class_id).copy()
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise ValueError(f"类别{class_id}不足{per_class}张图。")
        selected.extend(candidates[:per_class].tolist())
    result = np.asarray(sorted(selected), dtype=np.int64)
    if len(result) != count or len(np.unique(result)) != count:
        raise RuntimeError("分层图像抽样没有形成固定唯一集合。")
    return result


@torch.no_grad()
def encode_final_patches(model, images: torch.Tensor) -> torch.Tensor:
    visual = model.visual
    images = images.to(dtype=visual.conv1.weight.dtype)
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_token = visual.class_embedding.to(x.dtype)
    tokens = class_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat((tokens, x), dim=1)
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype)).permute(1, 0, 2)
    x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    return F.normalize(x.float(), dim=-1)


def window_from_peak(index: int, grid_side: int = 24, window_side: int = 4) -> tuple[int, int]:
    row, column = divmod(int(index), int(grid_side))
    maximum = int(grid_side) - int(window_side)
    return max(0, min(row - 1, maximum)), max(0, min(column - 1, maximum))


def windows_overlap(left: tuple[int, int], right: tuple[int, int], window_side: int = 4) -> bool:
    return not (
        left[0] + window_side <= right[0]
        or right[0] + window_side <= left[0]
        or left[1] + window_side <= right[1]
        or right[1] + window_side <= left[1]
    )


def window_edge_type(window: tuple[int, int], grid_side: int = 24, window_side: int = 4) -> str:
    maximum = grid_side - window_side
    return "edge" if window[0] in (0, maximum) or window[1] in (0, maximum) else "interior"


def window_mass(attention: np.ndarray, window: tuple[int, int], grid_side: int = 24, window_side: int = 4) -> float:
    matrix = np.asarray(attention).reshape(grid_side, grid_side)
    row, column = window
    return float(matrix[row:row + window_side, column:column + window_side].sum())


def select_nonoverlap_pair(attention: np.ndarray, grid_side: int = 24, window_side: int = 4):
    candidates = []
    seen = set()
    for index in np.argsort(np.asarray(attention), kind="mergesort")[::-1]:
        window = window_from_peak(int(index), grid_side, window_side)
        if window in seen:
            continue
        seen.add(window)
        if not candidates:
            candidates.append(window)
        elif not windows_overlap(candidates[0], window, window_side):
            candidates.append(window)
            break
    if len(candidates) != 2:
        return None
    masses = [window_mass(attention, window, grid_side, window_side) for window in candidates]
    return {"windows": candidates, "masses": masses}


def interaction_eta(original: float, score_a: float, score_b: float, score_union: float) -> float:
    drop_a = original - score_a
    drop_b = original - score_b
    drop_union = original - score_union
    return float(drop_union - drop_a - drop_b)


def magnitude_excess(target_eta: float, control_eta: float) -> float:
    return float(abs(target_eta) - abs(control_eta))


def fixed_attention_evidence(similarities: torch.Tensor, original_attention: torch.Tensor) -> torch.Tensor:
    if similarities.ndim != 2 or original_attention.ndim != 1:
        raise ValueError("固定Attention读出要求similarities=[batch,patch]、attention=[patch]。")
    if similarities.size(1) != original_attention.numel():
        raise ValueError("固定Attention读出的patch轴不一致。")
    return (similarities * original_attention.unsqueeze(0)).sum(dim=1)


@torch.no_grad()
def fixed_attention_logits(
    reader: SharedConceptReadout,
    patches: torch.Tensor,
    concepts: torch.Tensor,
    concept_index: int,
    original_attention: torch.Tensor,
) -> torch.Tensor:
    _, similarities, _ = reader.details(patches, concepts)
    evidence = fixed_attention_evidence(similarities[:, int(concept_index), :], original_attention)
    return reader.logit_scale.exp().clamp(max=100.0) * evidence + reader.bias


def contrastive_similarity_margin(
    similarities: torch.Tensor,
    target_index: int,
    competitor_indices: list[int] | tuple[int, ...],
) -> torch.Tensor:
    if similarities.ndim != 3:
        raise ValueError("对比margin要求similarities=[batch,concept,patch]。")
    if len(competitor_indices) < 2:
        raise ValueError("对比margin至少需要2个同角色竞争概念。")
    competitors = similarities[:, list(map(int, competitor_indices)), :].mean(dim=1)
    return similarities[:, int(target_index), :] - competitors


@torch.no_grad()
def fixed_contrastive_margin_logits(
    reader: SharedConceptReadout,
    patches: torch.Tensor,
    concepts: torch.Tensor,
    target_index: int,
    competitor_indices: list[int] | tuple[int, ...],
    original_attention: torch.Tensor,
) -> torch.Tensor:
    _, similarities, _ = reader.details(patches, concepts)
    margin = contrastive_similarity_margin(similarities, target_index, competitor_indices)
    evidence = fixed_attention_evidence(margin, original_attention)
    return reader.logit_scale.exp().clamp(max=100.0) * evidence


def choose_contrastive_neighbors(
    *, target_concept: int, class_id: int, clusters, query_similarities: np.ndarray,
    minimum_count: int, maximum_count: int, maximum_frequency_log_distance: float,
):
    target_role, target_members = clusters[int(target_concept)]
    target_frequency = max(len(target_members), 1)
    candidates = []
    for concept_index, (role, members) in enumerate(clusters):
        if concept_index == int(target_concept) or role != target_role or int(class_id) in members:
            continue
        frequency_log_distance = abs(math.log(max(len(members), 1) / target_frequency))
        if frequency_log_distance > maximum_frequency_log_distance:
            continue
        candidates.append(
            (
                -float(query_similarities[int(target_concept), concept_index]),
                frequency_log_distance,
                concept_index,
            )
        )
    selected = sorted(candidates)[: int(maximum_count)]
    if len(selected) < int(minimum_count):
        return None
    return {
        "indices": [int(row[2]) for row in selected],
        "query_cosines": [-float(row[0]) for row in selected],
        "frequency_log_distances": [float(row[1]) for row in selected],
    }


def _edge_flags(pair, grid_side: int, window_side: int):
    return sorted(window_edge_type(window, grid_side, window_side) for window in pair["windows"])


def choose_hard_control(
    *, target_pair, target_concept: int, class_id: int, attentions: np.ndarray,
    clusters, grid_side: int, window_side: int, maximum_log_distance: float,
    maximum_frequency_log_distance: float,
):
    target_role, target_members = clusters[target_concept]
    target_strength = sorted(target_pair["masses"], reverse=True)
    target_edges = _edge_flags(target_pair, grid_side, window_side)
    target_frequency = len(target_members)
    candidates = []
    for concept_index, (role, members) in enumerate(clusters):
        if concept_index == target_concept or role != target_role or class_id in members:
            continue
        pair = select_nonoverlap_pair(attentions[concept_index], grid_side, window_side)
        if pair is None or _edge_flags(pair, grid_side, window_side) != target_edges:
            continue
        target_attention_masses = [
            window_mass(attentions[target_concept], window, grid_side, window_side)
            for window in pair["windows"]
        ]
        strength = sorted(target_attention_masses, reverse=True)
        log_distance = max(abs(math.log((left + 1e-12) / (right + 1e-12))) for left, right in zip(strength, target_strength))
        frequency_log_distance = abs(math.log(max(len(members), 1) / max(target_frequency, 1)))
        if log_distance > maximum_log_distance or frequency_log_distance > maximum_frequency_log_distance:
            continue
        score = log_distance + 0.20 * frequency_log_distance
        candidates.append(
            (score, log_distance, frequency_log_distance, concept_index, pair, target_attention_masses)
        )
    if not candidates:
        return None
    score, log_distance, frequency_log_distance, concept_index, pair, target_attention_masses = min(
        candidates, key=lambda row: (row[0], row[3])
    )
    return {
        "concept_index": int(concept_index),
        "windows": pair["windows"],
        "proposal_concept_attention_masses": pair["masses"],
        "target_concept_attention_masses": target_attention_masses,
        "attention_log_distance": float(log_distance),
        "frequency_log_distance": float(frequency_log_distance),
        "target_frequency": int(target_frequency),
        "control_frequency": int(len(clusters[concept_index][1])),
        "match_score": float(score),
    }


def random_pair_like(
    target_pair, *, grid_side: int, window_side: int, seed: int, avoid_target: bool = True,
):
    rng = np.random.default_rng(seed)
    target_types = [window_edge_type(window, grid_side, window_side) for window in target_pair["windows"]]
    all_windows = [(row, column) for row in range(grid_side - window_side + 1) for column in range(grid_side - window_side + 1)]
    rng.shuffle(all_windows)
    forbidden = set(target_pair["windows"]) if avoid_target else set()
    def valid_random_window(window, required_type):
        return (
            window not in forbidden
            and window_edge_type(window, grid_side, window_side) == required_type
            and not any(windows_overlap(window, target, window_side) for target in target_pair["windows"])
        )

    first_candidates = [window for window in all_windows if valid_random_window(window, target_types[0])]
    second_candidates = [window for window in all_windows if valid_random_window(window, target_types[1])]
    for first in first_candidates:
        for second in second_candidates:
            if first != second and not windows_overlap(first, second, window_side):
                return {"windows": [first, second]}
    return None


def gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = image.shape[0]
    horizontal = kernel.reshape(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.reshape(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    value = image.unsqueeze(0)
    value = F.conv2d(F.pad(value, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    value = F.conv2d(F.pad(value, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)
    return value[0]


def perturb_windows(
    image: torch.Tensor, windows, *, mode: str, blurred: torch.Tensor,
    patch_pixels: int = 14, window_patch_side: int = 4,
) -> torch.Tensor:
    output = image.clone()
    for row, column in windows:
        top, left = row * patch_pixels, column * patch_pixels
        bottom = top + window_patch_side * patch_pixels
        right = left + window_patch_side * patch_pixels
        if mode == "mean_fill":
            output[:, top:bottom, left:right] = 0.0
        elif mode == "local_blur":
            output[:, top:bottom, left:right] = blurred[:, top:bottom, left:right]
        else:
            raise ValueError(f"未知扰动：{mode}")
    return output


def pair_variants(image, pair, mode, blurred, window_patch_side):
    left, right = pair["windows"]
    return [
        perturb_windows(image, [left], mode=mode, blurred=blurred, window_patch_side=window_patch_side),
        perturb_windows(image, [right], mode=mode, blurred=blurred, window_patch_side=window_patch_side),
        perturb_windows(image, [left, right], mode=mode, blurred=blurred, window_patch_side=window_patch_side),
    ]


def _prepare_identity(args, config):
    require_clean_code_tree()
    if current_code_commit() != args.expected_commit:
        raise ValueError("IDEA-169代码commit身份错误。")
    if sha256_file(args.config) != args.expected_config_sha:
        raise ValueError("IDEA-169 config SHA错误。")


def prepare(args, config: dict, device: torch.device):
    _prepare_identity(args, config)
    asset_identity = validate_assets(config)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True

    import clip

    clip_model, _ = clip.load(config["clip_checkpoint"], device=device, jit=False)
    clip_model.eval()
    values = setup_concepts(config, clip_model, device)
    patches = np.load(config["final_patches"], mmap_mode="r")
    if patches.shape != (7057, 576, 768) or patches.dtype != np.float16:
        raise ValueError("正式576-patch资产shape/dtype错误。")
    reader, training = train_reader(values, patches, config, device)
    train_scores = reader_scores(reader, patches, values["train_rows"], values["queries"], device)
    evaluation_scores = reader_scores(reader, patches, values["evaluation_rows"], values["queries"], device)
    train_targets = values["targets"][values["train_rows"]]
    evaluation_targets = values["targets"][values["evaluation_rows"]]
    thresholds = calibrate_thresholds(train_scores, train_targets)
    aucs = np.asarray([roc_auc(evaluation_targets[:, index] > 0.5, evaluation_scores[:, index]) for index in range(evaluation_scores.shape[1])])
    selected_rows = stratified_rows(
        values["labels"], values["evaluation_classes"], int(config["evaluation_image_count"]), int(config["seed"]) + 800,
    )
    if not np.all(np.isin(values["labels"][selected_rows], values["evaluation_classes"])):
        raise ValueError("500图抽样越过pseudo-unseen类别边界。")

    checkpoint = args.output.with_suffix(".pth")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": reader.state_dict(),
        "concept_queries": values["queries"],
        "clusters": [(int(role), list(map(int, members))) for role, members in values["clusters"]],
        "names": values["names"],
        "thresholds": torch.from_numpy(thresholds),
        "train_classes": values["train_classes"],
        "evaluation_classes": values["evaluation_classes"],
        "selected_rows": torch.from_numpy(selected_rows),
        "selected_labels": torch.from_numpy(values["labels"][selected_rows].astype(np.int64)),
        "code_commit": args.expected_commit,
        "config_sha256": args.expected_config_sha,
        "asset_identity": asset_identity,
    }
    torch.save(payload, checkpoint)
    result = {
        "schema_version": "gzsl-paper.contrastive-concept-interaction-prepared.v1",
        "idea_id": "IDEA-169",
        "code_commit": args.expected_commit,
        "config_sha256": args.expected_config_sha,
        "asset_identity": asset_identity,
        "environment": environment(device),
        "training": training,
        "reader_concept_auc": metric_summary(aucs),
        "reader_gate_pass": float(np.median(aucs)) >= float(config["reader_auc_gate"]),
        "train_classes": values["train_classes"],
        "evaluation_classes": values["evaluation_classes"],
        "eligible_indices": list(map(int, config["eligible_indices"])),
        "concept_axis": [
            {"role": int(role), "members": list(map(int, members))}
            for role, members in values["clusters"]
        ],
        "thresholds": thresholds.tolist(),
        "selected_rows": selected_rows.tolist(),
        "selected_labels": values["labels"][selected_rows].astype(int).tolist(),
        "selected_image_count": int(len(selected_rows)),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "formal_unseen_images_used": False,
        "pseudo_unseen_images_used_for_gradient": False,
        "all_200_class_texts_used": True,
        "human_annotations_used": False,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reader_gate_pass": result["reader_gate_pass"], "reader_auc": result["reader_concept_auc"], "checkpoint": str(checkpoint)}, ensure_ascii=False))


def load_prepared(path: Path, expected_commit: str, expected_config_sha: str):
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != "gzsl-paper.contrastive-concept-interaction-prepared.v1"
        or result.get("idea_id") != "IDEA-169"
        or result.get("code_commit") != expected_commit
        or result.get("config_sha256") != expected_config_sha
        or result.get("formal_unseen_images_used") is not False
        or result.get("pseudo_unseen_images_used_for_gradient") is not False
        or result.get("all_200_class_texts_used") is not True
        or result.get("human_annotations_used") is not False
    ):
        raise ValueError("IDEA-169 prepared结果身份错误。")
    checkpoint = Path(result["checkpoint_path"])
    if sha256_file(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("IDEA-169 prepared checkpoint SHA错误。")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("code_commit") != expected_commit or payload.get("config_sha256") != expected_config_sha:
        raise ValueError("IDEA-169 checkpoint内部身份错误。")
    concept_axis = [
        {"role": int(role), "members": list(map(int, members))}
        for role, members in payload["clusters"]
    ]
    if (
        payload.get("asset_identity") != result.get("asset_identity")
        or payload.get("train_classes") != result.get("train_classes")
        or payload.get("evaluation_classes") != result.get("evaluation_classes")
        or payload["selected_rows"].tolist() != result.get("selected_rows")
        or payload["selected_labels"].tolist() != result.get("selected_labels")
        or payload["thresholds"].tolist() != result.get("thresholds")
        or concept_axis != result.get("concept_axis")
    ):
        raise ValueError("IDEA-169 prepared JSON与checkpoint语义身份不一致。")
    return result, payload


@torch.no_grad()
def intervene(args, config: dict, device: torch.device):
    _prepare_identity(args, config)
    prepared, payload = load_prepared(args.prepared, args.expected_commit, args.expected_config_sha)
    if validate_assets(config) != prepared["asset_identity"]:
        raise ValueError("intervene阶段资产身份与prepare不一致。")
    if not prepared["reader_gate_pass"]:
        raise RuntimeError("共享Reader没有复现，禁止执行交互Gate。")
    if int(args.shard_count) != int(config["shard_count"]) or not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("IDEA-169必须恰好使用配置声明的两个shard。")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True

    import clip

    clip_model, preprocess = clip.load(config["clip_checkpoint"], device=device, jit=False)
    clip_model.eval()
    reader = SharedConceptReadout(rank=int(config["reader_rank"])).to(device)
    reader.load_state_dict(payload["model_state_dict"], strict=True)
    reader.eval()
    concepts = payload["concept_queries"].to(device)
    adapted_queries = adapted_text_queries(reader, concepts)
    query_similarities = (adapted_queries @ adapted_queries.T).cpu().numpy()
    clusters = [(int(role), list(map(int, members))) for role, members in payload["clusters"]]
    names = list(payload["names"])
    thresholds = payload["thresholds"].numpy()
    selected_rows = payload["selected_rows"].numpy()
    shard_rows = selected_rows[int(args.shard_index)::int(args.shard_count)]

    source = yaml.safe_load(Path(config["source_config"]).read_text(encoding="utf-8"))
    split = load_xlsa_split(source["res101"], source["att_splits"])
    labels = torch.load(config["train_labels"], map_location="cpu", weights_only=True).long().numpy()
    expected_labels = split.labels.index_select(0, split.train_indices).numpy()
    if not np.array_equal(labels, expected_labels):
        raise ValueError("原图与576-patch训练行序不一致。")
    cached_patches = np.load(config["final_patches"], mmap_mode="r")
    grid_side = int(config["token_grid_side"])
    window_side = int(config["window_patch_side"])
    records = []
    image_summaries = []

    for train_row in shard_rows:
        train_row = int(train_row)
        class_id = int(labels[train_row])
        global_index = int(split.train_indices[train_row])
        image_path = resolve_xlsa_image_path(source["raw_root"], split.image_files[global_index], source["image_path_anchors"])
        with Image.open(image_path) as handle:
            image = preprocess(handle.convert("RGB"))
        image = image.to(device)
        raw_patches = encode_final_patches(clip_model, image.unsqueeze(0))
        cached = torch.from_numpy(np.asarray(cached_patches[[train_row]], dtype=np.float16).copy()).to(device).float()
        patch_cosine = float(F.cosine_similarity(raw_patches, cached, dim=-1).mean())
        logits, similarities, attentions = reader.details(raw_patches, concepts)
        logits_np = logits[0].cpu().numpy()
        attentions_np = attentions[0].cpu().numpy()
        true_concepts = [index for index, (_, members) in enumerate(clusters) if class_id in members]
        candidate_concepts = [index for index in true_concepts if logits_np[index] >= thresholds[index]]
        candidate_concepts = sorted(candidate_concepts, key=lambda index: (-float(logits_np[index]), index))[:int(config["max_concepts_per_image"])]
        blurred = gaussian_blur(image, int(config["blur_kernel_size"]), float(config["blur_sigma"]))
        accepted = 0
        neighbor_failures = 0

        for concept_index in candidate_concepts:
            contrastive_neighbors = choose_contrastive_neighbors(
                target_concept=concept_index,
                class_id=class_id,
                clusters=clusters,
                query_similarities=query_similarities,
                minimum_count=int(config["contrastive_minimum_neighbors"]),
                maximum_count=int(config["contrastive_maximum_neighbors"]),
                maximum_frequency_log_distance=float(config["hard_frequency_log_distance_max"]),
            )
            if contrastive_neighbors is None:
                neighbor_failures += 1
                continue
            target_pair = select_nonoverlap_pair(attentions_np[concept_index], grid_side, window_side)
            if target_pair is None:
                continue
            hard_pair = choose_hard_control(
                target_pair=target_pair,
                target_concept=concept_index,
                class_id=class_id,
                attentions=attentions_np,
                clusters=clusters,
                grid_side=grid_side,
                window_side=window_side,
                maximum_log_distance=float(config["hard_attention_log_distance_max"]),
                maximum_frequency_log_distance=float(config["hard_frequency_log_distance_max"]),
            )
            random_pair = random_pair_like(
                target_pair,
                grid_side=grid_side,
                window_side=window_side,
                seed=int(config["seed"]) + train_row * 1009 + concept_index * 9173,
            )
            if hard_pair is None or random_pair is None:
                continue
            original_margin = float(
                reader.logit_scale.exp().clamp(max=100.0)
                * fixed_attention_evidence(
                    contrastive_similarity_margin(
                        similarities,
                        concept_index,
                        contrastive_neighbors["indices"],
                    ),
                    attentions[0, concept_index].detach(),
                )[0]
            )
            record = {
                "train_row": train_row,
                "class_id": class_id,
                "concept_index": int(concept_index),
                "concept_name": names[concept_index],
                "original_logit": float(logits_np[concept_index]),
                "original_contrastive_margin": original_margin,
                "threshold": float(thresholds[concept_index]),
                "contrastive_neighbor_indices": contrastive_neighbors["indices"],
                "contrastive_neighbor_names": [names[index] for index in contrastive_neighbors["indices"]],
                "contrastive_neighbor_query_cosines": contrastive_neighbors["query_cosines"],
                "contrastive_neighbor_frequency_log_distances": contrastive_neighbors["frequency_log_distances"],
                "target_windows": [list(window) for window in target_pair["windows"]],
                "target_attention_masses": list(map(float, target_pair["masses"])),
                "hard_concept_index": int(hard_pair["concept_index"]),
                "hard_windows": [list(window) for window in hard_pair["windows"]],
                "hard_proposal_concept_attention_masses": list(map(float, hard_pair["proposal_concept_attention_masses"])),
                "hard_target_concept_attention_masses": list(map(float, hard_pair["target_concept_attention_masses"])),
                "hard_attention_log_distance": float(hard_pair["attention_log_distance"]),
                "hard_frequency_log_distance": float(hard_pair["frequency_log_distance"]),
                "target_concept_frequency": int(hard_pair["target_frequency"]),
                "hard_concept_frequency": int(hard_pair["control_frequency"]),
                "hard_target_window_overlap": any(
                    windows_overlap(target_window, hard_window, window_side)
                    for target_window in target_pair["windows"]
                    for hard_window in hard_pair["windows"]
                ),
                "random_windows": [list(window) for window in random_pair["windows"]],
                "primary_readout": "fixed_original_attention_contrast_margin",
                "perturbations": {},
            }
            pairs = {"target": target_pair, "hard": hard_pair, "random": random_pair}
            for mode in PERTURBATIONS:
                variants = []
                for pair_name in ("target", "hard", "random"):
                    variants.extend(pair_variants(image, pairs[pair_name], mode, blurred, window_side))
                perturbed_patches = encode_final_patches(clip_model, torch.stack(variants))
                contrastive_logits = fixed_contrastive_margin_logits(
                    reader,
                    perturbed_patches,
                    concepts,
                    concept_index,
                    contrastive_neighbors["indices"],
                    attentions[0, concept_index].detach(),
                ).cpu().numpy()
                fixed_target_logits = fixed_attention_logits(
                    reader,
                    perturbed_patches,
                    concepts,
                    concept_index,
                    attentions[0, concept_index].detach(),
                ).cpu().numpy()
                dynamic_logits = reader(perturbed_patches, concepts)[:, concept_index].cpu().numpy()
                row = {}
                for pair_position, pair_name in enumerate(("target", "hard", "random")):
                    score_a, score_b, score_union = map(
                        float, contrastive_logits[pair_position * 3:(pair_position + 1) * 3]
                    )
                    fixed_target_a, fixed_target_b, fixed_target_union = map(
                        float, fixed_target_logits[pair_position * 3:(pair_position + 1) * 3]
                    )
                    dynamic_a, dynamic_b, dynamic_union = map(
                        float, dynamic_logits[pair_position * 3:(pair_position + 1) * 3]
                    )
                    row[pair_name] = {
                        "score_a": score_a,
                        "score_b": score_b,
                        "score_union": score_union,
                        "eta": interaction_eta(original_margin, score_a, score_b, score_union),
                        "fixed_target_eta_diagnostic": interaction_eta(
                            float(logits_np[concept_index]),
                            fixed_target_a,
                            fixed_target_b,
                            fixed_target_union,
                        ),
                        "dynamic_attention_eta_diagnostic": interaction_eta(
                            float(logits_np[concept_index]), dynamic_a, dynamic_b, dynamic_union
                        ),
                    }
                row["hard_magnitude_excess"] = magnitude_excess(row["target"]["eta"], row["hard"]["eta"])
                row["random_magnitude_excess"] = magnitude_excess(row["target"]["eta"], row["random"]["eta"])
                record["perturbations"][mode] = row
            records.append(record)
            accepted += 1
        image_summaries.append(
            {
                "train_row": train_row,
                "class_id": class_id,
                "patch_cosine": patch_cosine,
                "true_concept_count": len(true_concepts),
                "threshold_candidate_count": len(candidate_concepts),
                "contrastive_neighbor_failure_count": neighbor_failures,
                "accepted_interaction_count": accepted,
            }
        )

    result = {
        "schema_version": "gzsl-paper.contrastive-concept-interaction-shard.v1",
        "idea_id": "IDEA-169",
        "code_commit": args.expected_commit,
        "config_sha256": args.expected_config_sha,
        "prepared_sha256": sha256_file(args.prepared),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "environment": environment(device),
        "image_rows": list(map(int, shard_rows)),
        "image_summaries": image_summaries,
        "records": records,
        "formal_unseen_images_used": False,
        "pseudo_unseen_images_used_for_gradient": False,
        "all_200_class_texts_used": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"shard": int(args.shard_index), "images": len(shard_rows), "interaction_pairs": len(records)}, ensure_ascii=False))


def hierarchical_bootstrap(values, class_ids, image_ids, replicates: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    image_ids = np.asarray(image_ids, dtype=np.int64)
    if not (len(values) == len(class_ids) == len(image_ids)) or len(values) == 0:
        raise ValueError("层级bootstrap输入为空或轴不一致。")
    image_values = defaultdict(list)
    image_class = {}
    for value, class_id, image_id in zip(values, class_ids, image_ids):
        image_values[int(image_id)].append(float(value))
        image_class[int(image_id)] = int(class_id)
    per_image = {image_id: float(np.mean(rows)) for image_id, rows in image_values.items()}
    class_images = defaultdict(list)
    for image_id, class_id in image_class.items():
        class_images[class_id].append(image_id)
    for class_id in class_images:
        class_images[class_id] = sorted(class_images[class_id])
    classes = np.asarray(sorted(class_images), dtype=np.int64)
    class_means = [np.mean([per_image[image_id] for image_id in class_images[int(class_id)]]) for class_id in classes]
    point = float(np.mean(class_means))
    class_means_array = np.asarray(class_means, dtype=np.float64)
    scale = float(class_means_array.std(ddof=1)) if len(class_means_array) > 1 else 0.0
    standardized = point / max(scale, 1e-12)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled_classes = rng.choice(classes, size=len(classes), replace=True)
        sampled_class_means = []
        for class_id in sampled_classes:
            images = np.asarray(class_images[int(class_id)], dtype=np.int64)
            sampled_images = rng.choice(images, size=len(images), replace=True)
            sampled_class_means.append(np.mean([per_image[int(image_id)] for image_id in sampled_images]))
        draws[replicate] = float(np.mean(sampled_class_means))
    return {
        "point": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "standardized_effect": float(standardized),
        "observation_count": int(len(values)),
        "image_count": int(len(per_image)),
        "class_count": int(len(classes)),
    }


def comparable_environment(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "gpu_uuid"}


def merge(args, config: dict):
    _prepare_identity(args, config)
    prepared, _ = load_prepared(args.prepared, args.expected_commit, args.expected_config_sha)
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.shards]
    if len(shards) != int(config["shard_count"]):
        raise ValueError("IDEA-169 merge必须恰好接收两个shard。")
    expected_rows = sorted(map(int, prepared["selected_rows"]))
    expected_label_by_row = {
        int(row): int(label)
        for row, label in zip(prepared["selected_rows"], prepared["selected_labels"])
    }
    delivered_rows = sorted(row for shard in shards for row in shard["image_rows"])
    if delivered_rows != expected_rows or len(delivered_rows) != len(set(delivered_rows)):
        raise ValueError("双卡shard没有唯一、完整覆盖固定500图。")
    if sorted(int(shard["shard_index"]) for shard in shards) != list(range(len(shards))):
        raise ValueError("shard编号不完整。")
    for shard, path in zip(shards, args.shards):
        if (
            shard.get("schema_version") != "gzsl-paper.contrastive-concept-interaction-shard.v1"
            or shard.get("idea_id") != "IDEA-169"
            or shard.get("code_commit") != args.expected_commit
            or shard.get("config_sha256") != args.expected_config_sha
            or shard.get("prepared_sha256") != sha256_file(args.prepared)
            or int(shard.get("shard_count", -1)) != int(config["shard_count"])
            or shard.get("formal_unseen_images_used") is not False
            or shard.get("pseudo_unseen_images_used_for_gradient") is not False
            or shard.get("all_200_class_texts_used") is not True
        ):
            raise ValueError(f"shard身份错误：{path}")
        shard_rows = list(map(int, shard["image_rows"]))
        summary_rows = [int(row["train_row"]) for row in shard.get("image_summaries", [])]
        if (
            sorted(summary_rows) != sorted(shard_rows)
            or len(summary_rows) != len(set(summary_rows))
            or any(int(row["class_id"]) != expected_label_by_row.get(int(row["train_row"])) for row in shard.get("image_summaries", []))
        ):
            raise ValueError(f"shard summary没有唯一覆盖本片图像：{path}")
        seen_record_keys = set()
        shard_row_set = set(shard_rows)
        for record in shard.get("records", []):
            row = int(record["train_row"])
            key = (row, int(record["concept_index"]))
            if (
                row not in shard_row_set
                or int(record["class_id"]) != expected_label_by_row.get(row)
                or key in seen_record_keys
                or record.get("primary_readout") != "fixed_original_attention_contrast_margin"
            ):
                raise ValueError(f"shard record归属或读出口径错误：{path}")
            seen_record_keys.add(key)
    expected_environment = comparable_environment(prepared["environment"])
    if any(comparable_environment(shard["environment"]) != expected_environment for shard in shards):
        raise ValueError("prepare与双卡shard的环境/GPU型号不一致。")
    records = [record for shard in shards for record in shard["records"]]
    summaries = [row for shard in shards for row in shard["image_summaries"]]
    class_ids = np.asarray([record["class_id"] for record in records], dtype=np.int64)
    image_ids = np.asarray([record["train_row"] for record in records], dtype=np.int64)
    enough_pairs = len(records) >= int(config["minimum_interaction_pairs"])
    enough_classes = len(np.unique(class_ids)) >= int(config["minimum_interaction_classes"]) if records else False
    parity = np.asarray([row["patch_cosine"] for row in summaries], dtype=np.float64)
    patch_identity_pass = bool(
        len(parity) == len(expected_rows)
        and len(parity) > 0
        and float(parity.min()) >= float(config["patch_identity_gate"])
    )

    statistics = {}
    effect_conditions = {}
    if records:
        for perturbation_index, perturbation in enumerate(PERTURBATIONS):
            statistics[perturbation] = {}
            for control_index, control in enumerate(CONTROLS):
                values = np.asarray(
                    [record["perturbations"][perturbation][f"{control}_magnitude_excess"] for record in records],
                    dtype=np.float64,
                )
                row = hierarchical_bootstrap(
                    values,
                    class_ids,
                    image_ids,
                    int(config["bootstrap_replicates"]),
                    int(config["seed"]) + 10000 + perturbation_index * 100 + control_index,
                )
                statistics[perturbation][control] = row
                effect_conditions[f"{perturbation}_{control}"] = (
                    row["ci95_low"] > 0.0
                    and row["standardized_effect"] >= float(config["standardized_effect_gate"])
                )
        eta_mean = np.asarray([record["perturbations"]["mean_fill"]["target"]["eta"] for record in records])
        eta_blur = np.asarray([record["perturbations"]["local_blur"]["target"]["eta"] for record in records])
        concordance = np.sign(eta_mean) * np.sign(eta_blur)
        sign_stability = hierarchical_bootstrap(
            concordance,
            class_ids,
            image_ids,
            int(config["bootstrap_replicates"]),
            int(config["seed"]) + 12000,
        )
        sign_stability_pass = sign_stability["ci95_low"] > 0.0
        sign_report = {
            perturbation: {
                "negative_fraction_complement_candidate": float(np.mean(np.asarray([record["perturbations"][perturbation]["target"]["eta"] for record in records]) < 0)),
                "positive_fraction_redundancy_candidate": float(np.mean(np.asarray([record["perturbations"][perturbation]["target"]["eta"] for record in records]) > 0)),
                "zero_fraction": float(np.mean(np.asarray([record["perturbations"][perturbation]["target"]["eta"] for record in records]) == 0)),
            }
            for perturbation in PERTURBATIONS
        }
    else:
        sign_stability = None
        sign_stability_pass = False
        sign_report = {}

    conditions = {
        "reader_reproduced": bool(prepared["reader_gate_pass"]),
        "fixed_500_images_complete": delivered_rows == expected_rows,
        "minimum_interaction_pairs": enough_pairs,
        "minimum_interaction_classes": enough_classes,
        "raw_cached_patch_identity": patch_identity_pass,
        **effect_conditions,
        "cross_perturbation_sign_stability": sign_stability_pass,
    }
    decision = "gate_pass_proof_of_path" if all(conditions.values()) else "gate_fail_stop_direction"
    result = {
        "schema_version": "gzsl-paper.contrastive-concept-interaction-result.v1",
        "idea_id": "IDEA-169",
        "code_commit": args.expected_commit,
        "config_sha256": args.expected_config_sha,
        "decision": decision,
        "performance_status": "proof_of_path" if decision == "gate_pass_proof_of_path" else "rejected_at_gate0",
        "conditions": conditions,
        "reader_concept_auc": prepared["reader_concept_auc"],
        "evaluation_image_count": len(delivered_rows),
        "interaction_pair_count": len(records),
        "interaction_class_count": int(len(np.unique(class_ids))) if records else 0,
        "images_with_interactions": int(len(set(image_ids.tolist()))) if records else 0,
        "patch_identity": {
            "mean": float(parity.mean()) if len(parity) else None,
            "minimum": float(parity.min()) if len(parity) else None,
            "gate": float(config["patch_identity_gate"]),
        },
        "magnitude_excess_statistics": statistics,
        "cross_perturbation_sign_stability": sign_stability,
        "sign_interpretation_candidates_only": sign_report,
        "formal_unseen_images_used": False,
        "pseudo_unseen_images_used_for_gradient": False,
        "all_200_class_texts_used": True,
        "reports_H_U_S_ZS": False,
        "interpretation_boundary": "eta<0仅为互补候选，eta>0仅为冗余候选；输入扰动不等于真实因果删除。",
        "perturbation_semantics": {
            "mean_fill": "CLIP归一化空间的0，即CLIP全局通道均值颜色；不等同真实删除。",
            "local_blur": "只用全图高斯模糊值替换固定窗口；不等同生成式补全。",
        },
        "shard_sha256": {str(path): sha256_file(path) for path in args.shards},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "pairs": len(records), "classes": result["interaction_class_count"], "conditions": conditions}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    intervene_parser = commands.add_parser("intervene")
    intervene_parser.add_argument("--prepared", type=Path, required=True)
    intervene_parser.add_argument("--shard-index", type=int, required=True)
    intervene_parser.add_argument("--shard-count", type=int, default=2)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--prepared", type=Path, required=True)
    merge_parser.add_argument("--shards", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        prepare(args, config, torch.device(args.device))
    elif args.command == "intervene":
        intervene(args, config, torch.device(args.device))
    else:
        merge(args, config)


if __name__ == "__main__":
    main()
