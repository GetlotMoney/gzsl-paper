"""Minimal class-disjoint falsification for tri-state visual evidence predicates."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from tools.diagnose_intermediate_patch_concepts import phrase_embeddings
from tools.diagnose_learnable_concept_readout import prompted_embeddings
from tools.gzsl_data import load_xlsa_split, resolve_xlsa_image_path
from tools.run_contract import current_code_commit, require_clean_code_tree
from tools.runtime import sha256_file


ROLE_COUNT = 6
EMBEDDING_DIMENSION = 768


class PredicateReader(nn.Module):
    """One class-agnostic text-to-patch reader shared by every role and class."""

    def __init__(self, rank: int = 64):
        super().__init__()
        self.visual_down = nn.Linear(EMBEDDING_DIMENSION, rank, bias=False)
        self.visual_up = nn.Linear(rank, EMBEDDING_DIMENSION, bias=False)
        self.text_down = nn.Linear(EMBEDDING_DIMENSION, rank, bias=False)
        self.text_up = nn.Linear(rank, EMBEDDING_DIMENSION, bias=False)
        nn.init.zeros_(self.visual_up.weight)
        nn.init.zeros_(self.text_up.weight)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def evidence(
        self,
        patches: torch.Tensor,
        predicates: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        patches = F.normalize(patches.float(), dim=-1)
        predicates = F.normalize(predicates.float(), dim=-1)
        visual = F.normalize(
            patches + self.visual_up(F.gelu(self.visual_down(patches))), dim=-1
        )
        text = F.normalize(
            predicates + self.text_up(F.gelu(self.text_down(predicates))), dim=-1
        )
        similarities = torch.matmul(visual, text.T)
        attention = torch.softmax(similarities / 0.07, dim=1)
        pooled = (attention * similarities).sum(dim=1)
        logits = self.logit_scale.exp().clamp(max=100.0) * pooled + self.bias
        return (logits, attention) if return_attention else logits

    def forward(self, patches: torch.Tensor, predicates: torch.Tensor) -> torch.Tensor:
        return self.evidence(patches, predicates)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "idea_id",
        "dataset",
        "role_texts",
        "role_texts_sha256",
        "clip_checkpoint",
        "clip_checkpoint_sha256",
        "source_config",
        "source_config_sha256",
        "visual_asset_manifest",
        "visual_asset_manifest_sha256",
        "role_asset_manifest",
        "role_asset_manifest_sha256",
        "train_labels",
        "train_features",
        "role_sentence_embeds",
        "final_patches",
        "seed",
        "train_class_count",
        "evaluation_class_count",
        "rank",
        "bag_size",
        "hard_negative_count",
        "batch_class_count",
        "updates",
        "learning_rate",
        "weight_decay",
        "evaluation_batch_size",
        "deletion_count",
        "pairwise_accuracy_gate",
        "error_correction_gate",
        "correct_damage_gate",
        "deletion_gate",
        "raw_patch_cosine_gate",
        "unseen_images_used",
        "human_annotations_used",
    }
    actual = set(config) if isinstance(config, dict) else set()
    if actual != required:
        raise ValueError(f"三态诊断配置字段错误：缺少={sorted(required-actual)}，多出={sorted(actual-required)}")
    if (
        config["schema_version"] != "gzsl-paper.tristate-predicate-diagnostic.v1"
        or config["idea_id"] != "IDEA-163"
        or config["dataset"] != "CUB"
        or config["unseen_images_used"] is not False
        or config["human_annotations_used"] is not False
    ):
        raise ValueError("三态诊断身份或数据边界错误。")
    if int(config["train_class_count"]) != 100 or int(config["evaluation_class_count"]) != 50:
        raise ValueError("三态诊断固定100/50 class-disjoint划分。")
    return config


def split_classes(labels: np.ndarray, seed: int, train_count: int):
    classes = np.unique(labels)
    if len(classes) != 150:
        raise ValueError("三态诊断固定150个formal-seen类别。")
    order = np.random.default_rng(seed).permutation(classes)
    return np.sort(order[:train_count]), np.sort(order[train_count:])


def nearest_same_role(predicates: torch.Tensor, class_ids: np.ndarray, count: int) -> np.ndarray:
    """Return global class ids of text-nearest alternatives for each class and role."""
    values = F.normalize(predicates[class_ids].float(), dim=-1)
    result = np.empty((len(class_ids), ROLE_COUNT, count), dtype=np.int64)
    for role in range(ROLE_COUNT):
        similarities = values[:, role] @ values[:, role].T
        similarities.fill_diagonal_(-9)
        local = similarities.topk(count, dim=1).indices.numpy()
        result[:, role] = class_ids[local]
    return result


def class_rows(labels: np.ndarray, classes: np.ndarray) -> dict[int, np.ndarray]:
    return {int(class_id): np.flatnonzero(labels == class_id) for class_id in classes}


def shuffled_query_map(classes: np.ndarray, seed: int, enabled: bool) -> dict[int, int]:
    if not enabled:
        return {int(value): int(value) for value in classes}
    rng = np.random.default_rng(seed + 1000)
    shuffled = rng.permutation(classes)
    while np.any(shuffled == classes):
        shuffled = rng.permutation(classes)
    return {int(source): int(target) for source, target in zip(classes, shuffled)}


def train_reader(
    *,
    patches: np.ndarray,
    labels: np.ndarray,
    predicates: torch.Tensor,
    train_classes: np.ndarray,
    config: dict,
    device: torch.device,
    shuffle_labels: bool,
):
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    reader = PredicateReader(int(config["rank"])).to(device)
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    rows_by_class = class_rows(labels, train_classes)
    query_map = shuffled_query_map(train_classes, int(config["seed"]), shuffle_labels)
    alternatives = nearest_same_role(
        predicates,
        train_classes,
        int(config["hard_negative_count"]),
    )
    local_index = {int(class_id): index for index, class_id in enumerate(train_classes)}
    rng = np.random.default_rng(int(config["seed"]))
    losses = []
    reader.train()
    for _ in range(int(config["updates"])):
        sources = rng.choice(
            train_classes,
            size=int(config["batch_class_count"]),
            replace=False,
        )
        objective = []
        for source in sources:
            source = int(source)
            mapped = query_map[source]
            available = rows_by_class[source]
            bag_rows = rng.choice(
                available,
                size=int(config["bag_size"]),
                replace=len(available) < int(config["bag_size"]),
            )
            bag = torch.from_numpy(np.asarray(patches[bag_rows], dtype=np.float16).copy()).to(device)
            neighbor_row = alternatives[local_index[mapped]]
            query_ids = []
            for role in range(ROLE_COUNT):
                query_ids.append([mapped, *neighbor_row[role].tolist()])
            queries = torch.stack(
                [predicates[class_id, role] for role, ids in enumerate(query_ids) for class_id in ids]
            ).to(device)
            logits = reader(bag, queries).reshape(len(bag_rows), ROLE_COUNT, -1)
            bag_logits = torch.logsumexp(logits, dim=0) - math.log(len(bag_rows))
            targets = torch.zeros(ROLE_COUNT, dtype=torch.long, device=device)
            objective.append(F.cross_entropy(bag_logits, targets))
        loss = torch.stack(objective).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return reader.eval(), {
        "initial_loss_mean_20": float(np.mean(losses[:20])),
        "final_loss_mean_20": float(np.mean(losses[-20:])),
    }


@torch.no_grad()
def score_class_predicates(
    reader: PredicateReader,
    patches: np.ndarray,
    rows: np.ndarray,
    predicates: torch.Tensor,
    class_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    queries = predicates[class_ids].reshape(-1, EMBEDDING_DIMENSION).to(device)
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = torch.from_numpy(np.asarray(patches[batch_rows], dtype=np.float16).copy()).to(device)
        values = reader(batch, queries).reshape(len(batch_rows), len(class_ids), ROLE_COUNT)
        outputs.append(values.cpu())
    return torch.cat(outputs).numpy()


def _balanced_threshold(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative)).astype(np.float64)
    labels = np.concatenate(
        (np.ones(len(positive), dtype=np.int8), np.zeros(len(negative), dtype=np.int8))
    )
    order = np.argsort(values, kind="mergesort")[::-1]
    sorted_values = values[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1 - sorted_labels)
    tpr = true_positive / max(len(positive), 1)
    tnr = 1.0 - false_positive / max(len(negative), 1)
    balanced = 0.5 * (tpr + tnr)
    best = int(np.argmax(balanced))
    return float(sorted_values[best])


def calibrate_support_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    predicates: torch.Tensor,
    negative_count: int,
) -> np.ndarray:
    """Calibrate one absolute support threshold per role from fixed 1-vs-K pairs."""
    alternatives = nearest_same_role(predicates, class_ids, negative_count)
    positions = {int(class_id): index for index, class_id in enumerate(class_ids)}
    positive_by_role = [[] for _ in range(ROLE_COUNT)]
    negative_by_role = [[] for _ in range(ROLE_COUNT)]
    for row_index, class_id in enumerate(labels):
        local = positions[int(class_id)]
        for role in range(ROLE_COUNT):
            positive_by_role[role].append(scores[row_index, local, role])
            negative_local = [positions[int(value)] for value in alternatives[local, role]]
            negative_by_role[role].extend(scores[row_index, negative_local, role].tolist())
    return np.asarray(
        [
            _balanced_threshold(
                np.asarray(positive_by_role[role]), np.asarray(negative_by_role[role])
            )
            for role in range(ROLE_COUNT)
        ],
        dtype=np.float32,
    )


def tristate_ledger(
    scores: np.ndarray,
    class_ids: np.ndarray,
    predicates: torch.Tensor,
    thresholds: np.ndarray,
    negative_count: int,
):
    """Return class evidence plus support(+1), refute(-1), unknown(0) states."""
    alternatives = nearest_same_role(predicates, class_ids, negative_count)
    positions = {int(class_id): index for index, class_id in enumerate(class_ids)}
    alternative_scores = np.empty_like(scores)
    for local in range(len(class_ids)):
        for role in range(ROLE_COUNT):
            negative_local = [positions[int(value)] for value in alternatives[local, role]]
            alternative_scores[:, local, role] = scores[:, negative_local, role].max(axis=1)
    threshold = thresholds[None, None, :]
    support = (scores >= threshold) & (scores > alternative_scores)
    refute = (alternative_scores >= threshold) & (alternative_scores > scores)
    states = support.astype(np.int8) - refute.astype(np.int8)
    margins = scores - alternative_scores
    contributions = np.where(states != 0, margins, 0.0).astype(np.float32)
    observed = (states != 0).sum(axis=2)
    evidence = contributions.sum(axis=2) / np.maximum(observed, 1)
    return evidence, states, contributions, alternatives


def pairwise_hard_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    predicates: torch.Tensor,
    negative_count: int,
) -> float:
    alternatives = nearest_same_role(predicates, class_ids, negative_count)
    positions = {int(class_id): index for index, class_id in enumerate(class_ids)}
    correct = 0
    total = 0
    for row_index, class_id in enumerate(labels):
        local = positions[int(class_id)]
        for role in range(ROLE_COUNT):
            negatives = [positions[int(value)] for value in alternatives[local, role]]
            correct += int(scores[row_index, local, role] > scores[row_index, negatives, role].max())
            total += 1
    return correct / total


def mean8_predictions(
    features: torch.Tensor,
    labels: np.ndarray,
    rows: np.ndarray,
    role_embeddings: torch.Tensor,
    class_ids: np.ndarray,
):
    prototypes = F.normalize(role_embeddings.float().mean(dim=1), dim=-1)
    logits = F.normalize(features[rows].float(), dim=-1) @ prototypes[class_ids].T
    order = logits.argsort(dim=1, descending=True).numpy()
    predictions = class_ids[order[:, 0]]
    true_positions = {int(value): index for index, value in enumerate(class_ids)}
    true_local = np.asarray([true_positions[int(value)] for value in labels], dtype=np.int64)
    in_top5 = np.asarray([true_local[index] in order[index, :5] for index in range(len(rows))])
    return predictions, order, true_local, in_top5


def correction_metrics(
    evidence: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    parent_predictions: np.ndarray,
    true_local: np.ndarray,
    true_in_top5: np.ndarray,
):
    positions = {int(value): index for index, value in enumerate(class_ids)}
    parent_local = np.asarray([positions[int(value)] for value in parent_predictions])
    parent_correct = parent_predictions == labels
    eligible_errors = (~parent_correct) & true_in_top5
    preference = evidence[np.arange(len(labels)), true_local] > evidence[
        np.arange(len(labels)), parent_local
    ]
    correction = float(preference[eligible_errors].mean()) if eligible_errors.any() else 0.0
    evidence_predictions = class_ids[evidence.argmax(axis=1)]
    wrong = evidence.copy()
    wrong[np.arange(len(labels)), true_local] = -np.inf
    strict_wrong_preference = wrong.max(axis=1) > evidence[
        np.arange(len(labels)), true_local
    ]
    damage = float(strict_wrong_preference[parent_correct].mean())
    return {
        "parent_correct_count": int(parent_correct.sum()),
        "eligible_parent_error_count": int(eligible_errors.sum()),
        "error_pair_true_preferred_fraction": correction,
        "correct_sample_evidence_reversal_fraction": damage,
        "evidence_only_accuracy": float((evidence_predictions == labels).mean()),
    }


@torch.no_grad()
def encode_final_patches(clip_model, images: torch.Tensor) -> torch.Tensor:
    visual = clip_model.visual
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
    x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    return F.normalize(x.float(), dim=-1)


def predicate_contribution(logits: torch.Tensor, threshold: float) -> tuple[int, float]:
    """Interpret [candidate, fixed hard alternatives] as support/refute/unknown."""
    candidate = float(logits[0])
    alternative = float(logits[1:].max())
    if candidate >= threshold and candidate > alternative:
        return 1, candidate - alternative
    if alternative >= threshold and alternative > candidate:
        return -1, candidate - alternative
    return 0, 0.0


@torch.no_grad()
def deletion_test(
    *,
    reader: PredicateReader,
    clip_model,
    preprocess,
    split,
    source: dict,
    train_labels: np.ndarray,
    train_classes: np.ndarray,
    cached_patches: np.ndarray,
    predicates: torch.Tensor,
    thresholds: np.ndarray,
    negative_count: int,
    count: int,
    seed: int,
    device: torch.device,
):
    candidate_rows = np.flatnonzero(np.isin(train_labels, train_classes))
    rng = np.random.default_rng(seed + 3000)
    rng.shuffle(candidate_rows)
    alternatives = nearest_same_role(predicates, train_classes, negative_count)
    class_positions = {int(class_id): index for index, class_id in enumerate(train_classes)}
    selected_drops = []
    random_drops = []
    examples = []
    raw_patch_minimum_cosine = 1.0
    attempted = 0
    for train_row in candidate_rows:
        if len(selected_drops) >= count:
            break
        attempted += 1
        class_id = int(train_labels[train_row])
        local = class_positions[class_id]
        global_index = int(split.train_indices[train_row])
        path = resolve_xlsa_image_path(
            source["raw_root"], split.image_files[global_index], source["image_path_anchors"]
        )
        with Image.open(path) as handle:
            image = preprocess(handle.convert("RGB"))
        original_raw = encode_final_patches(clip_model, image.unsqueeze(0).to(device))
        cached = torch.from_numpy(
            np.asarray(cached_patches[[train_row]], dtype=np.float16).copy()
        ).to(device)
        raw_patch_minimum_cosine = min(
            raw_patch_minimum_cosine,
            float(F.cosine_similarity(original_raw, cached.float(), dim=-1).min()),
        )
        query_ids = [
            [class_id, *alternatives[local, role].tolist()] for role in range(ROLE_COUNT)
        ]
        all_queries = torch.stack(
            [predicates[value, role] for role, ids in enumerate(query_ids) for value in ids]
        ).to(device)
        logits, attention = reader.evidence(
            original_raw, all_queries, return_attention=True
        )
        role_logits = logits.reshape(ROLE_COUNT, negative_count + 1)
        support_candidates = []
        for role in range(ROLE_COUNT):
            state, contribution = predicate_contribution(
                role_logits[role], float(thresholds[role])
            )
            if state == 1:
                support_candidates.append((contribution, role))
        if not support_candidates:
            continue
        original_contribution, role = max(support_candidates)
        query_offset = role * (negative_count + 1)
        patch_index = int(attention[0, :, query_offset].argmax())
        random_index = int(rng.integers(0, 576))
        if random_index == patch_index:
            random_index = (random_index + 1) % 576
        selected_image = image.clone()
        random_image = image.clone()
        for target, index in ((selected_image, patch_index), (random_image, random_index)):
            row, column = divmod(index, 24)
            target[:, row * 14 : (row + 1) * 14, column * 14 : (column + 1) * 14] = 0.0
        masked = torch.stack((selected_image, random_image)).to(device)
        masked_patches = encode_final_patches(clip_model, masked)
        selected_queries = all_queries[query_offset : query_offset + negative_count + 1]
        masked_logits = reader(masked_patches, selected_queries)
        _, selected_contribution = predicate_contribution(
            masked_logits[0], float(thresholds[role])
        )
        _, random_contribution = predicate_contribution(
            masked_logits[1], float(thresholds[role])
        )
        selected_drop = original_contribution - selected_contribution
        random_drop = original_contribution - random_contribution
        selected_drops.append(selected_drop)
        random_drops.append(random_drop)
        if len(examples) < 20:
            examples.append(
                {
                    "train_row": int(train_row),
                    "class_id": class_id,
                    "role": role,
                    "selected_patch": patch_index,
                    "random_patch": random_index,
                    "original_support_contribution": original_contribution,
                    "selected_drop": selected_drop,
                    "random_drop": random_drop,
                }
            )
    if len(selected_drops) != count:
        raise RuntimeError(
            f"只有{len(selected_drops)}个唯一图像形成可删除support证据，要求{count}。"
        )
    selected_values = np.asarray(selected_drops)
    random_values = np.asarray(random_drops)
    return {
        "count": int(count),
        "attempted_unique_images": int(attempted),
        "raw_patch_minimum_cosine": raw_patch_minimum_cosine,
        "selected_drop_mean": float(selected_values.mean()),
        "random_drop_mean": float(random_values.mean()),
        "selected_drop_greater_fraction": float((selected_values > random_values).mean()),
        "examples": examples,
    }


def validate_asset_identity(config: dict) -> dict:
    visual_path = Path(config["visual_asset_manifest"])
    role_path = Path(config["role_asset_manifest"])
    source_path = Path(config["source_config"])
    if sha256_file(visual_path) != config["visual_asset_manifest_sha256"]:
        raise ValueError("正式576视觉manifest SHA错误。")
    if sha256_file(role_path) != config["role_asset_manifest_sha256"]:
        raise ValueError("text-v2角色manifest SHA错误。")
    if sha256_file(source_path) != config["source_config_sha256"]:
        raise ValueError("原图source config SHA错误。")
    if sha256_file(Path(config["role_texts"])) != config["role_texts_sha256"]:
        raise ValueError("角色原文SHA错误。")
    if sha256_file(Path(config["clip_checkpoint"])) != config["clip_checkpoint_sha256"]:
        raise ValueError("CLIP checkpoint SHA错误。")
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    role = json.loads(role_path.read_text(encoding="utf-8"))
    if (
        visual.get("schema_version") != "gzsl-paper.projected-patch-assets.v1"
        or visual.get("patch_shape") != [576, 768]
        or visual.get("patch_extraction", {}).get("patch_grid") != [24, 24]
        or role.get("schema_version") != "gzsl-paper.clip-assets.v1"
        or visual.get("class_order_sha256") != role.get("class_order_sha256")
    ):
        raise ValueError("视觉与角色资产schema、patch或类别轴身份错误。")
    bindings = {
        "train_labels": (visual_path.parent, "train_labels.pt", visual),
        "train_features": (visual_path.parent, "train_features.pt", visual),
        "final_patches": (visual_path.parent, "train_patch_features.npy", visual),
        "role_sentence_embeds": (role_path.parent, "role_sentence_embeds.pt", role),
    }
    for config_key, (parent, filename, manifest) in bindings.items():
        path = Path(config[config_key])
        if path.resolve() != (parent / filename).resolve():
            raise ValueError(f"{config_key}没有绑定manifest声明路径。")
        if sha256_file(path) != manifest.get("outputs_sha256", {}).get(filename):
            raise ValueError(f"{config_key}文件SHA与manifest不一致。")
    return {
        "visual_asset_manifest": str(visual_path),
        "visual_asset_manifest_sha256": config["visual_asset_manifest_sha256"],
        "visual_asset_id": visual.get("asset_id"),
        "role_asset_manifest": str(role_path),
        "role_asset_manifest_sha256": config["role_asset_manifest_sha256"],
        "role_asset_id": role.get("asset_id"),
        "class_order_sha256": visual.get("class_order_sha256"),
        "role_texts_sha256": config["role_texts_sha256"],
        "clip_checkpoint_sha256": config["clip_checkpoint_sha256"],
        "source_config_sha256": config["source_config_sha256"],
    }


def environment_fingerprint(device: torch.device) -> dict:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
    }


def write_ledger(
    path: Path,
    *,
    states: np.ndarray,
    contributions: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    thresholds: np.ndarray,
    alternatives: np.ndarray,
) -> str:
    np.savez_compressed(
        path,
        states=states.astype(np.int8),
        contributions=contributions.astype(np.float32),
        labels=labels.astype(np.int64),
        class_ids=class_ids.astype(np.int64),
        thresholds=thresholds.astype(np.float32),
        hard_alternatives=alternatives.astype(np.int64),
    )
    return sha256_file(path)


def run(
    config: dict,
    config_path: Path,
    config_sha256: str,
    expected_commit: str,
    output: Path,
    device: torch.device,
    shuffle_labels: bool,
):
    import clip

    require_clean_code_tree()
    if current_code_commit() != expected_commit:
        raise ValueError("三态诊断expected commit与clean HEAD不一致。")
    if sha256_file(config_path) != config_sha256:
        raise ValueError("三态诊断config SHA不一致。")
    output.parent.mkdir(parents=True, exist_ok=True)
    asset_identity = validate_asset_identity(config)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    labels = torch.load(config["train_labels"], map_location="cpu", weights_only=True).long().numpy()
    features = torch.load(config["train_features"], map_location="cpu", weights_only=True)
    role_embeddings = torch.load(
        config["role_sentence_embeds"], map_location="cpu", weights_only=True
    )
    patches = np.load(config["final_patches"], mmap_mode="r")
    if (
        labels.shape != (7057,)
        or features.shape != (7057, 768)
        or role_embeddings.shape != (200, 8, 768)
        or patches.shape != (7057, 576, 768)
        or patches.dtype != np.float16
    ):
        raise ValueError("三态诊断正式资产shape错误。")
    clip_model, preprocess = clip.load(config["clip_checkpoint"], device=device, jit=False)
    clip_model.eval()
    role_texts = json.loads(Path(config["role_texts"]).read_text(encoding="utf-8"))
    _, phrases = phrase_embeddings(clip_model, role_texts["descriptions"], device)
    predicates = prompted_embeddings(clip_model, phrases, device).permute(1, 0, 2).cpu()
    train_classes, evaluation_classes = split_classes(
        labels, int(config["seed"]), int(config["train_class_count"])
    )
    train_rows = np.flatnonzero(np.isin(labels, train_classes))
    evaluation_rows = np.flatnonzero(np.isin(labels, evaluation_classes))
    reader, training = train_reader(
        patches=patches,
        labels=labels,
        predicates=predicates,
        train_classes=train_classes,
        config=config,
        device=device,
        shuffle_labels=shuffle_labels,
    )
    train_scores = score_class_predicates(
        reader,
        patches,
        train_rows,
        predicates,
        train_classes,
        device,
        int(config["evaluation_batch_size"]),
    )
    thresholds = calibrate_support_thresholds(
        train_scores,
        labels[train_rows],
        train_classes,
        predicates,
        int(config["hard_negative_count"]),
    )
    evaluation_scores = score_class_predicates(
        reader,
        patches,
        evaluation_rows,
        predicates,
        evaluation_classes,
        device,
        int(config["evaluation_batch_size"]),
    )
    evaluation_labels = labels[evaluation_rows]
    pairwise = pairwise_hard_accuracy(
        evaluation_scores,
        evaluation_labels,
        evaluation_classes,
        predicates,
        int(config["hard_negative_count"]),
    )
    tri_evidence, states, contributions, alternatives = tristate_ledger(
        evaluation_scores,
        evaluation_classes,
        predicates,
        thresholds,
        int(config["hard_negative_count"]),
    )
    parent_predictions, _, true_local, true_in_top5 = mean8_predictions(
        features,
        evaluation_labels,
        evaluation_rows,
        role_embeddings,
        evaluation_classes,
    )
    correction = correction_metrics(
        tri_evidence,
        evaluation_labels,
        evaluation_classes,
        parent_predictions,
        true_local,
        true_in_top5,
    )
    deletion = None
    if not shuffle_labels:
        source = yaml.safe_load(Path(config["source_config"]).read_text(encoding="utf-8"))
        split = load_xlsa_split(source["res101"], source["att_splits"])
        expected = split.labels.index_select(0, split.train_indices).numpy()
        if not np.array_equal(expected, labels):
            raise ValueError("原图顺序与正式train标签不一致。")
        deletion = deletion_test(
            reader=reader,
            clip_model=clip_model,
            preprocess=preprocess,
            split=split,
            source=source,
            train_labels=labels,
            train_classes=train_classes,
            cached_patches=patches,
            predicates=predicates,
            thresholds=thresholds,
            negative_count=int(config["hard_negative_count"]),
            count=int(config["deletion_count"]),
            seed=int(config["seed"]),
            device=device,
        )
    ledger_path = output.with_suffix(".ledger.npz")
    ledger_sha = write_ledger(
        ledger_path,
        states=states,
        contributions=contributions,
        labels=evaluation_labels,
        class_ids=evaluation_classes,
        thresholds=thresholds,
        alternatives=alternatives,
    )
    checkpoint_path = output.with_suffix(".pth")
    checkpoint = {
        "model_state_dict": reader.state_dict(),
        "code_commit": expected_commit,
        "config_sha256": config_sha256,
        "asset_identity": asset_identity,
        "mode": "shuffled_predicate_control" if shuffle_labels else "real_tristate_predicates",
        "seed": int(config["seed"]),
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    result = {
        "mode": "shuffled_predicate_control" if shuffle_labels else "real_tristate_predicates",
        "code_commit": expected_commit,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "seed": int(config["seed"]),
        "asset_identity": asset_identity,
        "environment": environment_fingerprint(device),
        "training": training,
        "train_classes": train_classes.tolist(),
        "evaluation_classes": evaluation_classes.tolist(),
        "pairwise_hard_negative_accuracy": pairwise,
        "support_thresholds": thresholds.tolist(),
        "state_counts": {
            "support": int((states == 1).sum()),
            "refute": int((states == -1).sum()),
            "unobserved": int((states == 0).sum()),
        },
        "observed_fraction": float((states != 0).mean()),
        "mean8_and_evidence": correction,
        "deletion": deletion,
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "unseen_images_used": False,
        "human_annotations_used": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def _validate_result_artifacts(result: dict) -> None:
    for path_key, sha_key in (
        ("ledger_path", "ledger_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
    ):
        path = Path(result[path_key])
        if not path.is_file() or sha256_file(path) != result[sha_key]:
            raise ValueError(f"结果制品缺失或SHA错误：{path_key}")


def shuffled_failure_gates(shuffled: dict, config: dict) -> dict[str, bool]:
    return {
        "shuffled_pairwise_failed": shuffled["pairwise_hard_negative_accuracy"]
        < float(config["pairwise_accuracy_gate"]),
        "shuffled_error_correction_failed": shuffled["mean8_and_evidence"][
            "error_pair_true_preferred_fraction"
        ]
        < float(config["error_correction_gate"]),
        "shuffled_correct_damage_failed": shuffled["mean8_and_evidence"][
            "correct_sample_evidence_reversal_fraction"
        ]
        >= float(config["correct_damage_gate"]),
    }


def merge(
    config: dict,
    config_path: Path,
    config_sha256: str,
    expected_commit: str,
    real_path: Path,
    shuffled_path: Path,
    output: Path,
):
    require_clean_code_tree()
    if current_code_commit() != expected_commit or sha256_file(config_path) != config_sha256:
        raise ValueError("merge代码或config身份错误。")
    if real_path.resolve() == shuffled_path.resolve():
        raise ValueError("real与shuffled结果不得是同一文件。")
    real = json.loads(real_path.read_text(encoding="utf-8"))
    shuffled = json.loads(shuffled_path.read_text(encoding="utf-8"))
    if real.get("mode") != "real_tristate_predicates" or shuffled.get("mode") != "shuffled_predicate_control":
        raise ValueError("real/shuffled mode错误或被交换。")
    for result in (real, shuffled):
        if (
            result.get("code_commit") != expected_commit
            or result.get("config_sha256") != config_sha256
            or result.get("seed") != int(config["seed"])
            or result.get("unseen_images_used") is not False
            or result.get("human_annotations_used") is not False
        ):
            raise ValueError("待合并结果代码、配置、seed或数据边界身份不一致。")
        _validate_result_artifacts(result)
    if (
        real["train_classes"] != shuffled["train_classes"]
        or real["evaluation_classes"] != shuffled["evaluation_classes"]
        or real["asset_identity"] != shuffled["asset_identity"]
    ):
        raise ValueError("real/shuffled split或资产身份不一致。")
    if real.get("deletion") is None or shuffled.get("deletion") is not None:
        raise ValueError("删除验证必须且只能存在于real结果。")
    gates = {
        "pairwise_accuracy": real["pairwise_hard_negative_accuracy"]
        >= float(config["pairwise_accuracy_gate"]),
        "error_correction": real["mean8_and_evidence"]["error_pair_true_preferred_fraction"]
        >= float(config["error_correction_gate"]),
        "correct_damage": real["mean8_and_evidence"]["correct_sample_evidence_reversal_fraction"]
        < float(config["correct_damage_gate"]),
        "deletion": real["deletion"]["selected_drop_greater_fraction"]
        >= float(config["deletion_gate"]),
        "raw_patch_identity": real["deletion"]["raw_patch_minimum_cosine"]
        >= float(config["raw_patch_cosine_gate"]),
    }
    gates.update(shuffled_failure_gates(shuffled, config))
    result = {
        "schema_version": "gzsl-paper.tristate-predicate-diagnostic-result.v1",
        "idea_id": "IDEA-163",
        "code_commit": expected_commit,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "real": real,
        "shuffled": shuffled,
        "gates": gates,
        "decision": "minimal_falsification_pass" if all(gates.values()) else "minimal_falsification_fail",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--merge-real", type=Path)
    parser.add_argument("--merge-shuffled", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.merge_real or args.merge_shuffled:
        if not args.merge_real or not args.merge_shuffled:
            raise ValueError("合并必须同时提供real与shuffled结果。")
        merge(
            config,
            args.config,
            args.expected_config_sha,
            args.expected_commit,
            args.merge_real,
            args.merge_shuffled,
            args.output,
        )
    else:
        run(
            config,
            args.config,
            args.expected_config_sha,
            args.expected_commit,
            args.output,
            torch.device(args.device),
            args.shuffle_labels,
        )


if __name__ == "__main__":
    main()
