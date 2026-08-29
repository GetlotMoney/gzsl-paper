"""Gate constrained evidence-graph solvers on one frozen IDEA-162 reader."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment

from tools.diagnose_intermediate_patch_concepts import roc_auc
from tools.diagnose_learnable_concept_readout import (
    SharedConceptReadout,
    auc_values,
    probe_scores,
    setup,
)
from tools.gzsl_data import load_xlsa_split, resolve_xlsa_image_path
from tools.run_contract import current_code_commit, require_clean_code_tree
from tools.runtime import sha256_file


ROLE_COUNT = 6


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "idea_id", "dataset", "role_texts", "role_texts_sha256",
        "clip_checkpoint", "clip_checkpoint_sha256", "source_config", "source_config_sha256",
        "visual_asset_manifest", "visual_asset_manifest_sha256", "train_labels",
        "train_features", "final_patches", "seed", "reader_rank", "reader_updates",
        "reader_batch_size", "reader_learning_rate", "reader_weight_decay",
        "evaluation_image_count", "deletion_count", "hard_concept_minimum",
        "duplicate_rate_gate", "accuracy_gain_gate", "net_correction_gate",
        "deletion_gate", "assignment_identity_gate", "patch_identity_gate",
        "top_r_equivalence_tolerance",
        "solver_ms_gate", "eligible_indices",
        "unseen_images_used", "human_annotations_used",
    }
    actual = set(config) if isinstance(config, dict) else set()
    if actual != required:
        raise ValueError(f"IDEA-165配置错误：缺少={sorted(required-actual)}，多出={sorted(actual-required)}")
    if (
        config["schema_version"] != "gzsl-paper.constrained-evidence-search.v1"
        or config["idea_id"] != "IDEA-165"
        or config["dataset"] != "CUB"
        or config["unseen_images_used"] is not False
        or config["human_annotations_used"] is not False
    ):
        raise ValueError("IDEA-165配置身份或数据边界错误。")
    return config


def validate_assets(config: dict) -> dict:
    manifest_path = Path(config["visual_asset_manifest"])
    if sha256_file(manifest_path) != config["visual_asset_manifest_sha256"]:
        raise ValueError("576-patch manifest SHA错误。")
    if sha256_file(Path(config["role_texts"])) != config["role_texts_sha256"]:
        raise ValueError("角色文本SHA错误。")
    if sha256_file(Path(config["clip_checkpoint"])) != config["clip_checkpoint_sha256"]:
        raise ValueError("CLIP checkpoint SHA错误。")
    if sha256_file(Path(config["source_config"])) != config["source_config_sha256"]:
        raise ValueError("source config SHA错误。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "gzsl-paper.projected-patch-assets.v1"
        or manifest.get("patch_shape") != [576, 768]
        or manifest.get("clip_checkpoint_sha256") != config["clip_checkpoint_sha256"]
    ):
        raise ValueError("576-patch资产schema错误。")
    for key, filename in (
        ("train_labels", "train_labels.pt"),
        ("train_features", "train_features.pt"),
        ("final_patches", "train_patch_features.npy"),
    ):
        path = Path(config[key])
        if path.resolve() != (manifest_path.parent / filename).resolve():
            raise ValueError(f"{key}未绑定manifest路径。")
        if sha256_file(path) != manifest["outputs_sha256"].get(filename):
            raise ValueError(f"{key} SHA错误。")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": config["visual_asset_manifest_sha256"],
        "asset_id": manifest.get("asset_id"),
        "class_order_sha256": manifest.get("class_order_sha256"),
        "role_texts_sha256": config["role_texts_sha256"],
        "clip_checkpoint_sha256": config["clip_checkpoint_sha256"],
    }


def metric_summary(values: np.ndarray) -> dict:
    return {
        "median_auc": float(np.median(values)),
        "mean_auc": float(values.mean()),
        "fraction_ge_0_60": float((values >= 0.60).mean()),
    }


def environment(device):
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
    }


def train_reader(values, patches, config, device, shuffled: bool):
    torch.manual_seed(int(config["seed"]))
    rng = np.random.default_rng(int(config["seed"]))
    concepts = values["concepts"].to(device)
    train_rows = values["train_rows"]
    targets = values["targets"][train_rows].copy()
    if shuffled:
        targets = targets[np.random.default_rng(int(config["seed"]) + 1000).permutation(len(targets))]
    positives = targets.sum(axis=0)
    pos_weight = torch.from_numpy((len(targets) - positives) / np.maximum(positives, 1.0)).to(device)
    reader = SharedConceptReadout(rank=int(config["reader_rank"])).to(device)
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=float(config["reader_learning_rate"]),
        weight_decay=float(config["reader_weight_decay"]),
    )
    losses = []
    reader.train()
    for _ in range(int(config["reader_updates"])):
        local = rng.integers(0, len(train_rows), size=int(config["reader_batch_size"]))
        rows = train_rows[local]
        batch = torch.from_numpy(np.asarray(patches[rows], dtype=np.float16).copy()).to(device)
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


def adapted_patch_scores(reader, patches: torch.Tensor, concepts: torch.Tensor):
    patches = F.normalize(patches.float(), dim=-1)
    concepts = F.normalize(concepts.float(), dim=-1)
    visual = F.normalize(
        patches + reader.visual_up(F.gelu(reader.visual_down(patches))), dim=-1
    )
    text = F.normalize(
        concepts + reader.text_up(F.gelu(reader.text_down(concepts))), dim=-1
    )
    return torch.matmul(visual, text.T).permute(0, 2, 1)


@torch.no_grad()
def image_concept_max_scores(reader, patches, rows, concepts, device, batch_size=16):
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start+batch_size]
        batch = torch.from_numpy(np.asarray(patches[batch_rows], dtype=np.float16).copy()).to(device)
        outputs.append(adapted_patch_scores(reader, batch, concepts).amax(dim=2).cpu())
    return torch.cat(outputs).numpy()


def balanced_threshold(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative))
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    order = np.argsort(values, kind="mergesort")[::-1]
    labels = labels[order]
    tpr = np.cumsum(labels) / max(len(positive), 1)
    tnr = 1.0 - np.cumsum(1 - labels) / max(len(negative), 1)
    return float(values[order[int(np.argmax(0.5 * (tpr + tnr)))]])


def calibrate_thresholds(max_scores, targets):
    return np.asarray(
        [balanced_threshold(max_scores[targets[:, k] > 0.5, k], max_scores[targets[:, k] <= 0.5, k]) for k in range(max_scores.shape[1])],
        dtype=np.float32,
    )


def concept_auc(max_scores, targets):
    return np.asarray([roc_auc(targets[:, k] > 0.5, max_scores[:, k]) for k in range(max_scores.shape[1])])


def stratified_rows(labels: np.ndarray, classes: list[int], count: int, seed: int):
    rng = np.random.default_rng(seed)
    per_class, remainder = divmod(count, len(classes))
    rows = []
    for rank, class_id in enumerate(classes):
        candidates = np.flatnonzero(labels == class_id).copy()
        rng.shuffle(candidates)
        rows.extend(candidates[: per_class + int(rank < remainder)].tolist())
    return np.asarray(sorted(rows), dtype=np.int64)


def class_concepts(clusters, class_ids):
    mapping = {}
    for class_id in class_ids:
        nodes = []
        for concept_index, (role, members) in enumerate(clusters):
            if int(class_id) in members:
                nodes.append((int(role), int(concept_index)))
        mapping[int(class_id)] = sorted(nodes)
    return mapping


def pool_regions(edges: np.ndarray):
    return edges.reshape(edges.shape[0], 12, 2, 12, 2).mean(axis=(2, 4)).reshape(edges.shape[0], 144)


def independent_assignment(role_edges: np.ndarray):
    assignment = {}
    score = 0.0
    for role in range(role_edges.shape[0]):
        node = int(np.argmax(role_edges[role]))
        value = float(role_edges[role, node])
        if value > 0:
            assignment[role] = node
            score += value
    return score, assignment


def exact_assignment(role_edges: np.ndarray, capacity: int, *, top_r: bool = True):
    role_count, node_count = role_edges.shape
    if role_count == 0:
        return 0.0, {}, 0
    if top_r:
        candidates = sorted(
            set(
                int(value)
                for role in range(role_count)
                for value in np.argpartition(role_edges[role], -min(role_count, node_count))[-min(role_count, node_count):]
            )
        )
    else:
        candidates = list(range(node_count))
    state_count = 1 << role_count
    dp = np.full(state_count, -np.inf, dtype=np.float64)
    paths = [dict() for _ in range(state_count)]
    dp[0] = 0.0
    transitions = 0
    for node in candidates:
        new_dp = dp.copy()
        new_paths = [value.copy() for value in paths]
        for mask in range(state_count):
            if not np.isfinite(dp[mask]):
                continue
            remaining = [role for role in range(role_count) if not (mask >> role) & 1]
            for size in range(1, min(capacity, len(remaining)) + 1):
                for subset in itertools.combinations(remaining, size):
                    weight = sum(max(float(role_edges[role, node]), 0.0) for role in subset)
                    if weight <= 0:
                        continue
                    new_mask = mask | sum(1 << role for role in subset)
                    transitions += 1
                    if dp[mask] + weight > new_dp[new_mask]:
                        new_dp[new_mask] = dp[mask] + weight
                        new_paths[new_mask] = paths[mask].copy()
                        for role in subset:
                            if role_edges[role, node] > 0:
                                new_paths[new_mask][role] = node
        dp, paths = new_dp, new_paths
    best = int(np.argmax(dp))
    return float(dp[best]), paths[best], transitions


def fast_assignment(role_edges: np.ndarray, capacity: int):
    """Exact optional bipartite assignment; bitmask DP remains the audit oracle."""
    role_count, node_count = role_edges.shape
    if role_count == 0:
        return 0.0, {}, 0
    top_count = min(role_count, node_count)
    candidates = sorted(
        set(
            int(value)
            for role in range(role_count)
            for value in np.argpartition(role_edges[role], -top_count)[-top_count:]
        )
    )
    duplicated = [node for node in candidates for _ in range(capacity)]
    weights = np.zeros((role_count, len(duplicated) + role_count), dtype=np.float64)
    for column, node in enumerate(duplicated):
        weights[:, column] = np.maximum(role_edges[:, node], 0.0)
    rows, columns = linear_sum_assignment(weights, maximize=True)
    assignment = {}
    score = 0.0
    for role, column in zip(rows.tolist(), columns.tolist()):
        if column < len(duplicated) and weights[role, column] > 0:
            assignment[int(role)] = int(duplicated[column])
            score += float(weights[role, column])
    return score, assignment, int(weights.size)


def score_one_class(edges, nodes, mode):
    if not nodes:
        return 0.0, {}, 0, 0.0, {}, 0.0, 0.0
    role_edges = np.stack([edges[concept] for _, concept in nodes])
    if mode == "region_capacity1":
        role_edges = pool_regions(role_edges)
        capacity = 1
    else:
        capacity = 1 if mode == "patch_capacity1" else 2
    started = time.perf_counter()
    independent_score, independent_path = independent_assignment(role_edges)
    independent_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dp_score, dp_path, transitions = fast_assignment(role_edges, capacity)
    solver_seconds = time.perf_counter() - started
    denominator = len(nodes)
    return (
        dp_score / denominator,
        dp_path,
        transitions,
        independent_score / denominator,
        independent_path,
        independent_seconds,
        solver_seconds,
    )


def duplicate_assignment(path: dict):
    values = list(path.values())
    return len(values) >= 2 and len(values) != len(set(values))


@torch.no_grad()
def compute_edges(reader, patch_array, rows, concepts, thresholds, device, batch_size=8):
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start+batch_size]
        batch = torch.from_numpy(np.asarray(patch_array[batch_rows], dtype=np.float16).copy()).to(device)
        scores = adapted_patch_scores(reader, batch, concepts).cpu().numpy()
        outputs.append(scores - thresholds[None, :, None])
    return np.concatenate(outputs)


def evaluate_solvers(edges, labels, class_ids, mapping, modes):
    results = {}
    for mode in modes:
        dp_predictions, independent_predictions = [], []
        duplicates = []
        transition_values = []
        independent_seconds = 0.0
        solver_only_seconds = 0.0
        true_paths = []
        started = time.perf_counter()
        for image_index, image_edges in enumerate(edges):
            dp_scores, independent_scores = [], []
            class_paths = {}
            independent_paths = {}
            for class_id in class_ids:
                (
                    dp_score,
                    dp_path,
                    transitions,
                    independent_score,
                    independent_path,
                    independent_time,
                    solver_time,
                ) = score_one_class(
                    image_edges, mapping[int(class_id)], mode
                )
                dp_scores.append(dp_score)
                independent_scores.append(independent_score)
                class_paths[int(class_id)] = dp_path
                independent_paths[int(class_id)] = independent_path
                transition_values.append(transitions)
                independent_seconds += independent_time
                solver_only_seconds += solver_time
            dp_predictions.append(int(class_ids[int(np.argmax(dp_scores))]))
            independent_predictions.append(int(class_ids[int(np.argmax(independent_scores))]))
            true_class = int(labels[image_index])
            duplicates.append(duplicate_assignment(independent_paths[true_class]))
            true_paths.append(class_paths[true_class])
        elapsed = time.perf_counter() - started
        dp_predictions = np.asarray(dp_predictions)
        independent_predictions = np.asarray(independent_predictions)
        independent_correct = independent_predictions == labels
        dp_correct = dp_predictions == labels
        corrected = int((~independent_correct & dp_correct).sum())
        damaged = int((independent_correct & ~dp_correct).sum())
        results[mode] = {
            "independent_accuracy": float(independent_correct.mean()),
            "dp_accuracy": float(dp_correct.mean()),
            "accuracy_gain_percentage_points": 100.0 * float(dp_correct.mean() - independent_correct.mean()),
            "corrected": corrected,
            "damaged": damaged,
            "net_correction": corrected - damaged,
            "duplicate_rate": float(np.mean(duplicates)),
            "maximum_transitions_per_class": int(max(transition_values, default=0)),
            "solver_seconds": elapsed,
            "independent_seconds": independent_seconds,
            "assignment_seconds": solver_only_seconds,
            "assignment_ms_per_class": 1000.0 * solver_only_seconds / max(len(edges) * len(class_ids), 1),
            "dp_predictions": dp_predictions,
            "independent_predictions": independent_predictions,
            "true_paths": true_paths,
        }
    return results


def top_r_equivalence(edges, labels, mapping, modes, count=32):
    maximum = 0.0
    checked = 0
    for image_index in range(min(count, len(edges))):
        nodes = mapping[int(labels[image_index])]
        if not nodes:
            continue
        role_edges = np.stack([edges[image_index, concept] for _, concept in nodes])
        for mode in modes:
            current = pool_regions(role_edges) if mode == "region_capacity1" else role_edges
            capacity = 2 if mode == "patch_capacity2" else 1
            top_score, _, _ = exact_assignment(current, capacity, top_r=True)
            full_score, _, _ = exact_assignment(current, capacity, top_r=False)
            maximum = max(maximum, abs(top_score - full_score))
            checked += 1
    return {"checked": checked, "maximum_abs": maximum}


def decision_pre_gates(result, config):
    return (
        result["duplicate_rate"] >= float(config["duplicate_rate_gate"])
        and result["accuracy_gain_percentage_points"] >= float(config["accuracy_gain_gate"])
        and result["net_correction"] >= int(config["net_correction_gate"])
        and result["corrected"] > result["damaged"]
        and result["assignment_ms_per_class"] <= float(config["solver_ms_gate"])
    )


@torch.no_grad()
def encode_final_patches(model, images: torch.Tensor):
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


def mask_assignment(image: torch.Tensor, nodes, mode: str):
    output = image.clone()
    for node in sorted(set(int(value) for value in nodes)):
        if mode == "region_capacity1":
            row, column = divmod(node, 12)
            side = 28
        else:
            row, column = divmod(node, 24)
            side = 14
        output[:, row * side : (row + 1) * side, column * side : (column + 1) * side] = 0.0
    return output


def random_nodes(count: int, mode: str, seed: int, forbidden=()):
    node_count = 144 if mode == "region_capacity1" else 576
    rng = np.random.default_rng(seed)
    available = np.asarray(sorted(set(range(node_count)) - set(map(int, forbidden))))
    return rng.choice(available, size=count, replace=False).tolist()


def true_class_score(image_edges, nodes, mode):
    return score_one_class(image_edges, nodes, mode)[0:2]


@torch.no_grad()
def deletion_test(
    *, mode, reader, concept_queries, thresholds, mapping, evaluation_rows, labels,
    cached_paths, patch_array, split, source, clip_model, preprocess, device, count, seed,
):
    path_by_row = {
        int(row): path for row, path in zip(evaluation_rows.tolist(), cached_paths)
    }
    candidates = evaluation_rows.copy()
    np.random.default_rng(seed).shuffle(candidates)
    selected_better = []
    assignment_matches = []
    parity = []
    examples = []
    for train_row in candidates:
        if len(selected_better) >= count:
            break
        class_id = int(labels[train_row])
        nodes = mapping[class_id]
        if len(nodes) < 2:
            continue
        global_index = int(split.train_indices[train_row])
        path = resolve_xlsa_image_path(
            source["raw_root"], split.image_files[global_index], source["image_path_anchors"]
        )
        with Image.open(path) as handle:
            image = preprocess(handle.convert("RGB"))
        raw = encode_final_patches(clip_model, image.unsqueeze(0).to(device))
        cached = torch.from_numpy(np.asarray(patch_array[[train_row]], dtype=np.float16).copy()).to(device)
        parity.append(float(F.cosine_similarity(raw, cached.float(), dim=-1).mean()))
        edge = adapted_patch_scores(reader, raw, concept_queries.to(device))[0].cpu().numpy()
        edge = edge - thresholds[:, None]
        original_score, raw_assignment = true_class_score(edge, nodes, mode)
        cached_assignment = path_by_row[int(train_row)]
        assigned_nodes = list(cached_assignment.values())
        if not assigned_nodes:
            continue
        assignment_match = cached_assignment == raw_assignment
        random_assignment = random_nodes(
            len(set(assigned_nodes)),
            mode,
            seed + int(train_row) * 1009,
            forbidden=assigned_nodes,
        )
        variants = torch.stack(
            (
                mask_assignment(image, assigned_nodes, mode),
                mask_assignment(image, random_assignment, mode),
            )
        ).to(device)
        masked_patches = encode_final_patches(clip_model, variants)
        masked_edges = adapted_patch_scores(reader, masked_patches, concept_queries.to(device)).cpu().numpy()
        masked_edges = masked_edges - thresholds[None, :, None]
        selected_score, _ = true_class_score(masked_edges[0], nodes, mode)
        random_score, _ = true_class_score(masked_edges[1], nodes, mode)
        selected_drop = original_score - selected_score
        random_drop = original_score - random_score
        selected_better.append(selected_drop > 0 and selected_drop > random_drop)
        assignment_matches.append(assignment_match)
        if len(examples) < 20:
            examples.append(
                {
                    "train_row": int(train_row),
                    "class_id": class_id,
                    "assigned_nodes": assigned_nodes,
                    "cached_raw_assignment_match": assignment_match,
                    "random_nodes": random_assignment,
                    "selected_drop": selected_drop,
                    "random_drop": random_drop,
                }
            )
    if len(selected_better) < count:
        raise RuntimeError(f"只有{len(selected_better)}张图形成可删除assignment，要求{count}。")
    return {
        "count": count,
        "selected_drop_greater_fraction": float(np.mean(selected_better)),
        "cached_raw_assignment_match_fraction": float(np.mean(assignment_matches)),
        "mean_patch_cosine": float(np.mean(parity)),
        "minimum_image_mean_patch_cosine": float(np.min(parity)),
        "examples": examples,
    }


def run(config, config_path, config_sha, expected_commit, output, device, shuffled):
    require_clean_code_tree()
    if current_code_commit() != expected_commit or sha256_file(config_path) != config_sha:
        raise ValueError("IDEA-165代码/config身份错误。")
    asset_identity = validate_assets(config)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args = SimpleNamespace(
        role_texts=Path(config["role_texts"]),
        clip_checkpoint=Path(config["clip_checkpoint"]),
        train_labels=Path(config["train_labels"]),
        final_patches=Path(config["final_patches"]),
        seed=int(config["seed"]),
    )
    values = setup(args, device)
    expected_indices = [int(value) for value in config["eligible_indices"]]
    if values["eligible_indices"] != expected_indices or len(values["clusters"]) != 27:
        raise ValueError("IDEA-165没有复现冻结的27概念轴。")
    for class_id in range(200):
        seen_roles = [role for role, members in values["clusters"] if class_id in members]
        if len(seen_roles) != len(set(seen_roles)):
            raise ValueError("同一类别同一role被多个概念簇重复覆盖。")
    if len(values["train_classes"]) != 100 or len(values["evaluation_classes"]) != 50:
        raise ValueError("IDEA-165没有复现100/50类别轴。")
    patches = np.load(config["final_patches"], mmap_mode="r")
    reader, training = train_reader(values, patches, config, device, shuffled)
    concepts = values["concepts"].to(device)
    train_max = image_concept_max_scores(reader, patches, values["train_rows"], concepts, device)
    evaluation_max = image_concept_max_scores(reader, patches, values["evaluation_rows"], concepts, device)
    train_targets = values["targets"][values["train_rows"]]
    evaluation_targets = values["targets"][values["evaluation_rows"]]
    thresholds = calibrate_thresholds(train_max, train_targets)
    reader_logits = probe_scores(
        reader,
        patches,
        values["evaluation_rows"],
        concepts,
        device,
        16,
    )
    reader_auc = auc_values(evaluation_targets, reader_logits)
    edge_auc = concept_auc(evaluation_max, evaluation_targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output.with_suffix(".pth")
    concept_axis = [
        {"role": int(role), "members": list(map(int, members))}
        for role, members in values["clusters"]
    ]
    torch.save(
        {
            "model_state_dict": reader.state_dict(),
            "mode": "shuffled_reader" if shuffled else "real_reader",
            "code_commit": expected_commit,
            "config_sha256": config_sha,
            "asset_identity": asset_identity,
            "eligible_indices": expected_indices,
            "concept_axis": concept_axis,
            "thresholds": thresholds.tolist(),
        },
        checkpoint_path,
    )
    result = {
        "mode": "shuffled_reader" if shuffled else "real_reader",
        "code_commit": expected_commit,
        "config_sha256": config_sha,
        "asset_identity": asset_identity,
        "environment": environment(device),
        "training": training,
        "train_classes": values["train_classes"],
        "evaluation_classes": values["evaluation_classes"],
        "eligible_concept_count": len(values["eligible_indices"]),
        "eligible_indices": expected_indices,
        "concept_axis": concept_axis,
        "thresholds": thresholds.tolist(),
        "reader_concept_auc": metric_summary(reader_auc),
        "edge_max_concept_auc": metric_summary(edge_auc),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "unseen_images_used": False,
        "human_annotations_used": False,
    }
    if not shuffled:
        labels = values["labels"]
        evaluation_classes = values["evaluation_classes"]
        mapping = class_concepts(values["clusters"], evaluation_classes)
        eligible_classes = [class_id for class_id in evaluation_classes if len(mapping[int(class_id)]) >= int(config["hard_concept_minimum"])]
        evaluation_rows = stratified_rows(
            labels,
            eligible_classes,
            int(config["evaluation_image_count"]),
            int(config["seed"]) + 500,
        )
        evaluation_rows = evaluation_rows[np.isin(labels[evaluation_rows], eligible_classes)]
        if (
            len(evaluation_rows) != int(config["evaluation_image_count"])
            or len(np.unique(evaluation_rows)) != len(evaluation_rows)
            or any(len(mapping[int(labels[row])]) < int(config["hard_concept_minimum"]) for row in evaluation_rows)
        ):
            raise ValueError("IDEA-165固定500张唯一、真类至少2概念的评估合同失败。")
        edge_values = compute_edges(reader, patches, evaluation_rows, concepts, thresholds, device)
        modes = ("patch_capacity1", "patch_capacity2", "region_capacity1")
        solver_results = evaluate_solvers(
            edge_values,
            labels[evaluation_rows],
            np.asarray(evaluation_classes),
            mapping,
            modes,
        )
        equivalence = top_r_equivalence(
            edge_values,
            labels[evaluation_rows],
            mapping,
            modes,
        )
        deletions = {}
        prepass_modes = [
            mode for mode in modes
            if decision_pre_gates(solver_results[mode], config)
            and equivalence["maximum_abs"] <= float(config["top_r_equivalence_tolerance"])
        ]
        if prepass_modes:
            import clip

            clip_model, preprocess = clip.load(
                config["clip_checkpoint"], device=device, jit=False
            )
            clip_model.eval()
            source = yaml.safe_load(Path(config["source_config"]).read_text(encoding="utf-8"))
            split = load_xlsa_split(source["res101"], source["att_splits"])
            expected_labels = split.labels.index_select(0, split.train_indices).numpy()
            if not np.array_equal(expected_labels, labels):
                raise ValueError("删除验证原图与patch行序错误。")
            for mode in prepass_modes:
                deletions[mode] = deletion_test(
                    mode=mode,
                    reader=reader,
                    concept_queries=concepts,
                    thresholds=thresholds,
                    mapping=mapping,
                    evaluation_rows=evaluation_rows,
                    cached_paths=solver_results[mode]["true_paths"],
                    labels=labels,
                    patch_array=patches,
                    split=split,
                    source=source,
                    clip_model=clip_model,
                    preprocess=preprocess,
                    device=device,
                    count=int(config["deletion_count"]),
                    seed=int(config["seed"]) + 700,
                )
                row = deletions[mode]
                if (
                    row["selected_drop_greater_fraction"] >= float(config["deletion_gate"])
                    and row["cached_raw_assignment_match_fraction"] >= float(config["assignment_identity_gate"])
                    and min(row["mean_patch_cosine"], row["minimum_image_mean_patch_cosine"])
                    >= float(config["patch_identity_gate"])
                ):
                    break
        serialized = {}
        for mode, values_mode in solver_results.items():
            serialized[mode] = {
                key: value
                for key, value in values_mode.items()
                if key not in {"dp_predictions", "independent_predictions", "true_paths"}
            }
            serialized[mode]["pre_gates_pass"] = decision_pre_gates(values_mode, config)
            serialized[mode]["deletion"] = deletions.get(mode)
        result.update(
            {
                "evaluation_classes": list(map(int, evaluation_classes)),
                "eligible_classes": list(map(int, eligible_classes)),
                "evaluation_rows": evaluation_rows.tolist(),
                "solver_results": serialized,
                "top_r_equivalence": equivalence,
            }
        )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def merge(config, config_path, config_sha, expected_commit, real_path, shuffled_path, output):
    require_clean_code_tree()
    if current_code_commit() != expected_commit or sha256_file(config_path) != config_sha:
        raise ValueError("IDEA-165 merge身份错误。")
    real = json.loads(real_path.read_text(encoding="utf-8"))
    shuffled = json.loads(shuffled_path.read_text(encoding="utf-8"))
    if real.get("mode") != "real_reader" or shuffled.get("mode") != "shuffled_reader":
        raise ValueError("IDEA-165 real/shuffled mode错误。")
    for result in (real, shuffled):
        if result.get("code_commit") != expected_commit or result.get("config_sha256") != config_sha:
            raise ValueError("IDEA-165结果身份错误。")
        path = Path(result["checkpoint_path"])
        if not path.is_file() or sha256_file(path) != result["checkpoint_sha256"]:
            raise ValueError("IDEA-165 checkpoint SHA错误。")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("mode") != result["mode"]
            or payload.get("code_commit") != expected_commit
            or payload.get("config_sha256") != config_sha
            or payload.get("asset_identity") != result["asset_identity"]
            or payload.get("eligible_indices") != result["eligible_indices"]
            or payload.get("concept_axis") != result["concept_axis"]
            or payload.get("thresholds") != result["thresholds"]
        ):
            raise ValueError("IDEA-165 checkpoint内部身份错误。")
        probe = SharedConceptReadout(rank=int(config["reader_rank"]))
        probe.load_state_dict(payload["model_state_dict"], strict=True)
    if Path(real["checkpoint_path"]).resolve() == Path(shuffled["checkpoint_path"]).resolve():
        raise ValueError("real/shuffled不得共用checkpoint。")
    if (
        real["asset_identity"] != shuffled["asset_identity"]
        or real["train_classes"] != shuffled["train_classes"]
        or real["evaluation_classes"] != shuffled["evaluation_classes"]
        or real["eligible_indices"] != shuffled["eligible_indices"]
        or real["concept_axis"] != shuffled["concept_axis"]
    ):
        raise ValueError("real/shuffled资产或100/50 split不一致。")
    comparable_environment = set(real["environment"]) - {"gpu_uuid"}
    if comparable_environment != set(shuffled["environment"]) - {"gpu_uuid"} or any(
        real["environment"][key] != shuffled["environment"][key]
        for key in comparable_environment
    ):
        raise ValueError("real/shuffled环境或GPU型号不一致。")
    reader_gate = (
        real["reader_concept_auc"]["median_auc"] >= 0.65
        and shuffled["reader_concept_auc"]["median_auc"] <= 0.55
    )
    solver_decisions = {}
    ordered = ("patch_capacity1", "patch_capacity2", "region_capacity1")
    winner = None
    for attempt, mode in enumerate(ordered):
        row = real["solver_results"][mode]
        passed = (
            reader_gate
            and row["pre_gates_pass"]
            and real["top_r_equivalence"]["maximum_abs"]
            <= float(config["top_r_equivalence_tolerance"])
            and row["deletion"] is not None
            and row["deletion"]["selected_drop_greater_fraction"]
            >= float(config["deletion_gate"])
            and row["deletion"]["cached_raw_assignment_match_fraction"]
            >= float(config["assignment_identity_gate"])
            and min(
                row["deletion"]["mean_patch_cosine"],
                row["deletion"]["minimum_image_mean_patch_cosine"],
            )
            >= float(config["patch_identity_gate"])
        )
        solver_decisions[mode] = "pass" if passed else "fail"
        if passed and winner is None:
            winner = mode
            break
    result = {
        "schema_version": "gzsl-paper.constrained-evidence-search-result.v1",
        "idea_id": "IDEA-165",
        "code_commit": expected_commit,
        "config_sha256": config_sha,
        "reader_gate": reader_gate,
        "solver_decisions": solver_decisions,
        "winner": winner,
        "decision": "gate_pass" if winner else "gate_fail",
        "real": real,
        "shuffled": shuffled,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "reader_gate": reader_gate, "solver_decisions": solver_decisions}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shuffled-reader", action="store_true")
    parser.add_argument("--merge-real", type=Path)
    parser.add_argument("--merge-shuffled", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.merge_real or args.merge_shuffled:
        if not args.merge_real or not args.merge_shuffled:
            raise ValueError("merge必须同时提供real/shuffled。")
        merge(config, args.config, args.expected_config_sha, args.expected_commit, args.merge_real, args.merge_shuffled, args.output)
    else:
        run(config, args.config, args.expected_config_sha, args.expected_commit, args.output, torch.device(args.device), args.shuffled_reader)


if __name__ == "__main__":
    main()
