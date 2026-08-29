"""Run natural-prompt and class-disjoint learnable concept-readout diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.diagnose_intermediate_patch_concepts import (
    ROLE_COUNT,
    build_concepts,
    phrase_embeddings,
    roc_auc,
)


PROMPT_TEMPLATES = (
    "a photo of a bird with {phrase}",
    "a close-up photo of a bird showing {phrase}",
    "a bird whose visible features include {phrase}",
)


def metric_summary(values: np.ndarray) -> dict[str, float | int]:
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


@torch.no_grad()
def prompted_embeddings(model, phrases: list[list[str]], device: torch.device) -> torch.Tensor:
    import clip

    prompts = [
        PROMPT_TEMPLATES[template].format(phrase=phrases[class_id][role])
        for role in range(ROLE_COUNT)
        for class_id in range(200)
        for template in range(len(PROMPT_TEMPLATES))
    ]
    rows = []
    for start in range(0, len(prompts), 128):
        tokens = clip.tokenize(prompts[start : start + 128]).to(device)
        rows.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    values = torch.cat(rows).reshape(ROLE_COUNT, 200, len(PROMPT_TEMPLATES), 768)
    return F.normalize(values.mean(dim=2), dim=-1)


def setup(args, device: torch.device):
    import clip

    labels = torch.load(args.train_labels, map_location="cpu", weights_only=True).long().numpy()
    seen_classes = np.unique(labels)
    if len(labels) != 7057 or len(seen_classes) != 150:
        raise ValueError("概念探针固定CUB 7,057张formal-seen图像与150类。")
    text = json.loads(args.role_texts.read_text(encoding="utf-8"))
    model, _ = clip.load(str(args.clip_checkpoint), device=device, jit=False)
    model.eval()
    bare, phrases = phrase_embeddings(model, text["descriptions"], device)
    _, clusters, names = build_concepts(bare, phrases, set(seen_classes.tolist()))
    prompted = prompted_embeddings(model, phrases, device)
    concept_rows = []
    for role, members in clusters:
        concept_rows.append(F.normalize(prompted[role, members].mean(dim=0), dim=0))
    concepts = torch.stack(concept_rows)

    rng = np.random.default_rng(args.seed)
    class_order = rng.permutation(seen_classes)
    train_classes = set(class_order[:100].tolist())
    evaluation_classes = set(class_order[100:].tolist())
    eligible = []
    for index, (_, members) in enumerate(clusters):
        positive_train = train_classes & set(members)
        positive_evaluation = evaluation_classes & set(members)
        if len(positive_train) >= 2 and len(positive_evaluation) >= 1:
            eligible.append(index)
    if len(eligible) < 15:
        raise RuntimeError(f"可评估概念少于预注册门槛：{len(eligible)}")
    train_rows = np.flatnonzero(np.isin(labels, list(train_classes)))
    evaluation_rows = np.flatnonzero(np.isin(labels, list(evaluation_classes)))
    positive_classes = [set(clusters[index][1]) & set(seen_classes.tolist()) for index in eligible]
    targets = np.stack([np.isin(labels, list(classes)) for classes in positive_classes], axis=1).astype(np.float32)
    return {
        "labels": labels,
        "concepts": concepts[eligible],
        "clusters": [clusters[index] for index in eligible],
        "names": [names[index] for index in eligible],
        "eligible_indices": eligible,
        "train_classes": sorted(train_classes),
        "evaluation_classes": sorted(evaluation_classes),
        "train_rows": train_rows,
        "evaluation_rows": evaluation_rows,
        "targets": targets,
    }


def auc_values(targets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.asarray([roc_auc(targets[:, index], scores[:, index]) for index in range(scores.shape[1])])


@torch.no_grad()
def frozen_scores(patches: np.ndarray, rows: np.ndarray, concepts: torch.Tensor, device, batch_size):
    outputs = []
    concepts = concepts.to(device)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = torch.from_numpy(np.asarray(patches[batch_rows], dtype=np.float16).copy()).to(device)
        batch = F.normalize(batch.float(), dim=-1)
        outputs.append(torch.matmul(batch, concepts.T).amax(dim=1).cpu())
    return torch.cat(outputs).numpy()


class SharedConceptReadout(nn.Module):
    def __init__(self, rank: int = 64):
        super().__init__()
        self.visual_down = nn.Linear(768, rank, bias=False)
        self.visual_up = nn.Linear(rank, 768, bias=False)
        self.text_down = nn.Linear(768, rank, bias=False)
        self.text_up = nn.Linear(rank, 768, bias=False)
        nn.init.zeros_(self.visual_up.weight)
        nn.init.zeros_(self.text_up.weight)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, patches: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        patches = F.normalize(patches.float(), dim=-1)
        concepts = F.normalize(concepts.float(), dim=-1)
        visual = F.normalize(
            patches + self.visual_up(F.gelu(self.visual_down(patches))), dim=-1
        )
        text = F.normalize(
            concepts + self.text_up(F.gelu(self.text_down(concepts))), dim=-1
        )
        similarities = torch.matmul(visual, text.T)
        attention = torch.softmax(similarities / 0.07, dim=1)
        evidence = (attention * similarities).sum(dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * evidence + self.bias


@torch.no_grad()
def probe_scores(model, patches, rows, concepts, device, batch_size):
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = torch.from_numpy(np.asarray(patches[batch_rows], dtype=np.float16).copy()).to(device)
        outputs.append(model(batch, concepts).cpu())
    return torch.cat(outputs).numpy()


def baseline(args) -> None:
    device = torch.device(args.device)
    values = setup(args, device)
    patches = np.load(args.final_patches, mmap_mode="r")
    if patches.shape != (7057, 576, 768) or patches.dtype != np.float16:
        raise ValueError("正式576-patch资产shape/dtype错误。")
    all_rows = np.arange(len(values["labels"]))
    all_scores = frozen_scores(patches, all_rows, values["concepts"], device, args.eval_batch_size)
    evaluation_scores = all_scores[values["evaluation_rows"]]
    all_aucs = auc_values(values["targets"], all_scores)
    evaluation_aucs = auc_values(
        values["targets"][values["evaluation_rows"]], evaluation_scores
    )
    result = {
        "mode": "natural_prompt_frozen_baseline",
        "templates": list(PROMPT_TEMPLATES),
        "eligible_concept_count": len(values["eligible_indices"]),
        "eligible_indices": values["eligible_indices"],
        "train_classes": values["train_classes"],
        "evaluation_classes": values["evaluation_classes"],
        "all_formal_seen": metric_summary(all_aucs),
        "pseudo_unseen": metric_summary(evaluation_aucs),
        "per_concept": [
            {"name": name, "all_auc": float(all_aucs[index]), "pseudo_unseen_auc": float(evaluation_aucs[index])}
            for index, name in enumerate(values["names"])
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    values = setup(args, device)
    patches = np.load(args.final_patches, mmap_mode="r")
    concepts = values["concepts"].to(device)
    train_rows = values["train_rows"]
    train_targets = values["targets"][train_rows].copy()
    if args.shuffle_labels:
        rng_shuffle = np.random.default_rng(args.seed + 1000)
        train_targets = train_targets[rng_shuffle.permutation(len(train_targets))]
    positives = train_targets.sum(axis=0)
    pos_weight = torch.from_numpy((len(train_targets) - positives) / np.maximum(positives, 1.0)).to(device)
    model = SharedConceptReadout(rank=args.rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    losses = []
    model.train()
    for _ in range(args.updates):
        local = rng.integers(0, len(train_rows), size=args.batch_size)
        batch_rows = train_rows[local]
        batch = torch.from_numpy(np.asarray(patches[batch_rows], dtype=np.float16).copy()).to(device)
        target = torch.from_numpy(train_targets[local]).to(device)
        logits = model(batch, concepts)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    train_scores = probe_scores(
        model, patches, train_rows, concepts, device, args.eval_batch_size
    )
    evaluation_rows = values["evaluation_rows"]
    evaluation_scores = probe_scores(
        model, patches, evaluation_rows, concepts, device, args.eval_batch_size
    )
    true_train_targets = values["targets"][train_rows]
    train_aucs = auc_values(true_train_targets, train_scores)
    evaluation_aucs = auc_values(values["targets"][evaluation_rows], evaluation_scores)
    result = {
        "mode": "shuffled_label_control" if args.shuffle_labels else "real_label_probe",
        "eligible_concept_count": len(values["eligible_indices"]),
        "eligible_indices": values["eligible_indices"],
        "updates": args.updates,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "initial_loss_mean_20": float(np.mean(losses[:20])),
        "final_loss_mean_20": float(np.mean(losses[-20:])),
        "train": metric_summary(train_aucs),
        "pseudo_unseen": metric_summary(evaluation_aucs),
        "per_concept": [
            {"name": name, "train_auc": float(train_aucs[index]), "pseudo_unseen_auc": float(evaluation_aucs[index])}
            for index, name in enumerate(values["names"])
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def merge(args) -> None:
    baseline_result = json.loads(args.baseline.read_text(encoding="utf-8"))
    real = json.loads(args.real.read_text(encoding="utf-8"))
    shuffled = json.loads(args.shuffled.read_text(encoding="utf-8"))
    if not (
        baseline_result["eligible_indices"] == real["eligible_indices"] == shuffled["eligible_indices"]
    ):
        raise ValueError("三步诊断的概念轴不一致。")
    real_median = real["pseudo_unseen"]["median_auc"]
    baseline_median = baseline_result["pseudo_unseen"]["median_auc"]
    gain = real_median - baseline_median
    conditions = {
        "eligible_concepts_ge_15": real["eligible_concept_count"] >= 15,
        "real_pseudo_unseen_median_ge_0_65": real_median >= 0.65,
        "median_gain_ge_0_05": gain >= 0.05,
        "real_fraction_ge_0_60_at_least_0_60": real["pseudo_unseen"]["fraction_ge_0_60"] >= 0.60,
        "shuffled_median_le_0_55": shuffled["pseudo_unseen"]["median_auc"] <= 0.55,
    }
    result = {
        "schema_version": "gzsl-paper.learnable-concept-readout-diagnostic.v1",
        "baseline": baseline_result,
        "real_probe": real,
        "shuffled_control": shuffled,
        "pseudo_unseen_median_gain": gain,
        "success_conditions": conditions,
        "decision": "learnable_concept_signal_supported" if all(conditions.values()) else "learnable_concept_signal_not_supported",
        "unseen_images_used": False,
        "clip_frozen": True,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "baseline_pseudo_unseen": baseline_result["pseudo_unseen"],
        "real_pseudo_unseen": real["pseudo_unseen"],
        "shuffled_pseudo_unseen": shuffled["pseudo_unseen"],
        "gain": gain,
        "conditions": conditions,
    }, ensure_ascii=False))


def add_common(parser):
    parser.add_argument("--role-texts", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--final-patches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-batch-size", type=int, default=16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline_parser = commands.add_parser("baseline")
    add_common(baseline_parser)
    train_parser = commands.add_parser("train")
    add_common(train_parser)
    train_parser.add_argument("--updates", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--rank", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--shuffle-labels", action="store_true")
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--baseline", type=Path, required=True)
    merge_parser.add_argument("--real", type=Path, required=True)
    merge_parser.add_argument("--shuffled", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "baseline":
        baseline(args)
    elif args.command == "train":
        train(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
