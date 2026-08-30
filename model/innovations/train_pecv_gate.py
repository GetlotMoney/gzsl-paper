"""Train and evaluate the class-disjoint PECV proof gate."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.innovations.pecv import (
    PairwiseErrorCorrectingVerifier,
    corrected_topk_scores,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.pecv-gate.v1"
PATH_KEYS = (
    "train_features",
    "train_labels",
    "eval_features",
    "eval_labels",
    "role_text",
    "parent_prototypes",
    "hard_negative_receipt",
    "candidate_receipt",
)


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _load_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "idea_id",
        "base_commit",
        "device",
        "seed",
        "output_dir",
        "updates",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "hidden_dim",
        "max_correction",
        "top_k",
        "macro_gain_gate_pp",
        "semantic_control_drop_pp",
    }
    required.update(PATH_KEYS)
    required.update(f"{key}_sha256" for key in PATH_KEYS)
    if not isinstance(config, dict) or set(config) != required:
        actual = set(config) if isinstance(config, dict) else set()
        raise ValueError(
            f"PECV config fields mismatch; missing={sorted(required-actual)}, "
            f"extra={sorted(actual-required)}"
        )
    if (
        config["schema_version"] != SCHEMA
        or config["experiment_id"] != "V4-TRY-013"
        or config["idea_id"] != "IDEA-182"
        or config["base_commit"]
        != "52088f69d7ac4e574e7b63c28b21ac0da7789933"
        or int(config["seed"]) != 7
        or int(config["updates"]) != 1000
        or int(config["batch_size"]) != 64
        or float(config["learning_rate"]) != 1e-3
        or float(config["weight_decay"]) != 1e-4
        or int(config["hidden_dim"]) != 32
        or float(config["max_correction"]) != 4.0
        or int(config["top_k"]) != 5
        or float(config["macro_gain_gate_pp"]) != 1.0
        or float(config["semantic_control_drop_pp"]) != 0.5
    ):
        raise ValueError("PECV preregistered values changed.")
    for key in PATH_KEYS:
        file_path = Path(config[key])
        if not file_path.is_file() or sha256_file(file_path) != config[f"{key}_sha256"]:
            raise ValueError(f"PECV asset identity mismatch: {key}")
    return config, sha256_file(path)


def _load_assets(config: dict) -> dict[str, torch.Tensor]:
    arrays = {
        "train_features": torch.from_numpy(np.load(config["train_features"]).copy()).float(),
        "train_labels": torch.from_numpy(np.load(config["train_labels"]).copy()).long(),
        "eval_features": torch.from_numpy(np.load(config["eval_features"]).copy()).float(),
        "eval_labels": torch.from_numpy(np.load(config["eval_labels"]).copy()).long(),
        "role_text": torch.from_numpy(np.load(config["role_text"]).copy()).float(),
    }
    parent = torch.load(config["parent_prototypes"], map_location="cpu", weights_only=True)
    hard = torch.load(config["hard_negative_receipt"], map_location="cpu", weights_only=True)
    candidate = torch.load(config["candidate_receipt"], map_location="cpu", weights_only=True)
    arrays.update(
        prototypes=parent["prototypes"].float(),
        scale=parent["scale"].float(),
        seen_local=parent["seen_local"].long(),
        unseen_local=parent["unseen_local"].long(),
        local_to_global=parent["local_to_global"].long(),
        hard_true=hard["true_local"].long(),
        hard_negative=hard["negative_local"].long(),
        eval_top5=candidate["top5_local"].long(),
        candidate_true=candidate["true_local"].long(),
    )
    if arrays["train_features"].shape != (4702, 768):
        raise ValueError("PECV train feature shape changed.")
    if arrays["eval_features"].shape != (2355, 768):
        raise ValueError("PECV eval feature shape changed.")
    if arrays["role_text"].shape != (150, 8, 768):
        raise ValueError("PECV role text shape changed.")
    if arrays["prototypes"].shape != (150, 768) or arrays["eval_top5"].shape != (2355, 5):
        raise ValueError("PECV parent prototype or Top-5 shape changed.")
    if not torch.equal(arrays["train_labels"], arrays["hard_true"]):
        raise ValueError("PECV train rows do not match frozen hard-negative receipt.")
    if not torch.equal(arrays["eval_labels"], arrays["candidate_true"]):
        raise ValueError("PECV eval rows do not match frozen candidate receipt.")
    train_classes = torch.unique(arrays["train_labels"], sorted=True)
    eval_classes = torch.unique(arrays["eval_labels"], sorted=True)
    if not torch.equal(train_classes, arrays["seen_local"]):
        raise ValueError("PECV train class axis is not the frozen 100-class split.")
    if not torch.equal(eval_classes, arrays["unseen_local"]):
        raise ValueError("PECV eval class axis is not the frozen 50-class split.")
    if torch.isin(train_classes, eval_classes).any():
        raise ValueError("PECV train/eval classes overlap.")
    return arrays


def _macro_accuracy(labels: torch.Tensor, prediction: torch.Tensor, classes: torch.Tensor) -> float:
    return float(
        torch.stack(
            [prediction[labels.eq(class_id)].eq(class_id).double().mean() for class_id in classes]
        ).mean()
    )


@torch.no_grad()
def _evaluate(
    model: PairwiseErrorCorrectingVerifier | None,
    assets: dict[str, torch.Tensor],
    device: torch.device,
    *,
    shuffled_semantics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = assets["eval_features"].to(device)
    prototypes = assets["prototypes"].to(device)
    role_text = assets["role_text"].to(device)
    candidates = assets["eval_top5"].to(device)
    if shuffled_semantics:
        permutation = torch.roll(torch.arange(150, device=device), shifts=1)
        verifier_prototypes = prototypes.index_select(0, permutation)
        verifier_roles = role_text.index_select(0, permutation)
    else:
        verifier_prototypes = prototypes
        verifier_roles = role_text
    normalized = F.normalize(features, dim=-1)
    all_parent = normalized @ F.normalize(prototypes, dim=-1).T * assets["scale"].to(device)
    parent_top5 = all_parent.gather(1, candidates)
    corrected = corrected_topk_scores(
        parent_top5,
        features,
        candidates,
        verifier_prototypes,
        verifier_roles,
        model,
    )
    prediction = candidates.gather(1, corrected.argmax(dim=1, keepdim=True)).squeeze(1)
    return prediction.cpu(), corrected.cpu()


def run(config_path: Path, expected_commit: str) -> dict:
    config, config_sha = _load_config(config_path)
    commit = _current_commit()
    if commit != expected_commit:
        raise ValueError(f"PECV expected commit {expected_commit}, got {commit}.")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("PECV must run from a clean worktree.")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])
    assets = _load_assets(config)
    model = PairwiseErrorCorrectingVerifier(
        role_count=8,
        hidden_dim=int(config["hidden_dim"]),
        max_correction=float(config["max_correction"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    losses = []
    features = assets["train_features"].to(device)
    prototypes = assets["prototypes"].to(device)
    roles = assets["role_text"].to(device)
    scale = assets["scale"].to(device)
    true_local = assets["hard_true"].to(device)
    negative_local = assets["hard_negative"].to(device)
    for _ in range(int(config["updates"])):
        rows = torch.randperm(features.size(0), generator=generator)[: int(config["batch_size"])]
        rows = rows.to(device)
        image = features.index_select(0, rows)
        truth = true_local.index_select(0, rows)
        negative = negative_local.index_select(0, rows)
        parent_margin = (
            F.normalize(image, dim=-1)
            * (
                F.normalize(prototypes.index_select(0, truth), dim=-1)
                - F.normalize(prototypes.index_select(0, negative), dim=-1)
            )
        ).sum(dim=-1) * scale
        correction = model.correction(
            image,
            prototypes.index_select(0, truth),
            prototypes.index_select(0, negative),
            roles.index_select(0, truth),
            roles.index_select(0, negative),
        )
        loss = F.softplus(-(parent_margin + correction)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("PECV produced non-finite gradients.")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    parent_prediction, parent_scores = _evaluate(None, assets, device)
    full_prediction, full_scores = _evaluate(model, assets, device)
    shuffled_prediction, shuffled_scores = _evaluate(
        model, assets, device, shuffled_semantics=True
    )
    labels = assets["eval_labels"]
    classes = assets["unseen_local"]
    parent_macro = _macro_accuracy(labels, parent_prediction, classes)
    full_macro = _macro_accuracy(labels, full_prediction, classes)
    shuffled_macro = _macro_accuracy(labels, shuffled_prediction, classes)
    gain_pp = 100.0 * (full_macro - parent_macro)
    semantic_drop_pp = 100.0 * (full_macro - shuffled_macro)
    parent_correct = parent_prediction.eq(labels)
    full_correct = full_prediction.eq(labels)
    correction_count = int((~parent_correct & full_correct).sum())
    damage_count = int((parent_correct & ~full_correct).sum())
    coverage = assets["eval_top5"].eq(labels[:, None]).any(dim=1)
    off_exact = torch.equal(parent_scores, _evaluate(None, assets, device)[1])
    gate_passed = (
        gain_pp >= float(config["macro_gain_gate_pp"])
        and semantic_drop_pp >= float(config["semantic_control_drop_pp"])
        and off_exact
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "pecv_gate.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "code_commit": commit,
            "config_sha256": config_sha,
            "updates": int(config["updates"]),
        },
        checkpoint_path,
    )
    torch.save(
        {
            "labels_local": labels,
            "parent_prediction_local": parent_prediction,
            "full_prediction_local": full_prediction,
            "shuffled_prediction_local": shuffled_prediction,
            "parent_top5_scores": parent_scores,
            "full_top5_scores": full_scores,
            "shuffled_top5_scores": shuffled_scores,
        },
        output_dir / "evaluation_evidence.pth",
    )
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "idea_id": config["idea_id"],
        "code_commit": commit,
        "config_sha256": config_sha,
        "train_classes": 100,
        "eval_classes": 50,
        "eval_images_used_for_gradient": False,
        "parent_macro_top1": parent_macro,
        "full_macro_top1": full_macro,
        "shuffled_semantics_macro_top1": shuffled_macro,
        "macro_gain_pp": gain_pp,
        "semantic_control_drop_pp": semantic_drop_pp,
        "top5_micro_coverage": float(coverage.double().mean()),
        "correction_count": correction_count,
        "damage_count": damage_count,
        "net_correction": correction_count - damage_count,
        "module_off_exact_parent": off_exact,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "gate_passed": gate_passed,
        "decision": "keep_for_formal_gzsl" if gate_passed else "drop",
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.expected_commit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
