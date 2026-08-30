"""Train and evaluate the class-disjoint PECV proof gate."""

from __future__ import annotations

import argparse
import json
import platform
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
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    prepare_output_dir,
    require_finite_gradients,
    require_finite_model,
    require_finite_tensor_tree,
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
        "parent_run_commit",
        "parent_config_sha256",
        "parent_asset_implementation_commit",
        "parent_active_global_sha256",
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
        or config["parent_run_commit"]
        != "a4160e4410e39d2926ef3dac4a4120ea0ea60a06"
        or config["parent_config_sha256"]
        != "f99a982da2393cdfc0d123d126d274605f63ebe08b6a4564fb8f0b542a9ba1b5"
        or config["parent_asset_implementation_commit"]
        != "a4160e4410e39d2926ef3dac4a4120ea0ea60a06"
        or config["parent_active_global_sha256"]
        != "aaee779ba7fbb0908ec1839c990e4523defe5832fa5f4b3e840d4557f8c99f42"
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
        candidate_top1=candidate["top1_local"].long(),
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
    if (
        parent.get("code_commit") != config["parent_run_commit"]
        or parent.get("config_sha256") != config["parent_config_sha256"]
        or parent.get("stop_reason") != "completed_fixed_50"
        or parent.get("update") != 4700
        or parent.get("upstream_identity", {}).get("asset_implementation_commit")
        != config["parent_asset_implementation_commit"]
        or parent.get("upstream_identity", {}).get("active_global_sha256")
        != config["parent_active_global_sha256"]
    ):
        raise ValueError("PECV Parent upstream identity changed.")
    train_classes = torch.unique(arrays["train_labels"], sorted=True)
    eval_classes = torch.unique(arrays["eval_labels"], sorted=True)
    if not torch.equal(train_classes, arrays["seen_local"]):
        raise ValueError("PECV train class axis is not the frozen 100-class split.")
    if not torch.equal(eval_classes, arrays["unseen_local"]):
        raise ValueError("PECV eval class axis is not the frozen 50-class split.")
    if torch.isin(train_classes, eval_classes).any():
        raise ValueError("PECV train/eval classes overlap.")
    if (
        not torch.equal(torch.cat((train_classes, eval_classes)).sort().values, torch.arange(150))
        or arrays["local_to_global"].unique().numel() != 150
        or arrays["local_to_global"].numel() != 150
    ):
        raise ValueError("PECV local/global class axis is not a 150-class bijection.")
    if (
        arrays["hard_negative"].shape != arrays["train_labels"].shape
        or arrays["hard_negative"].lt(0).any()
        or arrays["hard_negative"].ge(150).any()
        or arrays["hard_negative"].eq(arrays["hard_true"]).any()
        or (~torch.isin(arrays["hard_negative"], train_classes)).any()
    ):
        raise ValueError("PECV hard-negative receipt has an invalid range or class.")
    if (
        arrays["eval_top5"].lt(0).any()
        or arrays["eval_top5"].ge(150).any()
        or torch.sort(arrays["eval_top5"], dim=1).values.diff(dim=1).eq(0).any()
        or not torch.equal(arrays["candidate_top1"], arrays["eval_top5"][:, 0])
    ):
        raise ValueError("PECV candidate receipt has invalid or duplicate Top-5 rows.")
    require_finite_tensor_tree(
        {
            "train_features": arrays["train_features"],
            "eval_features": arrays["eval_features"],
            "role_text": arrays["role_text"],
            "prototypes": arrays["prototypes"],
            "scale": arrays["scale"],
        },
        "pecv_assets",
    )
    if arrays["scale"].numel() != 1 or float(arrays["scale"]) <= 0:
        raise ValueError("PECV Parent scale must be a positive scalar.")
    return arrays


def _macro_accuracy(labels: torch.Tensor, prediction: torch.Tensor, classes: torch.Tensor) -> float:
    return float(
        torch.stack(
            [prediction[labels.eq(class_id)].eq(class_id).double().mean() for class_id in classes]
        ).mean()
    )


def _stable_topk_local(
    logits: torch.Tensor,
    eligible_local: torch.Tensor,
    local_to_global: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Rank by descending logit, breaking exact ties by ascending global id."""
    global_order = torch.argsort(
        local_to_global.index_select(0, eligible_local), stable=True
    )
    ordered_local = eligible_local.index_select(0, global_order)
    ordered_logits = logits.index_select(1, global_order)
    ranks = torch.argsort(ordered_logits, dim=1, descending=True, stable=True)
    return ordered_local.index_select(0, ranks[:, :top_k].reshape(-1)).reshape(-1, top_k)


def _truth_injected_train_candidates(
    all_parent_logits: torch.Tensor,
    truth: torch.Tensor,
    seen_local: torch.Tensor,
    local_to_global: torch.Tensor,
) -> torch.Tensor:
    """Return truth plus the Parent's four strongest wrong seen candidates."""
    seen_logits = all_parent_logits.index_select(1, seen_local)
    truth_positions = torch.searchsorted(seen_local, truth)
    if not torch.equal(seen_local.index_select(0, truth_positions), truth):
        raise ValueError("PECV truth label is absent from the seen candidate axis.")
    masked = seen_logits.clone()
    masked[torch.arange(masked.size(0), device=masked.device), truth_positions] = -torch.inf
    wrong = _stable_topk_local(masked, seen_local, local_to_global, 4)
    return torch.cat((truth[:, None], wrong), dim=1)


@torch.no_grad()
def _evaluate(
    model: PairwiseErrorCorrectingVerifier | None,
    assets: dict[str, torch.Tensor],
    device: torch.device,
    *,
    shuffled_semantics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    recomputed_top5 = _stable_topk_local(
        all_parent,
        torch.arange(150, device=device),
        assets["local_to_global"].to(device),
        5,
    )
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
    return prediction.cpu(), corrected.cpu(), recomputed_top5.cpu()


def run(config_path: Path, expected_commit: str, expected_config_sha: str) -> dict:
    config, config_sha = _load_config(config_path)
    if config_sha != expected_config_sha:
        raise ValueError(
            f"PECV expected config SHA {expected_config_sha}, got {config_sha}."
        )
    commit = _current_commit()
    if commit != expected_commit:
        raise ValueError(f"PECV expected commit {expected_commit}, got {commit}.")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("PECV must run from a clean worktree.")
    seed = int(config["seed"])
    reproducibility = configure_reproducibility(
        seed, strict_determinism=True, deterministic_warn_only=False
    )
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
    normalized_features = F.normalize(features, dim=-1)
    normalized_prototypes = F.normalize(prototypes, dim=-1)
    all_train_parent = normalized_features @ normalized_prototypes.T * scale
    train_candidates = _truth_injected_train_candidates(
        all_train_parent,
        true_local,
        assets["seen_local"].to(device),
        assets["local_to_global"].to(device),
    )
    if not torch.equal(train_candidates[:, 1], negative_local):
        raise ValueError("PECV recomputed strongest wrong class differs from frozen receipt.")
    for _ in range(int(config["updates"])):
        rows = torch.randperm(features.size(0), generator=generator)[: int(config["batch_size"])]
        rows = rows.to(device)
        image = features.index_select(0, rows)
        candidates = train_candidates.index_select(0, rows)
        parent_scores = all_train_parent.index_select(0, rows).gather(1, candidates)
        corrected_scores = corrected_topk_scores(
            parent_scores,
            image,
            candidates,
            prototypes,
            roles,
            model,
        )
        # Truth is explicitly at candidate position zero; all ten pair
        # interactions use the same forward as deployment.
        loss = F.cross_entropy(
            corrected_scores,
            torch.zeros(corrected_scores.size(0), dtype=torch.long, device=device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        require_finite_gradients(model)
        optimizer.step()
        require_finite_model(model)
        losses.append(float(loss.detach().cpu()))

    model.eval()
    parent_prediction, parent_scores, recomputed_top5 = _evaluate(None, assets, device)
    full_prediction, full_scores, _ = _evaluate(model, assets, device)
    shuffled_prediction, shuffled_scores, _ = _evaluate(
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
    net_correction = correction_count - damage_count
    coverage = assets["eval_top5"].eq(labels[:, None]).any(dim=1)
    macro_coverage = float(
        torch.stack(
            [coverage[labels.eq(class_id)].double().mean() for class_id in classes]
        ).mean()
    )
    off_exact = torch.equal(parent_prediction, assets["candidate_top1"]) and torch.equal(
        recomputed_top5, assets["eval_top5"]
    )
    gate_passed = (
        gain_pp >= float(config["macro_gain_gate_pp"])
        and semantic_drop_pp >= float(config["semantic_control_drop_pp"])
        and off_exact
        and net_correction > 0
    )
    output_dir = prepare_output_dir(config["output_dir"])
    checkpoint_path = output_dir / "pecv_gate.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "batch_generator_state": generator.get_state(),
        "code_commit": commit,
        "config_sha256": config_sha,
        "updates": int(config["updates"]),
        "losses": torch.tensor(losses, dtype=torch.float64),
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = PairwiseErrorCorrectingVerifier(
        role_count=8,
        hidden_dim=int(config["hidden_dim"]),
        max_correction=float(config["max_correction"]),
    ).to(device)
    restored.load_state_dict(loaded["model_state_dict"], strict=True)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    require_finite_model(restored)
    restored.eval()
    restored_prediction, restored_scores, _ = _evaluate(restored, assets, device)
    checkpoint_roundtrip = torch.equal(restored_prediction, full_prediction) and torch.equal(
        restored_scores, full_scores
    )
    if not checkpoint_roundtrip:
        raise ValueError("PECV checkpoint strict roundtrip changed evaluation output.")
    evidence_path = output_dir / "evaluation_evidence.pth"
    atomic_torch_save(
        evidence_path,
        {
            "labels_local": labels,
            "parent_prediction_local": parent_prediction,
            "full_prediction_local": full_prediction,
            "shuffled_prediction_local": shuffled_prediction,
            "parent_top5_scores": parent_scores,
            "full_top5_scores": full_scores,
            "shuffled_top5_scores": shuffled_scores,
            "recomputed_parent_top5_local": recomputed_top5,
        },
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
        "top5_macro_coverage": macro_coverage,
        "correction_count": correction_count,
        "damage_count": damage_count,
        "net_correction": net_correction,
        "module_off_exact_parent": off_exact,
        "checkpoint_roundtrip_verified": checkpoint_roundtrip,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "gate_passed": gate_passed,
        "decision": "keep_for_formal_gzsl" if gate_passed else "drop",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_evidence_sha256": sha256_file(evidence_path),
        "parent_identity": {
            "run_commit": config["parent_run_commit"],
            "config_sha256": config["parent_config_sha256"],
            "asset_implementation_commit": config["parent_asset_implementation_commit"],
            "active_global_sha256": config["parent_active_global_sha256"],
        },
        "reproducibility": reproducibility,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "tie_break": "descending score, then frozen Parent Top-5 order (itself tied by ascending global id)",
    }
    atomic_write_json(output_dir / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, args.expected_commit, args.expected_config_sha),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
