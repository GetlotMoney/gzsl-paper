"""Evaluate the fixed R2 checkpoint with the owner-selected PCLR inference tune."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml

from model.innovations.train_gtd_tst import build_model, load_assets, load_config
from tools.gzsl_data import per_class_accuracy
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v4-pclr-inference-tune.v1"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "dataset",
    "source_config",
    "source_config_sha256",
    "source_code_commit",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "source_metrics",
    "source_metrics_sha256",
    "source_module_off_history",
    "source_module_off_history_sha256",
    "parent_evaluation_history_sha256",
    "relation_asset_manifest_sha256",
    "candidate_top_k",
    "ridge_lambda",
    "potential_cap",
    "inference_relation_temperature",
    "correction_scale",
    "seen_logit_gamma",
    "required_h",
    "required_delta_h",
    "max_us_gap",
    "device",
    "test_used_for_selection",
    "test_used_for_hyperparameter_selection",
    "nested_official_test_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "human_annotations_used",
}


def load_inference_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    invalid = (
        not isinstance(config, dict)
        or actual != CONFIG_KEYS
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != "V4-TRY-023-R3"
        or config.get("dataset") != "CUB"
        or config.get("source_config_sha256")
        != "0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49"
        or config.get("source_code_commit")
        != "b0a756dd624e883eb50d19a2455ba06bdc73f118"
        or config.get("source_checkpoint_sha256")
        != "16b5071f21a3217e58a72315029c28b8cfd97b68f812641bd0145d3f5e0702ab"
        or config.get("source_metrics_sha256")
        != "3d64bd36e48304b025044b109c579001279400ccec075fc1246496c4f28e8578"
        or config.get("source_module_off_history_sha256")
        != "d5d7049e42209b9fcc9e73e8df0208d8df960132ff8c2993da916bef64074a05"
        or config.get("parent_evaluation_history_sha256")
        != "10591bb35a51949a1989ae3a918b50bca37c1f465a52c6bb5df5552c1b0a4779"
        or config.get("relation_asset_manifest_sha256")
        != "0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4"
        or int(config.get("candidate_top_k", -1)) != 17
        or float(config.get("ridge_lambda", -1)) != 0.3
        or float(config.get("potential_cap", -1)) != 0.5
        or float(config.get("inference_relation_temperature", -1)) != 0.2
        or float(config.get("correction_scale", -1)) != 6.95
        or float(config.get("seen_logit_gamma", -1)) != 0.575
        or float(config.get("required_h", -1)) != 80.070015
        or float(config.get("required_delta_h", -1)) != 1.0
        or float(config.get("max_us_gap", -1)) != 8.0
        or config.get("device") != "cuda:0"
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not True
        or config.get("nested_official_test_selection") is not True
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
    )
    if invalid:
        raise ValueError("PCLR R3 inference config identity or disclosure changed.")
    return config, sha256_file(path)


def _class_ids(model, logits: torch.Tensor, class_ids: torch.Tensor | None):
    if class_ids is None:
        return logits
    ids = model._validated_class_ids(class_ids, logits.device)
    return logits.index_select(1, ids)


@torch.no_grad()
def tuned_inference_logits(
    model,
    image_features: torch.Tensor,
    *,
    candidate_top_k: int,
    ridge_lambda: float,
    potential_cap: float,
    inference_relation_temperature: float,
    correction_scale: float,
    seen_logit_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return Raw Off, Calibrated Off, and tuned Full on the complete class axis."""
    parent = model.deployed_parent_logits(image_features)
    readout = model.read_images(image_features)
    relations = model.relation_embeddings.to(readout.device)
    scores = torch.einsum("bd,ekd->bek", readout, relations) / float(
        inference_relation_temperature
    )
    difference = scores[..., 0] - scores[..., 1]
    candidates = parent.detach().topk(int(candidate_top_k), dim=1).indices
    selected = torch.zeros_like(parent, dtype=torch.bool)
    selected.scatter_(1, candidates, True)
    edges = model.edge_index.to(parent.device)
    active = selected[:, edges[:, 0]] & selected[:, edges[:, 1]]
    difference = difference * active
    incidence = model.incidence.to(parent.device)
    mapping = torch.linalg.solve(
        incidence.T @ incidence
        + float(ridge_lambda) * torch.eye(model.class_count, device=parent.device),
        incidence.T,
    )
    potential = difference @ mapping.T
    potential -= potential.mean(dim=1, keepdim=True)
    norm = potential.abs().amax(dim=1, keepdim=True)
    potential = float(potential_cap) * potential / torch.maximum(
        norm, torch.full_like(norm, float(potential_cap))
    )
    correction = (
        float(correction_scale)
        * model.beta().detach()
        * parent.std(dim=1, unbiased=False, keepdim=True)
        * potential
    )
    calibrated = parent.clone()
    full = parent + correction
    seen = model.seen_classes.to(parent.device)
    calibrated[:, seen] -= float(seen_logit_gamma)
    full[:, seen] -= float(seen_logit_gamma)
    return parent, calibrated, full, float(active.float().mean())


def _transitions(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor) -> dict:
    old = before.eq(labels.cpu())
    new = after.eq(labels.cpu())
    return {
        "corrected_wrong_to_right": int((~old & new).sum()),
        "damaged_right_to_wrong": int((old & ~new).sum()),
        "net_correct": int(new.sum() - old.sum()),
    }


@torch.no_grad()
def evaluate(model, tensors: dict, config: dict, device: torch.device) -> dict:
    model.eval()
    outputs = {
        "raw": {"seen": [], "unseen": [], "zs": []},
        "calibrated": {"seen": [], "unseen": [], "zs": []},
        "full": {"seen": [], "unseen": [], "zs": []},
    }
    active_sum = 0.0
    active_count = 0
    unseen = model.unseen_classes.to(device)
    for split, features in (
        ("seen", tensors["test_seen_features"]),
        ("unseen", tensors["test_unseen_features"]),
    ):
        for start in range(0, len(features), 256):
            images = features[start : start + 256].to(device).float()
            raw, calibrated, full, active = tuned_inference_logits(
                model,
                images,
                candidate_top_k=int(config["candidate_top_k"]),
                ridge_lambda=float(config["ridge_lambda"]),
                potential_cap=float(config["potential_cap"]),
                inference_relation_temperature=float(
                    config["inference_relation_temperature"]
                ),
                correction_scale=float(config["correction_scale"]),
                seen_logit_gamma=float(config["seen_logit_gamma"]),
            )
            for name, logits in (
                ("raw", raw),
                ("calibrated", calibrated),
                ("full", full),
            ):
                outputs[name][split].append(logits.argmax(dim=1).cpu())
                if split == "unseen":
                    outputs[name]["zs"].append(
                        unseen[logits.index_select(1, unseen).argmax(dim=1)].cpu()
                    )
            active_sum += active * images.size(0)
            active_count += images.size(0)
    for group in outputs.values():
        for split in group:
            group[split] = torch.cat(group[split])
    seen = model.seen_classes.cpu()
    unseen_cpu = model.unseen_classes.cpu()
    labels_seen = tensors["test_seen_labels"].long()
    labels_unseen = tensors["test_unseen_labels"].long()

    def scores(predictions):
        seen_score = 100 * per_class_accuracy(
            labels_seen, predictions["seen"], seen
        )
        unseen_score = 100 * per_class_accuracy(
            labels_unseen, predictions["unseen"], unseen_cpu
        )
        zs = 100 * per_class_accuracy(labels_unseen, predictions["zs"], unseen_cpu)
        harmonic = 2 * seen_score * unseen_score / (seen_score + unseen_score)
        return {"U": float(unseen_score), "S": float(seen_score), "H": float(harmonic), "ZS": float(zs)}

    raw_scores = scores(outputs["raw"])
    calibrated_scores = scores(outputs["calibrated"])
    full_scores = scores(outputs["full"])
    transitions = {
        name: {
            "seen": _transitions(outputs[name]["seen"], outputs["full"]["seen"], labels_seen),
            "unseen": _transitions(
                outputs[name]["unseen"], outputs["full"]["unseen"], labels_unseen
            ),
            "zs": _transitions(outputs[name]["zs"], outputs["full"]["zs"], labels_unseen),
        }
        for name in ("raw", "calibrated")
    }
    return {
        "full_metrics": full_scores,
        "raw_off_metrics": raw_scores,
        "calibrated_off_metrics": calibrated_scores,
        "delta_H_vs_raw_off": full_scores["H"] - raw_scores["H"],
        "delta_H_vs_calibrated_off": full_scores["H"] - calibrated_scores["H"],
        "transitions": transitions,
        "active_edge_rate": active_sum / active_count,
        "effective_beta": float(config["correction_scale"]) * float(model.beta()),
        "effective_beta_max": float(config["correction_scale"]) * float(model.max_beta),
    }


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("PCLR R3 expected commit mismatch.")
    config, config_sha = load_inference_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != config["experiment_id"]:
        raise ValueError("PCLR R3 config SHA or output identity mismatch.")
    for key in ("source_config", "source_checkpoint", "source_metrics", "source_module_off_history"):
        path = Path(config[key])
        expected = config[f"{key}_sha256"]
        if not path.is_absolute() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"PCLR R3 source artifact mismatch: {key}")
    source_metrics = json.loads(Path(config["source_metrics"]).read_text(encoding="utf-8"))
    if (
        source_metrics.get("code_commit") != config["source_code_commit"]
        or source_metrics.get("config_sha256") != config["source_config_sha256"]
        or source_metrics.get("module_off_full_history_reproduced") is not True
    ):
        raise ValueError("PCLR R3 source RUN is not canonical.")
    source_config, source_sha = load_config(Path(config["source_config"]))
    if source_sha != config["source_config_sha256"]:
        raise ValueError("PCLR R3 source config loader mismatch.")
    device = torch.device(config["device"])
    tensors = load_assets(source_config)
    model = build_model(source_config, tensors, device)
    checkpoint = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=True)
    if (
        checkpoint.get("code_commit") != config["source_code_commit"]
        or checkpoint.get("config_sha256") != config["source_config_sha256"]
    ):
        raise ValueError("PCLR R3 source checkpoint identity mismatch.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = evaluate(model, tensors, config, device)
    parent_h = 79.070015
    full = result["full_metrics"]
    net = sum(
        result["transitions"]["raw"][split]["net_correct"]
        for split in ("seen", "unseen")
    )
    passed = (
        full["H"] >= float(config["required_h"])
        and full["H"] - parent_h >= float(config["required_delta_h"])
        and result["delta_H_vs_raw_off"] >= float(config["required_delta_h"])
        and abs(full["U"] - full["S"]) < float(config["max_us_gap"])
        and full["ZS"] >= 86.955839 - 0.5
        and net >= 20
    )
    result.update(
        {
            "schema_version": SCHEMA,
            "experiment_id": config["experiment_id"],
            "evaluation_code_commit": code_commit,
            "config_sha256": config_sha,
            "source_code_commit": config["source_code_commit"],
            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "parent_H": parent_h,
            "delta_H_vs_parent": full["H"] - parent_h,
            "net_joint_corrections_vs_raw": net,
            "full_gate_passed": passed,
            "decision": "keep_pclr_r3_inference_tune" if passed else "drop_pclr_r3_gate_failed",
            "test_used_for_selection": True,
            "test_used_for_hyperparameter_selection": True,
            "nested_official_test_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "human_annotations_used": False,
        }
    )
    output = prepare_output_dir(output_dir)
    (output / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    atomic_write_json(output / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()
