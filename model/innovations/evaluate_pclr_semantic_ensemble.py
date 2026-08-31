"""Evaluate the fixed PCLR checkpoint with role6/role0 class-semantic ensemble logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.evaluate_pclr_inference_tuned import (
    load_inference_config,
    tuned_inference_logits,
)
from model.innovations.train_gtd_tst import build_model, load_assets, load_config
from tools.gzsl_data import per_class_accuracy
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v4-pclr-semantic-ensemble.v1"
CONFIG_KEYS = {
    "schema_version", "experiment_id", "source_r3_config", "source_r3_config_sha256",
    "source_r3_metrics", "source_r3_metrics_sha256", "source_checkpoint",
    "source_checkpoint_sha256", "source_code_commit", "role0_weight", "role6_weight",
    "seen_logit_gamma", "required_h", "required_delta_h", "max_us_gap", "device",
    "test_used_for_selection", "test_used_for_hyperparameter_selection",
    "nested_official_test_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "human_annotations_used",
}


def load_semantic_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    invalid = (
        not isinstance(config, dict) or actual != CONFIG_KEYS
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != "V4-TRY-023-R4"
        or config.get("source_r3_config_sha256")
        != "8528b715c9bc6fcf1f21c4e9da0212cd9efab550efe2c038f24844d7a69766a3"
        or config.get("source_r3_metrics_sha256")
        != "39bea2dbf664dc421cd53b2a4f8d219b85f05b9279e7991b596c61d22aa4042a"
        or config.get("source_checkpoint_sha256")
        != "16b5071f21a3217e58a72315029c28b8cfd97b68f812641bd0145d3f5e0702ab"
        or config.get("source_code_commit") != "b0a756dd624e883eb50d19a2455ba06bdc73f118"
        or float(config.get("role0_weight", -1)) != 0.16
        or float(config.get("role6_weight", -1)) != 0.36
        or float(config.get("seen_logit_gamma", -1)) != 0.91
        or float(config.get("required_h", -1)) != 81.0
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
        raise ValueError("PCLR R4 semantic config identity or disclosure changed.")
    return config, sha256_file(path)


@torch.no_grad()
def semantic_ensemble_logits(model, images, *, role0_weight: float, role6_weight: float, gamma: float):
    raw, _, r3, _ = tuned_inference_logits(
        model,
        images,
        candidate_top_k=17,
        ridge_lambda=0.3,
        potential_cap=0.5,
        inference_relation_temperature=0.2,
        correction_scale=6.95,
        seen_logit_gamma=0.0,
    )
    roles = model.parent.tg_vpr.sentence_embeds.float()
    image = F.normalize(images.float(), dim=-1)

    def role_logits(index: int):
        value = image @ F.normalize(roles[:, index], dim=-1).T * model.scale()
        value -= value.mean(dim=1, keepdim=True)
        value = value / value.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return value * r3.std(dim=1, unbiased=False, keepdim=True)

    full = r3 + float(role0_weight) * role_logits(0) + float(role6_weight) * role_logits(6)
    full[:, model.seen_classes.to(full.device)] -= float(gamma)
    r3_control = r3.clone()
    r3_control[:, model.seen_classes.to(full.device)] -= 0.575
    return raw, r3_control, full


def _transitions(before, after, labels):
    old = before.eq(labels.cpu())
    new = after.eq(labels.cpu())
    return {
        "corrected_wrong_to_right": int((~old & new).sum()),
        "damaged_right_to_wrong": int((old & ~new).sum()),
        "net_correct": int(new.sum() - old.sum()),
    }


@torch.no_grad()
def evaluate(model, tensors, config, device):
    model.eval()
    seen = model.seen_classes.cpu()
    unseen = model.unseen_classes.cpu()
    unseen_device = unseen.to(device)
    outputs = {name: {"seen": [], "unseen": [], "zs": []} for name in ("raw", "r3", "r4")}
    for split, features in (("seen", tensors["test_seen_features"]), ("unseen", tensors["test_unseen_features"])):
        for start in range(0, len(features), 256):
            images = features[start:start+256].to(device).float()
            raw, r3, r4 = semantic_ensemble_logits(
                model, images, role0_weight=config["role0_weight"],
                role6_weight=config["role6_weight"], gamma=config["seen_logit_gamma"],
            )
            for name, logits in (("raw", raw), ("r3", r3), ("r4", r4)):
                if tuple(logits.shape) != (len(images), 200) or not torch.isfinite(logits).all():
                    raise RuntimeError(f"PCLR R4 {name} logits invalid.")
                outputs[name][split].append(logits.argmax(dim=1).cpu())
                if split == "unseen":
                    outputs[name]["zs"].append(
                        unseen_device[logits.index_select(1, unseen_device).argmax(dim=1)].cpu()
                    )
    for value in outputs.values():
        for split in value:
            value[split] = torch.cat(value[split])
    labels_seen = tensors["test_seen_labels"].long()
    labels_unseen = tensors["test_unseen_labels"].long()

    def metrics(value):
        s = 100 * per_class_accuracy(labels_seen, value["seen"], seen)
        u = 100 * per_class_accuracy(labels_unseen, value["unseen"], unseen)
        zs = 100 * per_class_accuracy(labels_unseen, value["zs"], unseen)
        return {"U": float(u), "S": float(s), "H": float(2*s*u/(s+u)), "ZS": float(zs)}

    scores = {name: metrics(value) for name, value in outputs.items()}
    transitions = {
        name: {
            "seen": _transitions(outputs[name]["seen"], outputs["r4"]["seen"], labels_seen),
            "unseen": _transitions(outputs[name]["unseen"], outputs["r4"]["unseen"], labels_unseen),
            "zs": _transitions(outputs[name]["zs"], outputs["r4"]["zs"], labels_unseen),
        }
        for name in ("raw", "r3")
    }
    return {"metrics": scores, "transitions": transitions}


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("PCLR R4 expected commit mismatch.")
    config, config_sha = load_semantic_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != config["experiment_id"]:
        raise ValueError("PCLR R4 config/output identity mismatch.")
    for key in ("source_r3_config", "source_r3_metrics", "source_checkpoint"):
        path = Path(config[key])
        if not path.is_absolute() or not path.is_file() or sha256_file(path) != config[f"{key}_sha256"]:
            raise ValueError(f"PCLR R4 source mismatch: {key}")
    r3_config, r3_sha = load_inference_config(Path(config["source_r3_config"]))
    r3_metrics = json.loads(Path(config["source_r3_metrics"]).read_text(encoding="utf-8"))
    if r3_sha != config["source_r3_config_sha256"] or r3_metrics.get("full_gate_passed") is not True:
        raise ValueError("PCLR R4 source R3 is not canonical.")
    source_config, source_sha = load_config(Path(r3_config["source_config"]))
    device = torch.device(config["device"])
    tensors = load_assets(source_config)
    model = build_model(source_config, tensors, device)
    checkpoint = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("code_commit") != config["source_code_commit"] or checkpoint.get("config_sha256") != source_sha:
        raise ValueError("PCLR R4 source checkpoint identity mismatch.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = evaluate(model, tensors, config, device)
    raw, r3, r4 = result["metrics"]["raw"], result["metrics"]["r3"], result["metrics"]["r4"]
    net = sum(result["transitions"]["raw"][split]["net_correct"] for split in ("seen", "unseen"))
    passed = (
        r4["H"] >= config["required_h"]
        and r4["H"] - 79.070015 >= config["required_delta_h"]
        and r4["H"] > r3["H"]
        and abs(r4["U"] - r4["S"]) < config["max_us_gap"]
        and r4["ZS"] >= 86.955839 - 0.5
        and net >= 20
    )
    result.update({
        "schema_version": SCHEMA, "experiment_id": config["experiment_id"],
        "evaluation_code_commit": code_commit, "config_sha256": config_sha,
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "delta_H_vs_parent": r4["H"] - 79.070015,
        "delta_H_vs_r3": r4["H"] - r3["H"],
        "net_joint_corrections_vs_raw": net, "full_gate_passed": passed,
        "decision": "keep_pclr_r4_semantic_ensemble" if passed else "drop_pclr_r4_gate_failed",
        "nested_official_test_selection": True, "test_used_for_selection": True,
        "test_used_for_hyperparameter_selection": True, "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False, "human_annotations_used": False,
    })
    output = prepare_output_dir(output_dir)
    (output / "config.snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_write_json(output / "metrics.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args(); run(args.config, args.output_dir, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()
