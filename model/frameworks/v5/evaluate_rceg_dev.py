"""Frozen 100/50 RCEG proof Gate with module-off and non-equivalence controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from model.frameworks.v5.rceg import RCEGModel
from model.frameworks.v5.rceg_data import load_rceg_subset, validate_bundle
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v5-rceg-dev-eval.v1"
CHECKPOINT_KEYS = {
    "full": "RCEG_FULL", "absolute_role": "RCEG_ABSOLUTE_ROLE",
    "reference_difficulty": "RCEG_REFERENCE_DIFFICULTY",
    "target_free": "RCEG_TARGET_FREE", "target_shuffle": "RCEG_TARGET_SHUFFLE",
    "role_shuffle": "RCEG_ROLE_SHUFFLE",
}
CONFIG_KEYS = {
    "schema_version", "experiment_id", "eval_manifest", "eval_manifest_sha256",
    "bundle_manifest", "bundle_manifest_sha256", "asset_generation_commit",
    "device", "batch_size", "candidate_chunk_size", "bootstrap_seed",
    "bootstrap_samples", "checkpoints", "unseen_images_used_for_gradient",
    "dev_unseen_text_used_for_gradient", "official_test_loaded",
    "pclr_online_inference",
}


def load_config(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("RCEG eval配置字段错误。")
    checkpoints = config.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(CHECKPOINT_KEYS):
        raise ValueError("RCEG eval checkpoint条件不完整。")
    for key, value in checkpoints.items():
        if not isinstance(value, dict) or set(value) != {"path", "sha256", "training_commit"}:
            raise ValueError(f"RCEG eval checkpoint字段错误：{key}")
    invalid = (
        config["schema_version"] != SCHEMA
        or int(config["batch_size"]) != 4
        or int(config["candidate_chunk_size"]) != 5
        or int(config["bootstrap_seed"]) != 7
        or int(config["bootstrap_samples"]) != 10000
        or config["unseen_images_used_for_gradient"] is not False
        or config["dev_unseen_text_used_for_gradient"] is not False
        or config["official_test_loaded"] is not False
        or config["pclr_online_inference"] is not False
    )
    if invalid:
        raise ValueError("RCEG eval固定协议错误。")
    return config, sha256_file(path)


def _role_shuffle_for_eval(
    roles: torch.Tensor, eval_class_ids: torch.Tensor, train_class_ids: torch.Tensor
) -> torch.Tensor:
    output = roles.clone()
    train_mask = torch.isin(eval_class_ids, train_class_ids)
    for mask in (train_mask, ~train_mask):
        positions = torch.where(mask)[0]
        if positions.numel() < 2:
            raise ValueError("RCEG role shuffle block至少需要两类。")
        output[positions] = roles[positions.roll(-1)]
    return output


def _load_checkpoint(spec: dict, expected_condition: str, bundle_id: str):
    path = Path(spec["path"])
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"RCEG checkpoint路径/SHA错误：{expected_condition}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("condition_id") != expected_condition
        or checkpoint.get("code_commit") != spec["training_commit"]
        or checkpoint.get("bundle_id") != bundle_id
        or checkpoint.get("unseen_images_used_for_gradient") is not False
        or checkpoint.get("dev_unseen_text_used_for_gradient") is not False
        or checkpoint.get("pclr_online_inference") is not False
    ):
        raise ValueError(f"RCEG checkpoint身份错误：{expected_condition}")
    return checkpoint


@torch.no_grad()
def _infer(
    checkpoint: dict,
    *,
    values: dict,
    visible,
    device: torch.device,
    batch_size: int,
    candidate_chunk_size: int,
    mode: str,
):
    roles = values["role_embeddings"].float()
    if checkpoint.get("role_shuffle"):
        roles = _role_shuffle_for_eval(
            roles, values["class_ids"].long(), checkpoint["class_ids"].long()
        )
    model = RCEGModel(
        values["name_embeddings"], roles, values["class_ids"],
        candidate_chunk_size=candidate_chunk_size,
    ).to(device)
    model.interaction_module.load_state_dict(checkpoint["interaction_state_dict"], strict=True)
    model.eval()
    rows = {key: [] for key in ("logits", "base_logits", "score", "name_error", "role_error")}
    count = len(values["labels"])
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        ids = torch.arange(start, end)
        visible_batch = torch.from_numpy(np.array(visible[start:end], copy=True)).to(device).float()
        target = None if mode == "target_free" else values["target"][start:end].to(device).float()
        output = model(
            values["image_cls"][start:end].to(device).float(),
            values["masked_cls"][start:end].to(device).float(),
            visible_batch, target, mode=mode,
        )
        for key in rows:
            if key in output:
                rows[key].append(output[key].cpu())
    return {key: torch.cat(value) for key, value in rows.items() if value}


def _per_class_vector(labels, predictions, classes):
    return torch.stack([
        predictions[labels.eq(class_id)].eq(class_id).float().mean()
        for class_id in classes
    ]).double()


def _bootstrap_difference(left, right, matrix):
    difference = 100.0 * (left - right)
    samples = difference[matrix].mean(dim=1)
    quantiles = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.double))
    return {
        "observed_pp": float(difference.mean()),
        "ci95": [float(quantiles[0]), float(quantiles[1])],
    }


def _condition_metrics(output, values):
    class_ids = values["class_ids"].long()
    labels = values["labels"].long()
    predictions = class_ids[output["logits"].argmax(dim=1)]
    classes = torch.unique(labels, sorted=True)
    return {
        "macro_top1": 100.0 * per_class_accuracy(labels, predictions, classes),
        "micro_top1": 100.0 * float(predictions.eq(labels).float().mean()),
        "prediction": predictions,
        "per_class": _per_class_vector(labels, predictions, classes),
    }


def run(config_path: Path, output_path: Path, expected_commit: str, expected_config_sha: str):
    require_clean_code_tree()
    config, config_sha = load_config(config_path)
    if current_code_commit() != expected_commit or config_sha != expected_config_sha:
        raise ValueError("RCEG eval commit/config SHA不匹配。")
    values, visible, manifest = load_rceg_subset(
        Path(config["eval_manifest"]), config["eval_manifest_sha256"],
        expected_subset="dev_eval", include_target=True,
    )
    bundle = validate_bundle(
        Path(config["bundle_manifest"]), config["bundle_manifest_sha256"],
        subset_name="dev_eval", subset_sha256=config["eval_manifest_sha256"],
    )
    common = manifest["common_identity"]
    if (
        common.get("code_commit") != config["asset_generation_commit"]
        or common.get("bundle_id") != bundle.get("common_identity", {}).get("bundle_id")
        or values["class_ids"].numel() != 150
        or torch.unique(values["labels"]).numel() != 50
    ):
        raise ValueError("RCEG eval资产边界错误。")
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    checkpoints = {
        key: _load_checkpoint(spec, CHECKPOINT_KEYS[key], common["bundle_id"])
        for key, spec in config["checkpoints"].items()
    }
    device = torch.device(config["device"])
    full_output = _infer(
        checkpoints["full"], values=values, visible=visible, device=device,
        batch_size=int(config["batch_size"]),
        candidate_chunk_size=int(config["candidate_chunk_size"]), mode="full",
    )
    outputs = {
        "full": full_output,
        "parent": {"logits": full_output["base_logits"]},
    }
    for name, mode in (("s_off", "s_off"), ("v_off", "v_off"), ("i_off", "i_off")):
        outputs[name] = _infer(
            checkpoints["full"], values=values, visible=visible, device=device,
            batch_size=int(config["batch_size"]),
            candidate_chunk_size=int(config["candidate_chunk_size"]), mode=mode,
        )
    for name, mode in (
        ("absolute_role", "absolute_role"),
        ("reference_difficulty", "reference_difficulty"),
        ("target_free", "target_free"),
        ("target_shuffle", "full"),
        ("role_shuffle", "full"),
    ):
        outputs[name] = _infer(
            checkpoints[name], values=values, visible=visible, device=device,
            batch_size=int(config["batch_size"]),
            candidate_chunk_size=int(config["candidate_chunk_size"]), mode=mode,
        )
    metrics = {name: _condition_metrics(output, values) for name, output in outputs.items()}
    classes = torch.unique(values["labels"].long(), sorted=True)
    generator = torch.Generator(device="cpu").manual_seed(int(config["bootstrap_seed"]))
    matrix = torch.randint(
        0, classes.numel(), (int(config["bootstrap_samples"]), classes.numel()),
        generator=generator,
    )
    comparisons = {
        name: _bootstrap_difference(metrics["full"]["per_class"], metrics[name]["per_class"], matrix)
        for name in ("parent", "s_off", "v_off", "i_off", "absolute_role", "reference_difficulty", "target_free")
    }
    labels = values["labels"].long()
    class_map = torch.full((200,), -1, dtype=torch.long)
    class_map[values["class_ids"].long()] = torch.arange(values["class_ids"].numel())
    true_pos = class_map[labels]
    base = full_output["base_logits"].clone()
    row = torch.arange(labels.numel())
    base[row, true_pos] = -torch.inf
    hard_wrong = base.argmax(dim=1)
    gain_win = full_output["score"][row, true_pos] > full_output["score"][row, hard_wrong]
    role_win = full_output["role_error"].mean(dim=-1)[row, true_pos] < full_output["role_error"].mean(dim=-1)[row, hard_wrong]
    def rate_payload(wins):
        vector = torch.stack([wins[labels.eq(class_id)].double().mean() for class_id in classes])
        samples = 100.0 * vector[matrix].mean(dim=1)
        ci = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.double))
        return {"macro_pct": 100.0 * float(vector.mean()), "ci95": [float(ci[0]), float(ci[1])]}
    direction = {"gain": rate_payload(gain_win), "absolute_role": rate_payload(role_win)}
    parent_prediction = metrics["parent"]["prediction"]
    full_prediction = metrics["full"]["prediction"]
    corrected = full_prediction.eq(labels) & parent_prediction.ne(labels)
    damaged = full_prediction.ne(labels) & parent_prediction.eq(labels)
    transitions = {
        "corrected": int(corrected.sum()), "damaged": int(damaged.sum()),
        "net": int(corrected.sum() - damaged.sum()),
        "changed": int(full_prediction.ne(parent_prediction).sum()),
    }
    full_gain = comparisons["parent"]["observed_pp"]
    shuffle_retention = {
        name: max(0.0, metrics[name]["macro_top1"] - metrics["parent"]["macro_top1"]) / max(full_gain, 1e-12)
        for name in ("target_shuffle", "role_shuffle")
    }
    gates = {
        "gain_direction": direction["gain"]["macro_pct"] >= 60.0 and direction["gain"]["ci95"][0] > 50.0,
        "absolute_direction": direction["absolute_role"]["macro_pct"] >= 60.0 and direction["absolute_role"]["ci95"][0] > 50.0,
        "prediction_changed": transitions["changed"] > 0,
        "net_correction_positive": transitions["net"] > 0,
        "parent_plus_1": comparisons["parent"]["observed_pp"] >= 1.0 and comparisons["parent"]["ci95"][0] > 0,
        "s_off_plus_1": comparisons["s_off"]["observed_pp"] >= 1.0 and comparisons["s_off"]["ci95"][0] > 0,
        "v_off_plus_1": comparisons["v_off"]["observed_pp"] >= 1.0 and comparisons["v_off"]["ci95"][0] > 0,
        "i_off_plus_1": comparisons["i_off"]["observed_pp"] >= 1.0 and comparisons["i_off"]["ci95"][0] > 0,
        "absolute_control_plus_0p5": comparisons["absolute_role"]["observed_pp"] >= 0.5 and comparisons["absolute_role"]["ci95"][0] > 0,
        "reference_control_plus_0p5": comparisons["reference_difficulty"]["observed_pp"] >= 0.5 and comparisons["reference_difficulty"]["ci95"][0] > 0,
        "target_free_plus_0p5": comparisons["target_free"]["observed_pp"] >= 0.5 and comparisons["target_free"]["ci95"][0] > 0,
        "target_shuffle_destroyed": shuffle_retention["target_shuffle"] <= 0.2,
        "role_shuffle_destroyed": shuffle_retention["role_shuffle"] <= 0.2,
    }
    gate_passed = all(gates.values())
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "code_commit": expected_commit,
        "config_sha256": config_sha,
        "asset_bundle_id": common["bundle_id"],
        "metrics": {name: {key: value for key, value in payload.items() if key not in {"prediction", "per_class"}} for name, payload in metrics.items()},
        "comparisons": comparisons, "direction": direction,
        "transitions": transitions, "shuffle_retention": shuffle_retention,
        "gates": gates, "gate_passed": gate_passed,
        "decision": "promote_to_formal" if gate_passed else "drop_or_owner_authorized_new_rescue",
        "unseen_images_used_for_gradient": False,
        "dev_unseen_text_used_for_gradient": False,
        "official_test_loaded": False, "pclr_online_inference": False,
    }
    output = prepare_output_dir(output_path)
    atomic_write_json(output / ("result.json" if gate_passed else "failure.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args()
    print(run(args.config, args.output, args.expected_commit, args.expected_config_sha))


if __name__ == "__main__":
    main()
