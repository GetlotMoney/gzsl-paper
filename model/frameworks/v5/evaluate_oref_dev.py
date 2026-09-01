"""Frozen preliminary OREF Gate: Full/off versus ordinary visible-token controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from model.frameworks.v5.oref import OREFModel
from model.frameworks.v5.oref_data import load_subset, validate_bundle
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v5-oref-dev-eval.v1"
CONDITIONS = {
    "full": "OREF_FULL", "ledger_mlp": "OREF_LEDGER_MLP",
    "filip": "OREF_FILIP", "signed_ledger": "OREF_SIGNED_LEDGER",
}


def _load_checkpoint(spec, expected, bundle_id):
    path = Path(spec["path"])
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"OREF checkpoint SHA错误：{expected}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        value.get("condition_id") != expected
        or value.get("code_commit") != spec["training_commit"]
        or value.get("bundle_id") != bundle_id
    ):
        raise ValueError(f"OREF checkpoint身份错误：{expected}")
    return value


@torch.no_grad()
def _infer(checkpoint, *, manifest, manifest_sha, bundle, bundle_sha, device, batch_size, chunk, mode):
    open_roles = mode != "s_off"
    open_patches = mode != "v_off"
    values, patches, subset = load_subset(
        Path(manifest), manifest_sha, subset="dev_eval",
        open_patches=open_patches, open_roles=open_roles,
    )
    validate_bundle(Path(bundle), bundle_sha, subset="dev_eval", subset_sha=manifest_sha)
    roles = values.get("role_embeddings")
    if roles is None:
        roles = torch.zeros(values["class_ids"].numel(), 8, 768)
    model = OREFModel(
        values["name_embeddings"], roles, values["class_ids"],
        candidate_chunk_size=chunk,
    ).to(device)
    model.visual_module.load_state_dict(checkpoint["visual_state_dict"], strict=True)
    model.interaction_module.ledger_mlp.load_state_dict(checkpoint["ledger_mlp_state_dict"], strict=True)
    model.eval()
    rows = {key: [] for key in ("logits", "base_logits", "score")}
    count = values["labels"].numel()
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        patch = None
        if open_patches:
            patch = torch.from_numpy(np.array(patches[start:end], copy=True)).to(device).float()
        output = model(values["image_cls"][start:end].to(device).float(), patch, mode=mode)
        for key in rows:
            if key in output:
                rows[key].append(output[key].cpu())
    result = {key: torch.cat(value) for key, value in rows.items() if value}
    result["labels"] = values["labels"].long()
    result["class_ids"] = values["class_ids"].long()
    result["image_order_sha256"] = subset["image_order_sha256"]
    result["opened_asset_keys"] = ["image_cls", "name_embeddings", "labels"] + (["role_embeddings"] if open_roles else []) + (["patch_tokens"] if open_patches else [])
    return result


def _metrics(output):
    predictions = output["class_ids"][output["logits"].argmax(1)]
    classes = torch.unique(output["labels"], sorted=True)
    vector = torch.stack([
        predictions[output["labels"].eq(class_id)].eq(class_id).double().mean()
        for class_id in classes
    ])
    return {
        "macro_top1": 100.0 * float(vector.mean()),
        "micro_top1": 100.0 * float(predictions.eq(output["labels"]).double().mean()),
        "per_class": vector, "prediction": predictions,
    }


def _comparison(full, other, matrix):
    diff = 100.0 * (full - other)
    samples = diff[matrix].mean(1)
    ci = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.double))
    return {"observed_pp": float(diff.mean()), "ci95": [float(ci[0]), float(ci[1])]}


def run(config_path, output_path, expected_commit, expected_config_sha):
    require_clean_code_tree()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(config_path)
    if (
        config.get("schema_version") != SCHEMA
        or config_sha != expected_config_sha or current_code_commit() != expected_commit
        or set(config.get("checkpoints", {})) != set(CONDITIONS)
        or int(config.get("batch_size", -1)) != 4
        or int(config.get("candidate_chunk_size", -1)) != 5
        or int(config.get("bootstrap_samples", -1)) != 10000
        or int(config.get("bootstrap_seed", -1)) != 7
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("official_test_loaded") is not False
        or config.get("pclr_online_inference") is not False
    ):
        raise ValueError("OREF eval配置身份错误。")
    # Read bundle identity without opening any condition-specific tensors.
    bundle_meta = validate_bundle(
        Path(config["bundle_manifest"]), config["bundle_manifest_sha256"],
        subset="dev_eval", subset_sha=config["eval_manifest_sha256"],
    )
    bundle_id = bundle_meta["common_identity"]["bundle_id"]
    checkpoints = {
        key: _load_checkpoint(config["checkpoints"][key], condition, bundle_id)
        for key, condition in CONDITIONS.items()
    }
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    device = torch.device(config["device"])
    common_args = dict(
        manifest=config["eval_manifest"], manifest_sha=config["eval_manifest_sha256"],
        bundle=config["bundle_manifest"], bundle_sha=config["bundle_manifest_sha256"],
        device=device, batch_size=4, chunk=5,
    )
    full = _infer(checkpoints["full"], mode="full", **common_args)
    outputs = {"full": full, "parent": {**full, "logits": full["base_logits"]}}
    for name, mode in (("s_off", "s_off"), ("v_off", "v_off"), ("i_off", "i_off")):
        outputs[name] = _infer(checkpoints["full"], mode=mode, **common_args)
    for name in ("ledger_mlp", "filip", "signed_ledger"):
        outputs[name] = _infer(checkpoints[name], mode=name, **common_args)
    order_shas = {output["image_order_sha256"] for output in outputs.values()}
    if len(order_shas) != 1 or next(iter(order_shas)) != config["targetfree_image_order_sha256"]:
        raise ValueError("OREF与RCEG Target-free eval rows不一致，必须重跑比较器。")
    targetfree_path = Path(config["targetfree_per_class_receipt"])
    if not targetfree_path.is_file() or sha256_file(targetfree_path) != config["targetfree_per_class_receipt_sha256"]:
        raise ValueError("OREF Target-free逐类收据路径/SHA错误。")
    targetfree_receipt = json.loads(targetfree_path.read_text(encoding="utf-8"))
    targetfree_vector = torch.tensor(targetfree_receipt.get("per_class", []), dtype=torch.double)
    if (
        targetfree_vector.shape != (50,)
        or targetfree_receipt.get("image_order_sha256") != config["targetfree_image_order_sha256"]
        or abs(100.0 * float(targetfree_vector.mean()) - float(config["targetfree_macro_top1"])) > 1e-8
    ):
        raise ValueError("OREF Target-free逐类收据身份/均值错误。")
    metrics = {name: _metrics(output) for name, output in outputs.items()}
    classes = torch.unique(full["labels"], sorted=True)
    generator = torch.Generator(device="cpu").manual_seed(7)
    matrix = torch.randint(0, 50, (10000, 50), generator=generator)
    comparisons = {
        name: _comparison(metrics["full"]["per_class"], metrics[name]["per_class"], matrix)
        for name in ("parent", "s_off", "v_off", "i_off", "ledger_mlp", "filip", "signed_ledger")
    }
    comparisons["targetfree"] = _comparison(
        metrics["full"]["per_class"], targetfree_vector, matrix
    )
    labels = full["labels"]
    class_map = torch.full((200,), -1, dtype=torch.long)
    class_map[full["class_ids"]] = torch.arange(full["class_ids"].numel())
    true = class_map[labels]
    rows = torch.arange(labels.numel())
    wrong = full["base_logits"].clone()
    wrong[rows, true] = -torch.inf
    hard_wrong = wrong.argmax(1)
    wins = full["score"][rows, true] > full["score"][rows, hard_wrong]
    direction_vector = torch.stack([wins[labels.eq(class_id)].double().mean() for class_id in classes])
    direction_samples = 100.0 * direction_vector[matrix].mean(1)
    direction_ci = torch.quantile(direction_samples, torch.tensor([0.025, 0.975], dtype=torch.double))
    parent_pred, full_pred = metrics["parent"]["prediction"], metrics["full"]["prediction"]
    corrected = full_pred.eq(labels) & parent_pred.ne(labels)
    damaged = full_pred.ne(labels) & parent_pred.eq(labels)
    transitions = {"corrected": int(corrected.sum()), "damaged": int(damaged.sum()), "net": int(corrected.sum()-damaged.sum()), "changed": int(full_pred.ne(parent_pred).sum())}
    gates = {
        "parent_plus_1": comparisons["parent"]["observed_pp"] >= 1 and comparisons["parent"]["ci95"][0] > 0,
        "targetfree_plus_0p5": comparisons["targetfree"]["observed_pp"] >= 0.5 and comparisons["targetfree"]["ci95"][0] > 0,
        **{f"{name}_plus_{'1' if name.endswith('off') else '0p5'}": comparisons[name]["observed_pp"] >= (1 if name.endswith("off") else 0.5) and comparisons[name]["ci95"][0] > 0 for name in ("s_off", "v_off", "i_off", "ledger_mlp", "filip", "signed_ledger")},
        "direction": 100.0 * float(direction_vector.mean()) >= 60 and float(direction_ci[0]) > 50,
        "net_positive": transitions["net"] > 0,
    }
    passed = all(gates.values())
    result = {
        "schema_version": SCHEMA, "code_commit": expected_commit,
        "config_sha256": config_sha, "asset_bundle_id": bundle_id,
        "metrics": {name: {"macro_top1": value["macro_top1"], "micro_top1": value["micro_top1"], "opened_asset_keys": outputs[name]["opened_asset_keys"]} for name, value in metrics.items()},
        "targetfree_comparator": {
            "macro_top1": config["targetfree_macro_top1"],
            "receipt_sha256": config["targetfree_receipt_sha256"],
            "per_class_receipt_sha256": config["targetfree_per_class_receipt_sha256"],
            "comparison": comparisons["targetfree"],
        },
        "comparisons": comparisons,
        "direction": {"macro_pct": 100.0 * float(direction_vector.mean()), "ci95": [float(direction_ci[0]), float(direction_ci[1])]},
        "transitions": transitions, "gates": gates, "gate_passed": passed,
        "decision": "continue_remaining_controls" if passed else "drop_oref_preliminary_gate",
        "unseen_images_used_for_gradient": False, "official_test_loaded": False,
        "pclr_online_inference": False,
    }
    output = prepare_output_dir(output_path)
    atomic_write_json(output / ("result.json" if passed else "failure.json"), result)
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
