"""Summarize matched TRY042-045 fresh one-stage results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.run_contract import atomic_write_json


CANDIDATES = ("V3-TRY-043", "V3-TRY-044", "V3-TRY-045")


def summarize(payloads: list[dict]) -> dict:
    by_id = {row.get("experiment_id"): row for row in payloads}
    expected = {"V3-TRY-042", *CANDIDATES}
    if set(by_id) != expected:
        raise ValueError(f"fresh汇总必须恰好包含{sorted(expected)}。")
    identity_fields = (
        "initial_tg_state_sha256",
        "initial_parent_state_sha256",
        "primary_batch_generator_initial_sha256",
        "primary_batches_updates_1_142_sha256",
    )
    control = by_id["V3-TRY-042"]
    for experiment_id, row in by_id.items():
        if row.get("initialization_strategy") != "fresh_seeded_tg":
            raise ValueError(f"{experiment_id}不是fresh TG初始化。")
        if row.get("loaded_training_checkpoints") != []:
            raise ValueError(f"{experiment_id}读取了训练checkpoint。")
        for field in identity_fields:
            if row.get(field) != control.get(field):
                raise ValueError(f"{experiment_id}的{field}与TRY042不匹配。")
    control_h = float(control["best_metrics"]["H"])
    candidates = []
    for experiment_id in CANDIDATES:
        row = by_id[experiment_id]
        best = row["best_metrics"]
        add_delta = float(best["H"]) - control_h
        module_delta = float(row["best_full_minus_off_delta"]["H"])
        gap = abs(float(best["U"]) - float(best["S"]))
        if add_delta >= 1.0 and module_delta >= 1.0 and gap < 8.0:
            decision = "strong_keep"
        elif add_delta >= 0.8 and module_delta >= 0.8 and gap < 8.0:
            decision = "weak_keep"
        else:
            decision = "drop"
        candidates.append({
            "experiment_id": experiment_id,
            "module": row["module"],
            "best_metrics": best,
            "cross_run_add_delta_H_vs_try042": add_delta,
            "same_checkpoint_full_minus_off_delta_H": module_delta,
            "gap_U_S": gap,
            "decision": decision,
        })
    return {
        "schema_version": "gzsl-paper.v3-fresh-effective-summary.v1",
        "control": {
            "experiment_id": "V3-TRY-042",
            "best_metrics": control["best_metrics"],
        },
        "matched_identity": {field: control[field] for field in identity_fields},
        "strong_threshold_H": 1.0,
        "weak_threshold_H": 0.8,
        "max_gap_U_S_exclusive": 8.0,
        "candidates": candidates,
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError("fresh汇总output必须是绝对路径。")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    result = summarize(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
