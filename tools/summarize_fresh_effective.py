"""Strictly summarize matched TRY042-045 fresh one-stage results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tools.run_contract import atomic_write_json
from tools.runtime import sha256_file


EXPECTED_MODULES = {
    "V3-TRY-042": "tg",
    "V3-TRY-043": "gtd",
    "V3-TRY-044": "mmt",
    "V3-TRY-045": "bd",
}
CANDIDATES = ("V3-TRY-043", "V3-TRY-044", "V3-TRY-045")
CONFIG_REFS = {
    experiment_id: f"config/tries/v3_try_{experiment_id[-3:]}_fresh_effective.yaml"
    for experiment_id in EXPECTED_MODULES
}
MATCHED_FIELDS = (
    "asset_id",
    "asset_manifest_sha256",
    "initial_tg_state_sha256",
    "initial_parent_state_sha256",
    "primary_batch_generator_initial_sha256",
    "primary_batches_updates_1_142_sha256",
    "post_build_cuda_rng_sha256",
    "update1_pre_main_cuda_rng_sha256",
    "random_seed",
    "batch_size",
    "total_updates",
    "eval_interval_steps",
    "tg_learning_rate",
)
METRICS = ("U", "S", "H", "ZS")


def _history_rows(value) -> list[dict]:
    if isinstance(value, dict):
        value = value.get("rows")
    if not isinstance(value, list):
        raise ValueError("fresh汇总history必须是rows列表。")
    return value


def _first_strict_max(rows: list[dict], metric: str) -> dict:
    best = rows[0]
    for row in rows[1:]:
        if float(row[metric]) > float(best[metric]):
            best = row
    return best


def _validate_history(experiment_id: str, result: dict, history: list[dict]) -> None:
    expected_updates = [0] + [141 * index for index in range(1, 151)] + [21171]
    if len(history) != 152 or [row.get("update") for row in history] != expected_updates:
        raise ValueError(f"{experiment_id}评估history不是完整152点。")
    if [row.get("evaluation_index") for row in history] != list(range(152)):
        raise ValueError(f"{experiment_id} evaluation_index不连续。")
    if int(result.get("history_length", -1)) != 152:
        raise ValueError(f"{experiment_id} result history_length错误。")
    for row in history:
        for metric in METRICS:
            expected_delta = float(row[metric]) - float(row["module_off_metrics"][metric])
            if float(row["full_minus_off_delta"][metric]) != expected_delta:
                raise ValueError(f"{experiment_id} Full-Off数值不自洽。")
    best_h = _first_strict_max(history, "H")
    if int(result.get("best_update", -1)) != int(best_h["update"]) or result.get("best_metrics") != best_h:
        raise ValueError(f"{experiment_id} best-H不是history首次严格最大值。")
    best_zs = _first_strict_max(history, "ZS")
    observation = result.get("best_zs_observation", {})
    if (
        int(observation.get("update", -1)) != int(best_zs["update"])
        or float(observation.get("ZS", float("nan"))) != float(best_zs["ZS"])
        or observation.get("metrics") != best_zs
    ):
        raise ValueError(f"{experiment_id} best-ZS不是history首次严格最大值。")


def summarize(
    payloads: list[dict], histories: dict[str, object] | list[object], *,
    expected_code_commit: str, repo_root: Path,
) -> dict:
    by_id = {row.get("experiment_id"): row for row in payloads}
    if set(by_id) != set(EXPECTED_MODULES) or len(payloads) != 4:
        raise ValueError(f"fresh汇总必须恰好包含{sorted(EXPECTED_MODULES)}。")
    if isinstance(histories, list):
        history_by_id = {
            row["experiment_id"]: history for row, history in zip(payloads, histories)
        }
    else:
        history_by_id = dict(histories)
    if set(history_by_id) != set(EXPECTED_MODULES):
        raise ValueError("fresh汇总缺少四RUN history。")
    repo_root = Path(repo_root).resolve()
    expected_code_commit = str(expected_code_commit)
    control = by_id["V3-TRY-042"]
    for experiment_id, expected_module in EXPECTED_MODULES.items():
        row = by_id[experiment_id]
        config_path = repo_root / CONFIG_REFS[experiment_id]
        if not config_path.is_file():
            raise ValueError(f"{experiment_id}仓库config不存在。")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if (
            row.get("module") != expected_module
            or config.get("experiment_id") != experiment_id
            or config.get("module") != expected_module
        ):
            raise ValueError(f"{experiment_id}的ID→module映射错误。")
        if row.get("code_commit") != expected_code_commit:
            raise ValueError(f"{experiment_id} code_commit不等于CLI expected commit。")
        actual_config_sha = sha256_file(config_path)
        if row.get("config_sha256") != actual_config_sha:
            raise ValueError(f"{experiment_id} config SHA与仓库重算不一致。")
        if (
            row.get("initialization_strategy") != "fresh_seeded_tg"
            or row.get("loaded_training_checkpoints") != []
            or config.get("tg_checkpoint") is not None
            or config.get("tg_checkpoint_sha256") is not None
            or config.get("pretrained_module_checkpoint") is not None
            or row.get("asset_id") != config.get("asset_id")
            or row.get("asset_manifest_sha256") != config.get("asset_manifest_sha256")
            or int(row.get("random_seed", -1)) != int(config.get("random_seed", -2))
            or int(row.get("batch_size", -1)) != int(config.get("batch_size", -2))
            or int(row.get("total_updates", -1)) != int(config.get("total_updates", -2))
            or int(row.get("eval_interval_steps", -1)) != int(config.get("eval_interval_steps", -2))
            or float(row.get("tg_learning_rate", -1.0)) != float(config.get("tg_learning_rate", -2.0))
        ):
            raise ValueError(f"{experiment_id} fresh/config/资产/预算身份错误。")
        _validate_history(experiment_id, row, _history_rows(history_by_id[experiment_id]))
    for experiment_id, row in by_id.items():
        for field in MATCHED_FIELDS:
            if row.get(field) != control.get(field):
                raise ValueError(f"{experiment_id}的{field}与TRY042不匹配。")

    control_history = _history_rows(history_by_id["V3-TRY-042"])
    control_h = float(control["best_metrics"]["H"])
    candidates = []
    for experiment_id in CANDIDATES:
        row = by_id[experiment_id]
        history = _history_rows(history_by_id[experiment_id])
        mismatches = []
        for index, (candidate_row, control_row) in enumerate(zip(history, control_history)):
            for metric in METRICS:
                if float(candidate_row["module_off_metrics"][metric]) != float(control_row[metric]):
                    mismatches.append({
                        "evaluation_index": index,
                        "update": int(control_row["update"]),
                        "metric": metric,
                        "candidate_off": float(candidate_row["module_off_metrics"][metric]),
                        "control_full": float(control_row[metric]),
                    })
        if mismatches:
            candidates.append({
                "experiment_id": experiment_id,
                "module": row["module"],
                "decision": "implementation_invalid",
                "trajectory_mismatch_count": len(mismatches),
                "first_trajectory_mismatch": mismatches[0],
                "cross_run_add_delta_H_vs_try042": None,
                "same_checkpoint_full_minus_off_delta_H": None,
            })
            continue
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
            "trajectory_mismatch_count": 0,
            "decision": decision,
        })
    return {
        "schema_version": "gzsl-paper.v3-fresh-effective-summary.v2",
        "expected_code_commit": expected_code_commit,
        "control": {"experiment_id": "V3-TRY-042", "best_metrics": control["best_metrics"]},
        "matched_identity": {field: control[field] for field in MATCHED_FIELDS},
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
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError("fresh汇总output必须是绝对路径。")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    histories = {
        row["experiment_id"]: json.loads(
            (path.parent / "evaluation_history.json").read_text(encoding="utf-8")
        )
        for path, row in zip(args.metrics, payloads)
    }
    result = summarize(
        payloads, histories, expected_code_commit=args.expected_commit,
        repo_root=args.repo_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
