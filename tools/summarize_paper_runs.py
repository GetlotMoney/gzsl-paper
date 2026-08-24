"""Aggregate final paper metrics without mixing protocols, assets, or checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("U", "S", "H", "ZS")
PROTOCOL = "chen_shiming_code_aligned_multidataset_test_selected_gzsl"


def summarize(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evaluation_protocol") != PROTOCOL:
            raise ValueError(f"混入非最终Chen-style协议：{path}")
        metrics = payload.get("best_metrics_percent")
        if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
            raise ValueError(f"metrics缺少同checkpoint U/S/H/ZS：{path}")
        rows.append(
            {
                "path": str(path),
                "dataset": payload["dataset"],
                "condition_id": payload["condition_id"],
                "training_strategy": payload["training_strategy"],
                "seed": int(payload["seed"]),
                "asset_id": payload["asset_id"],
                "code_commit": payload["code_commit"],
                "selected_iteration": payload["selected_iteration"],
                "selected_stage": payload["selected_stage"],
                **{key: float(metrics[key]) for key in METRICS},
            }
        )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["condition_id"], row["training_strategy"])].append(row)
    summaries = []
    for (dataset, condition, strategy), members in sorted(groups.items()):
        if len({row["seed"] for row in members}) != len(members):
            raise ValueError(f"{dataset}/{condition}/{strategy}包含重复seed。")
        if len({row["asset_id"] for row in members}) != 1:
            raise ValueError(f"{dataset}/{condition}/{strategy}跨asset拼接。")
        if len({row["code_commit"] for row in members}) != 1:
            raise ValueError(f"{dataset}/{condition}/{strategy}跨代码commit拼接。")
        best = max(members, key=lambda row: row["H"])
        summary = {
            "dataset": dataset,
            "condition_id": condition,
            "training_strategy": strategy,
            "asset_id": members[0]["asset_id"],
            "code_commit": members[0]["code_commit"],
            "seed_count": len(members),
            "highest_H_seed": best["seed"],
            "highest_seed_metrics": {key: best[key] for key in METRICS},
            "selected_iteration": best["selected_iteration"],
            "selected_stage": best["selected_stage"],
        }
        for key in METRICS:
            values = [row[key] for row in members]
            summary[f"{key}_mean"] = statistics.fmean(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
            summary[f"{key}_range"] = max(values) - min(values)
        summaries.append(summary)
    return {"schema_version": "gzsl-paper.paper-summary.v1", "runs": rows, "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"汇总输出已存在：{args.output}")
    result = summarize(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summaries"], ensure_ascii=False))


if __name__ == "__main__":
    main()
