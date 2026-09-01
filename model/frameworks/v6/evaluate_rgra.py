"""Evaluate a graph-free RGRA export with same-checkpoint controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.frameworks.v6.rgra import RGRAModel
from model.frameworks.v6.rgra_assets import load_rgra_eval_assets
from model.frameworks.v6.train_rgra import (
    EXPORT_SCHEMA,
    evaluate_all_conditions,
    load_config,
)
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


def run(
    config_path: Path,
    export_path: Path,
    output_dir: Path,
    expected_commit: str,
    expected_config_sha: str,
    expected_export_sha: str,
) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("RGRA eval expected commit mismatch.")
    config, config_sha = load_config(config_path)
    if config_sha != expected_config_sha:
        raise ValueError("RGRA eval expected config SHA mismatch.")
    if not export_path.is_file() or sha256_file(export_path) != expected_export_sha:
        raise ValueError("RGRA export path or SHA mismatch.")
    payload = torch.load(export_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != EXPORT_SCHEMA
        or payload.get("code_commit") != code_commit
        or payload.get("config_sha256") != config_sha
    ):
        raise ValueError("RGRA export identity mismatch.")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RGRA official evaluation requires CUDA.")
    model = RGRAModel.from_graph_free_state(payload["package"]).to(device).eval()
    assets = load_rgra_eval_assets(config)
    controls = evaluate_all_conditions(
        model, assets, device, int(config["eval_batch_size"])
    )
    result = {
        "schema_version": "gzsl-paper.v6-rgra-eval.v1",
        "experiment_id": config["experiment_id"],
        "evaluation_code_commit": code_commit,
        "config_sha256": config_sha,
        "export_sha256": expected_export_sha,
        "metrics": controls,
        "eval_asset_identity": assets.identity,
        "pclr_online_inference": False,
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
    }
    destination = prepare_output_dir(output_dir)
    atomic_write_json(destination / "metrics.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--expected-export-sha", required=True)
    args = parser.parse_args()
    run(
        args.config,
        args.export,
        args.output_dir,
        args.expected_commit,
        args.expected_config_sha,
        args.expected_export_sha,
    )


if __name__ == "__main__":
    main()
