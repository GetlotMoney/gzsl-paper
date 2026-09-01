"""Evaluate a saved RGRA checkpoint with same-checkpoint module-off controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.frameworks.v6.train_rgra import build_model, evaluate, load_assets, load_config
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


def run(config_path: Path, checkpoint_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str | None = None) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("RGRA eval expected-commit mismatch.")
    config, config_sha = load_config(config_path)
    if expected_config_sha is not None and config_sha != expected_config_sha:
        raise ValueError("RGRA eval expected-config-sha mismatch.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("code_commit") != expected_commit or checkpoint.get("config_sha256") != config_sha:
        raise ValueError("RGRA checkpoint identity mismatch.")
    device = torch.device(config["device"])
    tensors = load_assets(config)
    model = build_model(config, tensors, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = {
        "schema_version": "gzsl-paper.v6-rgra-eval.v1",
        "experiment_id": config["experiment_id"],
        "evaluation_code_commit": code_commit,
        "run_code_commit": expected_commit,
        "config_sha256": config_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics": evaluate(model, tensors, device),
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
    }
    out_dir = prepare_output_dir(output_dir)
    atomic_write_json(out_dir / "metrics.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha")
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()

