"""Create one SHA-bound final paper RUN config without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from model.frameworks.v4.model import CCGR_MODES, TG_MODES, TRANSPORT_MODES
from tools.runtime import sha256_file


PRESETS = {
    "B0_PURE_CLIP": ("off", "off", "off", "no_training"),
    "B1_MEAN8": ("off", "off", "off", "no_training"),
    "M1_TG_VPR": ("full", "off", "off", None),
    "M2_TST_NTR": ("full", "tangent_ntr", "off", None),
    "M3_CCGR": ("full", "tangent_ntr", "class_conditioned_four", None),
}


def build_config(args) -> dict:
    manifest_path = args.asset_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "gzsl-paper.clip-assets.v1":
        raise ValueError("asset manifest schema错误。")
    dataset = manifest["dataset"]
    if args.condition_id in PRESETS:
        tg_mode, transport_mode, ccgr_mode, forced_strategy = PRESETS[args.condition_id]
        strategy = forced_strategy or args.training_strategy
    else:
        tg_mode = args.tg_vpr_mode
        transport_mode = args.transport_mode
        ccgr_mode = args.ccgr_mode
        strategy = args.training_strategy
    if tg_mode not in TG_MODES or transport_mode not in TRANSPORT_MODES or ccgr_mode not in CCGR_MODES:
        raise ValueError("模块模式错误。")
    no_training = strategy == "no_training"
    return {
        "schema_version": "gzsl-paper.paper-v2-run.v1",
        "experiment_id": args.experiment_id,
        "condition_id": args.condition_id,
        "framework_id": "FRAMEWORK-V2",
        "dataset": dataset,
        "asset_manifest": str(manifest_path),
        "asset_manifest_sha256": sha256_file(manifest_path),
        "evaluation_protocol": "chen_shiming_code_aligned_multidataset_test_selected_gzsl",
        "test_used_for_selection": not no_training,
        "test_used_for_hyperparameter_selection": bool(args.hyperparameter_selection),
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
        "training_strategy": strategy,
        "selection_scope": "whole_run_whole_model_only",
        "nested_official_test_selection": False,
        "device": "cuda:0",
        "random_seed": int(args.seed),
        "batch_size": 50,
        "nominal_epochs": 200,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "end_to_end_learning_rate": float(args.end_to_end_lr),
        "stage1_learning_rate": float(args.stage1_lr),
        "stage2_learning_rate": float(args.stage2_lr),
        "stage3_learning_rate": float(args.stage3_lr),
        "tg_vpr_mode": tg_mode,
        "transport_mode": transport_mode,
        "ccgr_mode": ccgr_mode,
        "dropout": 0.5,
        "inner_ratio": 0.35,
        "outer_ratio": 0.65,
        "topology_weight": float(args.topology_weight),
        "temperature": 0.07,
        "transport_hidden_dim": 16,
        "generator_hidden_dim": 32,
        "max_transport_step": float(args.max_transport_step),
        "max_ntr_delta": 0.1,
        "max_generator_magnitude": float(args.max_generator_magnitude),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument(
        "--training-strategy",
        choices=("end_to_end_joint", "stagewise_50_100_50", "no_training"),
        default="stagewise_50_100_50",
    )
    parser.add_argument("--tg-vpr-mode", choices=sorted(TG_MODES), default="full")
    parser.add_argument("--transport-mode", choices=sorted(TRANSPORT_MODES), default="tangent_ntr")
    parser.add_argument("--ccgr-mode", choices=sorted(CCGR_MODES), default="class_conditioned_four")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--topology-weight", type=float, default=0.1)
    parser.add_argument("--max-transport-step", type=float, default=1.5)
    parser.add_argument("--max-generator-magnitude", type=float, default=0.2)
    parser.add_argument("--end-to-end-lr", type=float, default=0.0001)
    parser.add_argument("--stage1-lr", type=float, default=0.0001)
    parser.add_argument("--stage2-lr", type=float, default=0.0001)
    parser.add_argument("--stage3-lr", type=float, default=0.00001)
    parser.add_argument("--hyperparameter-selection", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出配置已存在：{args.output}")
    config = build_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
