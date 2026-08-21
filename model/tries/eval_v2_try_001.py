from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model.tg_vpr_h1 import TGVPRH1FixedEqual
from model.tg_vpr_h1 import train as h1
from model.tries.v2_try_001 import TGVPRH1UnseenValueTransfer
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


ATTEMPT_ID = "V2-TRY-001"
BASE_FRAMEWORK_COMMIT = "3dc078c0d52bf358bf24a26e48346c97de9e99ca"


def _build_model(model_type, config, tensors, seenclasses):
    centroids = h1.visual_centroids(
        tensors["train_features"], tensors["train_labels"].long(), seenclasses
    )
    return model_type(
        tensors["sentence_embeds"],
        seenclasses,
        centroids,
        dropout=config["dropout"],
        inner_ratio=config["inner_ratio"],
        outer_ratio=config["outer_ratio"],
        temperature=config["temperature"],
    )


def run(config_path, checkpoint_path, checkpoint_sha256, output_dir, expected_commit):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前HEAD不一致。")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("V2基线checkpoint SHA-256不匹配。")
    config, config_sha = h1.load_config(config_path)
    paths = h1.resolve_paths(config)
    input_sha = h1.verify_inputs(
        config, paths, h1.TRAINING_KEYS + h1.OFFICIAL_KEYS
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("framework_id") != "FRAMEWORK-V2":
        raise ValueError("checkpoint不是FRAMEWORK-V2。")
    if checkpoint.get("code_commit") != BASE_FRAMEWORK_COMMIT:
        raise ValueError("checkpoint不来自冻结V2代码。")

    tensors = {
        name: torch.load(paths[name], map_location="cpu", weights_only=True)
        for name in (
            "sentence_embeds",
            "train_features",
            "train_labels",
            "seen_features",
            "seen_labels",
            "unseen_features",
            "unseen_labels",
        )
    }
    seenclasses = torch.unique(tensors["train_labels"].long(), sorted=True)
    allclasses = torch.arange(200)
    unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V2-TRY-001要求CUDA评估。")

    baseline = _build_model(TGVPRH1FixedEqual, config, tensors, seenclasses)
    candidate = _build_model(
        TGVPRH1UnseenValueTransfer, config, tensors, seenclasses
    )
    baseline.load_state_dict(checkpoint["model_state_dict"], strict=True)
    candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    baseline = baseline.to(device).eval()
    candidate = candidate.to(device).eval()
    baseline_metrics = h1.evaluate(
        baseline, tensors, seenclasses, unseenclasses, device
    )
    candidate_metrics = h1.evaluate(
        candidate, tensors, seenclasses, unseenclasses, device
    )
    delta = {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in ("U", "S", "H", "ZS")
    }

    output_dir = prepare_output_dir(output_dir)
    atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
    metrics = {
        "attempt_id": ATTEMPT_ID,
        "idea_id": "IDEA-002",
        "framework_id": "FRAMEWORK-V2",
        "code_commit": code_commit,
        "base_framework_commit": BASE_FRAMEWORK_COMMIT,
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_protocol": h1.EVALUATION_PROTOCOL,
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
        "change": "eval-only shared Value transfer to unseen prototypes",
        "baseline_metrics_percent": baseline_metrics,
        "candidate_metrics_percent": candidate_metrics,
        "delta_percent_points": delta,
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "evaluation.log").open("x", encoding="utf-8") as stream:
        stream.write(f"attempt={ATTEMPT_ID}\n")
        stream.write(f"code_commit={code_commit}\n")
        stream.write(f"checkpoint_sha256={checkpoint_sha256}\n")
        stream.write(f"baseline={baseline_metrics}\n")
        stream.write(f"candidate={candidate_metrics}\n")
        stream.write(f"delta={delta}\n")
    print(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(
        args.config,
        args.checkpoint,
        args.checkpoint_sha256,
        args.output_dir,
        args.expected_commit,
    )


if __name__ == "__main__":
    main()

