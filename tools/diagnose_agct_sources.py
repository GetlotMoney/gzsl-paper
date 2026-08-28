from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model.candidates.v2.modules.agpt import AmbiguityGatedPatchTieBreaker
from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
from model.candidates.v2.modules.lpsr import orthogonal_local_text_residuals
from model.candidates.v2.modules.sdcr import SentenceDropoutConservativeRouting
from model.candidates.v2.modules.tigr import taxonomic_suffix_group_ids
from model.candidates.v2.trainers.train_agct import derive_train_threshold
from model.candidates.v2.trainers.train_agpt import load_config
from model.candidates.v2.trainers.train_ccpe import _precompute_scores
from model.candidates.v2.trainers.train_chen_style import OFFICIAL_KEYS, resolve_paths
from model.candidates.v2.trainers.train_sebc import _load_main
from model.frameworks.v2 import train as h1
from tools.cub_data import load_cub_split
from tools.diagnose_sdcr_errors import load_class_names
from tools.run_contract import atomic_write_json
from tools.runtime import sha256_file


STRATEGIES = (
    "baseline",
    "claude_max",
    "claude_min",
    "merge_max",
    "merge_min",
    "patch_max",
    "patch_min",
    "oracle_top2",
)


def apply_hard_tie_break(
    baseline_predictions: torch.Tensor,
    top_global_ids: torch.Tensor,
    source_top_scores: torch.Tensor,
    hard_gate: torch.Tensor,
    choose_max: bool,
) -> torch.Tensor:
    if tuple(top_global_ids.shape) != tuple(source_top_scores.shape):
        raise ValueError("source top2分数必须与top2类别逐项对应。")
    choice = (
        source_top_scores.argmax(dim=1)
        if choose_max
        else source_top_scores.argmin(dim=1)
    )
    selected = top_global_ids.gather(1, choice.unsqueeze(1)).squeeze(1)
    return torch.where(hard_gate.bool(), selected, baseline_predictions)


def transition_stats(
    labels: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    hard_gate: torch.Tensor,
    top_global_ids: torch.Tensor,
) -> dict[str, int | float]:
    labels = labels.cpu()
    baseline = baseline.cpu()
    candidate = candidate.cpu()
    hard_gate = hard_gate.cpu().bool()
    baseline_correct = baseline.eq(labels)
    candidate_correct = candidate.eq(labels)
    contains_true = top_global_ids.cpu().eq(labels.unsqueeze(1)).any(dim=1)
    corrected = ~baseline_correct & candidate_correct
    broken = baseline_correct & ~candidate_correct
    return {
        "sample_count": int(labels.numel()),
        "hard_gate_count": int(hard_gate.sum()),
        "hard_gate_rate": float(hard_gate.float().mean()),
        "gated_top2_contains_true_count": int((hard_gate & contains_true).sum()),
        "gated_baseline_wrong_count": int((hard_gate & ~baseline_correct).sum()),
        "corrected_count": int(corrected.sum()),
        "broken_count": int(broken.sum()),
        "net_correct_count": int(corrected.sum() - broken.sum()),
        "sample_accuracy_percent": float(candidate_correct.float().mean() * 100),
    }


@torch.no_grad()
def run(
    config_path: Path,
    claude_path: Path,
    claude_sha256: str,
    merge_path: Path,
    merge_sha256: str,
    output_json: Path,
    device_text: str,
) -> dict[str, object]:
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config)
    claude_path = claude_path.resolve()
    merge_path = merge_path.resolve()
    if sha256_file(claude_path) != claude_sha256:
        raise ValueError("AGCT oracle Claude SHA错误。")
    if sha256_file(merge_path) != merge_sha256:
        raise ValueError("AGCT oracle merge SHA错误。")
    device = torch.device(device_text)
    sentence = torch.load(paths["sentence_embeds"], map_location="cpu", weights_only=True)
    features = torch.load(paths["train_features"], map_location="cpu", weights_only=True)
    labels = torch.load(paths["train_labels"], map_location="cpu", weights_only=True).long()
    official = {
        name: torch.load(paths[name], map_location="cpu", weights_only=True)
        for name in OFFICIAL_KEYS
    }
    class_names_tensor = torch.load(
        Path(config["class_name_embeddings"]), map_location="cpu", weights_only=True
    ).to(device)
    sentence8 = torch.load(
        Path(config["eight_sentence_embeddings"]), map_location="cpu", weights_only=True
    ).to(device)
    claude = torch.load(claude_path, map_location="cpu", weights_only=True).to(device)
    merge = torch.load(merge_path, map_location="cpu", weights_only=True).to(device)
    seen_classes = torch.unique(labels, sorted=True)
    all_classes = torch.arange(200)
    unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
    checked_seen, checked_unseen = load_cub_split(
        paths["res101"], paths["att_splits"], labels,
        official["seen_labels"], official["unseen_labels"], "cpu"
    )
    if not torch.equal(checked_seen, seen_classes) or not torch.equal(
        checked_unseen, unseen_classes
    ):
        raise ValueError("AGCT oracle类别边界错误。")

    parent, sdrs = _load_main(
        config, sentence, labels, features,
        class_names_tensor, seen_classes, device
    )
    calibrator_payload = torch.load(
        Path(config["sebc_model"]), map_location="cpu", weights_only=False
    )
    calibrator = EpisodicBiasCalibration(
        float(calibrator_payload["config"]["max_gamma"])
    ).to(device)
    calibrator.load_state_dict(
        calibrator_payload["calibrator_state_dict"], strict=True
    )
    calibrator.eval()
    casr_payload = torch.load(
        Path(config["casr_model"]), map_location="cpu", weights_only=False
    )
    sdcr_payload = torch.load(
        Path(config["sdcr_model"]), map_location="cpu", weights_only=False
    )
    casr_weights = torch.softmax(
        casr_payload["aosr_state_dict"]["raw_sentence_weights"].float(), dim=0
    ).to(device)
    fixed_beta = float(sdcr_payload["fixed_beta"])
    sdcr = SentenceDropoutConservativeRouting(
        sentence8,
        class_names_tensor,
        casr_weights,
        fixed_beta,
        float(sdcr_payload["config"]["max_logit_residual"]),
        int(sdcr_payload["config"].get("drop_count", 1)),
    ).to(device)
    sdcr.load_state_dict(sdcr_payload["sdcr_state_dict"], strict=True)
    sdcr.eval()
    group_ids = taxonomic_suffix_group_ids(load_class_names(paths["att_splits"]))
    threshold, threshold_stats = derive_train_threshold(
        parent,
        sdrs,
        calibrator,
        sdcr,
        features,
        labels,
        seen_classes,
        group_ids,
        device,
        float(config["threshold_quantile"]),
    )
    gate_model = AmbiguityGatedPatchTieBreaker(
        sdcr.prototypes(use_dropout=False).detach(),
        fixed_beta,
        group_ids,
        threshold,
        float(config["margin_temperature"]),
        float(config["max_beta"]),
    ).to(device)
    names_n = F.normalize(class_names_tensor.float(), dim=-1)
    claude_n = F.normalize(claude.float(), dim=-1)
    merge_n = F.normalize(merge.float(), dim=-1)
    claude_orth = F.normalize(
        claude_n - (claude_n * names_n).sum(dim=-1, keepdim=True) * names_n,
        dim=-1,
    )
    merge_orth = F.normalize(
        merge_n - (merge_n * names_n).sum(dim=-1, keepdim=True) * names_n,
        dim=-1,
    )
    patch_text = orthogonal_local_text_residuals(sentence, class_names_tensor)
    print("precomputing source-oracle patch scores")
    patch_scores = _precompute_scores(config, patch_text, device)

    split_packages = {}

    def infer_split(split_name, split_features, split_labels, class_ids=None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = split_features.to(device).float()
        parent_logits = F.normalize(images, dim=-1) @ parent.prototypes().index_select(0, ids).T * parent.scale()
        parent_logits = sdrs(parent_logits, images, ids)
        seen_mask = torch.isin(ids.cpu(), seen_classes).to(device)
        parent_logits = calibrator(parent_logits, seen_mask)
        logits = parent_logits + gate_model.sdcr_beta * (
            F.normalize(images, dim=-1)
            @ gate_model.sdcr_prototypes.index_select(0, ids).T
        )
        gate, _, top_positions = gate_model.gate_values(logits, ids)
        hard_gate = gate >= 0.5
        top_global_ids = ids.index_select(0, top_positions.reshape(-1)).reshape_as(
            top_positions
        ).cpu()
        baseline = ids.index_select(0, logits.argmax(dim=1)).cpu()
        source_scores = {
            "claude": F.normalize(images, dim=-1)
            @ claude_orth.index_select(0, ids).T,
            "merge": F.normalize(images, dim=-1)
            @ merge_orth.index_select(0, ids).T,
            "patch": patch_scores[
                "unseen" if split_name == "unseen_zsl" else split_name
            ].to(device).float().index_select(1, ids),
        }
        predictions = {"baseline": baseline}
        transitions = {}
        for source_name, scores in source_scores.items():
            top_scores = scores.gather(1, top_positions).cpu()
            for orientation, choose_max in (("max", True), ("min", False)):
                strategy = f"{source_name}_{orientation}"
                predictions[strategy] = apply_hard_tie_break(
                    baseline,
                    top_global_ids,
                    top_scores,
                    hard_gate.cpu(),
                    choose_max,
                )
        labels_cpu = split_labels.cpu()
        contains_true = top_global_ids.eq(labels_cpu.unsqueeze(1)).any(dim=1)
        oracle = torch.where(
            hard_gate.cpu() & contains_true,
            labels_cpu,
            baseline,
        )
        predictions["oracle_top2"] = oracle
        for strategy, candidate in predictions.items():
            transitions[strategy] = transition_stats(
                labels_cpu,
                baseline,
                candidate,
                hard_gate.cpu(),
                top_global_ids,
            )
        split_packages[split_name] = {
            "predictions": predictions,
            "labels": labels_cpu,
            "transitions": transitions,
            "gate_mean": float(gate.mean()),
            "hard_gate_rate": float(hard_gate.float().mean()),
        }

    infer_split(
        "seen", official["seen_features"], official["seen_labels"]
    )
    infer_split(
        "unseen", official["unseen_features"], official["unseen_labels"]
    )
    infer_split(
        "unseen_zsl",
        official["unseen_features"],
        official["unseen_labels"],
        unseen_classes,
    )
    strategy_metrics = {}
    for strategy in STRATEGIES:
        seen_accuracy = h1.per_class_accuracy(
            split_packages["seen"]["labels"],
            split_packages["seen"]["predictions"][strategy],
            seen_classes,
        )
        unseen_accuracy = h1.per_class_accuracy(
            split_packages["unseen"]["labels"],
            split_packages["unseen"]["predictions"][strategy],
            unseen_classes,
        )
        zsl_accuracy = h1.per_class_accuracy(
            split_packages["unseen_zsl"]["labels"],
            split_packages["unseen_zsl"]["predictions"][strategy],
            unseen_classes,
        )
        strategy_metrics[strategy] = {
            "U": unseen_accuracy * 100,
            "S": seen_accuracy * 100,
            "H": 2 * seen_accuracy * unseen_accuracy / (seen_accuracy + unseen_accuracy) * 100,
            "ZS": zsl_accuracy * 100,
        }
    payload = {
        "source_config": str(config_path),
        "source_config_sha256": config_sha,
        "threshold": threshold,
        "threshold_stats": threshold_stats,
        "test_used_for_analysis": True,
        "unseen_images_used_for_gradient": False,
        "strategy_metrics_percent": strategy_metrics,
        "split_diagnostics": {
            split: {
                "gate_mean": package["gate_mean"],
                "hard_gate_rate": package["hard_gate_rate"],
                "transitions": package["transitions"],
            }
            for split, package in split_packages.items()
        },
        "source_sha256": {
            "claude": claude_sha256,
            "merge": merge_sha256,
            "patch": config["patch_sha256"],
        },
    }
    output_json = output_json.resolve()
    if output_json.exists():
        raise FileExistsError(f"AGCT source-oracle输出已存在：{output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_json, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--claude", type=Path, required=True)
    parser.add_argument("--claude-sha256", required=True)
    parser.add_argument("--merge", type=Path, required=True)
    parser.add_argument("--merge-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(
        args.config,
        args.claude,
        args.claude_sha256,
        args.merge,
        args.merge_sha256,
        args.output_json,
        args.device,
    )
    print(json.dumps(result["strategy_metrics_percent"], indent=2))


if __name__ == "__main__":
    main()
