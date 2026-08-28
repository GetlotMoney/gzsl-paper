from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.agpt import AmbiguityGatedPatchTieBreaker
from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
from model.candidates.v2.modules.lpsr import orthogonal_local_text_residuals
from model.candidates.v2.modules.sdcr import SentenceDropoutConservativeRouting
from model.candidates.v2.modules.tigr import taxonomic_suffix_group_ids
from model.candidates.v2.trainers.train_agct import derive_train_threshold
from model.candidates.v2.trainers.train_ccpe import _precompute_scores
from model.candidates.v2.trainers.train_chen_style import (
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.candidates.v2.trainers.train_sebc import _load_main
from model.frameworks.v2 import train as h1
from tools.cub_data import load_cub_split
from tools.diagnose_sdcr_errors import load_class_names
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
    require_finite_gradients,
)
from tools.runtime import sha256_file

EVALUATION_PROTOCOL = "chen_shiming_code_aligned_test_selected_gzsl"
CONFIG_KEYS = {
    "schema_version", "experiment_id", "idea_id", "framework_id", "dataset",
    "evaluation_protocol", "test_used_for_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "feature_provenance_complete", "base_model",
    "base_model_sha256", "sdrs_model", "sdrs_model_sha256", "sebc_model",
    "sebc_model_sha256", "casr_model", "casr_model_sha256", "sdcr_model",
    "sdcr_model_sha256", "parent_metrics_percent", "class_name_embeddings",
    "class_name_embeddings_sha256", "eight_sentence_embeddings",
    "eight_sentence_embeddings_sha256", "patch_inputs", "patch_sha256",
    "patch_top_k", "patch_chunk_size", "group_rule", "threshold_source",
    "threshold_quantile", "margin_temperature", "max_beta", "device",
    "random_seed", "batch_size", "epochs", "niters", "report_interval",
    "optimizer", "learning_rate", "weight_decay", "inputs", "expected_sha256",
    "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"AGPT配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if (
        config["schema_version"] != "gzsl-paper.agpt.v1"
        or config["experiment_id"] != "V2-INNOVATION-061"
        or config["idea_id"] != "IDEA-095"
    ):
        raise ValueError("AGPT身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("AGPT协议边界错误。")
    if config["feature_provenance_complete"] is not False:
        raise ValueError("AGPT patch provenance未完整。")
    if set(config["patch_inputs"]) != {"train", "seen", "unseen"} or set(
        config["patch_sha256"]
    ) != {"train", "seen", "unseen"}:
        raise ValueError("AGPT patch输入/SHA不完整。")
    if (
        int(config["patch_top_k"]) != 2
        or int(config["patch_chunk_size"]) != 16
        or config["group_rule"] != "class_name_last_token_min2"
        or config["threshold_source"] != "train_wrong_same_group_margin"
        or float(config["threshold_quantile"]) != 0.25
        or float(config["margin_temperature"]) != 0.1
        or float(config["max_beta"]) != 5.0
        or int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
        or config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.001
        or float(config["weight_decay"]) != 0.0
    ):
        raise ValueError("AGPT训练参数错误。")
    return config, sha256_file(path)


@torch.no_grad()
def evaluate(
    parent,
    sdrs,
    calibrator,
    model,
    tensors,
    scores,
    seen_classes,
    unseen_classes,
    device,
):
    gate_values = []

    def predict(features, patch_scores, class_ids=None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = features.to(device).float()
        parent_logits = F.normalize(images, dim=-1) @ parent.prototypes().index_select(0, ids).T * parent.scale()
        parent_logits = sdrs(parent_logits, images, ids)
        seen_mask = torch.isin(ids.cpu(), seen_classes).to(device)
        parent_logits = calibrator(parent_logits, seen_mask)
        base_logits = parent_logits + model.sdcr_beta * (
            F.normalize(images, dim=-1) @ model.sdcr_prototypes.index_select(0, ids).T
        )
        gate, _, _ = model.gate_values(base_logits, ids)
        gate_values.append(gate.cpu())
        predictions = model(
            parent_logits, images, patch_scores.to(device), ids
        ).argmax(1).cpu()
        return predictions if class_ids is None else class_ids[predictions]

    seen_predictions = predict(
        tensors["seen_features"], scores["seen"]
    )
    unseen_predictions = predict(
        tensors["unseen_features"], scores["unseen"]
    )
    zsl_predictions = predict(
        tensors["unseen_features"], scores["unseen"], unseen_classes
    )
    seen = h1.per_class_accuracy(tensors["seen_labels"], seen_predictions, seen_classes)
    unseen = h1.per_class_accuracy(
        tensors["unseen_labels"], unseen_predictions, unseen_classes
    )
    zsl = h1.per_class_accuracy(
        tensors["unseen_labels"], zsl_predictions, unseen_classes
    )
    return (
        {
            "U": unseen * 100,
            "S": seen * 100,
            "H": 2 * seen * unseen / (seen + unseen) * 100,
            "ZS": zsl * 100,
        },
        {
            "seen_gate_mean": float(gate_values[0].mean()),
            "unseen_gzsl_gate_mean": float(gate_values[1].mean()),
            "unseen_zsl_gate_mean": float(gate_values[2].mean()),
        },
    )


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    commit = current_code_commit()
    if commit != expected_commit:
        raise ValueError("expected-commit不一致。")
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config)
    input_sha = verify_inputs(config, paths)
    for key in (
        "base_model", "sdrs_model", "sebc_model", "casr_model", "sdcr_model",
        "class_name_embeddings", "eight_sentence_embeddings",
    ):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"AGPT {key} SHA错误。")
    for split, path_text in config["patch_inputs"].items():
        if sha256_file(h1.repo_path(path_text)) != config["patch_sha256"][split]:
            raise ValueError(f"AGPT {split} patch SHA错误。")
    device = torch.device(config["device"])
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    log = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    old_stdout = sys.stdout
    sys.stdout = h1.TeeStream(sys.stdout, log)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(
            seed, strict_determinism=True, deterministic_warn_only=False
        )
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
            raise ValueError("AGPT CUB类别边界错误。")

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
        for parameter in calibrator.parameters():
            parameter.requires_grad_(False)
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
        for parameter in sdcr.parameters():
            parameter.requires_grad_(False)
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
        text_residual = orthogonal_local_text_residuals(
            sentence, class_names_tensor
        )
        print("precomputing AGPT top2 patch scores")
        scores = _precompute_scores(config, text_residual, device)
        model = AmbiguityGatedPatchTieBreaker(
            sdcr.prototypes(use_dropout=False).detach(),
            fixed_beta,
            group_ids,
            threshold,
            float(config["margin_temperature"]),
            float(config["max_beta"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
        )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generator = torch.Generator().manual_seed(seed)
        class_ids = seen_classes.to(device)

        best_metrics, best_gate_stats = evaluate(
            parent, sdrs, calibrator, model, official, scores,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"AGPT初始态未复现SDCR：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = -1
        history = []
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "agpt_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "threshold_stats": threshold_stats,
                "config": config,
                "code_commit": commit,
                "reproducibility": reproducibility,
            },
        )

        for iteration in range(int(config["niters"])):
            indices = random_batch_indices(
                labels.numel(), int(config["batch_size"]), generator
            )
            images = features.index_select(0, indices).to(device).float()
            patch_scores = scores["train"].index_select(0, indices).to(device)
            targets = mapping[labels.index_select(0, indices)].to(device)
            parent_logits = F.normalize(images, dim=-1) @ parent.prototypes().index_select(0, class_ids).T * parent.scale()
            parent_logits = sdrs(parent_logits, images, class_ids)
            parent_logits = calibrator(
                parent_logits,
                torch.ones(150, dtype=torch.bool, device=device),
            )
            logits = model(
                parent_logits, images, patch_scores, class_ids
            )
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(model)
            optimizer.step()

            if iteration % int(config["report_interval"]) == 0:
                metrics, gate_stats = evaluate(
                    parent, sdrs, calibrator, model, official, scores,
                    seen_classes, unseen_classes, device
                )
                stats = {**model.stats(), **gate_stats}
                history.append(
                    {
                        "iteration": iteration,
                        "loss": float(loss.detach()),
                        "official_metrics_percent": metrics,
                        "agpt_stats": stats,
                    }
                )
                if metrics["H"] > best_h:
                    best_h = metrics["H"]
                    best_metrics = metrics
                    best_gate_stats = gate_stats
                    best_state = copy.deepcopy(model.state_dict())
                    best_iteration = iteration
                    atomic_torch_save(
                        output_dir / "model_best.pth",
                        {
                            "agpt_state_dict": best_state,
                            "best_metrics_percent": best_metrics,
                            "selected_iteration": best_iteration,
                            "threshold_stats": threshold_stats,
                            "config": config,
                            "code_commit": commit,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f} beta={stats['patch_beta']:.6f} "
                    f"gate_U={stats['unseen_gzsl_gate_mean']:.4f}"
                )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "agpt_state_dict": copy.deepcopy(model.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "history": history,
                "threshold_stats": threshold_stats,
                "config": config,
                "code_commit": commit,
            },
        )
        model.load_state_dict(best_state, strict=True)
        best_stats = {**model.stats(), **best_gate_stats}
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                "patch_files": config["patch_sha256"],
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "casr_model": config["casr_model_sha256"],
                "sdcr_model": config["sdcr_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
                "eight_sentence_embeddings": config["eight_sentence_embeddings_sha256"],
            },
        )
        metrics = {
            "experiment_id": config["experiment_id"],
            "idea_id": config["idea_id"],
            "run_id": run_id,
            "code_commit": commit,
            "config_sha256": config_sha,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "feature_provenance_complete": False,
            "parent_metrics_percent": expected_parent,
            "best_metrics_percent": best_metrics,
            "selected_iteration": best_iteration,
            "threshold_stats": threshold_stats,
            "agpt_stats": best_stats,
            "official_test_evaluation_count": len(history) + 1,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", metrics)
        print(metrics)
        return metrics
    finally:
        sys.stdout.flush()
        sys.stdout = old_stdout
        log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.run_id)


if __name__ == "__main__":
    main()
