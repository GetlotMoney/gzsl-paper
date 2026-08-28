from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.ccpe import class_conditioned_patch_scores
from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
from model.candidates.v2.modules.lpsr import orthogonal_local_text_residuals
from model.candidates.v2.modules.sdcr import SentenceDropoutConservativeRouting
from model.candidates.v2.trainers.train_chen_style import (
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.candidates.v2.trainers.train_clre import evaluate
from model.candidates.v2.trainers.train_sebc import _load_main
from model.frameworks.v2 import train as h1
from tools.cub_data import load_cub_split
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
    "class_name_embeddings_sha256", "train_patch", "train_patch_sha256",
    "patch_top_k", "patch_chunk_size", "reliability_amplitude", "sdcr_kl_weight",
    "device", "random_seed", "batch_size", "epochs", "niters", "report_interval",
    "optimizer", "learning_rate", "weight_decay", "inputs", "expected_sha256",
    "class_order_sha256",
}


def patch_reliability_weights(
    patch_scores: torch.Tensor,
    labels: torch.Tensor,
    amplitude: float,
) -> torch.Tensor:
    if patch_scores.ndim != 2 or patch_scores.shape[1] != 200:
        raise ValueError("PGSD patch分数必须是[N,200]。")
    if labels.ndim != 1 or labels.shape[0] != patch_scores.shape[0]:
        raise ValueError("PGSD标签必须与patch分数逐样本对应。")
    true_scores = patch_scores.gather(1, labels.long().unsqueeze(1)).squeeze(1)
    center = patch_scores.mean(dim=1)
    scale = patch_scores.std(dim=1, unbiased=False).clamp_min(1e-6)
    confidence = torch.tanh((true_scores - center) / scale)
    return 1.0 + float(amplitude) * confidence


def centered_patch_reliability_weights(
    patch_scores: torch.Tensor,
    labels: torch.Tensor,
    amplitude: float,
) -> torch.Tensor:
    raw = patch_reliability_weights(patch_scores, labels, 1.0) - 1.0
    centered = raw - raw.mean()
    normalized = centered / centered.abs().max().clamp_min(1e-6)
    return 1.0 + float(amplitude) * normalized


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    schema = config.get("schema_version") if isinstance(config, dict) else None
    expected_keys = (
        CONFIG_KEYS | {"center_weights"}
        if schema == "gzsl-paper.cpgsd.v1"
        else CONFIG_KEYS
    )
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"PGSD配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    identity = {
        "gzsl-paper.pgsd.v1": ("V2-INNOVATION-053", "IDEA-087"),
        "gzsl-paper.cpgsd.v1": ("V2-INNOVATION-054", "IDEA-088"),
    }.get(schema)
    if identity is None or (
        config["experiment_id"], config["idea_id"]
    ) != identity:
        raise ValueError("PGSD身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("PGSD协议边界错误。")
    if config["feature_provenance_complete"] is not False:
        raise ValueError("PGSD patch provenance未完整。")
    if (
        int(config["patch_top_k"]) != 2
        or int(config["patch_chunk_size"]) != 16
        or float(config["reliability_amplitude"]) != 0.25
        or float(config["sdcr_kl_weight"]) != 0.01
        or int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
        or config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.001
        or float(config["weight_decay"]) != 0.0
    ):
        raise ValueError("PGSD patch或训练参数错误。")
    if schema == "gzsl-paper.cpgsd.v1" and config["center_weights"] is not True:
        raise ValueError("CPGSD必须中心化样本权重。")
    return config, sha256_file(path)


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
        "class_name_embeddings",
    ):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"PGSD {key} SHA错误。")
    train_patch_path = h1.repo_path(config["train_patch"])
    if sha256_file(train_patch_path) != config["train_patch_sha256"]:
        raise ValueError("PGSD train patch SHA错误。")
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
        class_names = torch.load(
            Path(config["class_name_embeddings"]), map_location="cpu", weights_only=True
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
            raise ValueError("PGSD CUB类别边界错误。")

        parent, sdrs = _load_main(
            config, sentence, labels, features, class_names, seen_classes, device
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
        eight_sentences = torch.load(
            Path(sdcr_payload["config"]["eight_sentence_embeddings"]),
            map_location="cpu",
            weights_only=True,
        ).to(device)
        casr_weights = torch.softmax(
            casr_payload["aosr_state_dict"]["raw_sentence_weights"].float(), dim=0
        ).to(device)
        fixed_beta = float(sdcr_payload["fixed_beta"])
        sdcr = SentenceDropoutConservativeRouting(
            eight_sentences,
            class_names,
            casr_weights,
            fixed_beta,
            float(sdcr_payload["config"]["max_logit_residual"]),
            int(sdcr_payload["config"].get("drop_count", 1)),
        ).to(device)
        sdcr.load_state_dict(sdcr_payload["sdcr_state_dict"], strict=True)
        for parameter in sdcr.parameters():
            parameter.requires_grad_(False)
        sdcr.raw_weight_residual.requires_grad_(True)

        print("precomputing train-only patch reliability")
        train_patches = torch.load(
            train_patch_path, map_location="cpu", weights_only=True
        )
        text_residual = orthogonal_local_text_residuals(sentence, class_names)
        patch_scores = class_conditioned_patch_scores(
            train_patches,
            text_residual,
            int(config["patch_top_k"]),
            device,
            chunk_size=int(config["patch_chunk_size"]),
        )
        del train_patches
        if config["schema_version"] == "gzsl-paper.cpgsd.v1":
            sample_weights = centered_patch_reliability_weights(
                patch_scores, labels, float(config["reliability_amplitude"])
            )
        else:
            sample_weights = patch_reliability_weights(
                patch_scores, labels, float(config["reliability_amplitude"])
            )
        weight_stats = {
            "mean": float(sample_weights.mean()),
            "std": float(sample_weights.std(unbiased=False)),
            "min": float(sample_weights.min()),
            "max": float(sample_weights.max()),
        }

        optimizer = torch.optim.Adam(
            [sdcr.raw_weight_residual],
            lr=float(config["learning_rate"]),
            weight_decay=0.0,
        )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generator = torch.Generator().manual_seed(seed)
        parent_prototypes = parent.prototypes().detach()
        scale = parent.scale().detach()
        class_ids = seen_classes.to(device)
        seen_mask = torch.ones(150, dtype=torch.bool, device=device)
        mask_counts = [0] * 8

        sdcr.eval()
        best_metrics = evaluate(
            parent, sdrs, calibrator, sdcr, official,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"PGSD初始态未复现SDCR：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(sdcr.state_dict())
        best_iteration = -1
        history = []
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "sdcr_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "sample_weight_stats": weight_stats,
                "config": config,
                "code_commit": commit,
                "reproducibility": reproducibility,
            },
        )

        for iteration in range(int(config["niters"])):
            sdcr.train()
            indices = random_batch_indices(
                labels.numel(), int(config["batch_size"]), generator
            )
            images = features.index_select(0, indices).to(device).float()
            targets = mapping[labels.index_select(0, indices)].to(device)
            batch_weights = sample_weights.index_select(0, indices).to(device)
            normalized = F.normalize(images, dim=-1)
            logits = normalized @ parent_prototypes.index_select(0, class_ids).T * scale
            logits = sdrs(logits, images, class_ids)
            logits = calibrator(logits, seen_mask)
            logits = sdcr(logits, images, class_ids)
            for role in sdcr.last_masked_roles:
                mask_counts[role] += 1
            per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
            ce_loss = (per_sample_ce * batch_weights).mean()
            kl_loss = sdcr.kl_to_base()
            loss = ce_loss + float(config["sdcr_kl_weight"]) * kl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(sdcr)
            optimizer.step()

            if iteration % int(config["report_interval"]) == 0:
                sdcr.eval()
                metrics = evaluate(
                    parent, sdrs, calibrator, sdcr, official,
                    seen_classes, unseen_classes, device
                )
                sentence_stats = sdcr.weight_stats()
                history.append(
                    {
                        "iteration": iteration,
                        "loss": float(loss.detach()),
                        "ce_loss": float(ce_loss.detach()),
                        "kl_loss": float(kl_loss.detach()),
                        "official_metrics_percent": metrics,
                        "sentence_weight_stats": sentence_stats,
                        "mask_counts": list(mask_counts),
                    }
                )
                if metrics["H"] > best_h:
                    best_h = metrics["H"]
                    best_metrics = metrics
                    best_state = copy.deepcopy(sdcr.state_dict())
                    best_iteration = iteration
                    atomic_torch_save(
                        output_dir / "model_best.pth",
                        {
                            "sdcr_state_dict": best_state,
                            "best_metrics_percent": best_metrics,
                            "selected_iteration": best_iteration,
                            "sample_weight_stats": weight_stats,
                            "config": config,
                            "code_commit": commit,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f} weight_std={sentence_stats['std']:.6f}"
                )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "sdcr_state_dict": copy.deepcopy(sdcr.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "history": history,
                "mask_counts": mask_counts,
                "sample_weight_stats": weight_stats,
                "config": config,
                "code_commit": commit,
            },
        )
        sdcr.load_state_dict(best_state, strict=True)
        best_sentence_stats = sdcr.weight_stats()
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                "train_patch": config["train_patch_sha256"],
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "casr_model": config["casr_model_sha256"],
                "sdcr_model": config["sdcr_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
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
            "sample_weight_stats": weight_stats,
            "sentence_weight_stats": best_sentence_stats,
            "mask_counts": mask_counts,
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
