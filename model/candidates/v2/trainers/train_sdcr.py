from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
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
COMMON_CONFIG_KEYS = {
    "schema_version", "experiment_id", "idea_id", "framework_id", "dataset",
    "evaluation_protocol", "test_used_for_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "text_cache_provenance_complete", "base_model",
    "base_model_sha256", "sdrs_model", "sdrs_model_sha256", "sebc_model",
    "sebc_model_sha256", "casr_model", "casr_model_sha256",
    "parent_metrics_percent", "class_name_embeddings", "class_name_embeddings_sha256",
    "eight_sentence_embeddings", "eight_sentence_embeddings_sha256",
    "max_logit_residual", "kl_weight", "device", "random_seed", "batch_size",
    "epochs", "niters", "report_interval", "optimizer", "learning_rate",
    "weight_decay", "inputs", "expected_sha256", "class_order_sha256",
}


def sample_importance_mask(
    sentence_weights: torch.Tensor, generator: torch.Generator
) -> int:
    """按当前完整句权重采样一句；采样本身不参与梯度。"""
    probabilities = sentence_weights.detach().float().cpu()
    if tuple(probabilities.shape) != (8,):
        raise ValueError("IADR句权重必须是[8]。")
    if not torch.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError("IADR句权重必须有限且非负。")
    total = probabilities.sum()
    if float(total) <= 0:
        raise ValueError("IADR句权重之和必须大于0。")
    probabilities = probabilities / total
    return int(
        torch.multinomial(
            probabilities, 1, replacement=False, generator=generator
        ).item()
    )


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    schema = config.get("schema_version") if isinstance(config, dict) else None
    if schema == "gzsl-paper.sdcr.v2":
        expected_keys = COMMON_CONFIG_KEYS | {"drop_count"}
    elif schema == "gzsl-paper.sdcc.v1":
        expected_keys = COMMON_CONFIG_KEYS | {
            "consistency_weight", "distill_temperature"
        }
    elif schema == "gzsl-paper.wsdr.v1":
        expected_keys = COMMON_CONFIG_KEYS | {"candidate_masks"}
    elif schema == "gzsl-paper.iadr.v1":
        expected_keys = COMMON_CONFIG_KEYS | {"sampling_strategy"}
    else:
        expected_keys = COMMON_CONFIG_KEYS
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"SDCR配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    identity_by_schema = {
        "gzsl-paper.sdcr.v1": ("V2-INNOVATION-041", "IDEA-075"),
        "gzsl-paper.sdcr.v2": ("V2-INNOVATION-041", "IDEA-075"),
        "gzsl-paper.sdcc.v1": ("V2-INNOVATION-042", "IDEA-076"),
        "gzsl-paper.wsdr.v1": ("V2-INNOVATION-043", "IDEA-077"),
        "gzsl-paper.iadr.v1": ("V2-INNOVATION-044", "IDEA-078"),
    }
    identity = identity_by_schema.get(config["schema_version"])
    if (
        identity is None
        or config["experiment_id"] != identity[0]
        or config["idea_id"] != identity[1]
    ):
        raise ValueError("SDCR身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("SDCR协议边界错误。")
    if config["text_cache_provenance_complete"] is not False:
        raise ValueError("SDCR文本cache provenance未完整。")
    if (
        float(config["max_logit_residual"]) != 0.5
        or float(config["kl_weight"]) != 0.01
        or int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
        or config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
    ):
        raise ValueError("SDCR训练参数错误。")
    expected_drop = 2 if schema == "gzsl-paper.sdcr.v2" else 1
    if int(config.get("drop_count", 1)) != expected_drop:
        raise ValueError(f"SDCR drop_count必须为{expected_drop}。")
    if schema == "gzsl-paper.sdcc.v1" and (
        float(config["consistency_weight"]) != 0.1
        or float(config["distill_temperature"]) != 1.0
    ):
        raise ValueError("SDCC一致性权重/温度错误。")
    if schema == "gzsl-paper.wsdr.v1" and int(config["candidate_masks"]) != 2:
        raise ValueError("WSDR candidate_masks必须为2。")
    if (
        schema == "gzsl-paper.iadr.v1"
        and config["sampling_strategy"] != "current_weight_proportional"
    ):
        raise ValueError("IADR必须按当前完整句权重采样mask。")
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
        "base_model", "sdrs_model", "sebc_model", "casr_model",
        "class_name_embeddings", "eight_sentence_embeddings",
    ):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"SDCR {key} SHA错误。")
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
        new_sentences = torch.load(
            Path(config["eight_sentence_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        seen_classes = torch.unique(labels, sorted=True)
        all_classes = torch.arange(200)
        unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seen_classes) or not torch.equal(checked_unseen, unseen_classes):
            raise ValueError("SDCR CUB类别边界错误。")

        parent, sdrs = _load_main(
            config, sentence, labels, features, class_names, seen_classes, device
        )
        sebc_payload = torch.load(
            Path(config["sebc_model"]), map_location="cpu", weights_only=False
        )
        calibrator = EpisodicBiasCalibration(
            float(sebc_payload["config"]["max_gamma"])
        ).to(device)
        calibrator.load_state_dict(
            sebc_payload["calibrator_state_dict"], strict=True
        )
        calibrator.eval()
        for parameter in calibrator.parameters():
            parameter.requires_grad_(False)
        casr_payload = torch.load(
            Path(config["casr_model"]), map_location="cpu", weights_only=False
        )
        fixed_beta = float(casr_payload["fixed_beta"])
        base_weights = torch.softmax(
            casr_payload["aosr_state_dict"]["raw_sentence_weights"].float(), dim=0
        ).to(device)
        model = SentenceDropoutConservativeRouting(
            new_sentences, class_names, base_weights, fixed_beta,
            float(config["max_logit_residual"]), int(config.get("drop_count", 1)),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
        )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generator = torch.Generator().manual_seed(seed)
        prototypes = parent.prototypes().detach()
        scale = parent.scale().detach()
        class_ids = seen_classes.to(device)
        seen_mask = torch.ones(150, dtype=torch.bool, device=device)
        mask_counts = [0] * 8

        model.eval()
        best_metrics = evaluate(
            parent, sdrs, calibrator, model, official,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"SDCR初始态未复现CASR：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = -1
        history = []
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "sdcr_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "fixed_beta": fixed_beta,
                "config": config,
                "code_commit": commit,
                "reproducibility": reproducibility,
            },
        )

        for iteration in range(int(config["niters"])):
            model.train()
            indices = random_batch_indices(
                labels.numel(), int(config["batch_size"]), generator
            )
            images = features.index_select(0, indices).to(device).float()
            targets = mapping[labels.index_select(0, indices)].to(device)
            normalized = F.normalize(images, dim=-1)
            base = normalized @ prototypes.index_select(0, class_ids).T * scale
            parent_logits = sdrs(base, images, class_ids)
            parent_logits = calibrator(parent_logits, seen_mask)
            if config["schema_version"] == "gzsl-paper.wsdr.v1":
                candidate_roles = torch.randperm(8, device=device)[:2]
                candidate_losses = []
                for role_tensor in candidate_roles:
                    role = int(role_tensor.item())
                    candidate_logits = model(
                        parent_logits, images, class_ids, mask_roles=[role]
                    )
                    candidate_losses.append(
                        F.cross_entropy(candidate_logits, targets)
                    )
                    mask_counts[role] += 1
                ce_loss = torch.stack(candidate_losses).max()
                logits = candidate_logits
            elif config["schema_version"] == "gzsl-paper.iadr.v1":
                role = sample_importance_mask(
                    model.full_sentence_weights(), generator
                )
                logits = model(
                    parent_logits, images, class_ids, mask_roles=[role]
                )
                mask_counts[role] += 1
                ce_loss = F.cross_entropy(logits, targets)
            else:
                logits = model(parent_logits, images, class_ids)
                for role in model.last_masked_roles:
                    mask_counts[role] += 1
                ce_loss = F.cross_entropy(logits, targets)
            kl_loss = model.kl_to_base()
            if config["schema_version"] == "gzsl-paper.sdcc.v1":
                with torch.no_grad():
                    model.eval()
                    teacher_logits = model(
                        parent_logits.detach(), images, class_ids
                    )
                    model.train()
                temperature = float(config["distill_temperature"])
                consistency_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
            else:
                consistency_loss = ce_loss.new_zeros(())
            loss = (
                ce_loss
                + float(config["kl_weight"]) * kl_loss
                + float(config.get("consistency_weight", 0.0)) * consistency_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(model)
            optimizer.step()

            if iteration % int(config["report_interval"]) == 0:
                model.eval()
                metrics = evaluate(
                    parent, sdrs, calibrator, model, official,
                    seen_classes, unseen_classes, device
                )
                stats = model.weight_stats()
                history.append(
                    {
                        "iteration": iteration,
                        "loss": float(loss.detach()),
                        "ce_loss": float(ce_loss.detach()),
                        "kl_loss": float(kl_loss.detach()),
                        "consistency_loss": float(consistency_loss.detach()),
                        "official_metrics_percent": metrics,
                        "weight_stats": stats,
                        "mask_counts": list(mask_counts),
                    }
                )
                if metrics["H"] > best_h:
                    best_h = metrics["H"]
                    best_metrics = metrics
                    best_state = copy.deepcopy(model.state_dict())
                    best_iteration = iteration
                    atomic_torch_save(
                        output_dir / "model_best.pth",
                        {
                            "sdcr_state_dict": best_state,
                            "best_metrics_percent": best_metrics,
                            "selected_iteration": best_iteration,
                            "fixed_beta": fixed_beta,
                            "config": config,
                            "code_commit": commit,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f} weight_std={stats['std']:.6f}"
                )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "sdcr_state_dict": copy.deepcopy(model.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "history": history,
                "mask_counts": mask_counts,
                "fixed_beta": fixed_beta,
                "config": config,
                "code_commit": commit,
            },
        )
        model.load_state_dict(best_state, strict=True)
        model.eval()
        best_stats = model.weight_stats()
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "casr_model": config["casr_model_sha256"],
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
            "parent_metrics_percent": expected_parent,
            "best_metrics_percent": best_metrics,
            "selected_iteration": best_iteration,
            "fixed_beta": fixed_beta,
            "sentence_weight_stats": best_stats,
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
