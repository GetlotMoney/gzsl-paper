from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.aosr import AdaptiveOrthogonalSentenceResidual
from model.innovations.ebc import EpisodicBiasCalibration
from model.innovations.train_chen_style import (
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.innovations.train_clre import evaluate
from model.innovations.train_sebc import _load_main
from model.tg_vpr_h1 import train as h1
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
    "sebc_model_sha256", "oesr_model", "oesr_model_sha256",
    "parent_metrics_percent", "class_name_embeddings", "class_name_embeddings_sha256",
    "eight_sentence_embeddings", "eight_sentence_embeddings_sha256", "device",
    "random_seed", "batch_size", "epochs", "niters", "report_interval", "optimizer",
    "learning_rate", "weight_decay", "inputs", "expected_sha256", "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    schema = config.get("schema_version") if isinstance(config, dict) else None
    expected_keys = (
        COMMON_CONFIG_KEYS | {"kl_weight"}
        if schema in ("gzsl-paper.casr.v1", "gzsl-paper.casr.v2")
        else COMMON_CONFIG_KEYS
    )
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"AOSR配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    identity_by_schema = {
        "gzsl-paper.aosr.v1": ("V2-INNOVATION-037", "IDEA-071"),
        "gzsl-paper.casr.v1": ("V2-INNOVATION-038", "IDEA-072"),
        "gzsl-paper.casr.v2": ("V2-INNOVATION-038", "IDEA-072"),
    }
    identity = identity_by_schema.get(config["schema_version"])
    if (
        identity is None
        or config["experiment_id"] != identity[0]
        or config["idea_id"] != identity[1]
    ):
        raise ValueError("AOSR身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("AOSR协议边界错误。")
    if config["text_cache_provenance_complete"] is not False:
        raise ValueError("AOSR文本cache provenance未完整。")
    if (
        int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
        or config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
    ):
        raise ValueError("AOSR训练参数错误。")
    expected_kl = 0.01 if schema == "gzsl-paper.casr.v2" else 0.1
    if schema in ("gzsl-paper.casr.v1", "gzsl-paper.casr.v2") and float(
        config["kl_weight"]
    ) != expected_kl:
        raise ValueError(f"CASR kl_weight必须为{expected_kl}。")
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
        "base_model", "sdrs_model", "sebc_model", "oesr_model",
        "class_name_embeddings", "eight_sentence_embeddings",
    ):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"AOSR {key} SHA错误。")
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
            raise ValueError("AOSR CUB类别边界错误。")

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
        oesr_payload = torch.load(
            Path(config["oesr_model"]), map_location="cpu", weights_only=False
        )
        fixed_beta = float(
            float(oesr_payload["config"]["max_beta"])
            * torch.tanh(oesr_payload["clre_state_dict"]["raw_beta"].float())
        )
        model = AdaptiveOrthogonalSentenceResidual(
            new_sentences, class_names, fixed_beta
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

        best_metrics = evaluate(
            parent, sdrs, calibrator, model, official,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"AOSR等权初始态未复现OESR：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = -1
        history = []
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "aosr_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "fixed_beta": fixed_beta,
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
            targets = mapping[labels.index_select(0, indices)].to(device)
            base = F.normalize(images, dim=-1) @ prototypes.index_select(0, class_ids).T * scale
            logits = sdrs(base, images, class_ids)
            logits = calibrator(logits, seen_mask)
            logits = model(logits, images, class_ids)
            ce_loss = F.cross_entropy(logits, targets)
            if config["schema_version"] in (
                "gzsl-paper.casr.v1", "gzsl-paper.casr.v2"
            ):
                weights = model.sentence_weights()
                kl_loss = (weights * torch.log(weights * 8.0)).sum()
                loss = ce_loss + float(config["kl_weight"]) * kl_loss
            else:
                kl_loss = ce_loss.new_zeros(())
                loss = ce_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(model)
            optimizer.step()

            if iteration % int(config["report_interval"]) == 0:
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
                        "official_metrics_percent": metrics,
                        "weight_stats": stats,
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
                            "aosr_state_dict": best_state,
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
                "aosr_state_dict": copy.deepcopy(model.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "history": history,
                "fixed_beta": fixed_beta,
                "config": config,
                "code_commit": commit,
            },
        )
        model.load_state_dict(best_state, strict=True)
        best_stats = model.weight_stats()
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "oesr_model": config["oesr_model_sha256"],
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
