from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.orer import GammaResidualCalibration
from model.innovations.train_chen_class_exclusive import balanced_fold_batch
from model.innovations.train_chen_style import OFFICIAL_KEYS, resolve_paths, verify_inputs
from model.innovations.train_sebc import _load_folds, _load_main
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
CONFIG_KEYS = {
    "schema_version", "experiment_id", "idea_id", "framework_id", "dataset",
    "evaluation_protocol", "test_used_for_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "text_cache_provenance_complete", "base_model",
    "base_model_sha256", "sdrs_model", "sdrs_model_sha256", "sebc_model",
    "sebc_model_sha256", "oclr_model", "oclr_model_sha256",
    "parent_metrics_percent", "class_name_embeddings", "class_name_embeddings_sha256",
    "claude_embeddings", "claude_embeddings_sha256", "fold_model_dir",
    "fold_model_sha256", "max_gamma_residual", "device", "random_seed", "epochs",
    "batch_half", "optimizer", "learning_rate", "weight_decay", "dropout",
    "inner_ratio", "outer_ratio", "temperature", "inputs", "expected_sha256",
    "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"ORER配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if (
        config["schema_version"] != "gzsl-paper.orer.v1"
        or config["experiment_id"] != "V2-INNOVATION-034"
        or config["idea_id"] != "IDEA-068"
    ):
        raise ValueError("ORER身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("ORER协议边界错误。")
    if config["text_cache_provenance_complete"] is not False:
        raise ValueError("ORER文本cache provenance未完整。")
    if set(config["fold_model_sha256"]) != {"0", "1", "2"}:
        raise ValueError("ORER必须绑定三个fold模型SHA。")
    if (
        int(config["epochs"]) != 20
        or int(config["batch_half"]) != 25
        or config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
        or float(config["max_gamma_residual"]) != 0.1
    ):
        raise ValueError("ORER训练参数错误。")
    return config, sha256_file(path)


@torch.no_grad()
def evaluate(
    parent, sdrs, oclr_prototypes, oclr_beta, calibrator, tensors,
    seen_classes, unseen_classes, device
):
    prototypes = parent.prototypes()

    def predict(features, class_ids=None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = features.to(device).float()
        normalized = F.normalize(images, dim=-1)
        logits = normalized @ prototypes.index_select(0, ids).T * parent.scale()
        logits = sdrs(logits, images, ids)
        logits = logits + float(oclr_beta) * normalized @ oclr_prototypes.index_select(0, ids).T
        seen_mask = torch.isin(ids.cpu(), seen_classes).to(device)
        predictions = calibrator(logits, seen_mask).argmax(1).cpu()
        return predictions if class_ids is None else class_ids[predictions]

    seen_predictions = predict(tensors["seen_features"])
    unseen_predictions = predict(tensors["unseen_features"])
    zsl_predictions = predict(tensors["unseen_features"], unseen_classes)
    seen = h1.per_class_accuracy(tensors["seen_labels"], seen_predictions, seen_classes)
    unseen = h1.per_class_accuracy(tensors["unseen_labels"], unseen_predictions, unseen_classes)
    zsl = h1.per_class_accuracy(tensors["unseen_labels"], zsl_predictions, unseen_classes)
    return {
        "U": unseen * 100,
        "S": seen * 100,
        "H": 2 * seen * unseen / (seen + unseen) * 100,
        "ZS": zsl * 100,
    }


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    commit = current_code_commit()
    if commit != expected_commit:
        raise ValueError("expected-commit不一致。")
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config)
    input_sha = verify_inputs(config, paths)
    for key in (
        "base_model", "sdrs_model", "sebc_model", "oclr_model",
        "class_name_embeddings", "claude_embeddings",
    ):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"ORER {key} SHA错误。")
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
        claude = torch.load(
            Path(config["claude_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        normalized_names = F.normalize(class_names.float(), dim=-1)
        normalized_claude = F.normalize(claude.float(), dim=-1)
        oclr_prototypes = F.normalize(
            normalized_claude
            - (normalized_claude * normalized_names).sum(-1, keepdim=True)
            * normalized_names,
            dim=-1,
        )
        seen_classes = torch.unique(labels, sorted=True)
        all_classes = torch.arange(200)
        unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seen_classes) or not torch.equal(checked_unseen, unseen_classes):
            raise ValueError("ORER CUB类别边界错误。")

        parent, sdrs = _load_main(
            config, sentence, labels, features, class_names, seen_classes, device
        )
        packages = _load_folds(
            config, sentence, labels, features, seen_classes, device
        )
        sebc_payload = torch.load(
            Path(config["sebc_model"]), map_location="cpu", weights_only=False
        )
        sebc_config = sebc_payload["config"]
        parent_gamma = float(
            float(sebc_config["max_gamma"])
            * torch.tanh(sebc_payload["calibrator_state_dict"]["raw_gamma"].float())
        )
        oclr_payload = torch.load(
            Path(config["oclr_model"]), map_location="cpu", weights_only=False
        )
        oclr_beta = float(
            float(oclr_payload["config"]["max_beta"])
            * torch.tanh(oclr_payload["clre_state_dict"]["raw_beta"].float())
        )
        calibrator = GammaResidualCalibration(
            parent_gamma, float(config["max_gamma_residual"])
        ).to(device)
        optimizer = torch.optim.Adam(
            calibrator.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
        )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generators = [
            torch.Generator().manual_seed(seed * 71000 + fold_id)
            for fold_id in range(3)
        ]
        half = int(config["batch_half"])
        class_ids = seen_classes.to(device)
        class_beta = sdrs.class_beta(class_ids).detach()

        best_metrics = evaluate(
            parent, sdrs, oclr_prototypes, oclr_beta, calibrator, official,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"ORER初始态未复现OCLR：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(calibrator.state_dict())
        best_epoch = 0
        history = []
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "orer_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_epoch": best_epoch,
                "oclr_beta": oclr_beta,
                "config": config,
                "code_commit": commit,
                "reproducibility": reproducibility,
            },
        )

        for epoch in range(1, int(config["epochs"]) + 1):
            loss_sum = 0.0
            updates = 0
            for fold_id, package in enumerate(packages):
                pseudo_seen = package["pseudo_seen"]
                pseudo_unseen = package["pseudo_unseen"]
                seen_count = torch.isin(labels, pseudo_seen).sum().item()
                unseen_count = torch.isin(labels, pseudo_unseen).sum().item()
                steps = min(seen_count // half, unseen_count // half)
                for _ in range(steps):
                    indices = balanced_fold_batch(
                        labels, pseudo_seen, pseudo_unseen, half, generators[fold_id]
                    )
                    images = features.index_select(0, indices).to(device).float()
                    normalized = F.normalize(images, dim=-1)
                    targets = mapping[labels.index_select(0, indices)].to(device)
                    logits = package["model"].logits(images, class_ids)
                    name_logits = normalized @ class_names.index_select(0, class_ids).T
                    logits = logits + name_logits * class_beta.unsqueeze(0)
                    logits = logits + oclr_beta * normalized @ oclr_prototypes.index_select(
                        0, class_ids
                    ).T
                    logits = calibrator(logits, package["pseudo_seen_mask"])
                    loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    require_finite_gradients(calibrator)
                    optimizer.step()
                    loss_sum += float(loss.detach())
                    updates += 1

            metrics = evaluate(
                parent, sdrs, oclr_prototypes, oclr_beta, calibrator, official,
                seen_classes, unseen_classes, device
            )
            residual = float(calibrator.residual().detach())
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / updates,
                    "updates": updates,
                    "official_metrics_percent": metrics,
                    "gamma_residual": residual,
                }
            )
            if metrics["H"] > best_h:
                best_h = metrics["H"]
                best_metrics = metrics
                best_state = copy.deepcopy(calibrator.state_dict())
                best_epoch = epoch
                atomic_torch_save(
                    output_dir / "model_best.pth",
                    {
                        "orer_state_dict": best_state,
                        "best_metrics_percent": best_metrics,
                        "selected_epoch": best_epoch,
                        "oclr_beta": oclr_beta,
                        "config": config,
                        "code_commit": commit,
                        "reproducibility": reproducibility,
                    },
                )
            print(
                f"epoch={epoch} H={metrics['H']:.6f} best_H={best_h:.6f} "
                f"gamma_residual={residual:.6f} updates={updates}"
            )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "orer_state_dict": copy.deepcopy(calibrator.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_epoch": best_epoch,
                "history": history,
                "oclr_beta": oclr_beta,
                "config": config,
                "code_commit": commit,
            },
        )
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                "fold_models": config["fold_model_sha256"],
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "oclr_model": config["oclr_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
                "claude_embeddings": config["claude_embeddings_sha256"],
            },
        )
        best_residual = float(
            torch.tanh(best_state["raw_residual"])
            * float(config["max_gamma_residual"])
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
            "selected_epoch": best_epoch,
            "parent_gamma": parent_gamma,
            "gamma_residual": best_residual,
            "final_gamma": parent_gamma + best_residual,
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
