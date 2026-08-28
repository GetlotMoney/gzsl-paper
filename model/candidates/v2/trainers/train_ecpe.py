from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.ccpe import (
    ClassConditionedPatchEvidence,
    class_conditioned_patch_scores,
)
from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
from model.candidates.v2.modules.lpsr import orthogonal_local_text_residuals
from model.candidates.v2.trainers.train_ccpe import evaluate
from model.candidates.v2.trainers.train_chen_class_exclusive import balanced_fold_batch
from model.candidates.v2.trainers.train_chen_style import OFFICIAL_KEYS, resolve_paths, verify_inputs
from model.candidates.v2.trainers.train_sebc import _load_folds, _load_main
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
    "sebc_model_sha256", "parent_metrics_percent", "comparison_H",
    "class_name_embeddings", "class_name_embeddings_sha256", "fold_model_dir",
    "fold_model_sha256", "patch_inputs", "patch_sha256", "patch_top_k",
    "patch_chunk_size", "device", "random_seed", "epochs", "batch_half",
    "optimizer", "learning_rate", "weight_decay", "max_beta", "dropout",
    "inner_ratio", "outer_ratio", "temperature", "inputs", "expected_sha256",
    "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"ECPE配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if (
        config["schema_version"] != "gzsl-paper.ecpe.v1"
        or config["experiment_id"] != "V2-INNOVATION-021"
        or config["idea_id"] != "IDEA-055"
    ):
        raise ValueError("ECPE身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("ECPE协议边界错误。")
    if config["feature_provenance_complete"] is not False:
        raise ValueError("遗留CLIP patch provenance未完整，不得标成完整。")
    if set(config["fold_model_sha256"]) != {"0", "1", "2"}:
        raise ValueError("ECPE必须绑定三个fold模型SHA。")
    if set(config["patch_inputs"]) != {"train", "seen", "unseen"}:
        raise ValueError("ECPE patch输入必须包含train/seen/unseen。")
    if set(config["patch_sha256"]) != {"train", "seen", "unseen"}:
        raise ValueError("ECPE patch SHA必须包含train/seen/unseen。")
    if (
        int(config["patch_top_k"]) != 2
        or int(config["patch_chunk_size"]) != 16
        or int(config["epochs"]) != 20
        or int(config["batch_half"]) != 25
    ):
        raise ValueError("ECPE patch或episode训练量错误。")
    if (
        config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
        or float(config["max_beta"]) != 10.0
        or abs(float(config["comparison_H"]) - 77.6665326315915) > 1e-9
    ):
        raise ValueError("ECPE优化参数或比较门槛错误。")
    return config, sha256_file(path)


def _precompute_scores(config, text_prototypes, device):
    scores = {}
    for split, path_text in config["patch_inputs"].items():
        path = h1.repo_path(path_text)
        if sha256_file(path) != config["patch_sha256"][split]:
            raise ValueError(f"ECPE {split} patch SHA错误。")
        patches = torch.load(path, map_location="cpu", weights_only=True)
        scores[split] = class_conditioned_patch_scores(
            patches, text_prototypes, int(config["patch_top_k"]), device,
            chunk_size=int(config["patch_chunk_size"]),
        )
        del patches
    return scores


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    commit = current_code_commit()
    if commit != expected_commit:
        raise ValueError("expected-commit不一致。")
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config)
    input_sha = verify_inputs(config, paths)
    for key in ("base_model", "sdrs_model", "sebc_model", "class_name_embeddings"):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"ECPE {key} SHA错误。")

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
        names = torch.load(
            Path(config["class_name_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        seen_classes = torch.unique(labels, sorted=True)
        all_classes = torch.arange(200)
        unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seen_classes) or not torch.equal(checked_unseen, unseen_classes):
            raise ValueError("ECPE CUB类别边界错误。")

        parent, sdrs = _load_main(
            config, sentence, labels, features, names, seen_classes, device
        )
        packages = _load_folds(
            config, sentence, labels, features, seen_classes, device
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

        local_text = orthogonal_local_text_residuals(sentence, names)
        print("precomputing ECPE top2 patch scores")
        scores = _precompute_scores(config, local_text, device)
        if scores["train"].shape != (labels.shape[0], 200):
            raise ValueError("ECPE train score形状错误。")
        if scores["seen"].shape != (official["seen_labels"].shape[0], 200):
            raise ValueError("ECPE test-seen score形状错误。")
        if scores["unseen"].shape != (official["unseen_labels"].shape[0], 200):
            raise ValueError("ECPE test-unseen score形状错误。")

        model = ClassConditionedPatchEvidence(float(config["max_beta"])).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
        )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generators = [
            torch.Generator().manual_seed(seed * 67000 + fold_id)
            for fold_id in range(3)
        ]
        half = int(config["batch_half"])
        class_ids = seen_classes.to(device)
        class_beta = sdrs.class_beta(class_ids).detach()

        best_metrics = evaluate(
            parent, sdrs, calibrator, model, official, scores,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"ECPE关闭态未复现SEBC父指标：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        history = []
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "ecpe_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_epoch": best_epoch,
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
                    targets = mapping[labels.index_select(0, indices)].to(device)
                    fold_logits = package["model"].logits(images, class_ids)
                    name_logits = F.normalize(images, dim=-1) @ names.index_select(
                        0, class_ids
                    ).T
                    fold_logits = fold_logits + name_logits * class_beta.unsqueeze(0)
                    fold_logits = calibrator(fold_logits, package["pseudo_seen_mask"])
                    patch_scores = scores["train"].index_select(0, indices).to(device)
                    logits = model(fold_logits, patch_scores, class_ids)
                    loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    require_finite_gradients(model)
                    optimizer.step()
                    loss_sum += float(loss.detach())
                    updates += 1

            metrics = evaluate(
                parent, sdrs, calibrator, model, official, scores,
                seen_classes, unseen_classes, device
            )
            beta = float(model.beta().detach())
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / updates,
                    "updates": updates,
                    "official_metrics_percent": metrics,
                    "beta": beta,
                }
            )
            if metrics["H"] > best_h:
                best_h = metrics["H"]
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                atomic_torch_save(
                    output_dir / "model_best.pth",
                    {
                        "ecpe_state_dict": best_state,
                        "best_metrics_percent": best_metrics,
                        "selected_epoch": best_epoch,
                        "config": config,
                        "code_commit": commit,
                        "reproducibility": reproducibility,
                    },
                )
            print(
                f"epoch={epoch} H={metrics['H']:.6f} best_H={best_h:.6f} "
                f"beta={beta:.6f} updates={updates}"
            )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "ecpe_state_dict": copy.deepcopy(model.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_epoch": best_epoch,
                "history": history,
                "config": config,
                "code_commit": commit,
            },
        )
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                "patch_files": config["patch_sha256"],
                "fold_models": config["fold_model_sha256"],
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
            },
        )
        best_beta = float(
            torch.tanh(best_state["raw_beta"]) * float(config["max_beta"])
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
            "comparison_H": float(config["comparison_H"]),
            "best_metrics_percent": best_metrics,
            "delta_vs_parent_percent_points": {
                key: best_metrics[key] - float(expected_parent[key])
                for key in ("U", "S", "H", "ZS")
            },
            "selected_epoch": best_epoch,
            "learned_beta": best_beta,
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
