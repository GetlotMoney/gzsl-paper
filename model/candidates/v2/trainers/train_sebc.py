from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
from model.frameworks.v4.tg import VariableClassTGVPR, fixed_class_folds
from model.candidates.v2.modules.sdrs import SemanticDisagreementResidualScaling
from model.candidates.v2.trainers.train_chen_class_exclusive import balanced_fold_batch
from model.candidates.v2.trainers.train_chen_style import OFFICIAL_KEYS, resolve_paths, verify_inputs
from model.candidates.v2.modules.unified_seen import UnifiedSeenPrototypeModel
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
    "strict_blind_claim", "base_model", "base_model_sha256", "sdrs_model",
    "sdrs_model_sha256", "parent_metrics_percent", "class_name_embeddings",
    "class_name_embeddings_sha256", "fold_model_dir", "fold_model_sha256", "device",
    "random_seed", "epochs", "batch_half", "optimizer", "learning_rate",
    "weight_decay", "max_gamma", "dropout", "inner_ratio", "outer_ratio",
    "temperature", "inputs", "expected_sha256", "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"SEBC配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    gamma_by_schema = {
        "gzsl-paper.sebc.v1": 2.0,
        "gzsl-paper.sebc.v2": 0.2,
    }
    if (
        config["schema_version"] not in gamma_by_schema
        or config["experiment_id"] != "V2-INNOVATION-013"
        or config["idea_id"] != "IDEA-047"
    ):
        raise ValueError("SEBC身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("SEBC协议边界错误。")
    if set(config["fold_model_sha256"]) != {"0", "1", "2"}:
        raise ValueError("SEBC必须绑定三个fold模型SHA。")
    if (
        int(config["epochs"]) != 20
        or int(config["batch_half"]) != 25
        or config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
        or float(config["max_gamma"]) != gamma_by_schema[config["schema_version"]]
    ):
        raise ValueError("SEBC训练参数错误。")
    return config, sha256_file(path)


def _load_main(config, sentence, labels, features, names, seen_classes, device):
    centroids = h1.visual_centroids(features, labels, seen_classes)
    base_payload = torch.load(Path(config["base_model"]), map_location="cpu", weights_only=False)
    parent_config = base_payload["config"]
    all_classes = torch.arange(200)
    parent = UnifiedSeenPrototypeModel(
        sentence, seen_classes, centroids, active_classes=all_classes,
        dropout=float(parent_config["dropout"]),
        inner_ratio=float(parent_config["inner_ratio"]),
        outer_ratio=float(parent_config["outer_ratio"]),
        temperature=float(parent_config["temperature"]),
        transport_hidden_dim=int(parent_config["transport_hidden_dim"]),
        generator_hidden_dim=int(parent_config["generator_hidden_dim"]),
        max_transport_step=float(parent_config["max_transport_step"]),
        max_generator_magnitude=float(parent_config["max_generator_magnitude"]),
    ).to(device)
    parent.load_state_dict(base_payload["model_state_dict"], strict=True)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    sdrs_payload = torch.load(Path(config["sdrs_model"]), map_location="cpu", weights_only=False)
    sdrs_config = sdrs_payload["config"]
    sdrs = SemanticDisagreementResidualScaling(
        parent.prototypes().detach(), names, seen_classes.to(device),
        float(sdrs_payload["base_beta"]), float(sdrs_config["max_delta"]),
    ).to(device)
    sdrs.load_state_dict(sdrs_payload["sdrs_state_dict"], strict=True)
    sdrs.eval()
    for parameter in sdrs.parameters():
        parameter.requires_grad_(False)
    return parent, sdrs


def _load_folds(config, sentence, labels, features, seen_classes, device):
    folds = fixed_class_folds(seen_classes)
    packages = []
    fold_dir = Path(config["fold_model_dir"])
    for fold_id, (pseudo_seen, pseudo_unseen) in enumerate(folds):
        path = fold_dir / f"fold_{fold_id}.pth"
        if sha256_file(path) != config["fold_model_sha256"][str(fold_id)]:
            raise ValueError(f"SEBC fold {fold_id}模型SHA错误。")
        mask = torch.isin(labels, pseudo_seen)
        positions = mask.nonzero(as_tuple=False).flatten()
        fold_labels = labels.index_select(0, positions)
        centroids = h1.visual_centroids(
            features.index_select(0, positions), fold_labels, pseudo_seen
        )
        model = VariableClassTGVPR(
            sentence, pseudo_seen, centroids,
            dropout=float(config["dropout"]),
            inner_ratio=float(config["inner_ratio"]),
            outer_ratio=float(config["outer_ratio"]),
            temperature=float(config["temperature"]),
        ).to(device)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.equal(payload["pseudo_seen"], pseudo_seen):
            raise ValueError(f"SEBC fold {fold_id} pseudo-seen身份错误。")
        if not torch.equal(payload["pseudo_unseen"], pseudo_unseen):
            raise ValueError(f"SEBC fold {fold_id} pseudo-unseen身份错误。")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        packages.append(
            {
                "model": model,
                "pseudo_seen": pseudo_seen,
                "pseudo_unseen": pseudo_unseen,
                "pseudo_seen_mask": torch.isin(seen_classes, pseudo_seen).to(device),
            }
        )
    return packages


@torch.no_grad()
def evaluate(parent, sdrs, calibrator, tensors, seen_classes, unseen_classes, device):
    prototypes = parent.prototypes()

    def predict(features, class_ids=None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = features.to(device).float()
        base = F.normalize(images, dim=-1) @ prototypes.index_select(0, ids).T * parent.scale()
        logits = sdrs(base, images, ids)
        mask = torch.isin(ids.cpu(), seen_classes).to(device)
        predictions = calibrator(logits, mask).argmax(1).cpu()
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
    if sha256_file(Path(config["base_model"])) != config["base_model_sha256"]:
        raise ValueError("SEBC基础模型SHA错误。")
    if sha256_file(Path(config["sdrs_model"])) != config["sdrs_model_sha256"]:
        raise ValueError("SEBC SDRS父模型SHA错误。")
    names_path = Path(config["class_name_embeddings"])
    if sha256_file(names_path) != config["class_name_embeddings_sha256"]:
        raise ValueError("SEBC类名cache SHA错误。")

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
        names = torch.load(names_path, map_location="cpu", weights_only=True).to(device)
        seen_classes = torch.unique(labels, sorted=True)
        all_classes = torch.arange(200)
        unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seen_classes) or not torch.equal(checked_unseen, unseen_classes):
            raise ValueError("SEBC CUB类别边界错误。")

        parent, sdrs = _load_main(
            config, sentence, labels, features, names, seen_classes, device
        )
        packages = _load_folds(
            config, sentence, labels, features, seen_classes, device
        )
        calibrator = EpisodicBiasCalibration(float(config["max_gamma"])).to(device)
        optimizer = torch.optim.Adam(
            calibrator.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
        )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generators = [
            torch.Generator().manual_seed(seed * 61000 + fold_id)
            for fold_id in range(3)
        ]
        class_beta = sdrs.class_beta(seen_classes.to(device)).detach()
        half = int(config["batch_half"])

        history = []
        best_metrics = evaluate(
            parent, sdrs, calibrator, official, seen_classes, unseen_classes, device
        )
        for key, expected in config["parent_metrics_percent"].items():
            if abs(best_metrics[key] - float(expected)) > 1e-5:
                raise ValueError(f"SEBC关闭态未复现SDRS父指标：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(calibrator.state_dict())
        best_epoch = 0
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "calibrator_state_dict": best_state,
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
                    base_logits = package["model"].logits(images, seen_classes.to(device))
                    name_logits = F.normalize(images, dim=-1) @ names.index_select(
                        0, seen_classes.to(device)
                    ).T
                    logits = base_logits + name_logits * class_beta.unsqueeze(0)
                    corrected = calibrator(logits, package["pseudo_seen_mask"])
                    loss = F.cross_entropy(corrected, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    require_finite_gradients(calibrator)
                    optimizer.step()
                    loss_sum += float(loss.detach())
                    updates += 1
            metrics = evaluate(
                parent, sdrs, calibrator, official, seen_classes, unseen_classes, device
            )
            gamma = float(calibrator.gamma().detach())
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / updates,
                    "updates": updates,
                    "official_metrics_percent": metrics,
                    "gamma": gamma,
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
                        "calibrator_state_dict": best_state,
                        "best_metrics_percent": best_metrics,
                        "selected_epoch": best_epoch,
                        "config": config,
                        "code_commit": commit,
                        "reproducibility": reproducibility,
                    },
                )
            print(
                f"epoch={epoch} H={metrics['H']:.6f} best_H={best_h:.6f} "
                f"gamma={gamma:.6f} updates={updates}"
            )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "calibrator_state_dict": copy.deepcopy(calibrator.state_dict()),
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
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
                "fold_models": config["fold_model_sha256"],
            },
        )
        best_raw = best_state["raw_gamma"]
        best_gamma = float(torch.tanh(best_raw) * float(config["max_gamma"]))
        expected_parent = config["parent_metrics_percent"]
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
            "delta_vs_parent_percent_points": {
                key: best_metrics[key] - float(expected_parent[key])
                for key in ("U", "S", "H", "ZS")
            },
            "selected_epoch": best_epoch,
            "learned_gamma": best_gamma,
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
