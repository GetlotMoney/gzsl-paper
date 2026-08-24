from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml

from model.innovations.elpt import (
    class_fold_sha256,
    semantic_balanced_class_folds,
)
from model.innovations.train_unified_seen import full_epoch_batches
from model.innovations.unified_seen import UnifiedSeenPrototypeModel
from model.tg_vpr_h1 import train as h1
from tools.prepare_cub_standard_validation import (
    SEEN_HOLDOUT_FRACTION,
    SPLIT_SEED,
    _per_class_fit_and_seen_validation,
)
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
    require_finite_gradients,
    require_finite_model,
)
from tools.runtime import sha256_file


EVALUATION_PROTOCOL = "threefold_class_disjoint_gzsl_validation"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "condition_id",
    "framework_id",
    "dataset",
    "evaluation_protocol",
    "validation_used_for_selection",
    "test_used_for_selection",
    "official_test_loaded",
    "validation_images_used_for_gradient",
    "pseudo_unseen_images_used_for_gradient",
    "expert_attributes_used",
    "historical_test_informed_architecture",
    "strict_blind_claim",
    "feature_backbone",
    "feature_provenance_complete",
    "final_test_eligible",
    "fold_method",
    "fold_count",
    "fold_sha256",
    "split_seed",
    "seen_holdout_fraction",
    "device",
    "random_seed",
    "batch_size",
    "epochs",
    "weight_decay",
    "dropout",
    "inner_ratio",
    "outer_ratio",
    "topology_weight",
    "temperature",
    "transport_hidden_dim",
    "generator_hidden_dim",
    "max_transport_step",
    "max_generator_magnitude",
    "lr_stages",
    "inputs",
    "expected_sha256",
    "class_order_sha256",
}
INPUT_KEYS = (
    "sentence_embeds",
    "train_features",
    "train_labels",
    "res101",
    "att_splits",
)


def load_config(path: Path) -> tuple[dict, str]:
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"三折validation配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if (
        config["schema_version"] != "gzsl-paper.pure-threefold-validation.v1"
        or config["experiment_id"] != "V2-TUNE-003"
        or config["condition_id"] != "BASELINE"
        or config["framework_id"] != "FRAMEWORK-V2"
        or config["dataset"] != "CUB"
        or config["evaluation_protocol"] != EVALUATION_PROTOCOL
    ):
        raise ValueError("三折validation身份错误。")
    required = {
        "validation_used_for_selection": True,
        "test_used_for_selection": False,
        "official_test_loaded": False,
        "validation_images_used_for_gradient": False,
        "pseudo_unseen_images_used_for_gradient": False,
        "expert_attributes_used": False,
        "historical_test_informed_architecture": True,
        "strict_blind_claim": False,
        "feature_provenance_complete": False,
        "final_test_eligible": False,
    }
    for name, expected in required.items():
        if config[name] is not expected:
            raise ValueError(f"三折validation边界错误：{name}必须为{expected}。")
    if (
        config["fold_method"] != "semantic_pca_round_robin_image_balance_v1"
        or int(config["fold_count"]) != 3
        or len(config["fold_sha256"]) != 64
        or int(config["split_seed"]) != SPLIT_SEED
        or float(config["seen_holdout_fraction"]) != SEEN_HOLDOUT_FRACTION
        or int(config["batch_size"]) != 64
        or int(config["epochs"]) != 50
        or [int(stage["epochs"]) for stage in config["lr_stages"]] != [20, 20, 10]
    ):
        raise ValueError("三折validation固定折、数据holdout或训练周期错误。")
    if set(config["inputs"]) != set(INPUT_KEYS) or set(
        config["expected_sha256"]
    ) != set(INPUT_KEYS):
        raise ValueError("三折validation输入字段错误。")
    return config, sha256_file(path)


def resolve_and_verify_inputs(config: dict) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {name: h1.repo_path(config["inputs"][name]) for name in INPUT_KEYS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少三折validation输入：" + ", ".join(missing))
    actual = {name: sha256_file(path) for name, path in paths.items()}
    mismatch = [
        name for name in INPUT_KEYS if actual[name] != config["expected_sha256"][name]
    ]
    if mismatch:
        raise ValueError("三折validation输入SHA不匹配：" + ", ".join(mismatch))
    names = sio.loadmat(paths["att_splits"], variable_names=["allclasses_names"])[
        "allclasses_names"
    ]
    serialized = json.dumps(
        [str(item[0][0]) for item in names], ensure_ascii=False, separators=(",", ":")
    )
    if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != config[
        "class_order_sha256"
    ]:
        raise ValueError("CUB类别顺序不匹配。")
    return paths, actual


def build_fold_splits(
    labels: torch.Tensor,
    folds: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int = SPLIT_SEED,
    holdout_fraction: float = SEEN_HOLDOUT_FRACTION,
) -> list[dict[str, torch.Tensor]]:
    positions = torch.arange(labels.numel())
    result = []
    for pseudo_seen, pseudo_unseen in folds:
        seen_positions = positions[torch.isin(labels, pseudo_seen)]
        fit_positions, val_seen_positions = _per_class_fit_and_seen_validation(
            seen_positions,
            labels,
            pseudo_seen,
            seed=int(seed),
            holdout_fraction=float(holdout_fraction),
        )
        val_unseen_positions = positions[torch.isin(labels, pseudo_unseen)]
        if (
            torch.isin(fit_positions, val_seen_positions).any()
            or torch.isin(fit_positions, val_unseen_positions).any()
            or torch.isin(val_seen_positions, val_unseen_positions).any()
        ):
            raise RuntimeError("三折fit/val-seen/val-unseen必须互斥。")
        result.append(
            {
                "pseudo_seen": pseudo_seen,
                "pseudo_unseen": pseudo_unseen,
                "fit_positions": fit_positions,
                "val_seen_positions": val_seen_positions,
                "val_unseen_positions": val_unseen_positions,
            }
        )
    return result


def _per_class_accuracy(
    labels: torch.Tensor, predictions: torch.Tensor, classes: torch.Tensor
) -> float:
    values = []
    for class_id in classes.cpu().long():
        mask = labels.cpu().long().eq(class_id)
        if not mask.any():
            raise ValueError(f"三折validation缺少类别{int(class_id)}。")
        values.append(predictions.cpu().long()[mask].eq(class_id).float().mean())
    return float(torch.stack(values).mean())


@torch.no_grad()
def evaluate_fold(
    model: UnifiedSeenPrototypeModel,
    features: torch.Tensor,
    labels: torch.Tensor,
    split: dict[str, torch.Tensor],
    allclasses: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    prototypes = model.prototypes().index_select(0, allclasses.to(device))

    def predict(positions: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        selected = model.prototypes().index_select(0, class_ids.to(device))
        logits = (
            F.normalize(features.index_select(0, positions).to(device).float(), dim=-1)
            @ selected.T
            * model.scale()
        )
        return class_ids[logits.argmax(dim=1).cpu()]

    del prototypes
    seen_labels = labels.index_select(0, split["val_seen_positions"])
    unseen_labels = labels.index_select(0, split["val_unseen_positions"])
    seen_prediction = predict(split["val_seen_positions"], allclasses)
    unseen_prediction = predict(split["val_unseen_positions"], allclasses)
    zsl_prediction = predict(split["val_unseen_positions"], split["pseudo_unseen"])
    seen = _per_class_accuracy(seen_labels, seen_prediction, split["pseudo_seen"])
    unseen = _per_class_accuracy(
        unseen_labels, unseen_prediction, split["pseudo_unseen"]
    )
    zsl = _per_class_accuracy(unseen_labels, zsl_prediction, split["pseudo_unseen"])
    harmonic = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    return {"U": unseen * 100, "S": seen * 100, "H": harmonic * 100, "ZS": zsl * 100}


def aggregate_fold_histories(
    histories: list[list[dict]],
) -> tuple[list[dict], dict]:
    if len(histories) != 3 or not histories or len({len(rows) for rows in histories}) != 1:
        raise ValueError("三折history必须包含三个等长fold。")
    aggregate = []
    for epoch_index in range(len(histories[0])):
        metrics = [rows[epoch_index]["validation_metrics_percent"] for rows in histories]
        mean = {
            key: sum(float(metric[key]) for metric in metrics) / 3.0
            for key in ("U", "S", "H", "ZS")
        }
        h_values = [float(metric["H"]) for metric in metrics]
        aggregate.append(
            {
                "epoch": epoch_index + 1,
                "mean_metrics_percent": mean,
                "min_H": min(h_values),
                "max_H": max(h_values),
                "range_H": max(h_values) - min(h_values),
                "fold_metrics_percent": metrics,
            }
        )
    selected = max(aggregate, key=lambda row: row["mean_metrics_percent"]["H"])
    return aggregate, selected


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    config, config_sha = load_config(config_path)
    paths, input_sha = resolve_and_verify_inputs(config)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("纯三折validation要求可见CUDA。")
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = h1.TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(
            seed, strict_determinism=True, deterministic_warn_only=False
        )
        sentence_embeds = torch.load(
            paths["sentence_embeds"], map_location="cpu", weights_only=True
        )
        features = torch.load(
            paths["train_features"], map_location="cpu", weights_only=True
        ).float()
        labels = torch.load(
            paths["train_labels"], map_location="cpu", weights_only=True
        ).long()
        classes = torch.unique(labels, sorted=True)
        if labels.numel() != 7057 or classes.numel() != 150:
            raise ValueError("纯三折必须使用150类、7057张trainval图像。")
        class_counts = torch.stack([labels.eq(class_id).sum() for class_id in classes])
        folds = semantic_balanced_class_folds(
            classes,
            sentence_embeds,
            class_counts,
            fold_count=int(config["fold_count"]),
        )
        fold_sha = class_fold_sha256(folds)
        if fold_sha != config["fold_sha256"]:
            raise ValueError("纯三折类别清单SHA不匹配。")
        fold_splits = build_fold_splits(
            labels,
            folds,
            seed=int(config["split_seed"]),
            holdout_fraction=float(config["seen_holdout_fraction"]),
        )
        fold_histories = []
        fold_best_states = []
        fold_summaries = []
        print(
            f"实验={config['experiment_id']} commit={code_commit} config_sha={config_sha} "
            f"fold_sha={fold_sha} official-test-loaded=false"
        )
        for fold_id, split in enumerate(fold_splits):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            fit_positions = split["fit_positions"]
            fit_labels = labels.index_select(0, fit_positions)
            pseudo_seen = split["pseudo_seen"]
            centroids = h1.visual_centroids(
                features.index_select(0, fit_positions), fit_labels, pseudo_seen
            )
            model = UnifiedSeenPrototypeModel(
                sentence_embeds,
                pseudo_seen,
                centroids,
                active_classes=classes,
                dropout=float(config["dropout"]),
                inner_ratio=float(config["inner_ratio"]),
                outer_ratio=float(config["outer_ratio"]),
                temperature=float(config["temperature"]),
                transport_hidden_dim=int(config["transport_hidden_dim"]),
                generator_hidden_dim=int(config["generator_hidden_dim"]),
                max_transport_step=float(config["max_transport_step"]),
                max_generator_magnitude=float(config["max_generator_magnitude"]),
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(config["lr_stages"][0]["lr"]),
                weight_decay=float(config["weight_decay"]),
            )
            stages = config["lr_stages"]
            boundaries = []
            total = 0
            for stage in stages:
                total += int(stage["epochs"])
                boundaries.append(total)
            active_stage = 0
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(stages[0]["epochs"]),
                eta_min=float(stages[0]["eta_min"]),
            )
            mapping = torch.full((200,), -1, dtype=torch.long)
            mapping[pseudo_seen] = torch.arange(pseudo_seen.numel())
            generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + fold_id)
            history = []
            best_h = float("-inf")
            best_state = None
            best_epoch = None
            for epoch in range(1, int(config["epochs"]) + 1):
                target_stage = next(
                    index for index, boundary in enumerate(boundaries) if epoch <= boundary
                )
                if target_stage != active_stage:
                    active_stage = target_stage
                    stage = stages[active_stage]
                    for group in optimizer.param_groups:
                        group["lr"] = float(stage["lr"])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=int(stage["epochs"]),
                        eta_min=float(stage["eta_min"]),
                    )
                model.train()
                loss_sum = 0.0
                sample_count = 0
                for relative in full_epoch_batches(
                    fit_positions.numel(), int(config["batch_size"]), generator
                ):
                    positions = fit_positions.index_select(0, relative)
                    images = features.index_select(0, positions).to(device)
                    targets = mapping[labels.index_select(0, positions)].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    ce = F.cross_entropy(model.logits(images, pseudo_seen), targets)
                    topology = model.topology_loss()
                    loss = ce + float(config["topology_weight"]) * topology
                    loss.backward()
                    require_finite_gradients(model)
                    optimizer.step()
                    loss_sum += float(loss.detach()) * images.size(0)
                    sample_count += images.size(0)
                scheduler.step()
                metrics = evaluate_fold(
                    model, features, labels, split, classes, device
                )
                row = {
                    "epoch": epoch,
                    "train_loss": loss_sum / sample_count,
                    "fit_sample_count": sample_count,
                    "validation_metrics_percent": metrics,
                    "diagnostics": model.diagnostics(),
                }
                history.append(row)
                if metrics["H"] > best_h:
                    best_h = metrics["H"]
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                print(
                    f"fold={fold_id} epoch={epoch} fit={sample_count} "
                    f"U={metrics['U']:.6f} S={metrics['S']:.6f} H={metrics['H']:.6f}"
                )
            require_finite_model(model)
            fold_histories.append(history)
            fold_best_states.append(best_state)
            fold_summaries.append(
                {
                    "fold_id": fold_id,
                    "pseudo_seen_classes": [int(value) for value in pseudo_seen],
                    "pseudo_unseen_classes": [
                        int(value) for value in split["pseudo_unseen"]
                    ],
                    "fit_images": int(split["fit_positions"].numel()),
                    "val_seen_images": int(split["val_seen_positions"].numel()),
                    "val_unseen_images": int(split["val_unseen_positions"].numel()),
                    "fold_best_epoch": best_epoch,
                    "fold_best_H": best_h,
                }
            )
        aggregate_history, selected = aggregate_fold_histories(fold_histories)
        checkpoint = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "run_id": run_id,
            "code_commit": code_commit,
            "config": config,
            "config_sha256": config_sha,
            "fold_sha256": fold_sha,
            "fold_summaries": fold_summaries,
            "fold_best_state_dicts": fold_best_states,
            "fold_histories": fold_histories,
            "aggregate_history": aggregate_history,
            "selected_aggregate_epoch": selected["epoch"],
            "selected_aggregate_metrics": selected,
            "reproducibility": reproducibility,
        }
        atomic_torch_save(output_dir / "model_best.pth", checkpoint)
        atomic_torch_save(output_dir / "checkpoint_last.pth", checkpoint)
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        metrics = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "run_id": run_id,
            "framework_id": config["framework_id"],
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "validation_used_for_selection": True,
            "test_used_for_selection": False,
            "official_test_loaded": False,
            "validation_images_used_for_gradient": False,
            "pseudo_unseen_images_used_for_gradient": False,
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "fold_sha256": fold_sha,
            "fold_summaries": fold_summaries,
            "selected_epoch": selected["epoch"],
            "mean_metrics_percent": selected["mean_metrics_percent"],
            "min_H": selected["min_H"],
            "max_H": selected["max_H"],
            "range_H": selected["range_H"],
            "fold_metrics_percent": selected["fold_metrics_percent"],
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": sha256_file(
                output_dir / "checkpoint_last.pth"
            ),
        }
        atomic_write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        return metrics
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.run_id)


if __name__ == "__main__":
    main()
