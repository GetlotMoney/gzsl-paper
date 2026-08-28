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

from model.frameworks.v4.tg import (
    class_fold_sha256,
    semantic_balanced_class_folds,
)
from model.candidates.v2.trainers.train_chen_class_exclusive import balanced_fold_batch
from model.candidates.v2.trainers.train_unified_seen import full_epoch_batches
from model.candidates.v2.modules.unified_expert import ExpertAttributeUnifiedModel
from model.candidates.v2.modules.unified_seen import UnifiedSeenPrototypeModel
from model.frameworks.v2 import train as h1
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


EVALUATION_PROTOCOL = "xlsa17_class_disjoint_gzsl_validation"
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
    "expert_attributes_used",
    "feature_backbone",
    "feature_provenance_complete",
    "historical_test_informed_architecture",
    "final_test_eligible",
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
    "max_attribute_residual",
    "lr_stages",
    "inputs",
    "expected_sha256",
    "class_order_sha256",
}
NESTED_CONFIG_KEYS = CONFIG_KEYS | {
    "inner_fold_scope",
    "inner_fold_method",
    "inner_fold_count",
    "inner_fold_sha256",
    "inner_episode_weight",
    "inner_batch_half",
    "inner_pseudo_unseen_images_used_for_gradient",
}
NESTED_V2_CONFIG_KEYS = NESTED_CONFIG_KEYS | {"inner_gradient_scope"}
NESTED_V3_CONFIG_KEYS = NESTED_V2_CONFIG_KEYS | {"inner_batch_mode"}
INPUT_KEYS = (
    "sentence_embeds",
    "train_features",
    "train_labels",
    "res101",
    "att_splits",
    "validation_split",
)


def load_config(path: Path) -> tuple[dict, str]:
    path = h1.repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"标准validation配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    schema = config.get("schema_version") if isinstance(config, dict) else None
    expected_keys = {
        "gzsl-paper.standard-nested-validation.v1": NESTED_CONFIG_KEYS,
        "gzsl-paper.standard-nested-validation.v2": NESTED_V2_CONFIG_KEYS,
        "gzsl-paper.standard-nested-validation.v3": NESTED_V3_CONFIG_KEYS,
    }.get(schema, CONFIG_KEYS)
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"标准validation配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    if schema not in (
        "gzsl-paper.standard-validation.v1",
        "gzsl-paper.standard-nested-validation.v1",
        "gzsl-paper.standard-nested-validation.v2",
        "gzsl-paper.standard-nested-validation.v3",
    ):
        raise ValueError("标准validation配置schema错误。")
    nested = schema in (
        "gzsl-paper.standard-nested-validation.v1",
        "gzsl-paper.standard-nested-validation.v2",
        "gzsl-paper.standard-nested-validation.v3",
    )
    expected_experiment = "V2-TUNE-002" if nested else "V2-TUNE-001"
    if config["experiment_id"] != expected_experiment:
        raise ValueError("标准validation实验身份错误。")
    allowed_conditions = (
        ("NESTED-THREE-MODULE",) if nested else ("NO-EXPERT", "EXPERT")
    )
    if config["condition_id"] not in allowed_conditions:
        raise ValueError("condition_id只允许NO-EXPERT或EXPERT。")
    if config["framework_id"] != "FRAMEWORK-V2" or config["dataset"] != "CUB":
        raise ValueError("标准validation只接受FRAMEWORK-V2/CUB。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("标准validation协议身份错误。")
    required_boundaries = {
        "validation_used_for_selection": True,
        "test_used_for_selection": False,
        "official_test_loaded": False,
        "validation_images_used_for_gradient": False,
    }
    for name, expected in required_boundaries.items():
        if config[name] is not expected:
            raise ValueError(f"标准validation边界错误：{name}必须为{expected}。")
    expected_expert = config["condition_id"] == "EXPERT"
    if config["expert_attributes_used"] is not expected_expert:
        raise ValueError("专家属性开关与condition_id不一致。")
    if config["feature_provenance_complete"] is not False:
        raise ValueError("当前遗留CLIP缓存只能明确标记为开发期来源不完整。")
    if config["historical_test_informed_architecture"] is not True:
        raise ValueError("必须披露当前结构受历史official test探索影响。")
    if config["final_test_eligible"] is not False:
        raise ValueError("来源不完整的遗留CLIP缓存不得标记为最终测试候选。")
    if int(config["epochs"]) != 50 or int(config["batch_size"]) != 64:
        raise ValueError("首次validation固定50轮、batch size 64。")
    if [int(stage["epochs"]) for stage in config["lr_stages"]] != [20, 20, 10]:
        raise ValueError("validation训练固定20/20/10学习率阶段。")
    if nested and (
        config["inner_fold_scope"] != "outer_train_classes_only"
        or config["inner_fold_method"]
        != "semantic_pca_round_robin_image_balance_v1"
        or int(config["inner_fold_count"]) != 3
        or len(config["inner_fold_sha256"]) != 64
        or float(config["inner_episode_weight"]) not in (1.0, 0.1)
        or int(config["inner_batch_half"]) != 32
        or config["inner_pseudo_unseen_images_used_for_gradient"] is not True
    ):
        raise ValueError("嵌套validation内层三折配置错误。")
    if schema in (
        "gzsl-paper.standard-nested-validation.v2",
        "gzsl-paper.standard-nested-validation.v3",
    ) and config[
        "inner_gradient_scope"
    ] != "transport_generator_only":
        raise ValueError("嵌套validation v2只允许迁移与生成模块接收inner梯度。")
    if schema == "gzsl-paper.standard-nested-validation.v3" and config[
        "inner_batch_mode"
    ] != "pseudo_unseen_only":
        raise ValueError("嵌套validation v3只允许pseudo-unseen inner batch。")
    if set(config["inputs"]) != set(INPUT_KEYS):
        raise ValueError("validation输入字段不完整或包含非开发数据。")
    if set(config["expected_sha256"]) != set(INPUT_KEYS):
        raise ValueError("validation输入SHA字段不完整。")
    return config, sha256_file(path)


def resolve_and_verify_inputs(config: dict) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {name: h1.repo_path(config["inputs"][name]) for name in INPUT_KEYS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少标准validation输入：" + ", ".join(missing))
    actual = {name: sha256_file(path) for name, path in paths.items()}
    mismatch = [
        name for name in INPUT_KEYS if actual[name] != config["expected_sha256"][name]
    ]
    if mismatch:
        raise ValueError("标准validation输入SHA不匹配：" + ", ".join(mismatch))
    names = sio.loadmat(paths["att_splits"], variable_names=["allclasses_names"])[
        "allclasses_names"
    ]
    serialized = json.dumps(
        [str(item[0][0]) for item in names], ensure_ascii=False, separators=(",", ":")
    )
    if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != config["class_order_sha256"]:
        raise ValueError("CUB类别顺序不匹配。")
    return paths, actual


def _per_class_accuracy(labels, predictions, classes) -> float:
    values = []
    for class_id in classes.cpu().long():
        mask = labels.cpu().long() == class_id
        if not mask.any():
            raise ValueError(f"validation缺少类别{int(class_id)}。")
        values.append((predictions.cpu().long()[mask] == class_id).float().mean())
    return float(torch.stack(values).mean())


def pseudo_unseen_fold_batch(
    labels: torch.Tensor,
    pseudo_unseen: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    pool = torch.isin(labels, pseudo_unseen).nonzero(as_tuple=False).flatten()
    if int(batch_size) <= 0 or pool.numel() < int(batch_size):
        raise ValueError("pseudo-unseen inner batch样本不足。")
    return pool[
        torch.randperm(pool.numel(), generator=generator)[: int(batch_size)]
    ]


@torch.no_grad()
def evaluate_validation(
    model,
    features: torch.Tensor,
    labels: torch.Tensor,
    split: dict,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    seenclasses = split["dev_seen_classes"].long()
    unseenclasses = split["dev_unseen_classes"].long()
    competition = torch.cat((seenclasses, unseenclasses)).sort().values
    prototypes = model.prototypes()

    def predict(positions: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        selected = prototypes.index_select(0, class_ids.to(device))
        logits = (
            F.normalize(features.index_select(0, positions).to(device).float(), dim=-1)
            @ selected.T
            * model.scale()
        )
        return class_ids[logits.argmax(dim=1).cpu()]

    seen_positions = split["val_seen_positions"].long()
    unseen_positions = split["val_unseen_positions"].long()
    seen_labels = labels.index_select(0, seen_positions)
    unseen_labels = labels.index_select(0, unseen_positions)
    seen_prediction = predict(seen_positions, competition)
    unseen_prediction = predict(unseen_positions, competition)
    zsl_prediction = predict(unseen_positions, unseenclasses)
    seen = _per_class_accuracy(seen_labels, seen_prediction, seenclasses)
    unseen = _per_class_accuracy(unseen_labels, unseen_prediction, unseenclasses)
    zsl = _per_class_accuracy(unseen_labels, zsl_prediction, unseenclasses)
    harmonic = 2.0 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    return {"U": unseen * 100, "S": seen * 100, "H": harmonic * 100, "ZS": zsl * 100}


def _diagnostics(model) -> dict[str, float]:
    if isinstance(model, ExpertAttributeUnifiedModel):
        return model.diagnostics()
    return model.diagnostics()


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
        raise RuntimeError("标准validation训练要求可见CUDA。")
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
        )
        labels = torch.load(paths["train_labels"], map_location="cpu", weights_only=True).long()
        split = torch.load(paths["validation_split"], map_location="cpu", weights_only=True)
        seenclasses = split["dev_seen_classes"].long()
        unseenclasses = split["dev_unseen_classes"].long()
        activeclasses = torch.cat((seenclasses, unseenclasses)).sort().values
        fit_positions = split["fit_positions"].long()
        if seenclasses.numel() != 100 or unseenclasses.numel() != 50:
            raise ValueError("开发划分必须是100/50类别。")
        if torch.isin(seenclasses, unseenclasses).any():
            raise ValueError("开发seen与validation-unseen类别重叠。")
        fit_labels = labels.index_select(0, fit_positions)
        if not torch.equal(torch.unique(fit_labels, sorted=True), seenclasses):
            raise ValueError("梯度样本必须只覆盖100个开发seen类。")
        centroids = h1.visual_centroids(
            features.index_select(0, fit_positions), fit_labels, seenclasses
        )
        text_model = UnifiedSeenPrototypeModel(
            sentence_embeds,
            seenclasses,
            centroids,
            active_classes=activeclasses,
            dropout=float(config["dropout"]),
            inner_ratio=float(config["inner_ratio"]),
            outer_ratio=float(config["outer_ratio"]),
            temperature=float(config["temperature"]),
            transport_hidden_dim=int(config["transport_hidden_dim"]),
            generator_hidden_dim=int(config["generator_hidden_dim"]),
            max_transport_step=float(config["max_transport_step"]),
            max_generator_magnitude=float(config["max_generator_magnitude"]),
        )
        if config["condition_id"] == "EXPERT":
            attribute_mat = sio.loadmat(paths["att_splits"], variable_names=["att"])["att"]
            attributes = torch.from_numpy(attribute_mat.T).float()
            model = ExpertAttributeUnifiedModel(
                text_model,
                attributes,
                max_attribute_residual=float(config["max_attribute_residual"]),
            )
        else:
            model = text_model
        model = model.to(device)
        nested = config["schema_version"] in (
            "gzsl-paper.standard-nested-validation.v1",
            "gzsl-paper.standard-nested-validation.v2",
            "gzsl-paper.standard-nested-validation.v3",
        )
        detach_inner_tg = (
            config["schema_version"] in (
                "gzsl-paper.standard-nested-validation.v2",
                "gzsl-paper.standard-nested-validation.v3",
            )
        )
        inner_folds = []
        inner_fold_models = []
        inner_generators = []
        inner_manifest = None
        if nested:
            if config["condition_id"] != "NESTED-THREE-MODULE":
                raise ValueError("嵌套validation只允许无专家三模块条件。")
            class_sample_counts = torch.stack(
                [fit_labels.eq(class_id).sum() for class_id in seenclasses]
            )
            inner_folds = semantic_balanced_class_folds(
                seenclasses,
                sentence_embeds,
                class_sample_counts,
                fold_count=int(config["inner_fold_count"]),
            )
            actual_fold_sha = class_fold_sha256(inner_folds)
            if actual_fold_sha != config["inner_fold_sha256"]:
                raise ValueError("嵌套validation内层类别折SHA不匹配。")
            inner_manifest = {
                "method": config["inner_fold_method"],
                "sha256": actual_fold_sha,
                "pseudo_unseen_classes": [
                    [int(value) for value in pseudo_unseen]
                    for _, pseudo_unseen in inner_folds
                ],
            }
            for fold_id, (pseudo_seen, _) in enumerate(inner_folds):
                pseudo_seen_mask = torch.isin(fit_labels, pseudo_seen)
                pseudo_seen_positions = fit_positions[pseudo_seen_mask]
                pseudo_seen_labels = fit_labels[pseudo_seen_mask]
                pseudo_seen_centroids = h1.visual_centroids(
                    features.index_select(0, pseudo_seen_positions),
                    pseudo_seen_labels,
                    pseudo_seen,
                )
                fold_model = text_model.shared_fold_tg_vpr(
                    pseudo_seen,
                    pseudo_seen_centroids.to(device),
                ).to(device)
                inner_fold_models.append(fold_model)
                inner_generators.append(
                    torch.Generator(device="cpu").manual_seed(
                        seed * 100003 + fold_id
                    )
                )
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
        global_to_seen = torch.full((200,), -1, dtype=torch.long)
        global_to_seen[seenclasses] = torch.arange(seenclasses.numel())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        best_h = float("-inf")
        best_epoch = None
        best_state = None
        best_metrics = None
        history = []
        print(f"实验：{config['experiment_id']} 条件：{config['condition_id']}")
        print(f"代码commit：{code_commit} 配置SHA：{config_sha}")
        print(
            f"开发梯度图像={fit_positions.numel()} val-seen={split['val_seen_positions'].numel()} "
            f"val-unseen={split['val_unseen_positions'].numel()} official-test-loaded=false"
        )
        if nested:
            print(
                f"内层三折scope=outer-train-only fold_sha={inner_manifest['sha256']} "
                f"sizes={[len(values) for values in inner_manifest['pseudo_unseen_classes']]}"
            )

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
            for fold_model in inner_fold_models:
                fold_model.train()
            loss_sum = main_ce_sum = topology_sum = episode_sum = 0.0
            sample_count = 0
            inner_episode_sample_count = 0
            for relative_indices in full_epoch_batches(
                fit_positions.numel(), int(config["batch_size"]), generator
            ):
                positions = fit_positions.index_select(0, relative_indices)
                images = features.index_select(0, positions).to(device).float()
                targets = global_to_seen[labels.index_select(0, positions)].to(device)
                optimizer.zero_grad(set_to_none=True)
                ce = F.cross_entropy(model.logits(images, seenclasses), targets)
                topology = model.topology_loss()
                loss = ce + float(config["topology_weight"]) * topology
                episode_loss = ce.new_zeros(())
                if nested:
                    fold_losses = []
                    for fold_id, ((pseudo_seen, pseudo_unseen), fold_model) in enumerate(
                        zip(inner_folds, inner_fold_models)
                    ):
                        if config.get("inner_batch_mode") == "pseudo_unseen_only":
                            inner_relative = pseudo_unseen_fold_batch(
                                fit_labels,
                                pseudo_unseen,
                                2 * int(config["inner_batch_half"]),
                                inner_generators[fold_id],
                            )
                        else:
                            inner_relative = balanced_fold_batch(
                                fit_labels,
                                pseudo_seen,
                                pseudo_unseen,
                                int(config["inner_batch_half"]),
                                inner_generators[fold_id],
                            )
                        inner_positions = fit_positions.index_select(0, inner_relative)
                        inner_images = features.index_select(
                            0, inner_positions
                        ).to(device).float()
                        inner_targets = global_to_seen[
                            labels.index_select(0, inner_positions)
                        ].to(device)
                        fold_stages = text_model.prototype_stages_from_tg(
                            fold_model,
                            pseudo_seen.to(device),
                            detach_tg_inputs=detach_inner_tg,
                        )
                        competition = fold_stages["final"].index_select(
                            0, seenclasses.to(device)
                        )
                        inner_logits = (
                            F.normalize(inner_images, dim=-1)
                            @ competition.T
                            * (
                                text_model.scale().detach()
                                if detach_inner_tg
                                else text_model.scale()
                            )
                        )
                        fold_losses.append(
                            F.cross_entropy(inner_logits, inner_targets)
                        )
                        inner_episode_sample_count += inner_images.size(0)
                    episode_loss = torch.stack(fold_losses).mean()
                    loss = loss + float(config["inner_episode_weight"]) * episode_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("validation训练loss包含NaN/Inf。")
                loss.backward()
                require_finite_gradients(model)
                optimizer.step()
                loss_sum += float(loss.detach()) * images.size(0)
                main_ce_sum += float(ce.detach()) * images.size(0)
                topology_sum += float(topology.detach()) * images.size(0)
                episode_sum += float(episode_loss.detach()) * images.size(0)
                sample_count += images.size(0)
            if sample_count != fit_positions.numel():
                raise RuntimeError("每个epoch必须完整且唯一遍历开发梯度图像。")
            scheduler.step()
            metrics = evaluate_validation(model, features, labels, split, device)
            diagnostics = _diagnostics(model)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                "train_main_ce": main_ce_sum / sample_count,
                "train_topology": topology_sum / sample_count,
                "train_inner_episode_ce": episode_sum / sample_count,
                "sample_count": sample_count,
                "unique_sample_count": sample_count,
                "inner_episode_sample_count": inner_episode_sample_count,
                "validation_metrics_percent": metrics,
                "diagnostics": diagnostics,
            }
            history.append(row)
            print(
                f"epoch={epoch} samples={sample_count} loss={row['train_loss']:.6f} "
                f"val_U={metrics['U']:.6f} val_S={metrics['S']:.6f} val_H={metrics['H']:.6f}"
            )
            if metrics["H"] > best_h:
                best_h = metrics["H"]
                best_epoch = epoch
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())

        require_finite_model(model)
        final_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state, strict=True)
        checkpoint = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "run_id": run_id,
            "code_commit": code_commit,
            "config": config,
            "config_sha256": config_sha,
            "seed": seed,
            "selected_epoch": best_epoch,
            "best_validation_metrics_percent": best_metrics,
            "model_state_dict": best_state,
            "final_epoch_model_state_dict": final_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "inner_fold_manifest": inner_manifest,
            "reproducibility": reproducibility,
        }
        atomic_torch_save(output_dir / "model_best.pth", checkpoint)
        atomic_torch_save(output_dir / "checkpoint_last.pth", checkpoint)
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        atomic_write_json(
            output_dir / "metrics.json",
            {
                "experiment_id": config["experiment_id"],
                "condition_id": config["condition_id"],
                "run_id": run_id,
                "framework_id": config["framework_id"],
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "validation_used_for_selection": True,
                "test_used_for_selection": False,
                "official_test_loaded": False,
                "validation_images_used_for_gradient": False,
                "inner_pseudo_unseen_images_used_for_gradient": bool(nested),
                "expert_attributes_used": bool(config["expert_attributes_used"]),
                "feature_backbone": config["feature_backbone"],
                "feature_provenance_complete": False,
                "historical_test_informed_architecture": True,
                "final_test_eligible": False,
                "code_commit": code_commit,
                "config_sha256": config_sha,
                "seed": seed,
                "selected_epoch": best_epoch,
                "fit_images_per_epoch": int(fit_positions.numel()),
                "inner_episode_samples_per_epoch": (
                    int(history[-1]["inner_episode_sample_count"])
                    if nested
                    else 0
                ),
                "inner_fold_manifest": inner_manifest,
                "inner_gradient_scope": (
                    config.get("inner_gradient_scope", "all_three_modules")
                    if nested
                    else None
                ),
                "inner_batch_mode": (
                    config.get("inner_batch_mode", "balanced_pseudo_seen_unseen")
                    if nested
                    else None
                ),
                "best_validation_metrics_percent": best_metrics,
                "diagnostics": _diagnostics(model),
                "model_sha256": sha256_file(output_dir / "model_best.pth"),
                "checkpoint_last_sha256": sha256_file(
                    output_dir / "checkpoint_last.pth"
                ),
            },
        )
        print(
            "best_epoch={} val_U={U:.6f}% val_S={S:.6f}% val_H={H:.6f}% val_ZS={ZS:.6f}%".format(
                best_epoch, **best_metrics
            )
        )
        return best_metrics
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
