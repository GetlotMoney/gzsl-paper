"""Unified three-dataset Chen-style runner for the final FRAMEWORK-V2 paper matrix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.paper_v2 import (
    CCGR_MODES,
    RGVE_MODES,
    TG_MODES,
    TRANSPORT_MODES,
    PaperV2RGVEModel,
    PaperV2ThreeModuleModel,
)
from model.tg_vpr_h1 import train as h1
from model.visual_evidence import VISUAL_MODES, PaperV2VisualModel
from tools.gzsl_data import evaluate_prototypes, per_class_accuracy
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


EVALUATION_PROTOCOL = "chen_shiming_code_aligned_multidataset_test_selected_gzsl"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "condition_id",
    "framework_id",
    "dataset",
    "asset_manifest",
    "asset_manifest_sha256",
    "evaluation_protocol",
    "test_used_for_selection",
    "test_used_for_hyperparameter_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "training_strategy",
    "selection_scope",
    "nested_official_test_selection",
    "device",
    "random_seed",
    "batch_size",
    "nominal_epochs",
    "optimizer",
    "weight_decay",
    "end_to_end_learning_rate",
    "stage1_learning_rate",
    "stage2_learning_rate",
    "stage3_learning_rate",
    "tg_vpr_mode",
    "transport_mode",
    "ccgr_mode",
    "dropout",
    "inner_ratio",
    "outer_ratio",
    "topology_weight",
    "temperature",
    "transport_hidden_dim",
    "generator_hidden_dim",
    "max_transport_step",
    "max_ntr_delta",
    "max_generator_magnitude",
}
RGVE_CONFIG_KEYS = CONFIG_KEYS | {
    "eval_batch_size",
    "rgve_mode",
    "rgve_hidden_dim",
    "rgve_max_beta",
    "rgve_initial_temperature",
    "rgve_role_weight",
    "rgve_balance_weight",
    "rgve_calibration_weight",
    "rgve_role_margin",
}
VISUAL_CONFIG_KEYS = CONFIG_KEYS | {
    "eval_batch_size",
    "visual_mode",
    "visual_hidden_dim",
    "visual_max_beta",
    "visual_part_weight",
    "visual_diversity_weight",
    "visual_anchor_weight",
    "visual_hard_weight",
    "visual_hard_margin",
    "visual_lr_multiplier",
    "confusion_topk",
    "visual_scales",
}
VISUAL_V2_CONFIG_KEYS = VISUAL_CONFIG_KEYS | {"patch_cache_mode"}
ASSET_FILES = (
    "train_features.pt",
    "train_labels.pt",
    "test_seen_features.pt",
    "test_seen_labels.pt",
    "test_unseen_features.pt",
    "test_unseen_labels.pt",
    "class_name_embeds.pt",
    "role_sentence_embeds.pt",
)
PATCH_ASSET_FILES = (
    "train_patch_features.npy",
    "test_seen_patch_features.npy",
    "test_unseen_patch_features.npy",
)


def load_config(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"最终论文RUN配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    schema = config.get("schema_version") if isinstance(config, dict) else None
    expected_keys = {
        "gzsl-paper.paper-v2-rgve-run.v1": RGVE_CONFIG_KEYS,
        "gzsl-paper.paper-v2-visual-run.v1": VISUAL_CONFIG_KEYS,
        "gzsl-paper.paper-v2-visual-run.v2": VISUAL_V2_CONFIG_KEYS,
    }.get(schema, CONFIG_KEYS)
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"最终论文RUN配置字段错误；缺少={sorted(expected_keys-actual)}，多出={sorted(actual-expected_keys)}。"
        )
    if schema not in (
        "gzsl-paper.paper-v2-run.v1",
        "gzsl-paper.paper-v2-rgve-run.v1",
        "gzsl-paper.paper-v2-visual-run.v1",
        "gzsl-paper.paper-v2-visual-run.v2",
    ):
        raise ValueError("最终论文RUN schema错误。")
    if config["framework_id"] != "FRAMEWORK-V2" or config["dataset"] not in ("CUB", "AWA2", "SUN"):
        raise ValueError("最终论文RUN只接受FRAMEWORK-V2和CUB/AWA2/SUN。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("最终论文RUN评估协议身份错误。")
    if config["unseen_images_used_for_gradient"] is not False or config["strict_blind_claim"] is not False:
        raise ValueError("必须披露true-unseen无梯度且不作blind-test声明。")
    if config["nested_official_test_selection"] is not False:
        raise ValueError("分阶段RUN禁止嵌套official-test选模。")
    if config["selection_scope"] != "whole_run_whole_model_only":
        raise ValueError("只允许整次RUN的整模型全局H选模。")
    if config["training_strategy"] not in (
        "no_training",
        "end_to_end_joint",
        "stagewise_50_100_50",
        "modulewise_50_50_50_50",
        "modulewise_short_v5_joint150",
        "modulewise_short_v10_joint150",
    ):
        raise ValueError("未知训练策略。")
    no_training = config["training_strategy"] == "no_training"
    if no_training:
        if config["condition_id"] not in ("B0_PURE_CLIP", "B1_MEAN8"):
            raise ValueError("no_training只允许明确的B0_PURE_CLIP或B1_MEAN8原型来源。")
        if config["test_used_for_selection"] is not False:
            raise ValueError("no_training条件没有checkpoint选择。")
    elif config["test_used_for_selection"] is not True:
        raise ValueError("训练RUN固定使用official test H选checkpoint。")
    if int(config["batch_size"]) != 50:
        raise ValueError("Chen-style固定batch=50。")
    short_joint = config["training_strategy"] in (
        "modulewise_short_v5_joint150",
        "modulewise_short_v10_joint150",
    )
    expected_epochs = 150 if short_joint else 200
    if int(config["nominal_epochs"]) != expected_epochs:
        raise ValueError(f"当前训练策略固定{expected_epochs}名义epoch。")
    if config["optimizer"] != "Adam" or float(config["weight_decay"]) != 1e-4:
        raise ValueError("最终论文RUN固定Adam和weight_decay=1e-4。")
    if config["tg_vpr_mode"] not in TG_MODES:
        raise ValueError("tg_vpr_mode错误。")
    if config["transport_mode"] not in TRANSPORT_MODES:
        raise ValueError("transport_mode错误。")
    if config["ccgr_mode"] not in CCGR_MODES:
        raise ValueError("ccgr_mode错误。")
    if schema == "gzsl-paper.paper-v2-rgve-run.v1":
        if config["dataset"] != "CUB" or config["rgve_mode"] not in RGVE_MODES:
            raise ValueError("当前正式RGVE只接受CUB及已注册模式。")
        if int(config["eval_batch_size"]) <= 0 or int(config["rgve_hidden_dim"]) <= 0:
            raise ValueError("RGVE batch和hidden_dim必须为正数。")
        if float(config["rgve_max_beta"]) <= 0:
            raise ValueError("RGVE max_beta必须为正数。")
        for key in ("rgve_role_weight", "rgve_balance_weight", "rgve_calibration_weight"):
            if float(config[key]) < 0:
                raise ValueError(f"{key}不能为负数。")
    if schema in (
        "gzsl-paper.paper-v2-visual-run.v1",
        "gzsl-paper.paper-v2-visual-run.v2",
    ):
        if no_training or config["visual_mode"] not in VISUAL_MODES:
            raise ValueError("视觉筛选只接受已注册模式和正式训练策略。")
        if int(config["eval_batch_size"]) <= 0 or int(config["visual_hidden_dim"]) <= 0:
            raise ValueError("视觉eval batch和hidden_dim必须为正数。")
        if float(config["visual_max_beta"]) <= 0 or float(config["visual_lr_multiplier"]) <= 0:
            raise ValueError("视觉beta与学习率倍率必须为正数。")
        for key in (
            "visual_part_weight",
            "visual_diversity_weight",
            "visual_anchor_weight",
            "visual_hard_weight",
            "visual_hard_margin",
        ):
            if float(config[key]) < 0:
                raise ValueError(f"{key}不能为负数。")
        if int(config["confusion_topk"]) != 5 or list(config["visual_scales"]) != [24, 12, 6]:
            raise ValueError("视觉初筛固定topk=5及scales=[24,12,6]。")
        if float(config["topology_weight"]) != 0.1:
            raise ValueError("视觉初筛固定topology_weight=0.1。")
        if schema == "gzsl-paper.paper-v2-visual-run.v2":
            if config["patch_cache_mode"] not in ("none", "gpu_fp16"):
                raise ValueError("patch_cache_mode只允许none或gpu_fp16。")
            if config["visual_mode"] == "off" and config["patch_cache_mode"] != "none":
                raise ValueError("视觉off禁止无用patch缓存。")
            if config["visual_mode"] != "off" and config["patch_cache_mode"] != "gpu_fp16":
                raise ValueError("视觉on固定使用gpu_fp16常驻缓存。")
    return config, sha256_file(path)


def load_assets(config: dict) -> tuple[dict, dict, Path]:
    manifest_path = Path(config["asset_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"资产manifest不存在：{manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["asset_manifest_sha256"]:
        raise ValueError("资产manifest SHA不匹配。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_schema = manifest.get("schema_version")
    allowed_schemas = {"gzsl-paper.clip-assets.v1"}
    if config["schema_version"] in (
        "gzsl-paper.paper-v2-rgve-run.v1",
        "gzsl-paper.paper-v2-visual-run.v1",
        "gzsl-paper.paper-v2-visual-run.v2",
    ):
        allowed_schemas = {"gzsl-paper.rgve-local-patch-assets.v1"}
    if manifest_schema not in allowed_schemas or manifest.get("dataset") != config["dataset"]:
        raise ValueError("资产manifest身份错误。")
    expected_outputs = manifest.get("outputs_sha256", {})
    required_files = ASSET_FILES + (PATCH_ASSET_FILES if manifest_schema == "gzsl-paper.rgve-local-patch-assets.v1" else ())
    if not set(required_files).issubset(expected_outputs):
        raise ValueError("资产manifest缺少训练或评估缓存。")
    tensors = {}
    for filename in required_files:
        path = manifest_path.parent / filename
        if not path.is_file() or sha256_file(path) != expected_outputs[filename]:
            raise ValueError(f"资产文件缺失或SHA错误：{filename}")
        key = filename.removesuffix(".pt").removesuffix(".npy")
        tensors[key] = (
            np.load(path, mmap_mode="r")
            if filename.endswith(".npy")
            else torch.load(path, map_location="cpu", weights_only=True)
        )
    manifest = dict(manifest)
    class_count = int(manifest.get("class_count", tensors["role_sentence_embeds"].size(0)))
    counts = manifest.get("counts", {})
    manifest.setdefault("class_count", class_count)
    manifest.setdefault("train_count", int(counts.get("train", tensors["train_labels"].numel())))
    manifest.setdefault("test_seen_count", int(counts.get("test_seen", tensors["test_seen_labels"].numel())))
    manifest.setdefault("test_unseen_count", int(counts.get("test_unseen", tensors["test_unseen_labels"].numel())))
    seen_classes = torch.unique(tensors["train_labels"].long(), sorted=True).tolist()
    manifest.setdefault("seen_classes", seen_classes)
    manifest.setdefault("unseen_classes", [index for index in range(class_count) if index not in set(seen_classes)])
    manifest.setdefault("asset_id", manifest_path.parent.name)
    expected_shapes = {
        "train_features": (int(manifest["train_count"]), 768),
        "train_labels": (int(manifest["train_count"]),),
        "test_seen_features": (int(manifest["test_seen_count"]), 768),
        "test_seen_labels": (int(manifest["test_seen_count"]),),
        "test_unseen_features": (int(manifest["test_unseen_count"]), 768),
        "test_unseen_labels": (int(manifest["test_unseen_count"]),),
        "class_name_embeds": (class_count, 768),
        "role_sentence_embeds": (class_count, 8, 768),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"资产{name}形状错误：{tuple(tensors[name].shape)} != {shape}")
    if manifest_schema == "gzsl-paper.rgve-local-patch-assets.v1":
        patch_shape = tuple(int(value) for value in manifest["patch_shape"])
        for split, count in (
            ("train", manifest["train_count"]),
            ("test_seen", manifest["test_seen_count"]),
            ("test_unseen", manifest["test_unseen_count"]),
        ):
            actual_shape = tuple(tensors[f"{split}_patch_features"].shape)
            expected_shape = (int(count), *patch_shape)
            if actual_shape != expected_shape:
                raise ValueError(f"资产{split}_patch_features形状错误：{actual_shape} != {expected_shape}")
    return tensors, manifest, manifest_path


def random_batch_indices(count: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(int(count), generator=generator)[: int(batch_size)]


def report_interval_for_run(niters: int, nominal_epochs: int) -> int:
    if int(niters) <= 0 or int(nominal_epochs) <= 0:
        raise ValueError("niters和nominal_epochs必须为正数。")
    interval = int(niters) // int(nominal_epochs)
    if interval <= 0:
        raise ValueError("report_interval必须为正数。")
    return interval


def build_three_module_model(
    config: dict,
    tensors: dict,
    manifest: dict,
    device: torch.device,
    *,
    dropout_override: float | None = None,
) -> PaperV2ThreeModuleModel:
    seen_classes = torch.tensor(manifest["seen_classes"], dtype=torch.long)
    centroids = h1.visual_centroids(
        tensors["train_features"], tensors["train_labels"].long(), seen_classes
    )
    return PaperV2ThreeModuleModel(
        tensors["role_sentence_embeds"],
        seen_classes,
        centroids,
        tg_vpr_mode=config["tg_vpr_mode"],
        transport_mode=config["transport_mode"],
        ccgr_mode=config["ccgr_mode"],
        dropout=(float(config["dropout"]) if dropout_override is None else float(dropout_override)),
        inner_ratio=float(config["inner_ratio"]),
        outer_ratio=float(config["outer_ratio"]),
        temperature=float(config["temperature"]),
        transport_hidden_dim=int(config["transport_hidden_dim"]),
        generator_hidden_dim=int(config["generator_hidden_dim"]),
        max_transport_step=float(config["max_transport_step"]),
        max_ntr_delta=float(config["max_ntr_delta"]),
        max_generator_magnitude=float(config["max_generator_magnitude"]),
    ).to(device)


def build_run_model(
    config: dict,
    tensors: dict,
    manifest: dict,
    device: torch.device,
) -> PaperV2ThreeModuleModel | PaperV2RGVEModel | PaperV2VisualModel:
    parent = build_three_module_model(config, tensors, manifest, device)
    if config["schema_version"] == "gzsl-paper.paper-v2-rgve-run.v1":
        return PaperV2RGVEModel(
            parent,
            rgve_mode=config["rgve_mode"],
            hidden_dim=int(config["rgve_hidden_dim"]),
            max_beta=float(config["rgve_max_beta"]),
            initial_temperature=float(config["rgve_initial_temperature"]),
        ).to(device)
    if config["schema_version"] in (
        "gzsl-paper.paper-v2-visual-run.v1",
        "gzsl-paper.paper-v2-visual-run.v2",
    ):
        return PaperV2VisualModel(
            parent,
            visual_mode=config["visual_mode"],
            hidden_dim=int(config["visual_hidden_dim"]),
            max_beta=float(config["visual_max_beta"]),
            confusion_topk=int(config["confusion_topk"]),
            visual_scales=tuple(int(value) for value in config["visual_scales"]),
        ).to(device)
    return parent


def stage_for_iteration(ntrain: int, iteration: int) -> tuple[str, float]:
    if 0 <= iteration < ntrain:
        return "TG_ONLY", 0.25
    if ntrain <= iteration < 3 * ntrain:
        return "TRANSFER_CCGR", 0.50
    if 3 * ntrain <= iteration < 4 * ntrain:
        return "JOINT_FINETUNE", 0.25
    raise ValueError("iteration不属于50/100/50阶段。")


def modulewise_stage_for_iteration(ntrain: int, iteration: int) -> tuple[str, float]:
    if 0 <= iteration < ntrain:
        return "TG_ONLY", 0.25
    if ntrain <= iteration < 2 * ntrain:
        return "TST_NTR_ONLY", 0.25
    if 2 * ntrain <= iteration < 3 * ntrain:
        return "CCGR_ONLY", 0.25
    if 3 * ntrain <= iteration < 4 * ntrain:
        return "VISUAL_ONLY", 0.25
    raise ValueError("iteration不属于50/50/50/50模块式阶段。")


def short_modulewise_stage_for_iteration(
    ntrain: int,
    iteration: int,
    visual_nominal_epochs: int,
) -> tuple[str, float]:
    if int(visual_nominal_epochs) not in (5, 10):
        raise ValueError("短模块式Visual阶段只允许5或10名义epoch。")
    report_interval = (3 * int(ntrain)) // 150
    short = int(visual_nominal_epochs) * report_interval
    tg_end = int(ntrain)
    tst_end = tg_end + 5 * report_interval
    ccgr_end = tst_end + 5 * report_interval
    visual_end = ccgr_end + short
    if 0 <= iteration < tg_end:
        return "TG_ONLY", 0.25
    if tg_end <= iteration < tst_end:
        return "TST_NTR_ONLY", 5 / 200
    if tst_end <= iteration < ccgr_end:
        return "CCGR_ONLY", 5 / 200
    if ccgr_end <= iteration < visual_end:
        return "VISUAL_ONLY", int(visual_nominal_epochs) / 200
    if visual_end <= iteration < 3 * int(ntrain):
        return "JOINT_FINETUNE", (3 * int(ntrain) - visual_end) / (3 * int(ntrain))
    raise ValueError("iteration不属于短模块式+Joint阶段。")


def _active_groups(
    model: PaperV2ThreeModuleModel | PaperV2RGVEModel | PaperV2VisualModel,
    strategy: str,
    stage: str,
) -> list[str]:
    groups = model.parameter_groups()
    nonempty = {name for name, parameters in groups.items() if parameters}
    if strategy == "end_to_end_joint":
        return sorted(nonempty)
    if stage == "TG_ONLY":
        selected = {"tg_vpr"}
    elif stage == "TRANSFER_CCGR":
        selected = {"transport", "ntr", "ccgr_class", "ccgr_shared", "rgve", "visual"}
        if not (selected & nonempty):
            selected = {"tg_vpr"}
    elif stage == "JOINT_FINETUNE":
        selected = nonempty
    elif stage == "TST_NTR_ONLY":
        selected = {"transport", "ntr"}
    elif stage == "CCGR_ONLY":
        selected = {"ccgr_class", "ccgr_shared"}
    elif stage == "VISUAL_ONLY":
        selected = {"visual"}
    else:
        raise ValueError(f"未知阶段：{stage}")
    return sorted(selected & nonempty)


def set_trainable(
    model: PaperV2ThreeModuleModel | PaperV2RGVEModel | PaperV2VisualModel,
    group_names: list[str],
) -> list[torch.nn.Parameter]:
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = model.parameter_groups()
    active = []
    seen_ids = set()
    for name in group_names:
        for parameter in groups[name]:
            if id(parameter) not in seen_ids:
                parameter.requires_grad_(True)
                active.append(parameter)
                seen_ids.add(id(parameter))
    if not active:
        raise ValueError("当前训练阶段没有可训练参数。")
    return active


def _gradient_norms(
    model: PaperV2ThreeModuleModel | PaperV2RGVEModel | PaperV2VisualModel,
) -> dict[str, float]:
    result = {}
    for name, parameters in model.parameter_groups().items():
        values = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
        result[name] = float(torch.stack(values).norm()) if values else 0.0
    return result


def _load_patch_batch(memmap, indices: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(memmap, torch.Tensor):
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(np.asarray(indices, dtype=np.int64))
        return memmap.index_select(0, indices.to(memmap.device).long()).to(dtype=torch.float32)
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    values = np.asarray(memmap[np.asarray(indices, dtype=np.int64)])
    return torch.from_numpy(values.copy()).to(device=device, dtype=torch.float32)


def _cache_visual_patches(tensors: dict, config: dict, device: torch.device) -> None:
    if config.get("patch_cache_mode") != "gpu_fp16":
        return
    for split in ("train", "test_seen", "test_unseen"):
        key = f"{split}_patch_features"
        source = tensors[key]
        if isinstance(source, torch.Tensor):
            raise ValueError("patch缓存不得重复物化。")
        cpu_view = torch.from_numpy(np.asarray(source))
        tensors[key] = cpu_view.to(device=device, dtype=torch.float16)


@torch.no_grad()
def _evaluate_rgve_model(
    model: PaperV2RGVEModel | PaperV2VisualModel,
    tensors: dict,
    manifest: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    seen = torch.tensor(manifest["seen_classes"], dtype=torch.long)
    unseen = torch.tensor(manifest["unseen_classes"], dtype=torch.long)

    def predict(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        features = tensors[f"{split}_features"]
        patches = tensors[f"{split}_patch_features"]
        all_predictions = []
        zsl_predictions = []
        for start in range(0, len(features), int(batch_size)):
            end = min(start + int(batch_size), len(features))
            indices = np.arange(start, end)
            logits = model.logits(
                features[start:end].to(device).float(),
                _load_patch_batch(patches, indices, device),
            )
            all_predictions.append(logits.argmax(dim=1).cpu())
            unseen_logits = logits.index_select(1, unseen.to(device))
            zsl_predictions.append(unseen[unseen_logits.argmax(dim=1).cpu()])
        return torch.cat(all_predictions), torch.cat(zsl_predictions)

    seen_all, _ = predict("test_seen")
    unseen_all, unseen_zsl = predict("test_unseen")
    seen_labels = tensors["test_seen_labels"].long()
    unseen_labels = tensors["test_unseen_labels"].long()
    s = per_class_accuracy(seen_labels, seen_all, seen)
    u = per_class_accuracy(unseen_labels, unseen_all, unseen)
    z = per_class_accuracy(unseen_labels, unseen_zsl, unseen)
    h = 2 * s * u / (s + u) if s + u else 0.0
    return {"U": 100 * u, "S": 100 * s, "H": 100 * h, "ZS": 100 * z}


def _evaluate_model(model, tensors, manifest, device, eval_batch_size: int = 64) -> dict[str, float]:
    if isinstance(model, PaperV2VisualModel) and model.visual_mode == "off":
        model.eval()
        return evaluate_prototypes(
            model.prototypes(),
            model.scale(),
            tensors["test_seen_features"],
            tensors["test_seen_labels"],
            tensors["test_unseen_features"],
            tensors["test_unseen_labels"],
            torch.tensor(manifest["seen_classes"]),
            torch.tensor(manifest["unseen_classes"]),
            device=device,
        )
    if isinstance(model, (PaperV2RGVEModel, PaperV2VisualModel)):
        return _evaluate_rgve_model(model, tensors, manifest, device, eval_batch_size)
    if isinstance(model, PaperV2ThreeModuleModel):
        model.eval()
        prototypes = model.prototypes()
        scale = model.scale()
    else:
        prototypes, scale = model
    return evaluate_prototypes(
        prototypes,
        scale,
        tensors["test_seen_features"],
        tensors["test_seen_labels"],
        tensors["test_unseen_features"],
        tensors["test_unseen_labels"],
        torch.tensor(manifest["seen_classes"]),
        torch.tensor(manifest["unseen_classes"]),
        device=device,
    )


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    config, config_sha = load_config(config_path)
    tensors, manifest, manifest_path = load_assets(config)
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = h1.TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(seed, strict_determinism=True, deterministic_warn_only=False)
        device = torch.device(config["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("正式论文RUN要求CUDA。")
        if config["schema_version"] == "gzsl-paper.paper-v2-visual-run.v2":
            _cache_visual_patches(tensors, config, device)
        seen_classes = torch.tensor(manifest["seen_classes"], dtype=torch.long)
        train_labels = tensors["train_labels"].long()
        if not torch.equal(torch.unique(train_labels, sorted=True), seen_classes):
            raise ValueError("训练缓存seen类别与资产manifest不一致。")

        if config["training_strategy"] == "no_training":
            if config["condition_id"] == "B0_PURE_CLIP":
                prototypes = tensors["class_name_embeds"].to(device)
                scale = torch.tensor(1.0 / float(config["temperature"]), device=device)
            elif config["condition_id"] == "B1_MEAN8":
                prototypes = F.normalize(tensors["role_sentence_embeds"].mean(dim=1), dim=-1).to(device)
                scale = torch.tensor(1.0 / float(config["temperature"]), device=device)
            else:
                frozen_model = build_three_module_model(
                    config, tensors, manifest, device, dropout_override=0.0
                ).eval()
                if any(frozen_model.parameter_groups().values()):
                    raise ValueError("no_training内部消融仍含活动模块参数。")
                prototypes = frozen_model.prototypes()
                scale = frozen_model.scale()
            metrics = _evaluate_model((prototypes, scale), tensors, manifest, device)
            state = {
                "experiment_id": config["experiment_id"],
                "condition_id": config["condition_id"],
                "run_id": run_id,
                "code_commit": code_commit,
                "config": config,
                "config_sha256": config_sha,
                "best_metrics_percent": metrics,
                "prototypes": prototypes.cpu(),
                "scale": scale.cpu(),
            }
            atomic_torch_save(output_dir / "model_best.pth", state)
            atomic_torch_save(output_dir / "checkpoint_last.pth", state)
            history = [{"iteration": None, "nominal_epoch": None, "stage": "NO_TRAINING", "official_metrics_percent": metrics}]
            selected = {"iteration": None, "epoch": None, "stage": "NO_TRAINING", "metrics": metrics, "diagnostics": {}}
            stage_gradient_norms = {}
        else:
            model = build_run_model(config, tensors, manifest, device)
            ntrain = int(train_labels.numel())
            niters = ntrain * int(config["nominal_epochs"]) // int(config["batch_size"])
            short_joint = config["training_strategy"] in (
                "modulewise_short_v5_joint150",
                "modulewise_short_v10_joint150",
            )
            expected_iterations = (3 if short_joint else 4) * ntrain
            if niters != expected_iterations:
                raise ValueError("名义epoch与batch50没有得到预注册的总更新数。")
            report_interval = report_interval_for_run(
                niters, int(config["nominal_epochs"])
            )
            global_to_seen = torch.full((int(manifest["class_count"]),), -1, dtype=torch.long)
            global_to_seen[seen_classes] = torch.arange(seen_classes.numel())
            generator = torch.Generator(device="cpu").manual_seed(seed)
            current_stage = None
            optimizer = None
            history = []
            best_h = float("-inf")
            best_state = None
            selected = None
            best_zs = float("-inf")
            stage_gradient_norms = {}
            frozen_stage_anchor = None
            for iteration in range(niters):
                if config["training_strategy"] == "end_to_end_joint":
                    stage = "END_TO_END"
                    learning_rate = float(config["end_to_end_learning_rate"])
                elif config["training_strategy"] == "stagewise_50_100_50":
                    stage, _ = stage_for_iteration(ntrain, iteration)
                    learning_rate = {
                        "TG_ONLY": float(config["stage1_learning_rate"]),
                        "TRANSFER_CCGR": float(config["stage2_learning_rate"]),
                        "JOINT_FINETUNE": float(config["stage3_learning_rate"]),
                    }[stage]
                elif config["training_strategy"] == "modulewise_50_50_50_50":
                    stage, _ = modulewise_stage_for_iteration(ntrain, iteration)
                    learning_rate = {
                        "TG_ONLY": float(config["stage1_learning_rate"]),
                        "TST_NTR_ONLY": float(config["stage2_learning_rate"]),
                        "CCGR_ONLY": float(config["stage2_learning_rate"]),
                        "VISUAL_ONLY": float(config["stage2_learning_rate"]),
                    }[stage]
                else:
                    visual_epochs = (
                        5
                        if config["training_strategy"] == "modulewise_short_v5_joint150"
                        else 10
                    )
                    stage, _ = short_modulewise_stage_for_iteration(
                        ntrain, iteration, visual_epochs
                    )
                    learning_rate = {
                        "TG_ONLY": float(config["stage1_learning_rate"]),
                        "TST_NTR_ONLY": float(config["stage2_learning_rate"]),
                        "CCGR_ONLY": float(config["stage2_learning_rate"]),
                        "VISUAL_ONLY": float(config["stage2_learning_rate"]),
                        "JOINT_FINETUNE": float(config["stage3_learning_rate"]),
                    }[stage]
                stage_started = stage != current_stage
                if stage_started:
                    current_stage = stage
                    names = _active_groups(model, config["training_strategy"], stage)
                    frozen_stage_anchor = None
                    if names:
                        active = set_trainable(model, names)
                    elif (
                        isinstance(model, PaperV2VisualModel)
                        and config["training_strategy"] == "modulewise_50_50_50_50"
                        and stage == "VISUAL_ONLY"
                        and model.visual_mode == "off"
                    ):
                        model.zero_grad(set_to_none=True)
                        for parameter in model.parameters():
                            parameter.requires_grad_(False)
                        frozen_stage_anchor = torch.nn.Parameter(
                            torch.zeros((), device=device)
                        )
                        active = [frozen_stage_anchor]
                    else:
                        raise ValueError("当前模块式阶段没有合法可训练参数。")
                    if isinstance(model, PaperV2VisualModel) and "visual" in names:
                        visual_ids = {id(parameter) for parameter in model.parameter_groups()["visual"]}
                        base_active = [parameter for parameter in active if id(parameter) not in visual_ids]
                        visual_active = [parameter for parameter in active if id(parameter) in visual_ids]
                        parameter_groups = []
                        if base_active:
                            parameter_groups.append({"params": base_active, "lr": learning_rate})
                        if visual_active:
                            parameter_groups.append(
                                {
                                    "params": visual_active,
                                    "lr": learning_rate * float(config["visual_lr_multiplier"]),
                                }
                            )
                        optimizer = torch.optim.Adam(
                            parameter_groups,
                            lr=learning_rate,
                            weight_decay=float(config["weight_decay"]),
                        )
                    else:
                        optimizer = torch.optim.Adam(
                            active,
                            lr=learning_rate,
                            weight_decay=float(config["weight_decay"]),
                        )
                    print(f"stage={stage} start={iteration} lr={learning_rate} groups={names}")
                model.train()
                indices = random_batch_indices(ntrain, int(config["batch_size"]), generator)
                images = tensors["train_features"].index_select(0, indices).to(device).float()
                targets = global_to_seen[train_labels.index_select(0, indices)].to(device)
                optimizer.zero_grad(set_to_none=True)
                prototypes = model.prototypes()
                loss_role = prototypes.new_zeros(())
                loss_balance = prototypes.new_zeros(())
                loss_calibration = prototypes.new_zeros(())
                loss_part = prototypes.new_zeros(())
                loss_diversity = prototypes.new_zeros(())
                loss_anchor = prototypes.new_zeros(())
                loss_hard = prototypes.new_zeros(())
                seen_device = seen_classes.to(device)
                if isinstance(model, PaperV2VisualModel):
                    if config["visual_mode"] == "off":
                        logits = model.parent.logits(images, seen_device)
                        ce = F.cross_entropy(logits, targets)
                    else:
                        patches = _load_patch_batch(tensors["train_patch_features"], indices, device)
                        global_targets = seen_device.index_select(0, targets)
                        components = model.score_components(
                            images,
                            patches,
                            target_class_ids=global_targets,
                        )
                        final_scores = components["final_scores"]
                        assert isinstance(final_scores, torch.Tensor)
                        logits = final_scores.index_select(1, seen_device)
                        ce = F.cross_entropy(logits, targets)
                    if config["visual_mode"] != "off" and "visual" in names:
                        visual_losses = model.visual_losses(
                            components,
                            seen_device,
                            targets,
                            global_targets,
                            hard_margin=float(config["visual_hard_margin"]),
                        )
                        loss_part = visual_losses["part"]
                        loss_diversity = visual_losses["diversity"]
                        loss_anchor = visual_losses["anchor"]
                        loss_hard = visual_losses["hard"]
                elif isinstance(model, PaperV2RGVEModel):
                    patches = _load_patch_batch(tensors["train_patch_features"], indices, device)
                    components = model.score_components(images, patches)
                    logits = components["final_scores"].index_select(1, seen_device)
                    ce = F.cross_entropy(logits, targets)
                    if config["rgve_mode"] != "off":
                        role_seen = components["role_scores"].index_select(1, seen_device)
                        positive = role_seen.gather(
                            1, targets.view(-1, 1, 1).expand(-1, 1, 3)
                        ).squeeze(1)
                        negative = role_seen.clone()
                        negative.scatter_(
                            1,
                            targets.view(-1, 1, 1).expand(-1, 1, 3),
                            float("-inf"),
                        )
                        hardest = negative.max(dim=1).values
                        loss_role = F.relu(
                            float(config["rgve_role_margin"]) - positive + hardest
                        ).mean()
                        local_all = components["local_scores"]
                        local_seen = local_all.index_select(1, seen_device)
                        target_local = local_seen.gather(1, targets.unsqueeze(1)).squeeze(1)
                        non_target_seen = (local_seen.sum(dim=1) - target_local) / (
                            seen_device.numel() - 1
                        )
                        unseen_device = torch.tensor(
                            manifest["unseen_classes"], dtype=torch.long, device=device
                        )
                        unseen_mean = local_all.index_select(1, unseen_device).mean(dim=1)
                        loss_balance = (non_target_seen - unseen_mean).abs().mean()

                        global_detached = components["global_scores"].detach()
                        scale_detached = model.scale().detach()
                        parent_seen = (global_detached.index_select(1, seen_device) * scale_detached).clone()
                        parent_seen.scatter_(1, targets.unsqueeze(1), float("-inf"))
                        parent_unseen = global_detached.index_select(1, unseen_device) * scale_detached
                        final_calibration = (
                            global_detached + components["beta"] * local_all
                        ) * scale_detached
                        final_seen = final_calibration.index_select(1, seen_device).clone()
                        final_seen.scatter_(1, targets.unsqueeze(1), float("-inf"))
                        final_unseen = final_calibration.index_select(1, unseen_device)
                        parent_gap = torch.logsumexp(parent_seen, dim=1) - torch.logsumexp(
                            parent_unseen, dim=1
                        )
                        final_gap = torch.logsumexp(final_seen, dim=1) - torch.logsumexp(
                            final_unseen, dim=1
                        )
                        loss_calibration = F.relu(final_gap - parent_gap).square().mean()
                else:
                    seen_prototypes = prototypes.index_select(0, seen_classes.to(device))
                    logits = F.normalize(images, dim=-1) @ seen_prototypes.T * model.scale()
                    ce = F.cross_entropy(logits, targets)
                topology = model.topology_loss(prototypes)
                loss = ce + float(config["topology_weight"]) * topology
                if isinstance(model, PaperV2RGVEModel) and config["rgve_mode"] != "off":
                    loss = (
                        loss
                        + float(config["rgve_role_weight"]) * loss_role
                        + float(config["rgve_balance_weight"]) * loss_balance
                        + float(config["rgve_calibration_weight"]) * loss_calibration
                    )
                if isinstance(model, PaperV2VisualModel) and config["visual_mode"] != "off":
                    loss = (
                        loss
                        + float(config["visual_part_weight"]) * loss_part
                        + float(config["visual_diversity_weight"]) * loss_diversity
                        + float(config["visual_anchor_weight"]) * loss_anchor
                        + float(config["visual_hard_weight"]) * loss_hard
                    )
                if frozen_stage_anchor is not None:
                    loss = loss + 0.0 * frozen_stage_anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError("训练loss包含NaN/Inf。")
                loss.backward()
                require_finite_gradients(model)
                if stage_started:
                    stage_gradient_norms[stage] = _gradient_norms(model)
                optimizer.step()
                if iteration % report_interval == 0:
                    metrics = _evaluate_model(
                        model,
                        tensors,
                        manifest,
                        device,
                        eval_batch_size=int(config.get("eval_batch_size", 64)),
                    )
                    diagnostics = model.diagnostics()
                    epoch = iteration // report_interval
                    row = {
                        "iteration": iteration,
                        "nominal_epoch": epoch,
                        "stage": stage,
                        "train_loss": float(loss.detach()),
                        "train_ce": float(ce.detach()),
                        "train_topology": float(topology.detach()),
                        "train_rgve_role": float(loss_role.detach()),
                        "train_rgve_balance": float(loss_balance.detach()),
                        "train_rgve_calibration": float(loss_calibration.detach()),
                        "train_visual_part": float(loss_part.detach()),
                        "train_visual_diversity": float(loss_diversity.detach()),
                        "train_visual_anchor": float(loss_anchor.detach()),
                        "train_visual_hard": float(loss_hard.detach()),
                        "official_metrics_percent": metrics,
                        "diagnostics": diagnostics,
                    }
                    history.append(row)
                    best_zs = max(best_zs, metrics["ZS"])
                    if metrics["H"] > best_h:
                        best_h = metrics["H"]
                        best_state = copy.deepcopy(model.state_dict())
                        selected = {
                            "iteration": iteration,
                            "epoch": epoch,
                            "stage": stage,
                            "metrics": metrics,
                            "diagnostics": diagnostics,
                        }
                        atomic_torch_save(
                            output_dir / "model_best.pth",
                            {
                                "experiment_id": config["experiment_id"],
                                "condition_id": config["condition_id"],
                                "run_id": run_id,
                                "code_commit": code_commit,
                                "config": config,
                                "config_sha256": config_sha,
                                "selected": selected,
                                "model_state_dict": best_state,
                                "reproducibility": reproducibility,
                            },
                        )
                    print(
                        f"iter={iteration} epoch={epoch} stage={stage} "
                        f"U={metrics['U']:.6f} S={metrics['S']:.6f} H={metrics['H']:.6f} best_H={best_h:.6f}"
                    )
            require_finite_model(model)
            atomic_torch_save(
                output_dir / "checkpoint_last.pth",
                {
                    "experiment_id": config["experiment_id"],
                    "condition_id": config["condition_id"],
                    "run_id": run_id,
                    "code_commit": code_commit,
                    "config": config,
                    "config_sha256": config_sha,
                    "last_iteration": niters - 1,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_model_state_dict": best_state,
                    "selected": selected,
                    "history": history,
                    "stage_gradient_norms": stage_gradient_norms,
                    "reproducibility": reproducibility,
                },
            )
            selected["best_zs_observation_percent"] = best_zs

        atomic_write_json(output_dir / "evaluation_history.json", {"history": history})
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "asset_manifest": str(manifest_path),
                "asset_manifest_sha256": config["asset_manifest_sha256"],
                "asset_id": manifest["asset_id"],
                "files": manifest["outputs_sha256"],
            },
        )
        metrics_payload = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "run_id": run_id,
            "framework_id": config["framework_id"],
            "dataset": config["dataset"],
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "training_strategy": config["training_strategy"],
            "selection_scope": config["selection_scope"],
            "nested_official_test_selection": False,
            "test_used_for_selection": bool(config["test_used_for_selection"]),
            "test_used_for_hyperparameter_selection": bool(config["test_used_for_hyperparameter_selection"]),
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "asset_id": manifest["asset_id"],
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "seed": seed,
            "official_test_evaluation_count": len(history),
            "selected_iteration": selected["iteration"],
            "selected_nominal_epoch": selected["epoch"],
            "selected_stage": selected["stage"],
            "best_metrics_percent": selected["metrics"],
            "selected_diagnostics": selected["diagnostics"],
            "stage_gradient_norms": stage_gradient_norms,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
            "evaluation_history_sha256": sha256_file(output_dir / "evaluation_history.json"),
        }
        if "best_zs_observation_percent" in selected:
            metrics_payload["best_zs_observation_percent"] = selected["best_zs_observation_percent"]
        atomic_write_json(output_dir / "metrics.json", metrics_payload)
        print(json.dumps(metrics_payload, ensure_ascii=False))
        return metrics_payload
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
