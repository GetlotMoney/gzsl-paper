"""De-parent retrained ablation: freeze TG+GTD source, head on Mean8 text base.

Runs five fresh-from-scratch retrained conditions (Full / S-off / V-off /
I-off / V+I-off) where the TG+GTD source is fully frozen and the CompiledPCLR
head is built on the pure Mean8 text prototypes (``tg_vpr.base_prototypes()``).

Training signals per condition:
  - S (raw_role_weights): ordinary seen-only classification CE (TUNE013 contract).
  - V (Reader) / I (alpha): first-order class-held-out outer CE (TUNE014
    contract), restricted to the trainable subset of the condition.
  - No relation/direction CE (relation_loss_weight = 0.0), consistent with TUNE014.

A ``--baseline-only`` mode computes the B0 zero-training Mean8 baseline
(no head, no training) for the same official test splits.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v4.train import rank_modulo_class_folds
from model.frameworks.v6.compiled_pclr import (
    EMBED_DIM,
    CompiledPCLRExport,
    CompiledPCLRHead,
    initialized_reader_states,
)
from model.frameworks.v6.train_compiled_pclr import _learning_rate
from model.frameworks.v7 import train_one_text_seen_ce as tune013
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v7-no-parent-ablation.v1"
EXPERIMENT_ID = "V7-ABLATION-004_NO_PARENT_HEAD"
CODE_PARENT_COMMIT = "35cefc52896c383e1ec75a3adc5f78d218d616a3"
FORMAL_FULL_H = 80.51043185404096

CONDITIONS = {
    "Full": {
        "run_id": "RUN-FULL",
        "freeze_role_weights": False,
        "freeze_reader": False,
        "freeze_alpha": False,
        "semantic_enabled": True,
        "visual_enabled": True,
        "interaction_enabled": True,
    },
    "S-off": {
        "run_id": "RUN-S-OFF",
        "freeze_role_weights": True,
        "freeze_reader": False,
        "freeze_alpha": False,
        "semantic_enabled": False,
        "visual_enabled": True,
        "interaction_enabled": True,
    },
    "V-off": {
        "run_id": "RUN-V-OFF",
        "freeze_role_weights": False,
        "freeze_reader": True,
        "freeze_alpha": False,
        "semantic_enabled": True,
        "visual_enabled": False,
        "interaction_enabled": True,
    },
    "I-off": {
        "run_id": "RUN-I-OFF",
        "freeze_role_weights": False,
        "freeze_reader": False,
        "freeze_alpha": True,
        "semantic_enabled": True,
        "visual_enabled": True,
        "interaction_enabled": False,
    },
    "V+I-off": {
        "run_id": "RUN-VI-OFF",
        "freeze_role_weights": False,
        "freeze_reader": True,
        "freeze_alpha": True,
        "semantic_enabled": True,
        "visual_enabled": False,
        "interaction_enabled": False,
    },
}

ALL_META_PARAM_NAMES = (
    "reader_in.weight",
    "reader_in.bias",
    "reader_out.weight",
    "reader_out.bias",
    "raw_alpha",
)

CONFIG_KEYS = tune013.CONFIG_KEYS | {
    "ablation_experiment_id",
    "condition",
    "run_id",
    "code_parent_commit",
    "mean8_base",
    "source_frozen",
    "meta_algorithm",
    "meta_second_order",
    "meta_fold_schedule",
    "meta_inner_steps",
    "meta_inner_learning_rate",
    "meta_inner_batch_size",
    "meta_outer_batch_size",
    "meta_outer_candidate_scope",
    "meta_outer_loss_weight",
    "s_classification_gradient_source",
    "vi_classification_gradient_source",
    "temporary_inner_optimizer_steps_only",
    "freeze_role_weights",
    "freeze_reader",
    "freeze_alpha",
    "semantic_enabled",
    "visual_enabled",
    "interaction_enabled",
}


def _finite_metrics(metrics: dict) -> bool:
    return (
        isinstance(metrics, dict)
        and set(metrics) == {"U", "S", "H", "ZS"}
        and all(math.isfinite(float(metrics[name])) for name in ("U", "S", "H", "ZS"))
    )


def _expected_eval_count(identity: dict) -> int:
    return math.ceil(identity["total_updates"] / identity["eval_interval_steps"])


def _absolute_sha_file(config: dict, key: str) -> Path:
    path = Path(config[key])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"no-parent {key}必须是存在的绝对文件。")
    if sha256_file(path) != config[f"{key}_sha256"]:
        raise ValueError(f"no-parent {key} SHA不匹配。")
    return path


def load_no_parent_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    condition = CONDITIONS.get(config.get("condition")) if isinstance(config, dict) else None
    identity = tune013.IDENTITIES.get(config.get("dataset")) if isinstance(config, dict) else None
    invalid = (
        not isinstance(config, dict)
        or set(config) != CONFIG_KEYS
        or condition is None
        or identity is None
        or config.get("schema_version") != SCHEMA
        or config.get("ablation_experiment_id") != EXPERIMENT_ID
        or config.get("experiment_id") != f"{EXPERIMENT_ID}-{condition['run_id']}"
        or config.get("run_id") != condition["run_id"]
        or config.get("base_commit") != tune013.BASE_COMMIT
        or config.get("code_parent_commit") != CODE_PARENT_COMMIT
        or config.get("source_config_sha256") != identity["source_config_sha256"]
        or config.get("formal_checkpoint_usage") != "baseline_identity_only_not_training_initialization"
        or not _finite_metrics(config.get("formal_full_metrics_percent"))
        or config.get("device") not in {"cpu", "cuda", "cuda:0", "cuda:1"}
        or int(config.get("random_seed", -1)) != 7
        or int(config.get("batch_size", -1)) != 50
        or int(config.get("nominal_epochs", -1)) != 200
        or int(config.get("total_updates", -1)) != identity["total_updates"]
        or int(config.get("eval_interval_steps", -1)) != identity["eval_interval_steps"]
        or float(config.get("learning_rate", -1)) != 1e-4
        or float(config.get("min_learning_rate", -1)) != 1e-5
        or float(config.get("weight_decay", -1)) != 0.0
        or float(config.get("relation_loss_weight", -1)) != 0.0
        or float(config.get("ridge_lambda", -1)) != 0.3
        or float(config.get("relation_temperature", -1)) != 0.2
        or float(config.get("direction_temperature", -1)) != 0.07
        or int(config.get("top_k", -1)) != 3
        or float(config.get("seen_logit_gamma", -1)) != identity["seen_logit_gamma"]
        or float(config.get("alpha_max", -1)) != 2.0
        or abs(float(config.get("initial_alpha", -1)) - tune013.INITIAL_ALPHA) > 1e-12
        or float(config.get("role_weight_max", -1)) != 1.0
        or config.get("initial_role_weights") != tune013.INITIAL_ROLE_WEIGHTS
        or config.get("relation_embedding_mode") != "one_text_uniform_role_difference"
        or float(config.get("relation_endpoint_scale", -1)) != 0.5
        or config.get("classification_ce_scope") != "class_held_out_outer_for_vi_seen_only_for_s"
        or config.get("expected_direction_skip_seen_class_ids") != identity["direction_skip_seen_class_ids"]
        or config.get("best_selection_metric") != "official_condition_H_post_update"
        or int(config.get("official_test_evaluations", -1)) != _expected_eval_count(identity)
        or float(config.get("required_i_off_delta_h", -1)) != 0.0
        or float(config.get("required_v_off_delta_h", -1)) != 0.0
        or config.get("require_full_not_below_formal") is not False
        or config.get("fresh_source_initialization") is not True
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not True
        or config.get("nested_official_test_selection") is not False
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
        or config.get("expert_attributes_used") is not False
        or config.get("llm_world_knowledge_used") is not True
        or config.get("mean8_base") is not True
        or config.get("source_frozen") is not True
        or config.get("meta_algorithm") != "first_order_class_held_out_maml"
        or config.get("meta_second_order") is not False
        or config.get("meta_fold_schedule") != "rank_modulo_update_mod_3"
        or int(config.get("meta_inner_steps", -1)) != 1
        or float(config.get("meta_inner_learning_rate", -1)) <= 0.0
        or int(config.get("meta_inner_batch_size", -1)) != 50
        or int(config.get("meta_outer_batch_size", -1)) != 50
        or config.get("meta_outer_candidate_scope") != "all_train_seen_classes"
        or float(config.get("meta_outer_loss_weight", -1)) != 1.0
        or config.get("s_classification_gradient_source") != "ordinary_seen_batch_ce"
        or config.get("vi_classification_gradient_source") != "class_disjoint_pseudo_unseen_outer_ce"
        or config.get("temporary_inner_optimizer_steps_only") is not True
    )
    if invalid:
        raise ValueError("V7 no-parent ablation配置身份错误。")
    for name in (
        "freeze_role_weights",
        "freeze_reader",
        "freeze_alpha",
        "semantic_enabled",
        "visual_enabled",
        "interaction_enabled",
    ):
        if config.get(name) is not condition[name]:
            raise ValueError(f"V7 no-parent ablation {name}与condition合同不一致。")
    _absolute_sha_file(config, "source_config")
    _absolute_sha_file(config, "formal_checkpoint")
    return config, sha256_file(path)


def _condition(config: dict) -> dict:
    return CONDITIONS[str(config["condition"])]


def meta_param_names(config: dict) -> tuple[str, ...]:
    """可训练的 V/I 子集（按条件冻结过滤）。

    架构耦合说明：Reader 的唯一 logits 通道是 interaction 关系分支
    ``readout @ alpha*compiled_g``（compiled_pclr.py forward）。当 interaction
    被关闭（I-off / V+I-off）时关系块置零，Reader 的 outer 梯度恒为零并停留
    在 READER_SEED 初始化；因此 I-off 与 V+I-off 逐值等价，Full−I-off 度量的是
    "V+I 联合贡献"，不是 I 单独贡献。本函数只负责按 freeze 过滤可训练子集。
    """
    condition = _condition(config)
    names = []
    if not condition["freeze_reader"]:
        names.extend(("reader_in.weight", "reader_in.bias", "reader_out.weight", "reader_out.bias"))
    if not condition["freeze_alpha"]:
        names.append("raw_alpha")
    return tuple(names)


@torch.no_grad()
def mean8_prototypes(source: torch.nn.Module) -> torch.Tensor:
    return F.normalize(source.parent.tg_vpr.base_prototypes().detach().float(), dim=-1)


@torch.no_grad()
def build_mean8_head(source, config: dict, device: torch.device) -> tuple[CompiledPCLRHead, dict]:
    was_training = source.training
    source.eval()
    try:
        relations, edges, graph = tune013.one_text_edges_and_relations(
            source.parent.tg_vpr.sentence_embeds,
            top_k=int(config["top_k"]),
        )
        reader_in_state, reader_out_state = initialized_reader_states()
        head = CompiledPCLRHead(
            base_prototypes=source.parent.tg_vpr.base_prototypes(),
            role_prototypes=source.parent.tg_vpr.sentence_embeds,
            relation_embeddings=relations,
            edge_index=edges,
            seen_classes=source.seen_classes,
            scale=float(source.scale()),
            reader_in_state=reader_in_state,
            reader_out_state=reader_out_state,
            ridge_lambda=float(config["ridge_lambda"]),
            relation_temperature=float(config["relation_temperature"]),
            direction_temperature=float(config["direction_temperature"]),
            seen_logit_gamma=float(config["seen_logit_gamma"]),
            alpha_max=float(config["alpha_max"]),
            initial_alpha=float(config["initial_alpha"]),
            role_weight_max=float(config["role_weight_max"]),
            initial_role_weights=torch.tensor(config["initial_role_weights"]),
        ).to(device)
    finally:
        source.train(was_training)
    return head, graph


def load_frozen_source(config: dict, device: torch.device):
    source, tensors, source_config = tune013.load_training_source(config, device)
    source.requires_grad_(False)
    trainable = [name for name, parameter in source.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(f"no-parent source必须全部冻结，发现可训练参数：{trainable}")
    return source, tensors, source_config


def apply_ablation_trainability(head: CompiledPCLRHead, config: dict) -> None:
    condition = _condition(config)
    head.requires_grad_(True)
    if condition["freeze_role_weights"]:
        head.raw_role_weights.requires_grad_(False)
    if condition["freeze_alpha"]:
        head.raw_alpha.requires_grad_(False)
    if condition["freeze_reader"]:
        for parameter in (*head.reader_in.parameters(), *head.reader_out.parameters()):
            parameter.requires_grad_(False)


def _head_parameters(head: CompiledPCLRHead) -> list[torch.nn.Parameter]:
    return [parameter for parameter in head.parameters() if parameter.requires_grad]


def condition_logits(
    head: CompiledPCLRHead,
    images: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    condition = _condition(config)
    return head(
        images,
        semantic_enabled=condition["semantic_enabled"],
        visual_enabled=condition["visual_enabled"],
        interaction_enabled=condition["interaction_enabled"],
    )


@torch.no_grad()
def condition_export(head: CompiledPCLRHead, config: dict) -> CompiledPCLRExport:
    condition = _condition(config)
    if condition["visual_enabled"]:
        reader_in_weight = head.reader_in.weight.detach().cpu().clone()
        reader_in_bias = head.reader_in.bias.detach().cpu().clone()
        reader_out_weight = head.reader_out.weight.detach().cpu().clone()
        reader_out_bias = head.reader_out.bias.detach().cpu().clone()
    else:
        reader_in_weight = torch.zeros_like(head.reader_in.weight.detach().cpu())
        reader_in_bias = torch.zeros_like(head.reader_in.bias.detach().cpu())
        reader_out_weight = torch.zeros_like(head.reader_out.weight.detach().cpu())
        reader_out_bias = torch.zeros_like(head.reader_out.bias.detach().cpu())
    return CompiledPCLRExport(
        q=head.export_q(
            semantic_enabled=condition["semantic_enabled"],
            interaction_enabled=condition["interaction_enabled"],
        ).detach().cpu().clone(),
        bias=head.seen_bias.detach().cpu().clone(),
        reader_in_weight=reader_in_weight,
        reader_in_bias=reader_in_bias,
        reader_out_weight=reader_out_weight,
        reader_out_bias=reader_out_bias,
    )


def gradient_receipt(head: CompiledPCLRHead) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name, parameter in head.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f"冻结参数不应产生梯度：{name}")
            values[name] = None
            continue
        if parameter.grad is None:
            raise RuntimeError(f"训练参数缺少梯度：{name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"梯度包含NaN/Inf：{name}")
        values[name] = float(parameter.grad.detach().norm().cpu())
    return values


# ---------------------------------------------------------------------------
# First-order class-held-out episode (TUNE014 contract, parameterized by the
# trainable V/I subset of the current condition).
# ---------------------------------------------------------------------------


def _global_to_candidate(
    targets: torch.Tensor,
    candidate_classes: torch.Tensor,
    class_count: int,
) -> torch.Tensor:
    targets = targets.long()
    candidate_classes = candidate_classes.to(targets.device).long()
    mapping = torch.full((int(class_count),), -1, dtype=torch.long, device=targets.device)
    mapping[candidate_classes] = torch.arange(candidate_classes.numel(), device=targets.device)
    local = mapping.index_select(0, targets)
    if not bool(local.ge(0).all()):
        raise ValueError("no-parent分类CE targets必须全部属于候选类别集合。")
    return local


def candidate_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    candidate_classes: torch.Tensor,
    class_count: int,
) -> torch.Tensor:
    if targets.ndim != 1 or targets.numel() != logits.size(0):
        raise ValueError("no-parent targets必须是与batch等长的一维全局类别ID。")
    candidates = candidate_classes.to(logits.device).long()
    if candidates.ndim != 1 or candidates.numel() < 2 or candidates.unique().numel() != candidates.numel():
        raise ValueError("no-parent候选类别必须是至少2个唯一类别。")
    local_targets = _global_to_candidate(targets.to(logits.device), candidates, class_count)
    return F.cross_entropy(logits.index_select(1, candidates), local_targets)


def class_member_indices(labels: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    labels_cpu = labels.detach().cpu().long()
    class_cpu = class_ids.detach().cpu().long()
    mask = torch.isin(labels_cpu, class_cpu)
    ids = mask.nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
        raise ValueError("no-parent类别集合在训练标签中没有样本。")
    return ids


def sample_class_batch_indices(
    labels: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    ids = class_member_indices(labels, class_ids)
    if ids.numel() >= int(batch_size):
        order = torch.randperm(ids.numel(), generator=generator)[: int(batch_size)]
        return ids.index_select(0, order).to(device)
    draw = torch.randint(ids.numel(), (int(batch_size),), generator=generator)
    return ids.index_select(0, draw).to(device)


def _set_grad_none(head: CompiledPCLRHead, names: tuple[str, ...]) -> None:
    selected = set(names)
    for name, parameter in head.named_parameters():
        if name in selected:
            parameter.grad = None


def _copy_temp_grads_to_formal(
    temp_head: CompiledPCLRHead,
    formal_head: CompiledPCLRHead,
    names: tuple[str, ...],
    *,
    weight: float,
) -> dict[str, float]:
    temp_params = dict(temp_head.named_parameters())
    formal_params = dict(formal_head.named_parameters())
    norms: dict[str, float] = {}
    for name in names:
        grad = temp_params[name].grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"no-parent outer梯度缺失或非有限：{name}")
        value = grad.detach().to(formal_params[name].device) * float(weight)
        if formal_params[name].grad is None:
            formal_params[name].grad = value.clone()
        else:
            formal_params[name].grad.add_(value)
        norms[name] = float(value.norm().detach().cpu())
    return norms


def first_order_class_held_out_vi_episode(
    head: CompiledPCLRHead,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    pseudo_seen: torch.Tensor,
    pseudo_unseen: torch.Tensor,
    outer_candidate_classes: torch.Tensor,
    meta_param_names: tuple[str, ...],
    config: dict,
    generator: torch.Generator,
    inner_batch_size: int,
    outer_batch_size: int,
    inner_learning_rate: float,
    outer_loss_weight: float,
) -> dict:
    if not meta_param_names:
        raise ValueError("no-parent空meta参数不允许执行episode。")
    inner_set = set(int(value) for value in pseudo_seen.detach().cpu().tolist())
    outer_set = set(int(value) for value in pseudo_unseen.detach().cpu().tolist())
    if not inner_set or not outer_set or inner_set & outer_set:
        raise ValueError("no-parent inner和pseudo-unseen类别必须非空且不相交。")
    outer_candidate_set = set(int(value) for value in outer_candidate_classes.detach().cpu().tolist())
    if not outer_set.issubset(outer_candidate_set) or not inner_set.issubset(outer_candidate_set):
        raise ValueError("no-parent outer候选轴必须同时包含inner类和pseudo-unseen类。")
    device = next(head.parameters()).device
    temp_head = copy.deepcopy(head).to(device)
    temp_head.train()
    _set_grad_none(temp_head, tuple(name for name, _ in temp_head.named_parameters()))
    inner_ids = sample_class_batch_indices(
        train_labels,
        pseudo_seen,
        batch_size=int(inner_batch_size),
        generator=generator,
        device=device,
    )
    outer_ids = sample_class_batch_indices(
        train_labels,
        pseudo_unseen,
        batch_size=int(outer_batch_size),
        generator=generator,
        device=device,
    )
    inner_images = train_features.index_select(0, inner_ids).to(device).float()
    inner_labels = train_labels.index_select(0, inner_ids).to(device).long()
    outer_images = train_features.index_select(0, outer_ids).to(device).float()
    outer_labels = train_labels.index_select(0, outer_ids).to(device).long()
    if not bool(torch.isin(inner_labels.detach().cpu(), pseudo_seen.detach().cpu()).all()):
        raise RuntimeError("no-parent inner batch包含pseudo-unseen类别。")
    if not bool(torch.isin(outer_labels.detach().cpu(), pseudo_unseen.detach().cpu()).all()):
        raise RuntimeError("no-parent outer batch不属于pseudo-unseen类别。")

    # inner/outer forward 必须遵守当前条件的 enabled 开关（条件语义一致）
    def _logits(model: CompiledPCLRHead, images: torch.Tensor) -> torch.Tensor:
        condition = _condition(config)
        return model(
            images,
            semantic_enabled=condition["semantic_enabled"],
            visual_enabled=condition["visual_enabled"],
            interaction_enabled=condition["interaction_enabled"],
        )

    temp_params = dict(temp_head.named_parameters())
    vi_params = [temp_params[name] for name in meta_param_names]
    inner_loss = candidate_classification_loss(
        _logits(temp_head, inner_images),
        inner_labels,
        candidate_classes=pseudo_seen,
        class_count=head.class_count,
    )
    inner_grads = torch.autograd.grad(
        inner_loss,
        vi_params,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    with torch.no_grad():
        for parameter, grad in zip(vi_params, inner_grads):
            if not torch.isfinite(grad).all():
                raise FloatingPointError("no-parent inner梯度包含NaN/Inf。")
            parameter.add_(grad, alpha=-float(inner_learning_rate))

    _set_grad_none(temp_head, tuple(name for name, _ in temp_head.named_parameters()))
    outer_loss = candidate_classification_loss(
        _logits(temp_head, outer_images),
        outer_labels,
        candidate_classes=outer_candidate_classes,
        class_count=head.class_count,
    )
    outer_loss.backward()
    outer_gradient_norms = _copy_temp_grads_to_formal(
        temp_head,
        head,
        meta_param_names,
        weight=float(outer_loss_weight),
    )
    return {
        "inner_loss": float(inner_loss.detach().cpu()),
        "outer_loss": float(outer_loss.detach().cpu()),
        "outer_gradient_norms": outer_gradient_norms,
        "inner_class_ids": sorted(inner_set),
        "pseudo_unseen_class_ids": sorted(outer_set),
        "outer_candidate_class_ids": sorted(outer_candidate_set),
        "inner_batch_size": int(inner_ids.numel()),
        "outer_batch_size": int(outer_ids.numel()),
        "second_order": False,
        "temporary_inner_optimizer_steps_only": True,
    }


# ---------------------------------------------------------------------------
# Evaluation / baselines
# ---------------------------------------------------------------------------


def _scores(predictions: dict[str, torch.Tensor], tensors: dict, seen: torch.Tensor, unseen: torch.Tensor) -> dict[str, float]:
    s = 100.0 * per_class_accuracy(tensors["test_seen_labels"], predictions["seen"], seen)
    u = 100.0 * per_class_accuracy(tensors["test_unseen_labels"], predictions["unseen"], unseen)
    zs = 100.0 * per_class_accuracy(tensors["test_unseen_labels"], predictions["zs"], unseen)
    h = 2.0 * s * u / (s + u) if s + u else 0.0
    return {"U": float(u), "S": float(s), "H": float(h), "ZS": float(zs)}


@torch.no_grad()
def evaluate_condition(head: CompiledPCLRHead, source, tensors: dict, config: dict, device: torch.device) -> dict:
    head.eval()
    source.eval()
    seen = head.seen_classes.detach().cpu()
    all_classes = torch.arange(head.class_count)
    unseen_cpu = all_classes[~torch.isin(all_classes, seen)]
    unseen = unseen_cpu.to(device)
    outputs = {"seen": [], "unseen": [], "zs": []}
    parent = {"seen": [], "unseen": [], "zs": []}
    mean8 = mean8_prototypes(source).to(device)
    scale = source.scale().float()
    for split, features in (("seen", tensors["test_seen_features"]), ("unseen", tensors["test_unseen_features"])):
        for start in range(0, len(features), 256):
            images = features[start : start + 256].to(device).float()
            logits = condition_logits(head, images, config)
            outputs[split].append(logits.argmax(1).cpu())
            if split == "unseen":
                outputs["zs"].append(unseen[logits.index_select(1, unseen).argmax(1)].cpu())
            parent_logits = F.normalize(images, dim=-1) @ mean8.T * scale
            parent[split].append(parent_logits.argmax(1).cpu())
            if split == "unseen":
                parent["zs"].append(unseen[parent_logits.index_select(1, unseen).argmax(1)].cpu())
    for group in (outputs, parent):
        for split in group:
            group[split] = torch.cat(group[split])
    return {
        "condition": config["condition"],
        "metrics": _scores(outputs, tensors, seen, unseen_cpu),
        "mean8_baseline_metrics": _scores(parent, tensors, seen, unseen_cpu),
    }


def _training_context(config: dict, device: torch.device):
    identity = tune013.IDENTITIES[config["dataset"]]
    source, tensors, source_config = load_frozen_source(config, device)
    head, graph = build_mean8_head(source, config, device)
    skipped = tune013.validate_training_identity(head, tensors, identity)
    if skipped != identity["direction_skip_seen_class_ids"]:
        raise ValueError("no-parent CUB方向CE覆盖边界不匹配。")
    labels_cpu = tensors["train_labels"].long()
    seen = torch.unique(labels_cpu, sorted=True)
    seen_device = seen.to(device)
    global_to_seen = torch.full((identity["class_count"],), -1, dtype=torch.long, device=device)
    global_to_seen[seen_device] = torch.arange(len(seen), device=device)
    folds = rank_modulo_class_folds(seen)
    apply_ablation_trainability(head, config)
    return source, tensors, source_config, head, graph, skipped, seen_device, global_to_seen, folds


# ---------------------------------------------------------------------------
# micro-batch / run / baseline
# ---------------------------------------------------------------------------


def micro_batch(config_path: Path) -> dict:
    config, config_sha = load_no_parent_config(config_path)
    device = torch.device(config["device"])
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, _source_config, head, graph, skipped, seen_device, global_to_seen, folds = _training_context(config, device)
    train_features = tensors["train_features"].to(device).float()
    train_labels = tensors["train_labels"].to(device).long()
    images = train_features[:50]
    labels = train_labels[:50]
    generator = torch.Generator(device="cpu").manual_seed(7)
    episode_generator = torch.Generator(device="cpu").manual_seed(7 + 1_000_000)
    s_loss = tune013.seen_only_classification_loss(
        condition_logits(head, images, config),
        labels,
        seen_device=seen_device,
        global_to_seen=global_to_seen,
    )
    s_loss.backward()
    _set_grad_none(head, meta_param_names(config))
    meta_names = meta_param_names(config)
    episode = None
    if meta_names:
        episode = first_order_class_held_out_vi_episode(
            head,
            train_features,
            train_labels,
            pseudo_seen=folds[0][0],
            pseudo_unseen=folds[0][1],
            outer_candidate_classes=seen_device.detach().cpu(),
            meta_param_names=meta_names,
            config=config,
            generator=episode_generator,
            inner_batch_size=int(config["meta_inner_batch_size"]),
            outer_batch_size=int(config["meta_outer_batch_size"]),
            inner_learning_rate=float(config["meta_inner_learning_rate"]),
            outer_loss_weight=float(config["meta_outer_loss_weight"]),
        )
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "condition": config["condition"],
        "dataset": config["dataset"],
        "config_sha256": config_sha,
        "batch_size": 50,
        "s_seen_ce": float(s_loss.detach().cpu()),
        "meta_episode": episode,
        "head_gradient_norms": gradient_receipt(head),
        "mean8_base": True,
        "source_trainable_params": 0,
        "graph": graph,
        "direction_skip_seen_class_ids": skipped,
        "persistent_writes": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("no-parent expected commit错误。")
    config, config_sha = load_no_parent_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != config["run_id"]:
        raise ValueError("no-parent RUN身份错误。")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("no-parent正式RUN要求CUDA。")
    reproducibility = configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, _source_config, head, graph, skipped, seen_device, global_to_seen, folds = _training_context(config, device)
    train_features = tensors["train_features"].to(device).float()
    train_labels = tensors["train_labels"].to(device).long()
    head_optimizer = torch.optim.Adam(_head_parameters(head), lr=float(config["learning_rate"]), weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(7)
    # 主 batch 与 class-held-out episode 采样使用独立 generator，使五组条件的主
    # batch 随机数据流完全一致（episode 是否执行不影响主 batch 序列）。
    episode_generator = torch.Generator(device="cpu").manual_seed(7 + 1_000_000)
    output = prepare_output_dir(output_dir)
    (output / "config.snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log = (output / "training.log").open("w", encoding="utf-8", buffering=1)

    def emit(payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        print(line)
        log.write(line + "\n")

    history: list[dict] = []
    best: dict | None = None
    best_update = -1
    best_state = None
    best_zs: dict | None = None
    interval = {"s_seen_ce": 0.0, "vi_outer_ce": 0.0}
    steps = 0
    meta_names = meta_param_names(config)
    for update in range(1, int(config["total_updates"]) + 1):
        head.train()
        lr = _learning_rate(config, update)
        for group in head_optimizer.param_groups:
            group["lr"] = lr
        ids = torch.randperm(len(train_features), generator=generator)[:50].to(device)
        images = train_features[ids]
        labels = train_labels[ids]
        head_optimizer.zero_grad(set_to_none=True)
        s_loss = tune013.seen_only_classification_loss(
            condition_logits(head, images, config),
            labels,
            seen_device=seen_device,
            global_to_seen=global_to_seen,
        )
        s_loss.backward()
        _set_grad_none(head, meta_names)
        if meta_names:
            episode = first_order_class_held_out_vi_episode(
                head,
                train_features,
                train_labels,
                pseudo_seen=folds[(update - 1) % 3][0],
                pseudo_unseen=folds[(update - 1) % 3][1],
                outer_candidate_classes=seen_device.detach().cpu(),
                meta_param_names=meta_names,
                config=config,
                generator=episode_generator,
                inner_batch_size=int(config["meta_inner_batch_size"]),
                outer_batch_size=int(config["meta_outer_batch_size"]),
                inner_learning_rate=float(config["meta_inner_learning_rate"]),
                outer_loss_weight=float(config["meta_outer_loss_weight"]),
            )
            interval["vi_outer_ce"] += float(episode["outer_loss"])
        else:
            episode = None
        gradient_receipt(head)
        head_optimizer.step()
        interval["s_seen_ce"] += float(s_loss.detach().cpu())
        steps += 1
        if update % int(config["eval_interval_steps"]) != 0 and update != int(config["total_updates"]):
            continue
        evaluation = evaluate_condition(head, source, tensors, config, device)
        record = {
            "update": update,
            "head_lr": lr,
            "train_mean": {key: value / max(steps, 1) for key, value in interval.items()},
            "alpha": float(head.alpha().detach().cpu()),
            "role_weights": [float(value) for value in head.role_weights().detach().cpu()],
            **evaluation,
        }
        history.append(record)
        emit({"event": "evaluation", **record})
        interval = {key: 0.0 for key in interval}
        steps = 0
        if best is None or evaluation["metrics"]["H"] > best["metrics"]["H"]:
            best = copy.deepcopy(evaluation)
            best_update = update
            best_state = copy.deepcopy(head.state_dict())
        if best_zs is None or evaluation["metrics"]["ZS"] > best_zs["ZS"]:
            best_zs = {"update": update, "ZS": evaluation["metrics"]["ZS"], "metrics": copy.deepcopy(evaluation["metrics"])}
    if best_state is None or best is None:
        raise RuntimeError("no-parent没有post-update best checkpoint。")
    head.load_state_dict(best_state)
    final = evaluate_condition(head, source, tensors, config, device)
    export = condition_export(head, config)
    checkpoint = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "ablation_experiment_id": EXPERIMENT_ID,
        "condition": config["condition"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "best_update": best_update,
        "model_state_dict": best_state,
        "export": export.__dict__,
        "graph": graph,
    }
    atomic_torch_save(output / "model_best.pth", checkpoint)
    atomic_write_json(output / "evaluation_history.json", {"history": history})
    condition_h = float(final["metrics"]["H"])
    mean8_h = float(final["mean8_baseline_metrics"]["H"])
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "ablation_experiment_id": EXPERIMENT_ID,
        "condition": config["condition"],
        "dataset": config["dataset"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "source_config_sha256": config["source_config_sha256"],
        "formal_checkpoint_sha256": config["formal_checkpoint_sha256"],
        "best_update": best_update,
        "metrics": final["metrics"],
        "mean8_baseline_metrics": final["mean8_baseline_metrics"],
        "formal_full_metrics_percent": config["formal_full_metrics_percent"],
        "delta_H_condition_minus_mean8_baseline": float(condition_h - mean8_h),
        "delta_H_vs_formal_full": float(condition_h - FORMAL_FULL_H),
        "condition_contract": _condition(config),
        "meta_contract": {
            "mean8_base": True,
            "source_frozen": True,
            "meta_algorithm": config["meta_algorithm"],
            "meta_second_order": False,
            "meta_param_names": list(meta_names),
            "temporary_inner_optimizer_steps_only": True,
            "vi_classification_gradient_source": config["vi_classification_gradient_source"],
        },
        "architecture_note": {
            "reader_logits_channel": "readout @ alpha*compiled_g only",
            "i_off_reader_gradient": "zero (relation block zeroed -> Reader stays at READER_SEED init)",
            "i_off_equals_v_plus_i_off": True,
            "full_minus_i_off_interpretation": "joint V+I contribution, not I alone",
            "main_batch_generator_seed": 7,
            "episode_generator_seed": 1000007,
        },
        "best_zs_observation": best_zs,
        "decision": "diagnose_no_parent_ablation",
        "graph": graph,
        "direction_skip_seen_class_ids": skipped,
        "total_updates": int(config["total_updates"]),
        "official_test_evaluations": len(history),
        "fresh_source_initialization": True,
        "test_used_for_selection": True,
        "test_used_for_hyperparameter_selection": True,
        "nested_official_test_selection": False,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
        "human_annotations_used": False,
        "expert_attributes_used": False,
        "llm_world_knowledge_used": True,
        "reproducibility": reproducibility,
    }
    atomic_write_json(output / "metrics.json", result)
    emit({"event": "complete", **result})
    log.close()
    return result


def baseline(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict:
    """B0 zero-training Mean8 baseline: no head, no training, official eval."""
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("no-parent baseline expected commit错误。")
    config, config_sha = load_no_parent_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != "BASELINE-B0":
        raise ValueError("no-parent baseline身份错误。")
    if config["condition"] != "Full":
        raise ValueError("no-parent baseline必须使用Full条件config。")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("no-parent baseline正式RUN要求CUDA。")
    reproducibility = configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, _source_config, head, graph, skipped, _seen_device, _global_to_seen, _folds = _training_context(config, device)
    evaluation = evaluate_condition(head, source, tensors, config, device)
    b0 = evaluation["mean8_baseline_metrics"]
    if not _finite_metrics(b0):
        raise ValueError("no-parent B0 Mean8基线必须为有限U/S/H/ZS。")
    output = prepare_output_dir(output_dir)
    (output / "config.snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = {
        "schema_version": SCHEMA,
        "experiment_id": f"{EXPERIMENT_ID}-BASELINE-B0",
        "ablation_experiment_id": EXPERIMENT_ID,
        "condition": "B0",
        "dataset": config["dataset"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "metrics": b0,
        "mean8_baseline_metrics": b0,
        "formal_full_metrics_percent": config["formal_full_metrics_percent"],
        "delta_H_vs_formal_full": float(b0["H"] - FORMAL_FULL_H),
        "decision": "zero_training_mean8_baseline",
        "graph": graph,
        "direction_skip_seen_class_ids": skipped,
        "fresh_source_initialization": True,
        "test_used_for_selection": True,
        "test_used_for_hyperparameter_selection": True,
        "nested_official_test_selection": False,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
        "human_annotations_used": False,
        "expert_attributes_used": False,
        "llm_world_knowledge_used": True,
        "reproducibility": reproducibility,
    }
    atomic_write_json(output / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-config-sha")
    parser.add_argument("--micro-batch-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.micro_batch_only:
        micro_batch(args.config)
        return
    if args.baseline_only:
        if args.output_dir is None or not args.expected_commit or not args.expected_config_sha:
            parser.error("baseline缺少身份参数。")
        baseline(args.config, args.output_dir, args.expected_commit, args.expected_config_sha)
        return
    if args.output_dir is None or not args.expected_commit or not args.expected_config_sha:
        parser.error("正式RUN缺少身份参数。")
    run(args.config, args.output_dir, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()
