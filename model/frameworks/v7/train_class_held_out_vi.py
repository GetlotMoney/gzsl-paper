"""Train TUNE014 with class-held-out first-order V/I supervision on CUB."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v2 import train as h1
from model.frameworks.v4.train import (
    GroupwiseSchedule,
    rank_modulo_class_folds,
    refresh_oracle_targets,
    teacher_refresh_updates,
)
from model.frameworks.v6.compiled_pclr import CompiledPCLRHead
from model.frameworks.v6.train_compiled_pclr import (
    _finite_source_gradients,
    _gradient_receipt,
    _learning_rate,
    _parent_loss,
)
from model.frameworks.v7 import train_one_text_seen_ce as tune013
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v7-class-held-out-vi.v1"
CODE_PARENT_COMMIT = "35cefc52896c383e1ec75a3adc5f78d218d616a3"
META_PARAM_NAMES = (
    "reader_in.weight",
    "reader_in.bias",
    "reader_out.weight",
    "reader_out.bias",
    "raw_alpha",
)
S_PARAM_NAMES = ("raw_role_weights",)
IDENTITIES = {
    "CUB": {
        "experiment_id": "V7-TUNE-014-CUB-CLASS-HELD-OUT-VI",
        "source_config_sha256": tune013.IDENTITIES["CUB"]["source_config_sha256"],
        "seen_count": tune013.IDENTITIES["CUB"]["seen_count"],
        "class_count": tune013.IDENTITIES["CUB"]["class_count"],
        "total_updates": tune013.IDENTITIES["CUB"]["total_updates"],
        "eval_interval_steps": tune013.IDENTITIES["CUB"]["eval_interval_steps"],
        "seen_logit_gamma": tune013.IDENTITIES["CUB"]["seen_logit_gamma"],
        "direction_skip_seen_class_ids": tune013.IDENTITIES["CUB"]["direction_skip_seen_class_ids"],
    },
}
CONFIG_KEYS = tune013.CONFIG_KEYS | {
    "code_parent_commit",
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
}


def _absolute_sha_file(config: dict, key: str) -> Path:
    path = Path(config[key])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"TUNE014 {key}必须是存在的绝对文件。")
    if sha256_file(path) != config[f"{key}_sha256"]:
        raise ValueError(f"TUNE014 {key} SHA不匹配。")
    return path


def _expected_eval_count(identity: dict) -> int:
    return math.ceil(identity["total_updates"] / identity["eval_interval_steps"])


def load_class_held_out_vi_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    identity = IDENTITIES.get(config.get("dataset")) if isinstance(config, dict) else None
    invalid = (
        not isinstance(config, dict)
        or set(config) != CONFIG_KEYS
        or identity is None
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != identity["experiment_id"]
        or config.get("base_commit") != tune013.BASE_COMMIT
        or config.get("code_parent_commit") != CODE_PARENT_COMMIT
        or config.get("source_config_sha256") != identity["source_config_sha256"]
        or config.get("formal_checkpoint_usage") != "baseline_identity_only_not_training_initialization"
        or not tune013._finite_metrics(config.get("formal_full_metrics_percent"))
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
        or config.get("best_selection_metric") != "official_full_H_post_update"
        or int(config.get("official_test_evaluations", -1)) != _expected_eval_count(identity)
        or float(config.get("required_i_off_delta_h", -1)) != 0.0
        or float(config.get("required_v_off_delta_h", -1)) != 0.0
        or config.get("require_full_not_below_formal") is not True
        or config.get("fresh_source_initialization") is not True
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not True
        or config.get("nested_official_test_selection") is not False
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
        or config.get("expert_attributes_used") is not False
        or config.get("llm_world_knowledge_used") is not True
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
        raise ValueError("TUNE014 class-held-out V/I配置身份错误。")
    _absolute_sha_file(config, "source_config")
    _absolute_sha_file(config, "formal_checkpoint")
    return config, sha256_file(path)


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
        raise ValueError("TUNE014分类CE targets必须全部属于候选类别集合。")
    return local


def candidate_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    candidate_classes: torch.Tensor,
    class_count: int,
) -> torch.Tensor:
    if targets.ndim != 1 or targets.numel() != logits.size(0):
        raise ValueError("TUNE014 targets必须是与batch等长的一维全局类别ID。")
    candidates = candidate_classes.to(logits.device).long()
    if candidates.ndim != 1 or candidates.numel() < 2 or candidates.unique().numel() != candidates.numel():
        raise ValueError("TUNE014候选类别必须是至少2个唯一类别。")
    local_targets = _global_to_candidate(targets.to(logits.device), candidates, class_count)
    return F.cross_entropy(logits.index_select(1, candidates), local_targets)


def class_member_indices(
    labels: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    labels_cpu = labels.detach().cpu().long()
    class_cpu = class_ids.detach().cpu().long()
    mask = torch.isin(labels_cpu, class_cpu)
    ids = mask.nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
        raise ValueError("TUNE014类别集合在训练标签中没有样本。")
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
            raise RuntimeError(f"TUNE014 outer梯度缺失或非有限：{name}")
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
    generator: torch.Generator,
    inner_batch_size: int,
    outer_batch_size: int,
    inner_learning_rate: float,
    outer_loss_weight: float,
) -> dict:
    inner_set = set(int(value) for value in pseudo_seen.detach().cpu().tolist())
    outer_set = set(int(value) for value in pseudo_unseen.detach().cpu().tolist())
    if not inner_set or not outer_set or inner_set & outer_set:
        raise ValueError("TUNE014 inner和pseudo-unseen类别必须非空且不相交。")
    outer_candidate_set = set(int(value) for value in outer_candidate_classes.detach().cpu().tolist())
    if not outer_set.issubset(outer_candidate_set) or not inner_set.issubset(outer_candidate_set):
        raise ValueError("TUNE014 outer候选轴必须同时包含inner类和pseudo-unseen类。")
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
        raise RuntimeError("TUNE014 inner batch包含pseudo-unseen类别。")
    if not bool(torch.isin(outer_labels.detach().cpu(), pseudo_unseen.detach().cpu()).all()):
        raise RuntimeError("TUNE014 outer batch不属于pseudo-unseen类别。")

    temp_params = dict(temp_head.named_parameters())
    vi_params = [temp_params[name] for name in META_PARAM_NAMES]
    inner_loss = candidate_classification_loss(
        temp_head(inner_images),
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
                raise FloatingPointError("TUNE014 inner梯度包含NaN/Inf。")
            parameter.add_(grad, alpha=-float(inner_learning_rate))

    _set_grad_none(temp_head, tuple(name for name, _ in temp_head.named_parameters()))
    outer_loss = candidate_classification_loss(
        temp_head(outer_images),
        outer_labels,
        candidate_classes=outer_candidate_classes,
        class_count=head.class_count,
    )
    outer_loss.backward()
    outer_gradient_norms = _copy_temp_grads_to_formal(
        temp_head,
        head,
        META_PARAM_NAMES,
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


def ordinary_seen_s_loss(
    head: CompiledPCLRHead,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    seen_device: torch.Tensor,
    global_to_seen: torch.Tensor,
) -> torch.Tensor:
    logits = head(images)
    return tune013.seen_only_classification_loss(
        logits,
        targets,
        seen_device=seen_device,
        global_to_seen=global_to_seen,
    )


def class_held_out_contract(
    metrics: dict[str, dict[str, float]],
    formal: dict[str, float],
    config: dict,
    *,
    best_update: int,
) -> tuple[bool, dict]:
    passed, deltas = tune013.contract(metrics, formal, config)
    return bool(passed and int(best_update) > 0), deltas


def _training_context(config: dict, device: torch.device):
    identity = IDENTITIES[config["dataset"]]
    source, tensors, source_config = tune013.load_training_source(config, device)
    head, graph = tune013.build_head(source, config, device)
    skipped = tune013.validate_training_identity(head, tensors, identity)
    if skipped != identity["direction_skip_seen_class_ids"]:
        raise ValueError("TUNE014 CUB direction CE覆盖边界不匹配。")
    controls = tune013.relation_controls(head, seed=7)
    labels_cpu = tensors["train_labels"].long()
    seen = torch.unique(labels_cpu, sorted=True)
    seen_device = seen.to(device)
    global_to_seen = torch.full((identity["class_count"],), -1, dtype=torch.long, device=device)
    global_to_seen[seen_device] = torch.arange(len(seen), device=device)
    centroids = h1.visual_centroids(tensors["train_features"], labels_cpu, seen).to(device)
    folds = rank_modulo_class_folds(seen)
    packages = refresh_oracle_targets(source, centroids, folds, float(source_config["theta_penalty"]))
    return source, tensors, source_config, head, graph, controls, skipped, seen_device, global_to_seen, centroids, folds, packages


def micro_batch(config_path: Path) -> dict:
    config, config_sha = load_class_held_out_vi_config(config_path)
    device = torch.device(config["device"])
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, source_config, head, graph, _controls, skipped, seen_device, global_to_seen, _centroids, folds, packages = _training_context(config, device)
    train_features = tensors["train_features"].to(device).float()
    train_labels = tensors["train_labels"].to(device).long()
    images = train_features[:50]
    labels = train_labels[:50]
    generator = torch.Generator(device="cpu").manual_seed(7)
    parent = _parent_loss(source, images, labels, seen_device=seen_device, global_to_seen=global_to_seen, fold_package=packages[0], source_config=source_config)
    s_loss = ordinary_seen_s_loss(head, images, labels, seen_device=seen_device, global_to_seen=global_to_seen)
    parent["total"].backward()
    s_loss.backward()
    _set_grad_none(head, META_PARAM_NAMES)
    episode = first_order_class_held_out_vi_episode(
        head,
        train_features,
        train_labels,
        pseudo_seen=folds[0][0],
        pseudo_unseen=folds[0][1],
        outer_candidate_classes=seen_device.detach().cpu(),
        generator=generator,
        inner_batch_size=int(config["meta_inner_batch_size"]),
        outer_batch_size=int(config["meta_outer_batch_size"]),
        inner_learning_rate=float(config["meta_inner_learning_rate"]),
        outer_loss_weight=float(config["meta_outer_loss_weight"]),
    )
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "config_sha256": config_sha,
        "batch_size": 50,
        "joint_loss_components": {
            "parent": float(parent["total"].detach().cpu()),
            "s_seen_ce": float(s_loss.detach().cpu()),
            "vi_outer_ce": episode["outer_loss"],
        },
        "head_gradient_norms": _gradient_receipt(head),
        "source_gradient_count": len(_finite_source_gradients(source)),
        "graph": graph,
        "direction_skip_seen_class_ids": skipped,
        "meta_episode": episode,
        "test_tensors_used_for_gradient": False,
        "persistent_writes": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("TUNE014 expected commit错误。")
    config, config_sha = load_class_held_out_vi_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != config["experiment_id"]:
        raise ValueError("TUNE014 RUN身份错误。")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TUNE014正式RUN要求CUDA。")
    reproducibility = configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, source_config, head, graph, controls, skipped, seen_device, global_to_seen, centroids, folds, packages = _training_context(config, device)
    train_features = tensors["train_features"].to(device).float()
    train_labels = tensors["train_labels"].to(device).long()
    refresh = set(teacher_refresh_updates(train_count=len(train_features), nominal_epochs=200, batch_size=50))
    refresh_count = 1
    parent_optimizer = torch.optim.Adam(
        [
            {"params": list(source.parent.parameters()), "lr": float(source_config["tg_learning_rate"])},
            {"params": list(source.gate.parameters()), "lr": float(source_config["gate_learning_rate"])},
        ],
        weight_decay=float(source_config["weight_decay"]),
    )
    warmup = len(train_features) * int(source_config["gate_warmup_epochs"]) // 50
    parent_scheduler = GroupwiseSchedule(
        parent_optimizer,
        total_updates=int(config["total_updates"]),
        warmup_updates=warmup,
        tg_min_multiplier=float(source_config["tg_min_learning_rate"]) / float(source_config["tg_learning_rate"]),
        gate_min_multiplier=float(source_config["gate_min_learning_rate"]) / float(source_config["gate_learning_rate"]),
    )
    head_optimizer = torch.optim.Adam(head.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(7)
    output = prepare_output_dir(output_dir)
    (output / "config.snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log = (output / "training.log").open("w", encoding="utf-8", buffering=1)

    def emit(payload: dict) -> None:
        line = json.dumps(payload, sort_keys=True)
        print(line)
        log.write(line + "\n")

    history: list[dict] = []
    best: dict | None = None
    best_update = -1
    best_state = None
    best_source_state = None
    best_zs: dict | None = None
    interval = {"parent": 0.0, "s_seen_ce": 0.0, "vi_outer_ce": 0.0}
    steps = 0
    for update in range(1, int(config["total_updates"]) + 1):
        if update in refresh and update != 1:
            packages = refresh_oracle_targets(source, centroids, folds, float(source_config["theta_penalty"]))
            refresh_count += 1
        source.train()
        head.train()
        parent_scheduler.set_for_update(update)
        lr = _learning_rate(config, update)
        for group in head_optimizer.param_groups:
            group["lr"] = lr
        ids = torch.randperm(len(train_features), generator=generator)[:50].to(device)
        images = train_features[ids]
        labels = train_labels[ids]
        parent_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        parent = _parent_loss(
            source,
            images,
            labels,
            seen_device=seen_device,
            global_to_seen=global_to_seen,
            fold_package=packages[(update - 1) % 3],
            source_config=source_config,
        )
        s_loss = ordinary_seen_s_loss(head, images, labels, seen_device=seen_device, global_to_seen=global_to_seen)
        parent["total"].backward()
        s_loss.backward()
        _set_grad_none(head, META_PARAM_NAMES)
        episode = first_order_class_held_out_vi_episode(
            head,
            train_features,
            train_labels,
            pseudo_seen=folds[(update - 1) % 3][0],
            pseudo_unseen=folds[(update - 1) % 3][1],
            outer_candidate_classes=seen_device.detach().cpu(),
            generator=generator,
            inner_batch_size=int(config["meta_inner_batch_size"]),
            outer_batch_size=int(config["meta_outer_batch_size"]),
            inner_learning_rate=float(config["meta_inner_learning_rate"]),
            outer_loss_weight=float(config["meta_outer_loss_weight"]),
        )
        _gradient_receipt(head)
        _finite_source_gradients(source)
        parent_optimizer.step()
        head_optimizer.step()
        head.sync_source_prototypes(source)
        interval["parent"] += float(parent["total"].detach().cpu())
        interval["s_seen_ce"] += float(s_loss.detach().cpu())
        interval["vi_outer_ce"] += float(episode["outer_loss"])
        steps += 1
        if update % int(config["eval_interval_steps"]) != 0 and update != int(config["total_updates"]):
            continue
        ev = tune013.evaluate(head, source, tensors, controls, device)
        record = {
            "update": update,
            "head_lr": lr,
            "train_mean": {key: value / max(steps, 1) for key, value in interval.items()},
            "alpha": float(head.alpha().detach().cpu()),
            "role_weights": [float(value) for value in head.role_weights().detach().cpu()],
            **ev,
        }
        history.append(record)
        emit({"event": "evaluation", **record})
        interval = {key: 0.0 for key in interval}
        steps = 0
        if best is None or ev["metrics"]["full"]["H"] > best["metrics"]["full"]["H"]:
            best = copy.deepcopy(ev)
            best_update = update
            best_state = copy.deepcopy(head.state_dict())
            best_source_state = copy.deepcopy(source.state_dict())
        if best_zs is None or ev["metrics"]["full"]["ZS"] > best_zs["ZS"]:
            best_zs = {"update": update, "ZS": ev["metrics"]["full"]["ZS"], "metrics": copy.deepcopy(ev["metrics"]["full"])}
    if best_state is None or best_source_state is None or best is None or best_update <= 0:
        raise RuntimeError("TUNE014没有post-update best checkpoint。")
    source.load_state_dict(best_source_state)
    head.load_state_dict(best_state)
    final = tune013.evaluate(head, source, tensors, controls, device)
    passed, deltas = class_held_out_contract(final["metrics"], config["formal_full_metrics_percent"], config, best_update=best_update)
    export = head.export()
    checkpoint = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "best_update": best_update,
        "model_state_dict": best_state,
        "source_model_state_dict": best_source_state,
        "export": export.__dict__,
        "graph": graph,
    }
    atomic_torch_save(output / "model_best.pth", checkpoint)
    atomic_write_json(output / "evaluation_history.json", {"history": history})
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "source_config_sha256": config["source_config_sha256"],
        "formal_checkpoint_sha256": config["formal_checkpoint_sha256"],
        "best_update": best_update,
        "metrics": final["metrics"],
        "parent_metrics": final["parent_metrics"],
        "formal_full_metrics_percent": config["formal_full_metrics_percent"],
        "delta_H_vs_formal_full": float(final["metrics"]["full"]["H"] - float(config["formal_full_metrics_percent"]["H"])),
        "module_off_delta_H": deltas,
        "best_zs_observation": best_zs,
        "contract_passed": passed,
        "classification_ce_scope": config["classification_ce_scope"],
        "meta_algorithm": config["meta_algorithm"],
        "meta_second_order": False,
        "temporary_inner_optimizer_steps_only": True,
        "decision": "keep_tune014_class_held_out_vi" if passed else "drop_tune014_contract_failed",
        "graph": graph,
        "direction_skip_seen_class_ids": skipped,
        "teacher_refresh_count": refresh_count,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-config-sha")
    parser.add_argument("--micro-batch-only", action="store_true")
    args = parser.parse_args()
    if args.micro_batch_only:
        micro_batch(args.config)
        return
    if args.output_dir is None or not args.expected_commit or not args.expected_config_sha:
        parser.error("正式RUN缺少身份参数。")
    run(args.config, args.output_dir, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()
