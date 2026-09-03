"""Train fresh V7 seen-only CE S/V/I ablation conditions on CUB."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import yaml

from model.frameworks.v2 import train as h1
from model.frameworks.v4.train import (
    GroupwiseSchedule,
    rank_modulo_class_folds,
    refresh_oracle_targets,
    teacher_refresh_updates,
)
from model.frameworks.v6.compiled_pclr import CompiledPCLRExport, CompiledPCLRHead
from model.frameworks.v6.train_compiled_pclr import (
    _finite_source_gradients,
    _gradient_receipt,
    _learning_rate,
    _parent_loss,
)
from model.frameworks.v7 import train_one_text_seen_ce as tune013
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
    repo_path,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v7-seen-ce-retrained-ablation.v1"
EXPERIMENT_ID = "V7-ABLATION-002_SEEN_CE_RETRAINED_SVI"
FULL_REFERENCE_H = 79.94579718163422
FULL_REFERENCE_CONFIG_SHA = "cadceb65cb9596ea8f9769424154636938120691f306766c8c769168d1f711b3"
FULL_REFERENCE_CODE_COMMIT = "35cefc52896c383e1ec75a3adc5f78d218d616a3"
FULL_REFERENCE_METRICS_SHA = "08fd38f284eced2ccd4ea49e1faae1f1163ce5b3821eb7373f9acedf9201f34e"
FULL_REFERENCE_MODEL_SHA = "7349e869c1ffef41c3cfe265486880a1c455feb442ff326ae8403bd9a9a39328"


CONDITIONS = {
    "S-off": {
        "run_id": "RUN-S-OFF",
        "freeze_role_weights": True,
        "freeze_reader": False,
        "freeze_alpha": False,
        "semantic_enabled": False,
        "visual_enabled": True,
        "interaction_enabled": True,
        "direction_loss_enabled": True,
    },
    "V-off": {
        "run_id": "RUN-V-OFF",
        "freeze_role_weights": False,
        "freeze_reader": True,
        "freeze_alpha": False,
        "semantic_enabled": True,
        "visual_enabled": False,
        "interaction_enabled": True,
        "direction_loss_enabled": False,
    },
    "I-off": {
        "run_id": "RUN-I-OFF",
        "freeze_role_weights": False,
        "freeze_reader": False,
        "freeze_alpha": True,
        "semantic_enabled": True,
        "visual_enabled": True,
        "interaction_enabled": False,
        "direction_loss_enabled": True,
    },
    "V+I-off": {
        "run_id": "RUN-VI-OFF",
        "freeze_role_weights": False,
        "freeze_reader": True,
        "freeze_alpha": True,
        "semantic_enabled": True,
        "visual_enabled": False,
        "interaction_enabled": False,
        "direction_loss_enabled": False,
    },
}

CONFIG_KEYS = tune013.CONFIG_KEYS | {
    "ablation_experiment_id",
    "condition",
    "run_id",
    "full_reference_tune_experiment_id",
    "full_reference_code_commit",
    "full_reference_config",
    "full_reference_config_sha256",
    "full_reference_metrics_percent",
    "full_reference_metrics_uri",
    "full_reference_metrics_sha256",
    "full_reference_model_uri",
    "full_reference_model_sha256",
    "freeze_role_weights",
    "freeze_reader",
    "freeze_alpha",
    "semantic_enabled",
    "visual_enabled",
    "interaction_enabled",
    "direction_loss_enabled",
}


def load_ablation_config(path: Path) -> tuple[dict, str]:
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
        or config.get("source_config_sha256") != identity["source_config_sha256"]
        or config.get("formal_checkpoint_usage") != "baseline_identity_only_not_training_initialization"
        or not tune013._finite_metrics(config.get("formal_full_metrics_percent"))
        or config.get("full_reference_tune_experiment_id") != "V7-TUNE-013-CUB-ONE-TEXT-SEEN-CE"
        or config.get("full_reference_code_commit") != FULL_REFERENCE_CODE_COMMIT
        or config.get("full_reference_config_sha256") != FULL_REFERENCE_CONFIG_SHA
        or config.get("full_reference_metrics_percent") != {"H": FULL_REFERENCE_H}
        or config.get("full_reference_metrics_sha256") != FULL_REFERENCE_METRICS_SHA
        or config.get("full_reference_model_sha256") != FULL_REFERENCE_MODEL_SHA
        or config.get("device") not in {"cpu", "cuda", "cuda:0", "cuda:1"}
        or int(config.get("random_seed", -1)) != 7
        or int(config.get("batch_size", -1)) != 50
        or int(config.get("nominal_epochs", -1)) != 200
        or int(config.get("total_updates", -1)) != identity["total_updates"]
        or int(config.get("eval_interval_steps", -1)) != identity["eval_interval_steps"]
        or float(config.get("learning_rate", -1)) != 1e-4
        or float(config.get("min_learning_rate", -1)) != 1e-5
        or float(config.get("weight_decay", -1)) != 0.0
        or float(config.get("relation_loss_weight", -1)) != 1.0
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
        or config.get("classification_ce_scope") != "seen_only_train_classes"
        or config.get("expected_direction_skip_seen_class_ids") != identity["direction_skip_seen_class_ids"]
        or config.get("best_selection_metric") != "official_condition_H_post_update"
        or int(config.get("official_test_evaluations", -1)) != tune013._expected_eval_count(identity)
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
    )
    if invalid:
        raise ValueError("V7 seen-CE retrained ablation配置身份错误。")
    full_reference_config = repo_path(config["full_reference_config"])
    if not full_reference_config.is_file() or sha256_file(full_reference_config) != FULL_REFERENCE_CONFIG_SHA:
        raise ValueError("V7 seen-CE ablation Full reference config身份错误。")
    full_reference_metrics = repo_path(config["full_reference_metrics_uri"])
    if not full_reference_metrics.is_file() or sha256_file(full_reference_metrics) != FULL_REFERENCE_METRICS_SHA:
        raise ValueError("V7 seen-CE ablation Full reference metrics身份错误。")
    full_reference_model = repo_path(config["full_reference_model_uri"])
    if not full_reference_model.is_file() or sha256_file(full_reference_model) != FULL_REFERENCE_MODEL_SHA:
        raise ValueError("V7 seen-CE ablation Full reference model身份错误。")
    for name in (
        "freeze_role_weights",
        "freeze_reader",
        "freeze_alpha",
        "semantic_enabled",
        "visual_enabled",
        "interaction_enabled",
        "direction_loss_enabled",
    ):
        if config.get(name) is not condition[name]:
            raise ValueError(f"V7 seen-CE ablation {name}与condition合同不一致。")
    tune013._absolute_sha_file(config, "source_config")
    tune013._absolute_sha_file(config, "formal_checkpoint")
    return config, sha256_file(path)


def _condition(config: dict) -> dict:
    return CONDITIONS[str(config["condition"])]


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


def training_losses(
    head: CompiledPCLRHead,
    images: torch.Tensor,
    targets: torch.Tensor,
    config: dict,
    *,
    seen_device: torch.Tensor,
    global_to_seen: torch.Tensor,
) -> dict[str, torch.Tensor]:
    logits = condition_logits(head, images, config)
    classification = tune013.seen_only_classification_loss(
        logits,
        targets,
        seen_device=seen_device,
        global_to_seen=global_to_seen,
    )
    if _condition(config)["direction_loss_enabled"]:
        relation = head.relation_direction_loss(images, targets)
    else:
        relation = classification.sum() * 0.0
    total = classification + float(config["relation_loss_weight"]) * relation
    return {
        "total": total,
        "classification": classification,
        "relation": relation,
    }


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
    prototypes = torch.nn.functional.normalize(source.prototypes().float(), dim=-1)
    scale = source.scale().float()
    for split, features in (("seen", tensors["test_seen_features"]), ("unseen", tensors["test_unseen_features"])):
        for start in range(0, len(features), 256):
            images = features[start : start + 256].to(device).float()
            logits = condition_logits(head, images, config)
            outputs[split].append(logits.argmax(1).cpu())
            if split == "unseen":
                outputs["zs"].append(unseen[logits.index_select(1, unseen).argmax(1)].cpu())
            parent_logits = torch.nn.functional.normalize(images, dim=-1) @ prototypes.T * scale
            parent[split].append(parent_logits.argmax(1).cpu())
            if split == "unseen":
                parent["zs"].append(unseen[parent_logits.index_select(1, unseen).argmax(1)].cpu())
    for group in (outputs, parent):
        for split in group:
            group[split] = torch.cat(group[split])
    return {
        "condition": config["condition"],
        "metrics": _scores(outputs, tensors, seen, unseen_cpu),
        "parent_metrics": _scores(parent, tensors, seen, unseen_cpu),
    }


def _training_context(config: dict, device: torch.device):
    source, tensors, source_config, head, graph, _controls, skipped, seen_device, global_to_seen, centroids, folds, packages = tune013._training_context(config, device)
    apply_ablation_trainability(head, config)
    return source, tensors, source_config, head, graph, skipped, seen_device, global_to_seen, centroids, folds, packages


def micro_batch(config_path: Path) -> dict:
    config, config_sha = load_ablation_config(config_path)
    device = torch.device(config["device"])
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, source_config, head, graph, skipped, seen_device, global_to_seen, _centroids, _folds, packages = _training_context(config, device)
    images = tensors["train_features"][:50].to(device).float()
    labels = tensors["train_labels"][:50].to(device).long()
    parent = _parent_loss(source, images, labels, seen_device=seen_device, global_to_seen=global_to_seen, fold_package=packages[0], source_config=source_config)
    losses = training_losses(head, images, labels, config, seen_device=seen_device, global_to_seen=global_to_seen)
    total = parent["total"] + losses["total"]
    total.backward()
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "condition": config["condition"],
        "config_sha256": config_sha,
        "joint_loss": float(total.detach().cpu()),
        "parent_loss": float(parent["total"].detach().cpu()),
        "head_loss": float(losses["total"].detach().cpu()),
        "head_gradient_norms": gradient_receipt(head),
        "source_gradient_count": len(_finite_source_gradients(source)),
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
        raise ValueError("V7 seen-CE ablation expected commit错误。")
    config, config_sha = load_ablation_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != config["run_id"]:
        raise ValueError("V7 seen-CE ablation RUN身份错误。")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V7 seen-CE ablation正式RUN要求CUDA。")
    reproducibility = configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, source_config, head, graph, skipped, seen_device, global_to_seen, centroids, folds, packages = _training_context(config, device)
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
    head_optimizer = torch.optim.Adam(_head_parameters(head), lr=float(config["learning_rate"]), weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(7)
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
    best_source_state = None
    best_zs: dict | None = None
    interval = {"joint": 0.0, "parent": 0.0, "classification": 0.0, "relation": 0.0}
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
        losses = training_losses(head, images, labels, config, seen_device=seen_device, global_to_seen=global_to_seen)
        joint = parent["total"] + losses["total"]
        if not torch.isfinite(joint):
            raise FloatingPointError("V7 seen-CE ablation joint loss非有限。")
        joint.backward()
        gradient_receipt(head)
        _finite_source_gradients(source)
        parent_optimizer.step()
        head_optimizer.step()
        head.sync_source_prototypes(source)
        for key, value in (("joint", joint), ("parent", parent["total"]), ("classification", losses["classification"]), ("relation", losses["relation"])):
            interval[key] += float(value.detach().cpu())
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
            best_source_state = copy.deepcopy(source.state_dict())
        if best_zs is None or evaluation["metrics"]["ZS"] > best_zs["ZS"]:
            best_zs = {"update": update, "ZS": evaluation["metrics"]["ZS"], "metrics": copy.deepcopy(evaluation["metrics"])}
    if best_state is None or best_source_state is None or best is None:
        raise RuntimeError("V7 seen-CE ablation没有post-update best checkpoint。")
    source.load_state_dict(best_source_state)
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
        "source_model_state_dict": best_source_state,
        "export": export.__dict__,
        "graph": graph,
    }
    atomic_torch_save(output / "model_best.pth", checkpoint)
    atomic_write_json(output / "evaluation_history.json", {"history": history})
    delta_h = float(config["full_reference_metrics_percent"]["H"] - final["metrics"]["H"])
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "ablation_experiment_id": EXPERIMENT_ID,
        "condition": config["condition"],
        "dataset": config["dataset"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "source_config_sha256": config["source_config_sha256"],
        "best_update": best_update,
        "metrics": final["metrics"],
        "parent_metrics": final["parent_metrics"],
        "full_reference": {
            "experiment_id": config["full_reference_tune_experiment_id"],
            "code_commit": config["full_reference_code_commit"],
            "config_sha256": config["full_reference_config_sha256"],
            "metrics_percent": config["full_reference_metrics_percent"],
            "metrics_uri": config["full_reference_metrics_uri"],
            "metrics_sha256": config["full_reference_metrics_sha256"],
            "model_uri": config["full_reference_model_uri"],
            "model_sha256": config["full_reference_model_sha256"],
        },
        "delta_H_full_reference_minus_condition": delta_h,
        "condition_contract": _condition(config),
        "best_zs_observation": best_zs,
        "decision": "diagnose_retrained_ablation",
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
