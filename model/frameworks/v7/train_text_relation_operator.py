"""Train TUNE015 text-relation visual operator on CUB.

The TUNE015 head has no per-image Reader.  Training learns a shared
identity-residual low-rank operator from fixed text edge directions to
trainval-seen visual centroid differences, then uses the compiled class matrix
for seen-only CE.
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

from model.frameworks.v2 import train as h1
from model.frameworks.v4.train import (
    GroupwiseSchedule,
    build_model,
    load_assets,
    load_config,
    rank_modulo_class_folds,
    refresh_oracle_targets,
    teacher_refresh_updates,
)
from model.frameworks.v6.train_compiled_pclr import (
    _finite_source_gradients,
    _gradient_receipt,
    _learning_rate,
    _parent_loss,
)
from model.frameworks.v7.relation_operator import (
    TextRelationOperatorHead,
    text_relation_graph,
    visual_edge_targets,
)
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


SCHEMA = "gzsl-paper.v7-text-relation-operator.v1"
BASE_COMMIT = "35cefc52896c383e1ec75a3adc5f78d218d616a3"
INITIAL_ALPHA = 0.7258594751358033
INITIAL_ROLE_WEIGHTS = [0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.36, 0.0]
IDENTITIES = {
    "CUB": {
        "experiment_id": "V7-TUNE-015-CUB-TEXT-RELATION-OPERATOR",
        "source_config_sha256": "0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49",
        "seen_count": 150,
        "class_count": 200,
        "total_updates": 28228,
        "eval_interval_steps": 141,
        "seen_logit_gamma": 0.91,
    },
}
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "dataset",
    "base_commit",
    "source_config",
    "source_config_sha256",
    "formal_checkpoint",
    "formal_checkpoint_sha256",
    "formal_checkpoint_usage",
    "formal_full_metrics_percent",
    "device",
    "random_seed",
    "batch_size",
    "nominal_epochs",
    "total_updates",
    "eval_interval_steps",
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "relation_loss_weight",
    "ridge_lambda",
    "relation_temperature",
    "top_k",
    "seen_logit_gamma",
    "alpha_max",
    "initial_alpha",
    "role_weight_max",
    "initial_role_weights",
    "operator_rank",
    "operator_init_std",
    "operator_residual_mode",
    "relation_embedding_mode",
    "classification_ce_scope",
    "direction_alignment_source",
    "best_selection_metric",
    "official_test_evaluations",
    "required_i_off_delta_h",
    "require_full_not_below_formal",
    "fresh_source_initialization",
    "test_used_for_selection",
    "test_used_for_hyperparameter_selection",
    "nested_official_test_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "human_annotations_used",
    "expert_attributes_used",
    "llm_world_knowledge_used",
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
        raise ValueError(f"TUNE015 {key}必须是存在的绝对文件。")
    if sha256_file(path) != config[f"{key}_sha256"]:
        raise ValueError(f"TUNE015 {key} SHA不匹配。")
    return path


def load_text_relation_operator_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    identity = IDENTITIES.get(config.get("dataset")) if isinstance(config, dict) else None
    invalid = (
        not isinstance(config, dict)
        or set(config) != CONFIG_KEYS
        or identity is None
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != identity["experiment_id"]
        or config.get("base_commit") != BASE_COMMIT
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
        or float(config.get("relation_loss_weight", -1)) != 1.0
        or float(config.get("ridge_lambda", -1)) != 0.3
        or float(config.get("relation_temperature", -1)) != 0.2
        or int(config.get("top_k", -1)) != 3
        or float(config.get("seen_logit_gamma", -1)) != identity["seen_logit_gamma"]
        or float(config.get("alpha_max", -1)) != 2.0
        or abs(float(config.get("initial_alpha", -1)) - INITIAL_ALPHA) > 1e-12
        or float(config.get("role_weight_max", -1)) != 1.0
        or config.get("initial_role_weights") != INITIAL_ROLE_WEIGHTS
        or int(config.get("operator_rank", -1)) != 32
        or float(config.get("operator_init_std", -1)) != 0.01
        or config.get("operator_residual_mode") != "identity_residual_unit_normalized"
        or config.get("relation_embedding_mode") != "one_text_direction_to_visual_centroid_difference"
        or config.get("classification_ce_scope") != "seen_only_train_classes"
        or config.get("direction_alignment_source") != "trainval_seen_visual_centroids_only"
        or config.get("best_selection_metric") != "official_full_H_post_update"
        or int(config.get("official_test_evaluations", -1)) != _expected_eval_count(identity)
        or float(config.get("required_i_off_delta_h", -1)) != 0.0
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
    )
    if invalid:
        raise ValueError("TUNE015 text-relation operator配置身份错误。")
    _absolute_sha_file(config, "source_config")
    _absolute_sha_file(config, "formal_checkpoint")
    return config, sha256_file(path)


def load_training_source(config: dict, device: torch.device):
    source_path = _absolute_sha_file(config, "source_config")
    source_config, source_sha = load_config(source_path)
    if source_sha != config["source_config_sha256"] or source_config["dataset"] != config["dataset"]:
        raise ValueError("TUNE015 source config身份错误。")
    tensors = load_assets(source_config)
    source = build_model(source_config, tensors, device)
    source.requires_grad_(False)
    for parameter in tuple(source.parent.parameters()) + tuple(source.gate.parameters()):
        parameter.requires_grad_(True)
    return source, tensors, source_config


@torch.no_grad()
def build_head(
    source,
    tensors: dict,
    config: dict,
    device: torch.device,
) -> tuple[TextRelationOperatorHead, dict]:
    was_training = source.training
    source.eval()
    try:
        text_directions, edges, graph = text_relation_graph(
            source.parent.tg_vpr.sentence_embeds,
            top_k=int(config["top_k"]),
        )
        labels = tensors["train_labels"].long()
        seen = torch.unique(labels, sorted=True)
        centroids = h1.visual_centroids(tensors["train_features"], labels, seen)
        seen_edge_mask, visual_targets = visual_edge_targets(
            centroids,
            seen,
            edges,
            int(IDENTITIES[config["dataset"]]["class_count"]),
        )
        head = TextRelationOperatorHead(
            base_prototypes=source.prototypes(),
            role_prototypes=source.parent.tg_vpr.sentence_embeds,
            text_directions=text_directions,
            edge_index=edges,
            seen_classes=source.seen_classes,
            visual_seen_edge_mask=seen_edge_mask,
            visual_seen_targets=visual_targets,
            scale=float(source.scale()),
            ridge_lambda=float(config["ridge_lambda"]),
            relation_temperature=float(config["relation_temperature"]),
            seen_logit_gamma=float(config["seen_logit_gamma"]),
            alpha_max=float(config["alpha_max"]),
            initial_alpha=float(config["initial_alpha"]),
            role_weight_max=float(config["role_weight_max"]),
            initial_role_weights=torch.tensor(config["initial_role_weights"]),
            operator_rank=int(config["operator_rank"]),
            operator_init_std=float(config["operator_init_std"]),
        ).to(device)
    finally:
        source.train(was_training)
    graph = {
        **graph,
        "seen_seen_edge_count": int(head.visual_seen_edge_mask.sum().detach().cpu()),
        "unseen_touching_edge_count": int((~head.visual_seen_edge_mask).sum().detach().cpu()),
        "operator_rank": int(config["operator_rank"]),
        "reader_removed": True,
    }
    return head, graph


def validate_training_identity(head: TextRelationOperatorHead, tensors: dict, identity: dict) -> None:
    labels = tensors["train_labels"].long()
    seen = torch.unique(labels, sorted=True)
    if seen.numel() != identity["seen_count"]:
        raise ValueError("TUNE015 trainval seen类别数不匹配。")
    if int(seen.min()) < 0 or int(seen.max()) >= identity["class_count"]:
        raise ValueError("TUNE015 trainval类别ID超出类别轴。")
    if head.class_count != identity["class_count"] or head.seen_classes.cpu().numel() != identity["seen_count"]:
        raise ValueError("TUNE015 head类别身份不匹配。")
    if int(head.visual_seen_edge_mask.sum()) <= 0:
        raise ValueError("TUNE015 缺少seen-seen边视觉监督。")
    if int((~head.visual_seen_edge_mask).sum()) <= 0:
        raise ValueError("TUNE015 缺少unseen相关边，无法证明共享A外推。")


def _condition_logits(head: TextRelationOperatorHead, images: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "full": head(images),
        "s_off": head(images, semantic_enabled=False),
        "i_off": head(images, interaction_enabled=False),
    }


def _scores(predictions: dict[str, torch.Tensor], tensors: dict, seen: torch.Tensor, unseen: torch.Tensor) -> dict[str, float]:
    s = 100.0 * per_class_accuracy(tensors["test_seen_labels"], predictions["seen"], seen)
    u = 100.0 * per_class_accuracy(tensors["test_unseen_labels"], predictions["unseen"], unseen)
    zs = 100.0 * per_class_accuracy(tensors["test_unseen_labels"], predictions["zs"], unseen)
    h = 2.0 * s * u / (s + u) if s + u else 0.0
    return {"U": float(u), "S": float(s), "H": float(h), "ZS": float(zs)}


@torch.no_grad()
def evaluate(head: TextRelationOperatorHead, source, tensors: dict, device: torch.device) -> dict:
    head.eval()
    source.eval()
    seen = head.seen_classes.detach().cpu()
    all_classes = torch.arange(head.class_count)
    unseen_cpu = all_classes[~torch.isin(all_classes, seen)]
    unseen = unseen_cpu.to(device)
    outputs = {name: {"seen": [], "unseen": [], "zs": []} for name in ("full", "s_off", "i_off")}
    parent = {"seen": [], "unseen": [], "zs": []}
    prototypes = F.normalize(source.prototypes().float(), dim=-1)
    scale = source.scale().float()
    for split, features in (("seen", tensors["test_seen_features"]), ("unseen", tensors["test_unseen_features"])):
        for start in range(0, len(features), 256):
            images = features[start : start + 256].to(device).float()
            for name, logits in _condition_logits(head, images).items():
                outputs[name][split].append(logits.argmax(1).cpu())
                if split == "unseen":
                    outputs[name]["zs"].append(unseen[logits.index_select(1, unseen).argmax(1)].cpu())
            parent_logits = F.normalize(images, dim=-1) @ prototypes.T * scale
            parent[split].append(parent_logits.argmax(1).cpu())
            if split == "unseen":
                parent["zs"].append(unseen[parent_logits.index_select(1, unseen).argmax(1)].cpu())
    for group in (*outputs.values(), parent):
        for split in group:
            group[split] = torch.cat(group[split])
    return {
        "metrics": {name: _scores(value, tensors, seen, unseen_cpu) for name, value in outputs.items()},
        "parent_metrics": _scores(parent, tensors, seen, unseen_cpu),
    }


def contract(metrics: dict[str, dict[str, float]], formal: dict[str, float], config: dict) -> tuple[bool, dict]:
    full_h = float(metrics["full"]["H"])
    deltas = {name: float(full_h - metrics[name]["H"]) for name in ("s_off", "i_off")}
    passed = (
        full_h >= float(formal["H"])
        and deltas["i_off"] >= float(config["required_i_off_delta_h"])
    )
    return passed, deltas


def _training_context(config: dict, device: torch.device):
    identity = IDENTITIES[config["dataset"]]
    source, tensors, source_config = load_training_source(config, device)
    head, graph = build_head(source, tensors, config, device)
    validate_training_identity(head, tensors, identity)
    labels_cpu = tensors["train_labels"].long()
    seen = torch.unique(labels_cpu, sorted=True)
    seen_device = seen.to(device)
    global_to_seen = torch.full((identity["class_count"],), -1, dtype=torch.long, device=device)
    global_to_seen[seen_device] = torch.arange(len(seen), device=device)
    centroids = h1.visual_centroids(tensors["train_features"], labels_cpu, seen).to(device)
    folds = rank_modulo_class_folds(seen)
    packages = refresh_oracle_targets(source, centroids, folds, float(source_config["theta_penalty"]))
    return source, tensors, source_config, head, graph, seen_device, global_to_seen, centroids, folds, packages


def micro_batch(config_path: Path) -> dict:
    config, config_sha = load_text_relation_operator_config(config_path)
    device = torch.device(config["device"])
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, source_config, head, graph, seen_device, global_to_seen, _centroids, _folds, packages = _training_context(config, device)
    images = tensors["train_features"][:50].to(device).float()
    labels = tensors["train_labels"][:50].to(device).long()
    parent = _parent_loss(
        source,
        images,
        labels,
        seen_device=seen_device,
        global_to_seen=global_to_seen,
        fold_package=packages[0],
        source_config=source_config,
    )
    losses = head.training_losses(
        images,
        labels,
        seen_device=seen_device,
        global_to_seen=global_to_seen,
        relation_loss_weight=float(config["relation_loss_weight"]),
    )
    total = parent["total"] + losses["total"]
    total.backward()
    export = head.export()
    result = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "config_sha256": config_sha,
        "batch_size": 50,
        "joint_loss": float(total.detach().cpu()),
        "parent_loss": float(parent["total"].detach().cpu()),
        "head_loss": float(losses["total"].detach().cpu()),
        "head_gradient_norms": _gradient_receipt(head),
        "source_gradient_count": len(_finite_source_gradients(source)),
        "export_keys": sorted(export.__dict__),
        "q_shape": list(export.q.shape),
        "graph": graph,
        "persistent_writes": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("TUNE015 expected commit错误。")
    config, config_sha = load_text_relation_operator_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != config["experiment_id"]:
        raise ValueError("TUNE015 RUN身份错误。")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TUNE015正式RUN要求CUDA。")
    reproducibility = configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, source_config, head, graph, seen_device, global_to_seen, centroids, folds, packages = _training_context(config, device)
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
        losses = head.training_losses(
            images,
            labels,
            seen_device=seen_device,
            global_to_seen=global_to_seen,
            relation_loss_weight=float(config["relation_loss_weight"]),
        )
        joint = parent["total"] + losses["total"]
        if not torch.isfinite(joint):
            raise FloatingPointError("TUNE015 joint loss非有限。")
        joint.backward()
        _gradient_receipt(head)
        _finite_source_gradients(source)
        parent_optimizer.step()
        head_optimizer.step()
        head.sync_source_prototypes(source)
        for key, value in (("joint", joint), ("parent", parent["total"]), ("classification", losses["classification"]), ("relation", losses["relation"])):
            interval[key] += float(value.detach().cpu())
        steps += 1
        if update % int(config["eval_interval_steps"]) != 0 and update != int(config["total_updates"]):
            continue
        ev = evaluate(head, source, tensors, device)
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
    if best_state is None or best_source_state is None or best is None:
        raise RuntimeError("TUNE015没有post-update best checkpoint。")
    source.load_state_dict(best_source_state)
    head.load_state_dict(best_state)
    final = evaluate(head, source, tensors, device)
    passed, deltas = contract(final["metrics"], config["formal_full_metrics_percent"], config)
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
        "direction_alignment_source": config["direction_alignment_source"],
        "decision": "keep_tune015_text_relation_operator" if passed else "drop_tune015_contract_failed",
        "graph": graph,
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
