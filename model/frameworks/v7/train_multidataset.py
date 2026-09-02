"""Run fixed-200 FRAMEWORK-V7 confirmation on AWA2 or SUN."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
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
from model.frameworks.v6.compiled_pclr import CompiledPCLRHead
from model.frameworks.v6.train_compiled_pclr import (
    _finite_source_gradients,
    _gradient_receipt,
    _learning_rate,
    _parent_loss,
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


SCHEMA = "gzsl-paper.v7-multidataset-confirm.v1"
ASSET_SCHEMA = "gzsl-paper.v7-relation-asset.v1"
BASE_COMMIT = "b32a16f848c34f8e09d03b27d2f22ed445b9a295"
IDENTITIES = {
    "AWA2": {
        "experiment_id": "V7-CONFIRM-001-AWA2",
        "source_config_sha256": "44976e5e77a907112a88d295ee70296a2a22ef1c15c5b1f1d286249e1451a529",
        "train_count": 23527,
        "seen_count": 40,
        "class_count": 50,
        "total_updates": 94108,
        "eval_interval_steps": 470,
        "seen_logit_gamma": 0.05,
        "direction_skip_seen_class_ids": [],
    },
    "SUN": {
        "experiment_id": "V7-CONFIRM-001-SUN",
        "source_config_sha256": "75a4035f783e92ba2cc70c3e7a633791abb80f3844a4f7b29ce1d172b4935cf1",
        "train_count": 10320,
        "seen_count": 645,
        "class_count": 717,
        "total_updates": 41280,
        "eval_interval_steps": 206,
        "seen_logit_gamma": 0.15,
        "direction_skip_seen_class_ids": [102],
    },
}
CONFIG_KEYS = {
    "schema_version", "experiment_id", "dataset", "base_commit",
    "source_config", "source_config_sha256", "relation_manifest",
    "relation_manifest_sha256", "device", "random_seed", "batch_size",
    "nominal_epochs", "total_updates", "eval_interval_steps", "learning_rate",
    "min_learning_rate", "weight_decay", "relation_loss_weight", "ridge_lambda",
    "relation_temperature", "direction_temperature", "seen_logit_gamma",
    "alpha_max", "initial_alpha", "role_weight_max", "initial_role_weights",
    "candidate_top_k", "required_module_delta_h", "max_us_gap",
    "direction_skip_seen_class_ids", "test_used_for_selection",
    "test_used_for_hyperparameter_selection", "nested_official_test_selection",
    "unseen_images_used_for_gradient", "strict_blind_claim",
    "human_annotations_used", "expert_attributes_used", "llm_world_knowledge_used",
}


def load_confirmation_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    identity = IDENTITIES.get(config.get("dataset")) if isinstance(config, dict) else None
    invalid = (
        not isinstance(config, dict) or set(config) != CONFIG_KEYS or identity is None
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != identity["experiment_id"]
        or config.get("base_commit") != BASE_COMMIT
        or config.get("source_config_sha256") != identity["source_config_sha256"]
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
        or float(config.get("seen_logit_gamma", -1)) != identity["seen_logit_gamma"]
        or float(config.get("alpha_max", -1)) != 2.0
        or abs(float(config.get("initial_alpha", -1)) - 0.7258594751358033) > 1e-12
        or float(config.get("role_weight_max", -1)) != 1.0
        or config.get("initial_role_weights") != [0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.36, 0.0]
        or config.get("candidate_top_k") is not None
        or float(config.get("required_module_delta_h", -1)) != 1.0
        or float(config.get("max_us_gap", -1)) != 8.0
        or config.get("direction_skip_seen_class_ids") != identity["direction_skip_seen_class_ids"]
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not True
        or config.get("nested_official_test_selection") is not True
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
        or config.get("expert_attributes_used") is not False
        or config.get("llm_world_knowledge_used") is not True
    )
    if invalid:
        raise ValueError("V7多数据集确认配置身份错误。")
    return config, sha256_file(path)


def _absolute_sha_file(config: dict, key: str) -> Path:
    path = Path(config[key])
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != config[f"{key}_sha256"]:
        raise ValueError(f"V7多数据集{key}身份错误。")
    return path


def load_relation_asset(config: dict, identity: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    manifest_path = _absolute_sha_file(config, "relation_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("schema_version") != ASSET_SCHEMA
        or manifest.get("dataset") != config["dataset"]
        or manifest.get("class_count") != identity["class_count"]
        or manifest.get("seen_count") != identity["seen_count"]
        or manifest.get("human_annotations_used") is not False
        or manifest.get("llm_world_knowledge_used") is not True
        or manifest.get("visible_only_annotation") is not True
        or not isinstance(outputs, dict)
    ):
        raise ValueError("V7多数据集关系manifest合同错误。")
    for name in ("relation_sentence_embeds.pt", "edge_index.pt", "relation_texts.json"):
        if sha256_file(manifest_path.parent / name) != outputs.get(name):
            raise ValueError(f"V7多数据集关系资产SHA错误：{name}")
    relations = torch.load(manifest_path.parent / "relation_sentence_embeds.pt", map_location="cpu", weights_only=True)
    edges = torch.load(manifest_path.parent / "edge_index.pt", map_location="cpu", weights_only=True)
    edge_count = int(manifest["edge_count"])
    if (
        tuple(relations.shape) != (edge_count, 2, 768)
        or tuple(edges.shape) != (edge_count, 2)
        or edges.dtype != torch.int64
        or not torch.isfinite(relations).all()
        or int(edges.min()) < 0
        or int(edges.max()) >= identity["class_count"]
    ):
        raise ValueError("V7多数据集关系张量合同错误。")
    return relations, edges, manifest


def load_training_source(config: dict, device: torch.device):
    source_path = _absolute_sha_file(config, "source_config")
    source_config, source_sha = load_config(source_path)
    if source_sha != config["source_config_sha256"] or source_config["dataset"] != config["dataset"]:
        raise ValueError("V7多数据集source config身份错误。")
    tensors = load_assets(source_config)
    source = build_model(source_config, tensors, device)
    source.requires_grad_(False)
    for parameter in tuple(source.parent.parameters()) + tuple(source.gate.parameters()):
        parameter.requires_grad_(True)
    return source, tensors, source_config


def validate_training_identity(source, tensors: dict, identity: dict, configured_skips: list[int]) -> None:
    labels = tensors["train_labels"].long()
    seen = torch.unique(labels, sorted=True)
    if len(labels) != identity["train_count"] or seen.numel() != identity["seen_count"]:
        raise ValueError("V7多数据集trainval split身份错误。")
    if int(seen.min()) < 0 or int(seen.max()) >= identity["class_count"]:
        raise ValueError("V7多数据集训练标签超出类别轴。")
    edges = source.edge_index.detach().cpu()
    seen_mask = torch.zeros(identity["class_count"], dtype=torch.bool)
    seen_mask[seen] = True
    seen_edges = edges[seen_mask[edges[:, 0]] & seen_mask[edges[:, 1]]]
    covered = torch.unique(seen_edges.reshape(-1)) if seen_edges.numel() else torch.empty(0, dtype=torch.long)
    uncovered = seen[~torch.isin(seen, covered)].tolist()
    if uncovered != configured_skips:
        raise ValueError(f"V7多数据集方向CE跳过类别不匹配：{uncovered}")


@torch.no_grad()
def build_head(source, relations, edges, config: dict, device: torch.device) -> CompiledPCLRHead:
    was_training = source.training
    source.eval()
    try:
        head = CompiledPCLRHead(
            base_prototypes=source.prototypes(),
            role_prototypes=source.parent.tg_vpr.sentence_embeds,
            relation_embeddings=relations,
            edge_index=edges,
            seen_classes=source.seen_classes,
            scale=float(source.scale()),
            reader_in_state=(source.reader_in.weight, source.reader_in.bias),
            reader_out_state=(source.reader_out.weight, source.reader_out.bias),
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
    return head


def _condition_logits(head, images):
    return {
        "full": head(images),
        "s_off": head(images, semantic_enabled=False),
        "v_off": head(images, visual_enabled=False),
        "i_off": head(images, interaction_enabled=False),
    }


def _scores(predictions, tensors, seen, unseen):
    s = 100.0 * per_class_accuracy(tensors["test_seen_labels"], predictions["seen"], seen)
    u = 100.0 * per_class_accuracy(tensors["test_unseen_labels"], predictions["unseen"], unseen)
    zs = 100.0 * per_class_accuracy(tensors["test_unseen_labels"], predictions["zs"], unseen)
    h = 2.0 * s * u / (s + u) if s + u else 0.0
    return {"U": float(u), "S": float(s), "H": float(h), "ZS": float(zs)}


@torch.no_grad()
def evaluate(head, source, tensors, device):
    head.eval(); source.eval()
    seen = head.seen_classes.cpu()
    all_classes = torch.arange(head.class_count)
    unseen_cpu = all_classes[~torch.isin(all_classes, seen)]
    unseen = unseen_cpu.to(device)
    outputs = {name: {"seen": [], "unseen": [], "zs": []} for name in ("full", "s_off", "v_off", "i_off")}
    parent = {"seen": [], "unseen": [], "zs": []}
    prototypes = torch.nn.functional.normalize(source.prototypes().float(), dim=-1)
    scale = source.scale().float()
    for split, features in (("seen", tensors["test_seen_features"]), ("unseen", tensors["test_unseen_features"])):
        for start in range(0, len(features), 256):
            images = features[start:start+256].to(device).float()
            for name, logits in _condition_logits(head, images).items():
                outputs[name][split].append(logits.argmax(1).cpu())
                if split == "unseen":
                    outputs[name]["zs"].append(unseen[logits.index_select(1, unseen).argmax(1)].cpu())
            parent_logits = torch.nn.functional.normalize(images, dim=-1) @ prototypes.T * scale
            parent[split].append(parent_logits.argmax(1).cpu())
            if split == "unseen":
                parent["zs"].append(unseen[parent_logits.index_select(1, unseen).argmax(1)].cpu())
    for group in (*outputs.values(), parent):
        for split in group: group[split] = torch.cat(group[split])
    return {
        "metrics": {name: _scores(value, tensors, seen, unseen_cpu) for name, value in outputs.items()},
        "parent_metrics": _scores(parent, tensors, seen, unseen_cpu),
    }


def _contract(metrics, parent_best_h, best_update, config):
    full = metrics["full"]
    deltas = {name: float(full["H"] - metrics[name]["H"]) for name in ("s_off", "v_off", "i_off")}
    passed = (
        int(best_update) > 0 and float(full["H"]) > float(parent_best_h)
        and all(value >= float(config["required_module_delta_h"]) for value in deltas.values())
        and abs(float(full["U"] - full["S"])) < float(config["max_us_gap"])
    )
    return passed, deltas


def micro_batch(config_path: Path) -> dict:
    config, config_sha = load_confirmation_config(config_path)
    device = torch.device(config["device"])
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    identity = IDENTITIES[config["dataset"]]
    source, tensors, source_config = load_training_source(config, device)
    relations, edges, manifest = load_relation_asset(config, identity)
    head = build_head(source, relations, edges, config, device)
    validate_training_identity(head, tensors, identity, config["direction_skip_seen_class_ids"])
    labels_cpu = tensors["train_labels"].long(); seen = torch.unique(labels_cpu, sorted=True); seen_device = seen.to(device)
    global_to_seen = torch.full((identity["class_count"],), -1, dtype=torch.long, device=device)
    global_to_seen[seen_device] = torch.arange(len(seen), device=device)
    centroids = h1.visual_centroids(tensors["train_features"], labels_cpu, seen).to(device)
    packages = refresh_oracle_targets(source, centroids, rank_modulo_class_folds(seen), float(source_config["theta_penalty"]))
    images = tensors["train_features"][:50].to(device).float(); labels = labels_cpu[:50].to(device)
    parent = _parent_loss(source, images, labels, seen_device=seen_device, global_to_seen=global_to_seen, fold_package=packages[0], source_config=source_config)
    losses = head.training_losses(images, labels, relation_loss_weight=1.0)
    total = parent["total"] + losses["total"]
    total.backward(); head_gradients = _gradient_receipt(head); source_gradients = _finite_source_gradients(source)
    result = {
        "experiment_id": config["experiment_id"], "dataset": config["dataset"], "config_sha256": config_sha,
        "asset_id": manifest["asset_id"], "batch_size": 50, "joint_loss": float(total.detach().cpu()),
        "head_gradient_norms": head_gradients, "source_gradient_count": len(source_gradients),
        "q_shape": list(head.export().q.shape), "direction_skip_seen_class_ids": config["direction_skip_seen_class_ids"],
        "finite": True, "persistent_writes": False,
    }
    print(json.dumps(result, sort_keys=True)); return result


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str):
    require_clean_code_tree(); code_commit = current_code_commit()
    if code_commit != expected_commit: raise ValueError("V7多数据集expected commit错误。")
    config, config_sha = load_confirmation_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != config["experiment_id"]:
        raise ValueError("V7多数据集RUN身份错误。")
    device = torch.device(config["device"]); identity = IDENTITIES[config["dataset"]]
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V7多数据集正式RUN要求CUDA。")
    reproducibility = configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    source, tensors, source_config = load_training_source(config, device)
    relations, edges, manifest = load_relation_asset(config, identity)
    head = build_head(source, relations, edges, config, device)
    validate_training_identity(head, tensors, identity, config["direction_skip_seen_class_ids"])
    train_features = tensors["train_features"].to(device).float(); train_labels = tensors["train_labels"].to(device).long()
    seen = torch.unique(train_labels.cpu(), sorted=True); seen_device = seen.to(device)
    global_to_seen = torch.full((identity["class_count"],), -1, dtype=torch.long, device=device)
    global_to_seen[seen_device] = torch.arange(len(seen), device=device)
    centroids = h1.visual_centroids(tensors["train_features"], tensors["train_labels"].long(), seen).to(device)
    folds = rank_modulo_class_folds(seen)
    refresh = set(teacher_refresh_updates(train_count=len(train_features), nominal_epochs=200, batch_size=50))
    packages = refresh_oracle_targets(source, centroids, folds, float(source_config["theta_penalty"])); refresh_count = 1
    parent_parameters=list(source.parent.parameters()); gate_parameters=list(source.gate.parameters())
    parent_optimizer=torch.optim.Adam([{"params":parent_parameters,"lr":float(source_config["tg_learning_rate"])},{"params":gate_parameters,"lr":float(source_config["gate_learning_rate"])}],weight_decay=float(source_config["weight_decay"]))
    warmup=len(train_features)*int(source_config["gate_warmup_epochs"])//50
    parent_scheduler=GroupwiseSchedule(parent_optimizer,total_updates=int(config["total_updates"]),warmup_updates=warmup,tg_min_multiplier=float(source_config["tg_min_learning_rate"])/float(source_config["tg_learning_rate"]),gate_min_multiplier=float(source_config["gate_min_learning_rate"])/float(source_config["gate_learning_rate"]))
    head_optimizer=torch.optim.Adam(head.parameters(),lr=float(config["learning_rate"]),weight_decay=0.0)
    generator=torch.Generator(device="cpu").manual_seed(7)
    output=prepare_output_dir(output_dir); (output/"config.snapshot.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    log=(output/"training.log").open("w",encoding="utf-8",buffering=1)
    def emit(v):
        line=json.dumps(v,sort_keys=True); print(line); log.write(line+"\n")
    initial=evaluate(head,source,tensors,device); history=[{"update":0,**initial}]; emit({"event":"initial","update":0,**initial})
    best=None; best_update=-1; best_state=None; best_source_state=None; parent_best=None; parent_best_update=-1
    best_zs={"update":0,"ZS":initial["metrics"]["full"]["ZS"],"metrics":initial["metrics"]["full"]}
    parent_best_zs={"update":0,"ZS":initial["parent_metrics"]["ZS"],"metrics":initial["parent_metrics"]}
    interval={"joint":0.0,"parent":0.0,"classification":0.0,"relation":0.0}; steps=0
    for update in range(1,int(config["total_updates"])+1):
        if update in refresh and update!=1:
            packages=refresh_oracle_targets(source,centroids,folds,float(source_config["theta_penalty"])); refresh_count+=1
        source.train(); head.train(); parent_scheduler.set_for_update(update); lr=_learning_rate(config,update)
        for group in head_optimizer.param_groups: group["lr"]=lr
        ids=torch.randperm(len(train_features),generator=generator)[:50].to(device); images=train_features[ids]; labels=train_labels[ids]
        parent_optimizer.zero_grad(set_to_none=True); head_optimizer.zero_grad(set_to_none=True)
        parent=_parent_loss(source,images,labels,seen_device=seen_device,global_to_seen=global_to_seen,fold_package=packages[(update-1)%3],source_config=source_config)
        losses=head.training_losses(images,labels,relation_loss_weight=1.0); joint=parent["total"]+losses["total"]
        if not torch.isfinite(joint): raise FloatingPointError("V7多数据集loss非有限。")
        joint.backward(); _gradient_receipt(head); _finite_source_gradients(source); parent_optimizer.step(); head_optimizer.step(); head.sync_source_prototypes(source)
        for key,val in (("joint",joint),("parent",parent["total"]),("classification",losses["classification"]),("relation",losses["relation"])): interval[key]+=float(val.detach().cpu())
        steps+=1
        if update%int(config["eval_interval_steps"])!=0 and update!=int(config["total_updates"]): continue
        ev=evaluate(head,source,tensors,device); record={"update":update,"head_lr":lr,"train_mean":{k:v/steps for k,v in interval.items()},"alpha":float(head.alpha().detach().cpu()),"role_weights":[float(v) for v in head.role_weights().detach().cpu()],**ev}; history.append(record); emit({"event":"evaluation",**record}); interval={k:0.0 for k in interval}; steps=0
        if best is None or ev["metrics"]["full"]["H"]>best["metrics"]["full"]["H"]:
            best=copy.deepcopy(ev); best_update=update; best_state=copy.deepcopy(head.state_dict()); best_source_state=copy.deepcopy(source.state_dict())
        if parent_best is None or ev["parent_metrics"]["H"]>parent_best["H"]:
            parent_best=copy.deepcopy(ev["parent_metrics"]); parent_best_update=update
        if ev["metrics"]["full"]["ZS"]>best_zs["ZS"]: best_zs={"update":update,"ZS":ev["metrics"]["full"]["ZS"],"metrics":copy.deepcopy(ev["metrics"]["full"])}
        if ev["parent_metrics"]["ZS"]>parent_best_zs["ZS"]: parent_best_zs={"update":update,"ZS":ev["parent_metrics"]["ZS"],"metrics":copy.deepcopy(ev["parent_metrics"])}
    if best_state is None or best_source_state is None or parent_best is None: raise RuntimeError("V7多数据集没有best checkpoint。")
    source.load_state_dict(best_source_state)
    head.load_state_dict(best_state)
    final=evaluate(head,source,tensors,device); passed,deltas=_contract(final["metrics"],parent_best["H"],best_update,config); export=head.export()
    checkpoint={"schema_version":SCHEMA,"experiment_id":config["experiment_id"],"code_commit":code_commit,"config_sha256":config_sha,"best_update":best_update,"model_state_dict":best_state,"source_model_state_dict":best_source_state,"export":export.__dict__}
    atomic_torch_save(output/"model_best.pth",checkpoint); atomic_write_json(output/"evaluation_history.json",history)
    result={"schema_version":SCHEMA,"experiment_id":config["experiment_id"],"dataset":config["dataset"],"code_commit":code_commit,"config_sha256":config_sha,"relation_asset_manifest_sha256":config["relation_manifest_sha256"],"best_update":best_update,"metrics":final["metrics"],"parent_best_metrics":parent_best,"parent_best_update":parent_best_update,"delta_H_vs_parent":float(final["metrics"]["full"]["H"]-parent_best["H"]),"module_off_delta_H":deltas,"best_zs_observation":best_zs,"parent_best_zs_observation":parent_best_zs,"module_contract_passed":passed,"decision":"keep_v7_multidataset" if passed else "drop_v7_multidataset_contract_failed","direction_skip_seen_class_ids":config["direction_skip_seen_class_ids"],"teacher_refresh_count":refresh_count,"total_updates":int(config["total_updates"]),"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"strict_blind_claim":False,"llm_world_knowledge_used":True,"reproducibility":reproducibility}
    atomic_write_json(output/"metrics.json",result); emit({"event":"complete",**result}); log.close(); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path); parser.add_argument("--expected-commit"); parser.add_argument("--expected-config-sha"); parser.add_argument("--micro-batch-only",action="store_true"); args=parser.parse_args()
    if args.micro_batch_only: micro_batch(args.config); return
    if args.output_dir is None or not args.expected_commit or not args.expected_config_sha: parser.error("正式RUN缺少身份参数。")
    run(args.config,args.output_dir,args.expected_commit,args.expected_config_sha)


if __name__=="__main__": main()
