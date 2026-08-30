"""Evaluate generic Top-3 PCLR relations on fixed AWA2 or SUN TG+GTD checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.train_gtd_tst import build_model, load_assets, load_config
from tools.gzsl_data import per_class_accuracy
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v4-pclr-multidataset-generic-eval.v1"
CONFIG_KEYS = {
    "schema_version", "experiment_id", "dataset", "source_config",
    "source_config_sha256", "source_code_commit", "source_checkpoint",
    "source_checkpoint_sha256", "source_metrics", "source_metrics_sha256",
    "relation_manifest", "relation_manifest_sha256", "candidate_top_k",
    "ridge_lambda", "potential_cap", "relation_temperature", "relation_strength",
    "seen_logit_gamma", "device", "test_used_for_selection",
    "test_used_for_hyperparameter_selection", "nested_official_test_selection",
    "unseen_images_used_for_gradient", "strict_blind_claim", "human_annotations_used",
    "llm_world_knowledge_used", "generic_class_name_directions",
}
IDENTITIES = {
    "AWA2": {
        "experiment_id": "V4-CONFIRM-003-AWA2",
        "source_config_sha256": "44976e5e77a907112a88d295ee70296a2a22ef1c15c5b1f1d286249e1451a529",
        "source_checkpoint_sha256": "a1c3b90a7922d2e33a5fd40e0381b1f5cd08e18c54a55e4a7b5b9acfc4b2d019",
        "source_metrics_sha256": "02ee575bb88de1d69f2b60bc827ec2968603c4e86602ad1d71fd481d9a4c9042",
        "relation_manifest_sha256": "f93c9a690ce068614bc9792e6b60e989a4fe1fefebeb5df4a0273737e7bdadb2",
        "candidate_top_k": 5,
        "seen_logit_gamma": 0.05,
    },
    "SUN": {
        "experiment_id": "V4-CONFIRM-003-SUN",
        "source_config_sha256": "75a4035f783e92ba2cc70c3e7a633791abb80f3844a4f7b29ce1d172b4935cf1",
        "source_checkpoint_sha256": "336ba9259f584f575626a3b5034dfc87ff390b91b9205dd1022d895c109d0174",
        "source_metrics_sha256": "2137bd31a99df1a72ce85778a9efdd393b8e00ec17737d986ccf544a16a4d35c",
        "relation_manifest_sha256": "385744e7532ddf12862c3926878bac64422a902cdeeb87e30a9667045d4093b8",
        "candidate_top_k": 60,
        "seen_logit_gamma": 0.15,
    },
}


def load_multidataset_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    identity = IDENTITIES.get(config.get("dataset")) if isinstance(config, dict) else None
    invalid = (
        not isinstance(config, dict) or actual != CONFIG_KEYS or identity is None
        or config.get("schema_version") != SCHEMA
        or any(config.get(key) != value for key, value in identity.items())
        or config.get("source_code_commit") != "4013cca894b00933f6bfed0a125690c66e54cba1"
        or float(config.get("ridge_lambda", -1)) != 0.3
        or float(config.get("potential_cap", -1)) != 0.5
        or float(config.get("relation_temperature", -1)) != 0.2
        or float(config.get("relation_strength", -1)) != 1.95
        or config.get("device") not in {"cuda:0", "cuda:1"}
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not True
        or config.get("nested_official_test_selection") is not True
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
        or config.get("llm_world_knowledge_used") is not False
        or config.get("generic_class_name_directions") is not True
    )
    if invalid:
        raise ValueError("PCLR multidataset config identity or disclosure changed.")
    return config, sha256_file(path)


def load_relation_asset(
    config: dict,
    *,
    class_count: int,
    seen_count: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    manifest_path = Path(config["relation_manifest"])
    if (
        not manifest_path.is_absolute() or not manifest_path.is_file()
        or sha256_file(manifest_path) != config["relation_manifest_sha256"]
    ):
        raise ValueError("PCLR multidataset relation manifest mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs_sha256")
    required_outputs = {
        "relation_sentence_embeds.pt",
        "edge_index.pt",
        "relation_texts.json",
    }
    if (
        manifest.get("schema_version") != "gzsl-paper.pclr-generic-relation-asset.v1"
        or manifest.get("dataset") != config["dataset"]
        or manifest.get("class_count") != int(class_count)
        or manifest.get("seen_count") != int(seen_count)
        or not isinstance(manifest.get("edge_count"), int)
        or int(manifest["edge_count"]) <= 0
        or manifest.get("direction_count") != 2 * int(manifest["edge_count"])
        or manifest.get("embedding_dimension") != 768
        or manifest.get("graph_source") != "OpenAI_CLIP_class_name_template_union_top3"
        or manifest.get("human_annotations_used") is not False
        or manifest.get("llm_world_knowledge_used") is not False
        or manifest.get("generic_class_name_directions") is not True
        or not isinstance(outputs, dict)
        or set(outputs) != required_outputs
    ):
        raise ValueError("PCLR multidataset relation asset schema changed.")
    for name, digest in outputs.items():
        if sha256_file(manifest_path.parent / name) != digest:
            raise ValueError(f"PCLR multidataset relation output mismatch: {name}")
    relations = torch.load(
        manifest_path.parent / "relation_sentence_embeds.pt", map_location="cpu", weights_only=True
    )
    edges = torch.load(
        manifest_path.parent / "edge_index.pt", map_location="cpu", weights_only=True
    )
    if (
        relations.dtype != torch.float32
        or edges.dtype != torch.int64
        or tuple(relations.shape) != (manifest["edge_count"], 2, 768)
        or tuple(edges.shape) != (manifest["edge_count"], 2)
        or not torch.isfinite(relations).all()
        or not torch.allclose(relations.norm(dim=-1), torch.ones_like(relations[..., 0]), atol=1e-4)
        or int(edges.min()) < 0
        or int(edges.max()) >= int(class_count)
        or not bool((edges[:, 0] < edges[:, 1]).all())
        or torch.unique(edges, dim=0).size(0) != int(manifest["edge_count"])
    ):
        raise ValueError("PCLR multidataset relation tensor contract changed.")
    degrees = torch.zeros(int(class_count), dtype=torch.long)
    degrees.scatter_add_(0, edges[:, 0], torch.ones(len(edges), dtype=torch.long))
    degrees.scatter_add_(0, edges[:, 1], torch.ones(len(edges), dtype=torch.long))
    if bool(degrees.eq(0).any()):
        raise ValueError("PCLR multidataset relation graph contains an uncovered class.")
    return relations, edges, manifest


def transitions(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor) -> dict:
    old = before.eq(labels.cpu())
    new = after.eq(labels.cpu())
    return {"corrected_wrong_to_right": int((~old & new).sum()),
            "damaged_right_to_wrong": int((old & ~new).sum()),
            "net_correct": int(new.sum() - old.sum())}


@torch.no_grad()
def evaluate(model, tensors, relations, edges, config, device):
    model.eval()
    class_count = int(tensors["role_sentence_embeds"].size(0))
    relations = relations.to(device)
    edges = edges.to(device)
    incidence = torch.zeros(len(edges), class_count, device=device)
    rows = torch.arange(len(edges), device=device)
    incidence[rows, edges[:, 0]] = 1
    incidence[rows, edges[:, 1]] = -1
    mapping = torch.linalg.solve(
        incidence.T @ incidence + float(config["ridge_lambda"]) * torch.eye(class_count, device=device),
        incidence.T,
    )
    prototypes = F.normalize(model.prototypes().float(), dim=-1)
    seen = model.seen_classes.cpu()
    unseen = model.unseen_classes.cpu()
    seen_device = seen.to(device)
    unseen_device = unseen.to(device)
    predictions = {name: {"seen": [], "unseen": [], "zs": []} for name in ("raw", "full")}
    active_sum = 0.0
    active_count = 0
    for split, features in (("seen", tensors["test_seen_features"]), ("unseen", tensors["test_unseen_features"])):
        for start in range(0, len(features), 256):
            images = features[start : start + 256].to(device).float()
            readout = F.normalize(images, dim=-1)
            raw = readout @ prototypes.T * model.scale()
            scores = torch.einsum("bd,ekd->bek", readout, relations) / float(config["relation_temperature"])
            difference = scores[..., 0] - scores[..., 1]
            candidates = raw.topk(int(config["candidate_top_k"]), dim=1).indices
            selected = torch.zeros_like(raw, dtype=torch.bool)
            selected.scatter_(1, candidates, True)
            active = selected[:, edges[:, 0]] & selected[:, edges[:, 1]]
            difference *= active
            potential = difference @ mapping.T
            potential -= potential.mean(dim=1, keepdim=True)
            norm = potential.abs().amax(dim=1, keepdim=True)
            potential = float(config["potential_cap"]) * potential / torch.maximum(
                norm, torch.full_like(norm, float(config["potential_cap"]))
            )
            full = raw + float(config["relation_strength"]) * raw.std(
                dim=1, unbiased=False, keepdim=True
            ) * potential
            full[:, seen_device] -= float(config["seen_logit_gamma"])
            for name, logits in (("raw", raw), ("full", full)):
                predictions[name][split].append(logits.argmax(dim=1).cpu())
                if split == "unseen":
                    predictions[name]["zs"].append(
                        unseen_device[logits.index_select(1, unseen_device).argmax(dim=1)].cpu()
                    )
            active_sum += float(active.float().mean()) * images.size(0)
            active_count += images.size(0)
    for value in predictions.values():
        for split in value:
            value[split] = torch.cat(value[split])
    labels_seen = tensors["test_seen_labels"].long()
    labels_unseen = tensors["test_unseen_labels"].long()
    def metrics(value):
        s = 100 * per_class_accuracy(labels_seen, value["seen"], seen)
        u = 100 * per_class_accuracy(labels_unseen, value["unseen"], unseen)
        zs = 100 * per_class_accuracy(labels_unseen, value["zs"], unseen)
        return {"U": float(u), "S": float(s), "H": float(2*s*u/(s+u)), "ZS": float(zs)}
    raw_metrics, full_metrics = metrics(predictions["raw"]), metrics(predictions["full"])
    return {
        "raw_metrics": raw_metrics,
        "full_metrics": full_metrics,
        "delta_H": full_metrics["H"] - raw_metrics["H"],
        "delta_ZS": full_metrics["ZS"] - raw_metrics["ZS"],
        "raw_gap": abs(raw_metrics["U"] - raw_metrics["S"]),
        "full_gap": abs(full_metrics["U"] - full_metrics["S"]),
        "transitions": {
            "seen": transitions(predictions["raw"]["seen"], predictions["full"]["seen"], labels_seen),
            "unseen": transitions(predictions["raw"]["unseen"], predictions["full"]["unseen"], labels_unseen),
            "zs": transitions(predictions["raw"]["zs"], predictions["full"]["zs"], labels_unseen),
        },
        "active_edge_rate": active_sum / active_count,
    }


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("PCLR multidataset expected commit mismatch.")
    config, config_sha = load_multidataset_config(config_path)
    if config_sha != expected_config_sha or output_dir.name != config["experiment_id"]:
        raise ValueError("PCLR multidataset config/output identity mismatch.")
    for key in ("source_config", "source_checkpoint", "source_metrics"):
        path = Path(config[key])
        if not path.is_absolute() or not path.is_file() or sha256_file(path) != config[f"{key}_sha256"]:
            raise ValueError(f"PCLR multidataset source mismatch: {key}")
    source_metrics = json.loads(Path(config["source_metrics"]).read_text(encoding="utf-8"))
    source_config, source_sha = load_config(Path(config["source_config"]))
    if source_sha != config["source_config_sha256"]:
        raise ValueError("PCLR multidataset source config loader mismatch.")
    device = torch.device(config["device"])
    tensors = load_assets(source_config)
    model = build_model(source_config, tensors, device)
    checkpoint = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("code_commit") != config["source_code_commit"] or checkpoint.get("config_sha256") != source_sha:
        raise ValueError("PCLR multidataset checkpoint identity mismatch.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    relations, edges, manifest = load_relation_asset(
        config,
        class_count=int(tensors["role_sentence_embeds"].size(0)),
        seen_count=int(model.seen_classes.numel()),
    )
    result = evaluate(model, tensors, relations, edges, config, device)
    expected = source_metrics["best_metrics"]
    if any(abs(result["raw_metrics"][key] - float(expected[key])) > 1e-6 for key in ("U", "S", "H", "ZS")):
        raise RuntimeError("PCLR multidataset Raw checkpoint did not reproduce source best.")
    passed = result["delta_H"] > 0 and result["delta_ZS"] >= -0.5 and result["full_gap"] <= result["raw_gap"]
    result.update({
        "schema_version": SCHEMA, "experiment_id": config["experiment_id"], "dataset": config["dataset"],
        "evaluation_code_commit": code_commit, "config_sha256": config_sha,
        "source_code_commit": config["source_code_commit"], "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "relation_manifest_sha256": config["relation_manifest_sha256"], "relation_edge_count": manifest["edge_count"],
        "gate_passed": passed, "decision": "keep_multidataset_generic_gate" if passed else "drop_multidataset_generic_gate",
        "test_used_for_selection": True, "test_used_for_hyperparameter_selection": True,
        "nested_official_test_selection": True, "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False, "human_annotations_used": False, "llm_world_knowledge_used": False,
        "generic_class_name_directions": True,
    })
    output = prepare_output_dir(output_dir)
    (output / "config.snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_write_json(output / "metrics.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()
