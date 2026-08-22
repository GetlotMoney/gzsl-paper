from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.elpt import VariableClassTGVPR, fixed_class_folds, topology_loss
from model.innovations.sgt import (
    GraphResidualClassifier,
    GraphTransportStrength,
    apply_graph_residual,
    semantic_graph_residual,
)
from model.innovations.train_elpt import (
    FrozenPrototypeClassifier,
    _candidate_prototypes,
    _fold_package,
    _load_fold_checkpoint,
)
from model.innovations.tst import TangentStepGate, tangent_transport
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS = {
    "schema_version", "attempt_id", "idea_id", "framework_id", "base_config",
    "base_checkpoint", "base_checkpoint_sha256", "tst_gate_model",
    "tst_gate_model_sha256", "fold_checkpoint_dir", "seed", "epochs",
    "batch_half", "lr", "weight_decay", "topology_weight", "top_k",
    "graph_temperature", "max_strength", "initial_strength", "parent_metrics_percent",
}


class TeeStream:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path: Path):
    path = path.resolve(); config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(f"SGT配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"] != "gzsl-paper.sgt.v1" or config["attempt_id"] != "V2-TRY-044" or config["idea_id"] != "IDEA-013":
        raise ValueError("SGT首次TRY身份错误。")
    if int(config["epochs"]) != 20 or int(config["batch_half"]) != 32:
        raise ValueError("SGT固定20 epoch与32/32平衡batch。")
    if float(config["lr"]) != 0.01 or float(config["weight_decay"]) != 0.0:
        raise ValueError("SGT固定Adam lr=0.01。")
    if int(config["top_k"]) != 5 or float(config["graph_temperature"]) != 0.05:
        raise ValueError("SGT首次TRY固定top-5和temperature=0.05。")
    if float(config["max_strength"]) != 0.5 or float(config["initial_strength"]) != 0.1:
        raise ValueError("SGT传播强度身份错误。")
    if set(config["parent_metrics_percent"]) != {"U", "S", "H", "ZS"}:
        raise ValueError("SGT父指标不完整。")
    return config, sha256_file(path)


def _load_gate(config, device):
    path = Path(config["tst_gate_model"])
    if sha256_file(path) != config["tst_gate_model_sha256"]: raise ValueError("SGT父TST gate SHA不匹配。")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    gate = TangentStepGate(input_dim=4, max_step=1.5); gate.load_state_dict(payload["gate_state_dict"], strict=True)
    for parameter in gate.parameters(): parameter.requires_grad_(False)
    return gate.to(device).eval()


def run(config_path: Path, output_dir: Path, expected_commit: str):
    require_clean_code_tree(); code_commit = current_code_commit()
    if code_commit != expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config, config_sha = load_config(config_path)
    base_path = Path(config["base_config"])
    if not base_path.is_absolute(): base_path = Path.cwd() / base_path
    base_config, base_config_sha = h1.load_config(base_path); paths = h1.resolve_paths(base_config)
    input_sha = h1.verify_inputs(base_config, paths, h1.TRAINING_KEYS)
    checkpoint_path = Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path) != config["base_checkpoint_sha256"]: raise ValueError("SGT父checkpoint SHA不匹配。")
    device = torch.device(base_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("SGT要求CUDA。")
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle: yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1); original_stdout = sys.stdout; sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["seed"]); configure_reproducibility(seed, strict_determinism=True, deterministic_warn_only=False)
        tensors = {name: torch.load(paths[name], map_location="cpu", weights_only=True) for name in ("sentence_embeds", "train_features", "train_labels")}
        labels = tensors["train_labels"].long(); seenclasses = torch.unique(labels, sorted=True); allclasses = torch.arange(200); unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        folds = fixed_class_folds(seenclasses); tst_gate = _load_gate(config, device); packages = []
        for fold_id, (pseudo_seen, pseudo_unseen) in enumerate(folds):
            fold_model = _load_fold_checkpoint(fold_id, pseudo_seen, tensors["sentence_embeds"], tensors["train_features"], labels, base_config, device, config["fold_checkpoint_dir"])
            package = _fold_package(fold_model, pseudo_seen, pseudo_unseen, tensors, seenclasses, device, "summary")
            with torch.no_grad():
                base_all = package["base_all"].to(device); tst_all = package["fold_full"].to(device).clone(); pu = pseudo_unseen.to(device)
                step = tst_gate(package["gate_features"].to(device)); tst_all[pu] = tangent_transport(base_all.index_select(0, pu), package["value"].to(device), step)
                graph = semantic_graph_residual(base_all, package["fold_full"].to(device).index_select(0, pseudo_seen.to(device)), pseudo_seen, pseudo_unseen, top_k=config["top_k"], temperature=config["graph_temperature"])
            package["tst_all"] = tst_all; package["graph_residual"] = graph; packages.append(package); del fold_model
        strength = GraphTransportStrength(config["max_strength"], config["initial_strength"]).to(device); optimizer = torch.optim.Adam(strength.parameters(), lr=float(config["lr"]), weight_decay=0.0)
        mapping = torch.full((200,), -1, dtype=torch.long); mapping[seenclasses] = torch.arange(150); generators = [torch.Generator(device="cpu").manual_seed(seed*13000+i) for i in range(3)]; half = int(config["batch_half"]); history = []
        for epoch in range(1, int(config["epochs"])+1):
            loss_sum = 0.0; count = 0
            for fold_id, package in enumerate(packages):
                steps = min(package["seen_indices"].numel()//half, package["unseen_indices"].numel()//half)
                for _ in range(steps):
                    g = generators[fold_id]; si = package["seen_indices"][torch.randperm(package["seen_indices"].numel(), generator=g)[:half]]; ui = package["unseen_indices"][torch.randperm(package["unseen_indices"].numel(), generator=g)[:half]]; indices = torch.cat((si, ui))
                    images = tensors["train_features"][indices].to(device).float(); targets = mapping[labels[indices]].to(device); final = package["tst_all"].clone(); pu = package["pseudo_unseen"].to(device); final[pu] = apply_graph_residual(final.index_select(0, pu), package["graph_residual"], strength()); competition = final.index_select(0, seenclasses.to(device)); logits = F.normalize(images, dim=-1) @ competition.T * package["scale"].to(device); ce = F.cross_entropy(logits, targets); topo = topology_loss(package["base_all"].to(device).index_select(0, seenclasses.to(device)), competition); loss = ce + float(config["topology_weight"])*topo
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum += float(loss.detach()); count += 1
            row = {"epoch": epoch, "loss": loss_sum/count, "strength": float(strength().detach())}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} strength={row['strength']:.6f}")
        payload = {"attempt_id": config["attempt_id"], "code_commit": code_commit, "config": config, "strength_state_dict": copy.deepcopy(strength.state_dict()), "history": history}; torch.save(payload, output_dir / "graph_model.pth")
        # official test严格在SGT训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config, paths, h1.OFFICIAL_KEYS)); tensors.update({name: torch.load(paths[name], map_location="cpu", weights_only=True) for name in h1.OFFICIAL_KEYS})
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); centroids = h1.visual_centroids(tensors["train_features"], labels, seenclasses); parent = VariableClassTGVPR(tensors["sentence_embeds"], seenclasses, centroids, dropout=base_config["dropout"], inner_ratio=base_config["inner_ratio"], outer_ratio=base_config["outer_ratio"], temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"], strict=True); parent = parent.to(device).eval(); tst_prototypes, _ = _candidate_prototypes(parent, tst_gate, seenclasses, unseenclasses, device, "summary", folds, "tangent"); base_all = parent.base_prototypes(); graph = semantic_graph_residual(base_all, parent.prototypes().index_select(0, seenclasses.to(device)), seenclasses, unseenclasses, top_k=config["top_k"], temperature=config["graph_temperature"]); model = GraphResidualClassifier(tst_prototypes, unseenclasses, graph, strength, parent.scale()).to(device); parent_metrics = h1.evaluate(FrozenPrototypeClassifier(tst_prototypes, parent.scale()).to(device), tensors, seenclasses, unseenclasses, device); candidate_metrics = h1.evaluate(model, tensors, seenclasses, unseenclasses, device); delta = {key: candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; value = float(strength().detach()); success = delta["H"]>=0.20 and delta["U"]>=-2 and delta["S"]>=-2 and value < 0.49
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha}); metrics = {"attempt_id": config["attempt_id"], "idea_id": config["idea_id"], "framework_id": config["framework_id"], "code_commit": code_commit, "config_sha256": config_sha, "base_config_sha256": base_config_sha, "evaluation_protocol": h1.EVALUATION_PROTOCOL, "test_used_for_selection": True, "unseen_images_used_for_gradient": False, "recomputed_parent_metrics_percent": parent_metrics, "parent_metrics_percent": config["parent_metrics_percent"], "candidate_metrics_percent": candidate_metrics, "delta_vs_parent_percent_points": delta, "learned_strength": value, "success": success, "graph_model_sha256": sha256_file(output_dir / "graph_model.pth")}; atomic_write_json(output_dir / "metrics.json", metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout = original_stdout; log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--expected-commit", required=True); args = parser.parse_args(); run(args.config, args.output_dir, args.expected_commit)


if __name__ == "__main__": main()
