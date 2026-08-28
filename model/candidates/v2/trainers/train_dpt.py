from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.dpt import (
    AdaptiveDistributionalPrototypeClassifier,
    DistributionalPrototypeClassifier,
    text_resultant_lengths,
    text_uncertainty_features,
)
from model.frameworks.v4.tg import VariableClassTGVPR, fixed_class_folds
from model.candidates.v2.trainers.train_elpt import FrozenPrototypeClassifier, _candidate_prototypes
from model.frameworks.v4.tst import TangentStepGate
from model.frameworks.v2 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


CONFIG_KEYS = {
    "schema_version", "attempt_id", "idea_id", "framework_id",
    "base_config", "base_checkpoint", "base_checkpoint_sha256",
    "tst_gate_model", "tst_gate_model_sha256", "seed", "epochs",
    "batch_size", "lr", "weight_decay", "max_gamma", "initial_gamma",
    "parent_metrics_percent",
}
CONFIG_KEYS_V2 = CONFIG_KEYS | {"confidence_mode", "max_log_scale"}


class TeeStream:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path: Path):
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    expected = CONFIG_KEYS_V2 if isinstance(config, dict) and config.get("schema_version") in ("gzsl-paper.dpt.v2", "gzsl-paper.dpt.v3") else CONFIG_KEYS
    if not isinstance(config, dict) or actual != expected:
        raise ValueError(f"DPT配置字段错误；缺少={sorted(expected-actual)}，多出={sorted(actual-expected)}。")
    if config["schema_version"] not in ("gzsl-paper.dpt.v1", "gzsl-paper.dpt.v2", "gzsl-paper.dpt.v3"):
        raise ValueError("DPT schema错误。")
    if config["attempt_id"] not in ("V2-TRY-041", "V2-TRY-042", "V2-TRY-043") or config["idea_id"] != "IDEA-012":
        raise ValueError("DPT首次TRY身份错误。")
    if config["framework_id"] != "FRAMEWORK-V2":
        raise ValueError("DPT父框架错误。")
    if int(config["epochs"]) != 10 or int(config["batch_size"]) != 64:
        raise ValueError("DPT固定10 epoch与batch_size=64。")
    if float(config["lr"]) != 0.01 or float(config["weight_decay"]) != 0.0:
        raise ValueError("DPT固定Adam lr=0.01且无weight decay。")
    if float(config["max_gamma"]) != 2.0 or float(config["initial_gamma"]) != 0.05:
        raise ValueError("DPT gamma身份错误。")
    if set(config["parent_metrics_percent"]) != {"U", "S", "H", "ZS"}:
        raise ValueError("DPT父指标不完整。")
    config.setdefault("confidence_mode", "global_resultant")
    config.setdefault("max_log_scale", 0.1)
    expected_mode = {
        "V2-TRY-041": "global_resultant",
        "V2-TRY-042": "adaptive_gate",
        "V2-TRY-043": "centered_adaptive_gate",
    }[config["attempt_id"]]
    if config["confidence_mode"] != expected_mode:
        raise ValueError("DPT置信模式与TRY身份不匹配。")
    if float(config["max_log_scale"]) != 0.1:
        raise ValueError("DPT自适应log尺度固定为0.1。")
    return config, sha256_file(path)


def _load_tst_gate(config, device):
    path = Path(config["tst_gate_model"])
    if sha256_file(path) != config["tst_gate_model_sha256"]:
        raise ValueError("DPT父TST gate SHA不匹配。")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    gate = TangentStepGate(input_dim=4, max_step=1.5)
    gate.load_state_dict(payload["gate_state_dict"], strict=True)
    for parameter in gate.parameters(): parameter.requires_grad_(False)
    return gate.to(device).eval()


def run(config_path: Path, output_dir: Path, expected_commit: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    config, config_sha = load_config(config_path)
    base_path = Path(config["base_config"])
    if not base_path.is_absolute(): base_path = Path.cwd() / base_path
    base_config, base_config_sha = h1.load_config(base_path)
    paths = h1.resolve_paths(base_config)
    input_sha = h1.verify_inputs(base_config, paths, h1.TRAINING_KEYS)
    checkpoint_path = Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path) != config["base_checkpoint_sha256"]:
        raise ValueError("DPT父checkpoint SHA不匹配。")
    device = torch.device(base_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DPT正式TRY要求CUDA。")
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["seed"])
        configure_reproducibility(seed, strict_determinism=True, deterministic_warn_only=False)
        tensors = {name: torch.load(paths[name], map_location="cpu", weights_only=True) for name in ("sentence_embeds", "train_features", "train_labels")}
        labels = tensors["train_labels"].long()
        seenclasses = torch.unique(labels, sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        centroids = h1.visual_centroids(tensors["train_features"], labels, seenclasses)
        parent = VariableClassTGVPR(
            tensors["sentence_embeds"], seenclasses, centroids,
            dropout=base_config["dropout"], inner_ratio=base_config["inner_ratio"],
            outer_ratio=base_config["outer_ratio"], temperature=base_config["temperature"],
        )
        parent.load_state_dict(checkpoint["model_state_dict"], strict=True)
        parent = parent.to(device).eval()
        gate = _load_tst_gate(config, device)
        tst_prototypes, _ = _candidate_prototypes(
            parent, gate, seenclasses, unseenclasses, device,
            "summary", fixed_class_folds(seenclasses), "tangent"
        )
        if config["confidence_mode"] in ("adaptive_gate", "centered_adaptive_gate"):
            model = AdaptiveDistributionalPrototypeClassifier(
                tst_prototypes,
                text_uncertainty_features(tensors["sentence_embeds"]).to(device),
                parent.scale(),
                max_log_scale=config["max_log_scale"],
                seenclasses=seenclasses,
                center_seen_log_scale=(
                    config["confidence_mode"] == "centered_adaptive_gate"
                ),
            ).to(device)
        else:
            model = DistributionalPrototypeClassifier(
                tst_prototypes,
                text_resultant_lengths(tensors["sentence_embeds"]).to(device),
                seenclasses,
                parent.scale(),
                max_gamma=config["max_gamma"], initial_gamma=config["initial_gamma"],
            ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=0.0)
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seenclasses] = torch.arange(150)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        history = []
        for epoch in range(1, int(config["epochs"]) + 1):
            permutation = torch.randperm(labels.numel(), generator=generator)
            loss_sum = 0.0
            count = 0
            for start in range(0, labels.numel(), int(config["batch_size"])):
                indices = permutation[start:start + int(config["batch_size"])]
                features = tensors["train_features"][indices].to(device).float()
                targets = mapping[labels[indices]].to(device)
                loss = F.cross_entropy(model.logits(features, seenclasses), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("DPT梯度非有限。")
                optimizer.step()
                loss_sum += float(loss.detach()) * indices.numel()
                count += indices.numel()
            confidence = model.class_confidence().detach()
            gamma_value = (
                float(model.gamma().detach()) if hasattr(model, "gamma") else None
            )
            row = {
                "epoch": epoch, "train_ce": loss_sum / count,
                "gamma": gamma_value,
                "confidence_min": float(confidence.min()),
                "confidence_max": float(confidence.max()),
            }
            history.append(row)
            print(f"epoch={epoch} ce={row['train_ce']:.6f} gamma={row['gamma']} confidence=[{row['confidence_min']:.6f},{row['confidence_max']:.6f}]")
        payload = {"attempt_id": config["attempt_id"], "code_commit": code_commit, "config": config, "model_state_dict": copy.deepcopy(model.state_dict()), "history": history}
        torch.save(payload, output_dir / "distribution_model.pth")

        # official test严格在DPT训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config, paths, h1.OFFICIAL_KEYS))
        tensors.update({name: torch.load(paths[name], map_location="cpu", weights_only=True) for name in h1.OFFICIAL_KEYS})
        parent_classifier = FrozenPrototypeClassifier(tst_prototypes, parent.scale()).to(device)
        parent_metrics = h1.evaluate(parent_classifier, tensors, seenclasses, unseenclasses, device)
        candidate_metrics = h1.evaluate(model, tensors, seenclasses, unseenclasses, device)
        delta = {key: candidate_metrics[key] - float(config["parent_metrics_percent"][key]) for key in ("U", "S", "H", "ZS")}
        confidence = model.class_confidence().detach()
        confidence_stats = {"min": float(confidence.min()), "max": float(confidence.max()), "ratio": float(confidence.max()/confidence.min())}
        gamma = float(model.gamma().detach()) if hasattr(model, "gamma") else None
        gamma_safe = gamma is None or gamma < 1.96
        success = delta["H"] >= 0.20 and delta["U"] >= -2.0 and delta["S"] >= -2.0 and gamma_safe and confidence_stats["ratio"] < 1.5
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        metrics = {
            "attempt_id": config["attempt_id"], "idea_id": config["idea_id"],
            "framework_id": config["framework_id"], "code_commit": code_commit,
            "config_sha256": config_sha, "base_config_sha256": base_config_sha,
            "evaluation_protocol": h1.EVALUATION_PROTOCOL, "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "recomputed_parent_metrics_percent": parent_metrics,
            "parent_metrics_percent": config["parent_metrics_percent"],
            "candidate_metrics_percent": candidate_metrics,
            "delta_vs_parent_percent_points": delta,
            "learned_gamma": gamma, "confidence_stats": confidence_stats,
            "confidence_mode": config["confidence_mode"],
            "success": success,
            "distribution_model_sha256": sha256_file(output_dir / "distribution_model.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", metrics)
        print(metrics)
        return metrics
    finally:
        sys.stdout.flush(); sys.stdout = original_stdout; log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(); run(args.config, args.output_dir, args.expected_commit)


if __name__ == "__main__": main()
