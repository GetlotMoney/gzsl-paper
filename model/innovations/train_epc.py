from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.elpt import VariableClassTGVPR, fixed_class_folds
from model.innovations.epc import EpisodicPriorCalibration
from model.innovations.train_elpt import (
    FrozenPrototypeClassifier,
    _candidate_prototypes,
    _fold_package,
    _load_fold_checkpoint,
    load_config as load_tst_config,
)
from model.innovations.tst import TangentStepGate, tangent_transport
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


CONFIG_KEYS = {
    "schema_version",
    "attempt_id",
    "idea_id",
    "framework_id",
    "base_config",
    "base_checkpoint",
    "base_checkpoint_sha256",
    "tst_config",
    "tst_gate_model",
    "tst_gate_model_sha256",
    "fold_checkpoint_dir",
    "seed",
    "epochs",
    "batch_half",
    "lr",
    "weight_decay",
    "max_margin",
}
CONFIG_KEYS_V2 = CONFIG_KEYS | {"objective"}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_config(path: Path):
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    expected = CONFIG_KEYS_V2 if isinstance(config, dict) and config.get("schema_version") == "gzsl-paper.epc.v2" else CONFIG_KEYS
    if not isinstance(config, dict) or actual != expected:
        raise ValueError(
            f"EPC配置字段错误；缺少={sorted(expected-actual)}，"
            f"多出={sorted(actual-expected)}。"
        )
    if config["schema_version"] not in ("gzsl-paper.epc.v1", "gzsl-paper.epc.v2"):
        raise ValueError("EPC schema错误。")
    if (
        config["attempt_id"] not in ("V2-TRY-019", "V2-TRY-020")
        or config["idea_id"] != "IDEA-006"
        or config["framework_id"] != "FRAMEWORK-V2"
    ):
        raise ValueError("EPC首次TRY身份错误。")
    if int(config["epochs"]) != 10 or int(config["batch_half"]) != 32:
        raise ValueError("EPC固定10 epoch和32/32平衡batch。")
    if float(config["lr"]) != 0.01 or float(config["weight_decay"]) != 0.0:
        raise ValueError("EPC固定Adam lr=0.01且无weight decay。")
    if float(config["max_margin"]) != 0.5:
        raise ValueError("EPC固定边际范围[-0.5,0.5]。")
    if config["attempt_id"] == "V2-TRY-020" and config.get("objective") != "soft_harmonic":
        raise ValueError("EPC补救1必须使用soft_harmonic目标。")
    return config, sha256_file(path)


def episodic_soft_harmonic_loss(logits, targets, pseudo_unseen_mask):
    probability = logits.softmax(dim=-1)
    correct = probability.gather(1, targets.unsqueeze(1)).squeeze(1)
    unseen_soft = correct[pseudo_unseen_mask].mean()
    seen_soft = correct[~pseudo_unseen_mask].mean()
    soft_h = 2.0 * seen_soft * unseen_soft / (seen_soft + unseen_soft + 1e-12)
    return 1.0 - soft_h, seen_soft, unseen_soft


def _load_tst_gate(config, device):
    gate_path = Path(config["tst_gate_model"])
    if sha256_file(gate_path) != config["tst_gate_model_sha256"]:
        raise ValueError("TST gate SHA不匹配。")
    payload = torch.load(gate_path, map_location="cpu", weights_only=False)
    gate = TangentStepGate(max_step=1.5)
    gate.load_state_dict(payload["gate_state_dict"], strict=True)
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    return gate.to(device).eval()


def _adapted_fold_prototypes(package, gate, seenclasses, device):
    base_all = package["base_all"].to(device)
    prototypes = package["fold_full"].to(device).clone()
    pseudo_unseen = package["pseudo_unseen"].to(device)
    with torch.no_grad():
        step = gate(package["gate_features"].to(device))
        prototypes[pseudo_unseen] = tangent_transport(
            base_all.index_select(0, pseudo_unseen),
            package["value"].to(device),
            step,
        )
    return prototypes.index_select(0, seenclasses.to(device))


@torch.no_grad()
def evaluate_epc(
    prototypes,
    scale,
    calibration,
    tensors,
    seenclasses,
    unseenclasses,
    device,
):
    def predict(features, competition, calibrated):
        logits = F.normalize(features.to(device).float(), dim=-1) @ prototypes.index_select(
            0, competition.to(device)
        ).T * scale
        if calibrated:
            logits = calibration(logits, competition, unseenclasses)
        return competition[logits.argmax(dim=1).cpu()]

    allclasses = torch.arange(200)
    seen_pred = predict(tensors["seen_features"], allclasses, True)
    unseen_pred = predict(tensors["unseen_features"], allclasses, True)
    zsl_pred = predict(tensors["unseen_features"], unseenclasses, False)
    seen = h1.per_class_accuracy(tensors["seen_labels"], seen_pred, seenclasses)
    unseen = h1.per_class_accuracy(tensors["unseen_labels"], unseen_pred, unseenclasses)
    zsl = h1.per_class_accuracy(tensors["unseen_labels"], zsl_pred, unseenclasses)
    harmonic = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    return {"U": unseen * 100, "S": seen * 100, "H": harmonic * 100, "ZS": zsl * 100}


def run(config_path: Path, output_dir: Path, expected_commit: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    config, config_sha = load_config(config_path)
    base_config_path = Path(config["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = Path.cwd() / base_config_path
    base_config, base_config_sha = h1.load_config(base_config_path)
    tst_config_path = Path(config["tst_config"])
    if not tst_config_path.is_absolute():
        tst_config_path = Path.cwd() / tst_config_path
    tst_config, tst_config_sha = load_tst_config(tst_config_path)
    if tst_config["idea_id"] != "IDEA-005" or tst_config["seed"] != config["seed"]:
        raise ValueError("EPC父TST配置身份不匹配。")
    paths = h1.resolve_paths(base_config)
    input_sha = h1.verify_inputs(base_config, paths, h1.TRAINING_KEYS)
    if sha256_file(Path(config["base_checkpoint"])) != config["base_checkpoint_sha256"]:
        raise ValueError("V2基线checkpoint SHA不匹配。")
    device = torch.device(base_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EPC正式TRY要求CUDA。")

    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["seed"])
        configure_reproducibility(seed, strict_determinism=True, deterministic_warn_only=False)
        tensors = {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in ("sentence_embeds", "train_features", "train_labels")
        }
        seenclasses = torch.unique(tensors["train_labels"].long(), sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        if tensors["train_labels"].numel() != 7057 or unseenclasses.numel() != 50:
            raise ValueError("EPC训练边界必须是7057张seen图像和150/50类。")
        folds = fixed_class_folds(seenclasses)
        packages = []
        for fold_id, (pseudo_seen, pseudo_unseen) in enumerate(folds):
            fold_model = _load_fold_checkpoint(
                fold_id,
                pseudo_seen,
                tensors["sentence_embeds"],
                tensors["train_features"],
                tensors["train_labels"],
                base_config,
                device,
                config["fold_checkpoint_dir"],
            )
            packages.append(
                _fold_package(
                    fold_model,
                    pseudo_seen,
                    pseudo_unseen,
                    tensors,
                    seenclasses,
                    device,
                    "summary",
                )
            )
            del fold_model
        tst_gate = _load_tst_gate(config, device)
        fold_prototypes = [
            _adapted_fold_prototypes(package, tst_gate, seenclasses, device)
            for package in packages
        ]
        calibration = EpisodicPriorCalibration(config["max_margin"]).to(device)
        optimizer = torch.optim.Adam(
            calibration.parameters(), lr=float(config["lr"]), weight_decay=0.0
        )
        label_map = torch.full((200,), -1, dtype=torch.long)
        label_map[seenclasses] = torch.arange(150)
        generators = [
            torch.Generator(device="cpu").manual_seed(seed * 3000 + fold_id)
            for fold_id in range(3)
        ]
        half = int(config["batch_half"])
        history = []
        for epoch in range(1, int(config["epochs"]) + 1):
            loss_sum = 0.0
            steps_total = 0
            for fold_id, package in enumerate(packages):
                seen_indices = package["seen_indices"]
                unseen_indices = package["unseen_indices"]
                steps = min(seen_indices.numel() // half, unseen_indices.numel() // half)
                for _ in range(steps):
                    generator = generators[fold_id]
                    si = seen_indices[
                        torch.randperm(seen_indices.numel(), generator=generator)[:half]
                    ]
                    ui = unseen_indices[
                        torch.randperm(unseen_indices.numel(), generator=generator)[:half]
                    ]
                    indices = torch.cat((si, ui))
                    images = tensors["train_features"][indices].to(device).float()
                    targets = label_map[tensors["train_labels"].long()[indices]].to(device)
                    logits = F.normalize(images, dim=-1) @ fold_prototypes[fold_id].T
                    logits = logits * package["scale"].to(device)
                    logits = calibration(logits, seenclasses, package["pseudo_unseen"])
                    if config.get("objective", "cross_entropy") == "soft_harmonic":
                        pseudo_unseen_mask = torch.isin(
                            tensors["train_labels"].long()[indices],
                            package["pseudo_unseen"],
                        ).to(device)
                        loss, _, _ = episodic_soft_harmonic_loss(
                            logits, targets, pseudo_unseen_mask
                        )
                    else:
                        loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if not torch.isfinite(calibration.raw_margin.grad).all():
                        raise FloatingPointError("EPC边际梯度非有限。")
                    optimizer.step()
                    loss_sum += float(loss.detach())
                    steps_total += 1
            row = {
                "epoch": epoch,
                "train_ce": loss_sum / steps_total,
                "margin": float(calibration.margin().detach()),
            }
            history.append(row)
            print(f"epoch={epoch} train_ce={row['train_ce']:.6f} margin={row['margin']:.6f}")

        payload = {
            "attempt_id": config["attempt_id"],
            "code_commit": code_commit,
            "config": config,
            "calibration_state_dict": copy.deepcopy(calibration.state_dict()),
            "history": history,
        }
        torch.save(payload, output_dir / "calibration_model.pth")

        # official test严格在EPC训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config, paths, h1.OFFICIAL_KEYS))
        tensors.update(
            {
                name: torch.load(paths[name], map_location="cpu", weights_only=True)
                for name in h1.OFFICIAL_KEYS
            }
        )
        checkpoint = torch.load(
            Path(config["base_checkpoint"]), map_location="cpu", weights_only=False
        )
        parent = VariableClassTGVPR(
            tensors["sentence_embeds"],
            seenclasses,
            h1.visual_centroids(
                tensors["train_features"], tensors["train_labels"].long(), seenclasses
            ),
            dropout=base_config["dropout"],
            inner_ratio=base_config["inner_ratio"],
            outer_ratio=base_config["outer_ratio"],
            temperature=base_config["temperature"],
        )
        parent.load_state_dict(checkpoint["model_state_dict"], strict=True)
        parent = parent.to(device).eval()
        tst_prototypes, _ = _candidate_prototypes(
            parent,
            tst_gate,
            seenclasses,
            unseenclasses,
            device,
            "summary",
            folds,
            "tangent",
        )
        tst_classifier = FrozenPrototypeClassifier(tst_prototypes, parent.scale()).to(device)
        parent_metrics = h1.evaluate(
            tst_classifier, tensors, seenclasses, unseenclasses, device
        )
        candidate_metrics = evaluate_epc(
            tst_prototypes,
            parent.scale(),
            calibration,
            tensors,
            seenclasses,
            unseenclasses,
            device,
        )
        delta = {
            key: candidate_metrics[key] - parent_metrics[key]
            for key in ("U", "S", "H", "ZS")
        }
        margin = float(calibration.margin().detach())
        success = (
            delta["H"] >= 0.05
            and delta["U"] >= -2.0
            and delta["S"] >= -2.0
            and abs(margin) < float(config["max_margin"]) * 0.98
        )
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        metrics = {
            "attempt_id": config["attempt_id"],
            "idea_id": config["idea_id"],
            "framework_id": config["framework_id"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "base_config_sha256": base_config_sha,
            "tst_config_sha256": tst_config_sha,
            "base_checkpoint_sha256": config["base_checkpoint_sha256"],
            "tst_gate_model_sha256": config["tst_gate_model_sha256"],
            "evaluation_protocol": h1.EVALUATION_PROTOCOL,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "parent_metrics_percent": parent_metrics,
            "candidate_metrics_percent": candidate_metrics,
            "delta_percent_points": delta,
            "learned_margin": margin,
            "objective": config.get("objective", "cross_entropy"),
            "success": success,
            "calibration_model_sha256": sha256_file(
                output_dir / "calibration_model.pth"
            ),
        }
        atomic_write_json(output_dir / "metrics.json", metrics)
        print(metrics)
        return metrics
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit)


if __name__ == "__main__":
    main()
