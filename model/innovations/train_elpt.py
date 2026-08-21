from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.innovations.elpt import (
    ELPTGate,
    VariableClassTGVPR,
    blend_prototypes,
    fixed_class_folds,
    gate_features,
    topology_loss,
)
from model.tg_vpr_h1 import TGVPRH1FixedEqual
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "attempt_id",
    "idea_id",
    "framework_id",
    "base_config",
    "base_checkpoint",
    "base_checkpoint_sha256",
    "seed",
    "fold_count",
    "fold_epochs",
    "gate_epochs",
    "gate_batch_half",
    "gate_lr",
    "gate_weight_decay",
    "topology_weight",
}
OPTIONAL_CONFIG_KEYS = {"gate_max_alpha", "alpha_penalty", "fold_checkpoint_dir"}


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


class FrozenPrototypeClassifier(nn.Module):
    def __init__(self, prototypes: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("_prototypes", prototypes.detach())
        self.register_buffer("_scale", scale.detach())

    def prototypes(self):
        return self._prototypes

    def scale(self):
        return self._scale


def load_config(path: Path):
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ELPT配置必须是字典。")
    missing = REQUIRED_CONFIG_KEYS - set(raw)
    extra = set(raw) - REQUIRED_CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
    if missing or extra:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(
            f"ELPT配置字段错误；缺少={sorted(missing)}，多出={sorted(extra)}。"
        )
    raw.setdefault("gate_max_alpha", 1.0)
    raw.setdefault("alpha_penalty", 0.0)
    raw.setdefault("fold_checkpoint_dir", None)
    if raw["schema_version"] != "gzsl-paper.elpt.v1":
        raise ValueError("ELPT schema错误。")
    if raw["attempt_id"] not in {"V2-TRY-006", "V2-TRY-007"} or raw["idea_id"] != "IDEA-002":
        raise ValueError("ELPT首次TRY身份不匹配。")
    if raw["framework_id"] != "FRAMEWORK-V2":
        raise ValueError("ELPT只接受FRAMEWORK-V2。")
    if int(raw["fold_count"]) != 3 or int(raw["fold_epochs"]) != 50:
        raise ValueError("ELPT固定3折和50轮fold训练。")
    if int(raw["gate_epochs"]) != 20 or int(raw["gate_batch_half"]) != 32:
        raise ValueError("ELPT固定20轮gate训练和32/32平衡batch。")
    if float(raw["gate_lr"]) != 1e-3 or float(raw["gate_weight_decay"]) != 1e-4:
        raise ValueError("ELPT gate优化器参数不匹配。")
    if float(raw["topology_weight"]) != 0.1:
        raise ValueError("ELPT固定topology_weight=0.1。")
    if raw["attempt_id"] == "V2-TRY-006" and (
        float(raw["gate_max_alpha"]) != 1.0 or float(raw["alpha_penalty"]) != 0.0
    ):
        raise ValueError("TRY-006必须使用无上限补救的初始ELPT。")
    if raw["attempt_id"] == "V2-TRY-007" and (
        float(raw["gate_max_alpha"]) != 0.25 or float(raw["alpha_penalty"]) != 0.01
    ):
        raise ValueError("TRY-007必须使用0.25上限和0.01 alpha约束。")
    if raw["attempt_id"] == "V2-TRY-007" and not raw["fold_checkpoint_dir"]:
        raise ValueError("TRY-007必须复用TRY-006 fold checkpoint。")
    return raw, sha256_file(path)


def _stage_boundaries(stages):
    total = 0
    boundaries = []
    for stage in stages:
        total += int(stage["epochs"])
        boundaries.append(total)
    return boundaries


def _train_fold(
    fold_id,
    pseudo_seen,
    sentence_embeds,
    train_features,
    train_labels,
    base_config,
    device,
    output_dir,
    seed,
    print_log,
):
    mask = torch.isin(train_labels.long(), pseudo_seen)
    features = train_features[mask]
    labels = train_labels.long()[mask]
    centroids = h1.visual_centroids(train_features, train_labels.long(), pseudo_seen)
    model = VariableClassTGVPR(
        sentence_embeds,
        pseudo_seen,
        centroids,
        dropout=base_config["dropout"],
        inner_ratio=base_config["inner_ratio"],
        outer_ratio=base_config["outer_ratio"],
        temperature=base_config["temperature"],
    ).to(device)
    stages = base_config["lr_stages"]
    boundaries = _stage_boundaries(stages)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(stages[0]["lr"]),
        weight_decay=float(base_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(stages[0]["epochs"]),
        eta_min=float(stages[0]["eta_min"]),
    )
    active_stage = 0
    label_map = torch.full((200,), -1, dtype=torch.long)
    label_map[pseudo_seen] = torch.arange(pseudo_seen.numel())
    generator = torch.Generator(device="cpu").manual_seed(seed * 100 + fold_id)
    iters = labels.numel() // int(base_config["batch_size"])
    for epoch in range(1, 51):
        target_stage = next(i for i, boundary in enumerate(boundaries) if epoch <= boundary)
        if target_stage != active_stage:
            active_stage = target_stage
            stage = stages[active_stage]
            for group in optimizer.param_groups:
                group["lr"] = float(stage["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(stage["epochs"]),
                eta_min=float(stage["eta_min"]),
            )
        model.train()
        loss_sum = 0.0
        for _ in range(iters):
            indices = torch.randperm(labels.numel(), generator=generator)[: int(base_config["batch_size"])]
            batch = features[indices].to(device).float()
            targets = label_map[labels[indices]].to(device)
            optimizer.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model.logits(batch, pseudo_seen), targets)
            topo = model.topology_loss()
            loss = ce + float(base_config["topology_weight"]) * topo
            if not torch.isfinite(loss):
                raise FloatingPointError(f"fold {fold_id} loss非有限。")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        scheduler.step()
        if epoch in (1, 10, 20, 30, 40, 50):
            print_log(f"fold={fold_id} epoch={epoch} loss={loss_sum/iters:.6f}")
    model.eval()
    checkpoint = {
        "fold_id": fold_id,
        "seed": seed,
        "pseudo_seen": pseudo_seen.tolist(),
        "model_state_dict": copy.deepcopy(model.state_dict()),
    }
    torch.save(checkpoint, output_dir / f"fold_{fold_id}.pth")
    return model


def _load_fold_checkpoint(
    fold_id,
    pseudo_seen,
    sentence_embeds,
    train_features,
    train_labels,
    base_config,
    device,
    checkpoint_dir,
):
    path = Path(checkpoint_dir) / f"fold_{fold_id}.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("fold_id") != fold_id or checkpoint.get("pseudo_seen") != pseudo_seen.tolist():
        raise ValueError(f"fold {fold_id} checkpoint身份不匹配。")
    centroids = h1.visual_centroids(train_features, train_labels.long(), pseudo_seen)
    model = VariableClassTGVPR(
        sentence_embeds,
        pseudo_seen,
        centroids,
        dropout=base_config["dropout"],
        inner_ratio=base_config["inner_ratio"],
        outer_ratio=base_config["outer_ratio"],
        temperature=base_config["temperature"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _fold_package(model, pseudo_seen, pseudo_unseen, tensors, seenclasses, device):
    model.eval()
    with torch.no_grad():
        base_all = model.base_prototypes()
        fold_full = model.prototypes()
        value = model.value_candidate(pseudo_unseen.to(device))
        features = gate_features(
            base_all.index_select(0, pseudo_unseen.to(device)),
            value,
            base_all.index_select(0, pseudo_seen.to(device)),
        )
    seen_sample_mask = torch.isin(tensors["train_labels"].long(), pseudo_seen)
    unseen_sample_mask = torch.isin(tensors["train_labels"].long(), pseudo_unseen)
    return {
        "pseudo_seen": pseudo_seen,
        "pseudo_unseen": pseudo_unseen,
        "base_all": base_all.detach().cpu(),
        "fold_full": fold_full.detach().cpu(),
        "value": value.detach().cpu(),
        "gate_features": features.detach().cpu(),
        "seen_indices": seen_sample_mask.nonzero(as_tuple=False).flatten(),
        "unseen_indices": unseen_sample_mask.nonzero(as_tuple=False).flatten(),
        "scale": model.scale().detach().cpu(),
        "seenclasses": seenclasses,
    }


def _train_gate(packages, tensors, config, device, seed, print_log):
    gate = ELPTGate(max_alpha=float(config["gate_max_alpha"])).to(device)
    optimizer = torch.optim.Adam(
        gate.parameters(),
        lr=float(config["gate_lr"]),
        weight_decay=float(config["gate_weight_decay"]),
    )
    label_map = torch.full((200,), -1, dtype=torch.long)
    seenclasses = packages[0]["seenclasses"]
    label_map[seenclasses] = torch.arange(150)
    generators = [
        torch.Generator(device="cpu").manual_seed(seed * 1000 + fold_id)
        for fold_id in range(3)
    ]
    half = int(config["gate_batch_half"])
    for epoch in range(1, int(config["gate_epochs"]) + 1):
        gate.train()
        total_loss = 0.0
        total_steps = 0
        for fold_id, package in enumerate(packages):
            seen_indices = package["seen_indices"]
            unseen_indices = package["unseen_indices"]
            steps = min(seen_indices.numel() // half, unseen_indices.numel() // half)
            for _ in range(steps):
                generator = generators[fold_id]
                si = seen_indices[torch.randperm(seen_indices.numel(), generator=generator)[:half]]
                ui = unseen_indices[torch.randperm(unseen_indices.numel(), generator=generator)[:half]]
                indices = torch.cat((si, ui))
                images = tensors["train_features"][indices].to(device).float()
                targets = label_map[tensors["train_labels"].long()[indices]].to(device)
                base_all = package["base_all"].to(device)
                final_all = package["fold_full"].to(device).clone()
                features = package["gate_features"].to(device)
                alpha = gate(features)
                pseudo_unseen = package["pseudo_unseen"].to(device)
                final_all[pseudo_unseen] = blend_prototypes(
                    base_all.index_select(0, pseudo_unseen),
                    package["value"].to(device),
                    alpha,
                )
                competition = final_all.index_select(0, seenclasses.to(device))
                logits = F.normalize(images, dim=-1) @ competition.T * package["scale"].to(device)
                ce = F.cross_entropy(logits, targets)
                topo = topology_loss(
                    base_all.index_select(0, seenclasses.to(device)), competition
                )
                alpha_regularization = alpha.square().mean()
                loss = (
                    ce
                    + float(config["topology_weight"]) * topo
                    + float(config["alpha_penalty"]) * alpha_regularization
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("ELPT gate loss非有限。")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in gate.parameters()
                ):
                    raise FloatingPointError("ELPT gate梯度非有限。")
                optimizer.step()
                total_loss += float(loss.detach())
                total_steps += 1
        print_log(f"gate_epoch={epoch} loss={total_loss/total_steps:.6f}")
    return gate.eval()


def _candidate_prototypes(model, gate, seenclasses, unseenclasses, device):
    model.eval()
    with torch.no_grad():
        base_all = model.base_prototypes()
        final_all = model.prototypes().clone()
        value = model.value_candidate(unseenclasses.to(device))
        features = gate_features(
            base_all.index_select(0, unseenclasses.to(device)),
            value,
            base_all.index_select(0, seenclasses.to(device)),
        )
        alpha = gate(features)
        final_all[unseenclasses.to(device)] = blend_prototypes(
            base_all.index_select(0, unseenclasses.to(device)), value, alpha
        )
    return final_all, alpha


def run(config_path: Path, output_dir: Path, expected_commit: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前HEAD不一致。")
    config, config_sha = load_config(config_path)
    base_config_path = Path(config["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = Path.cwd() / base_config_path
    base_config, base_config_sha = h1.load_config(base_config_path)
    paths = h1.resolve_paths(base_config)
    input_sha = h1.verify_inputs(base_config, paths, h1.TRAINING_KEYS)
    checkpoint_path = Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path) != config["base_checkpoint_sha256"]:
        raise ValueError("V2基线checkpoint SHA不匹配。")
    device = torch.device(base_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ELPT正式TRY要求CUDA。")

    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)

    def print_log(value):
        print(value)

    try:
        seed = int(config["seed"])
        configure_reproducibility(seed, strict_determinism=True, deterministic_warn_only=False)
        tensors = {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in ("sentence_embeds", "train_features", "train_labels")
        }
        if set(tensors) != {"sentence_embeds", "train_features", "train_labels"}:
            raise RuntimeError("ELPT训练阶段只能加载seen训练输入。")
        seenclasses = torch.unique(tensors["train_labels"].long(), sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        folds = fixed_class_folds(seenclasses)
        packages = []
        for fold_id, (pseudo_seen, pseudo_unseen) in enumerate(folds):
            if config["fold_checkpoint_dir"]:
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
                print_log(f"fold={fold_id} reused_from={config['fold_checkpoint_dir']}")
            else:
                fold_model = _train_fold(
                    fold_id,
                    pseudo_seen,
                    tensors["sentence_embeds"],
                    tensors["train_features"],
                    tensors["train_labels"],
                    base_config,
                    device,
                    output_dir,
                    seed,
                    print_log,
                )
            packages.append(
                _fold_package(
                    fold_model,
                    pseudo_seen,
                    pseudo_unseen,
                    tensors,
                    seenclasses,
                    device,
                )
            )
            del fold_model
            torch.cuda.empty_cache()

        gate = _train_gate(packages, tensors, config, device, seed, print_log)
        torch.save(
            {
                "attempt_id": config["attempt_id"],
                "code_commit": code_commit,
                "config": config,
                "gate_state_dict": copy.deepcopy(gate.state_dict()),
            },
            output_dir / "gate_model.pth",
        )

        # official test只在所有训练完成后加载。
        input_sha.update(h1.verify_inputs(base_config, paths, h1.OFFICIAL_KEYS))
        tensors.update(
            {
                name: torch.load(paths[name], map_location="cpu", weights_only=True)
                for name in h1.OFFICIAL_KEYS
            }
        )
        baseline_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        baseline = TGVPRH1FixedEqual(
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
        baseline.load_state_dict(baseline_checkpoint["model_state_dict"], strict=True)
        variable = VariableClassTGVPR(
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
        variable.load_state_dict(baseline_checkpoint["model_state_dict"], strict=True)
        baseline = baseline.to(device).eval()
        variable = variable.to(device).eval()
        baseline_metrics = h1.evaluate(
            baseline, tensors, seenclasses, unseenclasses, device
        )
        candidate_prototypes, alpha = _candidate_prototypes(
            variable, gate, seenclasses, unseenclasses, device
        )
        candidate = FrozenPrototypeClassifier(
            candidate_prototypes, variable.scale()
        ).to(device)
        candidate_metrics = h1.evaluate(
            candidate, tensors, seenclasses, unseenclasses, device
        )
        delta = {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in ("U", "S", "H", "ZS")
        }
        alpha_stats = {
            "mean": float(alpha.mean()),
            "std": float(alpha.std(unbiased=False)),
            "min": float(alpha.min()),
            "max": float(alpha.max()),
        }
        success = (
            candidate_metrics["H"] > 74.023182
            and candidate_metrics["U"] > baseline_metrics["U"]
            and delta["S"] >= -2.0
            and 0.02 < alpha_stats["mean"] < 0.50
            and alpha_stats["std"] > 0.01
        )
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        metrics = {
            "attempt_id": config["attempt_id"],
            "idea_id": config["idea_id"],
            "framework_id": config["framework_id"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "base_config_sha256": base_config_sha,
            "base_checkpoint_sha256": config["base_checkpoint_sha256"],
            "evaluation_protocol": h1.EVALUATION_PROTOCOL,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "folds": [
                {
                    "pseudo_seen": pseudo_seen.tolist(),
                    "pseudo_unseen": pseudo_unseen.tolist(),
                }
                for pseudo_seen, pseudo_unseen in folds
            ],
            "baseline_metrics_percent": baseline_metrics,
            "candidate_metrics_percent": candidate_metrics,
            "delta_percent_points": delta,
            "alpha_stats": alpha_stats,
            "gate_max_alpha": float(config["gate_max_alpha"]),
            "alpha_penalty": float(config["alpha_penalty"]),
            "fold_checkpoint_dir": config["fold_checkpoint_dir"],
            "success": success,
            "gate_model_sha256": sha256_file(output_dir / "gate_model.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", metrics)
        print_log(metrics)
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
