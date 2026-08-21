from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.icgr import ICGRClassifier
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


CONFIG_KEYS = {
    "schema_version",
    "attempt_id",
    "idea_id",
    "framework_id",
    "base_config",
    "base_checkpoint",
    "base_checkpoint_sha256",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "hidden_dim",
}
CONFIG_KEYS_V2 = CONFIG_KEYS | {"router_input_mode"}


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


def load_config(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    expected = CONFIG_KEYS_V2 if isinstance(config, dict) and config.get("schema_version") == "gzsl-paper.icgr.v2" else CONFIG_KEYS
    if not isinstance(config, dict) or actual != expected:
        raise ValueError(
            f"ICGR配置字段不匹配；缺少={sorted(expected-actual)}，"
            f"多出={sorted(actual-expected)}。"
        )
    if config["schema_version"] not in ("gzsl-paper.icgr.v1", "gzsl-paper.icgr.v2"):
        raise ValueError("ICGR配置schema错误。")
    if config["idea_id"] != "IDEA-003" or config["framework_id"] != "FRAMEWORK-V2":
        raise ValueError("ICGR研究身份不匹配。")
    if int(config["epochs"]) != 10 or int(config["hidden_dim"]) != 64:
        raise ValueError("ICGR首次TRY固定训练10 epoch、隐藏维64。")
    if int(config["batch_size"]) <= 0:
        raise ValueError("ICGR batch_size必须为正数。")
    if float(config["lr"]) != 1e-3 or float(config["weight_decay"]) != 1e-4:
        raise ValueError("ICGR首次TRY固定Adam lr=1e-3、weight_decay=1e-4。")
    if config.get("router_input_mode", "image_cls") not in (
        "image_cls",
        "image_cls_role_cosine",
    ):
        raise ValueError("ICGR路由输入模式错误。")
    return config, sha256_file(path)


def build_parent(tensors, base_config, checkpoint, seenclasses, device):
    parent = TGVPRH1FixedEqual(
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
    return parent.to(device).eval()


def _per_class(labels, predictions, classes) -> float:
    return h1.per_class_accuracy(labels, predictions, classes)


@torch.no_grad()
def evaluate_icgr(model, tensors, seenclasses, unseenclasses, device, batch_size=512):
    model.eval()

    def predict(features, class_ids=None):
        outputs = []
        weights = []
        for start in range(0, features.size(0), batch_size):
            batch = features[start : start + batch_size].to(device).float()
            outputs.append(model.logits(batch, class_ids).argmax(dim=1).cpu())
            weights.append(model.route_weights(batch).cpu())
        predictions = torch.cat(outputs)
        if class_ids is not None:
            predictions = class_ids[predictions]
        return predictions, torch.cat(weights)

    seen_pred, seen_weights = predict(tensors["seen_features"])
    unseen_pred, unseen_weights = predict(tensors["unseen_features"])
    unseen_only, _ = predict(tensors["unseen_features"], unseenclasses)
    seen = _per_class(tensors["seen_labels"], seen_pred, seenclasses)
    unseen = _per_class(tensors["unseen_labels"], unseen_pred, unseenclasses)
    zsl = _per_class(tensors["unseen_labels"], unseen_only, unseenclasses)
    harmonic = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    all_weights = torch.cat((seen_weights, unseen_weights))
    stats = {
        "mean": all_weights.mean(dim=0).tolist(),
        "seen_mean": seen_weights.mean(dim=0).tolist(),
        "unseen_mean": unseen_weights.mean(dim=0).tolist(),
        "min": all_weights.min(dim=0).values.tolist(),
        "max": all_weights.max(dim=0).values.tolist(),
    }
    return {
        "U": unseen * 100,
        "S": seen * 100,
        "H": harmonic * 100,
        "ZS": zsl * 100,
    }, stats


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
    paths = h1.resolve_paths(base_config)
    input_sha = h1.verify_inputs(base_config, paths, h1.TRAINING_KEYS)
    checkpoint_path = Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path) != config["base_checkpoint_sha256"]:
        raise ValueError("V2基线checkpoint SHA不匹配。")
    device = torch.device(base_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ICGR正式TRY要求CUDA。")

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
        if tensors["train_labels"].numel() != 7057:
            raise ValueError("ICGR必须使用全部7057张seen训练图像。")
        seenclasses = torch.unique(tensors["train_labels"].long(), sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        if seenclasses.numel() != 150 or unseenclasses.numel() != 50:
            raise ValueError("ICGR训练边界必须是150 seen / 50 true unseen类。")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        parent = build_parent(tensors, base_config, checkpoint, seenclasses, device)
        model = ICGRClassifier(
            parent,
            hidden_dim=int(config["hidden_dim"]),
            router_input_mode=config.get("router_input_mode", "image_cls"),
        ).to(device)
        initial = model.route_weights(tensors["train_features"][:4].to(device).float())
        expected = torch.full_like(initial, 1.0 / 3.0)
        if not torch.equal(initial, expected):
            raise RuntimeError("ICGR初始权重不是严格1/3。")
        with torch.no_grad():
            initial_error = float(
                (
                    model.logits(tensors["train_features"][:16].to(device).float())
                    - parent.logits(tensors["train_features"][:16].to(device).float())
                )
                .abs()
                .max()
            )
        if initial_error > 2e-5:
            raise RuntimeError("ICGR初始logits未复现父条件。")

        optimizer = torch.optim.Adam(
            model.router.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        labels = tensors["train_labels"].long()
        global_to_seen = torch.full((200,), -1, dtype=torch.long)
        global_to_seen[seenclasses] = torch.arange(150)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        history = []
        for epoch in range(1, int(config["epochs"]) + 1):
            model.parent.eval()
            model.router.train()
            permutation = torch.randperm(labels.numel(), generator=generator)
            loss_sum = 0.0
            sample_count = 0
            for start in range(0, labels.numel(), int(config["batch_size"])):
                indices = permutation[start : start + int(config["batch_size"])]
                features = tensors["train_features"][indices].to(device).float()
                targets = global_to_seen[labels[indices]].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.logits(features, seenclasses), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("ICGR loss包含NaN/Inf。")
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.router.parameters()
                ):
                    raise FloatingPointError("ICGR gate梯度包含NaN/Inf。")
                if any(parameter.grad is not None for parameter in model.parent.parameters()):
                    raise RuntimeError("冻结TG-VPR参数意外获得梯度。")
                optimizer.step()
                loss_sum += float(loss.detach()) * indices.numel()
                sample_count += indices.numel()
            row = {"epoch": epoch, "train_ce": loss_sum / sample_count}
            history.append(row)
            print(f"epoch={epoch} train_ce={row['train_ce']:.6f}")

        gate_payload = {
            "attempt_id": config["attempt_id"],
            "code_commit": code_commit,
            "config": config,
            "epoch": int(config["epochs"]),
            "router_state_dict": copy.deepcopy(model.router.state_dict()),
            "history": history,
        }
        torch.save(gate_payload, output_dir / "gate_model.pth")
        torch.save(gate_payload, output_dir / "checkpoint_last.pth")

        # official test严格在路由训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config, paths, h1.OFFICIAL_KEYS))
        tensors.update(
            {
                name: torch.load(paths[name], map_location="cpu", weights_only=True)
                for name in h1.OFFICIAL_KEYS
            }
        )
        baseline_metrics = h1.evaluate(parent, tensors, seenclasses, unseenclasses, device)
        candidate_metrics, weight_stats = evaluate_icgr(
            model, tensors, seenclasses, unseenclasses, device
        )
        delta = {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in ("U", "S", "H", "ZS")
        }
        success = (
            delta["H"] >= 0.20
            and delta["U"] >= -2.0
            and delta["S"] >= -2.0
            and min(weight_stats["mean"]) >= 0.05
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
            "baseline_off_initial_max_abs_error": initial_error,
            "baseline_metrics_percent": baseline_metrics,
            "candidate_metrics_percent": candidate_metrics,
            "delta_percent_points": delta,
            "group_weight_stats": weight_stats,
            "router_input_mode": config.get("router_input_mode", "image_cls"),
            "success": success,
            "gate_model_sha256": sha256_file(output_dir / "gate_model.pth"),
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
