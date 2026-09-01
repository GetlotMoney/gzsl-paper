"""Evaluate the standalone FRAMEWORK-V7 export on the fixed CUB protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from model.frameworks.v4.train import load_assets, load_config
from model.frameworks.v7.model import load_v7_checkpoint
from tools.gzsl_data import per_class_accuracy
from tools.runtime import sha256_file


EXPECTED_METRICS = {
    "U": 77.60691046714783,
    "S": 83.639657497406,
    "H": 80.51043185404096,
    "ZS": 88.4734034538269,
}


@torch.no_grad()
def evaluate(config_path: Path, device: torch.device) -> dict[str, float]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(config["source_checkpoint"])
    asset_config_path = Path(config["asset_source_config"])
    if (
        config.get("schema_version") != "gzsl-paper.framework-v7.deploy.v1"
        or not checkpoint_path.is_absolute()
        or not asset_config_path.is_absolute()
        or sha256_file(checkpoint_path) != config["source_checkpoint_sha256"]
        or sha256_file(asset_config_path) != config["asset_source_config_sha256"]
    ):
        raise ValueError("FRAMEWORK-V7 config或source身份错误。")
    asset_config, _ = load_config(asset_config_path)
    tensors = load_assets(asset_config)
    model, _ = load_v7_checkpoint(checkpoint_path, map_location="cpu")
    model = model.to(device).eval()
    seen = torch.unique(tensors["train_labels"].long(), sorted=True)
    all_classes = torch.arange(200)
    unseen_cpu = all_classes[~torch.isin(all_classes, seen)]
    unseen = unseen_cpu.to(device)
    predictions = {"seen": [], "unseen": [], "zs": []}
    for split, features in (
        ("seen", tensors["test_seen_features"]),
        ("unseen", tensors["test_unseen_features"]),
    ):
        for start in range(0, len(features), 256):
            images = features[start : start + 256].to(device).float()
            logits = model(images)
            predictions[split].append(logits.argmax(dim=1).cpu())
            if split == "unseen":
                predictions["zs"].append(
                    unseen[logits.index_select(1, unseen).argmax(dim=1)].cpu()
                )
    for split in predictions:
        predictions[split] = torch.cat(predictions[split])
    labels_seen = tensors["test_seen_labels"].long()
    labels_unseen = tensors["test_unseen_labels"].long()
    s = 100.0 * per_class_accuracy(labels_seen, predictions["seen"], seen)
    u = 100.0 * per_class_accuracy(labels_unseen, predictions["unseen"], unseen_cpu)
    zs = 100.0 * per_class_accuracy(labels_unseen, predictions["zs"], unseen_cpu)
    h = 2.0 * s * u / (s + u)
    metrics = {"U": float(u), "S": float(s), "H": float(h), "ZS": float(zs)}
    for name, expected in EXPECTED_METRICS.items():
        if abs(metrics[name] - expected) > 1e-6:
            raise RuntimeError(f"FRAMEWORK-V7 {name}未复现晋级结果。")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, torch.device(args.device)), sort_keys=True))


if __name__ == "__main__":
    main()
