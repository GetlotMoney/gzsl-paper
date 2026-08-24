"""Evaluate the final M3 checkpoint with CCGR and TST-NTR disabled in forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.paper_v2 import PaperV2ThreeModuleModel
from model.tg_vpr_h1 import train as h1
from model.train_paper_v2 import _evaluate_model, load_assets, load_config
from tools.runtime import sha256_file


def run(config_path: Path, model_path: Path, output: Path, device_name: str) -> dict:
    if output.exists():
        raise FileExistsError(f"module-off输出已存在：{output}")
    config, config_sha = load_config(config_path)
    if config["condition_id"] != "M3_CCGR":
        raise ValueError("module-off诊断只接受M3_CCGR配置。")
    tensors, manifest, _ = load_assets(config)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config_sha:
        raise ValueError("model_best与配置SHA不一致。")
    device = torch.device(device_name)
    seen_classes = torch.tensor(manifest["seen_classes"], dtype=torch.long)
    centroids = h1.visual_centroids(tensors["train_features"], tensors["train_labels"].long(), seen_classes)
    model = PaperV2ThreeModuleModel(
        tensors["role_sentence_embeds"],
        seen_classes,
        centroids,
        tg_vpr_mode=config["tg_vpr_mode"],
        transport_mode=config["transport_mode"],
        ccgr_mode=config["ccgr_mode"],
        dropout=float(config["dropout"]),
        inner_ratio=float(config["inner_ratio"]),
        outer_ratio=float(config["outer_ratio"]),
        temperature=float(config["temperature"]),
        transport_hidden_dim=int(config["transport_hidden_dim"]),
        generator_hidden_dim=int(config["generator_hidden_dim"]),
        max_transport_step=float(config["max_transport_step"]),
        max_ntr_delta=float(config["max_ntr_delta"]),
        max_generator_magnitude=float(config["max_generator_magnitude"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    original_transport = model.transport_mode
    original_ccgr = model.ccgr_mode
    full = _evaluate_model(model, tensors, manifest, device)
    model.ccgr_mode = "off"
    ccgr_off = _evaluate_model(model, tensors, manifest, device)
    model.transport_mode = "off"
    transport_ccgr_off = _evaluate_model(model, tensors, manifest, device)
    model.transport_mode = original_transport
    model.ccgr_mode = original_ccgr
    result = {
        "schema_version": "gzsl-paper.module-off.v1",
        "dataset": config["dataset"],
        "condition_id": config["condition_id"],
        "config_sha256": config_sha,
        "model_sha256": sha256_file(model_path),
        "full": full,
        "ccgr_off": ccgr_off,
        "tst_ntr_ccgr_off": transport_ccgr_off,
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.model, args.output, args.device), ensure_ascii=False))


if __name__ == "__main__":
    main()
