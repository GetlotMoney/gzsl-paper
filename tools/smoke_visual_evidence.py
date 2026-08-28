"""Run one real CUB train-batch forward/backward for visual screen configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model.candidates.v2.trainers.paper_v2 import (
    _active_groups,
    _cache_visual_patches,
    _gradient_norms,
    _load_patch_batch,
    build_run_model,
    load_assets,
    load_config,
    set_trainable,
)
from model.candidates.v3.idea_133_visual_evidence.module import PaperV2VisualModel
from tools.reproducibility import configure_reproducibility
from tools.run_contract import require_finite_gradients


def run(config_paths: list[Path], device_name: str, batch_size: int) -> list[dict]:
    if not config_paths or not 1 <= int(batch_size) <= 8:
        raise ValueError("smoke需要至少一个配置且batch_size位于[1,8]。")
    first, _ = load_config(config_paths[0])
    configure_reproducibility(
        int(first["random_seed"]),
        strict_determinism=True,
        deterministic_warn_only=False,
    )
    tensors, manifest, manifest_path = load_assets(first)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("真实视觉smoke要求CUDA。")
    _cache_visual_patches(tensors, first, device)
    seen = torch.tensor(manifest["seen_classes"], dtype=torch.long, device=device)
    labels = tensors["train_labels"][: int(batch_size)].long().to(device)
    global_to_seen = torch.full(
        (int(manifest["class_count"]),), -1, dtype=torch.long, device=device
    )
    global_to_seen[seen] = torch.arange(seen.numel(), device=device)
    targets = global_to_seen.index_select(0, labels)
    indices = torch.arange(int(batch_size))
    images = tensors["train_features"][: int(batch_size)].to(device).float()
    patches = _load_patch_batch(tensors["train_patch_features"], indices, device)
    results = []
    for path in config_paths:
        config, config_sha = load_config(path)
        if Path(config["asset_manifest"]) != manifest_path:
            raise ValueError("smoke配置必须共享同一资产manifest。")
        model = build_run_model(config, tensors, manifest, device).train()
        names = _active_groups(model, "end_to_end_joint", "END_TO_END")
        set_trainable(model, names)
        model.zero_grad(set_to_none=True)
        assert isinstance(model, PaperV2VisualModel)
        components = model.score_components(
            images,
            patches,
            target_class_ids=labels,
        )
        final_scores = components["final_scores"]
        assert isinstance(final_scores, torch.Tensor)
        final_ce = F.cross_entropy(final_scores.index_select(1, seen), targets)
        losses = model.visual_losses(
            components,
            seen,
            targets,
            labels,
            hard_margin=float(config["visual_hard_margin"]),
        )
        topology = model.topology_loss(model.prototypes())
        total = (
            final_ce
            + float(config["topology_weight"]) * topology
            + float(config["visual_part_weight"]) * losses["part"]
            + float(config["visual_diversity_weight"]) * losses["diversity"]
            + float(config["visual_anchor_weight"]) * losses["anchor"]
            + float(config["visual_hard_weight"]) * losses["hard"]
        )
        if not torch.isfinite(total):
            raise FloatingPointError("视觉smoke总loss非有限。")
        total.backward()
        require_finite_gradients(model)
        gradients = _gradient_norms(model)
        if config["visual_mode"] != "off" and gradients["visual"] <= 0:
            raise RuntimeError("启用视觉模块但真实batch视觉梯度为0。")
        results.append(
            {
                "config": str(path),
                "config_sha256": config_sha,
                "visual_mode": config["visual_mode"],
                "loss": float(total.detach()),
                "part_loss": float(losses["part"].detach()),
                "diversity_loss": float(losses["diversity"].detach()),
                "anchor_loss": float(losses["anchor"].detach()),
                "hard_loss": float(losses["hard"].detach()),
                "gradient_norms": gradients,
                "diagnostics": model.diagnostics(),
            }
        )
        del model
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, args.device, args.batch_size),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
