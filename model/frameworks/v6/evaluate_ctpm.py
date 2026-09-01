"""Evaluate a CTPM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from model.frameworks.v6.ctpm import CTPMModel
from model.frameworks.v6.ctpm_assets import CTPMAssets, load_ctpm_assets
from tools.gzsl_data import per_class_accuracy
from tools.run_contract import atomic_write_json


@torch.no_grad()
def predict(
    model: CTPMModel,
    features: torch.Tensor,
    patches: torch.Tensor,
    device: torch.device,
    *,
    class_ids: torch.Tensor | None = None,
    enable_s: bool = True,
    enable_v: bool = True,
    enable_i: bool = True,
    query_mode: str = "role_difference",
    no_l_role: bool = False,
    batch_size: int = 256,
) -> torch.Tensor:
    axis = torch.arange(model.class_count) if class_ids is None else class_ids.detach().cpu().long()
    predictions = []
    model.eval()
    for start in range(0, features.size(0), int(batch_size)):
        stop = min(start + int(batch_size), features.size(0))
        out = model(
            features[start:stop].to(device).float(),
            patches[start:stop].to(device).float(),
            class_ids=class_ids.to(device).long() if class_ids is not None else None,
            enable_s=enable_s,
            enable_v=enable_v,
            enable_i=enable_i,
            query_mode=query_mode,
            no_l_role=no_l_role,
        )
        if tuple(out.logits.shape) != (stop - start, axis.numel()) or not torch.isfinite(out.logits).all():
            raise ValueError("CTPM evaluation logits shape mismatch or NaN/Inf.")
        predictions.append(axis[out.logits.argmax(dim=1).detach().cpu()])
    return torch.cat(predictions)


def _metrics(seen_pred: torch.Tensor, unseen_pred: torch.Tensor, zs_pred: torch.Tensor, assets: CTPMAssets) -> dict[str, float]:
    s = 100.0 * per_class_accuracy(assets.test_seen_labels, seen_pred, assets.seen_classes)
    u = 100.0 * per_class_accuracy(assets.test_unseen_labels, unseen_pred, assets.unseen_classes)
    z = 100.0 * per_class_accuracy(assets.test_unseen_labels, zs_pred, assets.unseen_classes)
    h = 2.0 * s * u / (s + u) if s + u else 0.0
    return {"U": u, "S": s, "H": h, "ZS": z}


def _transitions(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    old = before.cpu().eq(labels.cpu())
    new = after.cpu().eq(labels.cpu())
    return {
        "corrected_wrong_to_right": int((~old & new).sum()),
        "damaged_right_to_wrong": int((old & ~new).sum()),
        "net_correct": int(new.sum() - old.sum()),
    }


@torch.no_grad()
def evaluate(model: CTPMModel, assets: CTPMAssets, device: torch.device, *, batch_size: int = 256) -> dict:
    variants = {
        "full": {},
        "parent": {"enable_s": False, "enable_v": False, "enable_i": False},
        "S_off": {"enable_s": False},
        "V_off": {"enable_v": False},
        "I_off": {"enable_i": False},
        "S_query_off": {"query_mode": "class_name_difference"},
        "margin_only_no_l_role": {"no_l_role": True},
    }
    predictions = {}
    for name, kwargs in variants.items():
        predictions[name] = {
            "seen": predict(model, assets.test_seen_features, assets.test_seen_patches, device, batch_size=batch_size, **kwargs),
            "unseen": predict(model, assets.test_unseen_features, assets.test_unseen_patches, device, batch_size=batch_size, **kwargs),
            "zs": predict(
                model,
                assets.test_unseen_features,
                assets.test_unseen_patches,
                device,
                class_ids=assets.unseen_classes,
                batch_size=batch_size,
                **kwargs,
            ),
        }
    scores = {
        name: _metrics(pred["seen"], pred["unseen"], pred["zs"], assets)
        for name, pred in predictions.items()
    }
    full = scores["full"]
    module_off = {name: scores[name] for name in ("S_off", "V_off", "I_off")}
    return {
        **full,
        "parent_metrics": scores["parent"],
        "module_off_metrics": module_off,
        "full_minus_parent_delta": {metric: full[metric] - scores["parent"][metric] for metric in ("U", "S", "H", "ZS")},
        "full_minus_off_delta": {
            name: {metric: full[metric] - scores[name][metric] for metric in ("U", "S", "H", "ZS")}
            for name in ("S_off", "V_off", "I_off")
        },
        "controls": {name: scores[name] for name in ("S_query_off", "margin_only_no_l_role")},
        "transitions_vs_parent": {
            "seen": _transitions(predictions["parent"]["seen"], predictions["full"]["seen"], assets.test_seen_labels),
            "unseen": _transitions(predictions["parent"]["unseen"], predictions["full"]["unseen"], assets.test_unseen_labels),
            "zs": _transitions(predictions["parent"]["zs"], predictions["full"]["zs"], assets.test_unseen_labels),
        },
    }


def build_model_from_checkpoint(checkpoint_path: Path, assets: CTPMAssets, device: torch.device) -> CTPMModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = CTPMModel(
        assets.class_name_embeds,
        assets.role_sentence_embeds,
        scale=float(config["logit_scale"]),
        hidden_dim=int(config["hidden_dim"]),
        patch_projection_dim=int(config["patch_projection_dim"]),
        max_margin=float(config["max_margin"]),
        max_role_weight=float(config["max_role_weight"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assets = load_ctpm_assets(config)
    device = torch.device(args.device)
    model = build_model_from_checkpoint(args.checkpoint, assets, device)
    result = evaluate(model, assets, device, batch_size=int(config["eval_batch_size"]))
    atomic_write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
