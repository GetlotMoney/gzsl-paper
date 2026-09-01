"""Same-checkpoint evaluation for CTPM and its pure module offs."""

from __future__ import annotations

import numpy as np
import torch

from model.frameworks.v6.ctpm import CTPMModel
from model.frameworks.v6.ctpm_assets import CTPMEvalAssets
from tools.gzsl_data import per_class_accuracy


def _patch_batch(patches: np.memmap, start: int, stop: int, device: torch.device):
    value = np.asarray(patches[start:stop], dtype=np.float32).copy()
    return torch.from_numpy(value).to(device)


@torch.no_grad()
def predict(
    model: CTPMModel,
    features: torch.Tensor,
    patches: np.memmap,
    device: torch.device,
    *,
    class_ids: torch.Tensor | None = None,
    enable_s: bool = True,
    enable_v: bool = True,
    enable_i: bool = True,
    query_mode: str = "role_difference",
    no_l_role: bool = False,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(model.class_count) if class_ids is None else class_ids.cpu().long()
    predictions, pair_membership = [], []
    model.eval()
    for start in range(0, features.size(0), batch_size):
        stop = min(start + batch_size, features.size(0))
        labels_axis = None if class_ids is None else class_ids.to(device).long()
        output = model(
            features[start:stop].to(device).float(),
            _patch_batch(patches, start, stop, device),
            class_ids=labels_axis,
            enable_s=enable_s,
            enable_v=enable_v,
            enable_i=enable_i,
            query_mode=query_mode,
            no_l_role=no_l_role,
        )
        if not torch.isfinite(output.logits).all():
            raise FloatingPointError("CTPM evaluation logits are non-finite.")
        predictions.append(axis[output.logits.argmax(dim=1).cpu()])
        pair_membership.append(output.top2_global.cpu())
    return torch.cat(predictions), torch.cat(pair_membership)


def _metrics(seen_pred, unseen_pred, zs_pred, assets: CTPMEvalAssets):
    s = 100 * per_class_accuracy(assets.test_seen_labels, seen_pred, assets.seen_classes)
    u = 100 * per_class_accuracy(
        assets.test_unseen_labels, unseen_pred, assets.unseen_classes
    )
    z = 100 * per_class_accuracy(assets.test_unseen_labels, zs_pred, assets.unseen_classes)
    return {"U": u, "S": s, "H": 2 * s * u / (s + u), "ZS": z}


def _transitions(before, after, labels):
    old, new = before.eq(labels), after.eq(labels)
    return {
        "corrected": int((~old & new).sum()),
        "damaged": int((old & ~new).sum()),
        "net": int(new.sum() - old.sum()),
    }


@torch.no_grad()
def evaluate(
    model: CTPMModel,
    assets: CTPMEvalAssets,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> dict:
    variants = {
        "full": {},
        "parent": {"enable_s": False, "enable_v": False, "enable_i": False},
        "S_off": {"enable_s": False},
        "V_off": {"enable_v": False},
        "I_off": {"enable_i": False},
        "S_query_off": {"query_mode": "class_name_difference"},
        "margin_only_no_l_role": {"no_l_role": True},
    }
    predictions, pairs, scores = {}, {}, {}
    for name, kwargs in variants.items():
        seen, seen_pair = predict(
            model, assets.test_seen_features, assets.test_seen_patches,
            device, batch_size=batch_size, **kwargs
        )
        unseen, unseen_pair = predict(
            model, assets.test_unseen_features, assets.test_unseen_patches,
            device, batch_size=batch_size, **kwargs
        )
        zs, _ = predict(
            model, assets.test_unseen_features, assets.test_unseen_patches,
            device, class_ids=assets.unseen_classes, batch_size=batch_size, **kwargs
        )
        predictions[name] = {"seen": seen, "unseen": unseen, "zs": zs}
        pairs[name] = {"seen": seen_pair, "unseen": unseen_pair}
        scores[name] = _metrics(seen, unseen, zs, assets)
    full = scores["full"]
    gaps = {
        name: {metric: full[metric] - scores[name][metric] for metric in ("U", "S", "H", "ZS")}
        for name in ("S_off", "V_off", "I_off")
    }
    parent_pair_equal = all(
        torch.equal(pairs["parent"][split], pairs[name][split])
        for name in variants
        for split in ("seen", "unseen")
    )
    if not parent_pair_equal:
        raise RuntimeError("CTPM candidate pair changed across module-off conditions.")
    stratified = {}
    for split, labels in (
        ("seen", assets.test_seen_labels),
        ("unseen", assets.test_unseen_labels),
    ):
        in_pair = labels[:, None].eq(pairs["parent"][split]).any(dim=1)
        stratified[split] = {
            "inside_top2": _transitions(
                predictions["parent"][split][in_pair],
                predictions["full"][split][in_pair], labels[in_pair]
            ),
            "outside_top2": _transitions(
                predictions["parent"][split][~in_pair],
                predictions["full"][split][~in_pair], labels[~in_pair]
            ),
        }
    return {
        **full,
        "parent_metrics": scores["parent"],
        "module_off_metrics": {name: scores[name] for name in ("S_off", "V_off", "I_off")},
        "full_minus_off_delta": gaps,
        "full_minus_parent_delta": {
            metric: full[metric] - scores["parent"][metric]
            for metric in ("U", "S", "H", "ZS")
        },
        "controls": {
            name: scores[name] for name in ("S_query_off", "margin_only_no_l_role")
        },
        "pair_identity_equal_across_offs": parent_pair_equal,
        "stratified_transitions_vs_parent": stratified,
    }


def build_model_from_checkpoint(checkpoint: dict, assets: CTPMEvalAssets, device):
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
