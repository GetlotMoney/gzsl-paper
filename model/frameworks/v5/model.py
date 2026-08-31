"""Stable deployment entry for the promoted FRAMEWORK-V5 computation."""

from __future__ import annotations

import torch

from model.frameworks.v4.evaluate_pclr_semantic_ensemble import semantic_ensemble_logits


V5_DEPLOYMENT = {
    "candidate_top_k": 17,
    "ridge_lambda": 0.3,
    "potential_cap": 0.5,
    "inference_relation_temperature": 0.2,
    "correction_scale": 6.95,
    "role0_weight": 0.16,
    "role6_weight": 0.36,
    "seen_logit_gamma": 0.91,
}


@torch.no_grad()
def v5_logits(model, image_features: torch.Tensor) -> torch.Tensor:
    """Return the complete 200-class V5 logits before any late class-axis slice."""
    _, _, full = semantic_ensemble_logits(
        model,
        image_features,
        role0_weight=V5_DEPLOYMENT["role0_weight"],
        role6_weight=V5_DEPLOYMENT["role6_weight"],
        gamma=V5_DEPLOYMENT["seen_logit_gamma"],
    )
    return full
