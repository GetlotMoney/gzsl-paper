from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceAwareAttributeResidual(nn.Module):
    """根据每张图的属性预测置信度微调冻结ARA的融合强度。"""

    def __init__(self, base_ara, max_beta_residual=4.0):
        super().__init__()
        self.base_ara = base_ara
        for parameter in self.base_ara.parameters():
            parameter.requires_grad_(False)
        self.max_beta_residual = float(max_beta_residual)
        self.gate = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def confidence_features(self, images, class_ids=None):
        raw = F.normalize(images.float(), dim=-1) @ self.base_ara.ridge_weight
        norm = raw.norm(dim=-1)
        scores = F.normalize(raw, dim=-1) @ (
            self.base_ara.class_attributes
            if class_ids is None
            else self.base_ara.class_attributes.index_select(
                0, class_ids.to(raw.device)
            )
        ).T
        top2 = scores.topk(2, dim=-1).values
        probabilities = F.softmax(scores, dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(scores.shape[1])
        return torch.stack((norm, top2[:, 0], top2[:, 0] - top2[:, 1], entropy), dim=1)

    def beta(self, images, class_ids=None):
        residual = self.max_beta_residual * torch.tanh(
            self.gate(self.confidence_features(images, class_ids))
        ).squeeze(-1)
        return self.base_ara.beta() + residual

    def logits(self, images, prototypes, scale, class_ids=None, enabled=True):
        selected = prototypes
        if class_ids is not None:
            selected = prototypes.index_select(0, class_ids.to(prototypes.device))
        parent = F.normalize(images.float(), dim=-1) @ selected.T * scale
        attribute = self.base_ara.attribute_logits(images, class_ids)
        if not enabled:
            return parent + self.base_ara.beta() * attribute
        return parent + self.beta(images, class_ids).unsqueeze(1) * attribute

    def residual_stats(self, images, class_ids=None):
        residual = self.beta(images, class_ids).detach() - self.base_ara.beta().detach()
        return {
            "mean": float(residual.mean()),
            "std": float(residual.std(unbiased=False)),
            "min": float(residual.min()),
            "max": float(residual.max()),
        }
