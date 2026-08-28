from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def attribute_pca_features(attributes, rank):
    normalized = F.normalize(attributes.detach().float(), dim=-1)
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    features = centered @ right[: int(rank)].T
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (features - mean) / std


class ClassConditionedAttributeResidual(nn.Module):
    """根据类别属性因子预测CRA的类别级beta残差。"""

    def __init__(self, base_ara, class_features, max_beta_residual=4.0):
        super().__init__()
        self.base_ara = base_ara
        for parameter in self.base_ara.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("class_features", class_features.detach().float())
        self.max_beta_residual = float(max_beta_residual)
        self.gate = nn.Sequential(
            nn.Linear(class_features.shape[1], 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def beta_values(self, class_ids=None):
        features = self.class_features
        if class_ids is not None:
            features = features.index_select(0, class_ids.to(features.device))
        residual = self.max_beta_residual * torch.tanh(self.gate(features)).squeeze(-1)
        return self.base_ara.beta() + residual

    def logits(self, images, prototypes, scale, class_ids=None, enabled=True):
        selected = prototypes
        if class_ids is not None:
            selected = prototypes.index_select(0, class_ids.to(prototypes.device))
        parent = F.normalize(images.float(), dim=-1) @ selected.T * scale
        attribute = self.base_ara.attribute_logits(images, class_ids)
        if not enabled:
            return parent + self.base_ara.beta() * attribute
        return parent + attribute * self.beta_values(class_ids).unsqueeze(0)

    def residual_stats(self, class_ids=None):
        residual = self.beta_values(class_ids).detach() - self.base_ara.beta().detach()
        return {
            "mean": float(residual.mean()),
            "std": float(residual.std(unbiased=False)),
            "min": float(residual.min()),
            "max": float(residual.max()),
        }
