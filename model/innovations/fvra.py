from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureVisualResidualAdapter(nn.Module):
    """冻结CLIP CLS上的零初始化低秩残差适配器。"""

    def __init__(self, hidden_dim: int = 64, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.network = nn.Sequential(
            nn.Linear(768, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 768),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, *, enabled=True):
        normalized = F.normalize(features.float(), dim=-1)
        if not enabled:
            return normalized
        return F.normalize(
            normalized + self.residual_scale * self.network(normalized), dim=-1
        )

    def residual_stats(self, features: torch.Tensor):
        normalized = F.normalize(features.float(), dim=-1)
        residual = self.residual_scale * self.network(normalized)
        norm = residual.norm(dim=-1)
        return {"mean": float(norm.mean()), "max": float(norm.max())}


class FeatureAdapterClassifier(nn.Module):
    def __init__(self, prototypes, scale, adapter):
        super().__init__()
        self.register_buffer("prototypes", F.normalize(prototypes.detach(), dim=-1))
        self.register_buffer("scale", scale.detach().clone())
        self.adapter = adapter

    def logits(self, features, class_ids=None, *, enabled=True):
        prototypes = self.prototypes
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return self.adapter(features, enabled=enabled) @ prototypes.T * self.scale
