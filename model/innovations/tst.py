from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TangentStepGate(nn.Module):
    """从类别几何量预测球面切空间迁移步长。"""

    def __init__(
        self,
        input_dim: int = 4,
        max_step: float = 1.5,
        initial_step: float = 0.1,
    ):
        super().__init__()
        if int(input_dim) != 4:
            raise ValueError("TST首次TRY固定4维几何输入。")
        if not 0.0 < float(initial_step) < float(max_step):
            raise ValueError("TST初始步长必须位于(0, max_step)。")
        self.max_step = float(max_step)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        initial_ratio = float(initial_step) / self.max_step
        nn.init.constant_(
            self.network[-1].bias,
            math.log(initial_ratio / (1.0 - initial_ratio)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_step * torch.sigmoid(self.network(features)).squeeze(-1)


def tangent_transport(
    base: torch.Tensor, value: torch.Tensor, step: torch.Tensor
) -> torch.Tensor:
    base = F.normalize(base, dim=-1)
    value = F.normalize(value, dim=-1)
    tangent = value - (value * base).sum(dim=-1, keepdim=True) * base
    return F.normalize(base + step.unsqueeze(-1) * tangent, dim=-1)


def centroid_alignment_loss(
    prototypes: torch.Tensor, visual_centroids: torch.Tensor
) -> torch.Tensor:
    prototypes = F.normalize(prototypes, dim=-1)
    visual_centroids = F.normalize(visual_centroids, dim=-1)
    return 1.0 - (prototypes * visual_centroids).sum(dim=-1).mean()
