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
        if int(input_dim) not in (4, 5, 8):
            raise ValueError("切空间gate输入只允许4/5维摘要或8维邻域向量。")
        if not 0.0 < float(initial_step) < float(max_step):
            raise ValueError("TST初始步长必须位于(0, max_step)。")
        self.max_step = float(max_step)
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), 16),
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


class NeighborhoodResidualGate(nn.Module):
    """在冻结4维TST gate上学习top-5邻域残差。"""

    def __init__(self, base_gate: TangentStepGate, max_delta: float = 0.1):
        super().__init__()
        if float(max_delta) <= 0.0:
            raise ValueError("NTR residual max_delta必须为正数。")
        self.base_gate = base_gate.eval()
        for parameter in self.base_gate.parameters():
            parameter.requires_grad_(False)
        self.max_delta = float(max_delta)
        self.residual = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != 8:
            raise ValueError("NTR residual gate要求8维top-5特征。")
        summary = torch.stack(
            (features[:, 0], features[:, 1], features[:, 2], features[:, 3]), dim=1
        )
        base_step = self.base_gate(summary)
        delta = self.max_delta * torch.tanh(self.residual(features[:, 3:])).squeeze(-1)
        return (base_step + delta).clamp(0.0, 1.5)


class SummaryResidualGate(nn.Module):
    """在冻结4维TST gate上学习有界双层元残差。"""

    def __init__(self, base_gate: TangentStepGate, max_delta: float = 0.1):
        super().__init__()
        self.base_gate = base_gate.eval()
        for parameter in self.base_gate.parameters():
            parameter.requires_grad_(False)
        self.max_delta = float(max_delta)
        self.residual = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != 4:
            raise ValueError("BMR residual gate要求4维摘要特征。")
        base_step = self.base_gate(features)
        delta = self.max_delta * torch.tanh(self.residual(features)).squeeze(-1)
        return (base_step + delta).clamp(0.0, 1.5)


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


def centroid_contrastive_loss(
    prototypes: torch.Tensor,
    visual_centroids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    prototypes = F.normalize(prototypes, dim=-1)
    visual_centroids = F.normalize(visual_centroids, dim=-1)
    logits = prototypes @ visual_centroids.T / float(temperature)
    targets = torch.arange(prototypes.size(0), device=prototypes.device)
    return F.cross_entropy(logits, targets)


def bidirectional_centroid_contrastive_loss(
    prototypes: torch.Tensor,
    visual_centroids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    prototypes = F.normalize(prototypes, dim=-1)
    visual_centroids = F.normalize(visual_centroids, dim=-1)
    logits = prototypes @ visual_centroids.T / float(temperature)
    targets = torch.arange(prototypes.size(0), device=prototypes.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )
