from __future__ import annotations

import torch
import torch.nn as nn


class GammaResidualCalibration(nn.Module):
    """在固定SEBC gamma附近学习有界残差。"""

    def __init__(self, parent_gamma: float, max_residual: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("parent_gamma", torch.tensor(float(parent_gamma)))
        self.max_residual = float(max_residual)
        self.raw_residual = nn.Parameter(torch.zeros(()))

    def residual(self) -> torch.Tensor:
        return self.max_residual * torch.tanh(self.raw_residual)

    def gamma(self) -> torch.Tensor:
        return self.parent_gamma + self.residual()

    def forward(
        self, logits: torch.Tensor, seen_mask: torch.Tensor, enabled: bool = True
    ) -> torch.Tensor:
        if not enabled:
            return logits - self.parent_gamma * seen_mask.to(logits.dtype).unsqueeze(0)
        return logits - self.gamma() * seen_mask.to(logits.dtype).unsqueeze(0)
