from __future__ import annotations

import torch
import torch.nn as nn


class EpisodicPriorCalibration(nn.Module):
    """从pseudo-unseen episode学习有界类别竞争边际。"""

    def __init__(self, max_margin: float = 0.5):
        super().__init__()
        if float(max_margin) <= 0.0:
            raise ValueError("EPC max_margin必须为正数。")
        self.max_margin = float(max_margin)
        self.raw_margin = nn.Parameter(torch.zeros(()))

    def margin(self) -> torch.Tensor:
        return self.max_margin * torch.tanh(self.raw_margin)

    def forward(
        self,
        logits: torch.Tensor,
        competition_classes: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.isin(
            competition_classes.to(logits.device), target_classes.to(logits.device)
        )
        return logits + self.margin() * mask.to(logits.dtype).unsqueeze(0)
