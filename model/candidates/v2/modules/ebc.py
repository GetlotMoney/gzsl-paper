from __future__ import annotations

import torch
import torch.nn as nn


class EpisodicBiasCalibration(nn.Module):
    """训练式全局seen logit扣减。"""

    def __init__(self, max_gamma=0.2):
        super().__init__()
        self.max_gamma = float(max_gamma)
        self.raw_gamma = nn.Parameter(torch.zeros(()))

    def gamma(self):
        return self.max_gamma * torch.tanh(self.raw_gamma)

    def forward(self, logits, seen_mask, enabled=True):
        if not enabled:
            return logits
        return logits - self.gamma() * seen_mask.to(logits.dtype).unsqueeze(0)
