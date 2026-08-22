from __future__ import annotations

import torch
import torch.nn as nn


class NormalizedGeometricVisualFusion(nn.Module):
    """在加性视觉logit与单位球面原型融合之间学习插值。"""

    def __init__(self):
        super().__init__()
        self.raw_eta = nn.Parameter(torch.zeros(()))

    def eta(self):
        return torch.tanh(self.raw_eta)

    def forward(self, additive_logits, normalized_logits, enabled=True):
        if not enabled:
            return additive_logits
        return additive_logits + self.eta() * (normalized_logits - additive_logits)
