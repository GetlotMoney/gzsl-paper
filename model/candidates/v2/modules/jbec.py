from __future__ import annotations

import torch
import torch.nn as nn


class JointBidirectionalEpisodicCalibration(nn.Module):
    """在冻结VEBC解附近联合微调VPA beta与seen gamma。"""

    def __init__(self, parent_beta, parent_gamma, max_beta_residual=2.0, max_gamma_residual=0.05):
        super().__init__()
        self.register_buffer("parent_beta", torch.as_tensor(float(parent_beta)))
        self.register_buffer("parent_gamma", torch.as_tensor(float(parent_gamma)))
        self.max_beta_residual = float(max_beta_residual)
        self.max_gamma_residual = float(max_gamma_residual)
        self.raw_beta_residual = nn.Parameter(torch.zeros(()))
        self.raw_gamma_residual = nn.Parameter(torch.zeros(()))

    def beta(self):
        return self.parent_beta + self.max_beta_residual * torch.tanh(self.raw_beta_residual)

    def gamma(self):
        return self.parent_gamma + self.max_gamma_residual * torch.tanh(self.raw_gamma_residual)

    def forward(self, cra_logits, visual_logits, seen_mask, enabled=True):
        if not enabled:
            return cra_logits + self.parent_beta * visual_logits - self.parent_gamma * seen_mask.to(cra_logits.dtype).unsqueeze(0)
        return cra_logits + self.beta() * visual_logits - self.gamma() * seen_mask.to(cra_logits.dtype).unsqueeze(0)
