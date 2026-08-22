from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def competition_features(
    logits: torch.Tensor, seen_mask: torch.Tensor, unseen_mask: torch.Tensor
) -> torch.Tensor:
    seen_logits = logits[:, seen_mask]
    unseen_logits = logits[:, unseen_mask]
    seen_prob = F.softmax(seen_logits, dim=-1)
    unseen_prob = F.softmax(unseen_logits, dim=-1)
    seen_entropy = -(seen_prob * seen_prob.clamp_min(1e-12).log()).sum(dim=-1)
    unseen_entropy = -(unseen_prob * unseen_prob.clamp_min(1e-12).log()).sum(dim=-1)
    seen_max = seen_logits.max(dim=-1).values
    unseen_max = unseen_logits.max(dim=-1).values
    return torch.stack(
        (
            seen_max,
            unseen_max,
            torch.logsumexp(seen_logits, dim=-1),
            torch.logsumexp(unseen_logits, dim=-1),
            seen_entropy,
            unseen_entropy,
            seen_max - unseen_max,
        ),
        dim=1,
    )


class EpisodicDomainCompetition(nn.Module):
    """根据每张图的seen/unseen竞争状态预测有界logit校正。"""

    def __init__(self, max_correction: float = 0.2):
        super().__init__()
        self.max_correction = float(max_correction)
        self.network = nn.Sequential(
            nn.Linear(7, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def correction(self, logits, seen_mask, unseen_mask):
        features = competition_features(logits, seen_mask, unseen_mask)
        return self.max_correction * torch.tanh(self.network(features)).squeeze(-1)

    def forward(self, logits, seen_mask, unseen_mask, *, enabled=True):
        if not enabled:
            return logits
        correction = self.correction(logits, seen_mask, unseen_mask)
        return logits + correction.unsqueeze(1) * unseen_mask.to(logits.dtype).unsqueeze(0)
