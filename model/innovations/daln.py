from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def prototype_density_features(prototypes: torch.Tensor) -> torch.Tensor:
    prototypes = F.normalize(prototypes.detach(), dim=-1)
    similarity = prototypes @ prototypes.T
    similarity.fill_diagonal_(-1.0)
    top10 = similarity.topk(10, dim=1).values
    return torch.stack(
        (
            top10[:, 0],
            top10[:, :5].mean(dim=1),
            top10.mean(dim=1),
            top10.std(dim=1, unbiased=False),
        ),
        dim=1,
    )


class DensityAwareLogitNormalizer(nn.Module):
    """根据最终原型密度预测零均值类别log尺度。"""

    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        density_features: torch.Tensor,
        seenclasses: torch.Tensor,
        scale: torch.Tensor,
        *,
        max_log_scale: float = 0.1,
    ):
        super().__init__()
        self.register_buffer(
            "parent_prototypes", F.normalize(parent_prototypes.detach(), dim=-1)
        )
        features = density_features.detach().float()
        self.register_buffer("density_features", features)
        self.register_buffer("feature_mean", features.mean(dim=0, keepdim=True))
        self.register_buffer(
            "feature_std", features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        )
        self.register_buffer("seenclasses", seenclasses.detach().cpu().long())
        self.register_buffer("_scale", scale.detach().clone())
        self.max_log_scale = float(max_log_scale)
        self.gate = nn.Sequential(
            nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def class_confidence(self):
        features = (self.density_features - self.feature_mean) / self.feature_std
        log_scale = self.max_log_scale * torch.tanh(self.gate(features)).squeeze(-1)
        seen = self.seenclasses.to(log_scale.device)
        log_scale = log_scale - log_scale.index_select(0, seen).mean()
        return torch.exp(log_scale)

    def prototypes(self, *, enabled=True):
        if not enabled:
            return self.parent_prototypes
        return self.parent_prototypes * self.class_confidence().unsqueeze(-1)

    def scale(self):
        return self._scale

    def logits(self, image_features, class_ids=None):
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return F.normalize(image_features.float(), dim=-1) @ prototypes.T * self.scale()
