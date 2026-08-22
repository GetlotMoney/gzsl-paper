from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def tangent_direction_basis(
    base_prototypes: torch.Tensor,
    value_prototypes: torch.Tensor,
    role_prototypes: torch.Tensor,
) -> torch.Tensor:
    base = F.normalize(base_prototypes, dim=-1)
    candidates = torch.cat(
        (value_prototypes.unsqueeze(1), role_prototypes), dim=1
    )
    candidates = F.normalize(candidates, dim=-1)
    tangent = candidates - (
        candidates * base.unsqueeze(1)
    ).sum(dim=-1, keepdim=True) * base.unsqueeze(1)
    return F.normalize(tangent, dim=-1)


class ClassConditionedGeometricGenerator(nn.Module):
    """在四个文本切向方向内生成类别条件残差。"""

    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        direction_basis: torch.Tensor,
        class_features: torch.Tensor,
        target_classes: torch.Tensor,
        scale: torch.Tensor,
        *,
        hidden_dim: int = 32,
        max_magnitude: float = 0.1,
        initial_magnitude: float = 0.02,
    ):
        super().__init__()
        if tuple(direction_basis.shape) != (200, 4, 768):
            raise ValueError("CCGR方向基必须是[200,4,768]。")
        if class_features.ndim != 2 or class_features.shape[0] != 200:
            raise ValueError("CCGR类别特征必须是[200,F]。")
        feature_dim = int(class_features.shape[1])
        if feature_dim not in (4, 8):
            raise ValueError("CCGR类别特征目前只支持4维均值或8维top-5向量。")
        self.register_buffer(
            "parent_prototypes", F.normalize(parent_prototypes.detach(), dim=-1)
        )
        self.register_buffer("direction_basis", direction_basis.detach())
        frozen_features = class_features.detach().float()
        self.register_buffer("class_features", frozen_features)
        self.register_buffer("feature_mean", frozen_features.mean(dim=0, keepdim=True))
        self.register_buffer(
            "feature_std",
            frozen_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6),
        )
        self.register_buffer("target_classes", target_classes.detach().cpu().long())
        self.register_buffer("_scale", scale.detach().clone())
        self.max_magnitude = float(max_magnitude)
        self.trunk = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.weight_head = nn.Linear(hidden_dim, 4)
        self.magnitude_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        nn.init.zeros_(self.magnitude_head.weight)
        ratio = float(initial_magnitude) / self.max_magnitude
        nn.init.constant_(
            self.magnitude_head.bias, math.log(ratio / (1.0 - ratio))
        )

    def _generate(self, parent_prototypes, direction_basis, class_features):
        parent_prototypes = F.normalize(parent_prototypes, dim=-1)
        features = (class_features - self.feature_mean) / self.feature_std
        hidden = self.trunk(features)
        weights = F.softmax(self.weight_head(hidden), dim=-1)
        magnitude = self.max_magnitude * torch.sigmoid(
            self.magnitude_head(hidden)
        ).squeeze(-1)
        direction = F.normalize(
            (weights.unsqueeze(-1) * direction_basis).sum(dim=1), dim=-1
        )
        direction = direction - (
            direction * parent_prototypes
        ).sum(dim=-1, keepdim=True) * parent_prototypes
        direction = F.normalize(direction, dim=-1)
        return F.normalize(
            parent_prototypes + magnitude.unsqueeze(-1) * direction, dim=-1
        )

    def magnitude_values(self, class_features=None):
        if class_features is None:
            class_features = self.class_features
        features = (class_features - self.feature_mean) / self.feature_std
        return self.max_magnitude * torch.sigmoid(
            self.magnitude_head(self.trunk(features))
        ).squeeze(-1)

    def generated_all(self):
        return self._generate(
            self.parent_prototypes, self.direction_basis, self.class_features
        )

    def generate_external(self, parent_prototypes, direction_basis, class_features):
        return self._generate(parent_prototypes, direction_basis, class_features)

    def prototypes(self, *, enabled=True):
        if not enabled:
            return self.parent_prototypes
        generated = self.generated_all()
        result = self.parent_prototypes.clone()
        target = self.target_classes.to(result.device)
        result[target] = generated.index_select(0, target)
        return result

    def scale(self):
        return self._scale

    def magnitude_stats(self):
        magnitude = self.magnitude_values()
        return {"mean": float(magnitude.mean()), "max": float(magnitude.max())}
