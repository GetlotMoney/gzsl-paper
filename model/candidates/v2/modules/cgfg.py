from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGaussianFeatureGenerator(nn.Module):
    """从语义原型预测有界视觉均值。"""

    def __init__(self, semantic_prototypes, hidden_dim=128, max_residual_norm=0.2):
        super().__init__()
        self.register_buffer(
            "semantic_prototypes", F.normalize(semantic_prototypes.detach(), dim=-1)
        )
        self.max_residual_norm = float(max_residual_norm)
        self.network = nn.Sequential(
            nn.Linear(768, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), 768)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def residual(self):
        raw = self.network(self.semantic_prototypes)
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return raw * (self.max_residual_norm / norm).clamp(max=1.0)

    def means(self):
        return F.normalize(self.semantic_prototypes + self.residual(), dim=-1)


class CosineLinearClassifier(nn.Module):
    def __init__(self, initial_prototypes, scale):
        super().__init__()
        self.weight = nn.Parameter(F.normalize(initial_prototypes.detach(), dim=-1).clone())
        self.register_buffer("scale", scale.detach().clone())

    def logits(self, features, class_ids=None):
        weight = F.normalize(self.weight, dim=-1)
        if class_ids is not None:
            weight = weight.index_select(0, class_ids.to(weight.device))
        return F.normalize(features.float(), dim=-1) @ weight.T * self.scale
