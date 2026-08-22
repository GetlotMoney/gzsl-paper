from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiRolePrototypeClassifier(nn.Module):
    """在TST父logit上增加三角色多原型匹配证据。"""

    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        role_prototypes: torch.Tensor,
        scale: torch.Tensor,
        *,
        role_temperature: float = 0.05,
        max_strength: float = 0.5,
        initial_strength: float = 0.05,
        learn_role_bias: bool = False,
        max_role_bias: float = 0.05,
    ):
        super().__init__()
        if tuple(parent_prototypes.shape) != (200, 768):
            raise ValueError("MPR父原型必须是[200,768]。")
        if tuple(role_prototypes.shape) != (200, 3, 768):
            raise ValueError("MPR角色原型必须是[200,3,768]。")
        self.register_buffer("parent_prototypes", F.normalize(parent_prototypes.detach(), dim=-1))
        self.register_buffer("role_prototypes", F.normalize(role_prototypes.detach(), dim=-1))
        self.register_buffer("_scale", scale.detach().clone())
        self.role_temperature = float(role_temperature)
        self.max_strength = float(max_strength)
        self.learn_role_bias = bool(learn_role_bias)
        self.max_role_bias = float(max_role_bias)
        self.raw_role_bias = nn.Parameter(
            torch.zeros(3), requires_grad=self.learn_role_bias
        )
        ratio = float(initial_strength) / self.max_strength
        self.raw_strength = nn.Parameter(torch.tensor(math.log(ratio / (1.0-ratio))))

    def strength(self):
        return self.max_strength * torch.sigmoid(self.raw_strength)

    def scale(self):
        return self._scale

    def role_bias(self):
        bias = self.max_role_bias * torch.tanh(self.raw_role_bias)
        return bias - bias.mean()

    def logits(self, image_features: torch.Tensor, class_ids=None, *, enabled=True):
        images = F.normalize(image_features.float(), dim=-1)
        parent = self.parent_prototypes
        roles = self.role_prototypes
        if class_ids is not None:
            ids = class_ids.to(parent.device); parent = parent.index_select(0, ids); roles = roles.index_select(0, ids)
        parent_score = images @ parent.T
        if not enabled:
            return parent_score * self.scale()
        role_score = torch.einsum("bd,crd->bcr", images, roles)
        if self.learn_role_bias:
            role_score = role_score + self.role_bias().view(1, 1, 3)
        soft_best = self.role_temperature * torch.logsumexp(
            role_score / self.role_temperature, dim=-1
        )
        evidence = soft_best - role_score.mean(dim=-1)
        return (parent_score + self.strength() * evidence) * self.scale()
