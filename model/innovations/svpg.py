from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.elpt import topology_loss


class SemanticVisualPrototypeGenerator(nn.Module):
    """共享语义到视觉残差映射，零初始化时严格等价于父原型。"""

    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        scale: torch.Tensor,
        *,
        hidden_dim: int = 128,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if tuple(parent_prototypes.shape) != (200, 768):
            raise ValueError("SVPG父原型必须是[200,768]。")
        self.register_buffer(
            "parent_prototypes", F.normalize(parent_prototypes.detach(), dim=-1)
        )
        self.register_buffer("_scale", scale.detach().clone())
        self.residual_scale = float(residual_scale)
        self.adapter = nn.Sequential(
            nn.Linear(768, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 768),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def prototypes(self, *, enabled=True):
        if not enabled:
            return self.parent_prototypes
        residual = self.adapter(self.parent_prototypes)
        return F.normalize(
            self.parent_prototypes + self.residual_scale * residual, dim=-1
        )

    def scale(self):
        return self._scale

    def topology_loss(self):
        return topology_loss(self.parent_prototypes, self.prototypes())

    def logits(self, image_features: torch.Tensor, class_ids=None):
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return F.normalize(image_features.float(), dim=-1) @ prototypes.T * self.scale()

    def residual_stats(self):
        residual = self.residual_scale * self.adapter(self.parent_prototypes)
        norm = residual.norm(dim=-1)
        return {"mean": float(norm.mean()), "max": float(norm.max())}
