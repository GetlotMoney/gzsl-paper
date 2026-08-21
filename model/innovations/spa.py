from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.elpt import topology_loss


class SeenPrototypeAnchor(nn.Module):
    """用seen训练图像中心对TST父原型做有界视觉锚定。"""

    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        seenclasses: torch.Tensor,
        visual_centroids: torch.Tensor,
        scale: torch.Tensor,
        *,
        max_strength: float = 0.1,
        initial_strength: float = 0.01,
    ):
        super().__init__()
        if tuple(parent_prototypes.shape) != (200, 768):
            raise ValueError("SPA父原型必须是[200,768]。")
        if tuple(visual_centroids.shape) != (150, 768):
            raise ValueError("SPA视觉中心必须是[150,768]。")
        if not 0.0 < float(initial_strength) < float(max_strength):
            raise ValueError("SPA初始强度必须位于(0,max_strength)。")
        self.register_buffer(
            "parent_prototypes", F.normalize(parent_prototypes.detach(), dim=-1)
        )
        self.register_buffer("seenclasses", seenclasses.detach().cpu().long())
        self.register_buffer(
            "visual_centroids", F.normalize(visual_centroids.detach(), dim=-1)
        )
        self.register_buffer("_scale", scale.detach().clone())
        self.max_strength = float(max_strength)
        ratio = float(initial_strength) / self.max_strength
        self.raw_strength = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    def strength(self) -> torch.Tensor:
        return self.max_strength * torch.sigmoid(self.raw_strength)

    def prototypes(self, *, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return self.parent_prototypes
        strength = self.strength()
        result = self.parent_prototypes.clone()
        seen = self.seenclasses.to(result.device)
        parent_seen = result.index_select(0, seen)
        result[seen] = F.normalize(
            (1.0 - strength) * parent_seen + strength * self.visual_centroids,
            dim=-1,
        )
        return result

    def scale(self) -> torch.Tensor:
        return self._scale

    def topology_loss(self) -> torch.Tensor:
        return topology_loss(self.parent_prototypes, self.prototypes())

    def logits(self, image_features: torch.Tensor, class_ids=None) -> torch.Tensor:
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return F.normalize(image_features.float(), dim=-1) @ prototypes.T * self.scale()
