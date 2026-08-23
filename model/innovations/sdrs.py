from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticDisagreementResidualScaling(nn.Module):
    """按类别语义分歧调整已经成立的类名残差强度。"""

    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        class_name_prototypes: torch.Tensor,
        seen_class_ids: torch.Tensor,
        base_beta: float,
        max_delta: float = 5.0,
    ) -> None:
        super().__init__()
        parent = F.normalize(parent_prototypes.detach().float(), dim=-1)
        names = F.normalize(class_name_prototypes.detach().float(), dim=-1)
        if parent.shape != names.shape:
            raise ValueError("父原型与类名原型形状必须一致。")
        disagreement = 1.0 - (parent * names).sum(dim=-1)
        seen_values = disagreement.index_select(0, seen_class_ids.long())
        center = seen_values.mean()
        scale = seen_values.std(unbiased=False).clamp_min(1e-6)
        normalized = ((disagreement - center) / (2.0 * scale)).clamp(-1.0, 1.0)
        self.register_buffer("class_name_prototypes", names)
        self.register_buffer("normalized_disagreement", normalized)
        self.register_buffer("base_beta", torch.tensor(float(base_beta)))
        self.max_delta = float(max_delta)
        self.raw_slope = nn.Parameter(torch.zeros(()))

    def delta(self) -> torch.Tensor:
        return self.max_delta * torch.tanh(self.raw_slope)

    def class_beta(self, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        disagreement = self.normalized_disagreement
        if class_ids is not None:
            disagreement = disagreement.index_select(0, class_ids.to(disagreement.device))
        return self.base_beta + self.delta() * disagreement

    def residual_logits(
        self, images: torch.Tensor, class_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        prototypes = self.class_name_prototypes
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return F.normalize(images.float(), dim=-1) @ prototypes.T

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return parent_logits
        residual = self.residual_logits(images, class_ids)
        return parent_logits + residual * self.class_beta(class_ids).unsqueeze(0)
