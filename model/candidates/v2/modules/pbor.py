from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialBiOrthogonalResidual(nn.Module):
    """固定OCLR beta，只学习TG父方向的有界部分去除系数。"""

    def __init__(
        self,
        claude_prototypes: torch.Tensor,
        class_name_prototypes: torch.Tensor,
        parent_prototypes: torch.Tensor,
        fixed_beta: float,
        max_parent_projection: float = 1.0,
    ) -> None:
        super().__init__()
        if not (
            claude_prototypes.shape
            == class_name_prototypes.shape
            == parent_prototypes.shape
            == (200, 768)
        ):
            raise ValueError("PBOR三个原型输入必须是[200,768]。")
        source = F.normalize(claude_prototypes.detach().float(), dim=-1)
        q1 = F.normalize(class_name_prototypes.detach().float(), dim=-1)
        parent = parent_prototypes.detach().float()
        q2 = F.normalize(parent - (parent * q1).sum(-1, keepdim=True) * q1, dim=-1)
        base = source - (source * q1).sum(-1, keepdim=True) * q1
        parent_component = (source * q2).sum(-1, keepdim=True) * q2
        self.register_buffer("base_residual", base)
        self.register_buffer("parent_component", parent_component)
        self.register_buffer("fixed_beta", torch.tensor(float(fixed_beta)))
        self.max_parent_projection = float(max_parent_projection)
        self.raw_parent_projection = nn.Parameter(torch.zeros(()))

    def parent_projection(self) -> torch.Tensor:
        return self.max_parent_projection * torch.tanh(self.raw_parent_projection)

    def prototypes(self, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        residual = F.normalize(
            self.base_residual
            - self.parent_projection() * self.parent_component,
            dim=-1,
        )
        if class_ids is not None:
            residual = residual.index_select(0, class_ids.to(residual.device))
        return residual

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return parent_logits
        residual_logits = F.normalize(images.float(), dim=-1) @ self.prototypes(class_ids).T
        return parent_logits + self.fixed_beta * residual_logits
