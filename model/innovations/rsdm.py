from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.sdm import SymmetricDiagonalMetric


class ResidualSymmetricDiagonalMetric(nn.Module):
    """只对固定SDCR残差分支同步变换图像与文本原型。"""

    def __init__(
        self,
        residual_prototypes: torch.Tensor,
        fixed_beta: float,
        max_log_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if tuple(residual_prototypes.shape) != (200, 768):
            raise ValueError("RSDM残差原型必须是[200,768]。")
        self.register_buffer(
            "residual_prototypes",
            residual_prototypes.detach().float(),
        )
        self.register_buffer("fixed_beta", torch.tensor(float(fixed_beta)))
        self.metric = SymmetricDiagonalMetric(
            dimension=768, max_log_weight=float(max_log_weight)
        )

    def stats(self) -> dict[str, float]:
        return self.metric.stats()

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        prototypes = self.residual_prototypes
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        if enabled:
            residual_logits = self.metric.logits(
                images.float(), prototypes, self.fixed_beta
            )
        else:
            residual_logits = (
                F.normalize(images.float(), dim=-1)
                @ F.normalize(prototypes, dim=-1).T
                * self.fixed_beta
            )
        return parent_logits + residual_logits
