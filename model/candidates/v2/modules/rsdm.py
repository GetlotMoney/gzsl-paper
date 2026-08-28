from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.candidates.v2.modules.sdm import SymmetricDiagonalMetric


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


class FullSemanticSymmetricDiagonalMetric(nn.Module):
    """对TG主原型、SDRS类名和SDCR残差三条语义分支共享同一度量。"""

    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        parent_scale: float,
        class_name_prototypes: torch.Tensor,
        class_beta: torch.Tensor,
        residual_prototypes: torch.Tensor,
        residual_beta: float,
        seen_class_ids: torch.Tensor,
        seen_gamma: float,
        max_log_weight: float = 0.1,
    ) -> None:
        super().__init__()
        for name, tensor in (
            ("TG主原型", parent_prototypes),
            ("类名原型", class_name_prototypes),
            ("SDCR残差原型", residual_prototypes),
        ):
            if tuple(tensor.shape) != (200, 768):
                raise ValueError(f"FSDM{name}必须是[200,768]。")
        if tuple(class_beta.shape) != (200,):
            raise ValueError("FSDM类名残差强度必须是[200]。")
        seen_mask = torch.zeros(200, dtype=torch.bool, device=parent_prototypes.device)
        seen_mask[seen_class_ids.to(seen_mask.device).long()] = True
        self.register_buffer("parent_prototypes", parent_prototypes.detach().float())
        self.register_buffer("parent_scale", torch.tensor(float(parent_scale)))
        self.register_buffer(
            "class_name_prototypes", class_name_prototypes.detach().float()
        )
        self.register_buffer("class_beta", class_beta.detach().float())
        self.register_buffer(
            "residual_prototypes", residual_prototypes.detach().float()
        )
        self.register_buffer("residual_beta", torch.tensor(float(residual_beta)))
        self.register_buffer("seen_mask", seen_mask)
        self.register_buffer("seen_gamma", torch.tensor(float(seen_gamma)))
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
        ids = (
            torch.arange(200, device=images.device)
            if class_ids is None
            else class_ids.to(images.device)
        )
        residual_prototypes = self.residual_prototypes.index_select(0, ids)
        if not enabled:
            residual = (
                F.normalize(images.float(), dim=-1)
                @ F.normalize(residual_prototypes, dim=-1).T
                * self.residual_beta
            )
            return parent_logits + residual

        parent_prototypes = self.parent_prototypes.index_select(0, ids)
        names = self.class_name_prototypes.index_select(0, ids)
        class_beta = self.class_beta.index_select(0, ids)
        seen_mask = self.seen_mask.index_select(0, ids)
        base = self.metric.logits(images.float(), parent_prototypes, self.parent_scale)
        name_residual = self.metric.logits(
            images.float(), names, torch.tensor(1.0, device=images.device)
        ) * class_beta.unsqueeze(0)
        sdcr_residual = self.metric.logits(
            images.float(), residual_prototypes, self.residual_beta
        )
        return (
            base
            + name_residual
            + sdcr_residual
            - self.seen_gamma * seen_mask.to(base.dtype).unsqueeze(0)
        )
