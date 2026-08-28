from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodContrastiveSemanticResidual(nn.Module):
    """在固定SDCR原型旁增加与近邻语义正交的类别判别方向。"""

    def __init__(
        self,
        base_prototypes: torch.Tensor,
        fixed_beta: float,
        neighbor_k: int = 5,
        max_gamma: float = 5.0,
    ) -> None:
        super().__init__()
        if tuple(base_prototypes.shape) != (200, 768):
            raise ValueError("NCSR基础原型必须是[200,768]。")
        if int(neighbor_k) != 5:
            raise ValueError("NCSR首次实验固定5个语义近邻。")
        base = F.normalize(base_prototypes.detach().float(), dim=-1)
        similarity = base @ base.T
        similarity.fill_diagonal_(-float("inf"))
        neighbor_indices = similarity.topk(int(neighbor_k), dim=-1).indices
        neighbor_prototypes = base.index_select(
            0, neighbor_indices.reshape(-1)
        ).reshape(200, int(neighbor_k), 768)
        neighbor_mean = F.normalize(neighbor_prototypes.mean(dim=1), dim=-1)
        difference = base - neighbor_mean
        contrastive = difference - (difference * base).sum(
            dim=-1, keepdim=True
        ) * base
        contrastive = F.normalize(contrastive, dim=-1)

        self.register_buffer("base_prototypes", base)
        self.register_buffer("contrastive_prototypes", contrastive)
        self.register_buffer("neighbor_indices", neighbor_indices)
        self.register_buffer(
            "mean_neighbor_similarity",
            similarity.gather(1, neighbor_indices).mean(),
        )
        self.register_buffer("fixed_beta", torch.tensor(float(fixed_beta)))
        self.max_gamma = float(max_gamma)
        self.raw_gamma = nn.Parameter(torch.zeros(()))

    def gamma(self) -> torch.Tensor:
        return self.max_gamma * torch.tanh(self.raw_gamma)

    def prototype_pair(
        self, class_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_prototypes
        contrastive = self.contrastive_prototypes
        if class_ids is not None:
            ids = class_ids.to(base.device)
            base = base.index_select(0, ids)
            contrastive = contrastive.index_select(0, ids)
        return base, contrastive

    def stats(self) -> dict[str, float]:
        orthogonality = (
            self.base_prototypes * self.contrastive_prototypes
        ).sum(dim=-1).abs()
        return {
            "gamma": float(self.gamma().detach()),
            "mean_neighbor_similarity": float(self.mean_neighbor_similarity),
            "max_base_contrastive_abs_cosine": float(orthogonality.max()),
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        base, contrastive = self.prototype_pair(class_ids)
        normalized = F.normalize(images.float(), dim=-1)
        logits = parent_logits + self.fixed_beta * (normalized @ base.T)
        if not enabled:
            return logits
        return logits + self.gamma() * (normalized @ contrastive.T)
