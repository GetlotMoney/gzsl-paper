from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCrossLLMMixture(nn.Module):
    """在两个已训练LLM残差端点之间学习全局混合比例。"""

    def __init__(
        self,
        claude_prototypes: torch.Tensor,
        merge_prototypes: torch.Tensor,
        claude_beta: float,
        merge_beta: float,
    ) -> None:
        super().__init__()
        if tuple(claude_prototypes.shape) != (200, 768):
            raise ValueError("Claude原型必须是[200,768]。")
        if tuple(merge_prototypes.shape) != (200, 768):
            raise ValueError("merge原型必须是[200,768]。")
        self.register_buffer(
            "claude_prototypes", F.normalize(claude_prototypes.detach().float(), dim=-1)
        )
        self.register_buffer(
            "merge_prototypes", F.normalize(merge_prototypes.detach().float(), dim=-1)
        )
        self.register_buffer("claude_beta", torch.tensor(float(claude_beta)))
        self.register_buffer("merge_beta", torch.tensor(float(merge_beta)))
        self.raw_mix = nn.Parameter(torch.zeros(()))

    def claude_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_mix)

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return parent_logits
        claude = self.claude_prototypes
        merge = self.merge_prototypes
        if class_ids is not None:
            ids = class_ids.to(claude.device)
            claude = claude.index_select(0, ids)
            merge = merge.index_select(0, ids)
        normalized = F.normalize(images.float(), dim=-1)
        claude_logits = normalized @ claude.T
        merge_logits = normalized @ merge.T
        weight = self.claude_weight()
        return (
            parent_logits
            + weight * self.claude_beta * claude_logits
            + (1.0 - weight) * self.merge_beta * merge_logits
        )
