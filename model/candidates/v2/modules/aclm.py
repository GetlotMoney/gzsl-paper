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


class ClassAdaptiveCrossLLMMixture(nn.Module):
    """按每类Claude/merge语义一致度学习类别混合权重。"""

    def __init__(
        self,
        claude_prototypes: torch.Tensor,
        merge_prototypes: torch.Tensor,
        seen_class_ids: torch.Tensor,
        claude_beta: float,
        merge_beta: float,
    ) -> None:
        super().__init__()
        claude = F.normalize(claude_prototypes.detach().float(), dim=-1)
        merge = F.normalize(merge_prototypes.detach().float(), dim=-1)
        agreement = (claude * merge).sum(dim=-1)
        seen_values = agreement.index_select(0, seen_class_ids.to(agreement.device))
        normalized = (
            (agreement - seen_values.mean())
            / seen_values.std(unbiased=False).clamp_min(1e-6)
        ).clamp(-2.0, 2.0)
        self.register_buffer("claude_prototypes", claude)
        self.register_buffer("merge_prototypes", merge)
        self.register_buffer("normalized_agreement", normalized)
        self.register_buffer("claude_beta", torch.tensor(float(claude_beta)))
        self.register_buffer("merge_beta", torch.tensor(float(merge_beta)))
        self.raw_bias = nn.Parameter(torch.zeros(()))
        self.raw_slope = nn.Parameter(torch.zeros(()))

    def claude_weight(self, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        agreement = self.normalized_agreement
        if class_ids is not None:
            agreement = agreement.index_select(0, class_ids.to(agreement.device))
        return torch.sigmoid(self.raw_bias + self.raw_slope * agreement)

    def weight_stats(self) -> dict[str, float]:
        weights = self.claude_weight().detach()
        return {
            "mean": float(weights.mean()),
            "std": float(weights.std(unbiased=False)),
            "min": float(weights.min()),
            "max": float(weights.max()),
        }

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
        weight = self.claude_weight(class_ids).unsqueeze(0)
        return (
            parent_logits
            + weight * self.claude_beta * claude_logits
            + (1.0 - weight) * self.merge_beta * merge_logits
        )
