from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLLMLocalEvidenceComposition(nn.Module):
    """固定Claude全局与CCPE局部证据，只协调局部分支比例。"""

    def __init__(
        self,
        claude_prototypes: torch.Tensor,
        claude_beta: float,
        patch_beta: float,
        max_patch_scale_residual: float = 0.25,
    ) -> None:
        super().__init__()
        if tuple(claude_prototypes.shape) != (200, 768):
            raise ValueError("Claude原型必须是[200,768]。")
        self.register_buffer(
            "claude_prototypes",
            F.normalize(claude_prototypes.detach().float(), dim=-1),
        )
        self.register_buffer("claude_beta", torch.tensor(float(claude_beta)))
        self.register_buffer("patch_beta", torch.tensor(float(patch_beta)))
        self.max_patch_scale_residual = float(max_patch_scale_residual)
        self.raw_patch_scale = nn.Parameter(torch.zeros(()))

    def patch_scale(self) -> torch.Tensor:
        return 1.0 + self.max_patch_scale_residual * torch.tanh(self.raw_patch_scale)

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return parent_logits
        prototypes = self.claude_prototypes
        local = patch_scores
        if class_ids is not None:
            ids = class_ids.to(prototypes.device)
            prototypes = prototypes.index_select(0, ids)
            if local.shape[1] != parent_logits.shape[1]:
                local = local.index_select(1, ids.to(local.device))
        if local.shape != parent_logits.shape:
            raise ValueError("局部patch分数与父logits形状不一致。")
        claude_logits = F.normalize(images.float(), dim=-1) @ prototypes.T
        return (
            parent_logits
            + self.claude_beta * claude_logits
            + self.patch_scale() * self.patch_beta * local
        )
