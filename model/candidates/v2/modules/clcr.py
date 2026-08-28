from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLLMComplementaryResidual(nn.Module):
    """固定GPT-5.6 SDCR残差，并学习一条独立Claude正交残差。"""

    def __init__(
        self,
        sdcr_prototypes: torch.Tensor,
        sdcr_beta: float,
        claude_prototypes: torch.Tensor,
        max_beta: float = 5.0,
    ) -> None:
        super().__init__()
        if tuple(sdcr_prototypes.shape) != (200, 768):
            raise ValueError("CLCR SDCR原型必须是[200,768]。")
        if tuple(claude_prototypes.shape) != (200, 768):
            raise ValueError("CLCR Claude原型必须是[200,768]。")
        sdcr = F.normalize(sdcr_prototypes.detach().float(), dim=-1)
        claude = F.normalize(claude_prototypes.detach().float(), dim=-1)
        self.register_buffer("sdcr_prototypes", sdcr)
        self.register_buffer("sdcr_beta", torch.tensor(float(sdcr_beta)))
        self.register_buffer("claude_prototypes", claude)
        self.register_buffer(
            "mean_cross_llm_cosine", (sdcr * claude).sum(dim=-1).mean()
        )
        self.max_beta = float(max_beta)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self) -> torch.Tensor:
        return self.max_beta * torch.tanh(self.raw_beta)

    def stats(self) -> dict[str, float]:
        return {
            "claude_beta": float(self.beta().detach()),
            "mean_cross_llm_cosine": float(self.mean_cross_llm_cosine),
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        sdcr = self.sdcr_prototypes
        claude = self.claude_prototypes
        if class_ids is not None:
            ids = class_ids.to(sdcr.device)
            sdcr = sdcr.index_select(0, ids)
            claude = claude.index_select(0, ids)
        normalized = F.normalize(images.float(), dim=-1)
        logits = parent_logits + self.sdcr_beta * (normalized @ sdcr.T)
        if not enabled:
            return logits
        return logits + self.beta() * (normalized @ claude.T)
