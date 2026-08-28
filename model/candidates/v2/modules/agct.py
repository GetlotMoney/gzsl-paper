from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmbiguityGatedCrossLLMTieBreaker(nn.Module):
    """仅对top2同族低margin样本使用Claude证据做二选一。"""

    def __init__(
        self,
        sdcr_prototypes: torch.Tensor,
        sdcr_beta: float,
        claude_prototypes: torch.Tensor,
        group_ids: torch.Tensor,
        margin_threshold: float,
        margin_temperature: float = 0.1,
        max_beta: float = 5.0,
        consensus_only: bool = False,
    ) -> None:
        super().__init__()
        if tuple(sdcr_prototypes.shape) != (200, 768):
            raise ValueError("AGCT SDCR原型必须是[200,768]。")
        if tuple(claude_prototypes.shape) != (200, 768):
            raise ValueError("AGCT Claude原型必须是[200,768]。")
        if tuple(group_ids.shape) != (200,):
            raise ValueError("AGCT group_ids必须是[200]。")
        if float(margin_threshold) <= 0 or float(margin_temperature) <= 0:
            raise ValueError("AGCT margin阈值和温度必须为正。")
        self.register_buffer(
            "sdcr_prototypes",
            F.normalize(sdcr_prototypes.detach().float(), dim=-1),
        )
        self.register_buffer("sdcr_beta", torch.tensor(float(sdcr_beta)))
        self.register_buffer(
            "claude_prototypes",
            F.normalize(claude_prototypes.detach().float(), dim=-1),
        )
        self.register_buffer("group_ids", group_ids.detach().long())
        self.register_buffer(
            "margin_threshold", torch.tensor(float(margin_threshold))
        )
        self.margin_temperature = float(margin_temperature)
        self.max_beta = float(max_beta)
        self.consensus_only = bool(consensus_only)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self) -> torch.Tensor:
        return self.max_beta * torch.tanh(self.raw_beta)

    def gate_values(
        self,
        logits: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = (
            torch.arange(200, device=logits.device)
            if class_ids is None
            else class_ids.to(logits.device)
        )
        top = logits.detach().topk(2, dim=1)
        margin = top.values[:, 0] - top.values[:, 1]
        global_top_ids = ids.index_select(0, top.indices.reshape(-1)).reshape_as(
            top.indices
        )
        groups = self.group_ids.index_select(
            0, global_top_ids.reshape(-1).to(self.group_ids.device)
        ).reshape_as(global_top_ids)
        same_group = groups[:, 0].eq(groups[:, 1]) & groups[:, 0].ge(0)
        soft_gate = torch.sigmoid(
            (self.margin_threshold - margin) / self.margin_temperature
        )
        gate = soft_gate * same_group.to(soft_gate.dtype)
        if self.consensus_only:
            if images is None:
                raise ValueError("CCTB计算共识门控时必须提供图像特征。")
            claude = self.claude_prototypes.index_select(0, ids)
            claude_logits = F.normalize(images.float(), dim=-1) @ claude.T
            claude_top_scores = claude_logits.gather(1, top.indices)
            agreement = claude_top_scores[:, 0] >= claude_top_scores[:, 1]
            gate = gate * agreement.to(gate.dtype)
        return gate, same_group, top.indices

    def stats(self) -> dict[str, float]:
        return {
            "tie_beta": float(self.beta().detach()),
            "margin_threshold": float(self.margin_threshold),
            "margin_temperature": self.margin_temperature,
            "consensus_only": self.consensus_only,
        }

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
        sdcr = self.sdcr_prototypes.index_select(0, ids)
        normalized = F.normalize(images.float(), dim=-1)
        logits = parent_logits + self.sdcr_beta * (normalized @ sdcr.T)
        if not enabled:
            return logits
        gate, _, top_positions = self.gate_values(logits, ids, images)
        claude = self.claude_prototypes.index_select(0, ids)
        claude_logits = normalized @ claude.T
        top_scores = claude_logits.gather(1, top_positions)
        centered = top_scores - top_scores.mean(dim=1, keepdim=True)
        correction = torch.zeros_like(logits)
        correction.scatter_(1, top_positions, centered)
        return logits + self.beta() * gate.unsqueeze(1) * correction


class MultiSourceAmbiguityGatedTieBreaker(
    AmbiguityGatedCrossLLMTieBreaker
):
    """在同一歧义gate内联合学习Claude与merge两条top2证据。"""

    def __init__(
        self,
        sdcr_prototypes: torch.Tensor,
        sdcr_beta: float,
        source_prototypes: torch.Tensor,
        group_ids: torch.Tensor,
        margin_threshold: float,
        margin_temperature: float = 0.1,
        max_beta: float = 5.0,
    ) -> None:
        if tuple(source_prototypes.shape) != (2, 200, 768):
            raise ValueError("MAGT文本源必须是[2,200,768]。")
        super().__init__(
            sdcr_prototypes,
            sdcr_beta,
            source_prototypes[0],
            group_ids,
            margin_threshold,
            margin_temperature,
            max_beta,
            consensus_only=False,
        )
        del self.raw_beta
        self.register_buffer(
            "source_prototypes",
            F.normalize(source_prototypes.detach().float(), dim=-1),
        )
        self.raw_betas = nn.Parameter(torch.zeros(2))
        self.register_buffer(
            "source_cosine",
            (
                self.source_prototypes[0] * self.source_prototypes[1]
            ).sum(dim=-1).mean(),
        )

    def betas(self) -> torch.Tensor:
        return self.max_beta * torch.tanh(self.raw_betas)

    def stats(self) -> dict[str, float | list[float]]:
        betas = self.betas().detach().cpu()
        return {
            "source_betas": [float(value) for value in betas],
            "claude_beta": float(betas[0]),
            "merge_beta": float(betas[1]),
            "source_cosine": float(self.source_cosine),
            "margin_threshold": float(self.margin_threshold),
            "margin_temperature": self.margin_temperature,
            "consensus_only": False,
        }

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
        sdcr = self.sdcr_prototypes.index_select(0, ids)
        normalized = F.normalize(images.float(), dim=-1)
        logits = parent_logits + self.sdcr_beta * (normalized @ sdcr.T)
        if not enabled:
            return logits
        gate, _, top_positions = self.gate_values(logits, ids, images)
        correction = torch.zeros_like(logits)
        betas = self.betas()
        for source_index in range(2):
            source = self.source_prototypes[source_index].index_select(0, ids)
            source_logits = normalized @ source.T
            top_scores = source_logits.gather(1, top_positions)
            centered = top_scores - top_scores.mean(dim=1, keepdim=True)
            local = torch.zeros_like(logits)
            local.scatter_(1, top_positions, centered)
            correction = correction + betas[source_index] * local
        return logits + gate.unsqueeze(1) * correction
