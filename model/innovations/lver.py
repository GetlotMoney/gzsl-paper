from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalViewEvidenceRouter(nn.Module):
    """Route four local CLIP views into a zero-sum parent top-3 correction."""

    def __init__(
        self,
        hidden_dim: int = 16,
        margin_threshold: float = 0.25,
        margin_temperature: float = 0.1,
        local_temperature: float = 0.07,
        max_strength: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("LVER hidden_dim必须为正。")
        if min(
            float(margin_threshold),
            float(margin_temperature),
            float(local_temperature),
            float(max_strength),
            float(eps),
        ) <= 0:
            raise ValueError("LVER阈值、温度、强度上限和eps必须为正。")
        self.router = nn.Sequential(
            nn.Linear(3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.raw_strength = nn.Parameter(torch.zeros(()))
        self.margin_threshold = float(margin_threshold)
        self.margin_temperature = float(margin_temperature)
        self.local_temperature = float(local_temperature)
        self.max_strength = float(max_strength)
        self.eps = float(eps)

    def strength(self) -> torch.Tensor:
        """Signed, bounded correction strength; exactly zero at initialization."""
        return self.max_strength * torch.tanh(self.raw_strength)

    def _validate_inputs(
        self,
        parent_logits: torch.Tensor,
        local_views: torch.Tensor,
        prototypes: torch.Tensor,
        global_features: torch.Tensor,
    ) -> None:
        if parent_logits.ndim != 2:
            raise ValueError("LVER parent_logits必须是[B,C]。")
        batch, classes = parent_logits.shape
        if tuple(local_views.shape) != (batch, 4, 768):
            raise ValueError("LVER local_views必须是[B,4,768]。")
        if tuple(prototypes.shape) != (classes, 768):
            raise ValueError("LVER prototypes必须是[C,768]。")
        if tuple(global_features.shape) != (batch, 768):
            raise ValueError("LVER global_features必须是[B,768]。")
        if classes < 3:
            raise ValueError("LVER至少需要3个候选类别。")
        if not (
            parent_logits.is_floating_point()
            and local_views.is_floating_point()
            and prototypes.is_floating_point()
            and global_features.is_floating_point()
        ):
            raise ValueError("LVER输入必须是浮点张量。")

    def components(
        self,
        parent_logits: torch.Tensor,
        local_views: torch.Tensor,
        prototypes: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return fixed candidates, routed residual, and ambiguity-consensus gate."""
        self._validate_inputs(parent_logits, local_views, prototypes, global_features)
        dtype = parent_logits.dtype
        device = parent_logits.device
        local = F.normalize(local_views.to(device=device, dtype=dtype), dim=-1, eps=self.eps)
        proto = F.normalize(prototypes.to(device=device, dtype=dtype), dim=-1, eps=self.eps)
        global_view = F.normalize(
            global_features.to(device=device, dtype=dtype), dim=-1, eps=self.eps
        )

        # Candidate membership is a parent decision and must not drift with local evidence.
        parent_top = parent_logits.detach().topk(3, dim=1)
        candidate_indices = parent_top.indices
        candidate_prototypes = proto[candidate_indices]
        local_scores = torch.einsum("bvd,bkd->bvk", local, candidate_prototypes)

        consistency = torch.einsum("bvd,bd->bv", local, global_view)
        local_sorted = local_scores.sort(dim=-1, descending=True).values
        local_margin = local_sorted[..., 0] - local_sorted[..., 1]
        parent_candidate_scores = parent_logits.gather(1, candidate_indices).detach()
        parent_centered = parent_candidate_scores - parent_candidate_scores.mean(
            dim=-1, keepdim=True
        )
        local_centered = local_scores - local_scores.mean(dim=-1, keepdim=True)
        ranking_agreement = F.cosine_similarity(
            local_centered,
            parent_centered.unsqueeze(1).expand_as(local_centered),
            dim=-1,
            eps=self.eps,
        )
        route_features = torch.stack(
            (consistency, local_margin, ranking_agreement), dim=-1
        )
        route_weights = F.softmax(self.router(route_features).squeeze(-1), dim=1)
        routed_scores = torch.sum(route_weights.unsqueeze(-1) * local_scores, dim=1)
        candidate_residual = routed_scores - routed_scores.mean(dim=-1, keepdim=True)

        parent_margin = parent_top.values[:, 0] - parent_top.values[:, 1]
        ambiguity = torch.sigmoid(
            (self.margin_threshold - parent_margin) / self.margin_temperature
        )
        view_probabilities = F.softmax(local_scores / self.local_temperature, dim=-1)
        mean_probability = view_probabilities.mean(dim=1)
        # Normalized concentration is 0 for uniform/disagreeing views and 1 for consensus.
        consensus = ((mean_probability.square().sum(dim=-1) - (1.0 / 3.0)) / (2.0 / 3.0)).clamp(
            0.0, 1.0
        )
        gate = ambiguity * consensus
        return {
            "candidate_indices": candidate_indices,
            "candidate_residual": candidate_residual,
            "route_weights": route_weights,
            "ambiguity": ambiguity,
            "consensus": consensus,
            "gate": gate,
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        local_views: torch.Tensor,
        prototypes: torch.Tensor,
        global_features: torch.Tensor,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return parent_logits
        parts = self.components(parent_logits, local_views, prototypes, global_features)
        candidate_delta = (
            self.strength()
            * parts["gate"].unsqueeze(1)
            * parts["candidate_residual"]
        )
        correction = torch.zeros_like(parent_logits)
        correction.scatter_(1, parts["candidate_indices"], candidate_delta)
        return parent_logits + correction
