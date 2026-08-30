"""One-stage TG+GTD+PECV model for formal CUB GZSL evaluation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model.innovations.gtd_tst import GTDTSTModel


class PairwiseErrorCorrectingVerifier(nn.Module):
    """Shared antisymmetric verifier with an exact zero initial state."""

    def __init__(self, role_count: int = 8, hidden_dim: int = 32, max_correction: float = 4.0):
        super().__init__()
        if role_count != 8 or hidden_dim != 32 or float(max_correction) != 4.0:
            raise ValueError("PECV formal first run fixes roles=8, hidden=32, max correction=4.")
        self.role_count = int(role_count)
        self.max_correction = float(max_correction)
        signed_dim = 1 + self.role_count
        self.reader = nn.Sequential(
            nn.Linear(signed_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.reader[-1].weight)
        nn.init.zeros_(self.reader[-1].bias)

    def correction(
        self,
        image_features: torch.Tensor,
        prototype_a: torch.Tensor,
        prototype_b: torch.Tensor,
        roles_a: torch.Tensor,
        roles_b: torch.Tensor,
    ) -> torch.Tensor:
        image = F.normalize(image_features.float(), dim=-1)
        proto_a = F.normalize(prototype_a.float(), dim=-1)
        proto_b = F.normalize(prototype_b.float(), dim=-1)
        role_a = F.normalize(roles_a.float(), dim=-1)
        role_b = F.normalize(roles_b.float(), dim=-1)
        proto_difference = (image * (proto_a - proto_b)).sum(dim=-1, keepdim=True)
        role_difference = torch.einsum("bd,brd->br", image, role_a - role_b)
        signed = torch.cat((proto_difference, role_difference), dim=-1)
        proximity = (proto_a * proto_b).sum(dim=-1, keepdim=True)
        invariant = torch.cat((signed.abs(), proximity), dim=-1)
        forward = self.reader(torch.cat((signed, invariant), dim=-1)).squeeze(-1)
        reverse = self.reader(torch.cat((-signed, invariant), dim=-1)).squeeze(-1)
        return self.max_correction * torch.tanh(0.5 * (forward - reverse))


def corrected_topk_scores(
    parent_scores: torch.Tensor,
    image_features: torch.Tensor,
    candidate_ids: torch.Tensor,
    prototypes: torch.Tensor,
    role_text: torch.Tensor,
    verifier: PairwiseErrorCorrectingVerifier | None,
) -> torch.Tensor:
    """Add zero-sum pair corrections; ``None`` is bitwise module-off."""
    if verifier is None:
        return parent_scores
    if parent_scores.ndim != 2 or candidate_ids.shape != parent_scores.shape:
        raise ValueError("PECV candidate scores and ids must share [batch,top_k].")
    top_k = parent_scores.size(1)
    if top_k < 2:
        return parent_scores
    residual = torch.zeros_like(parent_scores)
    for left in range(top_k):
        for right in range(left + 1, top_k):
            left_ids = candidate_ids[:, left]
            right_ids = candidate_ids[:, right]
            value = verifier.correction(
                image_features,
                prototypes.index_select(0, left_ids),
                prototypes.index_select(0, right_ids),
                role_text.index_select(0, left_ids),
                role_text.index_select(0, right_ids),
            )
            residual[:, left] += value
            residual[:, right] -= value
    return parent_scores + residual / float(top_k - 1)


def stable_topk_ids(
    logits: torch.Tensor,
    class_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Descending score with ascending global class id as exact tie break."""
    if logits.ndim != 2 or logits.size(1) != class_ids.numel():
        raise ValueError("PECV logits/class axis mismatch.")
    order = torch.argsort(class_ids, stable=True)
    ordered_ids = class_ids.index_select(0, order)
    ordered_logits = logits.index_select(1, order)
    ranks = torch.argsort(ordered_logits, dim=1, descending=True, stable=True)
    return ordered_ids.index_select(0, ranks[:, : int(top_k)].reshape(-1)).reshape(
        logits.size(0), int(top_k)
    )


class PECVGTDModel(nn.Module):
    """GTD backbone plus a shared Top-5 candidate verifier."""

    def __init__(self, backbone: GTDTSTModel):
        super().__init__()
        self.backbone = backbone
        self.verifier = PairwiseErrorCorrectingVerifier()

    @property
    def seen_classes(self) -> torch.Tensor:
        return self.backbone.seen_classes

    @property
    def unseen_classes(self) -> torch.Tensor:
        return self.backbone.unseen_classes

    def scale(self) -> torch.Tensor:
        return self.backbone.scale()

    def prototype_bundle(self) -> dict[str, torch.Tensor]:
        return self.backbone.prototype_bundle()

    def role_text(self) -> torch.Tensor:
        return self.backbone.parent.tg_vpr.sentence_embeds

    def corrected_scores(
        self,
        images: torch.Tensor,
        candidate_ids: torch.Tensor,
        parent_scores: torch.Tensor,
        *,
        enabled: bool,
        prototypes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current = self.backbone.prototypes() if prototypes is None else prototypes
        return corrected_topk_scores(
            parent_scores,
            images,
            candidate_ids,
            current,
            self.role_text(),
            self.verifier if enabled else None,
        )

    def training_candidate_scores(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        seen_classes: torch.Tensor,
        *,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truth-injected Top-5 using current strongest four wrong seen classes."""
        prototypes = self.backbone.prototypes()
        seen = seen_classes.to(images.device).long()
        seen_prototypes = prototypes.index_select(0, seen)
        all_scores = F.normalize(images.float(), dim=-1) @ seen_prototypes.T * self.scale()
        positions = torch.searchsorted(seen, labels)
        if not torch.equal(seen.index_select(0, positions), labels):
            raise ValueError("PECV formal training label is outside seen classes.")
        masked = all_scores.clone()
        masked[torch.arange(masked.size(0), device=images.device), positions] = -torch.inf
        wrong_ids = stable_topk_ids(masked, seen, 4)
        candidates = torch.cat((labels[:, None], wrong_ids), dim=1)
        seen_position_map = torch.full(
            (prototypes.size(0),), -1, dtype=torch.long, device=images.device
        )
        seen_position_map[seen] = torch.arange(seen.numel(), device=images.device)
        candidate_positions = seen_position_map.index_select(0, candidates.reshape(-1)).reshape_as(
            candidates
        )
        parent_scores = all_scores.gather(1, candidate_positions)
        return self.corrected_scores(
            images,
            candidates,
            parent_scores,
            enabled=enabled,
            prototypes=prototypes,
        ), candidates
