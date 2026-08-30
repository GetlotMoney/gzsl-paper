"""Pairwise error-correcting verifier for class-disjoint GZSL screening.

The verifier never receives a class id.  It compares two candidate classes from
their frozen role text and the same frozen image feature used by the parent.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PairwiseErrorCorrectingVerifier(nn.Module):
    """Learn an order-invariant residual margin for a candidate pair."""

    def __init__(self, role_count: int = 8, hidden_dim: int = 32, max_correction: float = 4.0):
        super().__init__()
        if role_count < 1 or hidden_dim < 1 or max_correction <= 0:
            raise ValueError("PECV dimensions and max_correction must be positive.")
        self.role_count = int(role_count)
        self.max_correction = float(max_correction)
        signed_dim = 1 + self.role_count
        # signed evidence, its magnitude, and candidate semantic proximity
        input_dim = signed_dim * 2 + 1
        self.reader = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _reader_input(
        self,
        image_features: torch.Tensor,
        prototype_a: torch.Tensor,
        prototype_b: torch.Tensor,
        roles_a: torch.Tensor,
        roles_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = F.normalize(image_features, dim=-1)
        proto_a = F.normalize(prototype_a, dim=-1)
        proto_b = F.normalize(prototype_b, dim=-1)
        role_a = F.normalize(roles_a, dim=-1)
        role_b = F.normalize(roles_b, dim=-1)
        prototype_difference = (image * (proto_a - proto_b)).sum(dim=-1, keepdim=True)
        role_difference = torch.einsum("bd,brd->br", image, role_a - role_b)
        signed = torch.cat((prototype_difference, role_difference), dim=-1)
        semantic_proximity = (proto_a * proto_b).sum(dim=-1, keepdim=True)
        invariant = torch.cat((signed.abs(), semantic_proximity), dim=-1)
        return signed, invariant

    def correction(
        self,
        image_features: torch.Tensor,
        prototype_a: torch.Tensor,
        prototype_b: torch.Tensor,
        roles_a: torch.Tensor,
        roles_b: torch.Tensor,
    ) -> torch.Tensor:
        """Return c(a,b), with c(b,a) == -c(a,b) by construction."""
        signed, invariant = self._reader_input(
            image_features, prototype_a, prototype_b, roles_a, roles_b
        )
        forward = self.reader(torch.cat((signed, invariant), dim=-1)).squeeze(-1)
        reverse = self.reader(torch.cat((-signed, invariant), dim=-1)).squeeze(-1)
        odd_score = 0.5 * (forward - reverse)
        return self.max_correction * torch.tanh(odd_score)


def corrected_topk_scores(
    parent_scores: torch.Tensor,
    image_features: torch.Tensor,
    candidate_local: torch.Tensor,
    prototypes: torch.Tensor,
    role_text: torch.Tensor,
    verifier: PairwiseErrorCorrectingVerifier | None,
) -> torch.Tensor:
    """Apply zero-sum pair corrections inside a frozen Parent Top-K set.

    Passing ``verifier=None`` is the exact module-off path.
    """
    if verifier is None:
        return parent_scores
    if parent_scores.ndim != 2 or candidate_local.shape != parent_scores.shape:
        raise ValueError("PECV parent scores and candidate ids must be [batch, top_k].")
    if image_features.size(0) != parent_scores.size(0):
        raise ValueError("PECV image batch does not match candidate batch.")
    top_k = parent_scores.size(1)
    if top_k < 2:
        return parent_scores
    residual = torch.zeros_like(parent_scores)
    for left in range(top_k):
        for right in range(left + 1, top_k):
            class_left = candidate_local[:, left]
            class_right = candidate_local[:, right]
            correction = verifier.correction(
                image_features,
                prototypes.index_select(0, class_left),
                prototypes.index_select(0, class_right),
                role_text.index_select(0, class_left),
                role_text.index_select(0, class_right),
            )
            residual[:, left] += correction
            residual[:, right] -= correction
    return parent_scores + residual / float(top_k - 1)
