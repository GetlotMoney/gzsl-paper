"""Pair-Contrast Patch Comparator (PCPC)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 768
ROLE_COUNT = 8


class PairContrastPatchComparator(nn.Module):
    """Rerank the parent's top-2 classes with same-patch signed evidence.

    The module has no class-specific parameters.  Candidate semantics enter only
    through the frozen per-class role text supplied to ``forward``.
    """

    def __init__(
        self,
        *,
        rank: int = 32,
        patch_temperature: float = 0.07,
        max_logit_correction: float = 1.0,
    ) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("PCPC rank必须为正数。")
        if float(patch_temperature) <= 0.0:
            raise ValueError("PCPC patch_temperature必须为正数。")
        if float(max_logit_correction) <= 0.0:
            raise ValueError("PCPC max_logit_correction必须为正数。")
        self.rank = int(rank)
        self.patch_temperature = float(patch_temperature)
        self.max_logit_correction = float(max_logit_correction)
        self.visual_projection = nn.Linear(FEATURE_DIM, self.rank, bias=False)
        self.text_projection = nn.Linear(FEATURE_DIM, self.rank, bias=False)
        with torch.no_grad():
            self.text_projection.weight.copy_(self.visual_projection.weight)
        # tanh keeps the correction bounded; zero is the exact neutral start.
        self.raw_strength = nn.Parameter(torch.zeros(()))

    def bounded_strength(self) -> torch.Tensor:
        return self.raw_strength.tanh() * self.max_logit_correction

    @staticmethod
    def _validate_inputs(
        parent_logits: torch.Tensor,
        patches: torch.Tensor,
        role_text: torch.Tensor,
    ) -> None:
        if parent_logits.ndim != 2 or parent_logits.size(1) < 2:
            raise ValueError("PCPC parent_logits必须是至少2类的[B,C]。")
        if (
            patches.ndim != 3
            or patches.size(0) != parent_logits.size(0)
            or patches.size(1) < 1
            or patches.size(2) != FEATURE_DIM
        ):
            raise ValueError("PCPC patches必须是与batch匹配的[B,N,768]。")
        if tuple(role_text.shape[1:]) != (ROLE_COUNT, FEATURE_DIM):
            raise ValueError("PCPC role_text必须是[C,8,768]。")
        if role_text.size(0) != parent_logits.size(1):
            raise ValueError("PCPC role_text类别轴必须与parent logits一致。")
        if not all(torch.isfinite(value).all() for value in (parent_logits, patches, role_text)):
            raise ValueError("PCPC输入包含NaN/Inf。")

    def pair_evidence(
        self,
        patches: torch.Tensor,
        role_text: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return signed A-vs-B evidence on identical patches for every role."""
        if candidate_ids.ndim != 2 or candidate_ids.size(1) != 2:
            raise ValueError("PCPC candidate_ids必须是[B,2]。")
        if patches.size(0) != candidate_ids.size(0):
            raise ValueError("PCPC candidate batch不匹配。")
        class_count = role_text.size(0)
        device = self.text_projection.weight.device
        candidates = candidate_ids.to(device=device, dtype=torch.long)
        if bool((candidates < 0).any()) or bool((candidates >= class_count).any()):
            raise ValueError("PCPC candidate超出类别轴。")
        if bool(candidates[:, 0].eq(candidates[:, 1]).any()):
            raise ValueError("PCPC候选对必须是两个不同类别。")

        # Assets are frozen by protocol. Detach makes that gradient boundary
        # explicit while gradients still flow into both shared projections.
        patch_values = patches.detach().to(device).float()
        text_values = role_text.detach().to(device).float()
        left = text_values.index_select(0, candidates[:, 0])
        right = text_values.index_select(0, candidates[:, 1])
        role_difference = F.normalize(left - right, dim=-1)
        visual = F.normalize(self.visual_projection(patch_values), dim=-1)
        contrast = F.normalize(self.text_projection(role_difference), dim=-1)
        signed_patch_role = torch.einsum("bnr,bkr->bnk", visual, contrast)
        patch_weights = F.softmax(
            signed_patch_role.abs() / self.patch_temperature, dim=1
        )
        role_evidence = (patch_weights * signed_patch_role).sum(dim=1)
        delta = role_evidence.mean(dim=1)
        if not all(
            torch.isfinite(value).all()
            for value in (signed_patch_role, patch_weights, role_evidence, delta)
        ):
            raise FloatingPointError("PCPC evidence包含NaN/Inf。")
        return {
            "delta": delta,
            "role_evidence": role_evidence,
            "patch_weights": patch_weights,
            "signed_patch_role": signed_patch_role,
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        patches: torch.Tensor,
        role_text: torch.Tensor,
        *,
        enabled: bool = True,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        self._validate_inputs(parent_logits, patches, role_text)
        if not enabled:
            if return_diagnostics:
                zeros = parent_logits.new_zeros(parent_logits.size(0))
                return parent_logits, {
                    "candidate_ids": parent_logits.topk(2, dim=1).indices,
                    "delta": zeros,
                    "scaled_delta": zeros,
                    "correction": torch.zeros_like(parent_logits),
                    "strength": self.bounded_strength(),
                }
            return parent_logits

        candidate_ids = parent_logits.topk(2, dim=1).indices
        evidence = self.pair_evidence(patches, role_text, candidate_ids)
        scaled_delta = self.bounded_strength() * evidence["delta"]
        correction = torch.zeros_like(parent_logits)
        correction.scatter_add_(
            1,
            candidate_ids,
            torch.stack((scaled_delta, -scaled_delta), dim=1),
        )
        corrected = parent_logits + correction
        if not torch.isfinite(corrected).all():
            raise FloatingPointError("PCPC corrected logits包含NaN/Inf。")
        if return_diagnostics:
            return corrected, {
                **evidence,
                "candidate_ids": candidate_ids,
                "scaled_delta": scaled_delta,
                "correction": correction,
                "strength": self.bounded_strength(),
            }
        return corrected


def pairwise_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seen_classes: torch.Tensor,
    *,
    margin: float = 0.02,
) -> torch.Tensor:
    """Seen-image ranking loss against the hardest competing class."""
    if logits.ndim != 2 or logits.size(1) < 2:
        raise ValueError("PCPC loss logits必须是至少2类的[B,C]。")
    labels = targets.to(device=logits.device, dtype=torch.long)
    seen = torch.as_tensor(seen_classes, device=logits.device, dtype=torch.long)
    if labels.ndim != 1 or labels.numel() != logits.size(0):
        raise ValueError("PCPC loss targets必须是[B]。")
    if seen.ndim != 1 or seen.numel() < 2 or seen.unique().numel() != seen.numel():
        raise ValueError("PCPC seen_classes必须是至少2个不重复类别。")
    if bool((seen < 0).any()) or bool((seen >= logits.size(1)).any()):
        raise ValueError("PCPC seen_classes超出logit类别轴。")
    if bool(~torch.isin(labels, seen).all()):
        raise ValueError("PCPC训练loss只接受seen图像标签。")
    if float(margin) < 0.0:
        raise ValueError("PCPC margin不得为负数。")

    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], -torch.inf)
    hard_negative = competitors.max(dim=1).values
    loss = F.softplus(float(margin) - true_logits + hard_negative).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("PCPC hard-negative loss包含NaN/Inf。")
    return loss
