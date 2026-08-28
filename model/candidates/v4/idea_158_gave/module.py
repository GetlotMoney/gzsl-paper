"""Geodesic-Aligned Visual Evidence (GAVE)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 768
ROLE_IDS = (0, 1, 2, 3, 4, 5, 7)  # six local roles + unique; overall is excluded


def geodesic_tangent(
    mean8: torch.Tensor,
    value: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the unit tangent from Mean8 toward the TG Value candidate."""
    if mean8.ndim < 2 or mean8.shape != value.shape or mean8.size(-1) != FEATURE_DIM:
        raise ValueError("GAVE Mean8/Value必须是相同shape且末维为768。")
    base = F.normalize(mean8.float(), dim=-1)
    target = F.normalize(value.float(), dim=-1)
    cosine = (base * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = target - cosine * base
    norm = tangent.norm(dim=-1, keepdim=True)
    valid = norm.squeeze(-1) > float(eps)
    direction = tangent / norm.clamp_min(float(eps))
    direction = torch.where(valid.unsqueeze(-1), direction, torch.zeros_like(direction))
    return direction, valid


class GeodesicAlignedVisualEvidence(nn.Module):
    """Validate a candidate's semantic transport direction with localized patches.

    Role text is used only to locate relevant patches. The evidence itself is
    patch alignment with the Mean8-to-Value geodesic tangent, which is distinct
    from direct patch-text similarity and candidate-pair text differences.
    """

    def __init__(
        self,
        *,
        top_candidates: int = 5,
        visible_roles: int = 3,
        role_temperature: float = 0.07,
        max_logit_correction: float = 1.0,
    ) -> None:
        super().__init__()
        if int(top_candidates) < 2:
            raise ValueError("GAVE至少需要2个候选类别。")
        if not 1 <= int(visible_roles) <= len(ROLE_IDS):
            raise ValueError("GAVE visible_roles超出local+unique角色数量。")
        if float(role_temperature) <= 0.0 or float(max_logit_correction) <= 0.0:
            raise ValueError("GAVE温度和最大修正必须为正数。")
        self.top_candidates = int(top_candidates)
        self.visible_roles = int(visible_roles)
        self.role_temperature = float(role_temperature)
        self.max_logit_correction = float(max_logit_correction)
        self.raw_strength = nn.Parameter(torch.zeros(()))

    def strength(self) -> torch.Tensor:
        return self.max_logit_correction * torch.tanh(self.raw_strength)

    @staticmethod
    def _validate_inputs(
        parent_logits: torch.Tensor,
        patches: torch.Tensor,
        role_text: torch.Tensor,
        mean8: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if parent_logits.ndim != 2:
            raise ValueError("GAVE parent_logits必须是[B,C]。")
        batch, classes = parent_logits.shape
        if patches.ndim != 3 or tuple(patches.shape[::2]) != (batch, FEATURE_DIM):
            raise ValueError("GAVE patches必须是[B,N,768]。")
        if tuple(role_text.shape) != (classes, 8, FEATURE_DIM):
            raise ValueError("GAVE role_text必须是[C,8,768]。")
        if tuple(mean8.shape) != (classes, FEATURE_DIM) or value.shape != mean8.shape:
            raise ValueError("GAVE Mean8/Value必须是[C,768]。")
        if not all(
            tensor.is_floating_point() and bool(torch.isfinite(tensor).all())
            for tensor in (parent_logits, patches, role_text, mean8, value)
        ):
            raise ValueError("GAVE输入必须是有限浮点张量。")

    def components(
        self,
        parent_logits: torch.Tensor,
        patches: torch.Tensor,
        role_text: torch.Tensor,
        mean8: torch.Tensor,
        value: torch.Tensor,
        *,
        candidate_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(parent_logits, patches, role_text, mean8, value)
        batch, classes = parent_logits.shape
        device = parent_logits.device
        dtype = parent_logits.dtype
        if candidate_ids is None:
            if self.top_candidates > classes:
                raise ValueError("GAVE top_candidates超过类别数量。")
            candidates = parent_logits.detach().topk(self.top_candidates, dim=1).indices
        else:
            candidates = torch.as_tensor(candidate_ids, device=device, dtype=torch.long)
            if candidates.ndim != 2 or candidates.size(0) != batch:
                raise ValueError("GAVE candidate_ids必须是[B,M]。")
            if candidates.size(1) < 2:
                raise ValueError("GAVE每行至少需要2个候选。")
            if any(row.unique().numel() != row.numel() for row in candidates):
                raise ValueError("GAVE每行候选不得重复。")
            if bool((candidates < 0).any()) or bool((candidates >= classes).any()):
                raise ValueError("GAVE candidate_ids超出类别轴。")

        patch_values = F.normalize(patches.detach().to(device=device, dtype=dtype), dim=-1)
        roles = F.normalize(role_text.detach().to(device=device, dtype=dtype), dim=-1)
        base = F.normalize(mean8.detach().to(device=device, dtype=dtype), dim=-1)
        target = F.normalize(value.detach().to(device=device, dtype=dtype), dim=-1)
        role_axis = torch.tensor(ROLE_IDS, device=device)
        candidate_roles = roles.index_select(1, role_axis)[candidates]
        candidate_base = base[candidates]
        candidate_target = target[candidates]
        direction, valid = geodesic_tangent(candidate_base, candidate_target)
        direction = direction.to(dtype=dtype)

        role_similarity = torch.einsum("bnd,bmrd->bmrn", patch_values, candidate_roles)
        role_attention = F.softmax(role_similarity / self.role_temperature, dim=-1)
        directional_alignment = torch.einsum("bnd,bmd->bmn", patch_values, direction)
        centered_alignment = directional_alignment - directional_alignment.mean(
            dim=-1, keepdim=True
        )
        role_evidence = torch.sum(
            role_attention * centered_alignment.unsqueeze(2), dim=-1
        )
        role_evidence = torch.where(
            valid.unsqueeze(-1), role_evidence, torch.zeros_like(role_evidence)
        )
        strongest_visible = role_evidence.topk(self.visible_roles, dim=-1).values
        coverage = strongest_visible.mean(dim=-1)
        relative_evidence = coverage - coverage.mean(dim=1, keepdim=True)
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                role_similarity,
                role_attention,
                directional_alignment,
                role_evidence,
                coverage,
                relative_evidence,
            )
        ):
            raise FloatingPointError("GAVE视觉证据包含NaN/Inf。")
        return {
            "candidate_ids": candidates,
            "role_attention": role_attention,
            "directional_alignment": directional_alignment,
            "role_evidence": role_evidence,
            "coverage": coverage,
            "relative_evidence": relative_evidence,
            "valid_direction": valid,
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        patches: torch.Tensor,
        role_text: torch.Tensor,
        mean8: torch.Tensor,
        value: torch.Tensor,
        *,
        enabled: bool = True,
        candidate_ids: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not enabled:
            return (parent_logits, {}) if return_diagnostics else parent_logits
        parts = self.components(
            parent_logits,
            patches,
            role_text,
            mean8,
            value,
            candidate_ids=candidate_ids,
        )
        candidate_delta = self.strength() * parts["relative_evidence"]
        correction = torch.zeros_like(parent_logits)
        correction.scatter_add_(1, parts["candidate_ids"], candidate_delta)
        corrected = parent_logits + correction
        if return_diagnostics:
            return corrected, {
                **parts,
                "candidate_delta": candidate_delta,
                "correction": correction,
                "strength": self.strength(),
            }
        return corrected
