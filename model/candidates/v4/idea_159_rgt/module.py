"""Refutation-Gated Transport (RGT)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.frameworks.v4.gtd import geodesic_points


FEATURE_DIM = 768
ROLE_IDS = (0, 1, 2, 3, 4, 5, 7)


class RefutationGatedTransport(nn.Module):
    """Use localized negative evidence only to attenuate an existing GTD angle."""

    def __init__(
        self,
        *,
        top_candidates: int = 5,
        visible_roles: int = 3,
        role_temperature: float = 0.07,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(top_candidates) < 2:
            raise ValueError("RGT至少需要2个候选类别。")
        if not 1 <= int(visible_roles) <= len(ROLE_IDS):
            raise ValueError("RGT visible_roles超出local+unique角色数量。")
        if float(role_temperature) <= 0.0 or float(eps) <= 0.0:
            raise ValueError("RGT温度和eps必须为正数。")
        self.top_candidates = int(top_candidates)
        self.visible_roles = int(visible_roles)
        self.role_temperature = float(role_temperature)
        self.eps = float(eps)

    @staticmethod
    def _validate(
        parent_logits: torch.Tensor,
        patches: torch.Tensor,
        role_text: torch.Tensor,
        mean8: torch.Tensor,
        direction: torch.Tensor,
        theta: torch.Tensor,
    ) -> None:
        if parent_logits.ndim != 2:
            raise ValueError("RGT parent_logits必须是[B,C]。")
        batch, classes = parent_logits.shape
        if patches.ndim != 3 or tuple(patches.shape[::2]) != (batch, FEATURE_DIM):
            raise ValueError("RGT patches必须是[B,N,768]。")
        if tuple(role_text.shape) != (classes, 8, FEATURE_DIM):
            raise ValueError("RGT role_text必须是[C,8,768]。")
        if tuple(mean8.shape) != (classes, FEATURE_DIM) or direction.shape != mean8.shape:
            raise ValueError("RGT Mean8/direction必须是[C,768]。")
        if tuple(theta.shape) != (classes,):
            raise ValueError("RGT theta必须是[C]。")
        if not all(
            tensor.is_floating_point() and bool(torch.isfinite(tensor).all())
            for tensor in (parent_logits, patches, role_text, mean8, direction, theta)
        ):
            raise ValueError("RGT输入必须是有限浮点张量。")
        if bool((theta < 0.0).any()):
            raise ValueError("RGT只接受非负GTD theta。")

    def refutation_components(
        self,
        parent_logits: torch.Tensor,
        patches: torch.Tensor,
        role_text: torch.Tensor,
        mean8: torch.Tensor,
        direction: torch.Tensor,
        theta: torch.Tensor,
        *,
        candidate_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate(parent_logits, patches, role_text, mean8, direction, theta)
        batch, classes = parent_logits.shape
        device = parent_logits.device
        dtype = parent_logits.dtype
        if candidate_ids is None:
            if self.top_candidates > classes:
                raise ValueError("RGT top_candidates超过类别数量。")
            candidates = parent_logits.detach().topk(self.top_candidates, dim=1).indices
        else:
            candidates = torch.as_tensor(candidate_ids, device=device, dtype=torch.long)
            if candidates.ndim != 2 or candidates.size(0) != batch or candidates.size(1) < 2:
                raise ValueError("RGT candidate_ids必须是[B,M]且M至少为2。")
            if any(row.unique().numel() != row.numel() for row in candidates):
                raise ValueError("RGT每行候选不得重复。")
            if bool((candidates < 0).any()) or bool((candidates >= classes).any()):
                raise ValueError("RGT candidate_ids超出类别轴。")

        patch_values = F.normalize(patches.detach().to(device=device, dtype=dtype), dim=-1)
        roles = F.normalize(role_text.detach().to(device=device, dtype=dtype), dim=-1)
        role_axis = torch.tensor(ROLE_IDS, device=device)
        candidate_roles = roles.index_select(1, role_axis)[candidates]
        candidate_direction = direction.detach().to(device=device, dtype=dtype)[candidates]
        candidate_theta = theta.detach().to(device=device, dtype=dtype)[candidates]

        role_similarity = torch.einsum("bnd,bmrd->bmrn", patch_values, candidate_roles)
        role_attention = F.softmax(role_similarity / self.role_temperature, dim=-1)
        directional_alignment = torch.einsum(
            "bnd,bmd->bmn", patch_values, candidate_direction
        )
        centered_alignment = directional_alignment - directional_alignment.mean(
            dim=-1, keepdim=True
        )
        role_evidence = torch.sum(
            role_attention * centered_alignment.unsqueeze(2), dim=-1
        )
        support = role_evidence.clamp_min(0.0).topk(
            self.visible_roles, dim=-1
        ).values.mean(dim=-1)
        refute = (-role_evidence).clamp_min(0.0).topk(
            self.visible_roles, dim=-1
        ).values.mean(dim=-1)
        excess_refutation = (refute - support).clamp_min(0.0)
        refutation_ratio = excess_refutation / (refute + support).clamp_min(self.eps)
        refutation_ratio = torch.where(
            candidate_theta > 0.0,
            refutation_ratio,
            torch.zeros_like(refutation_ratio),
        )
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                role_attention,
                role_evidence,
                support,
                refute,
                refutation_ratio,
            )
        ):
            raise FloatingPointError("RGT反驳证据包含NaN/Inf。")
        return {
            "candidate_ids": candidates,
            "candidate_theta": candidate_theta,
            "role_attention": role_attention,
            "role_evidence": role_evidence,
            "support": support,
            "refute": refute,
            "refutation_ratio": refutation_ratio,
        }

    def attenuated_logits(
        self,
        parent_logits: torch.Tensor,
        image_features: torch.Tensor,
        parent_prototypes: torch.Tensor,
        mean8: torch.Tensor,
        direction: torch.Tensor,
        theta: torch.Tensor,
        scale: torch.Tensor,
        components: dict[str, torch.Tensor],
        *,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("RGT strength必须位于[0,1]。")
        if float(strength) == 0.0:
            return parent_logits, components["candidate_theta"]
        candidates = components["candidate_ids"]
        original_theta = components["candidate_theta"]
        ratio = components["refutation_ratio"]
        adjusted_theta = original_theta * (1.0 - float(strength) * ratio)
        candidate_mean8 = mean8.to(parent_logits.device)[candidates]
        candidate_direction = direction.to(parent_logits.device)[candidates]
        batch, candidate_count, dim = candidate_mean8.shape
        adjusted = geodesic_points(
            candidate_mean8.reshape(batch * candidate_count, dim),
            candidate_direction.reshape(batch * candidate_count, dim),
            adjusted_theta.reshape(batch * candidate_count),
        ).reshape(batch, candidate_count, dim)
        original = parent_prototypes.to(parent_logits.device)[candidates]
        adjusted = torch.where(
            original_theta.gt(0.0).unsqueeze(-1), adjusted, original
        )
        images = F.normalize(image_features.to(parent_logits.device).float(), dim=-1)
        candidate_logits = torch.einsum("bd,bmd->bm", images, adjusted) * scale
        corrected = parent_logits.clone()
        corrected.scatter_(1, candidates, candidate_logits.to(parent_logits.dtype))
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError("RGT调整后logits包含NaN/Inf。")
        return corrected, adjusted_theta
