from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def residual_subspace(
    base_prototypes: torch.Tensor,
    adapted_prototypes: torch.Tensor,
    source_classes: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    base = F.normalize(base_prototypes, dim=-1)
    source_classes = source_classes.to(base.device)
    source_base = base.index_select(0, source_classes)
    source_adapted = F.normalize(adapted_prototypes, dim=-1)
    residual = source_adapted - (
        source_adapted * source_base
    ).sum(dim=-1, keepdim=True) * source_base
    _, _, vh = torch.linalg.svd(residual, full_matrices=False)
    return vh[: int(rank)]


def orthogonal_transport(
    base: torch.Tensor,
    value: torch.Tensor,
    step: torch.Tensor,
    basis: torch.Tensor,
    mix: torch.Tensor,
    projection_mode: str = "shared",
) -> torch.Tensor:
    base = F.normalize(base, dim=-1)
    value = F.normalize(value, dim=-1)
    tangent = value - (value * base).sum(dim=-1, keepdim=True) * base
    projected = (tangent @ basis.T) @ basis
    if projection_mode == "shared":
        alternative = projected
    elif projection_mode == "complement":
        alternative = tangent - projected
    else:
        raise ValueError("未知ORT投影模式。")
    blended = (1.0 - mix) * tangent + mix * alternative
    return F.normalize(base + step.unsqueeze(-1) * blended, dim=-1)


class OrthogonalMix(nn.Module):
    def __init__(self, initial_mix: float = 0.1):
        super().__init__()
        self.raw_mix = nn.Parameter(
            torch.tensor(math.log(float(initial_mix) / (1.0 - float(initial_mix))))
        )

    def forward(self):
        return torch.sigmoid(self.raw_mix)
