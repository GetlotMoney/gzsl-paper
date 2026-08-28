from __future__ import annotations

import torch
import torch.nn.functional as F


def classwise_bi_orthogonal_residual(
    source: torch.Tensor,
    first_direction: torch.Tensor,
    second_direction: torch.Tensor,
) -> torch.Tensor:
    """逐类对两个Gram–Schmidt方向去投影并归一化。"""
    if source.shape != first_direction.shape or source.shape != second_direction.shape:
        raise ValueError("source与两个方向形状必须一致。")
    if source.ndim != 2:
        raise ValueError("语义原型必须是[C,D]。")
    source_n = F.normalize(source.float(), dim=-1)
    q1 = F.normalize(first_direction.float(), dim=-1)
    second = second_direction.float() - (
        second_direction.float() * q1
    ).sum(dim=-1, keepdim=True) * q1
    q2 = F.normalize(second, dim=-1)
    residual = source_n
    residual = residual - (residual * q1).sum(dim=-1, keepdim=True) * q1
    residual = residual - (residual * q2).sum(dim=-1, keepdim=True) * q2
    residual = F.normalize(residual, dim=-1)
    if not torch.isfinite(residual).all():
        raise ValueError("二维正交语义残差包含NaN/Inf。")
    return residual
