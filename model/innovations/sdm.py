from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricDiagonalMetric(nn.Module):
    """对图像和语义原型施加同一个有界正对角度量。"""

    def __init__(self, dimension: int = 768, max_log_weight: float = 0.1):
        super().__init__()
        self.max_log_weight = float(max_log_weight)
        self.raw_log_weight = nn.Parameter(torch.zeros(dimension))

    def log_weight(self):
        value = self.max_log_weight * torch.tanh(self.raw_log_weight)
        return value - value.mean()

    def weight(self):
        return self.log_weight().exp()

    def transform(self, features):
        return F.normalize(features * self.weight(), dim=-1)

    def logits(self, images, prototypes, scale):
        return self.transform(images) @ self.transform(prototypes).T * scale

    def stats(self):
        value = self.weight().detach()
        return {
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "min": float(value.min()),
            "max": float(value.max()),
        }


class SymmetricLowRankMetric(nn.Module):
    """在冻结对角度量上沿seen类别中心主方向学习低秩缩放。"""

    def __init__(
        self,
        base_metric: SymmetricDiagonalMetric,
        basis: torch.Tensor,
        max_subspace_log_weight: float = 0.1,
        freeze_base_metric: bool = True,
    ):
        super().__init__()
        if basis.ndim != 2 or basis.shape[1] != base_metric.raw_log_weight.numel():
            raise ValueError("SDM低秩基形状错误。")
        self.base_metric = base_metric
        if freeze_base_metric:
            for parameter in self.base_metric.parameters():
                parameter.requires_grad_(False)
        self.register_buffer("basis", basis.detach().float())
        self.max_subspace_log_weight = float(max_subspace_log_weight)
        self.raw_subspace_log_weight = nn.Parameter(torch.zeros(basis.shape[0]))

    def subspace_log_weight(self):
        value = self.max_subspace_log_weight * torch.tanh(
            self.raw_subspace_log_weight
        )
        return value - value.mean()

    def transform(self, features):
        diagonal = features * self.base_metric.weight()
        coefficients = diagonal @ self.basis.T
        residual_scale = self.subspace_log_weight().exp() - 1.0
        transformed = diagonal + (coefficients * residual_scale) @ self.basis
        return F.normalize(transformed, dim=-1)

    def logits(self, images, prototypes, scale):
        return self.transform(images) @ self.transform(prototypes).T * scale

    def stats(self):
        base = self.base_metric.stats()
        subspace = self.subspace_log_weight().exp().detach()
        return {
            **base,
            "subspace_mean": float(subspace.mean()),
            "subspace_std": float(subspace.std(unbiased=False)),
            "subspace_min": float(subspace.min()),
            "subspace_max": float(subspace.max()),
        }
