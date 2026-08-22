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
