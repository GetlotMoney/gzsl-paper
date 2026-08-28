from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def semantic_factor_matrix(descriptions, rank):
    roles = F.normalize(descriptions.detach().float(), dim=-1)
    flattened = roles.reshape(roles.shape[0], -1)
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    factors = centered @ right[: int(rank)].T
    return F.normalize(factors, dim=-1)


def fit_ridge_factor_map(features, labels, class_factors, ridge):
    images = F.normalize(features.detach().float(), dim=-1)
    targets = class_factors.detach().float().index_select(0, labels.long())
    gram = images.T @ images
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + float(ridge) * identity, images.T @ targets)


class SemanticFactorAlignment(nn.Module):
    """从图像预测跨类别描述因子，并作为CCGR logit残差。"""

    def __init__(self, ridge_weight, class_factors, max_beta=20.0):
        super().__init__()
        self.register_buffer("ridge_weight", ridge_weight.detach().float())
        self.register_buffer(
            "class_factors", F.normalize(class_factors.detach().float(), dim=-1)
        )
        self.max_beta = float(max_beta)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self):
        return self.max_beta * torch.tanh(self.raw_beta)

    def factor_logits(self, images, class_ids=None):
        predicted = F.normalize(
            F.normalize(images.float(), dim=-1) @ self.ridge_weight, dim=-1
        )
        factors = self.class_factors
        if class_ids is not None:
            factors = factors.index_select(0, class_ids.to(factors.device))
        return predicted @ factors.T

    def logits(self, images, prototypes, scale, class_ids=None, enabled=True):
        selected = prototypes
        if class_ids is not None:
            selected = prototypes.index_select(0, class_ids.to(prototypes.device))
        parent = F.normalize(images.float(), dim=-1) @ selected.T * scale
        if not enabled:
            return parent
        return parent + self.beta() * self.factor_logits(images, class_ids)
