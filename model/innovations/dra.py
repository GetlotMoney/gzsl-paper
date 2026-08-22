from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_role_matrix(descriptions):
    roles = F.normalize(descriptions.detach().float(), dim=-1)
    return roles.reshape(roles.shape[0], -1)


def fit_ridge_description_map(features, labels, descriptions, ridge):
    images = F.normalize(features.detach().float(), dim=-1)
    targets = normalized_role_matrix(descriptions).index_select(0, labels.long())
    gram = images.T @ images
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + float(ridge) * identity, images.T @ targets)


class DescriptionResidualAlignment(nn.Module):
    """从图像预测八角色描述，并作为主语义logit的训练式残差。"""

    def __init__(self, ridge_weight, descriptions, max_beta=20.0):
        super().__init__()
        if descriptions.ndim != 3:
            raise ValueError("DRA描述张量必须是[C,R,D]。")
        self.role_count = int(descriptions.shape[1])
        self.role_dim = int(descriptions.shape[2])
        self.register_buffer("ridge_weight", ridge_weight.detach().float())
        self.register_buffer("description_matrix", normalized_role_matrix(descriptions))
        self.max_beta = float(max_beta)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self):
        return self.max_beta * torch.tanh(self.raw_beta)

    def description_logits(self, images, class_ids=None):
        predicted = F.normalize(images.float(), dim=-1) @ self.ridge_weight
        predicted = predicted.reshape(-1, self.role_count, self.role_dim)
        predicted = F.normalize(predicted, dim=-1).reshape(predicted.shape[0], -1)
        descriptions = self.description_matrix
        if class_ids is not None:
            descriptions = descriptions.index_select(0, class_ids.to(descriptions.device))
        return predicted @ descriptions.T / self.role_count

    def logits(self, images, prototypes, scale, class_ids=None, enabled=True):
        selected = prototypes
        if class_ids is not None:
            selected = prototypes.index_select(0, class_ids.to(prototypes.device))
        parent = F.normalize(images.float(), dim=-1) @ selected.T * scale
        if not enabled:
            return parent
        return parent + self.beta() * self.description_logits(images, class_ids)
