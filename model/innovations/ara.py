from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_ridge_attribute_map(features, labels, class_attributes, ridge):
    images = F.normalize(features.detach().float(), dim=-1)
    targets = F.normalize(class_attributes.detach().float(), dim=-1).index_select(
        0, labels.long()
    )
    gram = images.T @ images
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + float(ridge) * identity, images.T @ targets)


class AttributeResidualAlignment(nn.Module):
    """把seen图像训练的属性证据作为主语义logit的有界残差。"""

    def __init__(self, ridge_weight, class_attributes, max_beta=20.0):
        super().__init__()
        self.register_buffer("ridge_weight", ridge_weight.detach().float())
        self.register_buffer(
            "class_attributes", F.normalize(class_attributes.detach().float(), dim=-1)
        )
        self.max_beta = float(max_beta)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self):
        return self.max_beta * torch.tanh(self.raw_beta)

    def attribute_logits(self, images, class_ids=None):
        predicted = F.normalize(
            F.normalize(images.float(), dim=-1) @ self.ridge_weight, dim=-1
        )
        attributes = self.class_attributes
        if class_ids is not None:
            attributes = attributes.index_select(0, class_ids.to(attributes.device))
        return predicted @ attributes.T

    def logits(self, images, prototypes, scale, metric, class_ids=None, enabled=True):
        selected = prototypes
        if class_ids is not None:
            selected = prototypes.index_select(0, class_ids.to(prototypes.device))
        parent = metric.logits(images.float(), selected, scale)
        if not enabled:
            return parent
        return parent + self.beta() * self.attribute_logits(images, class_ids)
