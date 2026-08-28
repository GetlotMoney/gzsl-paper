from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_attribute_to_visual_map(attributes, seenclasses, centroids, ridge):
    semantics = F.normalize(attributes.detach().float(), dim=-1).index_select(
        0, seenclasses.long()
    )
    targets = F.normalize(centroids.detach().float(), dim=-1)
    gram = semantics.T @ semantics
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(
        gram + float(ridge) * identity, semantics.T @ targets
    )


class VisualPrototypeAttributeResidual(nn.Module):
    """用属性生成的视觉原型为冻结CRA增加反向映射证据。"""

    def __init__(self, base_cra, visual_prototypes, max_beta=20.0):
        super().__init__()
        self.base_cra = base_cra
        for parameter in self.base_cra.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "visual_prototypes", F.normalize(visual_prototypes.detach().float(), dim=-1)
        )
        self.max_beta = float(max_beta)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self):
        return self.max_beta * torch.tanh(self.raw_beta)

    def visual_logits(self, images, class_ids=None):
        prototypes = self.visual_prototypes
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return F.normalize(images.float(), dim=-1) @ prototypes.T

    def logits(self, images, prototypes, scale, metric, class_ids=None, enabled=True):
        parent = self.base_cra.logits(images, prototypes, scale, metric, class_ids)
        if not enabled:
            return parent
        return parent + self.beta() * self.visual_logits(images, class_ids)
