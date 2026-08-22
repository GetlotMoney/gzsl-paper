from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def semantic_graph_residual(
    base_prototypes: torch.Tensor,
    source_prototypes: torch.Tensor,
    source_classes: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    top_k: int = 5,
    temperature: float = 0.05,
    target_direction: torch.Tensor | None = None,
    alignment_temperature: float = 0.2,
) -> torch.Tensor:
    base = F.normalize(base_prototypes, dim=-1)
    source_classes = source_classes.to(base.device)
    target_classes = target_classes.to(base.device)
    source_base = base.index_select(0, source_classes)
    target_base = base.index_select(0, target_classes)
    source = F.normalize(source_prototypes, dim=-1)
    residual = source - (source * source_base).sum(dim=-1, keepdim=True) * source_base
    similarity = target_base @ source_base.T
    values, indices = similarity.topk(k=int(top_k), dim=1)
    selected = residual.index_select(0, indices.flatten()).view(
        target_classes.numel(), int(top_k), -1
    )
    edge_logits = values / float(temperature)
    if target_direction is not None:
        target_direction = F.normalize(target_direction, dim=-1)
        alignment = (
            F.normalize(selected, dim=-1) * target_direction.unsqueeze(1)
        ).sum(dim=-1)
        edge_logits = edge_logits + alignment / float(alignment_temperature)
    weights = F.softmax(edge_logits, dim=1)
    transported = (weights.unsqueeze(-1) * selected).sum(dim=1)
    return transported - (transported * target_base).sum(dim=-1, keepdim=True) * target_base


def apply_graph_residual(
    prototypes: torch.Tensor, graph_residual: torch.Tensor, strength: torch.Tensor
) -> torch.Tensor:
    return F.normalize(prototypes + strength * graph_residual, dim=-1)


class GraphTransportStrength(nn.Module):
    def __init__(self, max_strength: float = 0.5, initial_strength: float = 0.1):
        super().__init__()
        self.max_strength = float(max_strength)
        ratio = float(initial_strength) / self.max_strength
        self.raw_strength = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    def forward(self) -> torch.Tensor:
        return self.max_strength * torch.sigmoid(self.raw_strength)


class GraphResidualClassifier(nn.Module):
    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        target_classes: torch.Tensor,
        graph_residual: torch.Tensor,
        strength_module: GraphTransportStrength,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("parent_prototypes", parent_prototypes.detach())
        self.register_buffer("target_classes", target_classes.detach().cpu().long())
        self.register_buffer("graph_residual", graph_residual.detach())
        self.register_buffer("_scale", scale.detach().clone())
        self.strength_module = strength_module

    def prototypes(self, *, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return self.parent_prototypes
        result = self.parent_prototypes.clone()
        target = self.target_classes.to(result.device)
        result[target] = apply_graph_residual(
            result.index_select(0, target),
            self.graph_residual,
            self.strength_module(),
        )
        return result

    def scale(self):
        return self._scale
