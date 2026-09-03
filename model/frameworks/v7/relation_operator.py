"""Text-relation visual operator head for V7 TUNE015.

This head removes the per-image Reader path.  A shared identity-residual
low-rank operator maps fixed text edge directions into unit visual edge
directions, then a fixed ridge graph compiler turns those directions into class
relation prototypes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.frameworks.v6.compiled_pclr import (
    EMBED_DIM,
    ROLE_COUNT,
    _inverse_sigmoid_ratio,
    _inverse_tanh_ratio,
)


@dataclass(frozen=True)
class TextRelationOperatorExport:
    """Frozen deployment tensors for single-path ``x Q^T + b`` inference."""

    q: torch.Tensor
    bias: torch.Tensor


def _validate_seen(seen_classes: torch.Tensor, class_count: int) -> torch.Tensor:
    seen = torch.as_tensor(seen_classes).detach().cpu().long().clone()
    if (
        seen.ndim != 1
        or seen.numel() == 0
        or seen.numel() >= int(class_count)
        or seen.unique().numel() != seen.numel()
        or int(seen.min()) < 0
        or int(seen.max()) >= int(class_count)
    ):
        raise ValueError("TUNE015 seen_classes必须是合法唯一全局类别ID。")
    return seen.sort().values


def text_relation_graph(
    role_prototypes: torch.Tensor,
    *,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a symmetric Top-K union graph and one directed text vector per edge."""

    roles = torch.as_tensor(role_prototypes).detach().cpu().float()
    if roles.ndim != 3 or roles.size(1) != ROLE_COUNT or roles.size(2) != EMBED_DIM:
        raise ValueError("TUNE015 role_prototypes必须是[class_count,8,768]。")
    if not torch.isfinite(roles).all():
        raise ValueError("TUNE015 role_prototypes包含NaN/Inf。")
    class_count = int(roles.size(0))
    if not 0 < int(top_k) < class_count:
        raise ValueError("TUNE015 top_k必须在类别数范围内。")

    roles = F.normalize(roles, dim=-1)
    mean_roles = F.normalize(roles.mean(dim=1), dim=-1)
    cosine = mean_roles @ mean_roles.T
    cosine.fill_diagonal_(-float("inf"))
    neighbors = torch.topk(cosine, k=int(top_k), dim=1).indices
    pairs = {
        tuple(sorted((int(src), int(dst))))
        for src in range(class_count)
        for dst in neighbors[src].tolist()
        if int(src) != int(dst)
    }
    edges = torch.tensor(sorted(pairs), dtype=torch.long)
    if edges.numel() == 0:
        raise ValueError("TUNE015 text relation graph没有边。")
    direction = (roles[edges[:, 0]] - roles[edges[:, 1]]).mean(dim=1)
    direction = F.normalize(direction, dim=-1)
    degree = torch.bincount(edges.reshape(-1), minlength=class_count)
    graph = {
        "edge_count": int(edges.size(0)),
        "top_k": int(top_k),
        "min_degree": int(degree.min()),
        "max_degree": int(degree.max()),
        "mean_degree": float(degree.float().mean()),
        "text_direction_norm_mean": float(direction.norm(dim=1).mean()),
    }
    return direction, edges, graph


def visual_edge_targets(
    centroids: torch.Tensor,
    seen_classes: torch.Tensor,
    edges: torch.Tensor,
    class_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trainval-only normalized visual differences for seen-seen edges."""

    seen = _validate_seen(seen_classes, class_count)
    centers = torch.as_tensor(centroids).detach().float()
    if tuple(centers.shape) != (seen.numel(), EMBED_DIM):
        raise ValueError("TUNE015 seen visual centroids必须是[seen_count,768]。")
    if not torch.isfinite(centers).all():
        raise ValueError("TUNE015 visual centroids包含NaN/Inf。")
    edges_cpu = torch.as_tensor(edges).detach().cpu().long()
    if edges_cpu.ndim != 2 or edges_cpu.size(1) != 2:
        raise ValueError("TUNE015 edges必须是[edge_count,2]。")

    seen_mask = torch.zeros(int(class_count), dtype=torch.bool)
    seen_mask[seen] = True
    seen_edges = seen_mask.index_select(0, edges_cpu[:, 0]) & seen_mask.index_select(0, edges_cpu[:, 1])
    global_to_seen = torch.full((int(class_count),), -1, dtype=torch.long)
    global_to_seen[seen] = torch.arange(seen.numel())
    endpoints = global_to_seen.index_select(0, edges_cpu[seen_edges].reshape(-1))
    if endpoints.numel() == 0:
        raise ValueError("TUNE015 没有seen-seen边可用于视觉方向对齐。")
    endpoints = endpoints.reshape(-1, 2)
    diff = centers.index_select(0, endpoints[:, 0]) - centers.index_select(0, endpoints[:, 1])
    return seen_edges, F.normalize(diff, dim=-1)


def ridge_compile_map(
    edges: torch.Tensor,
    class_count: int,
    *,
    ridge_lambda: float,
) -> torch.Tensor:
    edges_cpu = torch.as_tensor(edges).detach().cpu().long()
    if edges_cpu.ndim != 2 or edges_cpu.size(1) != 2 or edges_cpu.numel() == 0:
        raise ValueError("TUNE015 edge_index必须是非空[edge_count,2]。")
    if int(edges_cpu.min()) < 0 or int(edges_cpu.max()) >= int(class_count):
        raise ValueError("TUNE015 edge端点超出类别轴。")
    if not math.isfinite(float(ridge_lambda)) or float(ridge_lambda) <= 0.0:
        raise ValueError("TUNE015 ridge_lambda必须为有限正数。")
    edge_count = int(edges_cpu.size(0))
    incidence = torch.zeros(edge_count, int(class_count), dtype=torch.float64)
    rows = torch.arange(edge_count)
    incidence[rows, edges_cpu[:, 0]] = 1.0
    incidence[rows, edges_cpu[:, 1]] = -1.0
    system = incidence.T @ incidence + float(ridge_lambda) * torch.eye(int(class_count), dtype=torch.float64)
    return torch.linalg.solve(system, incidence.T).float()


class TextRelationOperatorHead(nn.Module):
    """TG/GTD source plus shared text-to-visual relation operator."""

    def __init__(
        self,
        *,
        base_prototypes: torch.Tensor,
        role_prototypes: torch.Tensor,
        text_directions: torch.Tensor,
        edge_index: torch.Tensor,
        seen_classes: torch.Tensor,
        visual_seen_edge_mask: torch.Tensor,
        visual_seen_targets: torch.Tensor,
        scale: float,
        ridge_lambda: float = 0.3,
        relation_temperature: float = 0.2,
        seen_logit_gamma: float = 0.91,
        alpha_max: float = 2.0,
        initial_alpha: float = 0.7258594751358033,
        role_weight_max: float = 1.0,
        initial_role_weights: torch.Tensor | None = None,
        operator_rank: int = 32,
        operator_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        base = torch.as_tensor(base_prototypes).detach().cpu().float().clone()
        roles = torch.as_tensor(role_prototypes).detach().cpu().float().clone()
        directions = torch.as_tensor(text_directions).detach().cpu().float().clone()
        edges = torch.as_tensor(edge_index).detach().cpu().long().clone()
        if base.ndim != 2 or base.size(1) != EMBED_DIM or base.size(0) < 2:
            raise ValueError("TUNE015 base_prototypes必须是[class_count,768]。")
        class_count = int(base.size(0))
        if tuple(roles.shape) != (class_count, ROLE_COUNT, EMBED_DIM):
            raise ValueError("TUNE015 role_prototypes必须是[class_count,8,768]。")
        if tuple(directions.shape) != (edges.size(0), EMBED_DIM):
            raise ValueError("TUNE015 text_directions必须是[edge_count,768]。")
        if tuple(edges.shape) != (directions.size(0), 2):
            raise ValueError("TUNE015 edge_index shape错误。")
        if not torch.isfinite(base).all() or not torch.isfinite(roles).all() or not torch.isfinite(directions).all():
            raise ValueError("TUNE015 输入原型或方向包含NaN/Inf。")
        seen = _validate_seen(seen_classes, class_count)
        seen_edge_mask = torch.as_tensor(visual_seen_edge_mask).detach().cpu().bool().clone()
        visual_targets = torch.as_tensor(visual_seen_targets).detach().cpu().float().clone()
        if tuple(seen_edge_mask.shape) != (directions.size(0),):
            raise ValueError("TUNE015 visual_seen_edge_mask shape错误。")
        if tuple(visual_targets.shape) != (int(seen_edge_mask.sum()), EMBED_DIM):
            raise ValueError("TUNE015 visual_seen_targets shape错误。")
        if not torch.isfinite(visual_targets).all():
            raise ValueError("TUNE015 visual_seen_targets包含NaN/Inf。")
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError("TUNE015 scale必须为有限正数。")
        if not math.isfinite(float(relation_temperature)) or float(relation_temperature) <= 0.0:
            raise ValueError("TUNE015 relation_temperature必须为有限正数。")
        if not math.isfinite(float(seen_logit_gamma)) or float(seen_logit_gamma) < 0.0:
            raise ValueError("TUNE015 seen_logit_gamma必须为有限非负数。")
        if int(operator_rank) <= 0 or int(operator_rank) > EMBED_DIM:
            raise ValueError("TUNE015 operator_rank必须位于(0,768]。")
        if not math.isfinite(float(operator_init_std)) or float(operator_init_std) <= 0.0:
            raise ValueError("TUNE015 operator_init_std必须为有限正数。")

        self.register_buffer("base_q", F.normalize(base, dim=-1) * float(scale), persistent=True)
        self.register_buffer("role_q", F.normalize(roles, dim=-1) * float(scale), persistent=True)
        self.register_buffer("text_directions", F.normalize(directions, dim=-1), persistent=True)
        self.register_buffer("edge_index", edges, persistent=True)
        self.register_buffer("ridge_map", ridge_compile_map(edges, class_count, ridge_lambda=float(ridge_lambda)), persistent=True)
        self.register_buffer("visual_seen_edge_mask", seen_edge_mask, persistent=True)
        self.register_buffer("visual_seen_targets", F.normalize(visual_targets, dim=-1), persistent=True)
        self.register_buffer("seen_classes", seen, persistent=True)
        seen_bias = torch.zeros(class_count, dtype=torch.float32)
        seen_bias[seen] = -float(seen_logit_gamma)
        self.register_buffer("seen_bias", seen_bias, persistent=True)
        self.ridge_lambda = float(ridge_lambda)
        self.relation_temperature = float(relation_temperature)
        self.alpha_max = float(alpha_max)
        self.role_weight_max = float(role_weight_max)
        self.class_count = class_count
        self.edge_count = int(edges.size(0))
        self.operator_rank = int(operator_rank)

        generator = torch.Generator(device="cpu").manual_seed(31515)
        self.operator_down = nn.Parameter(
            torch.randn(self.operator_rank, EMBED_DIM, generator=generator) * float(operator_init_std)
        )
        self.operator_up = nn.Parameter(torch.zeros(EMBED_DIM, self.operator_rank))
        self.raw_alpha = nn.Parameter(
            torch.tensor(_inverse_sigmoid_ratio(float(initial_alpha), float(alpha_max)), dtype=torch.float32)
        )
        if initial_role_weights is None:
            initial_role_weights = torch.zeros(ROLE_COUNT, dtype=torch.float32)
            initial_role_weights[0] = 0.16
            initial_role_weights[6] = 0.36
        role_initial = torch.as_tensor(initial_role_weights).detach().cpu().float().clone()
        if tuple(role_initial.shape) != (ROLE_COUNT,):
            raise ValueError("TUNE015 initial_role_weights必须是[8]。")
        self.raw_role_weights = nn.Parameter(_inverse_tanh_ratio(role_initial, self.role_weight_max))

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def role_weights(self) -> torch.Tensor:
        return self.role_weight_max * torch.tanh(self.raw_role_weights)

    def operator_residual(self) -> torch.Tensor:
        down = self.operator_down.to(self.text_directions.device)
        up = self.operator_up.to(self.text_directions.device)
        hidden = self.text_directions @ down.T
        return hidden @ up.T

    def operator_edge_directions(self) -> torch.Tensor:
        raw = self.text_directions + self.operator_residual()
        return F.normalize(raw, dim=-1)

    def normalized_operator_edge_directions(self) -> torch.Tensor:
        return self.operator_edge_directions()

    def compiled_relation_q(self) -> torch.Tensor:
        compiled = self.ridge_map.to(self.text_directions.device) @ self.operator_edge_directions()
        return compiled / self.relation_temperature

    def image_q(self, *, semantic_enabled: bool = True) -> torch.Tensor:
        if not semantic_enabled:
            return self.base_q
        role_residual = torch.einsum("r,crd->cd", self.role_weights(), self.role_q)
        return self.base_q + role_residual

    def export_q(
        self,
        *,
        semantic_enabled: bool = True,
        interaction_enabled: bool = True,
    ) -> torch.Tensor:
        relation_q = self.alpha() * self.compiled_relation_q() if interaction_enabled else torch.zeros_like(self.base_q)
        return self.image_q(semantic_enabled=semantic_enabled) + relation_q

    @staticmethod
    def _validate_images(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 2 or images.size(1) != EMBED_DIM or images.size(0) == 0:
            raise ValueError("TUNE015 images必须是非空[batch,768]。")
        if not torch.isfinite(images).all():
            raise ValueError("TUNE015 images包含NaN/Inf。")
        return images.float()

    def forward(
        self,
        images: torch.Tensor,
        *,
        semantic_enabled: bool = True,
        interaction_enabled: bool = True,
    ) -> torch.Tensor:
        values = F.normalize(self._validate_images(images), dim=-1)
        logits = values @ self.export_q(
            semantic_enabled=semantic_enabled,
            interaction_enabled=interaction_enabled,
        ).T + self.seen_bias
        if tuple(logits.shape) != (values.size(0), self.class_count):
            raise RuntimeError("TUNE015 logits shape错误。")
        return logits

    def visual_direction_alignment_loss(self) -> torch.Tensor:
        selected = self.normalized_operator_edge_directions()[self.visual_seen_edge_mask.to(self.text_directions.device)]
        targets = self.visual_seen_targets.to(selected.device)
        return (1.0 - (selected * targets).sum(dim=-1)).mean()

    def training_losses(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        *,
        seen_device: torch.Tensor,
        global_to_seen: torch.Tensor,
        relation_loss_weight: float,
    ) -> dict[str, torch.Tensor]:
        logits = self(images)
        targets = torch.as_tensor(targets, device=logits.device).long()
        seen_targets = global_to_seen.to(logits.device).index_select(0, targets)
        if not bool(seen_targets.ge(0).all()):
            raise ValueError("TUNE015 seen-only分类CE只允许seen训练标签。")
        seen_logits = logits.index_select(1, seen_device.to(logits.device))
        classification = F.cross_entropy(seen_logits, seen_targets)
        relation = self.visual_direction_alignment_loss()
        total = classification + float(relation_loss_weight) * relation
        return {"total": total, "classification": classification, "relation": relation}

    @torch.no_grad()
    def sync_source_prototypes(self, source_model: nn.Module) -> None:
        was_training = source_model.training
        source_model.eval()
        try:
            scale = float(source_model.scale().detach())
            base = F.normalize(source_model.prototypes().detach().float(), dim=-1) * scale
            roles = F.normalize(source_model.parent.tg_vpr.sentence_embeds.detach().float(), dim=-1) * scale
            if base.shape != self.base_q.shape or roles.shape != self.role_q.shape:
                raise ValueError("TUNE015 source prototype shape发生变化。")
            self.base_q.copy_(base)
            self.role_q.copy_(roles)
        finally:
            source_model.train(was_training)

    @torch.no_grad()
    def export(self) -> TextRelationOperatorExport:
        return TextRelationOperatorExport(
            q=self.export_q().detach().cpu().clone(),
            bias=self.seen_bias.detach().cpu().clone(),
        )

    def parameter_contract(self) -> tuple[dict[str, object], ...]:
        return (
            {"name": "operator_down", "trainable": True, "losses": ("classification", "relation"), "export": "Q_relation_identity_residual"},
            {"name": "operator_up", "trainable": True, "losses": ("classification", "relation"), "export": "Q_relation_identity_residual"},
            {"name": "raw_alpha", "trainable": True, "losses": ("classification",), "export": "Q_relation"},
            {"name": "raw_role_weights", "trainable": True, "losses": ("classification",), "export": "Q_image"},
            {"name": "base_q", "trainable": False, "losses": (), "export": "Q_image"},
            {"name": "text_directions", "trainable": False, "losses": (), "export": "none"},
            {"name": "ridge_map", "trainable": False, "losses": (), "export": "none"},
            {"name": "seen_bias", "trainable": False, "losses": (), "export": "bias"},
        )


class TextRelationOperatorDeployment(nn.Module):
    """Execute only ``normalize(x) Q^T + b``."""

    def __init__(self, *, q: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        q_tensor = torch.as_tensor(q).detach().cpu().float().clone()
        bias_tensor = torch.as_tensor(bias).detach().cpu().float().clone()
        if q_tensor.ndim != 2 or q_tensor.size(1) != EMBED_DIM or q_tensor.size(0) < 2:
            raise ValueError("TUNE015 export q必须是[class_count,768]。")
        if tuple(bias_tensor.shape) != (q_tensor.size(0),):
            raise ValueError("TUNE015 export bias shape错误。")
        if not torch.isfinite(q_tensor).all() or not torch.isfinite(bias_tensor).all():
            raise ValueError("TUNE015 export包含NaN/Inf。")
        self.register_buffer("q", q_tensor, persistent=True)
        self.register_buffer("bias", bias_tensor, persistent=True)

    @classmethod
    def from_export(cls, export: dict[str, torch.Tensor]) -> "TextRelationOperatorDeployment":
        if not isinstance(export, dict) or set(export) != {"q", "bias"}:
            raise ValueError("TUNE015 export字段必须且只能包含q,bias。")
        return cls(q=export["q"], bias=export["bias"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 2 or images.size(1) != EMBED_DIM or images.size(0) == 0:
            raise ValueError("TUNE015部署图像特征必须是非空[batch,768]。")
        values = F.normalize(images.float(), dim=-1)
        return values @ self.q.to(values.device).T + self.bias.to(values.device)
