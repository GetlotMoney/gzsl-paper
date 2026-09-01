"""Graph-supervised PCLR head compiled to a frozen class matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_COUNT = 200
EDGE_COUNT = 438
EMBED_DIM = 768
ROLE_COUNT = 8
READER_HIDDEN_DIM = 64


@dataclass(frozen=True)
class CompiledPCLRExport:
    """Frozen deployment tensors for ``h(x) Q^T + b``."""

    q: torch.Tensor
    bias: torch.Tensor
    reader_in_weight: torch.Tensor
    reader_in_bias: torch.Tensor
    reader_out_weight: torch.Tensor
    reader_out_bias: torch.Tensor


def _inverse_sigmoid_ratio(value: float, maximum: float) -> float:
    if not 0.0 < float(value) < float(maximum):
        raise ValueError("初始值必须严格位于(0, maximum)内。")
    ratio = float(value) / float(maximum)
    return math.log(ratio / (1.0 - ratio))


def _inverse_tanh_ratio(value: torch.Tensor, maximum: float) -> torch.Tensor:
    ratio = value.float() / float(maximum)
    if bool(ratio.abs().ge(1.0).any()):
        raise ValueError("角色初值必须严格位于有界范围内。")
    return torch.atanh(ratio)


class CompiledPCLRHead(nn.Module):
    """Internalize frozen pairwise relations into an exportable GZSL head.

    Only the shared Reader, one bounded relation strength, and eight bounded
    role weights are trainable.  The TG/GTD base prototypes, relation graph,
    role text embeddings, and seen calibration bias remain frozen.
    """

    def __init__(
        self,
        *,
        base_prototypes: torch.Tensor,
        role_prototypes: torch.Tensor,
        relation_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        seen_classes: torch.Tensor,
        scale: float,
        reader_in_state: tuple[torch.Tensor, torch.Tensor],
        reader_out_state: tuple[torch.Tensor, torch.Tensor],
        ridge_lambda: float = 0.3,
        relation_temperature: float = 0.2,
        direction_temperature: float = 0.07,
        seen_logit_gamma: float = 0.91,
        alpha_max: float = 2.0,
        initial_alpha: float = 0.7258594751358033,
        role_weight_max: float = 1.0,
        initial_role_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        base = torch.as_tensor(base_prototypes).detach().cpu().float().clone()
        roles = torch.as_tensor(role_prototypes).detach().cpu().float().clone()
        relations = torch.as_tensor(relation_embeddings).detach().cpu().float().clone()
        edges = torch.as_tensor(edge_index).detach().cpu().long().clone()
        seen = torch.as_tensor(seen_classes).detach().cpu().long().clone()
        if tuple(base.shape) != (CLASS_COUNT, EMBED_DIM):
            raise ValueError("C-PCLR base_prototypes必须是[200,768]。")
        if tuple(roles.shape) != (CLASS_COUNT, ROLE_COUNT, EMBED_DIM):
            raise ValueError("C-PCLR role_prototypes必须是[200,8,768]。")
        if tuple(relations.shape) != (EDGE_COUNT, 2, EMBED_DIM):
            raise ValueError("C-PCLR relation_embeddings必须是[438,2,768]。")
        if tuple(edges.shape) != (EDGE_COUNT, 2):
            raise ValueError("C-PCLR edge_index必须是[438,2]。")
        if seen.ndim != 1 or seen.numel() != 150 or seen.unique().numel() != 150:
            raise ValueError("C-PCLR seen_classes必须是150个唯一全局类别ID。")
        if int(edges.min()) < 0 or int(edges.max()) >= CLASS_COUNT:
            raise ValueError("C-PCLR边端点超出类别轴。")
        if not torch.isfinite(base).all() or not torch.isfinite(roles).all():
            raise ValueError("C-PCLR类别原型包含NaN/Inf。")
        if not torch.isfinite(relations).all():
            raise ValueError("C-PCLR关系文本包含NaN/Inf。")
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError("C-PCLR scale必须为有限正数。")
        if not math.isfinite(float(ridge_lambda)) or float(ridge_lambda) <= 0.0:
            raise ValueError("C-PCLR ridge_lambda必须为有限正数。")
        if not math.isfinite(float(relation_temperature)) or float(relation_temperature) <= 0.0:
            raise ValueError("C-PCLR relation_temperature必须为有限正数。")
        if not math.isfinite(float(direction_temperature)) or float(direction_temperature) <= 0.0:
            raise ValueError("C-PCLR direction_temperature必须为有限正数。")
        if not math.isfinite(float(seen_logit_gamma)) or float(seen_logit_gamma) < 0.0:
            raise ValueError("C-PCLR seen_logit_gamma必须为有限非负数。")
        if not math.isfinite(float(role_weight_max)) or float(role_weight_max) <= 0.0:
            raise ValueError("C-PCLR role_weight_max必须为有限正数。")

        base_q = F.normalize(base, dim=-1) * float(scale)
        role_q = F.normalize(roles, dim=-1) * float(scale)
        incidence = torch.zeros(EDGE_COUNT, CLASS_COUNT, dtype=torch.float64)
        rows = torch.arange(EDGE_COUNT)
        incidence[rows, edges[:, 0]] = 1.0
        incidence[rows, edges[:, 1]] = -1.0
        system = incidence.T @ incidence + float(ridge_lambda) * torch.eye(
            CLASS_COUNT, dtype=torch.float64
        )
        mapping = torch.linalg.solve(system, incidence.T)
        direction = (relations[:, 0] - relations[:, 1]).double()
        compiled_g = (mapping @ direction / float(relation_temperature)).float()

        seen_mask = torch.zeros(CLASS_COUNT, dtype=torch.bool)
        seen_mask[seen] = True
        seen_edges = seen_mask.index_select(0, edges[:, 0]) & seen_mask.index_select(
            0, edges[:, 1]
        )
        seen_bias = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        seen_bias[seen] = -float(seen_logit_gamma)

        self.register_buffer("base_q", base_q, persistent=True)
        self.register_buffer("role_q", role_q, persistent=True)
        self.register_buffer("compiled_g", compiled_g, persistent=True)
        self.register_buffer("relation_embeddings", relations, persistent=True)
        self.register_buffer("edge_index", edges, persistent=True)
        self.register_buffer("seen_edge_mask", seen_edges, persistent=True)
        self.register_buffer("seen_classes", seen, persistent=True)
        self.register_buffer("seen_bias", seen_bias, persistent=True)
        self.ridge_lambda = float(ridge_lambda)
        self.relation_temperature = float(relation_temperature)
        self.direction_temperature = float(direction_temperature)
        self.alpha_max = float(alpha_max)
        self.role_weight_max = float(role_weight_max)

        self.reader_in = nn.Linear(EMBED_DIM, READER_HIDDEN_DIM)
        self.reader_out = nn.Linear(READER_HIDDEN_DIM, EMBED_DIM)
        self._load_linear_state(self.reader_in, reader_in_state, "reader_in")
        self._load_linear_state(self.reader_out, reader_out_state, "reader_out")

        self.raw_alpha = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid_ratio(float(initial_alpha), float(alpha_max)),
                dtype=torch.float32,
            )
        )
        if initial_role_weights is None:
            initial_role_weights = torch.zeros(ROLE_COUNT, dtype=torch.float32)
            initial_role_weights[0] = 0.16
            initial_role_weights[6] = 0.36
        role_initial = torch.as_tensor(initial_role_weights).detach().cpu().float().clone()
        if tuple(role_initial.shape) != (ROLE_COUNT,):
            raise ValueError("C-PCLR initial_role_weights必须是[8]。")
        self.raw_role_weights = nn.Parameter(
            _inverse_tanh_ratio(role_initial, self.role_weight_max)
        )

    @staticmethod
    def _load_linear_state(
        layer: nn.Linear,
        state: tuple[torch.Tensor, torch.Tensor],
        name: str,
    ) -> None:
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError(f"{name} state必须是(weight,bias)。")
        weight = torch.as_tensor(state[0]).detach().cpu().float()
        bias = torch.as_tensor(state[1]).detach().cpu().float()
        if weight.shape != layer.weight.shape or bias.shape != layer.bias.shape:
            raise ValueError(f"{name} state shape错误。")
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)

    @classmethod
    @torch.no_grad()
    def from_source_model(cls, source_model: nn.Module, **kwargs) -> "CompiledPCLRHead":
        """Construct from the immutable R2/V5 source model."""
        return cls(
            base_prototypes=source_model.prototypes().detach(),
            role_prototypes=source_model.parent.tg_vpr.sentence_embeds.detach(),
            relation_embeddings=source_model.relation_embeddings.detach(),
            edge_index=source_model.edge_index.detach(),
            seen_classes=source_model.seen_classes.detach(),
            scale=float(source_model.scale().detach()),
            reader_in_state=(
                source_model.reader_in.weight.detach(),
                source_model.reader_in.bias.detach(),
            ),
            reader_out_state=(
                source_model.reader_out.weight.detach(),
                source_model.reader_out.bias.detach(),
            ),
            **kwargs,
        )

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def role_weights(self) -> torch.Tensor:
        return self.role_weight_max * torch.tanh(self.raw_role_weights)

    @staticmethod
    def _validate_images(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 2 or images.size(1) != EMBED_DIM or images.size(0) == 0:
            raise ValueError("C-PCLR images必须是非空[batch,768]。")
        if not torch.isfinite(images).all():
            raise ValueError("C-PCLR images包含NaN/Inf。")
        return images.float()

    def read_images(self, images: torch.Tensor, *, visual_enabled: bool = True) -> torch.Tensor:
        values = self._validate_images(images).detach()
        if not visual_enabled:
            return F.normalize(values, dim=-1)
        residual = self.reader_out(F.gelu(self.reader_in(values)))
        return F.normalize(values + residual, dim=-1)

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
        relation_q = (
            self.alpha() * self.compiled_g
            if interaction_enabled
            else torch.zeros_like(self.compiled_g)
        )
        return torch.cat((self.image_q(semantic_enabled=semantic_enabled), relation_q), dim=1)

    def forward(
        self,
        images: torch.Tensor,
        *,
        semantic_enabled: bool = True,
        visual_enabled: bool = True,
        interaction_enabled: bool = True,
    ) -> torch.Tensor:
        values = self._validate_images(images)
        image = F.normalize(values, dim=-1)
        readout = self.read_images(values, visual_enabled=visual_enabled)
        h = torch.cat((image, readout), dim=1)
        logits = h @ self.export_q(
            semantic_enabled=semantic_enabled,
            interaction_enabled=interaction_enabled,
        ).T + self.seen_bias
        if tuple(logits.shape) != (values.size(0), CLASS_COUNT):
            raise RuntimeError("C-PCLR logits shape错误。")
        return logits

    def relation_direction_loss(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Seen-only CE on true-class incident seen-seen edges; no Top-K mining."""
        values = self._validate_images(images)
        targets = torch.as_tensor(targets, device=values.device).long()
        if targets.ndim != 1 or targets.numel() != values.size(0):
            raise ValueError("C-PCLR targets必须是与batch等长的一维全局类别ID。")
        if not bool(torch.isin(targets.detach().cpu(), self.seen_classes.cpu()).all()):
            raise ValueError("C-PCLR方向CE只允许seen训练标签。")
        readout = self.read_images(values, visual_enabled=True)
        relations = self.relation_embeddings.to(readout.device)
        scores = torch.einsum("bd,ekd->bek", readout, relations) / self.direction_temperature
        edges = self.edge_index.to(readout.device)
        incident_a = targets[:, None].eq(edges[None, :, 0])
        incident_b = targets[:, None].eq(edges[None, :, 1])
        incident = (incident_a | incident_b) & self.seen_edge_mask.to(readout.device)[None]
        counts = incident.sum(dim=1)
        if bool(counts.eq(0).any()):
            raise ValueError("C-PCLR seen真类缺少seen-seen incident edge。")
        direction_targets = incident_b.long()
        losses = -F.log_softmax(scores, dim=-1).gather(
            2, direction_targets.unsqueeze(-1)
        ).squeeze(-1)
        return (
            (losses * incident.to(losses.dtype)).sum(dim=1)
            / counts.to(losses.dtype)
        ).mean()

    def training_losses(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        *,
        relation_loss_weight: float,
    ) -> dict[str, torch.Tensor]:
        logits = self(images)
        classification = F.cross_entropy(logits, targets.long())
        relation = self.relation_direction_loss(images, targets)
        total = classification + float(relation_loss_weight) * relation
        return {
            "total": total,
            "classification": classification,
            "relation": relation,
        }

    @torch.no_grad()
    def export(self) -> CompiledPCLRExport:
        return CompiledPCLRExport(
            q=self.export_q().detach().cpu().clone(),
            bias=self.seen_bias.detach().cpu().clone(),
            reader_in_weight=self.reader_in.weight.detach().cpu().clone(),
            reader_in_bias=self.reader_in.bias.detach().cpu().clone(),
            reader_out_weight=self.reader_out.weight.detach().cpu().clone(),
            reader_out_bias=self.reader_out.bias.detach().cpu().clone(),
        )

    def parameter_contract(self) -> tuple[dict[str, object], ...]:
        """Machine-readable train/freeze/export boundary used by tests and receipts."""
        return (
            {"name": "reader_in", "trainable": True, "losses": ("classification", "relation"), "export": "reader"},
            {"name": "reader_out", "trainable": True, "losses": ("classification", "relation"), "export": "reader"},
            {"name": "raw_alpha", "trainable": True, "losses": ("classification",), "export": "Q_relation"},
            {"name": "raw_role_weights", "trainable": True, "losses": ("classification",), "export": "Q_image"},
            {"name": "base_q", "trainable": False, "losses": (), "export": "Q_image"},
            {"name": "compiled_g", "trainable": False, "losses": (), "export": "Q_relation"},
            {"name": "seen_bias", "trainable": False, "losses": (), "export": "bias"},
        )
