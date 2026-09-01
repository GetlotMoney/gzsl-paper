"""Anchored Role-Relation Alignment (ARRA) core module."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


EMBED_DIM = 768
CLASS_COUNT = 200
ROLE_COUNT = 8
PATCH_COUNT = 36
EDGE_COUNT = 438
HIDDEN_DIM = 64
RIDGE_LAMBDA = 0.3
RELATION_TEMPERATURE = 0.2
DIRECTION_TEMPERATURE = 0.07
SEEN_LOGIT_GAMMA = 0.575
ROLE_BOUND = 0.75

ARRACondition = Literal[
    "full",
    "s_off",
    "v_off",
    "i_off",
    "additive",
    "shuffled",
]


@dataclass(frozen=True)
class ARRAComponents:
    logits: torch.Tensor
    semantic_logits: torch.Tensor
    visual_logits: torch.Tensor
    relation_logits: torch.Tensor
    calibrated_bias: torch.Tensor
    z: torch.Tensor
    zr: torch.Tensor
    g: torch.Tensor
    attention: torch.Tensor
    beta: torch.Tensor
    alpha: torch.Tensor
    delta: torch.Tensor


def _logit(value: float) -> float:
    if not 0.0 < float(value) < 1.0:
        raise ValueError("logit input must be in (0,1).")
    return math.log(float(value) / (1.0 - float(value)))


def _atanh(value: float) -> float:
    if not -1.0 < float(value) < 1.0:
        raise ValueError("atanh input must be in (-1,1).")
    return 0.5 * math.log((1.0 + float(value)) / (1.0 - float(value)))


def _validate_matrix(
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    value = torch.as_tensor(tensor).detach().cpu().float().clone()
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return value


def _validate_seen_classes(seen_classes: torch.Tensor) -> torch.Tensor:
    seen = torch.as_tensor(seen_classes).detach().cpu().long().sort().values
    if seen.ndim != 1 or seen.numel() == 0:
        raise ValueError("seen_classes must be a non-empty 1D tensor.")
    if seen.unique().numel() != seen.numel():
        raise ValueError("seen_classes must not contain duplicates.")
    if int(seen.min()) < 0 or int(seen.max()) >= CLASS_COUNT:
        raise ValueError("seen_classes contains ids outside the 200-class axis.")
    return seen


def _validate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    edges = torch.as_tensor(edge_index).detach().cpu().long().clone()
    if tuple(edges.shape) != (EDGE_COUNT, 2):
        raise ValueError(f"edge_index must have shape ({EDGE_COUNT}, 2).")
    if int(edges.min()) < 0 or int(edges.max()) >= CLASS_COUNT:
        raise ValueError("edge_index contains class ids outside the 200-class axis.")
    if bool(edges[:, 0].eq(edges[:, 1]).any()):
        raise ValueError("edge_index must not contain self loops.")
    canonical = edges.sort(dim=1).values
    if not torch.equal(edges, canonical):
        raise ValueError("edge_index must use the fixed a_id < b_id direction.")
    if canonical.unique(dim=0).size(0) != EDGE_COUNT:
        raise ValueError("edge_index contains duplicate undirected edges.")
    return edges


def compile_relation_field(
    relation_sentence_embeds: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> torch.Tensor:
    """Compile pairwise directional relation text into class residual vectors."""

    if not math.isfinite(float(ridge_lambda)) or float(ridge_lambda) <= 0.0:
        raise ValueError("ridge_lambda must be a finite positive value.")
    edges = _validate_edges(edge_index)
    relations = torch.as_tensor(relation_sentence_embeds).detach().cpu().float().clone()
    if tuple(relations.shape) == (EDGE_COUNT, 2, EMBED_DIM):
        directions = relations[:, 0] - relations[:, 1]
    elif tuple(relations.shape) == (EDGE_COUNT, EMBED_DIM):
        directions = relations
    else:
        raise ValueError(
            "relation_sentence_embeds must be [438,2,768] or precompiled directions [438,768]."
        )
    if not torch.isfinite(directions).all() or bool(directions.norm(dim=-1).eq(0).any()):
        raise ValueError("relation directions contain NaN or Inf.")

    incidence = torch.zeros(EDGE_COUNT, CLASS_COUNT, dtype=torch.float32)
    rows = torch.arange(EDGE_COUNT)
    incidence[rows, edges[:, 0]] = 1.0
    incidence[rows, edges[:, 1]] = -1.0
    system = incidence.T @ incidence + float(ridge_lambda) * torch.eye(
        CLASS_COUNT, dtype=torch.float32
    )
    laplacian_map = torch.linalg.solve(system, incidence.T)
    return F.normalize(laplacian_map @ directions, dim=-1)


class _ResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        if int(hidden_dim) != HIDDEN_DIM:
            raise ValueError("ARRA fixes adapter hidden_dim=64.")
        self.down = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.up = nn.Linear(HIDDEN_DIM, EMBED_DIM)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(value.float())))


class ARRAClassifier(nn.Module):
    """One-stage ARRA classifier with explicit S/V/I controls."""

    def __init__(
        self,
        class_prototypes: torch.Tensor,
        role_sentence_embeds: torch.Tensor,
        relation_sentence_embeds: torch.Tensor,
        edge_index: torch.Tensor,
        seen_classes: torch.Tensor,
        source_scale: torch.Tensor | float,
        *,
        seen_logit_gamma: float = SEEN_LOGIT_GAMMA,
        ridge_lambda: float = RIDGE_LAMBDA,
        initial_beta: float = 0.10,
        initial_alpha: float = 1.0,
        initial_delta: float = 0.0,
        init_seed: int = 20609,
    ) -> None:
        super().__init__()
        prototypes = _validate_matrix(
            class_prototypes, (CLASS_COUNT, EMBED_DIM), "class_prototypes"
        )
        roles = _validate_matrix(
            role_sentence_embeds,
            (CLASS_COUNT, ROLE_COUNT, EMBED_DIM),
            "role_sentence_embeds",
        )
        relations = torch.as_tensor(relation_sentence_embeds).detach().cpu().float().clone()
        edges = _validate_edges(edge_index)
        seen = _validate_seen_classes(seen_classes)
        scale = torch.as_tensor(source_scale).detach().cpu().float().clone()
        if scale.numel() != 1 or not torch.isfinite(scale).all() or float(scale) <= 0.0:
            raise ValueError("source_scale must be one finite positive scalar.")
        if not 0.0 < float(initial_beta) < 1.0:
            raise ValueError("initial_beta must be in (0,1).")
        if not 0.0 < float(initial_alpha) < 2.0:
            raise ValueError("initial_alpha must be in (0,2).")
        if not -1.0 < float(initial_delta) < 1.0:
            raise ValueError("initial_delta must be in (-1,1).")
        if not math.isfinite(float(seen_logit_gamma)) or float(seen_logit_gamma) < 0.0:
            raise ValueError("seen_logit_gamma must be finite and non-negative.")

        self.class_count = CLASS_COUNT
        self.seen_logit_gamma = float(seen_logit_gamma)
        self.ridge_lambda = float(ridge_lambda)
        self.register_buffer("class_prototypes", prototypes, persistent=True)
        self.register_buffer("role_sentence_embeds", F.normalize(roles, dim=-1), persistent=True)
        self.register_buffer("relation_sentence_embeds", relations, persistent=True)
        self.register_buffer("edge_index", edges, persistent=True)
        self.register_buffer("seen_classes", seen, persistent=True)
        unseen = torch.arange(CLASS_COUNT)[~torch.isin(torch.arange(CLASS_COUNT), seen)]
        self.register_buffer("unseen_classes", unseen, persistent=True)
        self.register_buffer("source_scale", scale.reshape(()), persistent=True)
        self.register_buffer(
            "compiled_relation_field",
            compile_relation_field(relations, edges, ridge_lambda=float(ridge_lambda)),
            persistent=True,
        )

        seen_bias = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        seen_bias[seen] = -float(seen_logit_gamma)
        self.register_buffer("seen_bias", seen_bias, persistent=True)

        self.visual_adapter = _ResidualAdapter()
        self.relation_reader = _ResidualAdapter()
        self.patch_query = nn.Linear(EMBED_DIM, HIDDEN_DIM, bias=False)
        self.patch_key = nn.Linear(EMBED_DIM, HIDDEN_DIM, bias=False)

        rng_state = torch.random.get_rng_state()
        try:
            generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
            nn.init.xavier_uniform_(self.patch_query.weight, generator=generator)
            nn.init.xavier_uniform_(self.patch_key.weight, generator=generator)
        finally:
            torch.random.set_rng_state(rng_state)

        role_init = torch.zeros(ROLE_COUNT, dtype=torch.float32)
        role_init[0] = 0.16
        role_init[6] = 0.36
        self.raw_role_weights = nn.Parameter(
            torch.tensor([_atanh(float(v) / ROLE_BOUND) for v in role_init])
        )
        self.raw_beta = nn.Parameter(torch.tensor(_logit(float(initial_beta))))
        self.raw_alpha = nn.Parameter(torch.tensor(_logit(float(initial_alpha) / 2.0)))
        self.raw_delta = nn.Parameter(torch.tensor(_atanh(float(initial_delta))))

    def role_weights(self) -> torch.Tensor:
        return ROLE_BOUND * torch.tanh(self.raw_role_weights)

    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_beta)

    def alpha(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.raw_alpha)

    def delta(self) -> torch.Tensor:
        return torch.tanh(self.raw_delta)

    def semantic_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.raw_role_weights,)

    def visual_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            tuple(self.visual_adapter.parameters())
            + tuple(self.patch_query.parameters())
            + tuple(self.patch_key.parameters())
            + (self.raw_beta,)
        )

    def interaction_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.relation_reader.parameters()) + (
            self.raw_alpha,
            self.raw_delta,
        )

    def optimizer_parameter_groups(self) -> list[dict[str, object]]:
        return [
            {
                "name": "role_relation",
                "params": self.semantic_parameters()
                + tuple(self.relation_reader.parameters())
                + (self.raw_alpha,),
                "lr": 3e-6,
                "weight_decay": 1e-3,
            },
            {
                "name": "visual_interaction",
                "params": self.visual_parameters() + (self.raw_delta,),
                "lr": 3e-5,
                "weight_decay": 1e-3,
            },
        ]

    @staticmethod
    def _validate_cls(cls_features: torch.Tensor) -> torch.Tensor:
        if cls_features.ndim != 2 or tuple(cls_features.shape[1:]) != (EMBED_DIM,):
            raise ValueError("cls_features must be [batch,768].")
        if cls_features.size(0) == 0 or not torch.isfinite(cls_features).all():
            raise ValueError("cls_features must be non-empty and finite.")
        return cls_features.float()

    @staticmethod
    def _validate_patches(patch_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tuple(patch_features.shape) != (batch_size, PATCH_COUNT, EMBED_DIM):
            raise ValueError("patch_features must be [batch,36,768].")
        if not torch.isfinite(patch_features).all():
            raise ValueError("patch_features contains NaN or Inf.")
        return patch_features.float()

    def _validated_targets(self, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        target_ids = torch.as_tensor(targets).detach().long()
        if target_ids.ndim != 1 or target_ids.numel() != batch_size:
            raise ValueError("targets must be a 1D tensor matching the batch size.")
        if int(target_ids.min()) < 0 or int(target_ids.max()) >= CLASS_COUNT:
            raise ValueError("targets contains ids outside the 200-class axis.")
        if not bool(torch.isin(target_ids.cpu(), self.seen_classes.cpu()).all()):
            raise ValueError("ARRA losses use only official seen training labels.")
        return target_ids

    def semantic_directions(self, *, enabled: bool = True) -> torch.Tensor:
        weights = self.role_weights() if enabled else self.raw_role_weights.new_zeros(ROLE_COUNT)
        return self.class_prototypes.to(weights.device) + torch.einsum(
            "r,crd->cd",
            weights,
            self.role_sentence_embeds.to(weights.device),
        )

    def visual_readout(self, cls_features: torch.Tensor, *, enabled: bool = True) -> torch.Tensor:
        cls = self._validate_cls(cls_features)
        if not enabled:
            return F.normalize(cls, dim=-1)
        return F.normalize(cls + self.visual_adapter(cls), dim=-1)

    def relation_readout(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z + self.relation_reader(z), dim=-1)

    def affine_reference_logits(self, cls_features: torch.Tensor) -> torch.Tensor:
        """Exact beta=0/delta=0 receipt path with no patch contribution."""
        x = F.normalize(self._validate_cls(cls_features), dim=-1)
        semantic = (
            x @ self.semantic_directions(enabled=True).to(x.device).T
            * self.source_scale.to(x.device)
        )
        zr = self.relation_readout(x)
        relation = (
            self.alpha().to(x.device)
            * (zr @ self.compiled_relation_field.to(x.device).T)
            / RELATION_TEMPERATURE
        )
        return semantic + relation + self.seen_bias.to(x.device)

    def role_patch_evidence(
        self,
        patch_features: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self._validate_patches(patch_features, int(patch_features.size(0)))
        if not enabled:
            zeros = patches.new_zeros((patches.size(0), CLASS_COUNT))
            neutral = patches.new_full((patches.size(0), CLASS_COUNT), 0.5)
            attn = patches.new_full(
                (patches.size(0), CLASS_COUNT, ROLE_COUNT, PATCH_COUNT),
                1.0 / PATCH_COUNT,
            )
            return zeros, neutral, attn

        adapted = F.normalize(patches + self.visual_adapter(patches), dim=-1)
        roles = self.role_sentence_embeds.to(adapted.device)
        query = self.patch_query(roles)
        key = self.patch_key(adapted)
        attention_logits = torch.einsum("crh,bnh->bcrn", query, key)
        attention_logits = attention_logits / math.sqrt(float(HIDDEN_DIM))
        attention = F.softmax(attention_logits, dim=-1)
        similarities = torch.einsum("bnd,crd->bcrn", adapted, roles)
        role_evidence = (attention * similarities).sum(dim=-1)
        visual_logits = role_evidence.mean(dim=-1) * self.source_scale.to(adapted.device)
        centered = visual_logits - visual_logits.mean(dim=1, keepdim=True)
        std = visual_logits.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        support = torch.sigmoid((centered / std).clamp(-5.0, 5.0))
        return visual_logits, support, attention

    def components(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        *,
        condition: ARRACondition = "full",
        beta_override: float | None = None,
        alpha_override: float | None = None,
        delta_override: float | None = None,
        shuffle_indices: torch.Tensor | None = None,
    ) -> ARRAComponents:
        if condition not in {"full", "s_off", "v_off", "i_off", "additive", "shuffled"}:
            raise ValueError(f"unknown ARRA condition: {condition}")
        cls = self._validate_cls(cls_features)
        patches = self._validate_patches(patch_features, cls.size(0))
        visual_enabled = condition != "v_off"
        semantic_enabled = condition != "s_off"
        interaction_enabled = condition != "i_off"

        z = self.visual_readout(cls, enabled=visual_enabled)
        q_s = self.semantic_directions(enabled=semantic_enabled).to(z.device)
        semantic_logits = z @ q_s.T * self.source_scale.to(z.device)

        visual_logits, g, attention = self.role_patch_evidence(
            patches, enabled=visual_enabled
        )
        beta = (
            z.new_tensor(float(beta_override))
            if beta_override is not None
            else self.beta().to(z.device)
        )
        alpha = (
            z.new_tensor(float(alpha_override))
            if alpha_override is not None
            else self.alpha().to(z.device)
        )
        delta = (
            z.new_tensor(float(delta_override))
            if delta_override is not None
            else self.delta().to(z.device)
        )
        if condition == "additive":
            delta = z.new_zeros(())
        if condition == "shuffled":
            if shuffle_indices is None:
                shuffle_indices = torch.stack(
                    [torch.randperm(CLASS_COUNT, device=z.device) for _ in range(z.size(0))]
                )
            else:
                shuffle_indices = shuffle_indices.to(z.device).long()
                if tuple(shuffle_indices.shape) != (z.size(0), CLASS_COUNT):
                    raise ValueError("shuffle_indices must be [batch,200].")
            g = g.gather(1, shuffle_indices)

        zr = self.relation_readout(z)
        relation_base = (
            zr @ self.compiled_relation_field.to(zr.device).T / RELATION_TEMPERATURE
        )
        relation_logits = alpha * (1.0 + delta * (2.0 * g - 1.0)) * relation_base
        if not interaction_enabled:
            relation_logits = torch.zeros_like(relation_logits)

        bias = self.seen_bias.to(z.device)
        logits = semantic_logits + beta * visual_logits + relation_logits + bias
        return ARRAComponents(
            logits=logits,
            semantic_logits=semantic_logits,
            visual_logits=visual_logits,
            relation_logits=relation_logits,
            calibrated_bias=bias,
            z=z,
            zr=zr,
            g=g,
            attention=attention,
            beta=beta,
            alpha=alpha,
            delta=delta,
        )

    def forward(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        *,
        condition: ARRACondition = "full",
        class_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        logits = self.components(
            cls_features,
            patch_features,
            condition=condition,
            **kwargs,
        ).logits
        if class_ids is None:
            return logits
        ids = torch.as_tensor(class_ids, device=logits.device).long()
        if ids.ndim != 1 or ids.numel() == 0 or ids.unique().numel() != ids.numel():
            raise ValueError("class_ids must be a non-empty 1D tensor without duplicates.")
        if int(ids.min()) < 0 or int(ids.max()) >= CLASS_COUNT:
            raise ValueError("class_ids contains ids outside the 200-class axis.")
        return logits.index_select(1, ids)

    def classification_loss(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = self(cls_features, patch_features)
        target_ids = self._validated_targets(targets, logits.size(0)).to(logits.device)
        seen = self.seen_classes.to(logits.device)
        seen_logits = logits.index_select(1, seen)
        global_to_seen = torch.full((CLASS_COUNT,), -1, dtype=torch.long, device=logits.device)
        global_to_seen[seen] = torch.arange(seen.numel(), device=logits.device)
        return F.cross_entropy(seen_logits, global_to_seen[target_ids])

    def topology_loss(self) -> torch.Tensor:
        qn = F.normalize(self.semantic_directions(enabled=True), dim=-1)
        pn = F.normalize(self.class_prototypes.to(qn.device), dim=-1)
        mask = ~torch.eye(CLASS_COUNT, dtype=torch.bool, device=qn.device)
        qv = (qn @ qn.T)[mask]
        pv = (pn @ pn.T)[mask].detach()
        qv = qv - qv.mean()
        pv = pv - pv.mean()
        return 1.0 - (qv * pv).sum() / (
            torch.sqrt(qv.square().sum() + 1e-8)
            * torch.sqrt(pv.square().sum() + 1e-8)
        )

    def direction_loss(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        components = self.components(cls_features, patch_features)
        target_ids = self._validated_targets(targets, components.logits.size(0)).to(
            components.logits.device
        )
        edges = self.edge_index.to(components.logits.device)
        relations = F.normalize(
            self.relation_sentence_embeds.to(components.logits.device),
            dim=-1,
        )
        scores = torch.einsum("bd,ekd->bek", components.zr, relations)
        scores = scores / DIRECTION_TEMPERATURE
        seen = torch.zeros(CLASS_COUNT, dtype=torch.bool, device=components.logits.device)
        seen[self.seen_classes.to(components.logits.device)] = True
        seen_edge = seen[edges[:, 0]] & seen[edges[:, 1]]
        incident_a = target_ids[:, None].eq(edges[None, :, 0])
        incident_b = target_ids[:, None].eq(edges[None, :, 1])
        incident = (incident_a | incident_b) & seen_edge[None]
        counts = incident.sum(dim=1)
        if bool(counts.eq(0).any()):
            raise ValueError("each target must have at least one seen-seen incident edge.")
        labels = incident_b.long()
        edge_losses = -F.log_softmax(scores, dim=-1).gather(
            2, labels.unsqueeze(-1)
        ).squeeze(-1)
        return (
            (edge_losses * incident.to(edge_losses.dtype)).sum(dim=1)
            / counts.to(edge_losses.dtype)
        ).mean()

    def losses(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        targets: torch.Tensor,
        *,
        topology_weight: float = 0.3,
        direction_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        cls = self.classification_loss(cls_features, patch_features, targets)
        topology = self.topology_loss()
        direction = self.direction_loss(cls_features, patch_features, targets)
        return {
            "total": cls + float(topology_weight) * topology + float(direction_weight) * direction,
            "cls": cls,
            "topology": topology,
            "direction": direction,
        }

    @torch.no_grad()
    def export_graph_free(self) -> dict[str, object]:
        return {
            "schema_version": "gzsl-paper.v6-arra-graph-free.v1",
            "class_prototypes": self.class_prototypes.detach().cpu().clone(),
            "role_sentence_embeds": self.role_sentence_embeds.detach().cpu().clone(),
            "compiled_relation_field": self.compiled_relation_field.detach().cpu().clone(),
            "seen_classes": self.seen_classes.detach().cpu().clone(),
            "source_scale": self.source_scale.detach().cpu().clone(),
            "seen_logit_gamma": self.seen_logit_gamma,
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.state_dict().items()
                if not key.startswith("relation_sentence_embeds")
                and not key.startswith("edge_index")
                and not key.startswith("compiled_relation_field")
                and not key.startswith("class_prototypes")
                and not key.startswith("role_sentence_embeds")
                and not key.startswith("seen_classes")
                and not key.startswith("unseen_classes")
                and not key.startswith("source_scale")
                and not key.startswith("seen_bias")
            },
        }


class ARRAGraphFreeClassifier(nn.Module):
    """Deployment-only ARRA path; it has no edge or relation-text tensors."""

    def __init__(
        self,
        class_prototypes: torch.Tensor,
        role_sentence_embeds: torch.Tensor,
        compiled_relation_field: torch.Tensor,
        seen_classes: torch.Tensor,
        source_scale: torch.Tensor | float,
        *,
        seen_logit_gamma: float = SEEN_LOGIT_GAMMA,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "class_prototypes",
            _validate_matrix(class_prototypes, (CLASS_COUNT, EMBED_DIM), "class_prototypes"),
            persistent=True,
        )
        self.register_buffer(
            "role_sentence_embeds",
            _validate_matrix(
                role_sentence_embeds,
                (CLASS_COUNT, ROLE_COUNT, EMBED_DIM),
                "role_sentence_embeds",
            ),
            persistent=True,
        )
        self.register_buffer(
            "compiled_relation_field",
            _validate_matrix(
                compiled_relation_field,
                (CLASS_COUNT, EMBED_DIM),
                "compiled_relation_field",
            ),
            persistent=True,
        )
        seen = _validate_seen_classes(seen_classes)
        scale = torch.as_tensor(source_scale).detach().cpu().float().clone()
        self.register_buffer("seen_classes", seen, persistent=True)
        self.register_buffer("source_scale", scale.reshape(()), persistent=True)
        seen_bias = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        seen_bias[seen] = -float(seen_logit_gamma)
        self.register_buffer("seen_bias", seen_bias, persistent=True)
        self.visual_adapter = _ResidualAdapter()
        self.relation_reader = _ResidualAdapter()
        self.patch_query = nn.Linear(EMBED_DIM, HIDDEN_DIM, bias=False)
        self.patch_key = nn.Linear(EMBED_DIM, HIDDEN_DIM, bias=False)
        self.raw_role_weights = nn.Parameter(torch.zeros(ROLE_COUNT))
        self.raw_beta = nn.Parameter(torch.zeros(()))
        self.raw_alpha = nn.Parameter(torch.zeros(()))
        self.raw_delta = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_export(cls, payload: dict[str, object]) -> "ARRAGraphFreeClassifier":
        if payload.get("schema_version") != "gzsl-paper.v6-arra-graph-free.v1":
            raise ValueError("unexpected ARRA graph-free export schema.")
        forbidden = {"edge_index", "relation_sentence_embeds", "relation_embeddings"}
        if forbidden & set(payload):
            raise ValueError("graph-free export must not contain relation graph assets.")
        model = cls(
            payload["class_prototypes"],  # type: ignore[arg-type]
            payload["role_sentence_embeds"],  # type: ignore[arg-type]
            payload["compiled_relation_field"],  # type: ignore[arg-type]
            payload["seen_classes"],  # type: ignore[arg-type]
            payload["source_scale"],  # type: ignore[arg-type]
            seen_logit_gamma=float(payload["seen_logit_gamma"]),
        )
        missing, unexpected = model.load_state_dict(
            payload["state_dict"],  # type: ignore[arg-type]
            strict=False,
        )
        allowed_missing = {
            "class_prototypes",
            "role_sentence_embeds",
            "compiled_relation_field",
            "seen_classes",
            "source_scale",
            "seen_bias",
        }
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(
                f"invalid graph-free state_dict: missing={missing}, unexpected={unexpected}."
            )
        return model

    def role_weights(self) -> torch.Tensor:
        return ROLE_BOUND * torch.tanh(self.raw_role_weights)

    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_beta)

    def alpha(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.raw_alpha)

    def delta(self) -> torch.Tensor:
        return torch.tanh(self.raw_delta)

    def _role_patch_evidence(
        self,
        patch_features: torch.Tensor,
        *,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = ARRAClassifier._validate_patches(patch_features, int(patch_features.size(0)))
        if not enabled:
            zeros = patches.new_zeros((patches.size(0), CLASS_COUNT))
            neutral = patches.new_full((patches.size(0), CLASS_COUNT), 0.5)
            attn = patches.new_full(
                (patches.size(0), CLASS_COUNT, ROLE_COUNT, PATCH_COUNT),
                1.0 / PATCH_COUNT,
            )
            return zeros, neutral, attn
        adapted = F.normalize(patches + self.visual_adapter(patches), dim=-1)
        roles = self.role_sentence_embeds.to(adapted.device)
        attention_logits = torch.einsum(
            "crh,bnh->bcrn",
            self.patch_query(roles),
            self.patch_key(adapted),
        ) / math.sqrt(float(HIDDEN_DIM))
        attention = F.softmax(attention_logits, dim=-1)
        similarities = torch.einsum("bnd,crd->bcrn", adapted, roles)
        role_evidence = (attention * similarities).sum(dim=-1)
        visual_logits = role_evidence.mean(dim=-1) * self.source_scale.to(adapted.device)
        centered = visual_logits - visual_logits.mean(dim=1, keepdim=True)
        std = visual_logits.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        support = torch.sigmoid((centered / std).clamp(-5.0, 5.0))
        return visual_logits, support, attention

    def components(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        *,
        condition: ARRACondition = "full",
        shuffle_indices: torch.Tensor | None = None,
    ) -> ARRAComponents:
        if condition not in {"full", "s_off", "v_off", "i_off", "additive", "shuffled"}:
            raise ValueError(f"unknown ARRA condition: {condition}")
        cls = ARRAClassifier._validate_cls(cls_features)
        patches = ARRAClassifier._validate_patches(patch_features, cls.size(0))
        visual_enabled = condition != "v_off"
        semantic_enabled = condition != "s_off"
        interaction_enabled = condition != "i_off"
        z = F.normalize(
            cls + (self.visual_adapter(cls) if visual_enabled else 0.0),
            dim=-1,
        )
        weights = self.role_weights() if semantic_enabled else self.raw_role_weights.new_zeros(ROLE_COUNT)
        q_s = self.class_prototypes.to(z.device) + torch.einsum(
            "r,crd->cd",
            weights,
            self.role_sentence_embeds.to(z.device),
        )
        semantic_logits = z @ q_s.T * self.source_scale.to(z.device)
        visual_logits, g, attention = self._role_patch_evidence(
            patches, enabled=visual_enabled
        )
        delta = self.delta().to(z.device)
        if condition == "additive":
            delta = z.new_zeros(())
        if condition == "shuffled":
            if shuffle_indices is None:
                shuffle_indices = torch.stack(
                    [torch.randperm(CLASS_COUNT, device=z.device) for _ in range(z.size(0))]
                )
            else:
                shuffle_indices = shuffle_indices.to(z.device).long()
            g = g.gather(1, shuffle_indices)
        zr = F.normalize(z + self.relation_reader(z), dim=-1)
        relation_base = zr @ self.compiled_relation_field.to(z.device).T / RELATION_TEMPERATURE
        relation_logits = self.alpha().to(z.device) * (1.0 + delta * (2.0 * g - 1.0)) * relation_base
        if not interaction_enabled:
            relation_logits = torch.zeros_like(relation_logits)
        logits = semantic_logits + self.beta().to(z.device) * visual_logits + relation_logits
        logits = logits + self.seen_bias.to(z.device)
        return ARRAComponents(
            logits=logits,
            semantic_logits=semantic_logits,
            visual_logits=visual_logits,
            relation_logits=relation_logits,
            calibrated_bias=self.seen_bias.to(z.device),
            z=z,
            zr=zr,
            g=g,
            attention=attention,
            beta=self.beta().to(z.device),
            alpha=self.alpha().to(z.device),
            delta=delta,
        )

    def forward(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        *,
        condition: ARRACondition = "full",
        class_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        logits = self.components(
            cls_features,
            patch_features,
            condition=condition,
            **kwargs,
        ).logits
        if class_ids is None:
            return logits
        ids = torch.as_tensor(class_ids, device=logits.device).long()
        if ids.ndim != 1 or ids.numel() == 0 or ids.unique().numel() != ids.numel():
            raise ValueError("class_ids must be a non-empty 1D tensor without duplicates.")
        if int(ids.min()) < 0 or int(ids.max()) >= CLASS_COUNT:
            raise ValueError("class_ids contains ids outside the 200-class axis.")
        return logits.index_select(1, ids)
