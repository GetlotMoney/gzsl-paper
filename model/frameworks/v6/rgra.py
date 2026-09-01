"""Role-Grounded Relation Alignment (RGRA)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EMBED_DIM = 768
ROLE_COUNT = 8
GROUP_COUNT = 3
PATCH_COUNT = 36
RGRA_CONDITIONS = {"full", "s_off", "v_off", "i_off", "additive", "shuffled"}


def _logit_from_bounded(value: float, maximum: float) -> torch.Tensor:
    ratio = float(value) / float(maximum)
    if not 0.0 < ratio < 1.0:
        raise ValueError("bounded scalar initialization must be inside (0, maximum).")
    return torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)


def raw_role_queries(role_sentence_embeds: torch.Tensor) -> torch.Tensor:
    if role_sentence_embeds.ndim != 3 or tuple(role_sentence_embeds.shape[1:]) != (
        ROLE_COUNT,
        EMBED_DIM,
    ):
        raise ValueError("role_sentence_embeds must be [class,8,768].")
    if not torch.isfinite(role_sentence_embeds).all():
        raise ValueError("role_sentence_embeds contains NaN/Inf.")
    roles = role_sentence_embeds.float()
    return F.normalize(
        torch.stack((roles[:, :6].mean(dim=1), roles[:, 6], roles[:, 7]), dim=1),
        dim=-1,
    )


def raw_role_groups(role_sentence_embeds: torch.Tensor) -> torch.Tensor:
    return raw_role_queries(role_sentence_embeds)


def mean8_prototypes(role_sentence_embeds: torch.Tensor) -> torch.Tensor:
    if role_sentence_embeds.ndim != 3 or tuple(role_sentence_embeds.shape[1:]) != (
        ROLE_COUNT,
        EMBED_DIM,
    ):
        raise ValueError("role_sentence_embeds must be [class,8,768].")
    return F.normalize(role_sentence_embeds.float().mean(dim=1), dim=-1)


def build_relation_field(
    relation_directions_or_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    ridge: float = 0.3,
    class_count: int = 200,
) -> torch.Tensor:
    edges = torch.as_tensor(edge_index).detach().cpu().long()
    values = torch.as_tensor(relation_directions_or_embeddings).detach().cpu().float()
    if values.ndim == 3:
        values = F.normalize(values[:, 0] - values[:, 1], dim=-1)
    if tuple(values.shape) != (edges.size(0), EMBED_DIM):
        raise ValueError("relation directions must be [edge,768] or [edge,2,768].")
    incidence = torch.zeros(edges.size(0), int(class_count), dtype=torch.float32)
    rows = torch.arange(edges.size(0))
    incidence[rows, edges[:, 0]] = 1.0
    incidence[rows, edges[:, 1]] = -1.0
    lhs = incidence.T @ incidence + float(ridge) * torch.eye(int(class_count))
    return F.normalize(torch.linalg.solve(lhs, incidence.T) @ values, dim=-1)


class BottleneckResidual(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.gelu(self.in_proj(value.float())))


class RoleSemanticComposer(nn.Module):
    def __init__(
        self,
        role_sentence_embeds: torch.Tensor,
        p_v5: torch.Tensor,
        *,
        hidden_dim: int,
        max_rho: float,
        initial_rho: float,
    ) -> None:
        super().__init__()
        self.max_rho = float(max_rho)
        self.register_buffer(
            "role_sentence_embeds",
            F.normalize(role_sentence_embeds.float(), dim=-1),
            persistent=True,
        )
        self.register_buffer("q_raw", raw_role_queries(role_sentence_embeds), persistent=True)
        self.register_buffer("p_mean8", mean8_prototypes(role_sentence_embeds), persistent=True)
        self.register_buffer("p_v5", F.normalize(p_v5.float(), dim=-1), persistent=True)
        self.adapter = BottleneckResidual(EMBED_DIM, hidden_dim)
        self.group_logits = nn.Parameter(torch.zeros(GROUP_COUNT))
        self.raw_rho = nn.Parameter(_logit_from_bounded(initial_rho, max_rho))

    def rho(self) -> torch.Tensor:
        return self.max_rho * torch.sigmoid(self.raw_rho)

    def group_weights(self, *, s_off: bool = False) -> torch.Tensor:
        if s_off:
            return self.q_raw.new_full((GROUP_COUNT,), 1.0 / GROUP_COUNT)
        return F.softmax(self.group_logits, dim=0)

    def role_queries(self, *, s_off: bool = False) -> torch.Tensor:
        if s_off:
            return self.q_raw
        return F.normalize(self.q_raw + self.adapter(self.q_raw), dim=-1)

    def prototypes(self, *, s_off: bool = False) -> torch.Tensor:
        if s_off:
            return self.p_mean8
        q = self.role_queries(s_off=False)
        grouped = F.normalize(
            (self.group_weights(s_off=False).view(1, GROUP_COUNT, 1) * q).sum(dim=1),
            dim=-1,
        )
        return F.normalize((1.0 - self.rho()) * self.p_v5 + self.rho() * grouped, dim=-1)


class RoleVisualAligner(nn.Module):
    def __init__(self, *, hidden_dim: int, max_beta: float, initial_beta: float) -> None:
        super().__init__()
        self.max_beta = float(max_beta)
        self.adapter = BottleneckResidual(EMBED_DIM, hidden_dim)
        self.query_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.key_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.key_proj.weight)
        self.raw_beta = nn.Parameter(_logit_from_bounded(initial_beta, max_beta))

    def beta(self) -> torch.Tensor:
        return self.max_beta * torch.sigmoid(self.raw_beta)


class RelationFieldMatcher(nn.Module):
    def __init__(
        self,
        relation_field: torch.Tensor,
        *,
        hidden_dim: int,
        max_alpha: float,
        initial_alpha: float,
    ) -> None:
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.reader = BottleneckResidual(EMBED_DIM, hidden_dim)
        self.raw_alpha = nn.Parameter(_logit_from_bounded(initial_alpha, max_alpha))
        self.register_buffer("relation_field", F.normalize(relation_field.float(), dim=-1))

    def alpha(self) -> torch.Tensor:
        return self.max_alpha * torch.sigmoid(self.raw_alpha)


class RGRAModel(nn.Module):
    """Three-module graph-free classifier trained through one final logits path."""

    def __init__(
        self,
        role_sentence_embeds: torch.Tensor,
        p_v5_or_seen_classes: torch.Tensor,
        relation_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        seen_classes: torch.Tensor | None = None,
        *,
        p_v5: torch.Tensor | None = None,
        class_count: int = 200,
        hidden_dim: int = 64,
        relation_ridge: float = 0.3,
        visual_temperature: float = 0.07,
        relation_temperature: float = 0.2,
        seen_logit_gamma: float = 0.91,
        max_rho_s: float = 0.5,
        initial_rho_s: float = 0.10,
        max_beta_v: float = 1.0,
        initial_beta_v: float = 0.10,
        max_alpha: float = 1.0,
        initial_alpha: float = 0.05,
        scale: torch.Tensor | float | None = None,
        reader_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if seen_classes is None:
            seen_input = p_v5_or_seen_classes
            p_v5_input = p_v5
        else:
            seen_input = seen_classes
            p_v5_input = p_v5_or_seen_classes
        if p_v5_input is None:
            p_v5_input = mean8_prototypes(role_sentence_embeds)
        if int(class_count) != int(role_sentence_embeds.size(0)):
            raise ValueError("class_count must match role_sentence_embeds.")
        seen = torch.as_tensor(seen_input).detach().cpu().long().sort().values
        if seen.ndim != 1 or seen.numel() == 0 or seen.unique().numel() != seen.numel():
            raise ValueError("seen_classes must be unique and non-empty.")
        if int(seen.min()) < 0 or int(seen.max()) >= int(class_count):
            raise ValueError("seen_classes out of class axis.")
        unseen = torch.arange(int(class_count))[~torch.isin(torch.arange(int(class_count)), seen)]
        edges = torch.as_tensor(edge_index).detach().cpu().long()
        relations = torch.as_tensor(relation_embeddings).detach().cpu().float()
        if edges.ndim != 2 or edges.size(1) != 2 or edges.size(0) == 0:
            raise ValueError("edge_index must be [edge,2].")
        if int(edges.min()) < 0 or int(edges.max()) >= int(class_count):
            raise ValueError("edge_index out of class axis.")
        if tuple(relations.shape) not in ((edges.size(0), 2, EMBED_DIM), (edges.size(0), EMBED_DIM)):
            raise ValueError("relation embeddings/directions shape mismatch.")
        relation_field = build_relation_field(relations, edges, relation_ridge, int(class_count))

        self.class_count = int(class_count)
        self.visual_temperature = float(visual_temperature)
        self.relation_temperature = float(relation_temperature)
        self.seen_logit_gamma = float(seen_logit_gamma)
        self.register_buffer("seen_classes", seen, persistent=True)
        self.register_buffer("unseen_classes", unseen, persistent=True)
        self.register_buffer("relation_embeddings", F.normalize(relations, dim=-1), persistent=False)
        self.register_buffer("edge_index", edges, persistent=False)
        self.rsc = RoleSemanticComposer(
            role_sentence_embeds,
            torch.as_tensor(p_v5_input).float(),
            hidden_dim=hidden_dim,
            max_rho=max_rho_s,
            initial_rho=initial_rho_s,
        )
        self.rva = RoleVisualAligner(
            hidden_dim=hidden_dim, max_beta=max_beta_v, initial_beta=initial_beta_v
        )
        self.rfm = RelationFieldMatcher(
            relation_field,
            hidden_dim=hidden_dim,
            max_alpha=max_alpha,
            initial_alpha=initial_alpha,
        )
        initial_scale = 1.0 / 0.07 if scale is None else float(torch.as_tensor(scale))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(initial_scale)))
        if reader_state_dict is not None:
            prefixed = {key.removeprefix("reader_").removeprefix("reader."): value for key, value in reader_state_dict.items()}
            translated = {
                key.replace("reader_in.", "in_proj.").replace("reader_out.", "out_proj."): value
                for key, value in prefixed.items()
            }
            self.rfm.reader.load_state_dict(translated, strict=False)

    @property
    def raw_alpha(self) -> torch.nn.Parameter:
        return self.rfm.raw_alpha

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def rho_s(self) -> torch.Tensor:
        return self.rsc.rho()

    def beta_v(self) -> torch.Tensor:
        return self.rva.beta()

    def alpha(self) -> torch.Tensor:
        return self.rfm.alpha()

    def semantic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.rsc.parameters()) + (self.logit_scale,)

    def visual_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.rva.parameters())

    def interaction_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.rfm.parameters())

    def training_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {
            "semantic": self.semantic_parameters(),
            "visual": self.visual_parameters(),
            "interaction": self.interaction_parameters(),
        }

    def semantic_queries(self, *, enabled: bool = True) -> torch.Tensor:
        return self.rsc.role_queries(s_off=not enabled)

    def prototypes(self, *, enabled: bool = True) -> torch.Tensor:
        return self.rsc.prototypes(s_off=not enabled)

    @staticmethod
    def _validate_inputs(
        cls_features: torch.Tensor, patch_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cls_features.ndim != 2 or cls_features.size(1) != EMBED_DIM:
            raise ValueError("cls_features must be [batch,768].")
        if (
            patch_features.ndim != 3
            or patch_features.size(0) != cls_features.size(0)
            or tuple(patch_features.shape[1:]) != (PATCH_COUNT, EMBED_DIM)
        ):
            raise ValueError("patch_features must be [batch,36,768].")
        if cls_features.size(0) == 0:
            raise ValueError("empty batch.")
        if not torch.isfinite(cls_features).all() or not torch.isfinite(patch_features).all():
            raise ValueError("RGRA inputs contain NaN/Inf.")
        return cls_features.float(), patch_features.float()

    def _visual_terms(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        q: torch.Tensor,
        *,
        enabled: bool,
    ) -> dict[str, torch.Tensor]:
        cls_features, patch_features = self._validate_inputs(cls_features, patch_features)
        if not enabled:
            z = F.normalize(cls_features, dim=-1)
            zeros = cls_features.new_zeros((cls_features.size(0), self.class_count))
            neutral = cls_features.new_full((cls_features.size(0), self.class_count), 0.5)
            uniform = cls_features.new_full(
                (cls_features.size(0), self.class_count, GROUP_COUNT, PATCH_COUNT),
                1.0 / PATCH_COUNT,
            )
            return {"z": z, "l_v": zeros, "support_gate": neutral, "attention": uniform}
        z = F.normalize(cls_features + self.rva.adapter(cls_features), dim=-1)
        queries = F.normalize(self.rva.query_proj(q), dim=-1)
        keys = F.normalize(self.rva.key_proj(patch_features), dim=-1)
        attention = F.softmax(
            torch.einsum("cgd,bnd->bcgn", queries, keys) / self.visual_temperature,
            dim=-1,
        )
        support_vectors = torch.einsum(
            "bcgn,bnd->bcgd", attention, F.normalize(patch_features, dim=-1)
        )
        role_scores = (support_vectors * q.unsqueeze(0)).sum(dim=-1) * self.scale()
        l_v = (self.rsc.group_weights().view(1, 1, GROUP_COUNT) * role_scores).sum(dim=-1)
        centered = l_v - l_v.mean(dim=1, keepdim=True)
        standardized = centered / l_v.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return {
            "z": z,
            "l_v": l_v,
            "support_gate": torch.sigmoid(standardized.clamp(-5.0, 5.0)),
            "attention": attention,
        }

    def _relation_score(self, z: torch.Tensor) -> torch.Tensor:
        z_r = F.normalize(z + self.rfm.reader(z), dim=-1)
        return z_r @ self.rfm.relation_field.to(z.device).T / self.relation_temperature

    def score_components(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        *,
        condition: str = "full",
        alpha_override: float | None = None,
        shuffle_seed: int = 7,
    ) -> dict[str, torch.Tensor]:
        if condition not in RGRA_CONDITIONS:
            raise ValueError(f"unknown RGRA condition: {condition}")
        q = self.rsc.role_queries(s_off=condition == "s_off")
        proto = self.rsc.prototypes(s_off=condition == "s_off")
        visual = self._visual_terms(
            cls_features, patch_features, q, enabled=condition != "v_off"
        )
        semantic_logits = visual["z"] @ proto.to(visual["z"].device).T * self.scale()
        visual_logits = visual["l_v"]
        relation_scores = self._relation_score(visual["z"])
        if condition == "additive":
            support_for_relation = torch.full_like(visual["support_gate"], 0.5)
        elif condition == "shuffled":
            generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
            perm = torch.randperm(self.class_count, generator=generator).to(
                visual["support_gate"].device
            )
            support_for_relation = visual["support_gate"].index_select(1, perm)
        else:
            support_for_relation = visual["support_gate"]
        alpha = self.alpha() if alpha_override is None else relation_scores.new_tensor(float(alpha_override))
        interaction_logits = (
            torch.zeros_like(relation_scores)
            if condition == "i_off"
            else alpha * support_for_relation * relation_scores
        )
        logits = semantic_logits + self.rva.beta() * visual_logits + interaction_logits
        logits[:, self.seen_classes.to(logits.device)] -= self.seen_logit_gamma
        return {
            "logits": logits,
            "semantic_logits": semantic_logits,
            "visual_logits": visual_logits,
            "interaction_logits": interaction_logits,
            "attention": visual["attention"],
            "support_gate": visual["support_gate"],
            "support_for_relation": support_for_relation,
            "relation_scores": relation_scores,
        }

    def logits(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        *,
        condition: str = "full",
        mode: str | None = None,
        alpha_override: float | None = None,
        shuffle_seed: int = 7,
    ) -> torch.Tensor:
        active = condition if mode is None else mode
        logits = self.score_components(
            cls_features,
            patch_features,
            condition=active,
            alpha_override=alpha_override,
            shuffle_seed=shuffle_seed,
        )["logits"]
        if class_ids is not None:
            logits = logits.index_select(1, class_ids.to(logits.device).long())
        return logits

    def forward(self, cls_features: torch.Tensor, patch_features: torch.Tensor) -> torch.Tensor:
        return self.logits(cls_features, patch_features)

    def _local_seen_targets(self, targets: torch.Tensor) -> torch.Tensor:
        targets_cpu = targets.detach().cpu().long()
        if not bool(torch.isin(targets_cpu, self.seen_classes.cpu()).all()):
            raise ValueError("RGRA classification_loss requires seen-class labels only.")
        global_to_seen = torch.full((self.class_count,), -1, dtype=torch.long)
        global_to_seen[self.seen_classes.cpu()] = torch.arange(self.seen_classes.numel())
        return global_to_seen.index_select(0, targets_cpu).to(targets.device)

    def classification_loss(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        seen = self.seen_classes.to(cls_features.device)
        return F.cross_entropy(
            self.logits(cls_features, patch_features).index_select(1, seen),
            self._local_seen_targets(targets).to(cls_features.device),
        )

    def topology_loss(self) -> torch.Tensor:
        base = F.normalize(self.rsc.p_mean8, dim=-1)
        current = self.rsc.prototypes(s_off=False)
        mask = ~torch.eye(self.class_count, dtype=torch.bool, device=current.device)
        x = (base @ base.T).detach()[mask]
        y = (current @ current.T)[mask]
        x = x - x.mean()
        y = y - y.mean()
        return 1.0 - (x * y).sum() / (
            torch.sqrt(x.square().sum() + 1e-8) * torch.sqrt(y.square().sum() + 1e-8)
        )

    def direction_loss(self, cls_features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        z = F.normalize(cls_features.float(), dim=-1)
        z_r = F.normalize(z + self.rfm.reader(z), dim=-1)
        relations = self.relation_embeddings.to(z_r.device)
        if relations.ndim == 2:
            relations = torch.stack((relations, -relations), dim=1)
        scores = torch.einsum("bd,ekd->bek", z_r, relations) / 0.07
        edges = self.edge_index.to(targets.device)
        left = targets[:, None].eq(edges[:, 0].view(1, -1))
        right = targets[:, None].eq(edges[:, 1].view(1, -1))
        mask = left | right
        if not bool(mask.any()):
            return cls_features.new_zeros(())
        labels = torch.where(
            left[mask],
            torch.zeros((), dtype=torch.long, device=targets.device),
            torch.ones((), dtype=torch.long, device=targets.device),
        )
        return F.cross_entropy(scores[mask], labels)

    def total_loss(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        targets: torch.Tensor,
        *,
        topology_weight: float = 0.3,
        direction_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        classification = self.classification_loss(cls_features, patch_features, targets)
        topology = self.topology_loss()
        direction = self.direction_loss(cls_features, targets.to(cls_features.device))
        return classification + float(topology_weight) * topology + float(direction_weight) * direction, {
            "classification_loss": classification,
            "topology_loss": topology,
            "direction_loss": direction,
        }

    def classification_gradient_norms(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        self.zero_grad(set_to_none=True)
        loss = self.classification_loss(cls_features, patch_features, targets)
        loss.backward()
        groups = {
            "rsc": self.semantic_parameters(),
            "rva": self.visual_parameters(),
            "rfm": self.interaction_parameters(),
        }
        result = {"classification_loss": float(loss.detach())}
        for name, params in groups.items():
            total = cls_features.new_zeros(())
            for param in params:
                if param.grad is not None:
                    total = total + param.grad.detach().float().norm()
            result[name] = float(total)
        return result

    @torch.no_grad()
    def export_classifier(self) -> dict[str, torch.Tensor]:
        return {
            "prototypes": self.rsc.prototypes(s_off=False).detach().cpu(),
            "relation_field": self.rfm.relation_field.detach().cpu(),
            "seen_classes": self.seen_classes.detach().cpu(),
            "group_weights": self.rsc.group_weights().detach().cpu(),
            "rho_s": self.rho_s().detach().cpu(),
            "beta_v": self.beta_v().detach().cpu(),
            "alpha": self.alpha().detach().cpu(),
            "seen_logit_gamma": torch.tensor(self.seen_logit_gamma),
        }

    @torch.no_grad()
    def export_graph_free_state(self) -> dict:
        return {
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.state_dict().items()
                if key
                not in {"relation_embeddings", "edge_index", "incidence", "laplacian_map"}
            },
            "metadata": self.export_classifier(),
        }
