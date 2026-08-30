"""Pairwise Contrastive Laplacian Reasoning (PCLR)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.gtd_tst import GTDTSTModel


RELATION_EDGE_COUNT = 438
EMBED_DIM = 768
READER_HIDDEN_DIM = 64
READER_SEED = 18601


class PCLRModel(GTDTSTModel):
    """GTD-TST plus frozen pairwise text evidence and graph potentials.

    The inherited :meth:`logits` method remains the exact TG+GTD training
    path.  :meth:`pclr_logits` is the deployed full-method path.  This split
    makes the three optimizer boundaries explicit: parent losses train only
    the inherited model, ``relation_loss`` trains only the shared reader, and
    ``beta_loss`` trains only ``raw_beta``.
    """

    def __init__(
        self,
        parent: nn.Module,
        seen_classes: torch.Tensor,
        relation_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        class_count: int = 200,
        hidden_dim: int = 16,
        max_transport_step: float = 1.5,
        grid_points: int = 33,
        reader_hidden_dim: int = READER_HIDDEN_DIM,
        reader_seed: int = READER_SEED,
        temperature: float = 0.07,
        ridge_lambda: float = 1.0,
        potential_cap: float = 0.5,
        max_beta: float = 0.25,
        initial_beta: float = 0.05,
        candidate_top_k: int | None = None,
        correction_scale: float = 1.0,
        seen_logit_gamma: float = 0.0,
    ) -> None:
        # The inherited GTD gate must consume exactly the same RNG draws as the
        # RUN-030 parent.  Only PCLR's extra reader is made RNG-neutral below.
        super().__init__(
            parent,
            seen_classes,
            class_count=class_count,
            hidden_dim=hidden_dim,
            max_transport_step=max_transport_step,
            grid_points=grid_points,
        )
        if int(class_count) != 200:
            raise ValueError("PCLR首轮固定200类全局类别轴。")
        if int(reader_hidden_dim) != READER_HIDDEN_DIM:
            raise ValueError("PCLR首轮固定reader hidden_dim=64。")
        if int(reader_seed) != READER_SEED:
            raise ValueError("PCLR首轮固定reader seed=18601。")
        if float(temperature) != 0.07:
            raise ValueError("PCLR首轮固定关系温度0.07。")
        if not math.isfinite(float(ridge_lambda)) or float(ridge_lambda) <= 0.0:
            raise ValueError("PCLR ridge_lambda必须为有限正数。")
        if float(potential_cap) <= 0.0:
            raise ValueError("PCLR potential_cap必须为正数。")
        if float(max_beta) != 0.25 or not 0.0 < float(initial_beta) < float(max_beta):
            raise ValueError("PCLR首轮固定max_beta=0.25且初始beta必须位于其内部。")
        if candidate_top_k is not None and not 2 <= int(candidate_top_k) < int(class_count):
            raise ValueError("Local-PCLR candidate_top_k必须位于[2,class_count)内。")
        if not math.isfinite(float(correction_scale)) or float(correction_scale) <= 0.0:
            raise ValueError("PCLR correction_scale必须为有限正数。")
        if not math.isfinite(float(seen_logit_gamma)) or float(seen_logit_gamma) < 0.0:
            raise ValueError("PCLR seen_logit_gamma必须为有限非负数。")

        relations = torch.as_tensor(relation_embeddings).detach().cpu().float().clone()
        edges = torch.as_tensor(edge_index).detach().cpu().long().clone()
        if tuple(relations.shape) != (RELATION_EDGE_COUNT, 2, EMBED_DIM):
            raise ValueError("PCLR关系embedding必须是[438,2,768]。")
        if tuple(edges.shape) != (RELATION_EDGE_COUNT, 2):
            raise ValueError("PCLR edge_index必须是[438,2]。")
        if not torch.isfinite(relations).all():
            raise ValueError("PCLR关系embedding包含NaN/Inf。")
        relation_norms = relations.norm(dim=-1)
        if not torch.allclose(
            relation_norms,
            torch.ones_like(relation_norms),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("PCLR关系embedding必须逐句L2归一化。")
        if int(edges.min()) < 0 or int(edges.max()) >= int(class_count):
            raise ValueError("PCLR边端点超出200类全局轴。")
        if bool(edges[:, 0].eq(edges[:, 1]).any()):
            raise ValueError("PCLR关系图不允许自环。")
        canonical_edges = edges.sort(dim=1).values
        if canonical_edges.unique(dim=0).size(0) != RELATION_EDGE_COUNT:
            raise ValueError("PCLR关系图包含重复无向边。")
        if not torch.equal(edges, canonical_edges):
            raise ValueError("PCLR edge_index每条边必须按a_id<b_id固定方向。")

        self.register_buffer("relation_embeddings", relations, persistent=True)
        self.register_buffer("edge_index", edges, persistent=True)
        self.temperature = float(temperature)
        self.ridge_lambda = float(ridge_lambda)
        self.potential_cap = float(potential_cap)
        self.max_beta = float(max_beta)
        self.candidate_top_k = (
            None if candidate_top_k is None else int(candidate_top_k)
        )
        self.correction_scale = float(correction_scale)
        self.seen_logit_gamma = float(seen_logit_gamma)

        # nn.Linear constructors consume the global CPU generator even though
        # both layers are immediately overwritten.  Restore only these extra
        # draws; W1 uses its own fixed generator and does not perturb RUN-030.
        reader_rng_state = torch.random.get_rng_state()
        try:
            self.reader_in = nn.Linear(EMBED_DIM, READER_HIDDEN_DIM)
            self.reader_out = nn.Linear(READER_HIDDEN_DIM, EMBED_DIM)
            reader_generator = torch.Generator(device="cpu").manual_seed(READER_SEED)
            nn.init.xavier_uniform_(self.reader_in.weight, generator=reader_generator)
            nn.init.zeros_(self.reader_in.bias)
            nn.init.zeros_(self.reader_out.weight)
            nn.init.zeros_(self.reader_out.bias)
        finally:
            torch.random.set_rng_state(reader_rng_state)

        raw_beta = math.log(float(initial_beta) / (float(max_beta) - float(initial_beta)))
        self.raw_beta = nn.Parameter(torch.tensor(raw_beta, dtype=torch.float32))

        incidence = torch.zeros(
            RELATION_EDGE_COUNT, int(class_count), dtype=torch.float32
        )
        rows = torch.arange(RELATION_EDGE_COUNT)
        incidence[rows, edges[:, 0]] = 1.0
        incidence[rows, edges[:, 1]] = -1.0
        regularized_laplacian = incidence.T @ incidence + self.ridge_lambda * torch.eye(
            int(class_count), dtype=torch.float32
        )
        laplacian_map = torch.linalg.solve(
            regularized_laplacian, incidence.T
        )
        self.register_buffer("incidence", incidence, persistent=True)
        self.register_buffer("laplacian_map", laplacian_map, persistent=True)

        seen_mask = torch.zeros(int(class_count), dtype=torch.bool)
        seen_mask[self.seen_classes] = True
        seen_edges = seen_mask.index_select(0, edges[:, 0]) & seen_mask.index_select(
            0, edges[:, 1]
        )
        self.register_buffer("seen_edge_mask", seen_edges, persistent=True)
        seen_degrees = torch.zeros(int(class_count), dtype=torch.long)
        for endpoint in (0, 1):
            seen_degrees.scatter_add_(
                0,
                edges[:, endpoint],
                seen_edges.long(),
            )
        if bool(seen_degrees.index_select(0, self.seen_classes).eq(0).any()):
            raise ValueError("PCLR每个seen类必须至少连接一条seen-seen边。")
        self.register_buffer("seen_incident_degree", seen_degrees, persistent=True)

    def beta(self) -> torch.Tensor:
        """Bounded global correction coefficient."""
        return self.max_beta * torch.sigmoid(self.raw_beta)

    def parent_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters owned by the unchanged TG+GTD parent objective."""
        return tuple(self.parent.parameters()) + tuple(self.gate.parameters())

    def relation_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters owned exclusively by the directional relation loss."""
        return tuple(self.reader_in.parameters()) + tuple(self.reader_out.parameters())

    def reader_parameters(self) -> tuple[nn.Parameter, ...]:
        """Alias used by trainers when naming the relation-reader optimizer."""
        return self.relation_parameters()

    def beta_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters owned exclusively by the detached beta objective."""
        return (self.raw_beta,)

    def training_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {
            "parent": self.parent_parameters(),
            "relation": self.relation_parameters(),
            "beta": self.beta_parameters(),
        }

    @staticmethod
    def _validated_images(image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 2 or image_features.size(1) != EMBED_DIM:
            raise ValueError("PCLR图像特征必须是[batch,768]。")
        if image_features.size(0) == 0 or not torch.isfinite(image_features).all():
            raise ValueError("PCLR图像batch必须非空且不含NaN/Inf。")
        return image_features

    def read_images(self, image_features: torch.Tensor) -> torch.Tensor:
        """Read frozen CLS features without allowing gradients into the backbone."""
        images = self._validated_images(image_features).detach().float()
        residual = self.reader_out(F.gelu(self.reader_in(images)))
        return F.normalize(images + residual, dim=-1)

    def relation_scores(self, image_features: torch.Tensor) -> torch.Tensor:
        """Return directional edge logits with shape ``[batch,438,2]``."""
        readout = self.read_images(image_features)
        relations = self.relation_embeddings.to(readout.device)
        return torch.einsum("bd,ekd->bek", readout, relations) / self.temperature

    def deployed_parent_logits(self, image_features: torch.Tensor) -> torch.Tensor:
        """Return the canonical normalized-prototype evaluation logits."""
        images = F.normalize(self._validated_images(image_features).float(), dim=-1)
        prototypes = F.normalize(self.prototypes().float(), dim=-1)
        logits = images @ prototypes.T * self.scale()
        if tuple(logits.shape) != (images.size(0), self.class_count):
            raise ValueError("PCLR parent logits shape错误。")
        return logits

    def candidate_edge_mask(self, parent_logits: torch.Tensor) -> torch.Tensor:
        """Select edges whose two endpoints are both in the detached Parent Top-K."""
        if self.candidate_top_k is None:
            return torch.ones(
                parent_logits.size(0),
                RELATION_EDGE_COUNT,
                dtype=torch.bool,
                device=parent_logits.device,
            )
        if (
            parent_logits.ndim != 2
            or tuple(parent_logits.shape[1:]) != (self.class_count,)
            or not torch.isfinite(parent_logits).all()
        ):
            raise ValueError("Local-PCLR parent logits必须是有限的[batch,200]。")
        candidate_ids = parent_logits.detach().topk(
            self.candidate_top_k, dim=1
        ).indices
        selected = torch.zeros_like(parent_logits, dtype=torch.bool)
        selected.scatter_(1, candidate_ids, True)
        edges = self.edge_index.to(parent_logits.device)
        return selected[:, edges[:, 0]] & selected[:, edges[:, 1]]

    def potentials_from_scores(
        self,
        relation_scores: torch.Tensor,
        parent_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve the fixed regularized graph inverse and bound node potentials."""
        if (
            relation_scores.ndim != 3
            or tuple(relation_scores.shape[1:]) != (RELATION_EDGE_COUNT, 2)
            or relation_scores.size(0) == 0
            or not torch.isfinite(relation_scores).all()
        ):
            raise ValueError("PCLR关系分数必须是有限的[batch,438,2]。")
        edge_difference = relation_scores[..., 0] - relation_scores[..., 1]
        if self.candidate_top_k is not None:
            if parent_logits is None:
                raise ValueError("Local-PCLR势能必须绑定Parent Top-K logits。")
            mask = self.candidate_edge_mask(parent_logits).to(edge_difference.dtype)
            edge_difference = edge_difference * mask
        potential = edge_difference @ self.laplacian_map.to(
            device=edge_difference.device, dtype=edge_difference.dtype
        ).T
        potential = potential - potential.mean(dim=1, keepdim=True)
        infinity_norm = potential.abs().amax(dim=1, keepdim=True)
        denominator = torch.maximum(
            infinity_norm,
            potential.new_full(infinity_norm.shape, self.potential_cap),
        )
        return self.potential_cap * potential / denominator

    def potentials(
        self,
        image_features: torch.Tensor,
        parent_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.potentials_from_scores(
            self.relation_scores(image_features), parent_logits
        )

    def _validated_class_ids(
        self,
        class_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        ids = torch.as_tensor(class_ids).detach().long()
        if (
            ids.ndim != 1
            or ids.numel() == 0
            or ids.unique().numel() != ids.numel()
            or int(ids.min()) < 0
            or int(ids.max()) >= self.class_count
        ):
            raise ValueError("PCLR class_ids必须是合法且无重复的一维全局类别ID。")
        return ids.to(device)

    def apply_seen_calibration(self, full_logits: torch.Tensor) -> torch.Tensor:
        """Subtract the fixed tuned gamma on the full 200-class axis."""
        if self.seen_logit_gamma == 0.0:
            return full_logits
        calibrated = full_logits.clone()
        seen = self.seen_classes.to(calibrated.device)
        calibrated[:, seen] = calibrated[:, seen] - self.seen_logit_gamma
        return calibrated

    def pclr_logits(
        self,
        image_features: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
        calibrated: bool = False,
    ) -> torch.Tensor:
        """Return full PCLR logits, slicing the class axis only at the end."""
        if not enabled:
            # Legacy PCLR retains its historical direct call.  Local-PCLR fixes
            # the reporting deviation by matching the canonical normalized-
            # prototype evaluation used by RUN-030.
            if self.candidate_top_k is None:
                if calibrated:
                    raise ValueError("Legacy PCLR不支持独立calibrated Off入口。")
                return super().logits(image_features, class_ids)
            parent_full = self.deployed_parent_logits(image_features)
            if calibrated:
                parent_full = self.apply_seen_calibration(parent_full)
            if class_ids is None:
                return parent_full
            ids = self._validated_class_ids(class_ids, parent_full.device)
            return parent_full.index_select(1, ids)

        if calibrated:
            raise ValueError("PCLR Full入口不接受重复calibrated标记。")
        parent_full = (
            super().logits(image_features, None)
            if self.candidate_top_k is None
            else self.deployed_parent_logits(image_features)
        )
        parent_std = parent_full.detach().std(
            dim=1, unbiased=False, keepdim=True
        )
        # Reader and beta have their own isolated objectives.  Detaching them
        # here prevents an accidental full-method CE from crossing boundaries.
        correction = (
            self.correction_scale
            * self.beta().detach()
            * parent_std
            * self.potentials(image_features, parent_full).detach()
        )
        full = self.apply_seen_calibration(parent_full + correction)
        if class_ids is not None:
            ids = self._validated_class_ids(class_ids, full.device)
            full = full.index_select(1, ids)
        return full

    def _validated_seen_targets(
        self, targets: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        targets = torch.as_tensor(targets, device=self.edge_index.device).long()
        if targets.ndim != 1 or targets.numel() != int(batch_size):
            raise ValueError("PCLR训练target必须是与batch等长的一维全局类别ID。")
        if int(targets.min()) < 0 or int(targets.max()) >= self.class_count:
            raise ValueError("PCLR训练target超出全局类别轴。")
        if not bool(torch.isin(targets.cpu(), self.seen_classes.cpu()).all()):
            raise ValueError("PCLR训练loss只允许official seen图像。")
        return targets

    def relation_loss(
        self, image_features: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Mean per-image CE over incident seen-seen directional edges."""
        scores = self.relation_scores(image_features)
        target_ids = self._validated_seen_targets(targets, scores.size(0)).to(scores.device)
        edges = self.edge_index.to(scores.device)
        incident_a = target_ids[:, None].eq(edges[None, :, 0])
        incident_b = target_ids[:, None].eq(edges[None, :, 1])
        incident = (incident_a | incident_b) & self.seen_edge_mask.to(scores.device)[None]
        counts = incident.sum(dim=1)
        if bool(counts.eq(0).any()):
            raise ValueError("PCLR训练图像的真类没有可用seen-seen incident edge。")
        labels = incident_b.long()
        edge_losses = -F.log_softmax(scores, dim=-1).gather(
            2, labels.unsqueeze(-1)
        ).squeeze(-1)
        per_image = (
            (edge_losses * incident.to(edge_losses.dtype)).sum(dim=1)
            / counts.to(edge_losses.dtype)
        )
        return per_image.mean()

    def beta_loss(
        self, image_features: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Seen CE whose only trainable dependency is ``raw_beta``."""
        parent_full = (
            super().logits(image_features, None)
            if self.candidate_top_k is None
            else self.deployed_parent_logits(image_features)
        ).detach()
        target_ids = self._validated_seen_targets(
            targets, parent_full.size(0)
        ).to(parent_full.device)
        parent_std = parent_full.std(dim=1, unbiased=False, keepdim=True).detach()
        potential = self.potentials(image_features, parent_full).detach()
        logits = (
            parent_full
            + self.correction_scale * self.beta() * parent_std * potential
        )
        return F.cross_entropy(logits, target_ids)

    @torch.no_grad()
    def pclr_diagnostics(
        self,
        image_features: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Explicit PCLR-only diagnostics entry point for the trainer."""
        return self.diagnostics(image_features, targets)

    @torch.no_grad()
    def diagnostics(
        self,
        image_features: torch.Tensor | list[dict[str, torch.Tensor]],
        targets: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """PCLR diagnostics, while retaining inherited GTD package support."""
        if isinstance(image_features, list):
            if targets is not None:
                raise ValueError("GTD package诊断不能同时传PCLR targets。")
            return super().diagnostics(image_features)
        scores = self.relation_scores(image_features)
        parent_logits = (
            None
            if self.candidate_top_k is None
            else self.deployed_parent_logits(image_features)
        )
        potential = self.potentials_from_scores(scores, parent_logits)
        output = {
            "beta": float(self.beta()),
            "effective_beta": self.correction_scale * float(self.beta()),
            "effective_beta_max": self.correction_scale * self.max_beta,
            "relation_margin_abs_mean": float(
                (scores[..., 0] - scores[..., 1]).abs().mean()
            ),
            "potential_mean_abs": float(potential.abs().mean()),
            "potential_abs_max": float(potential.abs().max()),
            "potential_signed_mean_abs": float(potential.mean(dim=1).abs().max()),
            "candidate_top_k": float(self.candidate_top_k or self.class_count),
            "correction_scale": self.correction_scale,
            "seen_logit_gamma": self.seen_logit_gamma,
        }
        if parent_logits is not None:
            output["active_edge_rate"] = float(
                self.candidate_edge_mask(parent_logits).float().mean()
            )
        if targets is not None:
            target_ids = self._validated_seen_targets(
                targets, scores.size(0)
            ).to(scores.device)
            edges = self.edge_index.to(scores.device)
            incident = (
                target_ids[:, None].eq(edges[None, :, 0])
                | target_ids[:, None].eq(edges[None, :, 1])
            ) & self.seen_edge_mask.to(scores.device)[None]
            counts = incident.sum(dim=1).float()
            output.update(
                {
                    "incident_edges_per_image_mean": float(counts.mean()),
                    "incident_edges_per_image_min": float(counts.min()),
                    "relation_loss": float(self.relation_loss(image_features, targets)),
                }
            )
        return output
