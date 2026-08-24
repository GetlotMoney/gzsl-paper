"""Dataset-agnostic three-module TG-VPR -> TST-NTR -> CCGR model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.ccgr import tangent_direction_basis
from model.innovations.elpt import VariableClassTGVPR
from model.innovations.tst import tangent_transport


TG_MODES = {"off", "grouped_no_value", "value_no_topology", "full"}
TRANSPORT_MODES = {"off", "euclidean", "tangent", "tangent_ntr"}
CCGR_MODES = {"off", "shared", "class_conditioned_value", "class_conditioned_four"}


class PaperV2ThreeModuleModel(nn.Module):
    """One implementation identity for the final three-dataset paper matrix."""

    def __init__(
        self,
        sentence_embeds: torch.Tensor,
        seen_classes: torch.Tensor,
        visual_centroids: torch.Tensor,
        *,
        tg_vpr_mode: str = "full",
        transport_mode: str = "tangent_ntr",
        ccgr_mode: str = "class_conditioned_four",
        dropout: float = 0.5,
        inner_ratio: float = 0.35,
        outer_ratio: float = 0.65,
        temperature: float = 0.07,
        transport_hidden_dim: int = 16,
        generator_hidden_dim: int = 32,
        max_transport_step: float = 1.5,
        max_ntr_delta: float = 0.1,
        max_generator_magnitude: float = 0.2,
    ):
        super().__init__()
        if tg_vpr_mode not in TG_MODES:
            raise ValueError(f"未知TG-VPR模式：{tg_vpr_mode}")
        if transport_mode not in TRANSPORT_MODES:
            raise ValueError(f"未知TST-NTR模式：{transport_mode}")
        if ccgr_mode not in CCGR_MODES:
            raise ValueError(f"未知CCGR模式：{ccgr_mode}")
        if sentence_embeds.ndim != 3 or tuple(sentence_embeds.shape[1:]) != (8, 768):
            raise ValueError("八角色文本必须是[class_count,8,768]。")
        classes = torch.as_tensor(seen_classes).detach().cpu().long().sort().values
        class_count = int(sentence_embeds.size(0))
        if classes.ndim != 1 or classes.numel() < 5 or classes.unique().numel() != classes.numel():
            raise ValueError("seen_classes必须包含至少5个唯一类别。")
        if int(classes.min()) < 0 or int(classes.max()) >= class_count:
            raise ValueError("seen_classes超出类别轴。")
        if float(max_transport_step) <= 0 or float(max_ntr_delta) <= 0:
            raise ValueError("迁移步长边界必须为正数。")
        if float(max_generator_magnitude) <= 0:
            raise ValueError("CCGR幅度边界必须为正数。")

        self.tg_vpr_mode = tg_vpr_mode
        self.transport_mode = transport_mode
        self.ccgr_mode = ccgr_mode
        self.class_count = class_count
        self.max_transport_step = float(max_transport_step)
        self.max_ntr_delta = float(max_ntr_delta)
        self.max_generator_magnitude = float(max_generator_magnitude)
        self.register_buffer("seen_classes", classes, persistent=True)
        self.tg_vpr = VariableClassTGVPR(
            sentence_embeds,
            classes,
            visual_centroids,
            dropout=dropout,
            inner_ratio=inner_ratio,
            outer_ratio=outer_ratio,
            temperature=temperature,
        )

        self.transport_trunk = nn.Sequential(nn.Linear(4, int(transport_hidden_dim)), nn.GELU())
        self.transport_head = nn.Linear(int(transport_hidden_dim), 1)
        nn.init.zeros_(self.transport_head.weight)
        nn.init.zeros_(self.transport_head.bias)
        self.ntr_residual = nn.Sequential(
            nn.Linear(5, int(transport_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(transport_hidden_dim), 1),
        )
        nn.init.zeros_(self.ntr_residual[-1].weight)
        nn.init.zeros_(self.ntr_residual[-1].bias)

        self.generator_trunk = nn.Sequential(nn.Linear(8, int(generator_hidden_dim)), nn.GELU())
        self.generator_weight_head = nn.Linear(int(generator_hidden_dim), 4)
        self.generator_magnitude_head = nn.Linear(int(generator_hidden_dim), 1)
        nn.init.zeros_(self.generator_weight_head.weight)
        nn.init.zeros_(self.generator_weight_head.bias)
        nn.init.zeros_(self.generator_magnitude_head.weight)
        nn.init.zeros_(self.generator_magnitude_head.bias)
        self.shared_generator_logits = nn.Parameter(torch.zeros(4))
        self.shared_generator_raw_magnitude = nn.Parameter(torch.zeros(()))

    def scale(self) -> torch.Tensor:
        return self.tg_vpr.scale()

    @staticmethod
    def _geometry_features(
        parent: torch.Tensor,
        value: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent = F.normalize(parent, dim=-1)
        value = F.normalize(value, dim=-1)
        support = F.normalize(support, dim=-1)
        if support.size(0) < 5:
            raise ValueError("TST-NTR至少需要5个seen支持类别。")
        top5 = (parent @ support.T).topk(5, dim=1).values
        cosine = (parent * value).sum(dim=-1, keepdim=True)
        displacement = (value - parent).norm(dim=-1, keepdim=True)
        summary = torch.cat(
            (cosine, displacement, top5.mean(dim=1, keepdim=True), top5.max(dim=1, keepdim=True).values),
            dim=1,
        )
        full = torch.cat((cosine, displacement, top5.mean(dim=1, keepdim=True), top5), dim=1)
        return summary, full

    def _tg_prototypes(self) -> torch.Tensor:
        if self.tg_vpr_mode == "off":
            return self.tg_vpr.base_prototypes()
        if self.tg_vpr_mode == "grouped_no_value":
            return F.normalize(self.tg_vpr.candidate_base_vectors(), dim=-1)
        return self.tg_vpr.prototypes()

    def prototype_stages(self) -> dict[str, torch.Tensor]:
        device = self.tg_vpr.sentence_embeds.device
        all_classes = torch.arange(self.class_count, device=device)
        tg = self._tg_prototypes()
        value = self.tg_vpr.value_candidate(all_classes)
        roles = self.tg_vpr.semantic_group_vectors()
        support = tg.index_select(0, self.seen_classes.to(device))
        transport_summary, transport_full = self._geometry_features(tg, value, support)
        if self.transport_mode == "off":
            step = tg.new_zeros((self.class_count,))
            transported = tg
        else:
            base_step = self.max_transport_step * torch.tanh(
                self.transport_head(self.transport_trunk(transport_summary))
            ).squeeze(-1)
            if self.transport_mode == "tangent_ntr":
                delta = self.max_ntr_delta * torch.tanh(
                    self.ntr_residual(transport_full[:, 3:])
                ).squeeze(-1)
                step = (base_step + delta).clamp(-self.max_transport_step, self.max_transport_step)
            else:
                step = base_step
            if self.transport_mode == "euclidean":
                transported = F.normalize(tg + step.unsqueeze(-1) * (value - tg), dim=-1)
            else:
                transported = tangent_transport(tg, value, step)

        direction_basis = tangent_direction_basis(transported, value, roles)
        _, generator_features = self._geometry_features(transported, value, support)
        if self.ccgr_mode == "off":
            weights = transported.new_full((self.class_count, 4), 0.25)
            magnitude = transported.new_zeros((self.class_count,))
            final = transported
        else:
            if self.ccgr_mode == "shared":
                weights = F.softmax(self.shared_generator_logits, dim=0).expand(self.class_count, -1)
                magnitude = (
                    self.max_generator_magnitude
                    * torch.tanh(self.shared_generator_raw_magnitude)
                ).expand(self.class_count)
            else:
                hidden = self.generator_trunk(generator_features)
                magnitude = self.max_generator_magnitude * torch.tanh(
                    self.generator_magnitude_head(hidden)
                ).squeeze(-1)
                if self.ccgr_mode == "class_conditioned_value":
                    weights = transported.new_zeros((self.class_count, 4))
                    weights[:, 0] = 1.0
                else:
                    weights = F.softmax(self.generator_weight_head(hidden), dim=-1)
            direction = F.normalize((weights.unsqueeze(-1) * direction_basis).sum(dim=1), dim=-1)
            direction = direction - (direction * transported).sum(dim=-1, keepdim=True) * transported
            direction = F.normalize(direction, dim=-1)
            final = F.normalize(transported + magnitude.unsqueeze(-1) * direction, dim=-1)
        return {
            "mean8": self.tg_vpr.base_prototypes(),
            "tg_vpr": tg,
            "transported": transported,
            "final": final,
            "transport_step": step,
            "generator_magnitude": magnitude,
            "generator_weights": weights,
        }

    def prototypes(self) -> torch.Tensor:
        return self.prototype_stages()["final"]

    def logits(self, image_features: torch.Tensor, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device).long())
        return F.normalize(image_features.float(), dim=-1) @ prototypes.T * self.scale()

    def topology_loss(self) -> torch.Tensor:
        base = self.tg_vpr.base_prototypes()
        adapted = self.prototypes()
        count = self.class_count
        off_diag = ~torch.eye(count, dtype=torch.bool, device=base.device)
        x = (base @ base.T).detach()[off_diag]
        y = (adapted @ adapted.T)[off_diag]
        x = x - x.mean()
        y = y - y.mean()
        return 1.0 - (x * y).sum() / (
            torch.sqrt(x.square().sum() + 1e-8) * torch.sqrt(y.square().sum() + 1e-8)
        )

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "tg_vpr": list(self.tg_vpr.parameters()),
            "transport": list(self.transport_trunk.parameters()) + list(self.transport_head.parameters()),
            "ntr": list(self.ntr_residual.parameters()),
            "ccgr_class": list(self.generator_trunk.parameters())
            + list(self.generator_weight_head.parameters())
            + list(self.generator_magnitude_head.parameters()),
            "ccgr_shared": [self.shared_generator_logits, self.shared_generator_raw_magnitude],
        }
        if self.tg_vpr_mode in {"off", "grouped_no_value"}:
            groups["tg_vpr"] = []
        if self.transport_mode == "off":
            groups["transport"] = []
            groups["ntr"] = []
        elif self.transport_mode != "tangent_ntr":
            groups["ntr"] = []
        if self.ccgr_mode == "off":
            groups["ccgr_class"] = []
            groups["ccgr_shared"] = []
        elif self.ccgr_mode == "shared":
            groups["ccgr_class"] = []
        else:
            groups["ccgr_shared"] = []
            if self.ccgr_mode == "class_conditioned_value":
                groups["ccgr_class"] = list(self.generator_trunk.parameters()) + list(
                    self.generator_magnitude_head.parameters()
                )
        return groups

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        stages = self.prototype_stages()
        step = stages["transport_step"]
        magnitude = stages["generator_magnitude"]
        weights = stages["generator_weights"]
        return {
            "transport_step_mean": float(step.mean()),
            "transport_step_std": float(step.std(unbiased=False)),
            "transport_step_max_abs": float(step.abs().max()),
            "generator_magnitude_mean": float(magnitude.mean()),
            "generator_magnitude_std": float(magnitude.std(unbiased=False)),
            "generator_magnitude_max_abs": float(magnitude.abs().max()),
            "generator_weight_min": float(weights.min()),
            "generator_weight_max": float(weights.max()),
        }
