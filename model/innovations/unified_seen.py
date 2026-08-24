from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.ccgr import tangent_direction_basis
from model.innovations.elpt import VariableClassTGVPR
from model.innovations.tst import tangent_transport


class UnifiedSeenPrototypeModel(nn.Module):
    """在全部seen样本上联合训练TG-VPR、语义迁移和类别条件修正。"""

    def __init__(
        self,
        sentence_embeds: torch.Tensor,
        seenclasses: torch.Tensor,
        visual_centroids: torch.Tensor,
        active_classes: torch.Tensor | None = None,
        *,
        dropout: float = 0.5,
        inner_ratio: float = 0.35,
        outer_ratio: float = 0.65,
        temperature: float = 0.07,
        transport_hidden_dim: int = 16,
        generator_hidden_dim: int = 32,
        max_transport_step: float = 1.5,
        max_generator_magnitude: float = 0.2,
    ):
        super().__init__()
        classes = torch.as_tensor(seenclasses).detach().cpu().long().sort().values
        if classes.numel() not in (100, 150) or classes.unique().numel() != classes.numel():
            raise ValueError("统一seen训练只接受100类开发训练或150类最终训练。")
        active = (
            torch.arange(200)
            if active_classes is None
            else torch.as_tensor(active_classes).detach().cpu().long().sort().values
        )
        if active.ndim != 1 or active.unique().numel() != active.numel():
            raise ValueError("active_classes必须是一维唯一类别编号。")
        if active.numel() not in (150, 200) or not torch.isin(classes, active).all():
            raise ValueError("active_classes必须包含训练类并固定为150或200类。")
        self.tg_vpr = VariableClassTGVPR(
            sentence_embeds,
            classes,
            visual_centroids,
            dropout=dropout,
            inner_ratio=inner_ratio,
            outer_ratio=outer_ratio,
            temperature=temperature,
        )
        self.register_buffer("seenclasses", classes, persistent=True)
        self.register_buffer("active_classes", active, persistent=True)
        self.max_transport_step = float(max_transport_step)
        self.max_generator_magnitude = float(max_generator_magnitude)
        if self.max_transport_step <= 0.0 or self.max_generator_magnitude <= 0.0:
            raise ValueError("统一训练的迁移步长和生成幅度上限必须为正数。")

        self.transport_trunk = nn.Sequential(
            nn.Linear(8, int(transport_hidden_dim)), nn.GELU()
        )
        self.transport_head = nn.Linear(int(transport_hidden_dim), 1)
        self.generator_trunk = nn.Sequential(
            nn.Linear(8, int(generator_hidden_dim)), nn.GELU()
        )
        self.generator_weight_head = nn.Linear(int(generator_hidden_dim), 4)
        self.generator_magnitude_head = nn.Linear(int(generator_hidden_dim), 1)

        # 两个残差出口均从0开始，初始forward严格等价于TG-VPR。
        nn.init.zeros_(self.transport_head.weight)
        nn.init.zeros_(self.transport_head.bias)
        nn.init.zeros_(self.generator_weight_head.weight)
        nn.init.zeros_(self.generator_weight_head.bias)
        nn.init.zeros_(self.generator_magnitude_head.weight)
        nn.init.zeros_(self.generator_magnitude_head.bias)

    def scale(self) -> torch.Tensor:
        return self.tg_vpr.scale()

    @staticmethod
    def _class_features(
        parent: torch.Tensor,
        value: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        parent = F.normalize(parent, dim=-1)
        value = F.normalize(value, dim=-1)
        support = F.normalize(support, dim=-1)
        top5 = (parent @ support.T).topk(5, dim=1).values
        cosine = (parent * value).sum(dim=-1, keepdim=True)
        displacement = (value - parent).norm(dim=-1, keepdim=True)
        return torch.cat(
            (cosine, displacement, top5.mean(dim=1, keepdim=True), top5), dim=1
        )

    def prototype_stages_from_tg(
        self,
        tg_vpr: VariableClassTGVPR,
        support_classes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """用共享迁移/生成权重处理任意100类或150类TG-VPR父模型。"""
        allclasses = torch.arange(200, device=tg_vpr.sentence_embeds.device)
        support_classes = torch.as_tensor(support_classes).to(allclasses.device).long()
        if support_classes.ndim != 1 or support_classes.numel() not in (100, 150):
            raise ValueError("外部TG父模型support必须包含100或150类。")
        tg_prototypes = tg_vpr.prototypes()
        value_prototypes = tg_vpr.value_candidate(allclasses)
        support = tg_prototypes.index_select(0, support_classes)
        transport_features = self._class_features(
            tg_prototypes, value_prototypes, support
        )
        transport_step = self.max_transport_step * torch.tanh(
            self.transport_head(self.transport_trunk(transport_features))
        ).squeeze(-1)
        transported = tangent_transport(
            tg_prototypes, value_prototypes, transport_step
        )

        role_prototypes = tg_vpr.semantic_group_vectors()
        direction_basis = tangent_direction_basis(
            transported, value_prototypes, role_prototypes
        )
        generator_features = self._class_features(
            transported, value_prototypes, support
        )
        hidden = self.generator_trunk(generator_features)
        direction_weights = F.softmax(
            self.generator_weight_head(hidden), dim=-1
        )
        magnitude = self.max_generator_magnitude * torch.tanh(
            self.generator_magnitude_head(hidden)
        ).squeeze(-1)
        direction = F.normalize(
            (direction_weights.unsqueeze(-1) * direction_basis).sum(dim=1), dim=-1
        )
        direction = direction - (
            direction * transported
        ).sum(dim=-1, keepdim=True) * transported
        direction = F.normalize(direction, dim=-1)
        final = F.normalize(
            transported + magnitude.unsqueeze(-1) * direction, dim=-1
        )
        return {
            "tg_vpr": tg_prototypes,
            "transported": transported,
            "final": final,
            "transport_step": transport_step,
            "generator_magnitude": magnitude,
            "generator_weights": direction_weights,
        }

    def shared_fold_tg_vpr(
        self,
        pseudo_seen_classes: torch.Tensor,
        pseudo_seen_centroids: torch.Tensor,
    ) -> VariableClassTGVPR:
        """创建共享TG-VPR可训练权重、但仅适配pseudo-seen类的fold父模型。"""
        fold = VariableClassTGVPR(
            self.tg_vpr.sentence_embeds,
            pseudo_seen_classes,
            pseudo_seen_centroids,
            dropout=float(self.tg_vpr.dropout.p),
            inner_ratio=self.tg_vpr.inner_ratio,
            outer_ratio=self.tg_vpr.outer_ratio,
            temperature=float(self.tg_vpr.logit_scale.detach().exp().reciprocal()),
        )
        fold.tg_value_projection = self.tg_vpr.tg_value_projection
        fold.tg_output_projection = self.tg_vpr.tg_output_projection
        fold.post_projection = self.tg_vpr.post_projection
        fold.layer_norm = self.tg_vpr.layer_norm
        fold.semantic_group_logits = self.tg_vpr.semantic_group_logits
        fold.logit_scale = self.tg_vpr.logit_scale
        return fold

    def prototype_stages(self) -> dict[str, torch.Tensor]:
        return self.prototype_stages_from_tg(self.tg_vpr, self.seenclasses)

    def prototypes(self) -> torch.Tensor:
        return self.prototype_stages()["final"]

    def logits(self, image_features: torch.Tensor, class_ids=None) -> torch.Tensor:
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(
                0, class_ids.to(prototypes.device).long()
            )
        return (
            F.normalize(image_features.float(), dim=-1)
            @ prototypes.T
            * self.scale()
        )

    def topology_loss(self) -> torch.Tensor:
        ids = self.active_classes.to(self.tg_vpr.sentence_embeds.device)
        base = self.tg_vpr.base_prototypes().index_select(0, ids)
        adapted = self.prototypes().index_select(0, ids)
        count = ids.numel()
        off_diag = ~torch.eye(count, dtype=torch.bool, device=base.device)
        x = (base @ base.T).detach()[off_diag]
        y = (adapted @ adapted.T)[off_diag]
        x = x - x.mean()
        y = y - y.mean()
        correlation = (x * y).sum() / (
            torch.sqrt(x.square().sum() + 1e-8)
            * torch.sqrt(y.square().sum() + 1e-8)
        )
        return 1.0 - correlation

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        stages = self.prototype_stages()
        step = stages["transport_step"]
        magnitude = stages["generator_magnitude"]
        return {
            "transport_step_mean": float(step.mean()),
            "transport_step_std": float(step.std(unbiased=False)),
            "transport_step_max_abs": float(step.abs().max()),
            "generator_magnitude_mean": float(magnitude.mean()),
            "generator_magnitude_std": float(magnitude.std(unbiased=False)),
            "generator_magnitude_max_abs": float(magnitude.abs().max()),
        }
