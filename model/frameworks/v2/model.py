from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TGVPRH1FixedEqual(nn.Module):
    """三组固定等权、单一768维Value路径的原型重参数化模块。"""

    def __init__(
        self,
        sentence_embeds: torch.Tensor,
        adapted_classes: torch.Tensor,
        visual_centroids: torch.Tensor,
        *,
        dropout: float = 0.5,
        inner_ratio: float = 0.35,
        outer_ratio: float = 0.65,
        temperature: float = 0.07,
    ):
        super().__init__()
        if tuple(sentence_embeds.shape) != (200, 8, 768):
            raise ValueError("sentence_embeds必须是[200, 8, 768]。")
        if not torch.isfinite(sentence_embeds).all():
            raise ValueError("sentence_embeds包含NaN/Inf。")
        classes = torch.as_tensor(adapted_classes).detach().cpu().long().sort().values
        if classes.ndim != 1 or classes.numel() != 150 or classes.unique().numel() != 150:
            raise ValueError("adapted_classes必须包含150个唯一seen类编号。")
        centroids = F.normalize(torch.as_tensor(visual_centroids).detach().float(), dim=-1)
        if tuple(centroids.shape) != (150, 768):
            raise ValueError("visual_centroids必须是[150, 768]。")
        if not 0.0 < float(inner_ratio) < 1.0:
            raise ValueError("inner_ratio必须位于(0, 1)。")
        if not 0.0 < float(outer_ratio) < 1.0:
            raise ValueError("outer_ratio必须位于(0, 1)。")
        if float(temperature) <= 0.0:
            raise ValueError("temperature必须为正数。")

        self.register_buffer(
            "sentence_embeds",
            F.normalize(sentence_embeds.detach().float(), dim=-1),
            persistent=True,
        )
        self.register_buffer("adapted_classes", classes, persistent=True)
        # 保留该buffer与源H1 checkpoint兼容；它不进入当前forward。
        self.register_buffer("visual_centroids", centroids, persistent=True)
        self.tg_value_projection = nn.Linear(768, 768)
        self.tg_output_projection = nn.Linear(768, 768)
        self.post_projection = nn.Linear(768, 768)
        self.dropout = nn.Dropout(float(dropout))
        self.layer_norm = nn.LayerNorm(768)
        # 保留该键与源checkpoint兼容；固定等权路径不会读取它，因此grad恒为None。
        self.semantic_group_logits = nn.Parameter(torch.zeros(3))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(temperature))))
        self.inner_ratio = float(inner_ratio)
        self.outer_ratio = float(outer_ratio)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def base_vectors(self) -> torch.Tensor:
        return self.sentence_embeds.mean(dim=1)

    def base_prototypes(self) -> torch.Tensor:
        return F.normalize(self.base_vectors(), dim=-1)

    def semantic_group_weights(self) -> torch.Tensor:
        return self.sentence_embeds.new_full((3,), 1.0 / 3.0)

    def semantic_group_vectors(self) -> torch.Tensor:
        local = F.normalize(self.sentence_embeds[:, :6].mean(dim=1), dim=-1)
        # 缓存物理顺序：前六句局部、第7句整体、第8句独特。
        unique = F.normalize(self.sentence_embeds[:, 7], dim=-1)
        overall = F.normalize(self.sentence_embeds[:, 6], dim=-1)
        return torch.stack((local, unique, overall), dim=1)

    def candidate_base_vectors(self) -> torch.Tensor:
        base = self.base_vectors()
        groups = self.semantic_group_vectors()
        weights = self.semantic_group_weights()
        grouped = F.normalize((weights.view(1, 3, 1) * groups).sum(dim=1), dim=-1)
        candidate = base.clone()
        candidate[self.adapted_classes] = grouped.index_select(0, self.adapted_classes)
        return candidate

    def transformed_groups(self) -> torch.Tensor:
        source = self.semantic_group_vectors().index_select(0, self.adapted_classes)
        batch, group_count, dim = source.shape
        value = self.tg_value_projection(source)
        value = value.view(batch, group_count, 1, dim).transpose(1, 2)
        weights = self.semantic_group_weights().view(1, 1, 1, group_count).expand(
            batch, 1, group_count, group_count
        )
        weights = F.dropout(weights, p=float(self.dropout.p), training=self.training)
        context = torch.einsum("bhqg,bhgd->bhqd", weights, value)
        context = context.transpose(1, 2).contiguous().view(batch, group_count, dim)
        context = self.tg_output_projection(context)
        context = self.dropout(self.post_projection(context))
        mixed = self.inner_ratio * context + (1.0 - self.inner_ratio) * source
        return self.layer_norm(2.0 * mixed)

    def prototype_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        groups = F.normalize(self.transformed_groups(), dim=-1)
        weights = self.semantic_group_weights().unsqueeze(0).expand(150, -1)
        base_vectors = self.candidate_base_vectors()
        base_scale = base_vectors.new_ones((200,))
        base_scale[self.adapted_classes] = 1.0 - self.outer_ratio
        base_part = base_scale.unsqueeze(-1) * base_vectors
        role_part = groups.new_zeros((200, 3, 768))
        role_part[self.adapted_classes] = self.outer_ratio * weights.unsqueeze(-1) * groups
        return base_part + role_part.sum(dim=1), base_part, role_part

    def prototypes(self, return_diagnostics: bool = False):
        enhanced, base_part, role_part = self.prototype_components()
        prototypes = F.normalize(enhanced, dim=-1)
        if return_diagnostics:
            return prototypes, {
                "base": self.base_prototypes(),
                "semantic_group_weights": self.semantic_group_weights(),
                "base_part": base_part,
                "role_part": role_part,
            }
        return prototypes

    def topology_loss(self) -> torch.Tensor:
        base = self.base_prototypes()
        adapted = self.prototypes()
        off_diag = ~torch.eye(200, dtype=torch.bool, device=base.device)
        x = (base @ base.T).detach()[off_diag]
        y = (adapted @ adapted.T)[off_diag]
        x = x - x.mean()
        y = y - y.mean()
        correlation = (x * y).sum() / (
            torch.sqrt(x.square().sum() + 1e-8)
            * torch.sqrt(y.square().sum() + 1e-8)
        )
        return 1.0 - correlation

    def logits(self, image_features: torch.Tensor, class_ids=None) -> torch.Tensor:
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return F.normalize(image_features.float(), dim=-1) @ prototypes.T * self.scale()

    def logit_components(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enhanced, base_part, role_part = self.prototype_components()
        denominator = enhanced.norm(dim=-1).clamp_min(1e-12)
        images = F.normalize(image_features.float(), dim=-1)
        base_logits = (images @ base_part.T) / denominator.unsqueeze(0)
        role_logits = torch.einsum("bd,crd->bcr", images, role_part)
        role_logits = role_logits / denominator.view(1, -1, 1)
        return base_logits * self.scale(), role_logits * self.scale()
