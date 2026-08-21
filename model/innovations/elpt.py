from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tg_vpr_h1 import TGVPRH1FixedEqual


class VariableClassTGVPR(TGVPRH1FixedEqual):
    """保持H1参数身份，允许任意非空adapted class子集。"""

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
        nn.Module.__init__(self)
        if tuple(sentence_embeds.shape) != (200, 8, 768):
            raise ValueError("sentence_embeds必须是[200, 8, 768]。")
        if not torch.isfinite(sentence_embeds).all():
            raise ValueError("sentence_embeds包含NaN/Inf。")
        classes = torch.as_tensor(adapted_classes).detach().cpu().long().sort().values
        if classes.ndim != 1 or classes.numel() == 0 or classes.unique().numel() != classes.numel():
            raise ValueError("adapted_classes必须是非空唯一类编号。")
        if classes.min() < 0 or classes.max() >= 200:
            raise ValueError("adapted_classes必须位于[0, 199]。")
        centroids = F.normalize(torch.as_tensor(visual_centroids).detach().float(), dim=-1)
        if tuple(centroids.shape) != (classes.numel(), 768):
            raise ValueError("visual_centroids数量必须等于adapted_classes。")
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
        self.register_buffer("visual_centroids", centroids, persistent=True)
        self.tg_value_projection = nn.Linear(768, 768)
        self.tg_output_projection = nn.Linear(768, 768)
        self.post_projection = nn.Linear(768, 768)
        self.dropout = nn.Dropout(float(dropout))
        self.layer_norm = nn.LayerNorm(768)
        self.semantic_group_logits = nn.Parameter(torch.zeros(3))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(temperature))))
        self.inner_ratio = float(inner_ratio)
        self.outer_ratio = float(outer_ratio)

    def prototype_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        groups = F.normalize(self.transformed_groups(), dim=-1)
        count = self.adapted_classes.numel()
        weights = self.semantic_group_weights().unsqueeze(0).expand(count, -1)
        base_vectors = self.candidate_base_vectors()
        base_scale = base_vectors.new_ones((200,))
        base_scale[self.adapted_classes] = 1.0 - self.outer_ratio
        base_part = base_scale.unsqueeze(-1) * base_vectors
        role_part = groups.new_zeros((200, 3, 768))
        role_part[self.adapted_classes] = self.outer_ratio * weights.unsqueeze(-1) * groups
        return base_part + role_part.sum(dim=1), base_part, role_part

    def value_candidate(self, class_ids: torch.Tensor) -> torch.Tensor:
        class_ids = class_ids.to(self.sentence_embeds.device).long()
        source = self.semantic_group_vectors().index_select(0, class_ids)
        batch, group_count, dim = source.shape
        value = self.tg_value_projection(source)
        value = value.view(batch, group_count, 1, dim).transpose(1, 2)
        weights = self.semantic_group_weights().view(1, 1, 1, group_count).expand(
            batch, 1, group_count, group_count
        )
        context = torch.einsum("bhqg,bhgd->bhqd", weights, value)
        context = context.transpose(1, 2).contiguous().view(batch, group_count, dim)
        context = self.tg_output_projection(context)
        context = self.post_projection(context)
        mixed = self.inner_ratio * context + (1.0 - self.inner_ratio) * source
        transformed = F.normalize(self.layer_norm(2.0 * mixed), dim=-1)
        role = (self.semantic_group_weights().view(1, 3, 1) * transformed).sum(dim=1)
        base = self.base_vectors().index_select(0, class_ids)
        return F.normalize(
            (1.0 - self.outer_ratio) * base + self.outer_ratio * role,
            dim=-1,
        )


class ELPTGate(nn.Module):
    """用类别语义几何学习每类迁移强度。"""

    def __init__(
        self,
        input_dim: int = 4,
        max_alpha: float = 1.0,
        initial_alpha: float = 0.1,
    ):
        super().__init__()
        if not 0.0 < float(max_alpha) <= 1.0:
            raise ValueError("max_alpha必须位于(0, 1]。")
        if not 0.0 < float(initial_alpha) < float(max_alpha):
            raise ValueError("initial_alpha必须位于(0, max_alpha)。")
        if int(input_dim) not in (4, 8):
            raise ValueError("ELPT gate input_dim只允许4或8。")
        self.input_dim = int(input_dim)
        self.max_alpha = float(max_alpha)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        raw_initial = float(initial_alpha) / self.max_alpha
        nn.init.constant_(
            self.network[-1].bias,
            math.log(raw_initial / (1.0 - raw_initial)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_alpha * torch.sigmoid(self.network(features)).squeeze(-1)


def fixed_class_folds(seenclasses: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    classes = torch.as_tensor(seenclasses).detach().cpu().long().sort().values
    if classes.numel() != 150 or classes.unique().numel() != 150:
        raise ValueError("ELPT固定要求150个seen类。")
    ranks = torch.arange(150)
    folds = []
    for fold_id in range(3):
        pseudo_unseen = classes[ranks.remainder(3) == fold_id]
        pseudo_seen = classes[ranks.remainder(3) != fold_id]
        folds.append((pseudo_seen, pseudo_unseen))
    return folds


def gate_features(
    base: torch.Tensor,
    value: torch.Tensor,
    support_base: torch.Tensor,
    mode: str = "summary",
) -> torch.Tensor:
    base = F.normalize(base, dim=-1)
    value = F.normalize(value, dim=-1)
    support_base = F.normalize(support_base, dim=-1)
    similarity = base @ support_base.T
    top5 = similarity.topk(k=5, dim=1).values
    cosine = (base * value).sum(dim=-1)
    displacement = (value - base).norm(dim=-1)
    if mode == "summary":
        return torch.stack(
            (cosine, displacement, top5.mean(dim=1), top5.max(dim=1).values),
            dim=1,
        )
    if mode == "top5_vector":
        return torch.cat(
            (cosine.unsqueeze(1), displacement.unsqueeze(1), top5.mean(dim=1, keepdim=True), top5),
            dim=1,
        )
    raise ValueError("未知ELPT gate feature mode。")


def blend_prototypes(base: torch.Tensor, value: torch.Tensor, alpha: torch.Tensor):
    return F.normalize(
        (1.0 - alpha.unsqueeze(-1)) * base + alpha.unsqueeze(-1) * value,
        dim=-1,
    )


def topology_loss(base: torch.Tensor, adapted: torch.Tensor) -> torch.Tensor:
    base = F.normalize(base, dim=-1)
    adapted = F.normalize(adapted, dim=-1)
    count = base.size(0)
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
