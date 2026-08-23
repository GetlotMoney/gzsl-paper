from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def taxonomic_suffix_group_ids(class_names: list[str]) -> torch.Tensor:
    if len(class_names) != 200:
        raise ValueError("TIGR必须接收200个类别名。")
    suffixes = [
        name.split(".", 1)[-1].strip().split("_")[-1].lower()
        for name in class_names
    ]
    counts = Counter(suffixes)
    grouped_suffixes = sorted(suffix for suffix, count in counts.items() if count >= 2)
    mapping = {suffix: index for index, suffix in enumerate(grouped_suffixes)}
    return torch.tensor([mapping.get(suffix, -1) for suffix in suffixes], dtype=torch.long)


class TaxonomicIntraGroupResidual(nn.Module):
    """固定SDCR残差，并增加类别相对同名族群中心的身份方向。"""

    def __init__(
        self,
        sdcr_prototypes: torch.Tensor,
        sdcr_beta: float,
        group_ids: torch.Tensor,
        max_beta: float = 5.0,
    ) -> None:
        super().__init__()
        if tuple(sdcr_prototypes.shape) != (200, 768):
            raise ValueError("TIGR SDCR原型必须是[200,768]。")
        if tuple(group_ids.shape) != (200,):
            raise ValueError("TIGR group_ids必须是[200]。")
        base = F.normalize(sdcr_prototypes.detach().float(), dim=-1)
        group_ids = group_ids.detach().long().to(base.device)
        identity = torch.zeros_like(base)
        valid_groups = group_ids[group_ids >= 0].unique(sorted=True)
        grouped_count = 0
        for group_id in valid_groups.tolist():
            mask = group_ids.eq(int(group_id))
            center = F.normalize(base[mask].mean(dim=0), dim=0)
            identity[mask] = F.normalize(base[mask] - center.unsqueeze(0), dim=-1)
            grouped_count += int(mask.sum())
        self.register_buffer("sdcr_prototypes", base)
        self.register_buffer("identity_prototypes", identity)
        self.register_buffer("group_ids", group_ids)
        self.register_buffer("sdcr_beta", torch.tensor(float(sdcr_beta)))
        self.group_count = int(valid_groups.numel())
        self.grouped_class_count = int(grouped_count)
        self.max_beta = float(max_beta)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self) -> torch.Tensor:
        return self.max_beta * torch.tanh(self.raw_beta)

    def stats(self) -> dict[str, float | int]:
        active = self.group_ids >= 0
        cosine = (
            self.sdcr_prototypes[active] * self.identity_prototypes[active]
        ).sum(dim=-1)
        return {
            "identity_beta": float(self.beta().detach()),
            "group_count": self.group_count,
            "grouped_class_count": self.grouped_class_count,
            "mean_base_identity_cosine": float(cosine.mean()),
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        base = self.sdcr_prototypes
        identity = self.identity_prototypes
        if class_ids is not None:
            ids = class_ids.to(base.device)
            base = base.index_select(0, ids)
            identity = identity.index_select(0, ids)
        normalized = F.normalize(images.float(), dim=-1)
        logits = parent_logits + self.sdcr_beta * (normalized @ base.T)
        if not enabled:
            return logits
        return logits + self.beta() * (normalized @ identity.T)


class TaxonomicWithinGroupLogitSharpening(nn.Module):
    """保持族群logit均值不变，只缩放族内类别差值。"""

    def __init__(
        self,
        sdcr_prototypes: torch.Tensor,
        sdcr_beta: float,
        group_ids: torch.Tensor,
        max_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if tuple(sdcr_prototypes.shape) != (200, 768):
            raise ValueError("TWLS SDCR原型必须是[200,768]。")
        if tuple(group_ids.shape) != (200,):
            raise ValueError("TWLS group_ids必须是[200]。")
        self.register_buffer(
            "sdcr_prototypes",
            F.normalize(sdcr_prototypes.detach().float(), dim=-1),
        )
        self.register_buffer("sdcr_beta", torch.tensor(float(sdcr_beta)))
        self.register_buffer("group_ids", group_ids.detach().long())
        valid = self.group_ids[self.group_ids >= 0]
        self.group_count = int(valid.unique().numel())
        self.grouped_class_count = int(valid.numel())
        self.max_alpha = float(max_alpha)
        self.raw_alpha = nn.Parameter(torch.zeros(()))

    def alpha(self) -> torch.Tensor:
        return self.max_alpha * torch.tanh(self.raw_alpha)

    def stats(self) -> dict[str, float | int]:
        return {
            "within_group_alpha": float(self.alpha().detach()),
            "group_count": self.group_count,
            "grouped_class_count": self.grouped_class_count,
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        prototypes = self.sdcr_prototypes
        ids = (
            torch.arange(200, device=prototypes.device)
            if class_ids is None
            else class_ids.to(prototypes.device)
        )
        prototypes = prototypes.index_select(0, ids)
        logits = parent_logits + self.sdcr_beta * (
            F.normalize(images.float(), dim=-1) @ prototypes.T
        )
        if not enabled:
            return logits
        local_groups = self.group_ids.index_select(0, ids)
        output = logits.clone()
        for group_id in local_groups[local_groups >= 0].unique(sorted=True).tolist():
            positions = local_groups.eq(int(group_id)).nonzero(as_tuple=False).flatten()
            if positions.numel() < 2:
                continue
            group_logits = logits.index_select(1, positions)
            centered = group_logits - group_logits.mean(dim=1, keepdim=True)
            output[:, positions] = group_logits + self.alpha() * centered
        return output


class TaxonomicPairwiseLogitDeconvolution(nn.Module):
    """按族内语义相似度做非均匀近邻高通，并保持组均值。"""

    def __init__(
        self,
        sdcr_prototypes: torch.Tensor,
        sdcr_beta: float,
        group_ids: torch.Tensor,
        similarity_temperature: float = 0.1,
        max_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if tuple(sdcr_prototypes.shape) != (200, 768):
            raise ValueError("TPLD SDCR原型必须是[200,768]。")
        if tuple(group_ids.shape) != (200,):
            raise ValueError("TPLD group_ids必须是[200]。")
        if float(similarity_temperature) <= 0:
            raise ValueError("TPLD相似度温度必须为正。")
        base = F.normalize(sdcr_prototypes.detach().float(), dim=-1)
        group_ids = group_ids.detach().long().to(base.device)
        affinity = torch.zeros(200, 200, dtype=base.dtype, device=base.device)
        valid_groups = group_ids[group_ids >= 0].unique(sorted=True)
        grouped_count = 0
        entropies = []
        for group_id in valid_groups.tolist():
            positions = group_ids.eq(int(group_id)).nonzero(as_tuple=False).flatten()
            if positions.numel() < 2:
                continue
            local = base.index_select(0, positions)
            similarity = local @ local.T / float(similarity_temperature)
            similarity.fill_diagonal_(-float("inf"))
            weights = torch.softmax(similarity, dim=-1)
            affinity[positions.unsqueeze(1), positions.unsqueeze(0)] = weights
            entropies.append(
                -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
            )
            grouped_count += int(positions.numel())
        self.register_buffer("sdcr_prototypes", base)
        self.register_buffer("sdcr_beta", torch.tensor(float(sdcr_beta)))
        self.register_buffer("group_ids", group_ids)
        self.register_buffer("affinity", affinity)
        self.register_buffer(
            "mean_affinity_entropy",
            torch.stack(entropies).mean() if entropies else torch.tensor(0.0),
        )
        self.group_count = int(valid_groups.numel())
        self.grouped_class_count = int(grouped_count)
        self.max_alpha = float(max_alpha)
        self.raw_alpha = nn.Parameter(torch.zeros(()))

    def alpha(self) -> torch.Tensor:
        return self.max_alpha * torch.tanh(self.raw_alpha)

    def stats(self) -> dict[str, float | int]:
        return {
            "pairwise_alpha": float(self.alpha().detach()),
            "group_count": self.group_count,
            "grouped_class_count": self.grouped_class_count,
            "mean_affinity_entropy": float(self.mean_affinity_entropy),
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        ids = (
            torch.arange(200, device=self.sdcr_prototypes.device)
            if class_ids is None
            else class_ids.to(self.sdcr_prototypes.device)
        )
        prototypes = self.sdcr_prototypes.index_select(0, ids)
        logits = parent_logits + self.sdcr_beta * (
            F.normalize(images.float(), dim=-1) @ prototypes.T
        )
        if not enabled:
            return logits
        affinity = self.affinity.index_select(0, ids).index_select(1, ids)
        row_sum = affinity.sum(dim=-1, keepdim=True)
        active = row_sum.squeeze(1) > 0
        affinity = affinity / row_sum.clamp_min(1e-8)
        neighbor_logits = logits @ affinity.T
        high_pass = logits - neighbor_logits
        high_pass[:, ~active] = 0.0
        local_groups = self.group_ids.index_select(0, ids)
        for group_id in local_groups[local_groups >= 0].unique(sorted=True).tolist():
            positions = local_groups.eq(int(group_id)).nonzero(as_tuple=False).flatten()
            if positions.numel() < 2:
                high_pass[:, positions] = 0.0
                continue
            high_pass[:, positions] -= high_pass[:, positions].mean(
                dim=1, keepdim=True
            )
        return logits + self.alpha() * high_pass
