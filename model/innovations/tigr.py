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
