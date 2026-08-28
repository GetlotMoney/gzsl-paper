from __future__ import annotations

import torch
import torch.nn.functional as F

from model.frameworks.v2 import TGVPRH1FixedEqual


class TGVPRH1UnseenValueTransfer(TGVPRH1FixedEqual):
    """保持seen路径不变，仅把共享Value变换应用到unseen语义。"""

    def __init__(self, *args, transfer_strength: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 < float(transfer_strength) <= 1.0:
            raise ValueError("transfer_strength必须位于(0, 1]。")
        self.transfer_strength = float(transfer_strength)

    def transformed_unseen_groups(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_classes = torch.arange(200, device=self.adapted_classes.device)
        unseen_classes = all_classes[
            ~torch.isin(all_classes, self.adapted_classes)
        ]
        source = self.semantic_group_vectors().index_select(0, unseen_classes)
        batch, group_count, dim = source.shape
        value = self.tg_value_projection(source)
        value = value.view(batch, group_count, 1, dim).transpose(1, 2)
        weights = self.semantic_group_weights().view(1, 1, 1, group_count).expand(
            batch, 1, group_count, group_count
        )
        context = torch.einsum("bhqg,bhgd->bhqd", weights, value)
        context = context.transpose(1, 2).contiguous().view(batch, group_count, dim)
        context = self.tg_output_projection(context)
        # TRY只做评估迁移，不为unseen引入新的随机dropout路径。
        context = self.post_projection(context)
        mixed = self.inner_ratio * context + (1.0 - self.inner_ratio) * source
        return unseen_classes, self.layer_norm(2.0 * mixed)

    def prototype_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, base_part, role_part = super().prototype_components()
        base_part = base_part.clone()
        role_part = role_part.clone()
        unseen_classes, transformed = self.transformed_unseen_groups()
        groups = F.normalize(transformed, dim=-1)
        weights = self.semantic_group_weights().view(1, 3, 1)
        base_vectors = self.candidate_base_vectors().index_select(0, unseen_classes)
        base_part[unseen_classes] = (1.0 - self.outer_ratio) * base_vectors
        role_part[unseen_classes] = self.outer_ratio * weights * groups
        enhanced = base_part + role_part.sum(dim=1)
        return enhanced, base_part, role_part

    def prototypes(self, return_diagnostics: bool = False):
        full_enhanced, base_part, role_part = self.prototype_components()
        full = F.normalize(full_enhanced, dim=-1)
        base_enhanced, _, _ = TGVPRH1FixedEqual.prototype_components(self)
        base = F.normalize(base_enhanced, dim=-1)
        all_classes = torch.arange(200, device=self.adapted_classes.device)
        unseen = all_classes[~torch.isin(all_classes, self.adapted_classes)]
        prototypes = base.clone()
        prototypes[unseen] = F.normalize(
            (1.0 - self.transfer_strength) * base.index_select(0, unseen)
            + self.transfer_strength * full.index_select(0, unseen),
            dim=-1,
        )
        if return_diagnostics:
            return prototypes, {
                "base": self.base_prototypes(),
                "semantic_group_weights": self.semantic_group_weights(),
                "base_part": base_part,
                "role_part": role_part,
                "transfer_strength": self.transfer_strength,
            }
        return prototypes
