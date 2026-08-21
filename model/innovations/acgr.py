from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.icgr import ICGRRouter


class AllClassCenteredGroupRouter(nn.Module):
    """在全部200类上路由均值为零的三组语义残差。"""

    def __init__(self, parent: nn.Module, hidden_dim: int = 64, role_scale: float = 0.65):
        super().__init__()
        if float(role_scale) <= 0.0:
            raise ValueError("ACGR role_scale必须为正数。")
        self.parent = parent.eval()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.router = ICGRRouter(input_dim=768, hidden_dim=hidden_dim)
        self.role_scale = float(role_scale)
        with torch.no_grad():
            self.register_buffer(
                "parent_prototypes",
                self.parent.prototypes().detach().clone(),
                persistent=False,
            )
            self.register_buffer(
                "semantic_groups",
                self.parent.semantic_group_vectors().detach().clone(),
                persistent=False,
            )
            self.register_buffer(
                "frozen_scale", self.parent.scale().detach().clone(), persistent=False
            )

    def router_inputs(self, image_features: torch.Tensor) -> torch.Tensor:
        return image_features.float()

    def route_weights(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.router(self.router_inputs(image_features))

    def component_logits(
        self, image_features: torch.Tensor, class_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = F.normalize(image_features.float(), dim=-1)
        parent_logits = images @ self.parent_prototypes.T * self.frozen_scale
        group_logits = torch.einsum("bd,crd->bcr", images, self.semantic_groups)
        centered_roles = self.role_scale * (
            group_logits - group_logits.mean(dim=-1, keepdim=True)
        ) * self.frozen_scale
        if class_ids is not None:
            class_ids = class_ids.to(parent_logits.device)
            parent_logits = parent_logits.index_select(1, class_ids)
            centered_roles = centered_roles.index_select(1, class_ids)
        return parent_logits, centered_roles

    @staticmethod
    def logits_from_weights(
        base_logits: torch.Tensor,
        role_logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return base_logits + torch.einsum("br,bcr->bc", weights, role_logits)

    def logits(
        self,
        image_features: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        *,
        enabled: bool = True,
    ) -> torch.Tensor:
        parent_logits, centered_roles = self.component_logits(image_features, class_ids)
        if not enabled:
            return parent_logits
        return self.logits_from_weights(
            parent_logits, centered_roles, self.route_weights(image_features)
        )

    def forward(
        self, image_features: torch.Tensor, class_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.logits(image_features, class_ids)
