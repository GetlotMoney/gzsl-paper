from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.candidates.v2.modules.unified_seen import UnifiedSeenPrototypeModel


class ExpertAttributeUnifiedModel(nn.Module):
    """在统一文本原型上增加CUB 312维专家属性切向残差。"""

    def __init__(
        self,
        text_model: UnifiedSeenPrototypeModel,
        attributes: torch.Tensor,
        *,
        max_attribute_residual: float = 0.5,
    ):
        super().__init__()
        if tuple(attributes.shape) != (200, 312):
            raise ValueError("CUB专家属性必须是[200,312]。")
        if not torch.isfinite(attributes).all():
            raise ValueError("CUB专家属性包含NaN/Inf。")
        self.text_model = text_model
        self.register_buffer(
            "attributes", F.normalize(attributes.detach().float(), dim=-1)
        )
        self.attribute_projection = nn.Linear(312, 768, bias=False)
        self.raw_attribute_residual = nn.Parameter(torch.tensor(0.0))
        self.max_attribute_residual = float(max_attribute_residual)
        if self.max_attribute_residual <= 0.0:
            raise ValueError("专家属性残差上限必须为正数。")

    def scale(self) -> torch.Tensor:
        return self.text_model.scale()

    def attribute_residual(self) -> torch.Tensor:
        return self.max_attribute_residual * torch.tanh(
            self.raw_attribute_residual
        )

    def prototypes(self) -> torch.Tensor:
        base = self.text_model.prototypes()
        attribute = F.normalize(self.attribute_projection(self.attributes), dim=-1)
        tangent = attribute - (attribute * base).sum(dim=-1, keepdim=True) * base
        tangent = F.normalize(tangent, dim=-1)
        return F.normalize(
            base + self.attribute_residual() * tangent,
            dim=-1,
        )

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
        ids = self.text_model.active_classes.to(self.attributes.device)
        base = self.text_model.tg_vpr.base_prototypes().index_select(0, ids)
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
        result = self.text_model.diagnostics()
        result["attribute_residual"] = float(self.attribute_residual())
        return result
