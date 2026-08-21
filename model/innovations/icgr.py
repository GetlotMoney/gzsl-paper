from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICGRRouter(nn.Module):
    """根据冻结的图像CLS为local/unique/overall分配权重。"""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.network(image_features.float()), dim=-1)
        if not torch.isfinite(weights).all():
            raise FloatingPointError("ICGR权重包含NaN/Inf。")
        return weights


class ICGRClassifier(nn.Module):
    """冻结TG-VPR，只训练图像条件三组路由。"""

    def __init__(
        self,
        parent: nn.Module,
        hidden_dim: int = 64,
        router_input_mode: str = "image_cls",
    ):
        super().__init__()
        if router_input_mode not in ("image_cls", "image_cls_role_cosine"):
            raise ValueError("未知ICGR路由输入模式。")
        self.router_input_mode = router_input_mode
        self.parent = parent.eval()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        input_dim = 771 if router_input_mode == "image_cls_role_cosine" else 768
        self.router = ICGRRouter(input_dim=input_dim, hidden_dim=hidden_dim)

        with torch.no_grad():
            enhanced, base_part, equal_role_part = self.parent.prototype_components()
            denominator = enhanced.norm(dim=-1).clamp_min(1e-12)
            self.register_buffer(
                "base_classifier",
                base_part / denominator.unsqueeze(-1),
                persistent=False,
            )
            self.register_buffer(
                "unweighted_role_classifier",
                3.0 * equal_role_part / denominator.view(-1, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "frozen_scale", self.parent.scale().detach().clone(), persistent=False
            )
            self.register_buffer(
                "semantic_groups",
                self.parent.semantic_group_vectors().detach().clone(),
                persistent=False,
            )

    def router_inputs(self, image_features: torch.Tensor) -> torch.Tensor:
        features = image_features.float()
        if self.router_input_mode == "image_cls":
            return features
        normalized = F.normalize(features, dim=-1)
        cosine = torch.einsum("bd,crd->bcr", normalized, self.semantic_groups)
        group_confidence = cosine.max(dim=1).values
        return torch.cat((features, group_confidence), dim=-1)

    def route_weights(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.router(self.router_inputs(image_features))

    def component_logits(
        self, image_features: torch.Tensor, class_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = F.normalize(image_features.float(), dim=-1)
        base_classifier = self.base_classifier
        role_classifier = self.unweighted_role_classifier
        if class_ids is not None:
            class_ids = class_ids.to(base_classifier.device)
            base_classifier = base_classifier.index_select(0, class_ids)
            role_classifier = role_classifier.index_select(0, class_ids)
        base_logits = images @ base_classifier.T * self.frozen_scale
        role_logits = torch.einsum("bd,crd->bcr", images, role_classifier)
        role_logits = role_logits * self.frozen_scale
        return base_logits, role_logits

    def parent_logits(
        self, image_features: torch.Tensor, class_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        base_logits, role_logits = self.component_logits(image_features, class_ids)
        return base_logits + role_logits.mean(dim=-1)

    def logits(
        self,
        image_features: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        *,
        enabled: bool = True,
    ) -> torch.Tensor:
        base_logits, role_logits = self.component_logits(image_features, class_ids)
        if not enabled:
            return base_logits + role_logits.mean(dim=-1)
        weights = self.route_weights(image_features)
        return base_logits + torch.einsum("br,bcr->bc", weights, role_logits)

    def forward(
        self, image_features: torch.Tensor, class_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.logits(image_features, class_ids)
