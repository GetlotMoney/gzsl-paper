from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiGeometrySentenceRouting(nn.Module):
    """用共享文本几何规则，为每个类别生成受限的八句权重。"""

    def __init__(
        self,
        sentence_embeddings: torch.Tensor,
        class_name_prototypes: torch.Tensor,
        parent_prototypes: torch.Tensor,
        seen_class_ids: torch.Tensor,
        base_sentence_weights: torch.Tensor,
        fixed_beta: float,
        max_logit_residual: float = 0.25,
    ) -> None:
        super().__init__()
        if tuple(sentence_embeddings.shape) != (200, 8, 768):
            raise ValueError("MGSR句子语义必须是[200,8,768]。")
        if tuple(class_name_prototypes.shape) != (200, 768):
            raise ValueError("MGSR类名原型必须是[200,768]。")
        if tuple(parent_prototypes.shape) != (200, 768):
            raise ValueError("MGSR父原型必须是[200,768]。")
        if tuple(base_sentence_weights.shape) != (8,):
            raise ValueError("MGSR基础句权重必须是[8]。")

        sentences = sentence_embeddings.detach().float()
        sentence_n = F.normalize(sentences, dim=-1)
        names = F.normalize(class_name_prototypes.detach().float(), dim=-1)
        parents = F.normalize(parent_prototypes.detach().float(), dim=-1)
        class_centers = F.normalize(sentence_n.mean(dim=1), dim=-1)
        seen = seen_class_ids.to(sentence_n.device).long()
        role_centers = F.normalize(
            sentence_n.index_select(0, seen).mean(dim=0), dim=-1
        )

        geometry = torch.stack(
            (
                torch.einsum("crd,cd->cr", sentence_n, names),
                torch.einsum("crd,cd->cr", sentence_n, parents),
                torch.einsum("crd,cd->cr", sentence_n, class_centers),
                torch.einsum("crd,rd->cr", sentence_n, role_centers),
            ),
            dim=-1,
        )
        seen_geometry = geometry.index_select(0, seen)
        center = seen_geometry.mean(dim=0, keepdim=True)
        scale = seen_geometry.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        geometry = ((geometry - center) / scale).clamp(-3.0, 3.0)

        base = base_sentence_weights.detach().float().clamp_min(1e-8)
        base = base / base.sum()
        self.register_buffer("sentence_embeddings", sentences)
        self.register_buffer("class_name_prototypes", names)
        self.register_buffer("geometry_features", geometry)
        self.register_buffer("base_log_weights", base.log())
        self.register_buffer("fixed_beta", torch.tensor(float(fixed_beta)))
        self.max_logit_residual = float(max_logit_residual)
        self.raw_geometry_coefficients = nn.Parameter(torch.zeros(4))

    def logit_residuals(
        self, class_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.geometry_features
        if class_ids is not None:
            features = features.index_select(0, class_ids.to(features.device))
        raw = torch.einsum("crf,f->cr", features, self.raw_geometry_coefficients)
        return self.max_logit_residual * torch.tanh(raw)

    def class_sentence_weights(
        self, class_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = self.logit_residuals(class_ids)
        return torch.softmax(self.base_log_weights.unsqueeze(0) + residual, dim=-1)

    def kl_to_base(self, class_ids: torch.Tensor) -> torch.Tensor:
        weights = self.class_sentence_weights(class_ids)
        return (
            weights
            * (weights.clamp_min(1e-8).log() - self.base_log_weights.unsqueeze(0))
        ).sum(dim=-1).mean()

    def prototypes(self, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        sentences = self.sentence_embeddings
        names = self.class_name_prototypes
        if class_ids is not None:
            ids = class_ids.to(sentences.device)
            sentences = sentences.index_select(0, ids)
            names = names.index_select(0, ids)
        weights = self.class_sentence_weights(class_ids)
        mixed = F.normalize(torch.einsum("cr,crd->cd", weights, sentences), dim=-1)
        return F.normalize(
            mixed - (mixed * names).sum(dim=-1, keepdim=True) * names,
            dim=-1,
        )

    def routing_stats(self) -> dict[str, object]:
        weights = self.class_sentence_weights().detach()
        residuals = self.logit_residuals().detach()
        return {
            "geometry_coefficients": [
                float(value) for value in self.raw_geometry_coefficients.detach().cpu()
            ],
            "mean_weights": [float(value) for value in weights.mean(dim=0).cpu()],
            "class_variation": float(weights.std(dim=0, unbiased=False).mean()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "residual_min": float(residuals.min()),
            "residual_max": float(residuals.max()),
        }

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return parent_logits
        residual_logits = F.normalize(images.float(), dim=-1) @ self.prototypes(class_ids).T
        return parent_logits + self.fixed_beta * residual_logits
