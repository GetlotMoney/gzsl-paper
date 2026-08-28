from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassConditionedSentenceRouting(nn.Module):
    """固定CASR全局权重，用句子-类名独立度学习共享类别斜率。"""

    def __init__(
        self,
        sentence_embeddings: torch.Tensor,
        class_name_prototypes: torch.Tensor,
        seen_class_ids: torch.Tensor,
        base_sentence_weights: torch.Tensor,
        fixed_beta: float,
        max_delta: float = 2.0,
    ) -> None:
        super().__init__()
        if tuple(sentence_embeddings.shape) != (200, 8, 768):
            raise ValueError("CCSR句子语义必须是[200,8,768]。")
        if tuple(class_name_prototypes.shape) != (200, 768):
            raise ValueError("CCSR类名原型必须是[200,768]。")
        if tuple(base_sentence_weights.shape) != (8,):
            raise ValueError("CCSR基础句权重必须是[8]。")
        sentences = sentence_embeddings.detach().float()
        names = F.normalize(class_name_prototypes.detach().float(), dim=-1)
        sentence_n = F.normalize(sentences, dim=-1)
        uniqueness = 1.0 - torch.einsum("crd,cd->cr", sentence_n, names)
        seen = seen_class_ids.to(uniqueness.device).long()
        seen_values = uniqueness.index_select(0, seen)
        center = seen_values.mean(dim=0, keepdim=True)
        scale = seen_values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        features = ((uniqueness - center) / scale).clamp(-2.0, 2.0)
        base = base_sentence_weights.detach().float().clamp_min(1e-8)
        base = base / base.sum()
        self.register_buffer("sentence_embeddings", sentences)
        self.register_buffer("class_name_prototypes", names)
        self.register_buffer("uniqueness_features", features)
        self.register_buffer("base_log_weights", base.log())
        self.register_buffer("fixed_beta", torch.tensor(float(fixed_beta)))
        self.max_delta = float(max_delta)
        self.raw_delta = nn.Parameter(torch.zeros(()))

    def delta(self) -> torch.Tensor:
        return self.max_delta * torch.tanh(self.raw_delta)

    def class_sentence_weights(
        self, class_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.uniqueness_features
        if class_ids is not None:
            features = features.index_select(0, class_ids.to(features.device))
        return torch.softmax(
            self.base_log_weights.unsqueeze(0) + self.delta() * features,
            dim=-1,
        )

    def prototypes(self, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        sentences = self.sentence_embeddings
        names = self.class_name_prototypes
        weights = self.class_sentence_weights()
        mixed = torch.einsum("cr,crd->cd", weights, sentences)
        mixed = F.normalize(mixed, dim=-1)
        residual = F.normalize(
            mixed - (mixed * names).sum(dim=-1, keepdim=True) * names,
            dim=-1,
        )
        if class_ids is not None:
            residual = residual.index_select(0, class_ids.to(residual.device))
        return residual

    def routing_stats(self) -> dict[str, object]:
        weights = self.class_sentence_weights().detach()
        mean_weights = weights.mean(dim=0)
        return {
            "delta": float(self.delta().detach()),
            "mean_weights": [float(value) for value in mean_weights.cpu()],
            "class_variation": float(weights.std(dim=0, unbiased=False).mean()),
            "min": float(weights.min()),
            "max": float(weights.max()),
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
