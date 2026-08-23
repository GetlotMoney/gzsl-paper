from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageConditionedSentenceRouting(nn.Module):
    """在固定CASR权重附近按图像CLS动态路由八句。"""

    def __init__(
        self,
        sentence_embeddings: torch.Tensor,
        class_name_prototypes: torch.Tensor,
        base_sentence_weights: torch.Tensor,
        fixed_beta: float,
        hidden_dim: int = 32,
        max_logit_residual: float = 0.5,
    ) -> None:
        super().__init__()
        if tuple(sentence_embeddings.shape) != (200, 8, 768):
            raise ValueError("ICSR句子语义必须是[200,8,768]。")
        if tuple(class_name_prototypes.shape) != (200, 768):
            raise ValueError("ICSR类名原型必须是[200,768]。")
        if tuple(base_sentence_weights.shape) != (8,):
            raise ValueError("ICSR基础句权重必须是[8]。")
        base = base_sentence_weights.detach().float().clamp_min(1e-8)
        base = base / base.sum()
        self.register_buffer("sentence_embeddings", sentence_embeddings.detach().float())
        self.register_buffer(
            "class_name_prototypes",
            F.normalize(class_name_prototypes.detach().float(), dim=-1),
        )
        self.register_buffer("base_weights", base)
        self.register_buffer("base_log_weights", base.log())
        self.register_buffer("fixed_beta", torch.tensor(float(fixed_beta)))
        self.max_logit_residual = float(max_logit_residual)
        self.gate = nn.Sequential(
            nn.Linear(768, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 8),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def sentence_weights(self, images: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(images.float(), dim=-1)
        residual = self.max_logit_residual * torch.tanh(self.gate(normalized))
        return torch.softmax(self.base_log_weights.unsqueeze(0) + residual, dim=-1)

    def kl_to_base(self, images: torch.Tensor) -> torch.Tensor:
        weights = self.sentence_weights(images)
        return (
            weights
            * (weights.clamp_min(1e-8).log() - self.base_log_weights.unsqueeze(0))
        ).sum(dim=-1).mean()

    def routing_stats(self, images: torch.Tensor) -> dict[str, object]:
        weights = self.sentence_weights(images).detach()
        mean_weights = weights.mean(dim=0)
        return {
            "mean_weights": [float(value) for value in mean_weights.cpu()],
            "image_variation": float(weights.std(dim=0, unbiased=False).mean()),
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
        weights = self.sentence_weights(images)
        sentences = self.sentence_embeddings
        names = self.class_name_prototypes
        if class_ids is not None:
            ids = class_ids.to(sentences.device)
            sentences = sentences.index_select(0, ids)
            names = names.index_select(0, ids)
        mixed = torch.einsum("br,crd->bcd", weights, sentences)
        mixed = F.normalize(mixed, dim=-1)
        residual = F.normalize(
            mixed - (mixed * names.unsqueeze(0)).sum(dim=-1, keepdim=True)
            * names.unsqueeze(0),
            dim=-1,
        )
        normalized = F.normalize(images.float(), dim=-1)
        residual_logits = torch.einsum("bd,bcd->bc", normalized, residual)
        return parent_logits + self.fixed_beta * residual_logits
