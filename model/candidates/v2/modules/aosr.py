from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveOrthogonalSentenceResidual(nn.Module):
    """固定OESR beta，学习八句全局softmax权重。"""

    def __init__(
        self,
        sentence_embeddings: torch.Tensor,
        class_name_prototypes: torch.Tensor,
        fixed_beta: float,
    ) -> None:
        super().__init__()
        if tuple(sentence_embeddings.shape) != (200, 8, 768):
            raise ValueError("AOSR句子语义必须是[200,8,768]。")
        if tuple(class_name_prototypes.shape) != (200, 768):
            raise ValueError("AOSR类名原型必须是[200,768]。")
        self.register_buffer("sentence_embeddings", sentence_embeddings.detach().float())
        self.register_buffer(
            "class_name_prototypes",
            F.normalize(class_name_prototypes.detach().float(), dim=-1),
        )
        self.register_buffer("fixed_beta", torch.tensor(float(fixed_beta)))
        self.raw_sentence_weights = nn.Parameter(torch.zeros(8))

    def sentence_weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_sentence_weights, dim=0)

    def prototypes(self, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        mixed = torch.einsum(
            "r,crd->cd", self.sentence_weights(), self.sentence_embeddings
        )
        mixed = F.normalize(mixed, dim=-1)
        names = self.class_name_prototypes
        residual = F.normalize(
            mixed - (mixed * names).sum(dim=-1, keepdim=True) * names,
            dim=-1,
        )
        if class_ids is not None:
            residual = residual.index_select(0, class_ids.to(residual.device))
        return residual

    def weight_stats(self) -> dict[str, object]:
        weights = self.sentence_weights().detach()
        return {
            "values": [float(value) for value in weights.cpu()],
            "std": float(weights.std(unbiased=False)),
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
