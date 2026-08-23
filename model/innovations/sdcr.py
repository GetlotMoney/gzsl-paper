from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceDropoutConservativeRouting(nn.Module):
    """训练期每批屏蔽一句，推理恢复完整八句。"""

    def __init__(
        self,
        sentence_embeddings: torch.Tensor,
        class_name_prototypes: torch.Tensor,
        base_sentence_weights: torch.Tensor,
        fixed_beta: float,
        max_logit_residual: float = 0.5,
        drop_count: int = 1,
    ) -> None:
        super().__init__()
        if tuple(sentence_embeddings.shape) != (200, 8, 768):
            raise ValueError("SDCR句子语义必须是[200,8,768]。")
        if tuple(class_name_prototypes.shape) != (200, 768):
            raise ValueError("SDCR类名原型必须是[200,768]。")
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
        if int(drop_count) not in (1, 2):
            raise ValueError("SDCR drop_count只允许1或2。")
        self.drop_count = int(drop_count)
        self.raw_weight_residual = nn.Parameter(torch.zeros(8))
        self.last_masked_role = -1
        self.last_masked_roles: list[int] = []

    def weight_residual(self) -> torch.Tensor:
        return self.max_logit_residual * torch.tanh(self.raw_weight_residual)

    def full_sentence_weights(self) -> torch.Tensor:
        return torch.softmax(self.base_log_weights + self.weight_residual(), dim=0)

    def active_sentence_weights(self) -> torch.Tensor:
        logits = self.base_log_weights + self.weight_residual()
        if self.training:
            roles = torch.randperm(8, device=logits.device)[: self.drop_count]
            logits = logits.clone()
            logits[roles] = -1e9
            self.last_masked_roles = [int(role) for role in roles.cpu()]
            self.last_masked_role = self.last_masked_roles[0]
        else:
            self.last_masked_role = -1
            self.last_masked_roles = []
        return torch.softmax(logits, dim=0)

    def kl_to_base(self) -> torch.Tensor:
        weights = self.full_sentence_weights()
        return (
            weights
            * (weights.clamp_min(1e-8).log() - self.base_log_weights)
        ).sum()

    def prototypes(self, *, use_dropout: bool) -> torch.Tensor:
        weights = self.active_sentence_weights() if use_dropout else self.full_sentence_weights()
        mixed = torch.einsum("r,crd->cd", weights, self.sentence_embeddings)
        mixed = F.normalize(mixed, dim=-1)
        names = self.class_name_prototypes
        return F.normalize(
            mixed - (mixed * names).sum(dim=-1, keepdim=True) * names,
            dim=-1,
        )

    def weight_stats(self) -> dict[str, object]:
        weights = self.full_sentence_weights().detach()
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
        residual = self.prototypes(use_dropout=self.training)
        if class_ids is not None:
            residual = residual.index_select(0, class_ids.to(residual.device))
        residual_logits = F.normalize(images.float(), dim=-1) @ residual.T
        return parent_logits + self.fixed_beta * residual_logits
