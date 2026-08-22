from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def text_resultant_lengths(sentence_embeds: torch.Tensor) -> torch.Tensor:
    if tuple(sentence_embeds.shape) != (200, 8, 768):
        raise ValueError("DPT文本输入必须是[200,8,768]。")
    sentences = F.normalize(sentence_embeds.detach().float(), dim=-1)
    return sentences.mean(dim=1).norm(dim=-1).clamp_min(1e-6)


class DistributionalPrototypeClassifier(nn.Module):
    """用八描述合向量长度建模类别文本原型置信度。"""

    def __init__(
        self,
        parent_prototypes: torch.Tensor,
        resultant_lengths: torch.Tensor,
        seenclasses: torch.Tensor,
        scale: torch.Tensor,
        *,
        max_gamma: float = 2.0,
        initial_gamma: float = 0.05,
    ):
        super().__init__()
        if tuple(parent_prototypes.shape) != (200, 768):
            raise ValueError("DPT父原型必须是[200,768]。")
        if tuple(resultant_lengths.shape) != (200,):
            raise ValueError("DPT合向量长度必须是[200]。")
        if not 0.0 < float(initial_gamma) < float(max_gamma):
            raise ValueError("DPT初始gamma必须位于(0,max_gamma)。")
        self.register_buffer(
            "parent_prototypes", F.normalize(parent_prototypes.detach(), dim=-1)
        )
        self.register_buffer("resultant_lengths", resultant_lengths.detach().float())
        self.register_buffer("seenclasses", seenclasses.detach().cpu().long())
        self.register_buffer("_scale", scale.detach().clone())
        seen_index = self.seenclasses.to(self.resultant_lengths.device)
        seen_log = self.resultant_lengths.index_select(0, seen_index).log()
        self.register_buffer("seen_log_reference", seen_log.mean().detach())
        self.max_gamma = float(max_gamma)
        ratio = float(initial_gamma) / self.max_gamma
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    def gamma(self) -> torch.Tensor:
        return self.max_gamma * torch.sigmoid(self.raw_gamma)

    def class_confidence(self) -> torch.Tensor:
        centered_log = self.resultant_lengths.log() - self.seen_log_reference
        return torch.exp(self.gamma() * centered_log)

    def prototypes(self, *, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return self.parent_prototypes
        return self.parent_prototypes * self.class_confidence().unsqueeze(-1)

    def scale(self) -> torch.Tensor:
        return self._scale

    def logits(self, image_features: torch.Tensor, class_ids=None) -> torch.Tensor:
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return F.normalize(image_features.float(), dim=-1) @ prototypes.T * self.scale()
