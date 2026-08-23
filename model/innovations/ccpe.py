from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_conditioned_patch_scores(
    patches: torch.Tensor,
    text_prototypes: torch.Tensor,
    top_k: int,
    device: torch.device,
    chunk_size: int = 16,
) -> torch.Tensor:
    """每个类别独立选择最匹配的top-k patch，返回[N,C]局部证据。"""
    if patches.ndim != 3 or patches.shape[1:] != (576, 768):
        raise ValueError("patch缓存必须是[N,576,768]。")
    if text_prototypes.ndim != 2 or text_prototypes.shape[1] != 768:
        raise ValueError("文本原型必须是[C,768]。")
    if not 0 < int(top_k) < 576:
        raise ValueError("top_k必须位于(0,576)。")
    prototypes = F.normalize(text_prototypes.detach().float(), dim=-1).to(device)
    output = []
    for start in range(0, patches.shape[0], int(chunk_size)):
        batch = F.normalize(
            patches[start : start + int(chunk_size)].to(device).float(), dim=-1
        )
        similarities = batch @ prototypes.T
        scores = similarities.topk(int(top_k), dim=1, largest=True).values.mean(dim=1)
        output.append(scores.cpu())
    return torch.cat(output, dim=0)


class ClassConditionedPatchEvidence(nn.Module):
    """把预计算的类别条件局部patch证据作为有界logit残差。"""

    def __init__(self, max_beta: float = 10.0) -> None:
        super().__init__()
        self.max_beta = float(max_beta)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self) -> torch.Tensor:
        return self.max_beta * torch.tanh(self.raw_beta)

    def forward(
        self,
        parent_logits: torch.Tensor,
        patch_scores: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return parent_logits
        scores = patch_scores
        if class_ids is not None and scores.shape[1] != parent_logits.shape[1]:
            scores = scores.index_select(1, class_ids.to(scores.device))
        if scores.shape != parent_logits.shape:
            raise ValueError("patch_scores与parent_logits形状不一致。")
        return parent_logits + self.beta() * scores
