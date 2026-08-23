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


def spatially_coherent_patch_scores(
    patches: torch.Tensor,
    text_prototypes: torch.Tensor,
    device: torch.device,
    chunk_size: int = 16,
) -> torch.Tensor:
    """用top2相似度及其24x24网格距离形成空间一致局部证据。"""
    if patches.ndim != 3 or patches.shape[1:] != (576, 768):
        raise ValueError("patch缓存必须是[N,576,768]。")
    if text_prototypes.ndim != 2 or text_prototypes.shape[1] != 768:
        raise ValueError("文本原型必须是[C,768]。")
    prototypes = F.normalize(text_prototypes.detach().float(), dim=-1).to(device)
    output = []
    max_distance = float((2 * (23 ** 2)) ** 0.5)
    for start in range(0, patches.shape[0], int(chunk_size)):
        batch = F.normalize(
            patches[start : start + int(chunk_size)].to(device).float(), dim=-1
        )
        similarities = batch @ prototypes.T
        top = similarities.topk(2, dim=1, largest=True)
        rows = top.indices.div(24, rounding_mode="floor").float()
        columns = top.indices.remainder(24).float()
        distance = torch.sqrt(
            (rows[:, 0] - rows[:, 1]).square()
            + (columns[:, 0] - columns[:, 1]).square()
        ) / max_distance
        coherence = 1.0 - distance.clamp(0.0, 1.0)
        scores = top.values.mean(dim=1) * coherence
        output.append(scores.cpu())
    return torch.cat(output, dim=0)


def multi_part_patch_scores(
    patches: torch.Tensor,
    part_text_prototypes: torch.Tensor,
    device: torch.device,
    chunk_size: int = 4,
) -> torch.Tensor:
    """每个局部句子独立寻找最匹配patch，再按类别平均六个部位证据。"""
    if patches.ndim != 3 or patches.shape[1:] != (576, 768):
        raise ValueError("patch缓存必须是[N,576,768]。")
    if part_text_prototypes.ndim != 3 or part_text_prototypes.shape[-1] != 768:
        raise ValueError("局部文本原型必须是[C,R,768]。")
    class_count, role_count, _ = part_text_prototypes.shape
    prototypes = F.normalize(
        part_text_prototypes.detach().float().reshape(-1, 768), dim=-1
    ).to(device)
    output = []
    for start in range(0, patches.shape[0], int(chunk_size)):
        batch = F.normalize(
            patches[start : start + int(chunk_size)].to(device).float(), dim=-1
        )
        similarities = batch @ prototypes.T
        best_per_role = similarities.max(dim=1).values
        scores = best_per_role.view(-1, class_count, role_count).mean(dim=-1)
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
