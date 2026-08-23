from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_fgvd_local_features(
    patches: torch.Tensor,
    top_k: int,
    device: torch.device,
    chunk_size: int = 32,
) -> torch.Tensor:
    """按冻结patch离群强度选择top-k并平均成一个局部图像向量。"""
    if patches.ndim != 3 or patches.shape[1:] != (576, 768):
        raise ValueError("patch缓存必须是[N,576,768]。")
    if not 0 < int(top_k) < 576:
        raise ValueError("top_k必须位于(0,576)。")
    pooled = []
    for start in range(0, patches.shape[0], int(chunk_size)):
        batch = patches[start : start + int(chunk_size)].to(device).float()
        centered = batch - batch.mean(dim=1, keepdim=True)
        scores = centered.abs().mean(dim=-1)
        indices = scores.topk(int(top_k), dim=1, largest=True).indices
        selected = torch.gather(
            batch, 1, indices.unsqueeze(-1).expand(-1, -1, batch.shape[-1])
        )
        pooled.append(F.normalize(selected.mean(dim=1), dim=-1).cpu())
    return torch.cat(pooled, dim=0)


class LocalPatchSemanticResidual(nn.Module):
    """用局部patch对齐与类名身份正交的局部文本残差。"""

    def __init__(
        self,
        sentence_embeddings: torch.Tensor,
        class_name_prototypes: torch.Tensor,
        max_beta: float = 10.0,
    ) -> None:
        super().__init__()
        if tuple(sentence_embeddings.shape) != (200, 8, 768):
            raise ValueError("八角色语义必须是[200,8,768]。")
        if tuple(class_name_prototypes.shape) != (200, 768):
            raise ValueError("类名原型必须是[200,768]。")
        names = F.normalize(class_name_prototypes.detach().float(), dim=-1)
        local = F.normalize(
            sentence_embeddings.detach().float().to(names.device)[:, :6].mean(dim=1),
            dim=-1,
        )
        projection = (local * names).sum(dim=-1, keepdim=True) * names
        residual = F.normalize(local - projection, dim=-1)
        if not torch.isfinite(residual).all():
            raise ValueError("局部文本残差包含NaN/Inf。")
        self.register_buffer("local_text_residual", residual)
        self.max_beta = float(max_beta)
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def beta(self) -> torch.Tensor:
        return self.max_beta * torch.tanh(self.raw_beta)

    def residual_logits(
        self, local_features: torch.Tensor, class_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        prototypes = self.local_text_residual
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device))
        return F.normalize(local_features.float(), dim=-1) @ prototypes.T

    def forward(
        self,
        parent_logits: torch.Tensor,
        local_features: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return parent_logits
        return parent_logits + self.beta() * self.residual_logits(local_features, class_ids)
