from __future__ import annotations

import torch
import torch.nn.functional as F


def ridge_predict_local_visual(
    semantic_prototypes: torch.Tensor,
    seen_class_ids: torch.Tensor,
    visual_centroids: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    semantics = F.normalize(semantic_prototypes.float(), dim=-1)
    seen = seen_class_ids.to(semantics.device).long()
    targets = F.normalize(visual_centroids.float().to(semantics.device), dim=-1)
    if targets.shape != (seen.numel(), semantics.shape[1]):
        raise ValueError("局部视觉中心与seen语义形状不一致。")
    seen_semantics = semantics.index_select(0, seen)
    kernel = seen_semantics @ seen_semantics.T
    regularized = kernel + float(ridge) * torch.eye(seen.numel(), device=semantics.device)
    coefficients = torch.linalg.solve(regularized, targets)
    return F.normalize(semantics @ seen_semantics.T @ coefficients, dim=-1)


def fit_local_visual_prototypes(
    patches: torch.Tensor,
    labels: torch.Tensor,
    semantic_prototypes: torch.Tensor,
    seen_class_ids: torch.Tensor,
    *,
    top_k: int,
    ridge: float,
    device: torch.device,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从seen真类局部patch中心拟合语义到局部视觉原型的线性ridge映射。"""
    if patches.ndim != 3 or patches.shape[1:] != (576, 768):
        raise ValueError("patch缓存必须是[N,576,768]。")
    if labels.ndim != 1 or labels.shape[0] != patches.shape[0]:
        raise ValueError("patch标签数量不一致。")
    if tuple(semantic_prototypes.shape) != (200, 768):
        raise ValueError("局部语义原型必须是[200,768]。")
    if seen_class_ids.numel() != 150:
        raise ValueError("LVPG必须使用150个seen类。")
    if int(top_k) != 2 or float(ridge) <= 0.0:
        raise ValueError("LVPG固定top_k=2且ridge必须为正。")

    semantics = F.normalize(semantic_prototypes.detach().float(), dim=-1).to(device)
    seen = seen_class_ids.detach().cpu().long()
    global_to_seen = torch.full((200,), -1, dtype=torch.long)
    global_to_seen[seen] = torch.arange(150)
    sums = torch.zeros(150, 768, device=device)
    counts = torch.zeros(150, device=device)
    for start in range(0, patches.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), patches.shape[0])
        batch = F.normalize(patches[start:stop].to(device).float(), dim=-1)
        batch_labels = labels[start:stop].long()
        local_semantics = semantics.index_select(0, batch_labels.to(device))
        similarities = torch.einsum("bnd,bd->bn", batch, local_semantics)
        indices = similarities.topk(int(top_k), dim=1, largest=True).indices
        selected = torch.gather(
            batch, 1, indices.unsqueeze(-1).expand(-1, -1, 768)
        )
        local_vectors = F.normalize(selected.mean(dim=1), dim=-1)
        local_ids = global_to_seen.index_select(0, batch_labels).to(device)
        sums.index_add_(0, local_ids, local_vectors)
        counts.index_add_(0, local_ids, torch.ones_like(local_ids, dtype=torch.float))
    if (counts == 0).any():
        raise ValueError("LVPG存在没有训练图像的seen类。")
    visual_centroids = F.normalize(sums / counts.unsqueeze(1), dim=-1)

    predicted = ridge_predict_local_visual(
        semantics, seen.to(device), visual_centroids, float(ridge)
    )
    return predicted, visual_centroids
