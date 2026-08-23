from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def semantic_neighbor_adjacency(
    prototypes: torch.Tensor, neighbor_k: int, *, mutual_only: bool = False
) -> torch.Tensor:
    """由固定语义原型构建无向top-k邻接，不包含自身。"""
    if prototypes.ndim != 2 or prototypes.shape[0] != 200:
        raise ValueError("语义近邻原型必须是[200,D]。")
    if not 1 <= int(neighbor_k) < 200:
        raise ValueError("semantic neighbor_k必须位于[1,199]。")
    normalized = F.normalize(prototypes.detach().float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-torch.inf)
    neighbors = similarity.topk(int(neighbor_k), dim=1).indices
    directed = torch.zeros((200, 200), dtype=torch.bool, device=prototypes.device)
    directed.scatter_(1, neighbors, True)
    adjacency = directed & directed.T if mutual_only else directed | directed.T
    adjacency.fill_diagonal_(False)
    return adjacency


class GatedPairEvidenceSelector(nn.Module):
    """用parent margin与三种证据差值学习top1/top2成对校正。"""

    def __init__(
        self,
        sdcr_prototypes: torch.Tensor,
        sdcr_beta: float,
        claude_prototypes: torch.Tensor,
        merge_prototypes: torch.Tensor,
        group_ids: torch.Tensor,
        margin_threshold: float,
        margin_temperature: float,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        for name, tensor in (
            ("SDCR", sdcr_prototypes),
            ("Claude", claude_prototypes),
            ("merge", merge_prototypes),
        ):
            if tuple(tensor.shape) != (200, 768):
                raise ValueError(f"GPES {name}原型必须是[200,768]。")
        if tuple(group_ids.shape) != (200,):
            raise ValueError("GPES group_ids必须是[200]。")
        if (
            feature_mean.ndim != 1
            or feature_std.shape != feature_mean.shape
            or feature_mean.numel() not in (3, 4, 12)
        ):
            raise ValueError("GPES特征统计必须是[3]、[4]或[12]。")
        self.register_buffer(
            "sdcr_prototypes", F.normalize(sdcr_prototypes.detach().float(), dim=-1)
        )
        self.register_buffer("sdcr_beta", torch.tensor(float(sdcr_beta)))
        self.register_buffer(
            "claude_prototypes", F.normalize(claude_prototypes.detach().float(), dim=-1)
        )
        self.register_buffer(
            "merge_prototypes", F.normalize(merge_prototypes.detach().float(), dim=-1)
        )
        self.register_buffer("group_ids", group_ids.detach().long())
        self.register_buffer("margin_threshold", torch.tensor(float(margin_threshold)))
        self.register_buffer("feature_mean", feature_mean.detach().float())
        self.register_buffer("feature_std", feature_std.detach().float().clamp_min(1e-6))
        self.margin_temperature = float(margin_temperature)
        self.max_delta = float(max_delta)
        self.feature_dim = int(feature_mean.numel())
        self.selector_weight = nn.Parameter(torch.zeros(self.feature_dim))
        self.selector_bias = nn.Parameter(torch.zeros(()))

    def pair_delta(self, raw_features: torch.Tensor) -> torch.Tensor:
        normalized = (raw_features.float() - self.feature_mean) / self.feature_std
        raw = normalized @ self.selector_weight + self.selector_bias
        return self.max_delta * torch.tanh(raw)

    def corrected_pair_logits(
        self, pair_logits: torch.Tensor, raw_features: torch.Tensor
    ) -> torch.Tensor:
        margin = raw_features[:, 0]
        gate = torch.sigmoid(
            (self.margin_threshold - margin) / self.margin_temperature
        )
        delta = gate * self.pair_delta(raw_features)
        return pair_logits + torch.stack((delta, -delta), dim=1)

    def stats(self) -> dict[str, object]:
        return {
            "selector_weight": [
                float(value) for value in self.selector_weight.detach().cpu()
            ],
            "selector_bias": float(self.selector_bias.detach()),
            "margin_threshold": float(self.margin_threshold),
            "margin_temperature": self.margin_temperature,
        }

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor,
        ids: torch.Tensor,
    ):
        top = logits.detach().topk(2, dim=1)
        global_ids = ids.index_select(0, top.indices.reshape(-1)).reshape_as(
            top.indices
        )
        groups = self.group_ids.index_select(
            0, global_ids.reshape(-1).to(self.group_ids.device)
        ).reshape_as(global_ids)
        same_group = groups[:, 0].eq(groups[:, 1]) & groups[:, 0].ge(0)
        normalized_images = F.normalize(images.float(), dim=-1)
        claude_logits = normalized_images @ self.claude_prototypes.index_select(0, ids).T
        merge_logits = normalized_images @ self.merge_prototypes.index_select(0, ids).T
        local_patch = patch_scores.to(logits.device).float()
        if local_patch.shape[1] == 200 and ids.numel() != 200:
            local_patch = local_patch.index_select(1, ids)
        raw_features = torch.stack(
            (
                top.values[:, 0] - top.values[:, 1],
                claude_logits.gather(1, top.indices)[:, 0]
                - claude_logits.gather(1, top.indices)[:, 1],
                merge_logits.gather(1, top.indices)[:, 0]
                - merge_logits.gather(1, top.indices)[:, 1],
                local_patch.gather(1, top.indices)[:, 0]
                - local_patch.gather(1, top.indices)[:, 1],
            ),
            dim=1,
        )
        return top, global_ids, same_group, raw_features

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        ids = (
            torch.arange(200, device=images.device)
            if class_ids is None
            else class_ids.to(images.device)
        )
        logits = parent_logits + self.sdcr_beta * (
            F.normalize(images.float(), dim=-1)
            @ self.sdcr_prototypes.index_select(0, ids).T
        )
        if not enabled:
            return logits
        top, _, same_group, raw_features = self._top2_context(
            logits, images, patch_scores, ids
        )
        corrected_pair = self.corrected_pair_logits(top.values, raw_features)
        correction = corrected_pair - top.values
        correction = correction * same_group.to(correction.dtype).unsqueeze(1)
        output = logits.clone()
        output.scatter_add_(1, top.indices, correction)
        return output


class NonlinearGatedPairSelector(GatedPairEvidenceSelector):
    """用4→8→1小型MLP学习证据差值的非线性交互。"""

    def __init__(self, *args, hidden_dim: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.selector_weight
        del self.selector_bias
        self.selector = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)
        self.hidden_dim = int(hidden_dim)

    def pair_delta(self, raw_features: torch.Tensor) -> torch.Tensor:
        normalized = (raw_features.float() - self.feature_mean) / self.feature_std
        raw = self.selector(normalized).squeeze(1)
        return self.max_delta * torch.tanh(raw)

    def stats(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "first_layer_weight_norm": float(
                self.selector[0].weight.detach().norm()
            ),
            "output_weight_norm": float(
                self.selector[-1].weight.detach().norm()
            ),
            "output_bias": float(self.selector[-1].bias.detach()),
            "margin_threshold": float(self.margin_threshold),
            "margin_temperature": self.margin_temperature,
        }


class TextOnlyGatedPairSelector(GatedPairEvidenceSelector):
    """仅使用parent margin、Claude差和merge差的patch-free选择器。"""

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top = logits.detach().topk(2, dim=1)
        global_ids = ids.index_select(0, top.indices.reshape(-1)).reshape_as(
            top.indices
        )
        groups = self.group_ids.index_select(
            0, global_ids.reshape(-1).to(self.group_ids.device)
        ).reshape_as(global_ids)
        same_group = groups[:, 0].eq(groups[:, 1]) & groups[:, 0].ge(0)
        normalized_images = F.normalize(images.float(), dim=-1)
        claude_logits = normalized_images @ self.claude_prototypes.index_select(0, ids).T
        merge_logits = normalized_images @ self.merge_prototypes.index_select(0, ids).T
        raw_features = torch.stack(
            (
                top.values[:, 0] - top.values[:, 1],
                claude_logits.gather(1, top.indices)[:, 0]
                - claude_logits.gather(1, top.indices)[:, 1],
                merge_logits.gather(1, top.indices)[:, 0]
                - merge_logits.gather(1, top.indices)[:, 1],
            ),
            dim=1,
        )
        return top, global_ids, same_group, raw_features

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None = None,
        class_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        return super().forward(
            parent_logits, images, patch_scores, class_ids, enabled
        )


class SemanticGatedPairSelector(TextOnlyGatedPairSelector):
    """增加短类名差值的四语义特征patch-free选择器。"""

    def __init__(
        self, *args, class_name_prototypes: torch.Tensor, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if tuple(class_name_prototypes.shape) != (200, 768):
            raise ValueError("S-GWPS类名原型必须是[200,768]。")
        self.register_buffer(
            "class_name_prototypes",
            F.normalize(class_name_prototypes.detach().float(), dim=-1),
        )

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, same_group, text_features = super()._top2_context(
            logits, images, patch_scores, ids
        )
        class_logits = F.normalize(
            images.float(), dim=-1
        ) @ self.class_name_prototypes.index_select(0, ids).T
        class_diff = class_logits.gather(1, top.indices)[:, 0] - class_logits.gather(
            1, top.indices
        )[:, 1]
        return (
            top,
            global_ids,
            same_group,
            torch.cat((text_features, class_diff.unsqueeze(1)), dim=1),
        )


class RoleAwareGatedPairSelector(SemanticGatedPairSelector):
    """增加八个角色句差值，以patch-free方式保留细粒度语义分歧。"""

    def __init__(
        self, *args, role_sentence_prototypes: torch.Tensor, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if tuple(role_sentence_prototypes.shape) != (200, 8, 768):
            raise ValueError("R-GWPS角色句原型必须是[200,8,768]。")
        self.register_buffer(
            "role_sentence_prototypes",
            F.normalize(role_sentence_prototypes.detach().float(), dim=-1),
        )

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, same_group, semantic_features = super()._top2_context(
            logits, images, patch_scores, ids
        )
        role_logits = torch.einsum(
            "bd,crd->bcr",
            F.normalize(images.float(), dim=-1),
            self.role_sentence_prototypes.index_select(0, ids),
        )
        role_top2 = role_logits.gather(
            1, top.indices.unsqueeze(-1).expand(-1, -1, 8)
        )
        role_diffs = role_top2[:, 0] - role_top2[:, 1]
        return (
            top,
            global_ids,
            same_group,
            torch.cat((semantic_features, role_diffs), dim=1),
        )


class CenteredRoleGatedPairSelector(RoleAwareGatedPairSelector):
    """去掉八角色公共身份，只保留样本内标准化的相对角色分歧。"""

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, same_group, features = super()._top2_context(
            logits, images, patch_scores, ids
        )
        role_diffs = features[:, -8:]
        centered = role_diffs - role_diffs.mean(dim=1, keepdim=True)
        centered = centered / centered.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        return (
            top,
            global_ids,
            same_group,
            torch.cat((features[:, :-8], centered), dim=1),
        )


class SemanticNeighborPairSelector(CenteredRoleGatedPairSelector):
    """在类名族群之外，允许固定语义top-k邻居进入成对纠错。"""

    def __init__(
        self, *args, semantic_adjacency: torch.Tensor, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if tuple(semantic_adjacency.shape) != (200, 200):
            raise ValueError("SNPS语义邻接必须是[200,200]。")
        adjacency = semantic_adjacency.detach().bool().clone()
        adjacency.fill_diagonal_(False)
        if not torch.equal(adjacency, adjacency.T):
            raise ValueError("SNPS语义邻接必须对称。")
        self.register_buffer("semantic_adjacency", adjacency)

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, same_group, features = super()._top2_context(
            logits, images, patch_scores, ids
        )
        semantic_neighbor = self.semantic_adjacency[
            global_ids[:, 0], global_ids[:, 1]
        ]
        return top, global_ids, same_group | semantic_neighbor, features
