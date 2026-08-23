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


def reciprocal_neighbor_confidence(
    prototypes: torch.Tensor, neighbor_k: int
) -> torch.Tensor:
    """互为top-k记1，单向top-k记0.5，其余记0。"""
    normalized = F.normalize(prototypes.detach().float(), dim=-1)
    if normalized.ndim != 2 or normalized.shape[0] != 200:
        raise ValueError("语义近邻原型必须是[200,D]。")
    if not 1 <= int(neighbor_k) < 200:
        raise ValueError("semantic neighbor_k必须位于[1,199]。")
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-torch.inf)
    neighbors = similarity.topk(int(neighbor_k), dim=1).indices
    directed = torch.zeros((200, 200), dtype=torch.bool, device=prototypes.device)
    directed.scatter_(1, neighbors, True)
    confidence = 0.5 * (directed.float() + directed.T.float())
    confidence.fill_diagonal_(0.0)
    return confidence


def pair_role_distance_weights(
    role_sentence_prototypes: torch.Tensor, class_pairs: torch.Tensor
) -> torch.Tensor:
    """按两类在八个语义角色上的余弦距离生成均值为1的pair权重。"""
    if tuple(role_sentence_prototypes.shape) != (200, 8, 768):
        raise ValueError("pair角色原型必须是[200,8,768]。")
    if class_pairs.ndim != 2 or class_pairs.shape[1] != 2:
        raise ValueError("class_pairs必须是[N,2]。")
    normalized = F.normalize(role_sentence_prototypes.float(), dim=-1)
    first = normalized.index_select(0, class_pairs[:, 0].long())
    second = normalized.index_select(0, class_pairs[:, 1].long())
    distance = (1.0 - (first * second).sum(dim=-1)).clamp_min(0.0)
    return distance / distance.mean(dim=1, keepdim=True).clamp_min(1e-6)


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
            or feature_mean.numel() not in (3, 4, 12, 13)
        ):
            raise ValueError("GPES特征统计必须是[3]、[4]、[12]或[13]。")
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


class ReciprocalSemanticNeighborPairSelector(CenteredRoleGatedPairSelector):
    """按互惠性连续缩放语义邻居的训练外推修正。"""

    def __init__(
        self, *args, semantic_confidence: torch.Tensor, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if tuple(semantic_confidence.shape) != (200, 200):
            raise ValueError("R-SNPS语义置信度必须是[200,200]。")
        confidence = semantic_confidence.detach().float().clone()
        if not torch.isfinite(confidence).all() or bool(
            ((confidence < 0) | (confidence > 1)).any()
        ):
            raise ValueError("R-SNPS语义置信度必须有限且位于[0,1]。")
        confidence.fill_diagonal_(0.0)
        if not torch.allclose(confidence, confidence.T):
            raise ValueError("R-SNPS语义置信度必须对称。")
        self.register_buffer("semantic_confidence", confidence)

    def forward(
        self,
        parent_logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None = None,
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
        top, global_ids, same_group, raw_features = super()._top2_context(
            logits, images, patch_scores, ids
        )
        confidence = self.semantic_confidence[
            global_ids[:, 0], global_ids[:, 1]
        ]
        confidence = torch.where(
            same_group, torch.ones_like(confidence), confidence
        )
        corrected_pair = self.corrected_pair_logits(top.values, raw_features)
        correction = (corrected_pair - top.values) * confidence.unsqueeze(1)
        output = logits.clone()
        output.scatter_add_(1, top.indices, correction)
        return output


class TriadicCompetitionPairSelector(SemanticNeighborPairSelector):
    """用top2与top3间隔补充二元pair纠错的第三类竞争上下文。"""

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, related, features = super()._top2_context(
            logits, images, patch_scores, ids
        )
        top3 = logits.detach().topk(3, dim=1)
        third_gap = top3.values[:, 1] - top3.values[:, 2]
        return (
            top,
            global_ids,
            related,
            torch.cat((features, third_gap.unsqueeze(1)), dim=1),
        )


class PairDiscriminativeRoleSelector(SemanticNeighborPairSelector):
    """按当前类别对的角色文本距离重加权图像角色分歧。"""

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, related, features = super()._top2_context(
            logits, images, patch_scores, ids
        )
        weights = pair_role_distance_weights(
            self.role_sentence_prototypes, global_ids
        )
        weighted_roles = features[:, -8:] * weights
        return (
            top,
            global_ids,
            related,
            torch.cat((features[:, :-8], weighted_roles), dim=1),
        )


class RoleDisagreementScaleSelector(SemanticNeighborPairSelector):
    """在中心化角色方向之外恢复归一化前的角色分歧尺度。"""

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, related, features = super()._top2_context(
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
        role_scale = (role_top2[:, 0] - role_top2[:, 1]).std(
            dim=1, unbiased=False
        )
        return (
            top,
            global_ids,
            related,
            torch.cat((features, role_scale.unsqueeze(1)), dim=1),
        )


class StagedRoleDisagreementScaleSelector(RoleDisagreementScaleSelector):
    """冻结已训练12维SNPS选择器，只训练新增角色尺度系数。"""

    def __init__(
        self,
        *args,
        base_selector_weight: torch.Tensor,
        base_selector_bias: torch.Tensor,
        base_feature_mean: torch.Tensor,
        base_feature_std: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.feature_dim != 13:
            raise ValueError("S-RDSS输入必须是13维。")
        if tuple(base_selector_weight.shape) != (12,):
            raise ValueError("S-RDSS父selector_weight必须是[12]。")
        if tuple(base_feature_mean.shape) != (12,) or tuple(
            base_feature_std.shape
        ) != (12,):
            raise ValueError("S-RDSS父特征统计必须是[12]。")
        if not torch.allclose(
            self.feature_mean[:12].cpu(), base_feature_mean.float().cpu(), atol=1e-6
        ) or not torch.allclose(
            self.feature_std[:12].cpu(), base_feature_std.float().cpu(), atol=1e-6
        ):
            raise ValueError("S-RDSS前12维特征统计未复现SNPS父模型。")
        del self.selector_weight
        del self.selector_bias
        self.register_buffer(
            "base_selector_weight", base_selector_weight.detach().float()
        )
        self.register_buffer(
            "base_selector_bias", base_selector_bias.detach().float().reshape(())
        )
        self.register_buffer(
            "base_feature_mean", base_feature_mean.detach().float()
        )
        self.register_buffer(
            "base_feature_std", base_feature_std.detach().float().clamp_min(1e-6)
        )
        self.scale_weight = nn.Parameter(torch.zeros(()))

    def pair_delta(self, raw_features: torch.Tensor) -> torch.Tensor:
        base_features = (
            raw_features[:, :12].float() - self.base_feature_mean
        ) / self.base_feature_std
        scale_feature = (
            raw_features[:, 12].float() - self.feature_mean[12]
        ) / self.feature_std[12]
        raw = (
            base_features @ self.base_selector_weight
            + self.base_selector_bias
            + scale_feature * self.scale_weight
        )
        return self.max_delta * torch.tanh(raw)

    def stats(self) -> dict[str, object]:
        return {
            "scale_weight": float(self.scale_weight.detach()),
            "base_selector_weight_norm": float(self.base_selector_weight.norm()),
            "base_selector_bias": float(self.base_selector_bias),
            "margin_threshold": float(self.margin_threshold),
            "margin_temperature": self.margin_temperature,
        }


class TrustRegionRoleDisagreementScaleSelector(RoleDisagreementScaleSelector):
    """从SNPS初始化13维联合训练，并约束旧12维留在父权重邻域。"""

    def __init__(
        self,
        *args,
        base_selector_weight: torch.Tensor,
        base_selector_bias: torch.Tensor,
        base_feature_mean: torch.Tensor,
        base_feature_std: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.feature_dim != 13 or tuple(base_selector_weight.shape) != (12,):
            raise ValueError("TR-RDSS要求13维输入和12维父权重。")
        if not torch.allclose(
            self.feature_mean[:12].cpu(), base_feature_mean.float().cpu(), atol=1e-6
        ) or not torch.allclose(
            self.feature_std[:12].cpu(), base_feature_std.float().cpu(), atol=1e-6
        ):
            raise ValueError("TR-RDSS前12维特征统计未复现SNPS父模型。")
        with torch.no_grad():
            self.selector_weight[:12].copy_(base_selector_weight.float())
            self.selector_weight[12].zero_()
            self.selector_bias.copy_(base_selector_bias.float().reshape(()))
        self.register_buffer(
            "base_selector_weight", base_selector_weight.detach().float()
        )
        self.register_buffer(
            "base_selector_bias", base_selector_bias.detach().float().reshape(())
        )

    def trust_region_loss(self) -> torch.Tensor:
        return (
            (self.selector_weight[:12] - self.base_selector_weight).square().mean()
            + (self.selector_bias - self.base_selector_bias).square()
        )

    def stats(self) -> dict[str, object]:
        base_stats = super().stats()
        base_stats.update(
            {
                "role_scale_weight": float(self.selector_weight[12].detach()),
                "base_weight_drift": float(
                    (self.selector_weight[:12] - self.base_selector_weight)
                    .detach()
                    .norm()
                ),
                "base_bias_drift": float(
                    (self.selector_bias - self.base_selector_bias).detach().abs()
                ),
            }
        )
        return base_stats


class RoleVotePairSelector(SemanticNeighborPairSelector):
    """增加八角色对top1/top2的有符号多数投票。"""

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, related, features = super()._top2_context(
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
        vote = torch.sign(role_top2[:, 0] - role_top2[:, 1]).mean(dim=1)
        return (
            top,
            global_ids,
            related,
            torch.cat((features, vote.unsqueeze(1)), dim=1),
        )


class CrossSourceDisagreementSelector(SemanticNeighborPairSelector):
    """增加Claude与merge pair差值的绝对分歧。"""

    def _top2_context(
        self,
        logits: torch.Tensor,
        images: torch.Tensor,
        patch_scores: torch.Tensor | None,
        ids: torch.Tensor,
    ):
        top, global_ids, related, features = super()._top2_context(
            logits, images, patch_scores, ids
        )
        disagreement = (features[:, 1] - features[:, 2]).abs()
        return (
            top,
            global_ids,
            related,
            torch.cat((features, disagreement.unsqueeze(1)), dim=1),
        )


class RoleUncertaintyGatedSelector(StagedRoleDisagreementScaleSelector):
    """冻结SNPS方向，用非负gamma按角色分歧乘法衰减pair delta。"""

    def __init__(self, *args, max_gamma: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.scale_weight
        self.gamma = nn.Parameter(torch.zeros(()))
        self.max_gamma = float(max_gamma)
        if self.max_gamma <= 0:
            raise ValueError("RUGS max_gamma必须为正。")

    def pair_delta(self, raw_features: torch.Tensor) -> torch.Tensor:
        base_features = (
            raw_features[:, :12].float() - self.base_feature_mean
        ) / self.base_feature_std
        base_raw = (
            base_features @ self.base_selector_weight + self.base_selector_bias
        )
        base_delta = self.max_delta * torch.tanh(base_raw)
        scale_ratio = (
            raw_features[:, 12].float()
            / self.feature_mean[12].clamp_min(1e-6)
        ).clamp(min=0.0, max=10.0)
        return base_delta * torch.exp(-self.gamma * scale_ratio)

    @torch.no_grad()
    def project_parameters(self) -> None:
        self.gamma.clamp_(0.0, self.max_gamma)

    def stats(self) -> dict[str, object]:
        return {
            "gamma": float(self.gamma.detach()),
            "max_gamma": self.max_gamma,
            "base_selector_weight_norm": float(self.base_selector_weight.norm()),
            "base_selector_bias": float(self.base_selector_bias),
            "margin_threshold": float(self.margin_threshold),
            "margin_temperature": self.margin_temperature,
        }
