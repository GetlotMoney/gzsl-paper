from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if tuple(feature_mean.shape) != (4,) or tuple(feature_std.shape) != (4,):
            raise ValueError("GPES特征统计必须是[4]。")
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
        self.selector_weight = nn.Parameter(torch.zeros(4))
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
