"""Trainable visual-evidence candidates for the FRAMEWORK-V2 CUB screen."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


VISUAL_MODES = {
    "off",
    "spatial_rgve",
    "semantic_part_tokens",
    "confusion_local_refiner",
    "multiscale_part_tokens",
}


def _attention_overlap(attention: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine overlap for three [B,N,G] attention maps."""

    if attention.ndim != 3 or attention.size(-1) != 3:
        raise ValueError("视觉注意力必须为[B,N,3]。")
    normalized = F.normalize(attention.float().transpose(1, 2), dim=-1)
    return (
        (normalized[:, 0] * normalized[:, 1]).sum(dim=-1)
        + (normalized[:, 0] * normalized[:, 2]).sum(dim=-1)
        + (normalized[:, 1] * normalized[:, 2]).sum(dim=-1)
    ).mean() / 3.0


class SpatialPatchAdapter(nn.Module):
    """Zero-initialized 24x24 residual adapter with local spatial mixing."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("visual hidden_dim必须为正数。")
        self.down = nn.Conv2d(768, int(hidden_dim), kernel_size=1)
        self.spatial = nn.Conv2d(
            int(hidden_dim),
            int(hidden_dim),
            kernel_size=3,
            padding=1,
            groups=int(hidden_dim),
        )
        self.up = nn.Conv2d(int(hidden_dim), 768, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (576, 768):
            raise ValueError("视觉patch必须为[B,576,768]并保持24x24顺序。")
        original = F.normalize(patches.float(), dim=-1)
        grid = original.transpose(1, 2).reshape(patches.size(0), 768, 24, 24)
        residual = self.up(F.gelu(self.spatial(F.gelu(self.down(grid)))))
        residual = residual.flatten(2).transpose(1, 2)
        adapted = F.normalize(original + residual, dim=-1)
        anchor = (1.0 - (adapted * original).sum(dim=-1)).clamp_min(0.0).mean()
        return adapted, anchor


class ImageFusionGate(nn.Module):
    """Image-conditioned role weights and signed zero-initialized local strength."""

    def __init__(self, hidden_dim: int, max_beta: float):
        super().__init__()
        if float(max_beta) <= 0:
            raise ValueError("visual max_beta必须为正数。")
        self.trunk = nn.Sequential(nn.Linear(768, int(hidden_dim)), nn.GELU())
        self.output = nn.Linear(int(hidden_dim), 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.max_beta = float(max_beta)

    def forward(self, cls_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.output(self.trunk(F.normalize(cls_features.float(), dim=-1)))
        weights = F.softmax(values[:, :3], dim=-1)
        beta = self.max_beta * torch.tanh(values[:, 3])
        return weights, beta


class VisualEvidenceHead(nn.Module):
    """Four registered local-visual architectures sharing one score/loss contract."""

    def __init__(
        self,
        mode: str,
        *,
        hidden_dim: int,
        max_beta: float,
        confusion_topk: int,
        visual_scales: tuple[int, ...],
    ):
        super().__init__()
        if mode not in VISUAL_MODES:
            raise ValueError(f"未知visual mode：{mode}")
        if not 2 <= int(confusion_topk) <= 20:
            raise ValueError("confusion_topk必须位于[2,20]。")
        if tuple(int(value) for value in visual_scales) != (24, 12, 6):
            raise ValueError("visual_scales固定为[24,12,6]。")
        self.mode = str(mode)
        self.confusion_topk = int(confusion_topk)
        self.visual_scales = tuple(int(value) for value in visual_scales)
        self.adapter = SpatialPatchAdapter(int(hidden_dim))
        self.fusion = ImageFusionGate(int(hidden_dim), float(max_beta))
        self.part_offsets = nn.Parameter(torch.zeros(3, 768))
        self.scale_gate = nn.Linear(768, 3)
        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)

    @staticmethod
    def _deterministic_grid_pool(grid: torch.Tensor, size: int) -> torch.Tensor:
        if grid.ndim != 4 or tuple(grid.shape[-2:]) != (24, 24):
            raise ValueError("多尺度视觉池化输入必须为[B,D,24,24]。")
        if int(size) == 24:
            return grid
        if int(size) not in (12, 6):
            raise ValueError("多尺度视觉池化只接受24/12/6。")
        factor = 24 // int(size)
        return grid.reshape(
            grid.size(0), grid.size(1), int(size), factor, int(size), factor
        ).mean(dim=(3, 5))

    @staticmethod
    def _part_tokens(
        patches: torch.Tensor,
        group_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = F.normalize(group_queries.float(), dim=-1)
        similarities = torch.einsum("bnd,gd->bng", patches, queries)
        attention = F.softmax(similarities / 0.07, dim=1)
        parts = F.normalize(torch.einsum("bng,bnd->bgd", attention, patches), dim=-1)
        return parts, attention

    @staticmethod
    def _class_attention(
        patches: torch.Tensor,
        class_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B,C,G] scores and [B,N,C,G] class attention."""

        similarities = torch.einsum(
            "bnd,cgd->bncg",
            patches,
            F.normalize(class_queries.float(), dim=-1),
        )
        attention = F.softmax(similarities / 0.07, dim=1)
        scores = (attention * similarities).sum(dim=1)
        return scores, attention

    @staticmethod
    def _candidate_attention(
        patches: torch.Tensor,
        class_queries: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_queries = class_queries.index_select(
            0, candidate_ids.reshape(-1)
        ).reshape(candidate_ids.size(0), candidate_ids.size(1), 3, 768)
        similarities = torch.einsum(
            "bnd,bkgd->bnkg", patches, F.normalize(candidate_queries.float(), dim=-1)
        )
        attention = F.softmax(similarities / 0.07, dim=1)
        scores = (attention * similarities).sum(dim=1)
        return scores, attention

    @staticmethod
    def _target_attention(
        attention: torch.Tensor,
        target_class_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch = torch.arange(attention.size(0), device=attention.device)
        return attention[batch, :, target_class_ids.long(), :]

    def forward(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        class_queries: torch.Tensor,
        global_scores: torch.Tensor,
        *,
        target_class_ids: torch.Tensor | None,
        seen_classes: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        adapted, anchor = self.adapter(patch_features)
        weights, beta = self.fusion(cls_features)
        class_queries = F.normalize(class_queries.float(), dim=-1)
        mean_queries = F.normalize(
            class_queries.index_select(0, seen_classes.long()).mean(dim=0)
            + self.part_offsets,
            dim=-1,
        )
        class_count = int(class_queries.size(0))
        candidate_ids = None
        candidate_local = None
        candidate_parts = None

        if self.mode == "spatial_rgve":
            part_scores, all_attention = self._class_attention(adapted, class_queries)
            local_scores = (part_scores * weights[:, None, :]).sum(dim=-1)
            diversity = (
                _attention_overlap(self._target_attention(all_attention, target_class_ids))
                if target_class_ids is not None
                else adapted.new_zeros(())
            )
            attention_for_stats = (
                self._target_attention(all_attention, target_class_ids)
                if target_class_ids is not None
                else all_attention[:, :, 0, :]
            )
        elif self.mode in {"semantic_part_tokens", "multiscale_part_tokens"}:
            if self.mode == "semantic_part_tokens":
                parts, attention_for_stats = self._part_tokens(adapted, mean_queries)
            else:
                grid = adapted.transpose(1, 2).reshape(adapted.size(0), 768, 24, 24)
                scale_weights = F.softmax(
                    self.scale_gate(F.normalize(cls_features.float(), dim=-1)), dim=-1
                )
                scale_parts = []
                scale_attention = []
                for size in self.visual_scales:
                    pooled = self._deterministic_grid_pool(grid, size)
                    tokens = F.normalize(pooled.flatten(2).transpose(1, 2), dim=-1)
                    parts_at_scale, attention_at_scale = self._part_tokens(tokens, mean_queries)
                    scale_parts.append(parts_at_scale)
                    scale_attention.append(attention_at_scale)
                parts = F.normalize(
                    sum(
                        scale_weights[:, index, None, None] * value
                        for index, value in enumerate(scale_parts)
                    ),
                    dim=-1,
                )
                attention_for_stats = scale_attention[0]
            part_scores = torch.einsum("bgd,cgd->bcg", parts, class_queries)
            local_scores = (part_scores * weights[:, None, :]).sum(dim=-1)
            if self.mode == "multiscale_part_tokens":
                diversity = torch.stack(
                    [_attention_overlap(value) for value in scale_attention]
                ).mean()
            else:
                diversity = _attention_overlap(attention_for_stats)
        elif self.mode == "confusion_local_refiner":
            candidate_ids = global_scores.detach().topk(self.confusion_topk, dim=1).indices
            if target_class_ids is not None:
                contains = candidate_ids.eq(target_class_ids[:, None]).any(dim=1)
                candidate_ids = candidate_ids.clone()
                candidate_ids[~contains, -1] = target_class_ids[~contains]
            candidate_parts, candidate_attention = self._candidate_attention(
                adapted, class_queries, candidate_ids
            )
            candidate_local = (candidate_parts * weights[:, None, :]).sum(dim=-1)
            local_scores = global_scores.new_zeros(global_scores.shape)
            local_scores.scatter_(1, candidate_ids, candidate_local)
            attention_for_stats = candidate_attention[:, :, 0, :]
            diversity = _attention_overlap(attention_for_stats)
            part_scores = global_scores.new_zeros((global_scores.size(0), class_count, 3))
        else:
            part_scores = global_scores.new_zeros((global_scores.size(0), class_count, 3))
            local_scores = global_scores.new_zeros(global_scores.shape)
            diversity = global_scores.new_zeros(())
            attention_for_stats = global_scores.new_zeros((global_scores.size(0), 1, 3))

        return {
            "local_scores": local_scores,
            "part_scores": part_scores,
            "candidate_ids": candidate_ids,
            "candidate_local_scores": candidate_local,
            "candidate_part_scores": candidate_parts,
            "beta": beta,
            "group_weights": weights,
            "diversity_loss": diversity,
            "anchor_loss": anchor,
            "attention_overlap": _attention_overlap(attention_for_stats),
            "adapter_residual": (adapted - F.normalize(patch_features.float(), dim=-1))
            .norm(dim=-1)
            .mean(),
        }


class PaperV2VisualModel(nn.Module):
    """Existing M3 parent plus one registered trainable local-visual candidate."""

    def __init__(
        self,
        parent: nn.Module,
        *,
        visual_mode: str,
        hidden_dim: int,
        max_beta: float,
        confusion_topk: int,
        visual_scales: tuple[int, ...],
    ):
        super().__init__()
        if visual_mode not in VISUAL_MODES:
            raise ValueError(f"未知visual mode：{visual_mode}")
        self.parent = parent
        self.visual_mode = str(visual_mode)
        self.visual = VisualEvidenceHead(
            visual_mode,
            hidden_dim=int(hidden_dim),
            max_beta=float(max_beta),
            confusion_topk=int(confusion_topk),
            visual_scales=tuple(visual_scales),
        )
        self._last_visual_diagnostics: dict[str, float] = {}

    def prototypes(self) -> torch.Tensor:
        return self.parent.prototypes()

    def scale(self) -> torch.Tensor:
        return self.parent.scale()

    def topology_loss(self, adapted: torch.Tensor | None = None) -> torch.Tensor:
        return self.parent.topology_loss(adapted)

    def score_components(
        self,
        image_features: torch.Tensor,
        patch_features: torch.Tensor,
        *,
        target_class_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        prototypes = self.prototypes()
        global_scores = F.normalize(image_features.float(), dim=-1) @ prototypes.T
        if self.visual_mode == "off":
            components: dict[str, torch.Tensor | None] = {
                "local_scores": global_scores.new_zeros(global_scores.shape),
                "part_scores": global_scores.new_zeros((*global_scores.shape, 3)),
                "candidate_ids": None,
                "candidate_local_scores": None,
                "candidate_part_scores": None,
                "beta": global_scores.new_zeros((global_scores.size(0),)),
                "group_weights": global_scores.new_full((global_scores.size(0), 3), 1.0 / 3.0),
                "diversity_loss": global_scores.new_zeros(()),
                "anchor_loss": global_scores.new_zeros(()),
                "attention_overlap": global_scores.new_zeros(()),
                "adapter_residual": global_scores.new_zeros(()),
            }
        else:
            components = self.visual(
                image_features,
                patch_features,
                self.parent.tg_vpr.semantic_group_vectors(),
                global_scores,
                target_class_ids=target_class_ids,
                seen_classes=self.parent.seen_classes.to(global_scores.device),
            )
        local_scores = components["local_scores"]
        beta = components["beta"]
        assert isinstance(local_scores, torch.Tensor) and isinstance(beta, torch.Tensor)
        final_scores = (global_scores + beta[:, None] * local_scores) * self.scale()
        components.update({"global_scores": global_scores, "final_scores": final_scores})
        with torch.no_grad():
            weights = components["group_weights"]
            assert isinstance(weights, torch.Tensor)
            self._last_visual_diagnostics = {
                "visual_beta_mean": float(beta.detach().mean()),
                "visual_beta_std": float(beta.detach().std(unbiased=False)),
                "visual_beta_max_abs": float(beta.detach().abs().max()),
                "visual_group_weight_min": float(weights.detach().min()),
                "visual_group_weight_max": float(weights.detach().max()),
                "visual_attention_overlap": float(components["attention_overlap"].detach()),
                "visual_adapter_residual": float(components["adapter_residual"].detach()),
            }
        return components

    def logits(
        self,
        image_features: torch.Tensor,
        patch_features: torch.Tensor,
        class_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.score_components(image_features, patch_features)["final_scores"]
        assert isinstance(scores, torch.Tensor)
        if class_ids is not None:
            scores = scores.index_select(1, class_ids.to(scores.device).long())
        return scores

    def visual_losses(
        self,
        components: dict[str, torch.Tensor | None],
        seen_classes: torch.Tensor,
        seen_targets: torch.Tensor,
        global_targets: torch.Tensor,
        *,
        hard_margin: float,
    ) -> dict[str, torch.Tensor]:
        zero = self.prototypes().new_zeros(())
        if self.visual_mode == "off":
            return {"part": zero, "diversity": zero, "anchor": zero, "hard": zero}
        if self.visual_mode == "confusion_local_refiner":
            ids = components["candidate_ids"]
            scores = components["candidate_local_scores"]
            assert isinstance(ids, torch.Tensor) and isinstance(scores, torch.Tensor)
            matches = ids.eq(global_targets[:, None])
            if not bool(matches.any(dim=1).all()):
                raise RuntimeError("Confusion训练候选未包含真实seen标签。")
            positions = matches.float().argmax(dim=1)
            scaled = scores * self.scale()
            part = F.cross_entropy(scaled, positions)
            positive = scaled.gather(1, positions[:, None]).squeeze(1)
            negative = scaled.masked_fill(matches, float("-inf")).max(dim=1).values
            hard = F.relu(float(hard_margin) - positive + negative).mean()
        else:
            parts = components["part_scores"]
            assert isinstance(parts, torch.Tensor)
            seen_parts = parts.index_select(1, seen_classes.long()) * self.scale()
            part = sum(
                F.cross_entropy(seen_parts[:, :, group], seen_targets)
                for group in range(3)
            ) / 3.0
            hard = zero
        diversity = components["diversity_loss"]
        anchor = components["anchor_loss"]
        assert isinstance(diversity, torch.Tensor) and isinstance(anchor, torch.Tensor)
        return {"part": part, "diversity": diversity, "anchor": anchor, "hard": hard}

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = dict(self.parent.parameter_groups())
        groups["visual"] = (
            list(self.visual.parameters()) if self.visual_mode != "off" else []
        )
        return groups

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        values = self.parent.diagnostics()
        values.update(self._last_visual_diagnostics)
        values["visual_parameter_count"] = float(
            sum(parameter.numel() for parameter in self.visual.parameters())
            if self.visual_mode != "off"
            else 0
        )
        return values
