"""Balanced-decoupled tangent semantic transport (BD-TST)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.elpt import fixed_class_folds, gate_features, topology_loss
from model.innovations.tst import TangentStepGate, tangent_transport


CLASS_COUNT = 200
SEEN_COUNT = 150


def _validated_fold(
    seen_classes: torch.Tensor,
    pseudo_seen: torch.Tensor,
    pseudo_unseen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seen = torch.as_tensor(seen_classes).detach().cpu().long().sort().values
    fold_seen = torch.as_tensor(pseudo_seen).detach().cpu().long().sort().values
    fold_unseen = torch.as_tensor(pseudo_unseen).detach().cpu().long().sort().values
    if seen.numel() != SEEN_COUNT or seen.unique().numel() != SEEN_COUNT:
        raise ValueError("BD-TST固定要求150个seen训练类。")
    if fold_seen.numel() != 100 or fold_unseen.numel() != 50:
        raise ValueError("BD-TST每折固定100个pseudo-seen和50个pseudo-unseen类。")
    if torch.isin(fold_seen, fold_unseen).any():
        raise ValueError("BD-TST的pseudo-seen与pseudo-unseen必须互斥。")
    if not torch.equal(torch.cat((fold_seen, fold_unseen)).sort().values, seen):
        raise ValueError("BD-TST每折必须完整覆盖150个seen训练类。")
    return fold_seen, fold_unseen


class BalancedDecoupledTST(nn.Module):
    """TG parent and a Gate whose gradients come only from balanced seen folds."""

    def __init__(
        self,
        parent: nn.Module,
        seen_classes: torch.Tensor,
        *,
        gate_enabled: bool = True,
        gate_hidden_dim: int = 16,
        max_transport_step: float = 1.5,
        initial_transport_step: float = 0.1,
        gate_initialization_seed: int = 151,
    ):
        super().__init__()
        if int(gate_hidden_dim) != 16:
            raise ValueError("BD-TST首轮固定Gate hidden_dim=16。")
        if float(max_transport_step) != 1.5 or float(initial_transport_step) != 0.1:
            raise ValueError("BD-TST首轮固定max step=1.5、initial step=0.1。")
        seen = torch.as_tensor(seen_classes).detach().cpu().long().sort().values
        if seen.numel() != SEEN_COUNT or seen.unique().numel() != SEEN_COUNT:
            raise ValueError("BD-TST CUB固定150个唯一seen类。")
        all_classes = torch.arange(CLASS_COUNT)
        unseen = all_classes[~torch.isin(all_classes, seen)]
        if unseen.numel() != 50:
            raise ValueError("BD-TST CUB固定50个true-unseen类。")
        if not hasattr(parent, "tg_vpr") or not hasattr(parent, "parameter_groups"):
            raise TypeError("BD-TST parent必须提供TG-VPR和参数组接口。")
        if getattr(parent, "transport_mode", None) != "off" or getattr(
            parent, "ccgr_mode", None
        ) != "off":
            raise ValueError("BD-TST只能包裹纯TG parent，禁止继承TST/NTR/CCGR失败路径。")

        self.parent = parent
        self.gate_enabled = bool(gate_enabled)
        self.max_transport_step = float(max_transport_step)
        self.register_buffer("seen_classes", seen, persistent=True)
        self.register_buffer("unseen_classes", unseen, persistent=True)
        # Gate初始化不得推进TG/dropout使用的全局CPU RNG；这保证off路径训练轨迹可逐位复现。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(gate_initialization_seed))
            self.gate = TangentStepGate(
                input_dim=4,
                max_step=max_transport_step,
                initial_step=initial_transport_step,
            )
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        for parameter in self.parent.parameter_groups()["tg_vpr"]:
            parameter.requires_grad_(True)

    def scale(self) -> torch.Tensor:
        return self.parent.scale()

    def tg_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.parent.parameter_groups()["tg_vpr"]
            if parameter.requires_grad
        ]

    def gate_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.gate.parameters() if parameter.requires_grad]

    def folds(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return fixed_class_folds(self.seen_classes)

    @torch.no_grad()
    def _detached_value_candidates(self, class_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(class_ids).to(self.parent.tg_vpr.sentence_embeds.device).long()
        return self.parent.tg_vpr.value_candidate(ids).detach()

    @torch.no_grad()
    def _detached_fold_prototypes(self, pseudo_seen: torch.Tensor) -> torch.Tensor:
        """Inference-style TG fold using shared current weights and no auxiliary dropout RNG."""
        tg = self.parent.tg_vpr
        device = tg.sentence_embeds.device
        adapted = torch.as_tensor(pseudo_seen).to(device).long()
        source = tg.semantic_group_vectors().index_select(0, adapted)
        count, group_count, dim = source.shape
        value = tg.tg_value_projection(source)
        value = value.view(count, group_count, 1, dim).transpose(1, 2)
        weights = tg.semantic_group_weights().view(1, 1, 1, group_count).expand(
            count, 1, group_count, group_count
        )
        context = torch.einsum("bhqg,bhgd->bhqd", weights, value)
        context = context.transpose(1, 2).contiguous().view(count, group_count, dim)
        context = tg.tg_output_projection(context)
        context = tg.post_projection(context)
        mixed = tg.inner_ratio * context + (1.0 - tg.inner_ratio) * source
        groups = F.normalize(tg.layer_norm(2.0 * mixed), dim=-1)

        base_vectors = tg.base_vectors()
        semantic_groups = tg.semantic_group_vectors()
        grouped = F.normalize(
            (
                tg.semantic_group_weights().view(1, 3, 1)
                * semantic_groups
            ).sum(dim=1),
            dim=-1,
        )
        candidate = base_vectors.clone()
        candidate[adapted] = grouped.index_select(0, adapted)
        base_scale = candidate.new_ones((CLASS_COUNT,))
        base_scale[adapted] = 1.0 - tg.outer_ratio
        enhanced = base_scale.unsqueeze(-1) * candidate
        enhanced[adapted] = enhanced.index_select(0, adapted) + (
            tg.outer_ratio
            * tg.semantic_group_weights().view(1, 3, 1)
            * groups
        ).sum(dim=1)
        return F.normalize(enhanced, dim=-1).detach()

    def main_objective(
        self,
        image_features: torch.Tensor,
        targets_on_seen_axis: torch.Tensor,
        topology_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Standard TG batch: CE and topology have no path to the Gate."""
        if float(topology_weight) != 0.1:
            raise ValueError("BD-TST主路固定topology权重0.1。")
        prototypes = self.parent.prototypes()
        seen = self.seen_classes.to(prototypes.device)
        logits = (
            F.normalize(image_features.float(), dim=-1)
            @ prototypes.index_select(0, seen).T
            * self.parent.scale()
        )
        ce = F.cross_entropy(logits, targets_on_seen_axis.long())
        topology = self.parent.topology_loss(prototypes)
        return {
            "loss": ce + float(topology_weight) * topology,
            "ce": ce,
            "topology": topology,
            "logits": logits,
        }

    def auxiliary_objective(
        self,
        image_features: torch.Tensor,
        global_targets: torch.Tensor,
        pseudo_seen: torch.Tensor,
        pseudo_unseen: torch.Tensor,
        topology_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Balanced 25+25 images, but logits compete over all 150 fold classes."""
        if not self.gate_enabled:
            raise RuntimeError("Gate关闭时禁止执行辅助优化路径。")
        if float(topology_weight) != 0.1:
            raise ValueError("BD-TST辅助路固定fold-topology权重0.1。")
        fold_seen, fold_unseen = _validated_fold(
            self.seen_classes, pseudo_seen, pseudo_unseen
        )
        labels = torch.as_tensor(global_targets).detach().cpu().long()
        if labels.numel() != image_features.size(0) or not torch.isin(
            labels, self.seen_classes.cpu()
        ).all():
            raise ValueError("BD-TST辅助batch只能来自150类trainval图像。")
        pseudo_seen_count = int(torch.isin(labels, fold_seen).sum())
        pseudo_unseen_count = int(torch.isin(labels, fold_unseen).sum())
        if image_features.size(0) != 50 or (pseudo_seen_count, pseudo_unseen_count) != (
            25,
            25,
        ):
            raise ValueError("BD-TST辅助batch必须严格为25 pseudo-seen + 25 pseudo-unseen。")

        device = image_features.device
        seen = self.seen_classes.to(device)
        fold_seen_device = fold_seen.to(device)
        fold_unseen_device = fold_unseen.to(device)
        base_all = self.parent.tg_vpr.base_prototypes().detach()
        fold_all = self._detached_fold_prototypes(fold_seen).to(device).clone()
        value = self._detached_value_candidates(fold_unseen).to(device)
        features = gate_features(
            base_all.index_select(0, fold_unseen_device),
            value,
            base_all.index_select(0, fold_seen_device),
            mode="summary",
        ).detach()
        step = self.gate(features)
        fold_all[fold_unseen_device] = tangent_transport(
            base_all.index_select(0, fold_unseen_device), value, step
        )
        competition = fold_all.index_select(0, seen)
        global_to_seen = torch.full((CLASS_COUNT,), -1, dtype=torch.long, device=device)
        global_to_seen[seen] = torch.arange(SEEN_COUNT, device=device)
        targets = global_to_seen.index_select(0, labels.to(device))
        logits = (
            F.normalize(image_features.float(), dim=-1)
            @ competition.T
            * self.parent.scale().detach()
        )
        ce = F.cross_entropy(logits, targets)
        topology = topology_loss(base_all.index_select(0, seen), competition)
        return {
            "loss": ce + float(topology_weight) * topology,
            "ce": ce,
            "topology": topology,
            "logits": logits,
            "step": step,
            "gate_features": features,
        }

    def prototype_bundle(self) -> dict[str, torch.Tensor]:
        parent = self.parent.prototypes()
        if not self.gate_enabled:
            return {"parent": parent, "final": parent, "step": parent.new_zeros((50,))}
        unseen = self.unseen_classes.to(parent.device)
        seen = self.seen_classes.to(parent.device)
        base_all = self.parent.tg_vpr.base_prototypes().detach()
        value = self._detached_value_candidates(unseen).to(parent.device)
        features = gate_features(
            base_all.index_select(0, unseen),
            value,
            base_all.index_select(0, seen),
            mode="summary",
        ).detach()
        step = self.gate(features)
        final = parent.clone()
        final[unseen] = tangent_transport(
            base_all.index_select(0, unseen), value, step
        )
        return {"parent": parent, "final": final, "step": step}

    def prototypes(self) -> torch.Tensor:
        return self.prototype_bundle()["final"]

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        bundle = self.prototype_bundle()
        step = bundle["step"]
        return {
            "gate_enabled": float(self.gate_enabled),
            "true_unseen_step_mean": float(step.mean()),
            "true_unseen_step_std": float(step.std(unbiased=False)),
            "true_unseen_step_min": float(step.min()),
            "true_unseen_step_max": float(step.max()),
        }
