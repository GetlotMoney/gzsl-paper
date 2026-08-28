"""Minimum-margin target tangent transport for zero-shot class prototypes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.elpt import VariableClassTGVPR, fixed_class_folds


STATUS_ALREADY_SAFE = 0
STATUS_FEASIBLE_MOVE = 1
STATUS_HARMFUL_OR_INVALID = 2
STATUS_CAP_INFEASIBLE = 3


def _require_matrix(name: str, value: torch.Tensor, rows: int | None = None) -> None:
    if value.ndim != 2 or value.size(1) != 768 or (rows is not None and value.size(0) != rows):
        expected = f"[{rows},768]" if rows is not None else "[N,768]"
        raise ValueError(f"{name}必须是{expected}。")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}包含NaN/Inf。")


def geodesic_basis(
    base: torch.Tensor,
    value: torch.Tensor,
    *,
    global_theta_max: float,
    tangent_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unit tangent, per-class cap and direction validity."""
    _require_matrix("base", base)
    _require_matrix("value", value, base.size(0))
    if not 0.0 < float(global_theta_max) < math.pi / 2:
        raise ValueError("MMT全局theta上限必须位于(0,pi/2)。")
    if float(tangent_epsilon) <= 0.0:
        raise ValueError("MMT tangent_epsilon必须为正数。")
    base = F.normalize(base.float(), dim=-1)
    value = F.normalize(value.float(), dim=-1)
    cosine = (base * value).sum(dim=-1).clamp(-1.0, 1.0)
    tangent = value - cosine.unsqueeze(-1) * base
    norm = tangent.norm(dim=-1)
    valid = norm > float(tangent_epsilon)
    direction = tangent / norm.clamp_min(float(tangent_epsilon)).unsqueeze(-1)
    direction = torch.where(valid.unsqueeze(-1), direction, torch.zeros_like(direction))
    arc = torch.acos(cosine)
    cap = torch.minimum(arc, arc.new_full(arc.shape, float(global_theta_max)))
    cap = torch.where(valid, cap, torch.zeros_like(cap))
    return direction, cap, valid


def geodesic_transport(
    base: torch.Tensor,
    direction: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    _require_matrix("base", base)
    _require_matrix("direction", direction, base.size(0))
    if theta.ndim != 1 or theta.numel() != base.size(0) or not torch.isfinite(theta).all():
        raise ValueError("MMT theta必须是有限[N]向量。")
    base = F.normalize(base.float(), dim=-1)
    direction = direction.float()
    result = torch.cos(theta).unsqueeze(-1) * base + torch.sin(theta).unsqueeze(-1) * direction
    normalized = F.normalize(result, dim=-1)
    return torch.where(theta.eq(0).unsqueeze(-1), base, normalized)


def semantic_geometry_features(
    base: torch.Tensor,
    value: torch.Tensor,
    support: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Inference-safe eight-dimensional Mean8/Value/support geometry."""
    _require_matrix("base", base)
    _require_matrix("value", value, base.size(0))
    _require_matrix("support", support)
    if int(topk) != 5 or support.size(0) < int(topk):
        raise ValueError("MMT固定使用至少5个支持类的Top-5。")
    base = F.normalize(base.float(), dim=-1)
    value = F.normalize(value.float(), dim=-1)
    support = F.normalize(support.float(), dim=-1)
    similarities = base @ support.T
    top_values = torch.sort(similarities, dim=1, descending=True, stable=True).values[:, :5]
    cosine = (base * value).sum(dim=-1, keepdim=True)
    displacement = (value - base).norm(dim=-1, keepdim=True)
    return torch.cat((cosine, displacement, top_values.mean(dim=1, keepdim=True), top_values), dim=1)


def _soft_topk(
    scores: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    excluded_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError("MMT soft-negative分数必须是有限二维矩阵。")
    if int(topk) != 5 or scores.size(1) < int(topk):
        raise ValueError("MMT soft-negative固定Top-5。")
    if float(temperature) <= 0.0:
        raise ValueError("MMT soft-negative温度必须为正数。")
    masked = scores.clone()
    if excluded_positions is not None:
        positions = torch.as_tensor(excluded_positions, device=scores.device).long()
        if positions.shape != (scores.size(0),):
            raise ValueError("MMT排除位置必须逐行提供。")
        masked.scatter_(1, positions.unsqueeze(1), float("-inf"))
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    values = masked.gather(1, order[:, : int(topk)])
    if not torch.isfinite(values).all():
        raise ValueError("MMT Top-5包含无效候选。")
    weights = F.softmax(values / float(temperature), dim=1)
    return (weights * values).sum(dim=1), values


@dataclass(frozen=True)
class MarginTargetTable:
    class_ids: torch.Tensor
    features: torch.Tensor
    base: torch.Tensor
    direction: torch.Tensor
    theta_cap: torch.Tensor
    theta_target: torch.Tensor
    move_target: torch.Tensor
    credible: torch.Tensor
    status: torch.Tensor
    target_positive_base: torch.Tensor
    target_direction_score: torch.Tensor
    soft_negative: torch.Tensor
    fold_margin: torch.Tensor
    leak_base_scores: torch.Tensor
    leak_direction_scores: torch.Tensor

    def to(self, device: torch.device | str) -> "MarginTargetTable":
        return MarginTargetTable(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    def detached(self) -> "MarginTargetTable":
        return MarginTargetTable(
            **{name: value.detach() for name, value in self.__dict__.items()}
        )


def minimum_theta_target(
    *,
    positive_base: torch.Tensor,
    positive_direction: torch.Tensor,
    soft_negative: torch.Tensor,
    required_margin: torch.Tensor,
    theta_cap: torch.Tensor,
    direction_valid: bool,
    bisection_steps: int,
) -> tuple[torch.Tensor, int, bool]:
    """Solve the first feasible point of one scalar geodesic margin curve."""
    values = (
        positive_base,
        positive_direction,
        soft_negative,
        required_margin,
        theta_cap,
    )
    if any(value.numel() != 1 or not torch.isfinite(value).all() for value in values):
        raise ValueError("MMT theta求解器只接受有限标量。")
    if int(bisection_steps) < 8 or float(theta_cap) < 0.0:
        raise ValueError("MMT theta求解器边界错误。")
    f0 = positive_base - soft_negative - required_margin
    if f0 >= 0:
        return positive_base.new_zeros(()), STATUS_ALREADY_SAFE, True
    if not bool(direction_valid) or float(theta_cap) == 0.0:
        return positive_base.new_zeros(()), STATUS_HARMFUL_OR_INVALID, False
    endpoints = [positive_base.new_zeros(()), theta_cap]
    interior = torch.atan2(positive_direction, positive_base)
    if 0.0 < float(interior) < float(theta_cap):
        endpoints.append(interior)
    candidates = torch.stack(endpoints)
    scores = (
        positive_base * torch.cos(candidates)
        + positive_direction * torch.sin(candidates)
    )
    best_index = int(scores.argmax())
    peak = candidates[best_index]
    best_score = scores[best_index]
    if best_score <= positive_base + 1e-12:
        return positive_base.new_zeros(()), STATUS_HARMFUL_OR_INVALID, False
    f_peak = best_score - soft_negative - required_margin
    if f_peak < 0:
        return positive_base.new_zeros(()), STATUS_CAP_INFEASIBLE, False
    low = positive_base.new_zeros(())
    high = peak.clone()
    for _ in range(int(bisection_steps)):
        middle = 0.5 * (low + high)
        f_middle = (
            positive_base * torch.cos(middle)
            + positive_direction * torch.sin(middle)
            - soft_negative
            - required_margin
        )
        if f_middle >= 0:
            high = middle
        else:
            low = middle
    return high, STATUS_FEASIBLE_MOVE, True


@torch.no_grad()
def build_margin_target_table(
    *,
    mean8: torch.Tensor,
    value: torch.Tensor,
    tg_prototypes: torch.Tensor,
    visual_centroids: torch.Tensor,
    seen_classes: torch.Tensor,
    topk: int = 5,
    negative_temperature: float = 0.07,
    margin_quantile: float = 0.25,
    global_theta_max: float = math.pi / 6,
    tangent_epsilon: float = 1e-6,
    bisection_steps: int = 20,
) -> MarginTargetTable:
    """Build detached pseudo-unseen minimum-angle targets from seen classes only."""
    _require_matrix("mean8", mean8, 200)
    _require_matrix("value", value, 200)
    _require_matrix("tg_prototypes", tg_prototypes, 200)
    _require_matrix("visual_centroids", visual_centroids, 150)
    classes = torch.as_tensor(seen_classes).detach().cpu().long().sort().values
    if classes.numel() != 150 or classes.unique().numel() != 150:
        raise ValueError("MMT目标表固定要求150个seen类。")
    if not 0.0 < float(margin_quantile) < 1.0 or int(bisection_steps) < 8:
        raise ValueError("MMT margin quantile或二分步数错误。")

    # Targets are intentionally generated in CPU float64 for deterministic scalar roots.
    mean8 = F.normalize(mean8.detach().cpu().double(), dim=-1)
    value = F.normalize(value.detach().cpu().double(), dim=-1)
    tg = F.normalize(tg_prototypes.detach().cpu().double(), dim=-1)
    centers = F.normalize(visual_centroids.detach().cpu().double(), dim=-1)
    direction_all, cap_all, valid_all = geodesic_basis(
        mean8.float(), value.float(),
        global_theta_max=float(global_theta_max),
        tangent_epsilon=float(tangent_epsilon),
    )
    direction_all = direction_all.double()
    cap_all = cap_all.double()
    valid_all = valid_all.cpu()
    rank_of = {int(class_id): rank for rank, class_id in enumerate(classes.tolist())}

    rows: dict[int, dict[str, torch.Tensor | float | int | bool]] = {}
    for pseudo_seen, pseudo_unseen in fixed_class_folds(classes):
        support_ids = pseudo_seen.long()
        support = tg.index_select(0, support_ids)
        support_centers = centers.index_select(
            0, torch.tensor([rank_of[int(value)] for value in support_ids.tolist()])
        )
        support_base = mean8.index_select(0, support_ids)
        support_scores = support_centers @ support.T
        excluded = torch.arange(support_ids.numel())
        support_negative, _ = _soft_topk(
            support_scores,
            topk=topk,
            temperature=negative_temperature,
            excluded_positions=excluded,
        )
        support_positive = (support_centers * support_base).sum(dim=-1)
        fold_threshold = torch.quantile(
            support_positive - support_negative,
            float(margin_quantile),
        )

        target_ids = pseudo_unseen.long()
        target_ranks = torch.tensor([rank_of[int(value)] for value in target_ids.tolist()])
        target_centers = centers.index_select(0, target_ranks)
        target_base = mean8.index_select(0, target_ids)
        target_value = value.index_select(0, target_ids)
        target_direction = direction_all.index_select(0, target_ids)
        target_cap = cap_all.index_select(0, target_ids)
        target_valid = valid_all.index_select(0, target_ids)
        target_negative, _ = _soft_topk(
            target_centers @ support.T,
            topk=topk,
            temperature=negative_temperature,
        )
        features = semantic_geometry_features(
            target_base.float(),
            target_value.float(),
            support.float(),
            topk=topk,
        ).double()

        for local, class_id in enumerate(target_ids.tolist()):
            center = target_centers[local]
            base = target_base[local]
            direction = target_direction[local]
            cap = target_cap[local]
            soft_negative = target_negative[local]
            a = (center * base).sum()
            b = (center * direction).sum()
            theta, status, credible = minimum_theta_target(
                positive_base=a,
                positive_direction=b,
                soft_negative=soft_negative,
                required_margin=fold_threshold,
                theta_cap=cap,
                direction_valid=bool(target_valid[local]),
                bisection_steps=int(bisection_steps),
            )
            rows[int(class_id)] = {
                "features": features[local],
                "base": base,
                "direction": direction,
                "theta_cap": cap,
                "theta_target": theta,
                "move_target": status == STATUS_FEASIBLE_MOVE,
                "credible": credible,
                "status": status,
                "target_positive_base": a,
                "target_direction_score": b,
                "soft_negative": soft_negative,
                "fold_margin": fold_threshold,
            }

    ordered = classes.tolist()
    if set(rows) != set(ordered):
        raise RuntimeError("MMT三折目标没有完整覆盖150个seen类。")
    target_base = torch.stack([rows[class_id]["base"] for class_id in ordered])
    target_direction = torch.stack([rows[class_id]["direction"] for class_id in ordered])
    leak_base = centers @ target_base.T
    leak_direction = centers @ target_direction.T
    return MarginTargetTable(
        class_ids=classes,
        features=torch.stack([rows[class_id]["features"] for class_id in ordered]).float(),
        base=target_base.float(),
        direction=target_direction.float(),
        theta_cap=torch.stack([rows[class_id]["theta_cap"] for class_id in ordered]).float(),
        theta_target=torch.stack([rows[class_id]["theta_target"] for class_id in ordered]).float(),
        move_target=torch.tensor([rows[class_id]["move_target"] for class_id in ordered], dtype=torch.float32),
        credible=torch.tensor([rows[class_id]["credible"] for class_id in ordered], dtype=torch.bool),
        status=torch.tensor([rows[class_id]["status"] for class_id in ordered], dtype=torch.long),
        target_positive_base=torch.stack(
            [rows[class_id]["target_positive_base"] for class_id in ordered]
        ).float(),
        target_direction_score=torch.stack(
            [rows[class_id]["target_direction_score"] for class_id in ordered]
        ).float(),
        soft_negative=torch.stack([rows[class_id]["soft_negative"] for class_id in ordered]).float(),
        fold_margin=torch.stack([rows[class_id]["fold_margin"] for class_id in ordered]).float(),
        leak_base_scores=leak_base.float(),
        leak_direction_scores=leak_direction.float(),
    ).detached()


class MinimumMarginGate(nn.Module):
    """Shared two-head gate: move-or-abstain and conditional geodesic angle."""

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        initial_move_probability: float = 0.1,
        initial_theta_fraction: float = 1.0 / 6.0,
    ):
        super().__init__()
        if int(hidden_dim) != 16:
            raise ValueError("MMT首轮固定hidden_dim=16。")
        if not 0.0 < float(initial_move_probability) < 0.5:
            raise ValueError("MMT初始移动概率必须位于(0,0.5)。")
        if not 0.0 < float(initial_theta_fraction) < 1.0:
            raise ValueError("MMT初始theta比例必须位于(0,1)。")
        self.trunk = nn.Sequential(nn.Linear(8, int(hidden_dim)), nn.GELU())
        self.move_head = nn.Linear(int(hidden_dim), 1)
        self.angle_head = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.move_head.weight)
        nn.init.zeros_(self.angle_head.weight)
        nn.init.constant_(
            self.move_head.bias,
            math.log(float(initial_move_probability) / (1.0 - float(initial_move_probability))),
        )
        nn.init.constant_(
            self.angle_head.bias,
            math.log(float(initial_theta_fraction) / (1.0 - float(initial_theta_fraction))),
        )

    def forward(
        self,
        features: torch.Tensor,
        theta_cap: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.size(1) != 8:
            raise ValueError("MMT Gate features必须是[N,8]。")
        if theta_cap.shape != features.shape[:1] or not torch.isfinite(theta_cap).all():
            raise ValueError("MMT Gate theta_cap必须是有限[N]向量。")
        hidden = self.trunk(features.float())
        move_logit = self.move_head(hidden).squeeze(-1)
        move_probability = torch.sigmoid(move_logit)
        theta_amount = theta_cap.float() * torch.sigmoid(self.angle_head(hidden).squeeze(-1))
        hard_move = (move_probability >= 0.5).to(move_probability.dtype)
        gate = hard_move + move_probability - move_probability.detach() if self.training else hard_move
        theta = gate * theta_amount
        return {
            "move_logit": move_logit,
            "move_probability": move_probability,
            "hard_move": hard_move,
            "theta_amount": theta_amount,
            "theta": theta,
        }


class MMTTSTModel(nn.Module):
    def __init__(
        self,
        parent: VariableClassTGVPR,
        gate: MinimumMarginGate,
        *,
        global_theta_max: float = math.pi / 6,
        tangent_epsilon: float = 1e-6,
        topk: int = 5,
    ):
        super().__init__()
        self.parent = parent
        self.gate = gate
        self.global_theta_max = float(global_theta_max)
        self.tangent_epsilon = float(tangent_epsilon)
        self.topk = int(topk)
        seen = parent.adapted_classes.detach().cpu().long().sort().values
        if seen.numel() != 150:
            raise ValueError("MMT父TG必须包含150个seen类。")
        self.register_buffer("seen_classes", seen, persistent=True)

    def scale(self) -> torch.Tensor:
        return self.parent.scale()

    def parent_prototypes(self) -> torch.Tensor:
        return self.parent.prototypes()

    def topology_loss_pretransport(self, parent_prototypes: torch.Tensor) -> torch.Tensor:
        base = self.parent.base_prototypes()
        adapted = F.normalize(parent_prototypes, dim=-1)
        off_diag = ~torch.eye(200, dtype=torch.bool, device=base.device)
        x = (base @ base.T).detach()[off_diag]
        y = (adapted @ adapted.T)[off_diag]
        x = x - x.mean()
        y = y - y.mean()
        return 1.0 - (x * y).sum() / (
            torch.sqrt(x.square().sum() + 1e-8)
            * torch.sqrt(y.square().sum() + 1e-8)
        )

    def prototype_components(self) -> dict[str, torch.Tensor]:
        device = self.parent.sentence_embeds.device
        all_classes = torch.arange(200, device=device)
        parent = self.parent_prototypes()
        seen = self.seen_classes.to(device)
        unseen = all_classes[~torch.isin(all_classes, seen)]
        base_all = self.parent.base_prototypes()
        value_all = self.parent.value_candidate(all_classes)
        base = base_all.index_select(0, unseen)
        value = value_all.index_select(0, unseen)
        support = parent.index_select(0, seen)
        direction, cap, valid = geodesic_basis(
            base,
            value,
            global_theta_max=self.global_theta_max,
            tangent_epsilon=self.tangent_epsilon,
        )
        features = semantic_geometry_features(base, value, support, topk=self.topk)
        gate = self.gate(features, cap)
        theta = torch.where(valid, gate["theta"], torch.zeros_like(gate["theta"]))
        moved = geodesic_transport(base, direction, theta)
        parent_unseen = parent.index_select(0, unseen)
        active = valid & gate["hard_move"].bool()
        transported = torch.where(active.unsqueeze(-1), moved, parent_unseen)
        final = parent.clone()
        final[unseen] = transported
        return {
            "parent": parent,
            "final": final,
            "unseen_classes": unseen,
            "features": features,
            "direction": direction,
            "theta_cap": cap,
            "valid_direction": valid,
            **gate,
            "theta": theta,
        }

    def prototypes(self) -> torch.Tensor:
        return self.prototype_components()["final"]

    def logits(
        self,
        image_features: torch.Tensor,
        class_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device).long())
        return F.normalize(image_features.float(), dim=-1) @ prototypes.T * self.scale()


def mmt_losses(
    gate_output: dict[str, torch.Tensor],
    table: MarginTargetTable,
    *,
    margin_scale: float = 0.02,
    leak_tolerance: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Direct gate supervision. Every table tensor is detached teacher evidence."""
    if float(margin_scale) <= 0.0 or float(leak_tolerance) < 0.0:
        raise ValueError("MMT loss margin_scale/leak_tolerance错误。")
    move_target = table.move_target.float()
    positive = move_target > 0.5
    negative = ~positive
    zero = gate_output["move_logit"].sum() * 0.0
    parts = []
    if positive.any():
        parts.append(F.binary_cross_entropy_with_logits(gate_output["move_logit"][positive], move_target[positive]))
    if negative.any():
        parts.append(F.binary_cross_entropy_with_logits(gate_output["move_logit"][negative], move_target[negative]))
    move = torch.stack(parts).mean() if parts else zero
    theta_regression = (
        F.smooth_l1_loss(
            gate_output["theta_amount"][positive] / table.theta_cap[positive].clamp_min(1e-8),
            table.theta_target[positive] / table.theta_cap[positive].clamp_min(1e-8),
        )
        if positive.any()
        else zero
    )
    theta = gate_output["theta"]
    achieved_positive = (
        table.target_positive_base * torch.cos(theta)
        + table.target_direction_score * torch.sin(theta)
    )
    achieved_margin = achieved_positive - table.soft_negative
    margin = (
        F.relu(table.fold_margin[table.credible] - achieved_margin[table.credible]).mean()
        / float(margin_scale)
        if table.credible.any()
        else zero
    )
    soft_theta = gate_output["move_probability"] * gate_output["theta_amount"]
    zero_target = ~positive
    abstain = (
        (soft_theta[zero_target] / table.theta_cap[zero_target].clamp_min(1e-8)).square().mean()
        if zero_target.any()
        else zero
    )
    delta = (
        (torch.cos(theta) - 1.0).unsqueeze(0) * table.leak_base_scores
        + torch.sin(theta).unsqueeze(0) * table.leak_direction_scores
    )
    diagonal = torch.eye(delta.size(0), dtype=torch.bool, device=delta.device)
    delta = delta.masked_fill(diagonal, float("-inf"))
    leak_topk = min(5, delta.size(0) - 1)
    if leak_topk <= 0:
        raise ValueError("MMT leak loss至少需要两个类别。")
    worst = delta.topk(leak_topk, dim=0).values
    leak = F.relu(worst - float(leak_tolerance)).mean()
    return {
        "move": move,
        "theta": theta_regression,
        "margin": margin,
        "zero": abstain,
        "leak": leak,
        "achieved_margin_mean": achieved_margin.detach().mean(),
    }
