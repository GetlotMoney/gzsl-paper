"""Geodesic-target-distilled tangent semantic transport (GTD-TST)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 6
GEOMETRY_EPS = 1e-6


def _validated_classes(
    values: torch.Tensor,
    *,
    class_count: int,
    name: str,
    allow_empty: bool = False,
) -> torch.Tensor:
    classes = torch.as_tensor(values).detach().cpu().long().sort().values
    if classes.ndim != 1 or (classes.numel() == 0 and not allow_empty):
        raise ValueError(f"{name}必须是一维{'可空' if allow_empty else '非空'}类别集合。")
    if classes.unique().numel() != classes.numel():
        raise ValueError(f"{name}不得包含重复类别。")
    if classes.numel() and (int(classes.min()) < 0 or int(classes.max()) >= int(class_count)):
        raise ValueError(f"{name}超出全局类别轴。")
    return classes


@dataclass(frozen=True)
class GeodesicGeometry:
    mean8: torch.Tensor
    value: torch.Tensor
    direction: torch.Tensor
    angle_to_value: torch.Tensor
    angle_limit: torch.Tensor
    valid: torch.Tensor
    features: torch.Tensor


def geodesic_geometry(
    mean8: torch.Tensor,
    value: torch.Tensor,
    support_mean8: torch.Tensor,
    *,
    max_transport_step: float = 1.5,
    eps: float = GEOMETRY_EPS,
) -> GeodesicGeometry:
    """Build text-only geometry available for both pseudo- and true-unseen classes."""
    if mean8.ndim != 2 or value.shape != mean8.shape or mean8.size(1) != 768:
        raise ValueError("GTD mean8/value必须是相同shape的[class,768]。")
    if support_mean8.ndim != 2 or support_mean8.size(1) != 768 or support_mean8.size(0) < 5:
        raise ValueError("GTD support Mean8必须是至少5类的[class,768]。")
    if not 0.0 < float(max_transport_step):
        raise ValueError("GTD max_transport_step必须为正数。")
    base = F.normalize(mean8.float(), dim=-1)
    target = F.normalize(value.float(), dim=-1)
    support = F.normalize(support_mean8.float(), dim=-1)
    cosine = (base * target).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    tangent = target - cosine.unsqueeze(-1) * base
    tangent_norm = tangent.norm(dim=-1)
    # The tangent is undefined for coincident and antipodal endpoints.
    valid = tangent_norm > float(eps)
    direction = tangent / tangent_norm.clamp_min(float(eps)).unsqueeze(-1)
    direction = torch.where(valid.unsqueeze(-1), direction, torch.zeros_like(direction))
    cap = math.atan(float(max_transport_step))
    angle_limit = torch.minimum(angle, angle.new_full(angle.shape, cap))
    angle_limit = torch.where(valid, angle_limit, torch.zeros_like(angle_limit))

    similarity = base @ support.T
    top5_values, top5_positions = similarity.topk(5, dim=1)
    top5_support = support.index_select(0, top5_positions.reshape(-1)).reshape(
        base.size(0), 5, 768
    )
    top5_weights = F.softmax(top5_values / 0.07, dim=1)
    neighbor_center = F.normalize(
        (top5_weights.unsqueeze(-1) * top5_support).sum(dim=1), dim=-1
    )
    neighbor_tangent = neighbor_center - (
        neighbor_center * base
    ).sum(dim=-1, keepdim=True) * base
    neighbor_norm = neighbor_tangent.norm(dim=-1)
    neighbor_direction = neighbor_tangent / neighbor_norm.clamp_min(float(eps)).unsqueeze(-1)
    direction_alignment = (direction * neighbor_direction).sum(dim=-1)
    direction_alignment = torch.where(
        valid & (neighbor_norm > float(eps)),
        direction_alignment,
        torch.zeros_like(direction_alignment),
    )
    features = torch.stack(
        (
            cosine,
            angle / math.pi,
            top5_values.mean(dim=1),
            top5_values.max(dim=1).values,
            top5_values.std(dim=1, unbiased=False),
            direction_alignment,
        ),
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise ValueError("GTD文本几何特征包含NaN/Inf。")
    return GeodesicGeometry(
        mean8=base,
        value=target,
        direction=direction,
        angle_to_value=angle,
        angle_limit=angle_limit,
        valid=valid,
        features=features,
    )


def geodesic_points(
    mean8: torch.Tensor,
    direction: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Exact unit-sphere points for one or many angles per class."""
    if mean8.ndim != 2 or direction.shape != mean8.shape:
        raise ValueError("GTD geodesic mean8/direction shape错误。")
    if theta.ndim not in (1, 2) or theta.size(0) != mean8.size(0):
        raise ValueError("GTD theta必须是[class]或[class,grid]。")
    base = F.normalize(mean8.float(), dim=-1)
    tangent = direction.float()
    if theta.ndim == 1:
        points = torch.cos(theta).unsqueeze(-1) * base + torch.sin(theta).unsqueeze(-1) * tangent
    else:
        points = (
            torch.cos(theta).unsqueeze(-1) * base.unsqueeze(1)
            + torch.sin(theta).unsqueeze(-1) * tangent.unsqueeze(1)
        )
    return F.normalize(points, dim=-1)


def closed_form_alignment_angle(
    visual_centroid: torch.Tensor,
    mean8: torch.Tensor,
    direction: torch.Tensor,
    angle_limit: torch.Tensor,
) -> torch.Tensor:
    """Unregularized alignment optimum, used as a deterministic oracle check."""
    centroid = F.normalize(visual_centroid.float(), dim=-1)
    base = F.normalize(mean8.float(), dim=-1)
    tangent = direction.float()
    a = (centroid * base).sum(dim=-1)
    b = (centroid * tangent).sum(dim=-1)
    critical = torch.atan2(b, a).clamp_min(0.0)
    critical = torch.minimum(critical, angle_limit)
    candidates = torch.stack((torch.zeros_like(angle_limit), critical, angle_limit), dim=1)
    scores = a[:, None] * torch.cos(candidates) + b[:, None] * torch.sin(candidates)
    return candidates.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)


def select_oracle_targets(
    theta_grid: torch.Tensor,
    objective: torch.Tensor,
    valid: torch.Tensor,
    *,
    gain_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the first minimum; all ties therefore resolve to exact theta zero."""
    if (
        theta_grid.ndim != 2
        or objective.shape != theta_grid.shape
        or valid.shape != theta_grid.shape[:1]
        or theta_grid.size(1) != 33
        or not torch.isfinite(theta_grid).all()
        or not torch.isfinite(objective).all()
        or float(gain_epsilon) <= 0.0
    ):
        raise ValueError("GTD oracle grid/objective/valid边界错误。")
    best_index = objective.argmin(dim=1)
    rows = torch.arange(theta_grid.size(0), device=theta_grid.device)
    selected = theta_grid[rows, best_index]
    gain = objective[:, 0] - objective[rows, best_index]
    useful = valid.bool() & (gain > float(gain_epsilon))
    selected = torch.where(useful, selected, torch.zeros_like(selected))
    return selected, gain, useful


class GeodesicTargetGate(nn.Module):
    """Shared zero-initialized angle-ratio regressor."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        if int(hidden_dim) != 16:
            raise ValueError("GTD首轮固定Gate hidden_dim=16。")
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def raw_ratio(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != FEATURE_DIM:
            raise ValueError("GTD Gate输入必须是[class,6]。")
        return self.network(features.float()).squeeze(-1)


class GTDTSTModel(nn.Module):
    """TG parent plus unseen-only geodesic target distilled transport."""

    def __init__(
        self,
        parent: nn.Module,
        seen_classes: torch.Tensor,
        *,
        class_count: int = 200,
        hidden_dim: int = 16,
        max_transport_step: float = 1.5,
        grid_points: int = 33,
    ):
        super().__init__()
        if int(class_count) <= 1 or int(grid_points) != 33:
            raise ValueError("GTD要求至少2类和固定33点角度网格。")
        parent_class_count = int(parent.tg_vpr.sentence_embeds.size(0))
        if parent_class_count != int(class_count):
            raise ValueError("GTD class_count必须与TG父模型类别轴一致。")
        seen = _validated_classes(
            seen_classes, class_count=int(class_count), name="seen_classes"
        )
        if seen.numel() < 6 or seen.numel() >= int(class_count):
            raise ValueError("GTD要求至少6个seen类且必须保留true-unseen类。")
        unseen = torch.arange(int(class_count))[~torch.isin(torch.arange(int(class_count)), seen)]
        self.parent = parent
        self.gate = GeodesicTargetGate(hidden_dim)
        self.class_count = int(class_count)
        self.max_transport_step = float(max_transport_step)
        self.grid_points = int(grid_points)
        self.dead_zone = 1.0 / float(self.grid_points - 1)
        self.register_buffer("seen_classes", seen, persistent=True)
        self.register_buffer("unseen_classes", unseen, persistent=True)

    def scale(self) -> torch.Tensor:
        return self.parent.scale()

    def parent_prototypes(self) -> torch.Tensor:
        return self.parent.prototypes()

    def _geometry(
        self,
        target_classes: torch.Tensor,
        support_classes: torch.Tensor,
    ) -> GeodesicGeometry:
        device = self.parent.tg_vpr.sentence_embeds.device
        target_ids = _validated_classes(
            target_classes,
            class_count=self.class_count,
            name="target_classes",
        ).to(device)
        support_ids = _validated_classes(
            support_classes,
            class_count=self.class_count,
            name="support_classes",
        ).to(device)
        if torch.isin(target_ids, support_ids).any():
            raise ValueError("GTD target与support必须互斥。")
        mean8_all = self.parent.tg_vpr.base_prototypes()
        all_ids = torch.arange(self.class_count, device=device)
        value_all = self.parent.tg_vpr.value_candidate(all_ids)
        return geodesic_geometry(
            mean8_all.index_select(0, target_ids),
            value_all.index_select(0, target_ids),
            mean8_all.index_select(0, support_ids),
            max_transport_step=self.max_transport_step,
        )

    def deployed_ratio(self, raw_ratio: torch.Tensor) -> torch.Tensor:
        ratio = raw_ratio.clamp(0.0, 1.0)
        return torch.where(ratio < self.dead_zone, torch.zeros_like(ratio), ratio)

    def transported_subset(
        self,
        target_classes: torch.Tensor,
        support_classes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        geometry = self._geometry(target_classes, support_classes)
        raw_ratio = self.gate.raw_ratio(geometry.features)
        ratio = self.deployed_ratio(raw_ratio)
        theta = ratio * geometry.angle_limit
        theta = torch.where(geometry.valid, theta, torch.zeros_like(theta))
        transported = geodesic_points(geometry.mean8, geometry.direction, theta)
        return transported, {
            "raw_ratio": raw_ratio,
            "ratio": ratio,
            "theta": theta,
            "valid": geometry.valid,
            "features": geometry.features,
            "angle_limit": geometry.angle_limit,
        }

    def prototype_bundle(self) -> dict[str, torch.Tensor]:
        parent = self.parent_prototypes()
        unseen = self.unseen_classes.to(parent.device)
        seen = self.seen_classes.to(parent.device)
        mean8 = self.parent.tg_vpr.base_prototypes()
        if not torch.allclose(
            parent.index_select(0, unseen),
            mean8.index_select(0, unseen),
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError("GTD父TG的真实unseen原型不是Mean8，无法保证theta0关闭路径。")
        transported, diagnostics = self.transported_subset(unseen, seen)
        parent_unseen = parent.index_select(0, unseen)
        # Repeated normalization is numerically close but not bitwise identical.  A
        # true zero angle is the registered module-off path and must reuse the exact
        # parent tensor so logits/checkpoints reproduce TG without tolerance games.
        transported = torch.where(
            diagnostics["theta"].eq(0.0).unsqueeze(-1),
            parent_unseen,
            transported,
        )
        final = parent.clone()
        final[unseen] = transported
        return {"parent": parent, "final": final, **diagnostics}

    def prototypes(self) -> torch.Tensor:
        return self.prototype_bundle()["final"]

    def logits(
        self,
        image_features: torch.Tensor,
        class_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prototypes = self.prototypes()
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device).long())
        return F.normalize(image_features.float(), dim=-1) @ prototypes.T * self.scale()

    def topology_loss(self) -> torch.Tensor:
        return self.parent.topology_loss()

    def oracle_targets(
        self,
        visual_centroids: torch.Tensor,
        pseudo_seen: torch.Tensor,
        pseudo_unseen: torch.Tensor,
        *,
        theta_penalty: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Detached seen-only CE+theta^2 targets for one pseudo-unseen fold."""
        if float(theta_penalty) != 0.1:
            raise ValueError("GTD首轮固定theta_penalty=0.1。")
        device = self.parent.tg_vpr.sentence_embeds.device
        pseudo_seen_cpu = _validated_classes(
            pseudo_seen, class_count=self.class_count, name="pseudo_seen"
        )
        pseudo_unseen_cpu = _validated_classes(
            pseudo_unseen, class_count=self.class_count, name="pseudo_unseen"
        )
        if torch.isin(pseudo_seen_cpu, pseudo_unseen_cpu).any():
            raise ValueError("GTD pseudo_seen/pseudo_unseen必须互斥。")
        joined = torch.cat((pseudo_seen_cpu, pseudo_unseen_cpu)).sort().values
        if not torch.equal(joined, self.seen_classes.cpu()):
            raise ValueError("GTD三折必须完整覆盖全部seen类。")
        expected_centroid_shape = (self.seen_classes.numel(), 768)
        if tuple(visual_centroids.shape) != expected_centroid_shape:
            raise ValueError(
                f"GTD seen视觉中心必须是{expected_centroid_shape}。"
            )
        pseudo_seen_ids = pseudo_seen_cpu.to(device)
        target_ids = pseudo_unseen_cpu.to(device)
        geometry = self._geometry(target_ids, pseudo_seen_ids)
        count = int(target_ids.numel())
        grid_ratio = torch.linspace(0.0, 1.0, self.grid_points, device=device)
        theta_grid = geometry.angle_limit[:, None] * grid_ratio[None, :]
        candidates = geodesic_points(geometry.mean8, geometry.direction, theta_grid)

        parent = self.parent_prototypes().detach().clone()
        mean8_all = self.parent.tg_vpr.base_prototypes().detach()
        parent[target_ids] = mean8_all.index_select(0, target_ids)
        seen_ids = self.seen_classes.to(device)
        competition = parent.index_select(0, seen_ids)
        global_to_seen = torch.full((self.class_count,), -1, dtype=torch.long, device=device)
        global_to_seen[seen_ids] = torch.arange(seen_ids.numel(), device=device)
        target_positions = global_to_seen.index_select(0, target_ids)
        if bool((target_positions < 0).any()):
            raise ValueError("GTD pseudo-unseen类别未映射到seen竞争轴。")
        centroids = F.normalize(visual_centroids.detach().to(device).float(), dim=-1)
        target_centroids = centroids.index_select(0, target_positions)
        scale = self.scale().detach()
        fixed_logits = target_centroids @ competition.T * scale
        fixed_logits.scatter_(
            1,
            target_positions[:, None],
            torch.full((count, 1), -torch.inf, device=device),
        )
        other_logsumexp = torch.logsumexp(fixed_logits, dim=1)
        true_logits = torch.einsum("cd,ckd->ck", target_centroids, candidates) * scale
        ce = torch.logaddexp(true_logits, other_logsumexp[:, None]) - true_logits
        objective = ce + float(theta_penalty) * theta_grid.square()
        best_theta, gain, useful = select_oracle_targets(
            theta_grid,
            objective,
            geometry.valid,
        )
        target_ratio = torch.where(
            geometry.angle_limit > GEOMETRY_EPS,
            best_theta / geometry.angle_limit.clamp_min(GEOMETRY_EPS),
            torch.zeros_like(best_theta),
        )
        closed_form = closed_form_alignment_angle(
            target_centroids,
            geometry.mean8,
            geometry.direction,
            geometry.angle_limit,
        )
        return {
            "class_ids": target_ids.detach(),
            "features": geometry.features.detach(),
            "target_ratio": target_ratio.detach(),
            "target_theta": best_theta.detach(),
            "oracle_gain": gain.detach(),
            "valid": geometry.valid.detach(),
            "move_mask": useful.detach(),
            "angle_limit": geometry.angle_limit.detach(),
            "closed_form_theta": closed_form.detach(),
        }

    @torch.no_grad()
    def diagnostics(self, packages: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        if len(packages) != 3:
            raise ValueError("GTD诊断固定要求三个seen折。")
        features = torch.cat([item["features"] for item in packages])
        target = torch.cat([item["target_ratio"] for item in packages])
        gain = torch.cat([item["oracle_gain"] for item in packages])
        valid = torch.cat([item["valid"] for item in packages])
        raw = self.gate.raw_ratio(features)
        deployed = self.deployed_ratio(raw)
        target_zero = target <= 0.0
        prediction_zero = deployed <= 0.0
        centered_raw = raw - raw.mean()
        centered_target = target - target.mean()
        denominator = torch.sqrt(centered_raw.square().sum() * centered_target.square().sum())
        correlation = (
            (centered_raw * centered_target).sum() / denominator
            if float(denominator) > 0.0
            else raw.new_zeros(())
        )
        bundle = self.prototype_bundle()
        theta = bundle["theta"]
        ratio = bundle["ratio"]
        return {
            "oracle_zero_target_rate": float(target_zero.float().mean()),
            "oracle_gain_mean": float(gain.mean()),
            "oracle_gain_max": float(gain.max()),
            "oracle_degenerate_rate": float((~valid).float().mean()),
            "gate_target_mae": float((raw - target).abs().mean()),
            "gate_target_correlation": float(correlation),
            "gate_false_move_rate": float((target_zero & ~prediction_zero).float().mean()),
            "gate_missed_move_rate": float((~target_zero & prediction_zero).float().mean()),
            "seen_raw_ratio_mean": float(raw.mean()),
            "seen_deployed_ratio_mean": float(deployed.mean()),
            "unseen_move_rate": float((ratio > 0).float().mean()),
            "unseen_theta_mean_degrees": float(torch.rad2deg(theta).mean()),
            "unseen_theta_p95_degrees": float(torch.quantile(torch.rad2deg(theta), 0.95)),
            "unseen_ratio_saturation_rate": float((ratio >= 1.0 - 1e-6).float().mean()),
        }
