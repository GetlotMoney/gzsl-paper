"""Continuous Top-2 Patch Margin model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CTPMOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    role_logits: torch.Tensor
    correction: torch.Tensor
    top2_local: torch.Tensor
    top2_global: torch.Tensor
    margin0: torch.Tensor
    interaction_input: torch.Tensor
    d_s: torch.Tensor
    d_v: torch.Tensor
    d_i: torch.Tensor
    d_total: torch.Tensor
    attention: torch.Tensor
    semantic_evidence: torch.Tensor
    visual_evidence: torch.Tensor
    pair_logits: torch.Tensor
    pair_mask: torch.Tensor | None = None


def _check_matrix(name: str, value: torch.Tensor, shape_tail: tuple[int, ...]) -> None:
    if value.ndim != 1 + len(shape_tail) or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{name} shape must be [N,{','.join(map(str, shape_tail))}].")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{name} contains NaN/Inf.")


def stable_top2(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.size(1) < 2:
        raise ValueError("logits must be [B,C] with C>=2.")
    return torch.argsort(logits, dim=1, descending=True, stable=True)[:, :2]


class TinyMarginMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        with torch.no_grad():
            values = torch.linspace(-1.0, 1.0, hidden_dim).view(1, hidden_dim) * 1e-3
            self.net[-1].weight.copy_(values)
            self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CTPMModel(nn.Module):
    """Fixed class-name logits plus one role-conditioned Top1/Top2 patch correction."""

    def __init__(
        self,
        class_name_embeds: torch.Tensor,
        role_sentence_embeds: torch.Tensor,
        *,
        scale: float = 1.0 / 0.07,
        hidden_dim: int = 32,
        patch_projection_dim: int = 64,
        max_margin: float = 2.0,
        max_role_weight: float = 0.75,
    ):
        super().__init__()
        _check_matrix("class_name_embeds", class_name_embeds, (768,))
        _check_matrix("role_sentence_embeds", role_sentence_embeds, (8, 768))
        if class_name_embeds.size(0) != role_sentence_embeds.size(0):
            raise ValueError("class_name_embeds and role_sentence_embeds class axes differ.")
        if int(hidden_dim) <= 0 or int(patch_projection_dim) <= 0:
            raise ValueError("hidden_dim and patch_projection_dim must be positive.")
        if float(scale) <= 0 or float(max_margin) <= 0 or float(max_role_weight) <= 0:
            raise ValueError("scale, max_margin and max_role_weight must be positive.")

        self.class_count = int(class_name_embeds.size(0))
        self.scale_value = float(scale)
        self.max_margin = float(max_margin)
        self.max_role_weight = float(max_role_weight)
        self.register_buffer("class_name_embeds", F.normalize(class_name_embeds.float(), dim=-1))
        self.register_buffer("role_sentence_embeds", F.normalize(role_sentence_embeds.float(), dim=-1))

        self.raw_role_weights = nn.Parameter(torch.zeros(8))
        self.semantic_margin = TinyMarginMLP(9, int(hidden_dim))
        self.patch_query = nn.Linear(768, int(patch_projection_dim), bias=False)
        self.patch_key = nn.Linear(768, int(patch_projection_dim), bias=False)
        self.visual_margin = TinyMarginMLP(8, int(hidden_dim))
        self.interaction_margin = TinyMarginMLP(5, int(hidden_dim))

    def scale(self) -> torch.Tensor:
        return torch.tensor(self.scale_value, device=self.class_name_embeds.device)

    def base_logits(self, image_features: torch.Tensor, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        images = F.normalize(image_features.float(), dim=-1)
        prototypes = self.class_name_embeds
        if class_ids is not None:
            prototypes = prototypes.index_select(0, class_ids.to(prototypes.device).long())
        return images @ prototypes.T * self.scale_value

    def _active_axis(self, class_ids: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if class_ids is None:
            return torch.arange(self.class_count, device=device)
        axis = class_ids.to(device).long()
        if axis.ndim != 1 or axis.numel() < 2 or int(axis.min()) < 0 or int(axis.max()) >= self.class_count:
            raise ValueError("class_ids must be a valid one-dimensional class axis with at least two classes.")
        return axis

    def _queries(self, top2_global: torch.Tensor, query_mode: str) -> torch.Tensor:
        c1 = top2_global[:, 0]
        c2 = top2_global[:, 1]
        if query_mode == "role_difference":
            return F.normalize(self.role_sentence_embeds.index_select(0, c2) - self.role_sentence_embeds.index_select(0, c1), dim=-1)
        if query_mode == "class_name_difference":
            q = F.normalize(self.class_name_embeds.index_select(0, c2) - self.class_name_embeds.index_select(0, c1), dim=-1)
            return q[:, None, :].expand(-1, 8, -1)
        raise ValueError(f"unknown query_mode: {query_mode}")

    def _role_logits(self, image_features: torch.Tensor, axis: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled:
            return image_features.new_zeros((image_features.size(0), axis.numel()))
        weights = self.max_role_weight * torch.tanh(self.raw_role_weights)
        role_mix = torch.einsum("r,crd->cd", weights, self.role_sentence_embeds)
        active = role_mix.index_select(0, axis.to(role_mix.device))
        return F.normalize(image_features.float(), dim=-1) @ active.T * self.scale_value

    def forward(
        self,
        image_features: torch.Tensor,
        patch_features: torch.Tensor,
        *,
        class_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        enable_s: bool = True,
        enable_v: bool = True,
        enable_i: bool = True,
        query_mode: str = "role_difference",
        no_l_role: bool = False,
    ) -> CTPMOutput:
        if image_features.ndim != 2 or image_features.size(1) != 768:
            raise ValueError("image_features must be [B,768].")
        if patch_features.ndim != 3 or tuple(patch_features.shape[1:]) != (36, 768):
            raise ValueError("patch_features must be [B,36,768].")
        if patch_features.size(0) != image_features.size(0):
            raise ValueError("image and patch batch sizes differ.")
        device = image_features.device
        axis = self._active_axis(class_ids, device)
        images = F.normalize(image_features.float(), dim=-1)
        patches = F.normalize(patch_features.float(), dim=-1)
        base = self.base_logits(images, axis)
        top2_local = stable_top2(base.detach())
        top2_global = axis.index_select(0, top2_local.reshape(-1)).view(-1, 2)
        margin0 = base.gather(1, top2_local[:, :1]).squeeze(1) - base.gather(1, top2_local[:, 1:2]).squeeze(1)
        queries = self._queries(top2_global, query_mode)

        semantic_evidence = torch.einsum("bd,brd->br", images, queries)
        d_s = self.max_margin * torch.tanh(self.semantic_margin(torch.cat((semantic_evidence, margin0[:, None]), dim=1)))
        if not enable_s:
            d_s = torch.zeros_like(d_s)
        role_logits = self._role_logits(images, axis, bool(enable_s and not no_l_role))

        q_proj = F.normalize(self.patch_query(queries), dim=-1)
        p_proj = F.normalize(self.patch_key(patches), dim=-1)
        attn_logits = torch.einsum("brh,bnh->brn", q_proj, p_proj) / math.sqrt(float(q_proj.size(-1)))
        attention = F.softmax(attn_logits, dim=-1)
        signed_patch = torch.einsum("bnd,brd->brn", patches, queries)
        visual_evidence = (attention * signed_patch).sum(dim=-1)
        d_v = self.max_margin * torch.tanh(self.visual_margin(visual_evidence))
        if not enable_v:
            d_v = torch.zeros_like(d_v)
            visual_evidence_for_i = torch.zeros_like(visual_evidence)
        else:
            visual_evidence_for_i = visual_evidence

        with torch.no_grad():
            probs = F.softmax(base, dim=1).clamp_min(1e-12)
            h0 = -(probs * probs.log()).sum(dim=1) / math.log(float(base.size(1)))
        i_input = torch.stack((margin0, d_s, d_v, d_s * d_v, h0), dim=1)
        d_i = self.max_margin * torch.tanh(self.interaction_margin(i_input))
        if not enable_i:
            d_i = torch.zeros_like(d_i)
        d_total = d_s + d_v + d_i

        logits = base + role_logits
        correction = torch.zeros_like(logits)
        correction.scatter_add_(1, top2_local[:, :1], -0.5 * d_total[:, None])
        correction.scatter_add_(1, top2_local[:, 1:2], 0.5 * d_total[:, None])
        logits = logits + correction
        pair_logits = logits.gather(1, top2_local)

        pair_mask = None
        if labels is not None:
            labels = labels.to(device).long()
            pair_mask = labels[:, None].eq(top2_global).any(dim=1)
        return CTPMOutput(
            logits=logits,
            base_logits=base,
            role_logits=role_logits,
            correction=correction,
            top2_local=top2_local,
            top2_global=top2_global,
            margin0=margin0,
            interaction_input=i_input,
            d_s=d_s,
            d_v=d_v,
            d_i=d_i,
            d_total=d_total,
            attention=attention,
            semantic_evidence=semantic_evidence,
            visual_evidence=visual_evidence_for_i,
            pair_logits=pair_logits,
            pair_mask=pair_mask,
        )

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "semantic": list(self.semantic_margin.parameters()) + [self.raw_role_weights],
            "visual": list(self.patch_query.parameters()) + list(self.patch_key.parameters()) + list(self.visual_margin.parameters()),
            "interaction": list(self.interaction_margin.parameters()),
        }


def pair_ce_loss(output: CTPMOutput, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(output.logits.device).long()
    pair_target = labels.eq(output.top2_global[:, 1]).long()
    mask = labels[:, None].eq(output.top2_global).any(dim=1)
    if not bool(mask.any()):
        return output.logits.sum() * 0.0
    return F.cross_entropy(output.pair_logits[mask], pair_target[mask])


def pair_scatter(top2_local: torch.Tensor, margin: torch.Tensor, class_count: int) -> torch.Tensor:
    """Return the fixed antisymmetric CTPM correction for a candidate margin."""
    if top2_local.ndim != 2 or tuple(top2_local.shape[1:]) != (2,):
        raise ValueError("top2_local must be [B,2].")
    if margin.ndim != 1 or margin.size(0) != top2_local.size(0):
        raise ValueError("margin must be [B] for the same candidate pairs.")
    correction = margin.new_zeros((margin.size(0), int(class_count)))
    correction.scatter_add_(1, top2_local[:, :1], -0.5 * margin[:, None])
    correction.scatter_add_(1, top2_local[:, 1:2], 0.5 * margin[:, None])
    return correction


def balanced_pair_ce(
    pair_logits: torch.Tensor, top2_global: torch.Tensor, labels: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Equal-mass c1/c2 CE, with an exact graph-zero when one side is absent."""
    labels = labels.to(pair_logits.device).long()
    target = labels.eq(top2_global[:, 1]).long()
    in_pair = labels[:, None].eq(top2_global).any(dim=1)
    c1 = in_pair & target.eq(0)
    c2 = in_pair & target.eq(1)
    if not bool(c1.any()) or not bool(c2.any()):
        return pair_logits.sum() * 0.0, True
    loss = 0.5 * F.cross_entropy(pair_logits[c1], target[c1])
    loss = loss + 0.5 * F.cross_entropy(pair_logits[c2], target[c2])
    return loss, False


def isolated_interaction_margin(model: CTPMModel, output: CTPMOutput) -> torch.Tensor:
    """Recompute CMI with detached S/V prefixes so only I receives this loss."""
    h0 = output.interaction_input[:, 4].detach()
    i_input = torch.stack(
        (
            output.margin0.detach(), output.d_s.detach(), output.d_v.detach(),
            (output.d_s * output.d_v).detach(), h0,
        ),
        dim=1,
    )
    return model.max_margin * torch.tanh(model.interaction_margin(i_input))


def attention_diversity_loss(attention: torch.Tensor) -> torch.Tensor:
    if attention.ndim != 3 or attention.size(1) != 8 or attention.size(2) != 36:
        raise ValueError("attention must be [B,8,36].")
    rows = F.normalize(attention.float().clamp_min(1e-12), dim=-1)
    sim = torch.einsum("brn,bsn->brs", rows, rows)
    off_diag = ~torch.eye(8, dtype=torch.bool, device=attention.device)
    return sim[:, off_diag].square().mean()


def ctpm_loss(
    output: CTPMOutput,
    labels: torch.Tensor,
    *,
    pair_loss_weight: float = 0.1,
    attention_diversity_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = labels.to(output.logits.device).long()
    full_ce = F.cross_entropy(output.logits, labels)
    pair_ce = pair_ce_loss(output, labels)
    diversity = attention_diversity_loss(output.attention)
    total = full_ce + float(pair_loss_weight) * pair_ce + float(attention_diversity_weight) * diversity
    pair_mask = labels[:, None].eq(output.top2_global).any(dim=1)
    return total, {
        "total": total.detach(),
        "full_ce": full_ce.detach(),
        "pair_ce": pair_ce.detach(),
        "attention_diversity": diversity.detach(),
        "pair_count": pair_mask.sum().detach(),
    }
