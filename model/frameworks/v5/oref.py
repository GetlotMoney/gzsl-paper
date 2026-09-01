"""Observable Role Entailment Field (OREF) with exactly S/V/I modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DIM = 768
ROLES = 8
HIDDEN = 64
EPS = 1e-6
TAU_PATCH = 0.07
TAU_ROLE = 0.1
MARGIN_SCALE = 0.2
REFUTATION_WEIGHT = 2.0
BASE_TEMPERATURE = 0.07


def standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(1, keepdim=True)) / torch.sqrt(
        values.var(1, keepdim=True, unbiased=False) + EPS
    )


def stable_rivals(logits: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or class_ids.ndim != 1 or logits.size(1) != class_ids.numel():
        raise ValueError("OREF logits/class axis形状错误。")
    id_order = torch.argsort(class_ids, stable=True)
    ranked = id_order[torch.argsort(
        logits.index_select(1, id_order), dim=1, descending=True, stable=True
    )]
    top1, top2 = ranked[:, 0], ranked[:, 1]
    candidates = torch.arange(class_ids.numel(), device=logits.device)[None]
    return torch.where(candidates.eq(top1[:, None]), top2[:, None], top1[:, None])


class DynamicRoleClaimModule(nn.Module):
    """S: candidate-vs-rival role claims and name-only off path."""

    def __init__(self, names: torch.Tensor, roles: torch.Tensor, class_ids: torch.Tensor):
        super().__init__()
        if names.ndim != 2 or names.size(1) != DIM:
            raise ValueError("OREF name embeddings必须是[C,768]。")
        if roles.shape != (names.size(0), ROLES, DIM):
            raise ValueError("OREF role embeddings必须是[C,8,768]。")
        if class_ids.shape != (names.size(0),):
            raise ValueError("OREF class_ids与文本轴不一致。")
        self.register_buffer("names", F.normalize(names.float(), dim=-1))
        self.register_buffer("roles", F.normalize(roles.float(), dim=-1))
        self.register_buffer("class_ids", class_ids.long())

    def parent_logits(self, image_cls: torch.Tensor) -> torch.Tensor:
        return F.normalize(image_cls.float(), dim=-1) @ self.names.T / BASE_TEMPERATURE

    def name_chunk(self, rivals: torch.Tensor, start: int, end: int):
        batch = rivals.size(0)
        candidate = self.names[start:end][None].expand(batch, -1, -1)
        rival = self.names[rivals[:, start:end]]
        query = F.normalize(candidate - rival, dim=-1)
        return query[:, :, None, :].expand(-1, -1, ROLES, -1)

    def role_chunk(self, rivals: torch.Tensor, start: int, end: int):
        batch = rivals.size(0)
        candidate = self.roles[start:end][None].expand(batch, -1, -1, -1)
        rival = self.roles[rivals[:, start:end]]
        return F.normalize(candidate - rival, dim=-1)


class VisibleWitnessField(nn.Module):
    """V: shared patch adapter and signed support/refutation ledger."""

    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(DIM, HIDDEN, bias=False)
        self.output_projection = nn.Linear(HIDDEN, DIM, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def adapt(self, patches: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            patches.float()
            + self.output_projection(F.gelu(self.input_projection(patches.float()))),
            dim=-1,
        )

    @staticmethod
    def global_ledger(image_cls: torch.Tensor, queries: torch.Tensor):
        similarity = torch.einsum("bd,bckd->bck", F.normalize(image_cls.float(), dim=-1), queries)
        support, refutation = similarity, -similarity
        margin = 2.0 * similarity
        observable = torch.ones_like(margin)
        entailment = torch.tanh(margin / MARGIN_SCALE)
        ledger = torch.stack((support, refutation, margin, observable, entailment), dim=-1)
        return ledger, similarity

    @staticmethod
    def patch_ledger(adapted: torch.Tensor, queries: torch.Tensor):
        raw = torch.einsum("bnd,bckd->bckn", adapted, queries)
        scaled = raw / TAU_PATCH
        count = raw.size(-1)
        support = TAU_PATCH * torch.logsumexp(scaled, dim=-1) - TAU_PATCH * math.log(count)
        refutation = TAU_PATCH * torch.logsumexp(-scaled, dim=-1) - TAU_PATCH * math.log(count)
        positive = F.softmax(scaled, dim=-1)
        negative = F.softmax(-scaled, dim=-1)
        log_count = math.log(count)
        h_pos = -(positive * torch.log(positive.clamp_min(EPS))).sum(-1)
        h_neg = -(negative * torch.log(negative.clamp_min(EPS))).sum(-1)
        observable = torch.maximum(1.0 - h_pos / log_count, 1.0 - h_neg / log_count)
        margin = support - refutation
        entailment = observable * torch.tanh(margin / MARGIN_SCALE)
        ledger = torch.stack((support, refutation, margin, observable, entailment), dim=-1)
        filip = raw.max(dim=-1).values.mean(dim=-1)
        return ledger, filip, raw.argmax(dim=-1), raw.argmin(dim=-1)


class FalsificationSolver(nn.Module):
    """I: fixed non-compensatory solver plus a stronger MLP control."""

    def __init__(self):
        super().__init__()
        self.ledger_mlp = nn.Sequential(
            nn.Linear(ROLES * 5, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1)
        )

    @staticmethod
    def falsification_score(ledger: torch.Tensor) -> torch.Tensor:
        entailment = ledger[..., 4]
        positive = F.relu(entailment).mean(dim=-1)
        negative = (
            TAU_ROLE * torch.logsumexp(F.relu(-entailment) / TAU_ROLE, dim=-1)
            - TAU_ROLE * math.log(ROLES)
        )
        return positive - REFUTATION_WEIGHT * negative

    @staticmethod
    def compensatory_score(ledger: torch.Tensor) -> torch.Tensor:
        return ledger[..., 4].mean(dim=-1)

    def mlp_score(self, ledger: torch.Tensor) -> torch.Tensor:
        return self.ledger_mlp(ledger.flatten(start_dim=-2)).squeeze(-1)


class OREFModel(nn.Module):
    """Exactly three top-level modules: semantic, visual and interaction."""

    def __init__(self, names, roles, class_ids, *, candidate_chunk_size: int = 5):
        super().__init__()
        self.semantic_module = DynamicRoleClaimModule(names, roles, class_ids)
        self.visual_module = VisibleWitnessField()
        self.interaction_module = FalsificationSolver()
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.call_counts = {}

    def reset_call_counts(self):
        self.call_counts = {
            "role_chunk": 0, "name_chunk": 0, "patch_adapter": 0,
            "falsification_solver": 0, "compensatory_solver": 0,
        }

    def forward(self, image_cls, patch_tokens, *, mode="full"):
        allowed = {"full", "parent", "s_off", "v_off", "i_off", "signed_ledger", "filip", "ledger_mlp"}
        if mode not in allowed:
            raise ValueError(f"OREF mode无效：{mode}")
        if not self.call_counts:
            self.reset_call_counts()
        parent = self.semantic_module.parent_logits(image_cls)
        if mode == "parent":
            return {"logits": parent, "base_logits": parent}
        if mode == "v_off":
            if patch_tokens is not None:
                raise ValueError("OREF V-off物理关闭时patch_tokens必须为None。")
            adapted = None
        else:
            if patch_tokens is None or patch_tokens.ndim != 3 or patch_tokens.size(-1) != DIM:
                raise ValueError("OREF patch tokens必须是[B,N,768]。")
            self.call_counts["patch_adapter"] += 1
            adapted = self.visual_module.adapt(patch_tokens)
        rivals = stable_rivals(parent.detach(), self.semantic_module.class_ids)
        score_rows, ledger_rows, support_ids, refute_ids = [], [], [], []
        for start in range(0, parent.size(1), self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, parent.size(1))
            queries = (
                self.semantic_module.name_chunk(rivals, start, end)
                if mode == "s_off"
                else self.semantic_module.role_chunk(rivals, start, end)
            )
            self.call_counts["name_chunk" if mode == "s_off" else "role_chunk"] += 1
            if mode == "v_off":
                ledger, filip = self.visual_module.global_ledger(image_cls, queries)
                support_id = refute_id = torch.full(
                    ledger.shape[:-1], -1, dtype=torch.long, device=ledger.device
                )
            else:
                ledger, filip, support_id, refute_id = self.visual_module.patch_ledger(adapted, queries)
            if mode == "filip":
                score = filip
            elif mode in {"i_off", "signed_ledger"}:
                self.call_counts["compensatory_solver"] += 1
                score = self.interaction_module.compensatory_score(ledger)
            elif mode == "ledger_mlp":
                score = self.interaction_module.mlp_score(ledger)
            else:
                self.call_counts["falsification_solver"] += 1
                score = self.interaction_module.falsification_score(ledger)
            score_rows.append(score)
            ledger_rows.append(ledger)
            support_ids.append(support_id)
            refute_ids.append(refute_id)
        score = torch.cat(score_rows, dim=1)
        score_z = standardize(score)
        base_std = torch.sqrt(parent.var(dim=1, keepdim=True, unbiased=False) + EPS)
        logits = parent + base_std * torch.tanh(score_z)
        return {
            "logits": logits, "base_logits": parent, "score": score,
            "score_z": score_z, "ledger": torch.cat(ledger_rows, dim=1),
            "argmax_support_patch": torch.cat(support_ids, dim=1),
            "argmax_refute_patch": torch.cat(refute_ids, dim=1),
            "rivals": rivals,
        }


def oref_loss(outputs, targets):
    rows = torch.arange(targets.numel(), device=targets.device)
    ce = F.cross_entropy(outputs["logits"], targets)
    wrong_logits = outputs["base_logits"].detach().clone()
    wrong_logits[rows, targets] = -torch.inf
    hard_wrong = wrong_logits.argmax(dim=1)
    score = outputs["score"]
    rank = F.softplus(0.1 - (score[rows, targets] - score[rows, hard_wrong])).mean()
    return {"total": ce + rank, "ce": ce, "rank": rank}
