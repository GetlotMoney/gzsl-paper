"""Role-Contrast Evidence Gain (RCEG) with exactly S/V/I modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TEXT_DIM = 768
TARGET_DIM = 1024
ROLE_COUNT = 8
MASK_COUNT = 4
HIDDEN_DIM = 64
EPS = 1e-6
ATTENTION_TEMPERATURE = 0.07
BASE_TEMPERATURE = 0.07


def _standardize(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=1, keepdim=True)
    variance = values.var(dim=1, keepdim=True, unbiased=False)
    return (values - mean) / torch.sqrt(variance + EPS)


def stable_rival_positions(
    parent_logits: torch.Tensor, class_ids: torch.Tensor
) -> torch.Tensor:
    """Return each candidate's highest-parent-logit different class."""
    if parent_logits.ndim != 2 or class_ids.ndim != 1:
        raise ValueError("RCEG parent logits/class axis形状错误。")
    if parent_logits.size(1) != class_ids.numel() or class_ids.unique().numel() != class_ids.numel():
        raise ValueError("RCEG parent logits与唯一类别轴不一致。")
    if class_ids.numel() < 2:
        raise ValueError("RCEG rival至少需要两个候选类别。")
    id_order = torch.argsort(class_ids, stable=True)
    ordered_logits = parent_logits.index_select(1, id_order)
    ranked_in_id_order = torch.argsort(
        ordered_logits, dim=1, descending=True, stable=True
    )
    ranked = id_order[ranked_in_id_order]
    top1 = ranked[:, 0]
    top2 = ranked[:, 1]
    candidate_positions = torch.arange(
        class_ids.numel(), device=parent_logits.device
    ).view(1, -1)
    return torch.where(candidate_positions.eq(top1[:, None]), top2[:, None], top1[:, None])


class RoleContrastSemanticModule(nn.Module):
    """S: fixed name anchor plus matched 6+1+1 role contrasts."""

    def __init__(
        self,
        name_embeddings: torch.Tensor,
        role_embeddings: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> None:
        super().__init__()
        if name_embeddings.ndim != 2 or name_embeddings.size(1) != TEXT_DIM:
            raise ValueError("RCEG name embeddings必须是[C,768]。")
        if role_embeddings.shape != (name_embeddings.size(0), ROLE_COUNT, TEXT_DIM):
            raise ValueError("RCEG role embeddings必须是[C,8,768]。")
        if class_ids.shape != (name_embeddings.size(0),):
            raise ValueError("RCEG class_ids必须与文本类别轴一致。")
        self.register_buffer(
            "name_embeddings", F.normalize(name_embeddings.detach().float(), dim=-1)
        )
        self.register_buffer(
            "role_embeddings", F.normalize(role_embeddings.detach().float(), dim=-1)
        )
        self.register_buffer("class_ids", class_ids.detach().long())

    def parent_logits(self, image_cls: torch.Tensor) -> torch.Tensor:
        if image_cls.ndim != 2 or image_cls.size(1) != TEXT_DIM:
            raise ValueError("RCEG image CLS必须是[B,768]。")
        return (
            F.normalize(image_cls.float(), dim=-1)
            @ self.name_embeddings.T
            / BASE_TEMPERATURE
        )

    def chunk(
        self,
        rival_positions: torch.Tensor,
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        batch = rival_positions.size(0)
        candidate_name = self.name_embeddings[start:end]
        candidate_role = self.role_embeddings[start:end]
        rival_index = rival_positions[:, start:end]
        rival_name = self.name_embeddings[rival_index]
        rival_role = self.role_embeddings[rival_index]
        candidate_name = candidate_name.view(1, end - start, TEXT_DIM).expand(
            batch, -1, -1
        )
        candidate_role = candidate_role.view(
            1, end - start, ROLE_COUNT, TEXT_DIM
        ).expand(batch, -1, -1, -1)
        name_query = F.normalize(candidate_name - rival_name, dim=-1)
        role_query = F.normalize(candidate_role - rival_role, dim=-1)
        name_candidate_roles = candidate_name[:, :, None, :].expand(
            -1, -1, ROLE_COUNT, -1
        )
        name_rival_roles = rival_name[:, :, None, :].expand(
            -1, -1, ROLE_COUNT, -1
        )
        name_query_roles = name_query[:, :, None, :].expand(
            -1, -1, ROLE_COUNT, -1
        )
        return {
            "role_query": role_query,
            "name_query": name_query,
            "role_triplet": torch.cat(
                (candidate_role, rival_role, role_query), dim=-1
            ),
            "name_triplet": torch.cat(
                (name_candidate_roles, name_rival_roles, name_query_roles), dim=-1
            ),
        }


class MaskedRoleEvidenceModule(nn.Module):
    """V: fixed positive/negative evidence from non-target visible tokens."""

    @staticmethod
    @torch.no_grad()
    def role_evidence(
        visible_tokens: torch.Tensor, role_queries: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.einsum(
            "bmnd,bckd->bcmkn", visible_tokens.float(), role_queries.float()
        ) / ATTENTION_TEMPERATURE
        positive = F.softmax(scores, dim=-1)
        negative = F.softmax(-scores, dim=-1)
        return torch.einsum(
            "bcmkn,bmnd->bcmkd", positive - negative, visible_tokens.float()
        )

    @staticmethod
    @torch.no_grad()
    def name_evidence(
        visible_tokens: torch.Tensor, name_queries: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.einsum(
            "bmnd,bcd->bcmn", visible_tokens.float(), name_queries.float()
        ) / ATTENTION_TEMPERATURE
        weights = F.softmax(scores, dim=-1) - F.softmax(-scores, dim=-1)
        evidence = torch.einsum(
            "bcmn,bmnd->bcmd", weights, visible_tokens.float()
        )
        return evidence[:, :, :, None, :].expand(-1, -1, -1, ROLE_COUNT, -1)


class NameAnchoredGainModule(nn.Module):
    """I: shared name/role predictor and nested evidence-gain test."""

    def __init__(self) -> None:
        super().__init__()
        self.global_projection = nn.Linear(TEXT_DIM, HIDDEN_DIM, bias=False)
        self.evidence_projection = nn.Linear(TEXT_DIM, HIDDEN_DIM, bias=False)
        self.semantic_projection = nn.Linear(3 * TEXT_DIM, HIDDEN_DIM, bias=False)
        self.output_projection = nn.Linear(HIDDEN_DIM, TARGET_DIM, bias=False)
        self.mask_embedding = nn.Parameter(torch.zeros(MASK_COUNT, HIDDEN_DIM))

    def predict(
        self,
        masked_cls: torch.Tensor,
        evidence: torch.Tensor,
        semantic_triplet: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_code = self.global_projection(masked_cls.float())
        evidence_code = self.evidence_projection(evidence.float())
        semantic_code = self.semantic_projection(semantic_triplet.float())
        hidden = F.gelu(
            global_code[:, None, :, None, :]
            + evidence_code
            + semantic_code[:, :, None, :, :]
            + evidence_code * semantic_code[:, :, None, :, :]
            + self.mask_embedding[None, None, :, None, :]
        ).mean(dim=3)
        raw = self.output_projection(hidden)
        return raw, F.normalize(raw, dim=-1)

    @staticmethod
    def error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target[:, None, :, :].float()).square().sum(dim=-1)


class RCEGModel(nn.Module):
    """Exactly three modules: semantic, visual and interaction."""

    def __init__(
        self,
        name_embeddings: torch.Tensor,
        role_embeddings: torch.Tensor,
        class_ids: torch.Tensor,
        *,
        candidate_chunk_size: int = 5,
    ) -> None:
        super().__init__()
        if int(candidate_chunk_size) <= 0:
            raise ValueError("RCEG candidate_chunk_size必须为正整数。")
        self.semantic_module = RoleContrastSemanticModule(
            name_embeddings, role_embeddings, class_ids
        )
        self.visual_module = MaskedRoleEvidenceModule()
        self.interaction_module = NameAnchoredGainModule()
        self.candidate_chunk_size = int(candidate_chunk_size)

    def forward(
        self,
        image_cls: torch.Tensor,
        masked_cls: torch.Tensor,
        visible_tokens: torch.Tensor,
        target: torch.Tensor | None,
        *,
        mode: str = "full",
    ) -> dict[str, torch.Tensor]:
        allowed = {
            "full", "s_off", "v_off", "i_off", "absolute_role",
            "reference_difficulty", "target_free", "parent",
        }
        if mode not in allowed:
            raise ValueError(f"RCEG mode无效：{mode}")
        batch = image_cls.size(0)
        if masked_cls.shape != (batch, MASK_COUNT, TEXT_DIM):
            raise ValueError("RCEG masked CLS必须是[B,4,768]。")
        if visible_tokens.ndim != 4 or visible_tokens.shape[:2] != (batch, MASK_COUNT):
            raise ValueError("RCEG visible tokens必须是[B,4,N,768]。")
        if visible_tokens.size(-1) != TEXT_DIM:
            raise ValueError("RCEG visible token宽度必须是768。")
        if mode != "target_free":
            if target is None or target.shape != (batch, MASK_COUNT, TARGET_DIM):
                raise ValueError("RCEG target必须是[B,4,1024]。")
        parent_logits = self.semantic_module.parent_logits(image_cls)
        if mode == "parent":
            return {"logits": parent_logits, "base_logits": parent_logits}
        rival_positions = stable_rival_positions(
            parent_logits.detach(), self.semantic_module.class_ids
        )
        scores, name_errors, role_errors = [], [], []
        class_count = parent_logits.size(1)
        for start in range(0, class_count, self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, class_count)
            semantic = self.semantic_module.chunk(rival_positions, start, end)
            if mode == "v_off":
                shape = (
                    batch, end - start, MASK_COUNT, ROLE_COUNT, TEXT_DIM
                )
                role_evidence = visible_tokens.new_zeros(shape, dtype=torch.float32)
                name_evidence = visible_tokens.new_zeros(shape, dtype=torch.float32)
            elif mode == "s_off":
                name_evidence = self.visual_module.name_evidence(
                    visible_tokens, semantic["name_query"]
                )
                role_evidence = name_evidence
            else:
                role_evidence = self.visual_module.role_evidence(
                    visible_tokens, semantic["role_query"]
                )
                name_evidence = self.visual_module.name_evidence(
                    visible_tokens, semantic["name_query"]
                )
            name_raw, name_prediction = self.interaction_module.predict(
                masked_cls, name_evidence, semantic["name_triplet"]
            )
            if mode == "s_off":
                role_raw, role_prediction = name_raw, name_prediction
            else:
                role_raw, role_prediction = self.interaction_module.predict(
                    masked_cls, role_evidence, semantic["role_triplet"]
                )
            if mode == "target_free":
                scores.append(-role_raw.square().sum(dim=-1).mean(dim=-1))
                continue
            current_name_error = self.interaction_module.error(name_prediction, target)
            current_role_error = self.interaction_module.error(role_prediction, target)
            name_errors.append(current_name_error)
            role_errors.append(current_role_error)
            if mode in {"i_off", "absolute_role"}:
                current_score = -torch.log(current_role_error.mean(dim=-1) + EPS)
            elif mode == "reference_difficulty":
                current_score = torch.log(current_name_error.mean(dim=-1) + EPS)
            else:
                current_score = torch.log(
                    (current_name_error + EPS) / (current_role_error + EPS)
                ).mean(dim=-1)
            scores.append(current_score)
        score = torch.cat(scores, dim=1)
        score_z = _standardize(score)
        base_std = torch.sqrt(parent_logits.var(dim=1, keepdim=True, unbiased=False) + EPS)
        logits = parent_logits + base_std * torch.tanh(score_z)
        output = {
            "logits": logits,
            "base_logits": parent_logits,
            "score": score,
            "score_z": score_z,
            "rival_positions": rival_positions,
        }
        if name_errors:
            output["name_error"] = torch.cat(name_errors, dim=1)
            output["role_error"] = torch.cat(role_errors, dim=1)
        return output


def rceg_loss(
    outputs: dict[str, torch.Tensor],
    target_positions: torch.Tensor,
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    logits = outputs["logits"]
    if target_positions.shape != (logits.size(0),):
        raise ValueError("RCEG训练target position形状错误。")
    ce = F.cross_entropy(logits, target_positions.long())
    rows = torch.arange(logits.size(0), device=logits.device)
    parent_wrong = outputs["base_logits"].detach().clone()
    parent_wrong[rows, target_positions] = -torch.inf
    hard_wrong = parent_wrong.argmax(dim=1)
    score = outputs["score"]
    true_score = score[rows, target_positions]
    wrong_score = score[rows, hard_wrong]
    rank = F.softplus(0.1 - (true_score - wrong_score)).mean()
    if mode == "full":
        sign = 0.5 * (
            F.softplus(-true_score).mean() + F.softplus(wrong_score).mean()
        )
    else:
        sign = ce.new_zeros(())
    total = ce + rank + sign
    return {"total": total, "ce": ce, "rank": rank, "sign": sign}
