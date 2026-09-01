"""Counterfactual Utility Active View (CUAV) with S/V/I modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DIM = 768
ACTIONS = 25
HIDDEN = 64
EPS = 1e-6
TEMPERATURE = 0.07


def standardize(values):
    return (values - values.mean(-1, keepdim=True)) / torch.sqrt(
        values.var(-1, keepdim=True, unbiased=False) + EPS
    )


def stable_top2(logits, class_ids):
    id_order = torch.argsort(class_ids, stable=True)
    ranked = id_order[torch.argsort(
        logits.index_select(1, id_order), dim=1, descending=True, stable=True
    )]
    return ranked[:, :2]


class NameAmbiguityState(nn.Module):
    """S: name-only parent ambiguity state."""

    def __init__(self, name_embeddings, class_ids):
        super().__init__()
        if name_embeddings.ndim != 2 or name_embeddings.size(1) != DIM:
            raise ValueError("CUAV name embeddings必须是[C,768]。")
        if class_ids.shape != (name_embeddings.size(0),):
            raise ValueError("CUAV class_ids与name轴不一致。")
        self.register_buffer("names", F.normalize(name_embeddings.float(), dim=-1))
        self.register_buffer("class_ids", class_ids.long())

    def forward(self, full_cls, *, semantic_off=False):
        parent = F.normalize(full_cls.float(), dim=-1) @ self.names.T / TEMPERATURE
        top2 = stable_top2(parent.detach(), self.class_ids)
        rows = torch.arange(parent.size(0), device=parent.device)
        query = F.normalize(self.names[top2[:, 0]] - self.names[top2[:, 1]], dim=-1)
        probabilities = F.softmax(parent, dim=1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(EPS))).sum(1)
        stats = torch.stack((
            parent[rows, top2[:, 0]] - parent[rows, top2[:, 1]],
            entropy, parent.mean(1), parent.std(1, unbiased=False),
        ), dim=1)
        if semantic_off:
            query = torch.zeros_like(query)
            stats = torch.zeros_like(stats)
        return {"parent_logits": parent, "query": query, "stats": stats, "top2": top2}


class CounterfactualGlimpsePolicy(nn.Module):
    """V: discrete 25-action policy; raw-crop execution is runtime-owned."""

    def __init__(self):
        super().__init__()
        self.image_projection = nn.Linear(DIM, HIDDEN, bias=False)
        self.query_projection = nn.Linear(DIM, HIDDEN, bias=False)
        self.stats_projection = nn.Linear(4, HIDDEN, bias=False)
        self.action_projection = nn.Linear(HIDDEN, ACTIONS, bias=False)
        nn.init.zeros_(self.action_projection.weight)

    def forward(self, full_cls, query, stats):
        hidden = F.gelu(
            self.image_projection(full_cls.float())
            + self.query_projection(query.float())
            + self.stats_projection(stats.float())
        )
        logits = self.action_projection(hidden)
        return {"policy_logits": logits, "policy": F.softmax(logits, dim=1), "action": logits.argmax(1)}


class FixedCropEvidenceUpdate(nn.Module):
    """I: fixed relative or absolute crop evidence update."""

    @staticmethod
    def crop_logits(crop_cls, names):
        return F.normalize(crop_cls.float(), dim=-1) @ names.T / TEMPERATURE

    @staticmethod
    def full_update(parent, crop_logits):
        delta = standardize(crop_logits - parent)
        scale = torch.sqrt(parent.var(-1, keepdim=True, unbiased=False) + EPS)
        return parent + scale * torch.tanh(delta)

    @staticmethod
    def interaction_off(parent, crop_logits):
        delta = standardize(crop_logits)
        scale = torch.sqrt(parent.var(-1, keepdim=True, unbiased=False) + EPS)
        return parent + scale * torch.tanh(delta)


class CUAVModel(nn.Module):
    """Exactly three top-level modules: semantic, visual policy and interaction."""

    def __init__(self, name_embeddings, class_ids):
        super().__init__()
        self.semantic_module = NameAmbiguityState(name_embeddings, class_ids)
        self.visual_module = CounterfactualGlimpsePolicy()
        self.interaction_module = FixedCropEvidenceUpdate()
        self.call_counts = {}

    def reset_call_counts(self):
        self.call_counts = {"semantic_state": 0, "policy": 0, "relative_update": 0, "absolute_update": 0}

    def policy(self, full_cls, *, semantic_off=False):
        if not self.call_counts:
            self.reset_call_counts()
        state = self.semantic_module(full_cls, semantic_off=semantic_off)
        self.call_counts["semantic_state"] += 0 if semantic_off else 1
        policy = self.visual_module(full_cls, state["query"], state["stats"])
        self.call_counts["policy"] += 1
        return {**state, **policy}

    def training_forward(self, full_cls, all_crop_cls, *, semantic_off=False, interaction_off=False):
        if all_crop_cls.ndim != 3 or all_crop_cls.shape[1:] != (ACTIONS, DIM):
            raise ValueError("CUAV training crop features必须是[B,25,768]。")
        output = self.policy(full_cls, semantic_off=semantic_off)
        crop_logits = torch.einsum(
            "bad,cd->bac", F.normalize(all_crop_cls.float(), dim=-1), self.semantic_module.names
        ) / TEMPERATURE
        parent = output["parent_logits"][:, None, :].expand_as(crop_logits)
        if interaction_off:
            self.call_counts["absolute_update"] += 1
            final = self.interaction_module.interaction_off(parent, crop_logits)
        else:
            self.call_counts["relative_update"] += 1
            final = self.interaction_module.full_update(parent, crop_logits)
        return {**output, "action_crop_logits": crop_logits, "action_final_logits": final}

    def selected_update(self, full_cls, crop_cls, *, semantic_off=False, interaction_off=False):
        output = self.policy(full_cls, semantic_off=semantic_off)
        crop_logits = self.interaction_module.crop_logits(crop_cls, self.semantic_module.names)
        if interaction_off:
            self.call_counts["absolute_update"] += 1
            final = self.interaction_module.interaction_off(output["parent_logits"], crop_logits)
        else:
            self.call_counts["relative_update"] += 1
            final = self.interaction_module.full_update(output["parent_logits"], crop_logits)
        return {**output, "crop_logits": crop_logits, "logits": final}


def cuav_action_losses(outputs, targets):
    batch, actions, classes = outputs["action_final_logits"].shape
    expanded_targets = targets[:, None].expand(-1, actions).reshape(-1)
    ce = F.cross_entropy(
        outputs["action_final_logits"].reshape(batch * actions, classes),
        expanded_targets, reduction="none",
    ).reshape(batch, actions)
    rows = torch.arange(batch, device=targets.device)
    parent_wrong = outputs["parent_logits"].detach().clone()
    parent_wrong[rows, targets] = -torch.inf
    wrong = parent_wrong.argmax(1)
    true_values = outputs["action_final_logits"].gather(
        2, targets[:, None, None].expand(-1, actions, 1)
    ).squeeze(-1)
    wrong_values = outputs["action_final_logits"].gather(
        2, wrong[:, None, None].expand(-1, actions, 1)
    ).squeeze(-1)
    rank = F.softplus(0.1 - (true_values - wrong_values))
    return ce + rank


def cuav_policy_loss(outputs, targets):
    action_losses = cuav_action_losses(outputs, targets)
    batch, actions, classes = outputs["action_final_logits"].shape
    expanded_targets = targets[:, None].expand(-1, actions).reshape(-1)
    ce_only = F.cross_entropy(
        outputs["action_final_logits"].reshape(batch * actions, classes),
        expanded_targets, reduction="none",
    ).reshape(batch, actions)
    rank_only = action_losses - ce_only
    expected = (outputs["policy"] * action_losses).sum(1).mean()
    return {
        "total": expected,
        "expected_ce": (outputs["policy"] * ce_only).sum(1).mean(),
        "expected_rank": (outputs["policy"] * rank_only).sum(1).mean(),
    }
