"""Semantic-Visual Risk Arbiter (SVRA) V6 framework pieces.

SVRA keeps the deployable contract to three modules:

S: eight role-conditioned natural-language difference questions represented by
   precomputed role text embeddings.
V: a 25-action window policy with an explicit zero-abstain action.
I: a parent-risk arbiter that decides whether to keep or swap the parent Top-2.

The main SVRA forward path intentionally does not consume crop features.  The
13-D ceiling helpers are exposed for offline diagnosis only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FEATURE_DIM: Final[int] = 768
ROLE_COUNT: Final[int] = 8
ACTION_COUNT: Final[int] = 25
PATCH_GRID: Final[int] = 24
PATCH_COUNT: Final[int] = PATCH_GRID * PATCH_GRID
WINDOW_SIZE: Final[int] = 6
WINDOW_STARTS: Final[tuple[int, ...]] = (0, 4, 9, 14, 18)
HIDDEN_DIM: Final[int] = 64
ACTION_HEAD_INPUT_DIM: Final[int] = 261
MAIN_RISK_INPUT_DIM: Final[int] = 4
CEILING_RISK_INPUT_DIM: Final[int] = 13
PAIR_TEMPERATURE: Final[float] = 0.07
LAYER_NORM_EPS: Final[float] = 1e-5
ATTENTION_DENOM_EPS: Final[float] = 1e-6
ACTION_GEOMETRY_SHA256: Final[str] = (
    "4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb"
)
JSVRA_ACTION_POS_WEIGHT: Final[float] = 6065 / 992
JSVRA_RISK_POS_WEIGHT: Final[float] = (4485 + 1550) / 1022
JSVRA_LOSS_COEFFICIENT: Final[float] = 1.0

ROLE_ORDER: Final[tuple[str, ...]] = (
    "beak",
    "head_features",
    "body_plumage",
    "wings",
    "tail",
    "legs",
    "overall_appearance",
    "unique_discriminative_features",
)


@dataclass(frozen=True)
class PairState:
    """Parent Top-2 state plus S-module questions."""

    parent_logits: Tensor
    top2: Tensor
    leader_ids: Tensor
    challenger_ids: Tensor
    leader_margin: Tensor
    parent_stats: Tensor
    questions: Tensor


@dataclass(frozen=True)
class ActionPolicyState:
    """V-module role-to-window policy state."""

    pair: PairState
    utility_logits: Tensor
    utility: Tensor
    policy_logits: Tensor
    policy: Tensor
    selected_action: Tensor
    selected_policy_confidence: Tensor
    trigger: Tensor
    attention: Tensor
    attention_mass: Tensor
    window_features: Tensor
    window_keys: Tensor
    window_values: Tensor
    role_values: Tensor
    action_head_input: Tensor

    @property
    def parent_logits(self) -> Tensor:
        return self.pair.parent_logits

    @property
    def top2(self) -> Tensor:
        return self.pair.top2

    @property
    def leader_ids(self) -> Tensor:
        return self.pair.leader_ids

    @property
    def challenger_ids(self) -> Tensor:
        return self.pair.challenger_ids

    @property
    def parent_stats(self) -> Tensor:
        return self.pair.parent_stats

    @property
    def risk_features13(self) -> Tensor:
        return build_ceiling_risk_inputs(
            self.pair.parent_stats,
            self.selected_policy_confidence,
            self.selected_action,
        )


@dataclass(frozen=True)
class SVRAOutput:
    """Full SVRA output after parent-risk keep/swap arbitration."""

    pair: PairState
    action_state: ActionPolicyState
    logits: Tensor
    parent_logits: Tensor
    risk_logits: Tensor
    swap_probability: Tensor
    swapped: Tensor


@dataclass(frozen=True)
class JointSVRATargets:
    """Full200 J-SVRA targets for one train batch."""

    action_targets26: Tensor
    risk_targets: Tensor
    joint_targets: Tensor
    top2_group: Tensor
    conflict_mask: Tensor
    best_action_margin: Tensor


@dataclass(frozen=True)
class JointSVRAOutput:
    """Differentiable J-SVRA forward values used by the precheck objective."""

    pair: PairState
    action_state: ActionPolicyState
    parent_logits: Tensor
    action_logits25: Tensor
    policy_logits26: Tensor
    max_action_logit: Tensor
    opportunity_probability: Tensor
    hard_trigger: Tensor
    risk_logits: Tensor
    risk_probability: Tensor
    joint_probability: Tensor
    logits: Tensor
    swapped: Tensor
    soft_hard_trigger_equal: bool


@dataclass(frozen=True)
class JointSVRALoss:
    """Raw and weighted loss receipts for the fixed IDEA-200 objective."""

    total: Tensor
    action: Tensor
    risk: Tensor
    joint: Tensor
    raw_means: dict[str, Tensor]
    weighted_means: dict[str, Tensor]
    weights: dict[str, float]
    soft_hard_trigger_equal: bool


def _require_rank(name: str, value: Tensor, rank: int) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {tuple(value.shape)}")


def _require_last_dim(name: str, value: Tensor, size: int) -> None:
    if value.shape[-1] != size:
        raise ValueError(
            f"{name} last dim must be {size}, got shape {tuple(value.shape)}"
        )


def _l2_normalize(value: Tensor) -> Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-12)


def _action_window_slices() -> tuple[tuple[int, int], ...]:
    return tuple((y, x) for y in WINDOW_STARTS for x in WINDOW_STARTS)


def make_action_positions(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the fixed [25, 8] normalized action-position matrix."""

    values: list[list[float]] = []
    for y, x in _action_window_slices():
        values.append(
            [
                x / PATCH_GRID,
                y / PATCH_GRID,
                (x + WINDOW_SIZE) / PATCH_GRID,
                (y + WINDOW_SIZE) / PATCH_GRID,
                (x + WINDOW_SIZE / 2) / PATCH_GRID,
                (y + WINDOW_SIZE / 2) / PATCH_GRID,
                WINDOW_SIZE / PATCH_GRID,
                WINDOW_SIZE / PATCH_GRID,
            ]
        )
    return torch.tensor(values, device=device, dtype=dtype)


def pool_action_windows(patch_tokens: Tensor) -> Tensor:
    """Pool 24x24 projected patch tokens into fixed 25 6x6 windows."""

    _require_rank("patch_tokens", patch_tokens, 3)
    if patch_tokens.shape[1] != PATCH_COUNT:
        raise ValueError(
            f"patch_tokens second dim must be {PATCH_COUNT}, got {patch_tokens.shape[1]}"
        )
    _require_last_dim("patch_tokens", patch_tokens, FEATURE_DIM)

    batch = patch_tokens.shape[0]
    grid = patch_tokens.float().reshape(batch, PATCH_GRID, PATCH_GRID, FEATURE_DIM)
    pooled = []
    for y, x in _action_window_slices():
        pooled.append(grid[:, y : y + WINDOW_SIZE, x : x + WINDOW_SIZE].mean(dim=(1, 2)))
    return torch.stack(pooled, dim=1)


def stable_top2_by_logit_then_class_id(logits: Tensor, class_ids: Tensor) -> Tensor:
    """Return local top-2 indices, breaking equal logits by smaller class id."""

    _require_rank("logits", logits, 2)
    _require_rank("class_ids", class_ids, 1)
    if logits.shape[1] != class_ids.numel():
        raise ValueError(
            "logits class axis and class_ids length mismatch: "
            f"{logits.shape[1]} vs {class_ids.numel()}"
        )
    if class_ids.numel() < 2:
        raise ValueError("SVRA requires at least two active classes")

    ascending_id_order = torch.argsort(class_ids, stable=True)
    logits_by_id = logits.index_select(1, ascending_id_order)
    ranked_by_id = torch.argsort(logits_by_id, dim=1, descending=True, stable=True)
    return ascending_id_order.index_select(0, ranked_by_id[:, :2].reshape(-1)).reshape(
        logits.shape[0], 2
    )


def _parent_logits_from_cls(
    cls_features: Tensor,
    name_embeddings: Tensor,
    *,
    temperature: float = PAIR_TEMPERATURE,
) -> Tensor:
    _require_rank("cls_features", cls_features, 2)
    _require_last_dim("cls_features", cls_features, FEATURE_DIM)
    return _l2_normalize(cls_features) @ _l2_normalize(name_embeddings).t() / temperature


class EightRoleSemanticQuestions(nn.Module):
    """S module: build eight role-conditioned leader/challenger questions."""

    def __init__(
        self,
        role_embeddings: Tensor,
        name_embeddings: Tensor,
        class_ids: Tensor,
    ) -> None:
        super().__init__()
        self._validate_assets(role_embeddings, name_embeddings, class_ids)

        self.register_buffer(
            "role_embeddings", _l2_normalize(role_embeddings), persistent=False
        )
        self.register_buffer(
            "name_embeddings", _l2_normalize(name_embeddings), persistent=False
        )
        self.register_buffer("class_ids", class_ids.long().clone(), persistent=False)

        self.role_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM, bias=False)
        self.name_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM, bias=False)
        self.role_id = nn.Embedding(ROLE_COUNT, HIDDEN_DIM)
        self.question_norm = nn.LayerNorm(
            HIDDEN_DIM, eps=LAYER_NORM_EPS, elementwise_affine=True
        )

    @staticmethod
    def _validate_assets(
        role_embeddings: Tensor,
        name_embeddings: Tensor,
        class_ids: Tensor,
    ) -> None:
        _require_rank("role_embeddings", role_embeddings, 3)
        _require_rank("name_embeddings", name_embeddings, 2)
        _require_rank("class_ids", class_ids, 1)
        if role_embeddings.shape[1:] != (ROLE_COUNT, FEATURE_DIM):
            raise ValueError(
                "role_embeddings must have shape [C, 8, 768], "
                f"got {tuple(role_embeddings.shape)}"
            )
        if name_embeddings.shape[1] != FEATURE_DIM:
            raise ValueError(
                "name_embeddings must have shape [C, 768], "
                f"got {tuple(name_embeddings.shape)}"
            )
        if role_embeddings.shape[0] != name_embeddings.shape[0]:
            raise ValueError("role_embeddings and name_embeddings class counts differ")
        if class_ids.numel() != name_embeddings.shape[0]:
            raise ValueError("class_ids length and embedding class count differ")
        if torch.unique(class_ids.long()).numel() != class_ids.numel():
            raise ValueError("class_ids must be unique on the active axis")

    def forward(self, full_cls: Tensor, *, semantic_off: bool = False) -> PairState:
        _require_rank("full_cls", full_cls, 2)
        _require_last_dim("full_cls", full_cls, FEATURE_DIM)

        names = self.name_embeddings.to(device=full_cls.device)
        class_ids = self.class_ids.to(device=full_cls.device)
        parent_logits = _parent_logits_from_cls(full_cls, names)
        top2 = stable_top2_by_logit_then_class_id(parent_logits, class_ids)

        row = torch.arange(full_cls.shape[0], device=full_cls.device)
        leader = top2[:, 0]
        challenger = top2[:, 1]
        leader_ids = class_ids.index_select(0, leader)
        challenger_ids = class_ids.index_select(0, challenger)

        leader_logits = parent_logits[row, leader]
        challenger_logits = parent_logits[row, challenger]
        probs = torch.softmax(parent_logits.float(), dim=1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        parent_stats = torch.stack(
            [
                leader_logits - challenger_logits,
                entropy,
                parent_logits.mean(dim=1),
                parent_logits.std(dim=1, unbiased=False),
            ],
            dim=1,
        ).float()

        if semantic_off:
            questions = full_cls.new_zeros(full_cls.shape[0], ROLE_COUNT, HIDDEN_DIM)
        else:
            name_delta = _l2_normalize(
                names.index_select(0, leader) - names.index_select(0, challenger)
            )
            roles = self.role_embeddings.to(device=full_cls.device)
            role_delta = _l2_normalize(
                roles.index_select(0, leader) - roles.index_select(0, challenger)
            )
            role_ids = torch.arange(ROLE_COUNT, device=full_cls.device)
            questions = self.question_norm(
                self.role_projection(role_delta)
                + self.name_projection(name_delta).unsqueeze(1)
                + self.role_id(role_ids).unsqueeze(0)
            )

        return PairState(
            parent_logits=parent_logits,
            top2=top2,
            leader_ids=leader_ids,
            challenger_ids=challenger_ids,
            leader_margin=leader_logits - challenger_logits,
            parent_stats=parent_stats,
            questions=questions,
        )


class RoleToWindowActionPolicy(nn.Module):
    """V module: role-to-window attention and explicit abstain+25 policy."""

    def __init__(self) -> None:
        super().__init__()
        window_input_dim = FEATURE_DIM + FEATURE_DIM + 8
        self.window_key = nn.Linear(window_input_dim, HIDDEN_DIM, bias=False)
        self.window_value = nn.Linear(window_input_dim, HIDDEN_DIM, bias=False)
        self.role_value = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.window_norm = nn.LayerNorm(
            HIDDEN_DIM, eps=LAYER_NORM_EPS, elementwise_affine=True
        )
        self.utility_hidden = nn.Linear(ACTION_HEAD_INPUT_DIM, HIDDEN_DIM, bias=False)
        self.utility_output = nn.Linear(HIDDEN_DIM, 1, bias=False)
        self.register_buffer(
            "action_positions", make_action_positions(), persistent=False
        )
        nn.init.zeros_(self.utility_output.weight)

    def _window_features(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        *,
        visual_off: bool,
    ) -> Tensor:
        z_full = _l2_normalize(full_cls)
        if visual_off:
            return z_full[:, None, :].expand(-1, ACTION_COUNT, -1)
        if patch_tokens is None:
            raise ValueError("patch_tokens is required unless visual_off=True")
        return pool_action_windows(patch_tokens).to(device=full_cls.device)

    def forward(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        pair: PairState,
        *,
        visual_off: bool = False,
    ) -> ActionPolicyState:
        _require_rank("full_cls", full_cls, 2)
        _require_last_dim("full_cls", full_cls, FEATURE_DIM)
        _require_rank("questions", pair.questions, 3)
        if pair.questions.shape[1:] != (ROLE_COUNT, HIDDEN_DIM):
            raise ValueError(
                "pair.questions must have shape [B, 8, 64], "
                f"got {tuple(pair.questions.shape)}"
            )
        if pair.parent_stats.shape != (full_cls.shape[0], MAIN_RISK_INPUT_DIM):
            raise ValueError(
                "pair.parent_stats must have shape [B, 4], "
                f"got {tuple(pair.parent_stats.shape)}"
            )

        windows = self._window_features(full_cls, patch_tokens, visual_off=visual_off)
        z_full = _l2_normalize(full_cls)
        positions = self.action_positions.to(
            device=full_cls.device, dtype=windows.dtype
        ).expand(full_cls.shape[0], -1, -1)
        window_input = torch.cat([windows, windows - z_full[:, None, :], positions], dim=2)

        keys = self.window_norm(self.window_key(window_input))
        values = self.window_value(window_input)
        role_values = self.role_value(pair.questions.float())

        scores = torch.einsum("brd,bad->bra", pair.questions.float(), keys)
        attention = torch.softmax(scores / math.sqrt(HIDDEN_DIM), dim=2)
        attention_mass = attention.sum(dim=1) / ROLE_COUNT
        role_mass = attention.sum(dim=1).unsqueeze(2)
        context = torch.einsum("bra,brd->bad", attention, role_values) / (
            role_mass + ATTENTION_DENOM_EPS
        )

        stats = pair.parent_stats.to(device=full_cls.device, dtype=keys.dtype)
        action_head_input = torch.cat(
            [
                keys,
                values,
                context,
                values * context,
                attention_mass.unsqueeze(2),
                stats[:, None, :].expand(-1, ACTION_COUNT, -1),
            ],
            dim=2,
        )
        if action_head_input.shape[2] != ACTION_HEAD_INPUT_DIM:
            raise RuntimeError(
                "SVRA action head input contract violated: "
                f"expected {ACTION_HEAD_INPUT_DIM}, got {action_head_input.shape[2]}"
            )

        utility_logits = self.utility_output(F.gelu(self.utility_hidden(action_head_input))).squeeze(2)
        abstain = torch.zeros(
            utility_logits.shape[0],
            1,
            device=utility_logits.device,
            dtype=utility_logits.dtype,
        )
        policy_logits = torch.cat([abstain, utility_logits], dim=1)
        policy = torch.softmax(policy_logits, dim=1)
        choice = policy_logits.argmax(dim=1)
        selected_action = (choice - 1).clamp_min(0)
        trigger = choice > 0
        utility = policy[:, 1:]
        selected_policy_confidence = policy.gather(1, choice[:, None]).squeeze(1)

        return ActionPolicyState(
            pair=pair,
            utility_logits=utility_logits,
            utility=utility,
            policy_logits=policy_logits,
            policy=policy,
            selected_action=selected_action,
            selected_policy_confidence=selected_policy_confidence,
            trigger=trigger,
            attention=attention,
            attention_mass=attention_mass,
            window_features=windows,
            window_keys=keys,
            window_values=values,
            role_values=role_values,
            action_head_input=action_head_input,
        )


class ParentRiskArbiter(nn.Module):
    """I module: decide whether selected evidence should swap parent Top-2."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(MAIN_RISK_INPUT_DIM, 32)
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, parent_stats: Tensor) -> Tensor:
        _require_rank("parent_stats", parent_stats, 2)
        _require_last_dim("parent_stats", parent_stats, MAIN_RISK_INPUT_DIM)
        return self.output(F.gelu(self.hidden(parent_stats.float()))).squeeze(1)

    @staticmethod
    def apply_keep_swap(
        parent_logits: Tensor,
        top2: Tensor,
        trigger: Tensor,
        risk_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _require_rank("parent_logits", parent_logits, 2)
        _require_rank("top2", top2, 2)
        if top2.shape != (parent_logits.shape[0], 2):
            raise ValueError(f"top2 must have shape [B, 2], got {tuple(top2.shape)}")
        if trigger.shape != (parent_logits.shape[0],):
            raise ValueError(f"trigger must have shape [B], got {tuple(trigger.shape)}")
        if risk_logits.shape != (parent_logits.shape[0],):
            raise ValueError(
                f"risk_logits must have shape [B], got {tuple(risk_logits.shape)}"
            )

        swap_probability = torch.sigmoid(risk_logits)
        swap = trigger & (swap_probability > 0.5)
        logits = parent_logits.clone()
        if swap.any():
            rows = torch.nonzero(swap, as_tuple=False).flatten()
            leaders = top2.index_select(0, rows)[:, 0]
            challengers = top2.index_select(0, rows)[:, 1]
            leader_values = logits[rows, leaders].clone()
            logits[rows, leaders] = logits[rows, challengers]
            logits[rows, challengers] = leader_values
        return logits, swap_probability, swap


class ParentRiskCeilingArbiter(nn.Module):
    """Offline diagnostic arbiter over 13-D parent/action/box features."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(CEILING_RISK_INPUT_DIM, 32)
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, ceiling_inputs: Tensor) -> Tensor:
        _require_rank("ceiling_inputs", ceiling_inputs, 2)
        _require_last_dim("ceiling_inputs", ceiling_inputs, CEILING_RISK_INPUT_DIM)
        return self.output(F.gelu(self.hidden(ceiling_inputs.float()))).squeeze(1)


def build_ceiling_risk_inputs(
    parent_stats: Tensor,
    selected_policy_confidence: Tensor,
    selected_action: Tensor,
    *,
    action_positions: Tensor | None = None,
) -> Tensor:
    """Build the offline 13-D ceiling input: 4 parent stats + 1 confidence + box8."""

    _require_rank("parent_stats", parent_stats, 2)
    _require_last_dim("parent_stats", parent_stats, MAIN_RISK_INPUT_DIM)
    if selected_policy_confidence.shape != (parent_stats.shape[0],):
        raise ValueError(
            "selected_policy_confidence must have shape [B], "
            f"got {tuple(selected_policy_confidence.shape)}"
        )
    if selected_action.shape != (parent_stats.shape[0],):
        raise ValueError(
            f"selected_action must have shape [B], got {tuple(selected_action.shape)}"
        )

    if action_positions is None:
        action_positions = make_action_positions(
            device=parent_stats.device, dtype=parent_stats.dtype
        )
    if action_positions.shape != (ACTION_COUNT, 8):
        raise ValueError(
            "action_positions must have shape [25, 8], "
            f"got {tuple(action_positions.shape)}"
        )
    selected_action = selected_action.to(device=parent_stats.device, dtype=torch.long)
    if selected_action.numel() and (
        int(selected_action.min().item()) < 0
        or int(selected_action.max().item()) >= ACTION_COUNT
    ):
        raise ValueError("selected_action values must be in [0, 25)")
    boxes = action_positions.to(device=parent_stats.device, dtype=parent_stats.dtype).index_select(
        0, selected_action
    )
    return torch.cat(
        [
            parent_stats.float(),
            selected_policy_confidence.to(device=parent_stats.device).float().unsqueeze(1),
            boxes.float(),
        ],
        dim=1,
    )


def ceiling_risk_mask(trigger: Tensor, top2_group: Tensor | None = None) -> Tensor:
    """Return rows eligible for the offline 13-D ceiling arbiter."""

    _require_rank("trigger", trigger, 1)
    mask = trigger.bool()
    if top2_group is not None:
        if top2_group.shape != trigger.shape:
            raise ValueError(
                f"top2_group must have shape {tuple(trigger.shape)}, got {tuple(top2_group.shape)}"
            )
        mask = mask & (top2_group < 2)
    return mask


class SemanticVisualRiskArbiter(nn.Module):
    """SVRA S/V/I wrapper with deploy and diagnostic risk heads."""

    def __init__(
        self,
        role_embeddings: Tensor,
        name_embeddings: Tensor,
        class_ids: Tensor,
        *,
        seed: int = 7,
    ) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.semantic = EightRoleSemanticQuestions(
                role_embeddings=role_embeddings,
                name_embeddings=name_embeddings,
                class_ids=class_ids,
            )
            self.visual = RoleToWindowActionPolicy()
            self.interaction = ParentRiskArbiter()
            self.trigger_arbiter13_ceiling = ParentRiskCeilingArbiter()
            self.allrow_arbiter4_control = ParentRiskArbiter()

    @property
    def name_embeddings(self) -> Tensor:
        return self.semantic.name_embeddings

    @property
    def class_ids(self) -> Tensor:
        return self.semantic.class_ids

    def parent_state(self, full_cls: Tensor, *, semantic_off: bool = False) -> PairState:
        return self.semantic(full_cls, semantic_off=semantic_off)

    def action_state(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        *,
        semantic_off: bool = False,
        visual_off: bool = False,
    ) -> ActionPolicyState:
        pair = self.parent_state(full_cls, semantic_off=semantic_off)
        return self.visual(full_cls, patch_tokens, pair, visual_off=visual_off)

    def utility_state(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        *,
        semantic_off: bool = False,
        visual_off: bool = False,
    ) -> ActionPolicyState:
        return self.action_state(
            full_cls,
            patch_tokens,
            semantic_off=semantic_off,
            visual_off=visual_off,
        )

    def policy_state(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        *,
        semantic_off: bool = False,
        visual_off: bool = False,
    ) -> ActionPolicyState:
        return self.action_state(
            full_cls,
            patch_tokens,
            semantic_off=semantic_off,
            visual_off=visual_off,
        )

    def risk_probability(self, features: Tensor, *, head: str) -> Tensor:
        if head == "triggered4d":
            logits = self.interaction(features)
        elif head == "all_row4d":
            logits = self.allrow_arbiter4_control(features)
        elif head == "ceiling13d":
            logits = self.trigger_arbiter13_ceiling(features)
        else:
            raise ValueError(f"unknown SVRA risk head: {head}")
        return torch.sigmoid(logits)

    def risk_probabilities(self, state: ActionPolicyState) -> dict[str, Tensor]:
        return {
            "triggered4d": self.risk_probability(
                state.parent_stats, head="triggered4d"
            ),
            "all_row4d": self.risk_probability(
                state.parent_stats, head="all_row4d"
            ),
            "ceiling13d": self.risk_probability(
                state.risk_features13, head="ceiling13d"
            ),
        }

    def forward(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        *,
        semantic_off: bool = False,
        visual_off: bool = False,
        interaction_off: bool = False,
    ) -> SVRAOutput:
        action_state = self.action_state(
            full_cls,
            patch_tokens,
            semantic_off=semantic_off,
            visual_off=visual_off,
        )
        pair = action_state.pair
        if interaction_off:
            batch = full_cls.shape[0]
            risk_logits = full_cls.new_zeros(batch)
            return SVRAOutput(
                pair=pair,
                action_state=action_state,
                logits=pair.parent_logits,
                parent_logits=pair.parent_logits,
                risk_logits=risk_logits,
                swap_probability=torch.sigmoid(risk_logits),
                swapped=torch.zeros(batch, dtype=torch.bool, device=full_cls.device),
            )

        risk_logits = self.interaction(pair.parent_stats)
        logits, swap_probability, swapped = self.interaction.apply_keep_swap(
            pair.parent_logits,
            pair.top2,
            action_state.trigger,
            risk_logits,
        )
        return SVRAOutput(
            pair=pair,
            action_state=action_state,
            logits=logits,
            parent_logits=pair.parent_logits,
            risk_logits=risk_logits,
            swap_probability=swap_probability,
            swapped=swapped,
        )

    def dense_utility_targets(
        self,
        full_cls: Tensor,
        all_crop_cls: Tensor,
        target_class_ids: Tensor,
        *,
        semantic_off: bool = False,
    ) -> tuple[Tensor, Tensor, PairState]:
        pair = self.parent_state(full_cls, semantic_off=semantic_off)
        targets, group = dense_utility_targets_from_logits(
            pair,
            all_crop_cls,
            self.name_embeddings.to(device=full_cls.device),
            target_class_ids,
            self.class_ids.to(device=full_cls.device),
        )
        return targets, group, pair

    def joint_forward(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        *,
        semantic_off: bool = False,
        visual_off: bool = False,
        interaction_off: bool = False,
    ) -> JointSVRAOutput:
        """Run the IDEA-200 differentiable conjunction path."""

        return joint_svra_forward(
            self,
            full_cls,
            patch_tokens,
            semantic_off=semantic_off,
            visual_off=visual_off,
            interaction_off=interaction_off,
        )


def dense_utility_targets_from_logits(
    pair: PairState,
    all_crop_cls: Tensor,
    name_embeddings: Tensor,
    target_class_ids: Tensor,
    class_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build [B,25] utility targets from all25 crop CLS for training only."""

    _require_rank("all_crop_cls", all_crop_cls, 3)
    if all_crop_cls.shape[1:] != (ACTION_COUNT, FEATURE_DIM):
        raise ValueError(
            "all_crop_cls must have shape [B,25,768], "
            f"got {tuple(all_crop_cls.shape)}"
        )
    if target_class_ids.shape != (all_crop_cls.shape[0],):
        raise ValueError(
            "target_class_ids must have shape [B], "
            f"got {tuple(target_class_ids.shape)}"
        )

    device = all_crop_cls.device
    names = name_embeddings.to(device=device)
    crop_logits = torch.einsum("bad,cd->bac", _l2_normalize(all_crop_cls), _l2_normalize(names))
    crop_logits = crop_logits / PAIR_TEMPERATURE

    local_truth = _global_to_local_indices(
        target_class_ids.to(device=device, dtype=torch.long),
        class_ids.to(device=device, dtype=torch.long),
    )
    row = torch.arange(all_crop_cls.shape[0], device=device)
    action = torch.arange(ACTION_COUNT, device=device)
    leader = pair.top2[:, 0].to(device=device)
    challenger = pair.top2[:, 1].to(device=device)

    leader_score = crop_logits[row[:, None], action, leader[:, None]]
    challenger_score = crop_logits[row[:, None], action, challenger[:, None]]
    crop_keeps_leader = leader_score >= challenger_score

    group = torch.full_like(local_truth, fill_value=2)
    group = torch.where(local_truth == leader, torch.zeros_like(group), group)
    group = torch.where(local_truth == challenger, torch.ones_like(group), group)

    targets = torch.zeros(
        all_crop_cls.shape[0],
        ACTION_COUNT,
        dtype=crop_logits.dtype,
        device=device,
    )
    leader_rows = group == 0
    challenger_rows = group == 1
    targets[leader_rows] = crop_keeps_leader[leader_rows].float()
    targets[challenger_rows] = (~crop_keeps_leader[challenger_rows]).float()
    return targets, group


def joint_action_targets_from_logits(
    pair: PairState,
    all_crop_cls: Tensor,
    name_embeddings: Tensor,
    target_class_ids: Tensor,
    class_ids: Tensor,
) -> JointSVRATargets:
    """Build J-SVRA 26-way abstain/action, risk, and joint targets.

    The action target is positive only when the ground-truth class is the Parent
    challenger and at least one of the 25 train-only counterfactual crop CLS
    vectors makes the challenger beat the leader. Challenger rows without such
    an action are the deliberate factorization-conflict rows: I-positive but
    V/joint-negative.
    """

    _require_rank("all_crop_cls", all_crop_cls, 3)
    if all_crop_cls.shape[1:] != (ACTION_COUNT, FEATURE_DIM):
        raise ValueError(
            "all_crop_cls must have shape [B,25,768], "
            f"got {tuple(all_crop_cls.shape)}"
        )
    if target_class_ids.shape != (all_crop_cls.shape[0],):
        raise ValueError(
            "target_class_ids must have shape [B], "
            f"got {tuple(target_class_ids.shape)}"
        )

    device = all_crop_cls.device
    names = name_embeddings.to(device=device)
    crop_logits = torch.einsum(
        "bad,cd->bac", _l2_normalize(all_crop_cls), _l2_normalize(names)
    )
    crop_logits = crop_logits / PAIR_TEMPERATURE

    local_truth = _global_to_local_indices(
        target_class_ids.to(device=device, dtype=torch.long),
        class_ids.to(device=device, dtype=torch.long),
    )
    row = torch.arange(all_crop_cls.shape[0], device=device)
    action = torch.arange(ACTION_COUNT, device=device)
    leader = pair.top2[:, 0].to(device=device)
    challenger = pair.top2[:, 1].to(device=device)

    leader_score = crop_logits[row[:, None], action, leader[:, None]]
    challenger_score = crop_logits[row[:, None], action, challenger[:, None]]
    best_action_margin, best_action = (challenger_score - leader_score).max(dim=1)

    top2_group = torch.full_like(local_truth, fill_value=2)
    top2_group = torch.where(local_truth == leader, torch.zeros_like(top2_group), top2_group)
    top2_group = torch.where(
        local_truth == challenger, torch.ones_like(top2_group), top2_group
    )
    action_positive = (top2_group == 1) & (best_action_margin > 0)
    action_targets26 = torch.zeros_like(local_truth)
    action_targets26 = torch.where(action_positive, best_action + 1, action_targets26)
    risk_targets = (top2_group == 1).float()
    joint_targets = action_positive.float()
    conflict_mask = (top2_group == 1) & ~action_positive
    return JointSVRATargets(
        action_targets26=action_targets26.long(),
        risk_targets=risk_targets,
        joint_targets=joint_targets,
        top2_group=top2_group.long(),
        conflict_mask=conflict_mask,
        best_action_margin=best_action_margin.float(),
    )


def joint_svra_forward(
    model: SemanticVisualRiskArbiter,
    full_cls: Tensor,
    patch_tokens: Tensor | None,
    *,
    semantic_off: bool = False,
    visual_off: bool = False,
    interaction_off: bool = False,
) -> JointSVRAOutput:
    """Run the J-SVRA soft training path and hard deployment-equivalent path."""

    action_state = model.action_state(
        full_cls,
        patch_tokens,
        semantic_off=semantic_off,
        visual_off=visual_off,
    )
    pair = action_state.pair
    max_action_logit = torch.amax(action_state.utility_logits, dim=1)
    opportunity_probability = torch.sigmoid(max_action_logit)
    hard_trigger = max_action_logit > 0
    if interaction_off:
        risk_logits = full_cls.new_zeros(full_cls.shape[0])
        risk_probability = torch.sigmoid(risk_logits)
        logits = pair.parent_logits
        swapped = torch.zeros(full_cls.shape[0], dtype=torch.bool, device=full_cls.device)
    else:
        risk_logits = model.interaction(pair.parent_stats)
        logits, risk_probability, swapped = model.interaction.apply_keep_swap(
            pair.parent_logits,
            pair.top2,
            hard_trigger,
            risk_logits,
        )
    soft_hard = torch.equal(opportunity_probability > 0.5, hard_trigger)
    return JointSVRAOutput(
        pair=pair,
        action_state=action_state,
        parent_logits=pair.parent_logits,
        action_logits25=action_state.utility_logits,
        policy_logits26=action_state.policy_logits,
        max_action_logit=max_action_logit,
        opportunity_probability=opportunity_probability,
        hard_trigger=hard_trigger,
        risk_logits=risk_logits,
        risk_probability=risk_probability,
        joint_probability=opportunity_probability * risk_probability,
        logits=logits,
        swapped=swapped,
        soft_hard_trigger_equal=bool(soft_hard),
    )


def joint_svra_loss(
    output: JointSVRAOutput,
    targets: JointSVRATargets,
    *,
    action_pos_weight: float = JSVRA_ACTION_POS_WEIGHT,
    risk_pos_weight: float = JSVRA_RISK_POS_WEIGHT,
    joint_pos_weight: float = JSVRA_ACTION_POS_WEIGHT,
) -> JointSVRALoss:
    """Compute IDEA-200 fixed three-loss objective with receipt-ready means."""

    if output.policy_logits26.shape[0] != targets.action_targets26.shape[0]:
        raise ValueError("policy logits and action targets batch size differ")
    if output.policy_logits26.shape[1] != ACTION_COUNT + 1:
        raise ValueError(
            "policy_logits26 must have shape [B,26], "
            f"got {tuple(output.policy_logits26.shape)}"
        )
    _validate_binary_target_shape("risk_targets", output.risk_logits, targets.risk_targets)
    _validate_binary_target_shape(
        "joint_targets", output.joint_probability, targets.joint_targets
    )

    action_raw = F.cross_entropy(
        output.policy_logits26.float(),
        targets.action_targets26.to(device=output.policy_logits26.device),
        reduction="none",
    )
    action_weight = torch.where(
        targets.action_targets26.to(device=output.policy_logits26.device) > 0,
        output.policy_logits26.new_full((), float(action_pos_weight)),
        output.policy_logits26.new_ones(()),
    )
    action_loss = (action_raw * action_weight).mean()

    risk_target = targets.risk_targets.to(device=output.risk_logits.device).float()
    risk_raw = F.binary_cross_entropy_with_logits(
        output.risk_logits.float(),
        risk_target,
        reduction="none",
    )
    risk_weight = torch.where(
        risk_target > 0,
        output.risk_logits.new_full((), float(risk_pos_weight)),
        output.risk_logits.new_ones(()),
    )
    risk_loss = (risk_raw * risk_weight).mean()

    joint_target = targets.joint_targets.to(device=output.joint_probability.device).float()
    joint_prob = output.joint_probability.float().clamp(1e-6, 1 - 1e-6)
    joint_raw = -(
        joint_target * joint_prob.log()
        + (1 - joint_target) * (1 - joint_prob).log()
    )
    joint_weight = torch.where(
        joint_target > 0,
        output.joint_probability.new_full((), float(joint_pos_weight)),
        output.joint_probability.new_ones(()),
    )
    joint_loss = (joint_raw * joint_weight).mean()

    return JointSVRALoss(
        total=action_loss + risk_loss + joint_loss,
        action=action_loss,
        risk=risk_loss,
        joint=joint_loss,
        raw_means={
            "action": action_raw.mean(),
            "risk": risk_raw.mean(),
            "joint": joint_raw.mean(),
        },
        weighted_means={
            "action": action_loss,
            "risk": risk_loss,
            "joint": joint_loss,
        },
        weights={
            "action_positive": float(action_pos_weight),
            "risk_positive": float(risk_pos_weight),
            "joint_positive": float(joint_pos_weight),
            "action": JSVRA_LOSS_COEFFICIENT,
            "risk": JSVRA_LOSS_COEFFICIENT,
            "joint": JSVRA_LOSS_COEFFICIENT,
        },
        soft_hard_trigger_equal=output.soft_hard_trigger_equal,
    )


def _validate_binary_target_shape(name: str, logits_or_prob: Tensor, target: Tensor) -> None:
    if logits_or_prob.shape != target.shape:
        raise ValueError(
            f"{name} must have shape {tuple(logits_or_prob.shape)}, got {tuple(target.shape)}"
        )


def risk_arbiter_targets(
    pair: PairState,
    target_class_ids: Tensor,
    class_ids: Tensor,
    *,
    trigger: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return swap targets, train mask, and top2 group for ParentRiskArbiter."""

    if target_class_ids.shape != (pair.top2.shape[0],):
        raise ValueError(
            "target_class_ids must have shape [B], "
            f"got {tuple(target_class_ids.shape)}"
        )
    device = pair.top2.device
    local_truth = _global_to_local_indices(
        target_class_ids.to(device=device, dtype=torch.long),
        class_ids.to(device=device, dtype=torch.long),
    )
    leader = pair.top2[:, 0]
    challenger = pair.top2[:, 1]
    group = torch.full_like(local_truth, fill_value=2)
    group = torch.where(local_truth == leader, torch.zeros_like(group), group)
    group = torch.where(local_truth == challenger, torch.ones_like(group), group)
    targets = (group == 1).float()
    mask = group < 2
    if trigger is not None:
        if trigger.shape != mask.shape:
            raise ValueError(f"trigger must have shape [B], got {tuple(trigger.shape)}")
        mask = mask & trigger.to(device=device).bool()
    return targets, mask, group


def _global_to_local_indices(labels: Tensor, class_ids: Tensor) -> Tensor:
    if labels.numel() == 0:
        return labels.long()
    if (labels < 0).any():
        raise ValueError("target_class_ids must be non-negative global class ids")
    max_id = int(torch.max(torch.cat([class_ids, labels])).item())
    lookup = torch.full((max_id + 1,), -1, dtype=torch.long, device=labels.device)
    lookup[class_ids] = torch.arange(class_ids.numel(), device=labels.device)
    return lookup[labels.long()]


def dense_utility_loss(utility_logits: Tensor, targets: Tensor) -> Tensor:
    """Elementwise BCEWithLogits loss for dense utility supervision."""

    if utility_logits.shape != targets.shape:
        raise ValueError(
            "utility_logits and targets must share shape, "
            f"got {tuple(utility_logits.shape)} vs {tuple(targets.shape)}"
        )
    if utility_logits.ndim != 2 or utility_logits.shape[1] != ACTION_COUNT:
        raise ValueError(
            f"utility tensors must have shape [B,25], got {tuple(utility_logits.shape)}"
        )
    return F.binary_cross_entropy_with_logits(utility_logits, targets.float())


def risk_arbiter_loss(risk_logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Masked BCEWithLogits loss for keep/swap supervision."""

    if risk_logits.shape != targets.shape or targets.shape != mask.shape:
        raise ValueError(
            "risk_logits, targets, and mask must share shape, "
            f"got {tuple(risk_logits.shape)}, {tuple(targets.shape)}, {tuple(mask.shape)}"
        )
    mask = mask.bool()
    if not bool(mask.any()):
        return risk_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(risk_logits[mask], targets.float()[mask])


__all__ = [
    "ACTION_COUNT",
    "ACTION_GEOMETRY_SHA256",
    "ACTION_HEAD_INPUT_DIM",
    "ATTENTION_DENOM_EPS",
    "CEILING_RISK_INPUT_DIM",
    "FEATURE_DIM",
    "HIDDEN_DIM",
    "JSVRA_ACTION_POS_WEIGHT",
    "JSVRA_LOSS_COEFFICIENT",
    "JSVRA_RISK_POS_WEIGHT",
    "LAYER_NORM_EPS",
    "MAIN_RISK_INPUT_DIM",
    "PAIR_TEMPERATURE",
    "PATCH_COUNT",
    "PATCH_GRID",
    "ROLE_COUNT",
    "ROLE_ORDER",
    "WINDOW_SIZE",
    "WINDOW_STARTS",
    "ActionPolicyState",
    "EightRoleSemanticQuestions",
    "JointSVRALoss",
    "JointSVRAOutput",
    "JointSVRATargets",
    "PairState",
    "ParentRiskArbiter",
    "ParentRiskCeilingArbiter",
    "RoleToWindowActionPolicy",
    "SVRAOutput",
    "SemanticVisualRiskArbiter",
    "build_ceiling_risk_inputs",
    "ceiling_risk_mask",
    "dense_utility_loss",
    "dense_utility_targets_from_logits",
    "joint_action_targets_from_logits",
    "joint_svra_forward",
    "joint_svra_loss",
    "make_action_positions",
    "pool_action_windows",
    "risk_arbiter_loss",
    "risk_arbiter_targets",
    "stable_top2_by_logit_then_class_id",
]
