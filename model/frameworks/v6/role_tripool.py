"""Role Tri-Pool signed window utility for FRAMEWORK-V6.

Each 6x6 window keeps, for every one of the eight semantic roles, the mean,
strongest challenger evidence and strongest leader evidence over its 36 CLIP
patches.  A shared MLP predicts the three-state executable action outcome
``damage / neutral / correction`` directly; there is no learned role-to-window
attention target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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
ROLE_STAT_COUNT: Final[int] = 3
ROLE_STAT_DIM: Final[int] = ROLE_COUNT * ROLE_STAT_COUNT
PARENT_FEATURE_DIM: Final[int] = 2
ACTION_HEAD_INPUT_DIM: Final[int] = ROLE_STAT_DIM + 8 + PARENT_FEATURE_DIM
TRISTATE_CLASS_COUNT: Final[int] = 3
DAMAGE_CLASS: Final[int] = 0
NEUTRAL_CLASS: Final[int] = 1
CORRECTION_CLASS: Final[int] = 2
PAIR_TEMPERATURE: Final[float] = 0.07
ABSTAIN_THRESHOLD: Final[float] = 0.0
LAYER_NORM_EPS: Final[float] = 1e-5
ACTION_GEOMETRY_SHA256: Final[str] = (
    "4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb"
)


@dataclass(frozen=True)
class PairState:
    """Parent Top-2 state and frozen challenger-minus-leader role evidence."""

    parent_logits: Tensor
    top2: Tensor
    leader_ids: Tensor
    challenger_ids: Tensor
    leader_margin: Tensor
    parent_stats: Tensor
    role_deltas: Tensor


@dataclass(frozen=True)
class UtilityState:
    """Per-window tri-state prediction and its explicit evidence features."""

    pair: PairState
    utility_logits: Tensor
    utility: Tensor
    selected_action: Tensor
    trigger: Tensor
    max_utility: Tensor
    tri_state_probabilities: Tensor
    role_patch_scores: Tensor
    role_statistics: Tensor
    action_head_input: Tensor


@dataclass(frozen=True)
class RWDGOutput:
    """Full S/V/I output after optional selected-crop pair verification."""

    pair: PairState
    utility_state: UtilityState
    logits: Tensor
    parent_logits: Tensor
    crop_margin: Tensor | None
    swapped: Tensor


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
    """Return the fixed `[25, 8]` normalized action-position matrix."""

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
    """Pool 24x24 projected patch tokens into the fixed 25 6x6 windows."""

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


def pool_role_window_statistics(
    patch_tokens: Tensor,
    role_deltas: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return per-patch role scores and `[B,25,8,3]` mean/max/min statistics.

    ``role_deltas`` is challenger role text minus leader role text.  Therefore a
    positive score supports Top2 and a negative score supports Top1.  Max and
    min preserve a small discriminative part that a 36-patch mean can dilute.
    """

    _require_rank("patch_tokens", patch_tokens, 3)
    _require_rank("role_deltas", role_deltas, 3)
    if patch_tokens.shape[1:] != (PATCH_COUNT, FEATURE_DIM):
        raise ValueError(
            f"patch_tokens must have shape [B,{PATCH_COUNT},{FEATURE_DIM}], "
            f"got {tuple(patch_tokens.shape)}"
        )
    if role_deltas.shape != (patch_tokens.shape[0], ROLE_COUNT, FEATURE_DIM):
        raise ValueError(
            "role_deltas must have shape [B,8,768], "
            f"got {tuple(role_deltas.shape)}"
        )

    batch = patch_tokens.shape[0]
    grid = _l2_normalize(patch_tokens).reshape(
        batch, PATCH_GRID, PATCH_GRID, FEATURE_DIM
    )
    scores = torch.einsum("bhwd,brd->brhw", grid, role_deltas.float())
    window_scores: list[Tensor] = []
    statistics: list[Tensor] = []
    for y, x in _action_window_slices():
        current = scores[:, :, y : y + WINDOW_SIZE, x : x + WINDOW_SIZE].reshape(
            batch, ROLE_COUNT, WINDOW_SIZE * WINDOW_SIZE
        )
        window_scores.append(current)
        statistics.append(
            torch.stack(
                [
                    current.mean(dim=2),
                    current.max(dim=2).values,
                    current.min(dim=2).values,
                ],
                dim=2,
            )
        )
    return torch.stack(window_scores, dim=1), torch.stack(statistics, dim=1)


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
        raise ValueError("RWDG requires at least two active classes")

    ascending_id_order = torch.argsort(class_ids, stable=True)
    logits_by_id = logits.index_select(1, ascending_id_order)
    ranked_by_id = torch.argsort(logits_by_id, dim=1, descending=True, stable=True)
    return ascending_id_order.index_select(0, ranked_by_id[:, :2].reshape(-1)).reshape(
        logits.shape[0], 2
    )


def _pair_logits_from_cls(
    cls_features: Tensor,
    name_embeddings: Tensor,
    *,
    temperature: float = PAIR_TEMPERATURE,
) -> Tensor:
    _require_rank("cls_features", cls_features, 2)
    _require_last_dim("cls_features", cls_features, FEATURE_DIM)
    return _l2_normalize(cls_features) @ _l2_normalize(name_embeddings).t() / temperature


class EightRolePairEvidence(nn.Module):
    """S module: expose frozen Top2-minus-Top1 role-text differences."""

    def __init__(
        self,
        role_embeddings: Tensor,
        name_embeddings: Tensor,
        class_ids: Tensor,
    ) -> None:
        super().__init__()
        self._validate_assets(role_embeddings, name_embeddings, class_ids)

        # The axis assets must not enter checkpoints: train100 checkpoints are
        # loaded with eval150 assets, so persistent buffers would shape-conflict.
        self.register_buffer(
            "role_embeddings", _l2_normalize(role_embeddings), persistent=False
        )
        self.register_buffer(
            "name_embeddings", _l2_normalize(name_embeddings), persistent=False
        )
        self.register_buffer("class_ids", class_ids.long().clone(), persistent=False)

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
        parent_logits = _pair_logits_from_cls(full_cls, names)
        top2 = stable_top2_by_logit_then_class_id(parent_logits, class_ids)

        row = torch.arange(full_cls.shape[0], device=full_cls.device)
        leader = top2[:, 0]
        challenger = top2[:, 1]
        leader_ids = class_ids.index_select(0, leader)
        challenger_ids = class_ids.index_select(0, challenger)

        name_delta = (
            names.index_select(0, challenger) - names.index_select(0, leader)
        )
        if semantic_off:
            role_delta = name_delta[:, None, :].expand(-1, ROLE_COUNT, -1)
        else:
            roles = self.role_embeddings.to(device=full_cls.device)
            role_delta = (
                roles.index_select(0, challenger) - roles.index_select(0, leader)
            )

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

        return PairState(
            parent_logits=parent_logits,
            top2=top2,
            leader_ids=leader_ids,
            challenger_ids=challenger_ids,
            leader_margin=leader_logits - challenger_logits,
            parent_stats=parent_stats,
            role_deltas=role_delta.float(),
        )


class RoleTriPoolUtility(nn.Module):
    """Shared window MLP over 24 role statistics and 10 fixed context values."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(
            ACTION_HEAD_INPUT_DIM, eps=LAYER_NORM_EPS, elementwise_affine=True
        )
        self.utility_hidden = nn.Linear(ACTION_HEAD_INPUT_DIM, HIDDEN_DIM, bias=False)
        self.utility_output = nn.Linear(HIDDEN_DIM, TRISTATE_CLASS_COUNT, bias=False)
        self.register_buffer(
            "action_positions", make_action_positions(), persistent=False
        )
        nn.init.zeros_(self.utility_output.weight)

    def _patch_features(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        *,
        visual_off: bool,
    ) -> Tensor:
        if visual_off:
            return _l2_normalize(full_cls)[:, None, :].expand(-1, PATCH_COUNT, -1)
        if patch_tokens is None:
            raise ValueError("patch_tokens is required unless visual_off=True")
        return patch_tokens.to(device=full_cls.device)

    def forward(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        pair: PairState,
        *,
        visual_off: bool = False,
    ) -> UtilityState:
        _require_rank("full_cls", full_cls, 2)
        _require_last_dim("full_cls", full_cls, FEATURE_DIM)
        _require_rank("role_deltas", pair.role_deltas, 3)
        if pair.role_deltas.shape[1:] != (ROLE_COUNT, FEATURE_DIM):
            raise ValueError(
                "pair.role_deltas must have shape [B,8,768], "
                f"got {tuple(pair.role_deltas.shape)}"
            )
        if pair.parent_stats.shape != (full_cls.shape[0], 4):
            raise ValueError(
                "pair.parent_stats must have shape [B, 4], "
                f"got {tuple(pair.parent_stats.shape)}"
            )

        patches = self._patch_features(full_cls, patch_tokens, visual_off=visual_off)
        role_patch_scores, role_statistics = pool_role_window_statistics(
            patches, pair.role_deltas.to(device=full_cls.device)
        )
        positions = self.action_positions.to(
            device=full_cls.device, dtype=role_statistics.dtype
        ).expand(full_cls.shape[0], -1, -1)
        # Only the Parent Top1-Top2 margin and entropy are admitted as global
        # context; mean/std logits from the attention predecessor are removed.
        parent_context = pair.parent_stats[:, :2].to(
            device=full_cls.device, dtype=role_statistics.dtype
        )
        action_head_input = torch.cat(
            [
                role_statistics.reshape(full_cls.shape[0], ACTION_COUNT, ROLE_STAT_DIM),
                positions,
                parent_context[:, None, :].expand(-1, ACTION_COUNT, -1),
            ],
            dim=2,
        )
        if action_head_input.shape[2] != ACTION_HEAD_INPUT_DIM:
            raise RuntimeError(
                "RoleTriPool action head input contract violated: "
                f"expected {ACTION_HEAD_INPUT_DIM}, got {action_head_input.shape[2]}"
            )

        hidden = F.gelu(self.utility_hidden(self.input_norm(action_head_input)))
        utility_logits = self.utility_output(hidden)
        probabilities = torch.softmax(utility_logits, dim=2)
        utility = probabilities[:, :, CORRECTION_CLASS] - probabilities[:, :, DAMAGE_CLASS]
        max_utility, selected_action = utility.max(dim=1)
        trigger = max_utility > ABSTAIN_THRESHOLD

        return UtilityState(
            pair=pair,
            utility_logits=utility_logits,
            utility=utility,
            selected_action=selected_action,
            trigger=trigger,
            max_utility=max_utility,
            tri_state_probabilities=probabilities,
            role_patch_scores=role_patch_scores,
            role_statistics=role_statistics,
            action_head_input=action_head_input,
        )


class SelectedCropPairVerifier(nn.Module):
    """I module: fixed keep/swap verifier for the chosen high-resolution crop."""

    def __init__(self, *, temperature: float = PAIR_TEMPERATURE) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def crop_margin(
        self,
        crop_cls: Tensor,
        name_embeddings: Tensor,
        top2: Tensor,
    ) -> Tensor:
        _require_rank("crop_cls", crop_cls, 2)
        _require_last_dim("crop_cls", crop_cls, FEATURE_DIM)
        logits = _pair_logits_from_cls(
            crop_cls, name_embeddings.to(device=crop_cls.device), temperature=self.temperature
        )
        row = torch.arange(crop_cls.shape[0], device=crop_cls.device)
        return logits[row, top2[:, 0]] - logits[row, top2[:, 1]]

    @staticmethod
    def apply_keep_swap(
        parent_logits: Tensor,
        top2: Tensor,
        crop_margin: Tensor,
        trigger: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _require_rank("parent_logits", parent_logits, 2)
        _require_rank("top2", top2, 2)
        if top2.shape != (parent_logits.shape[0], 2):
            raise ValueError(
                f"top2 must have shape [B, 2], got {tuple(top2.shape)}"
            )
        if crop_margin.shape != (parent_logits.shape[0],):
            raise ValueError(
                "crop_margin must have shape [B], "
                f"got {tuple(crop_margin.shape)}"
            )
        if trigger.shape != (parent_logits.shape[0],):
            raise ValueError(f"trigger must have shape [B], got {tuple(trigger.shape)}")

        logits = parent_logits.clone()
        swap = trigger & (crop_margin < 0)
        if swap.any():
            rows = torch.nonzero(swap, as_tuple=False).flatten()
            leaders = top2.index_select(0, rows)[:, 0]
            challengers = top2.index_select(0, rows)[:, 1]
            leader_values = logits[rows, leaders].clone()
            logits[rows, leaders] = logits[rows, challengers]
            logits[rows, challengers] = leader_values
        return logits, swap


class RoleTriPoolGlimpse(nn.Module):
    """Gate-0 role-tri-pool S/V/I wrapper.

    Asset tensors are non-persistent buffers so a train100 checkpoint contains
    only trainable parameters and can be loaded into an eval150 instance.
    """

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
            self.semantic = EightRolePairEvidence(
                role_embeddings=role_embeddings,
                name_embeddings=name_embeddings,
                class_ids=class_ids,
            )
            self.visual = RoleTriPoolUtility()
        self.interaction = SelectedCropPairVerifier()

    @property
    def name_embeddings(self) -> Tensor:
        return self.semantic.name_embeddings

    @property
    def class_ids(self) -> Tensor:
        return self.semantic.class_ids

    def parent_state(self, full_cls: Tensor, *, semantic_off: bool = False) -> PairState:
        return self.semantic(full_cls, semantic_off=semantic_off)

    def utility_state(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        *,
        semantic_off: bool = False,
        visual_off: bool = False,
    ) -> UtilityState:
        pair = self.parent_state(full_cls, semantic_off=semantic_off)
        return self.visual(full_cls, patch_tokens, pair, visual_off=visual_off)

    def forward(
        self,
        full_cls: Tensor,
        patch_tokens: Tensor | None,
        crop_cls: Tensor | None,
        *,
        semantic_off: bool = False,
        visual_off: bool = False,
        interaction_off: bool = False,
    ) -> RWDGOutput:
        utility_state = self.utility_state(
            full_cls,
            patch_tokens,
            semantic_off=semantic_off,
            visual_off=visual_off,
        )
        pair = utility_state.pair
        if interaction_off:
            batch = full_cls.shape[0]
            return RWDGOutput(
                pair=pair,
                utility_state=utility_state,
                logits=pair.parent_logits,
                parent_logits=pair.parent_logits,
                crop_margin=None,
                swapped=torch.zeros(batch, dtype=torch.bool, device=full_cls.device),
            )
        if crop_cls is None:
            raise ValueError("crop_cls is required unless interaction_off=True")
        margin = self.interaction.crop_margin(
            crop_cls,
            self.name_embeddings.to(device=full_cls.device),
            pair.top2,
        )
        logits, swapped = self.interaction.apply_keep_swap(
            pair.parent_logits,
            pair.top2,
            margin,
            utility_state.trigger,
        )
        return RWDGOutput(
            pair=pair,
            utility_state=utility_state,
            logits=logits,
            parent_logits=pair.parent_logits,
            crop_margin=margin,
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
        """Build `[B,25]` BCE targets from all25 crop CLS.

        Rows whose truth is outside the Parent Top-2 receive an all-zero target
        vector.  The returned group is 0=leader, 1=challenger, 2=outside.
        """

        _require_rank("all_crop_cls", all_crop_cls, 3)
        if all_crop_cls.shape[1:] != (ACTION_COUNT, FEATURE_DIM):
            raise ValueError(
                "all_crop_cls must have shape [B,25,768], "
                f"got {tuple(all_crop_cls.shape)}"
            )
        if target_class_ids.shape != (full_cls.shape[0],):
            raise ValueError(
                "target_class_ids must have shape [B], "
                f"got {tuple(target_class_ids.shape)}"
            )

        pair = self.parent_state(full_cls, semantic_off=semantic_off)
        names = self.name_embeddings.to(device=full_cls.device)
        crop_logits = torch.einsum(
            "bad,cd->bac", _l2_normalize(all_crop_cls.to(device=full_cls.device)), names
        ) / PAIR_TEMPERATURE

        target_class_ids = target_class_ids.to(device=full_cls.device, dtype=torch.long)
        local_truth = self._global_to_local_indices(target_class_ids, full_cls.device)
        row = torch.arange(full_cls.shape[0], device=full_cls.device)
        leader = pair.top2[:, 0]
        challenger = pair.top2[:, 1]

        leader_score = crop_logits[row[:, None], torch.arange(ACTION_COUNT, device=full_cls.device), leader[:, None]]
        challenger_score = crop_logits[
            row[:, None], torch.arange(ACTION_COUNT, device=full_cls.device), challenger[:, None]
        ]
        crop_keeps_leader = leader_score >= challenger_score

        group = torch.full_like(local_truth, fill_value=2)
        group = torch.where(local_truth == leader, torch.zeros_like(group), group)
        group = torch.where(local_truth == challenger, torch.ones_like(group), group)

        targets = torch.zeros(
            full_cls.shape[0],
            ACTION_COUNT,
            dtype=crop_logits.dtype,
            device=full_cls.device,
        )
        leader_rows = group == 0
        challenger_rows = group == 1
        targets[leader_rows] = crop_keeps_leader[leader_rows].float()
        targets[challenger_rows] = (~crop_keeps_leader[challenger_rows]).float()
        return targets, group, pair

    def signed_action_targets(
        self,
        full_cls: Tensor,
        all_crop_cls: Tensor,
        target_class_ids: Tensor,
        *,
        semantic_off: bool = False,
    ) -> tuple[Tensor, Tensor, PairState]:
        """Return `[B,25]` class ids: damage=0, neutral=1, correction=2.

        Labels compare each executable crop action with the Parent decision.
        Truth outside Parent Top2 is always neutral because swapping two wrong
        candidates cannot change correctness.
        """

        correctness, group, pair = self.dense_utility_targets(
            full_cls,
            all_crop_cls,
            target_class_ids,
            semantic_off=semantic_off,
        )
        labels = torch.full_like(correctness, fill_value=NEUTRAL_CLASS, dtype=torch.long)
        leader_rows = group == 0
        challenger_rows = group == 1
        labels[leader_rows[:, None] & ~correctness.bool()] = DAMAGE_CLASS
        labels[challenger_rows[:, None] & correctness.bool()] = CORRECTION_CLASS
        return labels, group, pair

    def _global_to_local_indices(self, labels: Tensor, device: torch.device) -> Tensor:
        if labels.numel() == 0:
            return labels.to(device=device, dtype=torch.long)
        if (labels < 0).any():
            raise ValueError("target_class_ids must be non-negative global class ids")
        class_ids = self.class_ids.to(device=device)
        max_id = int(torch.max(torch.cat([class_ids, labels.to(device=device)])).item())
        lookup = torch.full((max_id + 1,), -1, dtype=torch.long, device=device)
        lookup[class_ids] = torch.arange(class_ids.numel(), device=device)
        return lookup[labels.to(device=device, dtype=torch.long)]


def tri_state_action_loss(utility_logits: Tensor, targets: Tensor) -> Tensor:
    """Natural-distribution cross entropy over all 25 executable actions."""

    if utility_logits.shape != (*targets.shape, TRISTATE_CLASS_COUNT):
        raise ValueError(
            "utility_logits and targets must share shape, "
            f"got {tuple(utility_logits.shape)} vs {tuple(targets.shape)}"
        )
    if utility_logits.ndim != 3 or utility_logits.shape[1:] != (
        ACTION_COUNT,
        TRISTATE_CLASS_COUNT,
    ):
        raise ValueError(
            "utility logits must have shape [B,25,3], "
            f"got {tuple(utility_logits.shape)}"
        )
    if bool((targets < 0).any()) or bool((targets >= TRISTATE_CLASS_COUNT).any()):
        raise ValueError("tri-state targets must be class ids in [0,2]")
    return F.cross_entropy(
        utility_logits.reshape(-1, TRISTATE_CLASS_COUNT),
        targets.long().reshape(-1),
    )


# Compatibility alias for shared Gate-0 data and evaluator helpers.  The
# implementation itself is the new RoleTriPool path, not the RWDG attention path.
RoleWindowDenseGlimpse = RoleTriPoolGlimpse


__all__ = [
    "ABSTAIN_THRESHOLD",
    "ACTION_COUNT",
    "ACTION_GEOMETRY_SHA256",
    "ACTION_HEAD_INPUT_DIM",
    "FEATURE_DIM",
    "HIDDEN_DIM",
    "LAYER_NORM_EPS",
    "PAIR_TEMPERATURE",
    "PATCH_COUNT",
    "PATCH_GRID",
    "ROLE_COUNT",
    "WINDOW_SIZE",
    "WINDOW_STARTS",
    "CORRECTION_CLASS",
    "DAMAGE_CLASS",
    "NEUTRAL_CLASS",
    "ROLE_STAT_COUNT",
    "ROLE_STAT_DIM",
    "TRISTATE_CLASS_COUNT",
    "EightRolePairEvidence",
    "PairState",
    "RoleTriPoolUtility",
    "RoleTriPoolGlimpse",
    "RoleWindowDenseGlimpse",
    "RWDGOutput",
    "SelectedCropPairVerifier",
    "UtilityState",
    "tri_state_action_loss",
    "make_action_positions",
    "pool_action_windows",
    "pool_role_window_statistics",
    "stable_top2_by_logit_then_class_id",
]
