"""Gate-0 two-stage trainer for IDEA-199 / SVRA.

The trainer keeps the contract deliberately narrow:

* Stage 1 trains the S/V policy with the EAAC 26-way target
  (abstain plus the strongest corrective all25 action).
* Stage 2 freezes S/V and trains three trigger arbiters: the deployed 4D
  arbiter on Stage-1 triggered rows, a 13D ceiling on the same rows and batch
  trace, and an all-row 4D control.
* The formal path reads only dev_train targets/all25 crop features for gradient
  construction. It never loads evaluation rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_torch_save, atomic_write_json


TRAIN_SCHEMA = "gzsl-paper.v6-svra-gate0-train.v1"
FEATURE_DIM = 768
ACTION_COUNT = 25
EAAC_CLASS_COUNT = ACTION_COUNT + 1
PAIR_TEMPERATURE = 0.07
ACTION_GEOMETRY_SHA256 = (
    "4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb"
)

SVRA_PREREGISTERED_STAGE1_TARGET_COUNTS = {
    "total_rows": 4702,
    "abstain_rows": 4107,
    "action_rows": 595,
}

SVRA_PREREGISTERED_TRIGGER_COUNTS = {
    "triggered_rows": 574,
    "positive_rows": 300,
    "negative_rows": 274,
}

STRICT_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "condition_id",
        "text_manifest",
        "text_manifest_sha256",
        "role_tensor",
        "role_tensor_sha256",
        "name_tensor",
        "name_tensor_sha256",
        "patch_manifest",
        "patch_manifest_sha256",
        "cls_tensor",
        "cls_tensor_sha256",
        "patch_tensor",
        "patch_tensor_sha256",
        "action_bundle_manifest",
        "action_bundle_manifest_sha256",
        "dev_train_manifest_sha256",
        "dev_eval_manifest_sha256",
        "dev_eval_oracle_manifest_sha256",
        "att_splits_mat_path",
        "trainval_count",
        "oracle_receipt",
        "oracle_receipt_sha256",
        "action_geometry_sha256",
        "output_dir",
        "device",
        "seed",
        "stage1_batch_size",
        "stage1_updates",
        "stage1_lr",
        "stage1_weight_decay",
        "stage1_challenger_per_batch",
        "stage2_batch_size",
        "stage2_updates",
        "stage2_lr",
        "stage2_weight_decay",
        "stage2_positive_per_batch",
        "stage2_threshold",
        "expected_stage1_abstain_rows",
        "expected_stage1_action_rows",
        "expected_stage2_triggered_rows",
        "expected_stage2_positive_rows",
        "expected_stage2_negative_rows",
        "strict_sha",
        "validate_tensor_values",
        "require_clean_tree",
        "allow_cpu",
        "official_test_loaded",
        "unseen_images_used_for_gradient",
        "pclr_online_inference",
    }
)


@dataclass(frozen=True)
class Gate0TrainConfig:
    schema_version: str
    experiment_id: str
    condition_id: str
    text_manifest: str
    text_manifest_sha256: str
    role_tensor: str
    role_tensor_sha256: str
    name_tensor: str
    name_tensor_sha256: str
    patch_manifest: str
    patch_manifest_sha256: str
    cls_tensor: str
    cls_tensor_sha256: str
    patch_tensor: str
    patch_tensor_sha256: str
    action_bundle_manifest: str
    action_bundle_manifest_sha256: str
    dev_train_manifest_sha256: str
    dev_eval_manifest_sha256: str
    dev_eval_oracle_manifest_sha256: str
    att_splits_mat_path: str
    trainval_count: int
    oracle_receipt: str
    oracle_receipt_sha256: str
    action_geometry_sha256: str
    output_dir: str
    device: str = "cuda:0"
    seed: int = 7
    stage1_batch_size: int = 8
    stage1_updates: int = 1000
    stage1_lr: float = 1e-3
    stage1_weight_decay: float = 1e-4
    stage1_challenger_per_batch: int = 4
    stage2_batch_size: int = 32
    stage2_updates: int = 1000
    stage2_lr: float = 1e-3
    stage2_weight_decay: float = 1e-4
    stage2_positive_per_batch: int = 16
    stage2_threshold: float = 0.5
    expected_stage1_abstain_rows: int = 4107
    expected_stage1_action_rows: int = 595
    expected_stage2_triggered_rows: int = 574
    expected_stage2_positive_rows: int = 300
    expected_stage2_negative_rows: int = 274
    strict_sha: bool = True
    validate_tensor_values: bool = True
    require_clean_tree: bool = True
    allow_cpu: bool = False
    official_test_loaded: bool = False
    unseen_images_used_for_gradient: bool = False
    pclr_online_inference: bool = False


@dataclass(frozen=True)
class TrainSubsetTargets:
    labels: Tensor
    class_ids: Tensor
    crop_features: Any
    labels_path: str
    class_ids_path: str
    crop_features_path: str
    labels_sha256: str
    class_ids_sha256: str
    crop_features_sha256: str


@dataclass(frozen=True)
class EAACTargetPlan:
    targets26: Tensor
    groups: Tensor
    margins: Tensor
    dense_targets: Tensor
    stats: Mapping[str, Any]


@dataclass(frozen=True)
class TriggerPlan:
    triggered_rows: Tensor
    labels: Tensor
    selected_actions: Tensor
    features4: Tensor
    features13: Tensor
    allrow_rows: Tensor
    allrow_labels: Tensor
    allrow_features4: Tensor
    stats: Mapping[str, Any]


@dataclass(frozen=True)
class GradientGateReport:
    finite: bool
    nonzero: bool
    grad_abs_sum: float
    grad_max_abs: float


class LinearTriggerArbiter(nn.Module):
    """Core-compatible 2-layer BCE arbiter used for Stage 2 trigger approval."""

    def __init__(self, input_dim: int, *, seed: int = 7) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(input_dim, 32)
            self.output = nn.Linear(32, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.hidden.in_features:
            raise ValueError(
                f"features must have shape [B,{self.hidden.in_features}], "
                f"got {tuple(features.shape)}"
            )
        return self.output(F.gelu(self.hidden(features.float()))).squeeze(1)


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_strict_config(path: str | os.PathLike[str]) -> tuple[Gate0TrainConfig, str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"config must be a JSON object: {p}")
    got = {str(k) for k in raw.keys()}
    missing = STRICT_CONFIG_FIELDS - got
    extra = got - STRICT_CONFIG_FIELDS
    if missing or extra:
        raise RuntimeError(
            f"config fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return Gate0TrainConfig(**raw), sha256_file(p)


def validate_config(config: Gate0TrainConfig) -> None:
    if config.schema_version != TRAIN_SCHEMA:
        raise RuntimeError(f"schema_version must be {TRAIN_SCHEMA}")
    if config.condition_id != "SVRA_FULL":
        raise RuntimeError("condition_id must be SVRA_FULL")
    if not str(config.experiment_id):
        raise RuntimeError("experiment_id must be non-empty")
    if config.seed != 7:
        raise RuntimeError("SVRA Gate0 seed is fixed at 7")
    if config.stage1_batch_size != 8 or config.stage1_challenger_per_batch != 4:
        raise RuntimeError("Stage1 requires fixed 4:4 sampling with batch8")
    if config.stage1_updates != 1000:
        raise RuntimeError("Stage1 updates is fixed at 1000")
    if float(config.stage1_lr) != 1e-3 or float(config.stage1_weight_decay) != 1e-4:
        raise RuntimeError("Stage1 AdamW is fixed at lr=1e-3 wd=1e-4")
    if config.stage2_batch_size != 32 or config.stage2_positive_per_batch != 16:
        raise RuntimeError("Stage2 requires fixed 16+16 sampling with batch32")
    if config.stage2_updates != 1000:
        raise RuntimeError("Stage2 updates is fixed at 1000")
    if float(config.stage2_threshold) != 0.5:
        raise RuntimeError("Stage2 threshold is fixed at 0.5")
    if int(config.trainval_count) != 7057:
        raise RuntimeError("trainval_count must be 7057")
    if config.action_geometry_sha256 != ACTION_GEOMETRY_SHA256:
        raise RuntimeError("action_geometry_sha256 mismatch")
    if not config.strict_sha or not config.validate_tensor_values:
        raise RuntimeError("formal SVRA training requires strict asset validation")
    if config.allow_cpu:
        raise RuntimeError("formal SVRA training requires allow_cpu=false")
    if config.official_test_loaded is not False:
        raise RuntimeError("official_test_loaded must be false for training")
    if config.unseen_images_used_for_gradient is not False:
        raise RuntimeError("unseen_images_used_for_gradient must be false")
    if config.pclr_online_inference is not False:
        raise RuntimeError("pclr_online_inference must be false")


def set_reproducibility(seed: int) -> torch.Generator:
    configure_reproducibility(
        seed,
        strict_determinism=True,
        deterministic_warn_only=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def resolve_device(config: Gate0TrainConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device, but torch.cuda.is_available() is false")
    if device.type == "cpu" and not config.allow_cpu:
        raise RuntimeError("CPU training is disabled unless allow_cpu=true")
    return device


def git_commit_and_clean(repo_root: Path) -> tuple[str, bool, str]:
    commit = _git(repo_root, "rev-parse", "HEAD").strip()
    status = _git(repo_root, "status", "--porcelain").strip()
    return commit, status == "", status


def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def prepare_output_dir(output_dir: str | os.PathLike[str], repo_root: Path) -> Path:
    out = Path(output_dir).expanduser().resolve()
    repo = repo_root.resolve()
    try:
        out.relative_to(repo)
        inside_repo = True
    except ValueError:
        inside_repo = False
    if inside_repo:
        raise RuntimeError(f"output_dir must be outside the repository: {out}")
    if out.exists():
        raise RuntimeError(f"output_dir already exists; refusing overwrite: {out}")
    out.mkdir(parents=True)
    return out


def _l2_normalize(value: Tensor) -> Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-12)


def crop_leader_minus_challenger_margins(
    model: Any,
    all_crop_cls: Tensor,
    pair: Any,
) -> Tensor:
    if all_crop_cls.ndim != 3 or all_crop_cls.shape[1:] != (ACTION_COUNT, FEATURE_DIM):
        raise ValueError(
            "all_crop_cls must have shape [B,25,768], "
            f"got {tuple(all_crop_cls.shape)}"
        )
    names = model.name_embeddings.to(device=all_crop_cls.device)
    crop_logits = torch.einsum("bad,cd->bac", _l2_normalize(all_crop_cls), names)
    crop_logits = crop_logits / PAIR_TEMPERATURE
    rows = torch.arange(all_crop_cls.shape[0], device=all_crop_cls.device)
    actions = torch.arange(ACTION_COUNT, device=all_crop_cls.device)
    leader = pair.top2[:, 0]
    challenger = pair.top2[:, 1]
    leader_score = crop_logits[rows[:, None], actions, leader[:, None]]
    challenger_score = crop_logits[rows[:, None], actions, challenger[:, None]]
    return leader_score - challenger_score


def eaac_action_targets(
    model: Any,
    full_cls: Tensor,
    all_crop_cls: Tensor,
    target_class_ids: Tensor,
    *,
    semantic_off: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Any, Tensor]:
    dense_targets, groups, pair = model.dense_utility_targets(
        full_cls,
        all_crop_cls,
        target_class_ids,
        semantic_off=semantic_off,
    )
    margins = crop_leader_minus_challenger_margins(model, all_crop_cls, pair)
    targets26 = torch.zeros(full_cls.shape[0], dtype=torch.long, device=full_cls.device)
    correctable = groups.eq(1) & dense_targets.bool().any(dim=1)
    if bool(correctable.any()):
        masked = margins[correctable].masked_fill(
            ~dense_targets[correctable].bool(), float("inf")
        )
        targets26[correctable] = torch.argmin(masked, dim=1).long() + 1
    return targets26, groups, margins, pair, dense_targets


def eaac_policy_logits(utility_logits: Tensor) -> Tensor:
    if utility_logits.ndim != 2 or utility_logits.shape[1] != ACTION_COUNT:
        raise ValueError(f"EAAC utility logits must have shape [B,{ACTION_COUNT}]")
    return torch.cat([utility_logits.new_zeros((utility_logits.shape[0], 1)), utility_logits], dim=1)


def eaac_action_loss(utility_logits: Tensor, targets26: Tensor) -> Tensor:
    if targets26.shape != (utility_logits.shape[0],):
        raise ValueError("EAAC logits/target shape mismatch")
    targets26 = targets26.long()
    if bool((targets26 < 0).any()) or bool((targets26 >= EAAC_CLASS_COUNT).any()):
        raise ValueError(f"EAAC targets must be class ids in [0,{EAAC_CLASS_COUNT - 1}]")
    return F.cross_entropy(eaac_policy_logits(utility_logits), targets26)


eaac_policy_loss = eaac_action_loss
eaac_targets = eaac_action_targets


def _int_tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _float_tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _class_histogram(targets26: Tensor) -> list[int]:
    hist = torch.bincount(targets26.detach().cpu().long(), minlength=EAAC_CLASS_COUNT)
    return [int(x) for x in hist[:EAAC_CLASS_COUNT].tolist()]


def precompute_eaac_target_plan(
    model: Any,
    train_view: Any,
    targets: TrainSubsetTargets,
    *,
    batch_size: int,
    device: torch.device,
) -> EAACTargetPlan:
    model.eval()
    all_targets: list[Tensor] = []
    all_groups: list[Tensor] = []
    all_margins: list[Tensor] = []
    all_dense: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, int(train_view.size), int(batch_size)):
            rows = torch.arange(start, min(start + batch_size, int(train_view.size)), dtype=torch.long)
            full_cls, _, crops, labels = batch_from_rows(train_view, targets, rows, device=device)
            target26, groups, margins, _, dense_targets = eaac_action_targets(
                model, full_cls, crops, labels
            )
            all_targets.append(target26.detach().cpu())
            all_groups.append(groups.detach().cpu())
            all_margins.append(margins.detach().cpu())
            all_dense.append(dense_targets.detach().cpu())

    targets26 = torch.cat(all_targets).long()
    groups = torch.cat(all_groups).long()
    margins = torch.cat(all_margins).float()
    dense_targets = torch.cat(all_dense).bool()
    hist = _class_histogram(targets26)
    group_counts = {
        "leader": int(groups.eq(0).sum().item()),
        "challenger": int(groups.eq(1).sum().item()),
        "outside": int(groups.eq(2).sum().item()),
    }
    stats = {
        "target_policy": "svra_reuses_eaac_strongest_corrective_action_or_abstain",
        "row_count": int(targets26.numel()),
        "class_histogram": hist,
        "abstain_rows": int(targets26.eq(0).sum().item()),
        "action_rows": int(targets26.gt(0).sum().item()),
        "group_counts": group_counts,
        "target_sha256": _int_tensor_sha256(targets26),
        "group_sha256": _int_tensor_sha256(groups),
        "dense_target_sha256": _int_tensor_sha256(dense_targets.long()),
        "margin_sha256": _float_tensor_sha256(margins),
    }
    return EAACTargetPlan(
        targets26=targets26,
        groups=groups,
        margins=margins,
        dense_targets=dense_targets,
        stats=stats,
    )


def validate_stage1_target_contract(plan: EAACTargetPlan, config: Gate0TrainConfig) -> None:
    if int(plan.stats["row_count"]) != SVRA_PREREGISTERED_STAGE1_TARGET_COUNTS["total_rows"]:
        raise RuntimeError(f"Stage1 row count mismatch: {plan.stats['row_count']}")
    if int(plan.stats["abstain_rows"]) != int(config.expected_stage1_abstain_rows):
        raise RuntimeError("Stage1 abstain_rows mismatch")
    if int(plan.stats["action_rows"]) != int(config.expected_stage1_action_rows):
        raise RuntimeError("Stage1 action_rows mismatch")


class BalancedGroupSampler:
    """Seeded positive/negative sampler with independent reshuffle cycles."""

    def __init__(
        self,
        positive_mask: Tensor,
        *,
        batch_size: int,
        positive_per_batch: int,
        seed: int = 7,
        name: str = "BalancedGroupSampler",
    ) -> None:
        mask = positive_mask.detach().cpu().bool().reshape(-1)
        positives = torch.nonzero(mask, as_tuple=False).flatten().long()
        negatives = torch.nonzero(~mask, as_tuple=False).flatten().long()
        if positives.numel() < positive_per_batch:
            raise RuntimeError(f"{name} has too few positive rows")
        if negatives.numel() < batch_size - positive_per_batch:
            raise RuntimeError(f"{name} has too few negative rows")
        self.name = str(name)
        self.batch_size = int(batch_size)
        self.positive_per_batch = int(positive_per_batch)
        self.negative_per_batch = self.batch_size - self.positive_per_batch
        self.seed = int(seed)
        self.positive_indices = positives
        self.negative_indices = negatives
        self._pos_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self._neg_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self._batch_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self._pos_buffer = torch.empty(0, dtype=torch.long)
        self._neg_buffer = torch.empty(0, dtype=torch.long)

    def sample(self) -> Tensor:
        pos = self._take(True, self.positive_per_batch)
        neg = self._take(False, self.negative_per_batch)
        batch = torch.cat([pos, neg])
        order = torch.randperm(batch.numel(), generator=self._batch_gen)
        return batch.index_select(0, order).long()

    def next_batch(self) -> Tensor:
        return self.sample()

    def _take(self, positive: bool, count: int) -> Tensor:
        pool = self.positive_indices if positive else self.negative_indices
        gen = self._pos_gen if positive else self._neg_gen
        buffer = self._pos_buffer if positive else self._neg_buffer
        chunks: list[Tensor] = []
        remaining = int(count)
        while remaining > 0:
            if buffer.numel() == 0:
                buffer = pool.index_select(0, torch.randperm(pool.numel(), generator=gen))
            take = min(remaining, int(buffer.numel()))
            chunks.append(buffer[:take])
            buffer = buffer[take:]
            remaining -= take
        if positive:
            self._pos_buffer = buffer
        else:
            self._neg_buffer = buffer
        return torch.cat(chunks).long()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "positive_per_batch": self.positive_per_batch,
            "negative_per_batch": self.negative_per_batch,
            "positive_rows": int(self.positive_indices.numel()),
            "negative_rows": int(self.negative_indices.numel()),
            "reshuffle_policy": "independent positive/negative randperm cycles; shuffle within batch",
        }


def stage1_sampler_from_groups(
    groups: Tensor,
    *,
    batch_size: int = 8,
    challenger_per_batch: int = 4,
    seed: int = 7,
) -> BalancedGroupSampler:
    if batch_size != 8 or challenger_per_batch != 4:
        raise RuntimeError("Stage1 sampler requires fixed batch8, 4 challenger rows")
    return BalancedGroupSampler(
        groups.detach().cpu().long().eq(1),
        batch_size=batch_size,
        positive_per_batch=challenger_per_batch,
        seed=seed,
        name="Stage1Fixed4to4ChallengerSampler",
    )


def sampled_class_histogram(
    targets26: Tensor,
    groups: Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    challenger_per_batch: int = 4,
) -> Mapping[str, Any]:
    sampler = stage1_sampler_from_groups(
        groups,
        batch_size=batch_size,
        challenger_per_batch=challenger_per_batch,
        seed=seed,
    )
    hist = torch.zeros(EAAC_CLASS_COUNT, dtype=torch.long)
    for _ in range(int(updates)):
        rows = sampler.sample()
        hist += torch.bincount(
            targets26.index_select(0, rows).cpu().long(),
            minlength=EAAC_CLASS_COUNT,
        )[:EAAC_CLASS_COUNT].long()
    total = max(int(updates) * int(batch_size), 1)
    return {
        "updates": int(updates),
        "batch_size": int(batch_size),
        "sampled_rows": total,
        "class_histogram": [int(x) for x in hist.tolist()],
        "abstain_rows": int(hist[0].item()),
        "action_rows": int(hist[1:].sum().item()),
        "class_density": [float(x / total) for x in hist.tolist()],
        "sampler": sampler.state_dict(),
    }


def batch_from_rows(
    train_view: Any,
    targets: TrainSubsetTargets,
    rows: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    row_np = rows.detach().cpu().numpy().astype(np.int64, copy=False)
    batch = train_view.batch(row_np, include_patches=True, as_torch=True, device=device)
    labels = targets.labels.index_select(0, rows.cpu()).to(device=device)
    crops = _take_crop_rows(targets.crop_features, row_np).to(device=device, dtype=torch.float32)
    return batch["cls"], batch["patches"], crops, labels


def _take_crop_rows(crop_features: Any, rows: np.ndarray) -> Tensor:
    if torch.is_tensor(crop_features):
        return crop_features.index_select(0, torch.as_tensor(rows, dtype=torch.long)).float()
    return torch.as_tensor(np.asarray(crop_features[rows]), dtype=torch.float32)


def _utility_logits_from_state(state: Any) -> Tensor:
    logits = getattr(state, "utility_logits", state)
    if logits.ndim == 3 and logits.shape[2] == 1:
        logits = logits.squeeze(2)
    if logits.ndim != 2 or logits.shape[1] != ACTION_COUNT:
        raise ValueError(f"Stage1 utility logits must have shape [B,{ACTION_COUNT}]")
    return logits


def train_stage1_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    train_view: Any,
    targets: TrainSubsetTargets,
    rows: Tensor,
    target_plan: EAACTargetPlan,
    *,
    device: torch.device,
    apply_update: bool = True,
) -> tuple[float, Mapping[str, GradientGateReport]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    full_cls, patches, _, _ = batch_from_rows(train_view, targets, rows, device=device)
    utility = _policy_state_from_model(model, full_cls, patches)
    logits = _utility_logits_from_state(utility)
    target26 = target_plan.targets26.index_select(0, rows.cpu()).to(device=device)
    loss = eaac_action_loss(logits, target26)
    loss.backward()
    report = collect_gradient_report(model)
    if apply_update:
        optimizer.step()
    return float(loss.detach().cpu()), report


def _policy_state_from_model(
    model: Any,
    full_cls: Tensor,
    patches: Tensor | None,
    *,
    semantic_off: bool = False,
    visual_off: bool = False,
) -> Any:
    if not hasattr(model, "policy_state"):
        raise RuntimeError("SVRA model must expose policy_state")
    return model.policy_state(
        full_cls,
        patches,
        semantic_off=semantic_off,
        visual_off=visual_off,
    )


def _state_field(state: Any, name: str) -> Tensor:
    if hasattr(state, name):
        return getattr(state, name)
    pair = getattr(state, "pair", None)
    if pair is not None and hasattr(pair, name):
        return getattr(pair, name)
    raise RuntimeError(f"SVRA policy_state missing field: {name}")


def collect_gradient_report(module: nn.Module) -> Mapping[str, GradientGateReport]:
    out: dict[str, GradientGateReport] = {}
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            out[name] = gradient_gate_report(parameter.grad)
    if not out:
        raise RuntimeError("no trainable parameters found for gradient report")
    return out


def gradient_gate_report(grad: Tensor | None) -> GradientGateReport:
    if grad is None:
        return GradientGateReport(False, False, 0.0, 0.0)
    finite = bool(torch.isfinite(grad).all().detach().cpu())
    abs_grad = grad.detach().abs()
    abs_sum = float(abs_grad.sum().cpu())
    max_abs = float(abs_grad.max().cpu()) if abs_grad.numel() else 0.0
    return GradientGateReport(finite, abs_sum > 0.0, abs_sum, max_abs)


def assert_nonzero_gradients(report: Mapping[str, GradientGateReport], *, label: str) -> None:
    if not any(gate.finite and gate.nonzero for gate in report.values()):
        raise RuntimeError(f"{label} gradient gate failed: all trainable gradients are zero")
    bad = {name: gate for name, gate in report.items() if not gate.finite}
    if bad:
        raise RuntimeError(f"{label} gradient gate failed: non-finite gradients {bad}")


def assert_stage1_second_step_contract(
    report: Mapping[str, GradientGateReport],
) -> None:
    semantic = [
        name
        for name, gate in report.items()
        if name.startswith("semantic.") and gate.finite and gate.nonzero
    ]
    visual_upstream = [
        name
        for name, gate in report.items()
        if name.startswith("visual.")
        and not name.startswith("visual.utility_output.")
        and gate.finite
        and gate.nonzero
    ]
    if not semantic:
        raise RuntimeError("Stage1 step2 gradient gate failed: no semantic gradient")
    if not visual_upstream:
        raise RuntimeError(
            "Stage1 step2 gradient gate failed: no visual upstream gradient"
        )


def freeze_stage1_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def configure_stage1_trainable(model: nn.Module) -> list[nn.Parameter]:
    """Train only semantic/visual policy parameters when the core exposes them."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[nn.Parameter] = []
    for module_name in ("semantic", "visual"):
        module = getattr(model, module_name, None)
        if isinstance(module, nn.Module):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                selected.append(parameter)
    if not selected:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
            selected.append(parameter)
    if not selected:
        raise RuntimeError("SVRA Stage1 found no semantic/visual trainable parameters")
    return selected


def build_trigger_plan(
    model: Any,
    train_view: Any,
    targets: TrainSubsetTargets,
    target_plan: EAACTargetPlan,
    *,
    batch_size: int,
    threshold: float,
    device: torch.device,
) -> TriggerPlan:
    model.eval()
    logits_all: list[Tensor] = []
    parent_stats_all: list[Tensor] = []
    selected_actions_all: list[Tensor] = []
    selected_confidence_all: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, int(train_view.size), int(batch_size)):
            rows = torch.arange(start, min(start + batch_size, int(train_view.size)), dtype=torch.long)
            full_cls, patches, _, _ = batch_from_rows(train_view, targets, rows, device=device)
            state = _policy_state_from_model(model, full_cls, patches)
            logits_all.append(_utility_logits_from_state(state).detach().cpu())
            parent_stats_all.append(_state_field(state, "parent_stats").detach().cpu().float())
            selected_actions_all.append(_state_field(state, "selected_action").detach().cpu().long())
            selected_confidence_all.append(
                _state_field(state, "selected_policy_confidence").detach().cpu().float()
            )
    utility_logits = torch.cat(logits_all).float()
    parent_stats = torch.cat(parent_stats_all).float()
    state_selected_actions = torch.cat(selected_actions_all).long()
    state_selected_confidence = torch.cat(selected_confidence_all).float()
    policy_logits = eaac_policy_logits(utility_logits)
    probabilities = torch.softmax(policy_logits, dim=1)
    confidence, selected26 = probabilities.max(dim=1)
    selected_actions_all_tensor = (selected26 - 1).clamp(min=0, max=ACTION_COUNT - 1).long()
    if not torch.equal(state_selected_actions, selected_actions_all_tensor):
        raise RuntimeError("SVRA policy_state selected_action disagrees with policy logits")
    if not torch.allclose(state_selected_confidence, confidence.float(), atol=1e-6, rtol=1e-5):
        raise RuntimeError("SVRA policy_state selected_policy_confidence disagrees with policy logits")
    triggered_mask = selected26.gt(0)
    row_ids = torch.nonzero(triggered_mask, as_tuple=False).flatten().long()
    selected_actions = selected_actions_all_tensor.index_select(0, row_ids)
    labels = target_plan.groups.index_select(0, row_ids).eq(1).float()

    features4_all, features13_all = build_trigger_features(
        parent_stats=parent_stats,
        selected_policy_confidence=confidence.float(),
        selected_action=selected_actions_all_tensor,
    )
    features4 = features4_all.index_select(0, row_ids)
    features13 = features13_all.index_select(0, row_ids)

    allrow_labels = target_plan.groups.eq(1).float()
    stats = {
        "trigger_rule": "argmax([zero_abstain_logit, action_logits]) != abstain",
        "stage2_threshold": float(threshold),
        "triggered_rows": int(row_ids.numel()),
        "positive_rows": int(labels.eq(1).sum().item()),
        "negative_rows": int(labels.eq(0).sum().item()),
        "selected_action_histogram": [
            int(x)
            for x in torch.bincount(selected_actions.cpu(), minlength=ACTION_COUNT)[:ACTION_COUNT].tolist()
        ],
        "trigger_rows_sha256": _int_tensor_sha256(row_ids),
        "trigger_label_sha256": _int_tensor_sha256(labels.long()),
        "selected_action_sha256": _int_tensor_sha256(selected_actions),
        "features4_sha256": _float_tensor_sha256(features4),
        "features13_sha256": _float_tensor_sha256(features13),
        "allrow_label_sha256": _int_tensor_sha256(allrow_labels.long()),
        "allrow_features4_sha256": _float_tensor_sha256(features4_all),
        "probability_summary": probability_summary(probabilities, selected26),
        "group_stats": group_trigger_stats(target_plan.groups, triggered_mask, labels, row_ids),
    }
    return TriggerPlan(
        triggered_rows=row_ids,
        labels=labels,
        selected_actions=selected_actions,
        features4=features4,
        features13=features13,
        allrow_rows=torch.arange(target_plan.targets26.numel(), dtype=torch.long),
        allrow_labels=allrow_labels,
        allrow_features4=features4_all,
        stats=stats,
    )


def build_trigger_features(
    *,
    parent_stats: Tensor,
    selected_policy_confidence: Tensor,
    selected_action: Tensor,
) -> tuple[Tensor, Tensor]:
    from model.frameworks.v6.svra import (
        CEILING_RISK_INPUT_DIM,
        MAIN_RISK_INPUT_DIM,
        build_ceiling_risk_inputs,
    )

    if parent_stats.ndim != 2 or parent_stats.shape[1] != MAIN_RISK_INPUT_DIM:
        raise ValueError(f"parent_stats must have shape [B,4], got {tuple(parent_stats.shape)}")
    if selected_policy_confidence.shape != (parent_stats.shape[0],):
        raise ValueError("selected_policy_confidence must have shape [B]")
    if selected_action.shape != (parent_stats.shape[0],):
        raise ValueError("selected_action must have shape [B]")

    features4 = parent_stats.float()
    features13 = build_ceiling_risk_inputs(
        features4,
        selected_policy_confidence.float(),
        selected_action.long(),
    ).float()
    if features13.shape[1] != 13:
        raise RuntimeError(f"Stage2 ceiling feature contract expected 13D, got {features13.shape[1]}")
    if features13.shape[1] != CEILING_RISK_INPUT_DIM:
        raise RuntimeError(
            "Stage2 ceiling feature contract disagrees with SVRA core: "
            f"{features13.shape[1]} vs {CEILING_RISK_INPUT_DIM}"
        )
    return features4, features13


def validate_trigger_contract(plan: TriggerPlan, config: Gate0TrainConfig) -> None:
    if int(plan.stats["triggered_rows"]) != int(config.expected_stage2_triggered_rows):
        raise RuntimeError("Stage2 triggered_rows mismatch")
    if int(plan.stats["positive_rows"]) != int(config.expected_stage2_positive_rows):
        raise RuntimeError("Stage2 positive_rows mismatch")
    if int(plan.stats["negative_rows"]) != int(config.expected_stage2_negative_rows):
        raise RuntimeError("Stage2 negative_rows mismatch")


def probability_summary(probabilities: Tensor, selected26: Tensor) -> Mapping[str, Any]:
    selected = probabilities[torch.arange(probabilities.shape[0]), selected26]
    abstain = probabilities[:, 0]
    return {
        "selected_mean": float(selected.mean().item()),
        "selected_min": float(selected.min().item()),
        "selected_max": float(selected.max().item()),
        "abstain_mean": float(abstain.mean().item()),
        "abstain_min": float(abstain.min().item()),
        "abstain_max": float(abstain.max().item()),
    }


def group_trigger_stats(
    groups: Tensor,
    triggered_mask: Tensor,
    trigger_labels: Tensor,
    triggered_rows: Tensor,
) -> Mapping[str, Any]:
    out: dict[str, Any] = {}
    trigger_groups = groups.index_select(0, triggered_rows).long()
    for group_id, name in enumerate(("leader", "challenger", "outside")):
        base_mask = groups.eq(group_id)
        trig_mask = trigger_groups.eq(group_id)
        labels = trigger_labels[trig_mask]
        out[name] = {
            "rows": int(base_mask.sum().item()),
            "triggered": int(triggered_mask[base_mask].sum().item()),
            "positive": int(labels.eq(1).sum().item()),
            "negative": int(labels.eq(0).sum().item()),
        }
    return out


def make_balanced_batch_trace(
    labels: Tensor,
    *,
    updates: int,
    batch_size: int,
    positive_per_batch: int,
    seed: int,
    name: str,
) -> tuple[list[Tensor], Mapping[str, Any]]:
    sampler = BalancedGroupSampler(
        labels.detach().cpu().bool(),
        batch_size=batch_size,
        positive_per_batch=positive_per_batch,
        seed=seed,
        name=name,
    )
    trace = [sampler.sample() for _ in range(int(updates))]
    flat = torch.cat(trace).long() if trace else torch.empty(0, dtype=torch.long)
    receipt = {
        **dict(sampler.state_dict()),
        "updates": int(updates),
        "sampled_rows": int(flat.numel()),
        "batch_trace_sha256": _int_tensor_sha256(flat),
    }
    return trace, receipt


def train_arbiter_with_trace(
    arbiter: nn.Module,
    features: Tensor,
    labels: Tensor,
    trace: Sequence[Tensor],
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[list[float], Mapping[str, Mapping[str, GradientGateReport]], Mapping[str, Any]]:
    arbiter.to(device)
    features = features.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(arbiter.parameters(), lr=lr, weight_decay=weight_decay, foreach=False, fused=False)
    losses: list[float] = []
    step_reports: dict[str, Mapping[str, GradientGateReport]] = {}
    for step, rows in enumerate(trace):
        rows = rows.to(device=device)
        opt.zero_grad(set_to_none=True)
        logits = arbiter(features.index_select(0, rows))
        loss = F.binary_cross_entropy_with_logits(logits, labels.index_select(0, rows))
        loss.backward()
        report = collect_gradient_report(arbiter)
        if step == 0:
            assert_nonzero_gradients(report, label="Stage2")
            step_reports["step1"] = report
        elif step == 1:
            assert_nonzero_gradients(report, label="Stage2 step2")
            step_reports["step2"] = report
        opt.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        logits_all = arbiter(features)
        probs = torch.sigmoid(logits_all).detach().cpu()
    if "step1" not in step_reports:
        raise RuntimeError("Stage2 trace must include at least one update")
    if len(trace) > 1 and "step2" not in step_reports:
        raise RuntimeError("Stage2 trace did not capture step2 gradients")
    summary = {
        "first": losses[0],
        "last": losses[-1],
        "num_updates": len(losses),
        "probability_summary": {
            "mean": float(probs.mean().item()),
            "min": float(probs.min().item()),
            "max": float(probs.max().item()),
            "positive_mean": float(probs[labels.detach().cpu().bool()].mean().item()),
            "negative_mean": float(probs[~labels.detach().cpu().bool()].mean().item()),
        },
    }
    return losses, step_reports, summary


def train_stage2_arbiters(
    trigger_plan: TriggerPlan,
    config: Gate0TrainConfig,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    from model.frameworks.v6.svra import ParentRiskArbiter, ParentRiskCeilingArbiter

    trace, trace_receipt = make_balanced_batch_trace(
        trigger_plan.labels,
        updates=config.stage2_updates,
        batch_size=config.stage2_batch_size,
        positive_per_batch=config.stage2_positive_per_batch,
        seed=config.seed,
        name="Stage2TriggeredFixed16to16Sampler",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        arbiter4 = ParentRiskArbiter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        arbiter13 = ParentRiskCeilingArbiter()
    losses4, grad4, summary4 = train_arbiter_with_trace(
        arbiter4,
        trigger_plan.features4,
        trigger_plan.labels,
        trace,
        lr=config.stage2_lr,
        weight_decay=config.stage2_weight_decay,
        device=device,
    )
    losses13, grad13, summary13 = train_arbiter_with_trace(
        arbiter13,
        trigger_plan.features13,
        trigger_plan.labels,
        trace,
        lr=config.stage2_lr,
        weight_decay=config.stage2_weight_decay,
        device=device,
    )
    all_trace, all_trace_receipt = make_balanced_batch_trace(
        trigger_plan.allrow_labels,
        updates=config.stage2_updates,
        batch_size=config.stage2_batch_size,
        positive_per_batch=config.stage2_positive_per_batch,
        seed=config.seed,
        name="Stage2AllRow4DFixed16to16Sampler",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        allrow4 = ParentRiskArbiter()
    all_losses, all_grad, all_summary = train_arbiter_with_trace(
        allrow4,
        trigger_plan.allrow_features4,
        trigger_plan.allrow_labels,
        all_trace,
        lr=config.stage2_lr,
        weight_decay=config.stage2_weight_decay,
        device=device,
    )
    return {
        "arbiter4": arbiter4,
        "arbiter13": arbiter13,
        "allrow4": allrow4,
        "losses": {
            "triggered_4d": losses4,
            "triggered_13d": losses13,
            "allrow_4d": all_losses,
        },
        "loss_summary": {
            "triggered_4d": summary4,
            "triggered_13d": summary13,
            "allrow_4d": all_summary,
        },
        "gradient_gates": {
            "triggered_4d": {
                step: _gradient_report_to_json(report)
                for step, report in grad4.items()
            },
            "triggered_13d": {
                step: _gradient_report_to_json(report)
                for step, report in grad13.items()
            },
            "allrow_4d": {
                step: _gradient_report_to_json(report)
                for step, report in all_grad.items()
            },
        },
        "samplers": {
            "triggered_shared": trace_receipt,
            "allrow_4d": all_trace_receipt,
        },
    }


def _gradient_report_to_json(
    report: Mapping[str, GradientGateReport],
) -> Mapping[str, Mapping[str, float | bool]]:
    return {
        name: {
            "finite": gate.finite,
            "nonzero": gate.nonzero,
            "grad_abs_sum": gate.grad_abs_sum,
            "grad_max_abs": gate.grad_max_abs,
        }
        for name, gate in report.items()
    }


def _module_state(module: nn.Module) -> Mapping[str, Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def build_checkpoint_payload(
    model: nn.Module,
    stage2: Mapping[str, Any],
    *,
    config: Gate0TrainConfig,
    config_sha256: str,
    commit: str,
    train_targets: TrainSubsetTargets,
    target_plan: EAACTargetPlan,
    trigger_plan: TriggerPlan,
    stage1_losses: Sequence[float],
    stage1_grad1: Mapping[str, GradientGateReport],
    stage1_grad2: Mapping[str, GradientGateReport],
    stage1_sampler: BalancedGroupSampler,
    stage1_sampled_stats: Mapping[str, Any],
    asset_receipt: Mapping[str, Any],
    oracle_receipt: Mapping[str, str],
) -> Mapping[str, Any]:
    model_state = _module_state(model)
    forbidden = [
        key for key in model_state if key.endswith(("role_embeddings", "name_embeddings", "class_ids"))
    ]
    if forbidden:
        raise RuntimeError("checkpoint includes class-axis asset buffers: " + ", ".join(forbidden))
    return {
        "schema_version": TRAIN_SCHEMA,
        "experiment_id": config.experiment_id,
        "condition_id": config.condition_id,
        "code_commit": commit,
        "config_sha256": config_sha256,
        "method": "SVRA",
        "gate": "Gate0",
        "train_scope": "two_stage_policy_plus_trigger_arbiters",
        "official_test_loaded": config.official_test_loaded,
        "unseen_images_used_for_gradient": config.unseen_images_used_for_gradient,
        "pclr_online_inference": config.pclr_online_inference,
        "action_bundle_manifest_sha256": config.action_bundle_manifest_sha256,
        "policy_state_dict": model_state,
        "trigger_arbiter4_state_dict": _module_state(stage2["arbiter4"]),
        "trigger_arbiter13_ceiling_state_dict": _module_state(stage2["arbiter13"]),
        "allrow_arbiter4_control_state_dict": _module_state(stage2["allrow4"]),
        "config": asdict(config),
        "asset_receipt": dict(asset_receipt),
        "oracle_receipt": dict(oracle_receipt),
        "train_targets": {
            "labels_path": train_targets.labels_path,
            "labels_sha256": train_targets.labels_sha256,
            "class_ids_path": train_targets.class_ids_path,
            "class_ids_sha256": train_targets.class_ids_sha256,
            "crop_features_path": train_targets.crop_features_path,
            "crop_features_sha256": train_targets.crop_features_sha256,
        },
        "target_stats": target_plan.stats,
        "trigger_stats": trigger_plan.stats,
        "stage1": {
            "loss": {
                "first": float(stage1_losses[0]),
                "second": float(stage1_losses[1]),
                "last": float(stage1_losses[-1]),
                "num_updates": len(stage1_losses),
            },
            "gradient_gates": {
                "step1": _gradient_report_to_json(stage1_grad1),
                "step2": _gradient_report_to_json(stage1_grad2),
            },
            "sampler": stage1_sampler.state_dict(),
            "sampled_class_stats": stage1_sampled_stats,
        },
        "stage2": {
            "threshold": float(config.stage2_threshold),
            "loss_summary": stage2["loss_summary"],
            "gradient_gates": stage2["gradient_gates"],
            "samplers": stage2["samplers"],
        },
        "reproducibility_identity": {
            "code_commit": commit,
            "config_sha256": config_sha256,
            "seed": config.seed,
            "action_geometry_sha256": config.action_geometry_sha256,
            "oracle_receipt_sha256": oracle_receipt["sha256"],
            "target_sha256": target_plan.stats["target_sha256"],
            "trigger_rows_sha256": trigger_plan.stats["trigger_rows_sha256"],
            "trigger_label_sha256": trigger_plan.stats["trigger_label_sha256"],
            "stage2_shared_batch_trace_sha256": stage2["samplers"]["triggered_shared"]["batch_trace_sha256"],
            "allrow_batch_trace_sha256": stage2["samplers"]["allrow_4d"]["batch_trace_sha256"],
        },
        "opened_keys": [
            "text_manifest",
            "role_tensor",
            "name_tensor",
            "patch_manifest",
            "cls_tensor",
            "patch_tensor_safe_view",
            "action_bundle_manifest",
            "dev_train_manifest",
            "dev_train.labels",
            "dev_train.class_ids",
            "dev_train.crop_features",
        ],
    }


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(str(path), map_location="cpu")


def _first_tensor(value: Any, *, source: Path) -> Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            if torch.is_tensor(item):
                return item
    raise RuntimeError(f"no tensor found in {source}")


def _load_long_vector(path: Path, *, name: str) -> Tensor:
    value = _first_tensor(_torch_load_cpu(path), source=path) if path.suffix == ".pt" else torch.as_tensor(np.load(path))
    value = value.detach().cpu().long().reshape(-1)
    if value.numel() == 0:
        raise RuntimeError(f"{name} is empty")
    return value


def _load_crop_feature_table(path: Path) -> Any:
    if path.suffix == ".pt":
        tensor = _first_tensor(_torch_load_cpu(path), source=path).detach().cpu()
        if tuple(tensor.shape[1:]) != (ACTION_COUNT, FEATURE_DIM):
            raise RuntimeError(f"crop features must have trailing shape [25,768], got {tuple(tensor.shape)}")
        return tensor
    arr = np.load(path, mmap_mode="r")
    if tuple(arr.shape[1:]) != (ACTION_COUNT, FEATURE_DIM):
        raise RuntimeError(f"crop features must have trailing shape [25,768], got {arr.shape}")
    return arr


def load_dev_train_targets(assets: Any) -> TrainSubsetTargets:
    data_api = _load_data_api()
    labels_path = _resolve_first_subset_output(data_api, assets, "dev_train", ("labels.pt", "labels.npy", "targets.pt", "targets.npy"))
    class_ids_path = _resolve_first_subset_output(data_api, assets, "dev_train", ("class_ids.pt", "class_ids.npy", "classes.pt", "classes.npy"))
    crop_path = _resolve_first_subset_output(data_api, assets, "dev_train", ("crop_features.pt", "crop_features.npy", "all25_crop_features.pt", "all25_crop_features.npy"))
    labels_sha = _verified_subset_output_sha(assets, "dev_train", labels_path)
    class_ids_sha = _verified_subset_output_sha(assets, "dev_train", class_ids_path)
    crop_sha = _verified_subset_output_sha(assets, "dev_train", crop_path)
    return TrainSubsetTargets(
        labels=_load_long_vector(labels_path, name="labels"),
        class_ids=_load_long_vector(class_ids_path, name="class_ids"),
        crop_features=_load_crop_feature_table(crop_path),
        labels_path=str(labels_path),
        class_ids_path=str(class_ids_path),
        crop_features_path=str(crop_path),
        labels_sha256=labels_sha,
        class_ids_sha256=class_ids_sha,
        crop_features_sha256=crop_sha,
    )


def _verified_subset_output_sha(assets: Any, subset_name: str, path: Path) -> str:
    manifest = getattr(assets, f"{subset_name}_manifest", None)
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"{subset_name}: missing loaded subset manifest for SHA receipt")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping):
        raise RuntimeError(f"{subset_name}: subset manifest missing outputs_sha256")
    target = path.name.lower()
    for key, value in outputs.items():
        key_base = str(key).replace("\\", "/").split("/")[-1].lower()
        if key_base == target and isinstance(value, str):
            return value
    raise RuntimeError(f"{subset_name}: manifest missing SHA for resolved output {path.name}")


def _resolve_first_subset_output(
    data_api: Mapping[str, Any],
    assets: Any,
    subset_name: str,
    basenames: Sequence[str],
) -> Path:
    errors = []
    for filename in basenames:
        try:
            # The subset manifest itself is SHA-bound when assets are loaded.
            # Reuse its frozen output SHA instead of re-reading the 5.9 GB
            # all25 table merely to recompute a hash already in that manifest.
            return data_api["resolve_subset_output"](assets, subset_name, filename, verify_sha=False)
        except Exception as exc:  # pragma: no cover - formal data dependent.
            errors.append(f"{filename}: {exc}")
    raise RuntimeError(f"{subset_name}: no required output resolved: " + "; ".join(errors))


def _load_data_api() -> Mapping[str, Any]:
    try:
        from . import rwdg_data
    except ImportError:  # pragma: no cover
        import rwdg_data  # type: ignore[no-redef]
    return {
        "ManifestContract": rwdg_data.ManifestContract,
        "TensorContract": rwdg_data.TensorContract,
        "SVRAAssetConfig": rwdg_data.SVRAAssetConfig,
        "load_svra_gate_data": rwdg_data.load_svra_gate_data,
        "resolve_subset_output": rwdg_data.resolve_subset_output,
    }


def asset_config_from_train_config(config: Gate0TrainConfig) -> Any:
    data = _load_data_api()
    manifest = data["ManifestContract"]
    tensor = data["TensorContract"]
    asset = data["SVRAAssetConfig"]
    return asset(
        text_manifest=manifest(config.text_manifest, config.text_manifest_sha256),
        role_tensor=tensor(config.role_tensor, config.role_tensor_sha256, (200, 8, FEATURE_DIM), "float32"),
        name_tensor=tensor(config.name_tensor, config.name_tensor_sha256, (200, FEATURE_DIM), "float32"),
        patch_manifest=manifest(config.patch_manifest, config.patch_manifest_sha256),
        cls_tensor=tensor(config.cls_tensor, config.cls_tensor_sha256, (config.trainval_count, FEATURE_DIM), "float32"),
        patch_tensor=tensor(config.patch_tensor, config.patch_tensor_sha256, (config.trainval_count, 576, FEATURE_DIM), "float16"),
        action_bundle_manifest=manifest(
            config.action_bundle_manifest,
            config.action_bundle_manifest_sha256,
        ),
        dev_train_manifest_sha256=config.dev_train_manifest_sha256,
        dev_eval_manifest_sha256=config.dev_eval_manifest_sha256,
        dev_eval_oracle_manifest_sha256=config.dev_eval_oracle_manifest_sha256,
        att_splits_mat_path=config.att_splits_mat_path,
        trainval_count=config.trainval_count,
    )


def build_svra_model(
    role_embeddings: Tensor,
    name_embeddings: Tensor,
    class_ids: Tensor,
    *,
    device: torch.device,
    seed: int,
) -> nn.Module:
    from model.frameworks.v6.svra import SemanticVisualRiskArbiter

    active_ids = class_ids.to(dtype=torch.long)
    if role_embeddings.shape[0] != active_ids.numel():
        role_embeddings = role_embeddings.index_select(
            0,
            active_ids.to(device=role_embeddings.device),
        )
    if name_embeddings.shape[0] != active_ids.numel():
        name_embeddings = name_embeddings.index_select(
            0,
            active_ids.to(device=name_embeddings.device),
        )
    return SemanticVisualRiskArbiter(
        role_embeddings,
        name_embeddings,
        active_ids,
        seed=seed,
    ).to(device)


def verify_oracle_receipt(config: Gate0TrainConfig) -> Mapping[str, str]:
    path = Path(config.oracle_receipt).expanduser()
    if not path.is_file():
        raise RuntimeError(f"oracle_receipt is missing: {path}")
    path = path.resolve()
    actual_sha = sha256_file(path)
    if actual_sha.lower() != config.oracle_receipt_sha256.lower():
        raise RuntimeError(
            "oracle_receipt_sha256 mismatch: "
            f"got={actual_sha} expected={config.oracle_receipt_sha256}"
        )
    return {"path": str(path), "sha256": actual_sha}


def _asset_receipt_from_assets(assets: Any) -> Mapping[str, Any]:
    cfg = getattr(assets, "config", None)
    summaries = getattr(assets, "subset_summaries", {})
    if cfg is None:
        return {}
    def dc(value: Any) -> Any:
        return asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return {
        "text_manifest": dc(cfg.text_manifest),
        "role_tensor": dc(cfg.role_tensor),
        "name_tensor": dc(cfg.name_tensor),
        "patch_manifest": dc(cfg.patch_manifest),
        "cls_tensor": dc(cfg.cls_tensor),
        "patch_tensor": dc(cfg.patch_tensor),
        "action_bundle_manifest": dc(cfg.action_bundle_manifest),
        "dev_train_manifest_sha256": cfg.dev_train_manifest_sha256,
        "dev_eval_manifest_sha256": cfg.dev_eval_manifest_sha256,
        "dev_eval_oracle_manifest_sha256": cfg.dev_eval_oracle_manifest_sha256,
        "subset_summaries": dict(summaries),
    }


def train_full_gate0(
    config: Gate0TrainConfig,
    *,
    config_sha256: str,
    repo_root: Path,
    expected_commit: str,
) -> Path:
    validate_config(config)
    oracle_receipt = verify_oracle_receipt(config)
    commit, clean, status = git_commit_and_clean(repo_root)
    if commit != expected_commit:
        raise RuntimeError(f"expected commit mismatch: HEAD={commit}, expected={expected_commit}")
    if config.require_clean_tree and not clean:
        raise RuntimeError("repository must be clean before formal SVRA Gate0 training:\n" + status)
    device = resolve_device(config)
    set_reproducibility(config.seed)
    data_api = _load_data_api()
    assets, views = data_api["load_svra_gate_data"](
        asset_config_from_train_config(config),
        strict_sha=config.strict_sha,
        validate_tensor_values=config.validate_tensor_values,
        strict_eval_boundary=True,
    )
    try:
        train_view = views["dev_train"]
        train_targets = load_dev_train_targets(assets)
        model = build_svra_model(
            assets.role_embeddings,
            assets.name_embeddings,
            train_targets.class_ids,
            device=device,
            seed=config.seed,
        )
        target_plan = precompute_eaac_target_plan(
            model,
            train_view,
            train_targets,
            batch_size=config.stage1_batch_size,
            device=device,
        )
        validate_stage1_target_contract(target_plan, config)
        sampled_stats = sampled_class_histogram(
            target_plan.targets26,
            target_plan.groups,
            updates=config.stage1_updates,
            batch_size=config.stage1_batch_size,
            seed=config.seed,
            challenger_per_batch=config.stage1_challenger_per_batch,
        )
        sampler = stage1_sampler_from_groups(
            target_plan.groups,
            batch_size=config.stage1_batch_size,
            challenger_per_batch=config.stage1_challenger_per_batch,
            seed=config.seed,
        )
        stage1_parameters = configure_stage1_trainable(model)
        optimizer = torch.optim.AdamW(stage1_parameters, lr=config.stage1_lr, weight_decay=config.stage1_weight_decay, foreach=False, fused=False)
        stage1_losses: list[float] = []
        rows = sampler.sample()
        loss, grad1 = train_stage1_step(model, optimizer, train_view, train_targets, rows, target_plan, device=device)
        stage1_losses.append(loss)
        assert_nonzero_gradients(grad1, label="Stage1 step1")
        rows = sampler.sample()
        loss, grad2 = train_stage1_step(model, optimizer, train_view, train_targets, rows, target_plan, device=device)
        stage1_losses.append(loss)
        assert_nonzero_gradients(grad2, label="Stage1 step2")
        assert_stage1_second_step_contract(grad2)
        for _ in range(2, config.stage1_updates):
            rows = sampler.sample()
            loss, _ = train_stage1_step(model, optimizer, train_view, train_targets, rows, target_plan, device=device)
            stage1_losses.append(loss)

        freeze_stage1_model(model)
        trigger_plan = build_trigger_plan(
            model,
            train_view,
            train_targets,
            target_plan,
            batch_size=config.stage1_batch_size,
            threshold=config.stage2_threshold,
            device=device,
        )
        validate_trigger_contract(trigger_plan, config)
        stage2 = train_stage2_arbiters(trigger_plan, config, device=device)
        checkpoint = build_checkpoint_payload(
            model,
            stage2,
            config=config,
            config_sha256=config_sha256,
            commit=commit,
            train_targets=train_targets,
            target_plan=target_plan,
            trigger_plan=trigger_plan,
            stage1_losses=stage1_losses,
            stage1_grad1=grad1,
            stage1_grad2=grad2,
            stage1_sampler=sampler,
            stage1_sampled_stats=sampled_stats,
            asset_receipt=_asset_receipt_from_assets(assets),
            oracle_receipt=oracle_receipt,
        )
        output_dir = prepare_output_dir(config.output_dir, repo_root)
        checkpoint_path = output_dir / "svra_gate0_combined.pt"
        atomic_torch_save(checkpoint_path, checkpoint)
        checkpoint_sha = sha256_file(checkpoint_path)
        atomic_write_json(
            output_dir / "train_history.json",
            {
                "schema_version": TRAIN_SCHEMA,
                "experiment_id": config.experiment_id,
                "condition_id": config.condition_id,
                "code_commit": commit,
                "config_sha256": config_sha256,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "oracle_receipt": checkpoint["oracle_receipt"],
                "official_test_loaded": config.official_test_loaded,
                "unseen_images_used_for_gradient": config.unseen_images_used_for_gradient,
                "pclr_online_inference": config.pclr_online_inference,
                "target_stats": checkpoint["target_stats"],
                "trigger_stats": checkpoint["trigger_stats"],
                "stage1": checkpoint["stage1"],
                "stage2": checkpoint["stage2"],
            },
        )
        return checkpoint_path
    finally:
        close = getattr(assets, "close", None)
        if callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    config, config_sha = load_strict_config(args.config)
    if config_sha.lower() != args.expected_config_sha.lower():
        raise RuntimeError(f"config SHA mismatch: got {config_sha}, expected {args.expected_config_sha}")
    print(train_full_gate0(config, config_sha256=config_sha, repo_root=repo_root, expected_commit=args.expected_commit))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTION_COUNT",
    "ACTION_GEOMETRY_SHA256",
    "EAAC_CLASS_COUNT",
    "FEATURE_DIM",
    "Gate0TrainConfig",
    "GradientGateReport",
    "LinearTriggerArbiter",
    "STRICT_CONFIG_FIELDS",
    "SVRA_PREREGISTERED_STAGE1_TARGET_COUNTS",
    "SVRA_PREREGISTERED_TRIGGER_COUNTS",
    "TRAIN_SCHEMA",
    "TrainSubsetTargets",
    "TriggerPlan",
    "BalancedGroupSampler",
    "build_checkpoint_payload",
    "build_trigger_features",
    "build_trigger_plan",
    "eaac_action_loss",
    "eaac_action_targets",
    "eaac_policy_logits",
    "load_strict_config",
    "make_balanced_batch_trace",
    "precompute_eaac_target_plan",
    "sampled_class_histogram",
    "stage1_sampler_from_groups",
    "train_stage2_arbiters",
    "validate_config",
    "validate_stage1_target_contract",
    "validate_trigger_contract",
]
