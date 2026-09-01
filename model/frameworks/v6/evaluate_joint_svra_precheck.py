"""Official zero-crop precheck evaluator for IDEA-200 / Joint SVRA.

The evaluator reads official test features first, freezes every 200-class logit
tensor for every condition, and only then reads labels for U/S/H/ZS metrics.
No raw image, crop, or eval all25 table is opened in this path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from model.frameworks.v6.svra import (
    ACTION_COUNT,
    ACTION_GEOMETRY_SHA256,
    FEATURE_DIM,
    SemanticVisualRiskArbiter,
)
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v6-joint-svra-precheck-eval.v1"
CHECKPOINT_SCHEMA = "gzsl-paper.v6-joint-svra-precheck-train.v1"
FULL_CONDITION = "JOINT_SVRA_FULL"
NO_JOINT_CONDITION = "JOINT_SVRA_NO_JOINT"
SEQUENTIAL_CONDITION = "JOINT_SVRA_SEQUENTIAL"
CONDITION_TO_CONFIG_KEY = {
    "full": "full_checkpoint",
    "no_joint": "no_joint_checkpoint",
    "sequential": "sequential_checkpoint",
}
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "role_tensor",
    "role_tensor_sha256",
    "name_tensor",
    "name_tensor_sha256",
    "test_seen_cls",
    "test_seen_cls_sha256",
    "test_seen_patches",
    "test_seen_patches_sha256",
    "test_seen_labels",
    "test_seen_labels_sha256",
    "test_unseen_cls",
    "test_unseen_cls_sha256",
    "test_unseen_patches",
    "test_unseen_patches_sha256",
    "test_unseen_labels",
    "test_unseen_labels_sha256",
    "action_geometry_sha256",
    "full_checkpoint",
    "no_joint_checkpoint",
    "sequential_checkpoint",
    "device",
    "random_seed",
    "eval_batch_size",
    "bootstrap_seed",
    "bootstrap_samples",
    "module_contract_margin",
    "support_control_margin",
    "max_action_occupancy",
    "require_clean_tree",
    "test_used_for_selection",
    "test_used_for_hyperparameter_selection",
    "nested_official_test_selection",
    "strict_blind_claim",
    "official_test_loaded",
    "unseen_images_used_for_gradient",
    "pclr_online_inference",
}


class JointSVRAPrecheckError(RuntimeError):
    """Raised when the official Joint-SVRA precheck contract is violated."""


@dataclass(frozen=True)
class OfficialFeatures:
    seen_cls: torch.Tensor
    seen_patches: torch.Tensor
    unseen_cls: torch.Tensor
    unseen_patches: torch.Tensor


@dataclass(frozen=True)
class OfficialLabels:
    seen: torch.Tensor
    unseen: torch.Tensor
    seen_classes: torch.Tensor
    unseen_classes: torch.Tensor


@dataclass(frozen=True)
class FrozenSplit:
    logits: torch.Tensor
    parent_logits: torch.Tensor
    top2: torch.Tensor
    actions: torch.Tensor
    trigger: torch.Tensor
    swap: torch.Tensor
    soft_hard_trigger_equal: bool


@dataclass(frozen=True)
class FrozenCondition:
    name: str
    seen: FrozenSplit
    unseen: FrozenSplit

    @property
    def logits_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.logits, self.unseen.logits), dim=0)

    @property
    def actions_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.actions, self.unseen.actions), dim=0)

    @property
    def trigger_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.trigger, self.unseen.trigger), dim=0)

    @property
    def swap_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.swap, self.unseen.swap), dim=0)


def load_config(path: Path | str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != REQUIRED_CONFIG_KEYS:
        raise JointSVRAPrecheckError(
            "Joint SVRA precheck eval config字段错误；"
            f"缺少={sorted(REQUIRED_CONFIG_KEYS - actual)} 多出={sorted(actual - REQUIRED_CONFIG_KEYS)}"
        )
    invalid = (
        config["schema_version"] != SCHEMA
        or int(config["random_seed"]) != 7
        or int(config["bootstrap_seed"]) != 7
        or int(config["bootstrap_samples"]) != 10000
        or int(config["eval_batch_size"]) <= 0
        or float(config["module_contract_margin"]) != 1.0
        or float(config["support_control_margin"]) != 0.5
        or float(config["max_action_occupancy"]) != 0.70
        or config["action_geometry_sha256"] != ACTION_GEOMETRY_SHA256
        or config["test_used_for_selection"] is not True
        or config["test_used_for_hyperparameter_selection"] is not False
        or config["nested_official_test_selection"] is not True
        or config["strict_blind_claim"] is not False
        or config["official_test_loaded"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["pclr_online_inference"] is not False
    )
    if invalid:
        raise JointSVRAPrecheckError("Joint SVRA precheck eval固定协议错误。")
    return config, sha256_file(path)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _first_tensor(value: Any, *, source: Path) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        for item in value.values():
            if torch.is_tensor(item):
                return item.detach().cpu()
    raise JointSVRAPrecheckError(f"{source} 中没有tensor。")


def _load_tensor(path: str, sha256: str, *, shape: tuple[int, ...], name: str) -> torch.Tensor:
    file_path = Path(path)
    if not file_path.is_file() or sha256_file(file_path).lower() != str(sha256).lower():
        raise JointSVRAPrecheckError(f"{name} 路径或SHA错误。")
    tensor = _first_tensor(_torch_load(file_path), source=file_path)
    if tuple(int(x) for x in tensor.shape) != shape:
        raise JointSVRAPrecheckError(f"{name} shape错误：{tuple(tensor.shape)} != {shape}")
    if not bool(torch.isfinite(tensor).all()):
        raise JointSVRAPrecheckError(f"{name} 包含NaN/Inf。")
    return tensor


def load_text_tensors(config: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    roles = _load_tensor(
        config["role_tensor"],
        config["role_tensor_sha256"],
        shape=(200, 8, FEATURE_DIM),
        name="role_tensor",
    ).float()
    names = _load_tensor(
        config["name_tensor"],
        config["name_tensor_sha256"],
        shape=(200, FEATURE_DIM),
        name="name_tensor",
    ).float()
    return roles, names, torch.arange(200, dtype=torch.long)


def load_official_features(config: Mapping[str, Any]) -> OfficialFeatures:
    seen_cls = _load_tensor(
        config["test_seen_cls"],
        config["test_seen_cls_sha256"],
        shape=(1764, FEATURE_DIM),
        name="test_seen_cls",
    ).float()
    seen_patches = _load_tensor(
        config["test_seen_patches"],
        config["test_seen_patches_sha256"],
        shape=(1764, 576, FEATURE_DIM),
        name="test_seen_patches",
    )
    unseen_cls = _load_tensor(
        config["test_unseen_cls"],
        config["test_unseen_cls_sha256"],
        shape=(2967, FEATURE_DIM),
        name="test_unseen_cls",
    ).float()
    unseen_patches = _load_tensor(
        config["test_unseen_patches"],
        config["test_unseen_patches_sha256"],
        shape=(2967, 576, FEATURE_DIM),
        name="test_unseen_patches",
    )
    return OfficialFeatures(seen_cls, seen_patches, unseen_cls, unseen_patches)


def load_official_labels_after_logits(config: Mapping[str, Any]) -> OfficialLabels:
    seen = _load_tensor(
        config["test_seen_labels"],
        config["test_seen_labels_sha256"],
        shape=(1764,),
        name="test_seen_labels",
    ).long()
    unseen = _load_tensor(
        config["test_unseen_labels"],
        config["test_unseen_labels_sha256"],
        shape=(2967,),
        name="test_unseen_labels",
    ).long()
    seen_classes = torch.unique(seen, sorted=True)
    unseen_classes = torch.unique(unseen, sorted=True)
    all_classes = torch.cat((seen_classes, unseen_classes)).sort().values
    if seen_classes.numel() != 150 or unseen_classes.numel() != 50:
        raise JointSVRAPrecheckError("official seen/unseen类别数必须为150/50。")
    if not torch.equal(all_classes, torch.arange(200, dtype=torch.long)):
        raise JointSVRAPrecheckError("official seen/unseen类别必须完整覆盖200轴。")
    return OfficialLabels(seen, unseen, seen_classes, unseen_classes)


def load_checkpoint(
    spec: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_condition: str,
) -> Mapping[str, Any]:
    required = {"path", "sha256", "training_commit", "train_config_sha256"}
    if not isinstance(spec, Mapping) or set(spec) != required:
        raise JointSVRAPrecheckError("checkpoint字段必须精确包含path/sha256/training_commit/train_config_sha256。")
    path = Path(str(spec["path"]))
    if not path.is_file() or sha256_file(path).lower() != str(spec["sha256"]).lower():
        raise JointSVRAPrecheckError(f"{expected_condition} checkpoint路径或SHA错误。")
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, Mapping):
        raise JointSVRAPrecheckError(f"{expected_condition} checkpoint不是mapping。")
    invalid = (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("condition_id") != expected_condition
        or checkpoint.get("code_commit") != expected_commit
        or spec["training_commit"] != expected_commit
        or checkpoint.get("config_sha256") != spec["train_config_sha256"]
        or "state_dict" not in checkpoint
    )
    if invalid:
        raise JointSVRAPrecheckError(f"{expected_condition} checkpoint身份错误。")
    return checkpoint


def instantiate_model(
    roles: torch.Tensor,
    names: torch.Tensor,
    class_ids: torch.Tensor,
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> SemanticVisualRiskArbiter:
    model = SemanticVisualRiskArbiter(roles, names, class_ids, seed=7).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def apply_pair_swap(parent_logits: torch.Tensor, top2: torch.Tensor, swap: torch.Tensor) -> torch.Tensor:
    logits = parent_logits.clone()
    if parent_logits.ndim != 2 or top2.shape != (parent_logits.shape[0], 2):
        raise JointSVRAPrecheckError("parent_logits/top2 shape错误。")
    if swap.shape != (parent_logits.shape[0],):
        raise JointSVRAPrecheckError("swap shape错误。")
    rows = torch.nonzero(swap.bool(), as_tuple=False).flatten()
    if rows.numel():
        leaders = top2.index_select(0, rows)[:, 0]
        challengers = top2.index_select(0, rows)[:, 1]
        leader_values = logits[rows, leaders].clone()
        logits[rows, leaders] = logits[rows, challengers]
        logits[rows, challengers] = leader_values
    return logits


@torch.no_grad()
def freeze_split(
    model: SemanticVisualRiskArbiter,
    cls_features: torch.Tensor,
    patch_features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    semantic_off: bool = False,
    visual_off: bool = False,
    interaction_off: bool = False,
    control: str = "full",
) -> FrozenSplit:
    logits: list[torch.Tensor] = []
    parent_logits: list[torch.Tensor] = []
    top2_values: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    triggers: list[torch.Tensor] = []
    swaps: list[torch.Tensor] = []
    equivalence = True
    for start in range(0, cls_features.shape[0], batch_size):
        cls = cls_features[start : start + batch_size].to(device).float()
        if visual_off:
            patches = None
        else:
            patches = patch_features[start : start + batch_size].to(device).float()
        state = model.policy_state(
            cls,
            patches,
            semantic_off=semantic_off,
            visual_off=visual_off,
        )
        max_action_logit = state.utility_logits.max(dim=1).values
        soft_trigger = torch.sigmoid(max_action_logit) > 0.5
        hard_trigger = max_action_logit > 0
        equivalence = equivalence and bool(torch.equal(soft_trigger.cpu(), hard_trigger.cpu()))
        equivalence = equivalence and bool(torch.equal(state.trigger.detach().cpu().bool(), hard_trigger.cpu()))

        if interaction_off:
            risk_prob = torch.zeros_like(max_action_logit)
            swap = torch.zeros_like(hard_trigger)
        elif control == "always_swap":
            risk_prob = torch.ones_like(max_action_logit)
            swap = hard_trigger
        elif control == "all_row4d":
            risk_prob = model.risk_probability(state.parent_stats, head="all_row4d")
            swap = risk_prob > 0.5
        else:
            risk_prob = model.risk_probability(state.parent_stats, head="triggered4d")
            swap = hard_trigger & (risk_prob > 0.5)

        current_logits = apply_pair_swap(state.parent_logits, state.top2, swap)
        logits.append(current_logits.detach().cpu())
        parent_logits.append(state.parent_logits.detach().cpu())
        top2_values.append(state.top2.detach().cpu().long())
        actions.append(state.selected_action.detach().cpu().long())
        triggers.append(hard_trigger.detach().cpu().bool())
        swaps.append(swap.detach().cpu().bool())
    return FrozenSplit(
        logits=torch.cat(logits),
        parent_logits=torch.cat(parent_logits),
        top2=torch.cat(top2_values),
        actions=torch.cat(actions),
        trigger=torch.cat(triggers),
        swap=torch.cat(swaps),
        soft_hard_trigger_equal=equivalence,
    )


def freeze_condition(
    name: str,
    model: SemanticVisualRiskArbiter,
    features: OfficialFeatures,
    *,
    device: torch.device,
    batch_size: int,
    semantic_off: bool = False,
    visual_off: bool = False,
    interaction_off: bool = False,
    control: str = "full",
) -> FrozenCondition:
    seen = freeze_split(
        model,
        features.seen_cls,
        features.seen_patches,
        device=device,
        batch_size=batch_size,
        semantic_off=semantic_off,
        visual_off=visual_off,
        interaction_off=interaction_off,
        control=control,
    )
    unseen = freeze_split(
        model,
        features.unseen_cls,
        features.unseen_patches,
        device=device,
        batch_size=batch_size,
        semantic_off=semantic_off,
        visual_off=visual_off,
        interaction_off=interaction_off,
        control=control,
    )
    return FrozenCondition(name=name, seen=seen, unseen=unseen)


def parent_condition(full: FrozenCondition) -> FrozenCondition:
    seen = FrozenSplit(
        logits=full.seen.parent_logits,
        parent_logits=full.seen.parent_logits,
        top2=full.seen.top2,
        actions=full.seen.actions,
        trigger=torch.zeros_like(full.seen.trigger),
        swap=torch.zeros_like(full.seen.swap),
        soft_hard_trigger_equal=True,
    )
    unseen = FrozenSplit(
        logits=full.unseen.parent_logits,
        parent_logits=full.unseen.parent_logits,
        top2=full.unseen.top2,
        actions=full.unseen.actions,
        trigger=torch.zeros_like(full.unseen.trigger),
        swap=torch.zeros_like(full.unseen.swap),
        soft_hard_trigger_equal=True,
    )
    return FrozenCondition(name="parent", seen=seen, unseen=unseen)


def _per_class_vector(labels: torch.Tensor, predictions: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    values = []
    for class_id in classes.long():
        mask = labels.eq(class_id)
        if not bool(mask.any()):
            raise JointSVRAPrecheckError(f"评估split缺少类别{int(class_id)}。")
        values.append(predictions[mask].eq(class_id).double().mean())
    return torch.stack(values)


def condition_metrics(condition: FrozenCondition, labels: OfficialLabels) -> dict[str, Any]:
    seen_pred = condition.seen.logits.argmax(dim=1).cpu().long()
    unseen_pred = condition.unseen.logits.argmax(dim=1).cpu().long()
    unseen_axis = labels.unseen_classes.long()
    zsl_pred = unseen_axis[condition.unseen.logits.index_select(1, unseen_axis).argmax(dim=1)].cpu().long()
    seen_vec = _per_class_vector(labels.seen, seen_pred, labels.seen_classes)
    unseen_vec = _per_class_vector(labels.unseen, unseen_pred, labels.unseen_classes)
    zsl_vec = _per_class_vector(labels.unseen, zsl_pred, labels.unseen_classes)
    s = 100.0 * float(seen_vec.mean())
    u = 100.0 * float(unseen_vec.mean())
    zsl = 100.0 * float(zsl_vec.mean())
    h = 2.0 * s * u / (s + u) if s + u else 0.0
    return {
        "U": u,
        "S": s,
        "H": h,
        "ZS": zsl,
        "seen_per_class": seen_vec,
        "unseen_per_class": unseen_vec,
        "zsl_per_class": zsl_vec,
        "seen_prediction": seen_pred,
        "unseen_prediction": unseen_pred,
        "zsl_prediction": zsl_pred,
    }


def paired_h_comparison(
    full: Mapping[str, Any],
    other: Mapping[str, Any],
    seen_matrix: torch.Tensor,
    unseen_matrix: torch.Tensor,
) -> dict[str, Any]:
    full_s = full["seen_per_class"].double()
    full_u = full["unseen_per_class"].double()
    other_s = other["seen_per_class"].double()
    other_u = other["unseen_per_class"].double()
    sample_full_s = full_s[seen_matrix].mean(dim=1) * 100.0
    sample_full_u = full_u[unseen_matrix].mean(dim=1) * 100.0
    sample_other_s = other_s[seen_matrix].mean(dim=1) * 100.0
    sample_other_u = other_u[unseen_matrix].mean(dim=1) * 100.0
    full_h = 2.0 * sample_full_s * sample_full_u / (sample_full_s + sample_full_u).clamp_min(1e-12)
    other_h = 2.0 * sample_other_s * sample_other_u / (sample_other_s + sample_other_u).clamp_min(1e-12)
    samples = full_h - other_h
    ci = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.double))
    return {
        "observed_pp": float(full["H"] - other["H"]),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def transition_summary(parent: Mapping[str, Any], full: Mapping[str, Any], labels: OfficialLabels) -> dict[str, Any]:
    old_seen = parent["seen_prediction"].eq(labels.seen)
    new_seen = full["seen_prediction"].eq(labels.seen)
    old_unseen = parent["unseen_prediction"].eq(labels.unseen)
    new_unseen = full["unseen_prediction"].eq(labels.unseen)
    corrected = int((~old_seen & new_seen).sum() + (~old_unseen & new_unseen).sum())
    damaged = int((old_seen & ~new_seen).sum() + (old_unseen & ~new_unseen).sum())
    return {"corrected": corrected, "damaged": damaged, "net": corrected - damaged}


def group_statistics(full_condition: FrozenCondition, labels: OfficialLabels, parent: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    all_top2 = torch.cat((full_condition.seen.top2, full_condition.unseen.top2), dim=0)
    all_labels = torch.cat((labels.seen, labels.unseen)).long()
    all_trigger = full_condition.trigger_for_sha
    parent_pred = torch.cat((parent["seen_prediction"], parent["unseen_prediction"]))
    full_pred = torch.cat((full["seen_prediction"], full["unseen_prediction"]))
    leader = all_top2[:, 0]
    challenger = all_top2[:, 1]
    group = torch.full_like(all_labels, 2)
    group[all_labels.eq(leader)] = 0
    group[all_labels.eq(challenger)] = 1
    result = {}
    for value, name in ((0, "leader"), (1, "challenger"), (2, "outside")):
        mask = group.eq(value)
        parent_correct = parent_pred[mask].eq(all_labels[mask])
        full_correct = full_pred[mask].eq(all_labels[mask])
        result[name] = {
            "count": int(mask.sum()),
            "trigger": int((all_trigger & mask).sum()),
            "abstain": int((~all_trigger & mask).sum()),
            "corrected": int((~parent_correct & full_correct).sum()),
            "damaged": int((parent_correct & ~full_correct).sum()),
            "net": int(full_correct.sum() - parent_correct.sum()),
        }
    return result


def group_safety_gate(group_stats: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    leader = group_stats["leader"]
    challenger = group_stats["challenger"]
    outside = group_stats["outside"]
    leader_rate = leader["trigger"] / max(1, leader["count"])
    challenger_rate = challenger["trigger"] / max(1, challenger["count"])
    total_net = leader["net"] + challenger["net"] + outside["net"]
    total_trigger = leader["trigger"] + challenger["trigger"] + outside["trigger"]
    total_count = leader["count"] + challenger["count"] + outside["count"]
    gates = {
        "challenger_trigger_gt_leader_trigger": challenger_rate > leader_rate,
        "leader_damage_lt_challenger_corrections": leader["damaged"] < challenger["corrected"],
        "net_positive": total_net > 0,
        "has_trigger_and_abstain": 0 < total_trigger < total_count,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "leader_trigger_rate": leader_rate,
        "challenger_trigger_rate": challenger_rate,
    }


def action_summary(condition: FrozenCondition) -> dict[str, Any]:
    actions = condition.actions_for_sha
    trigger = condition.trigger_for_sha
    histogram = torch.bincount(actions[trigger], minlength=ACTION_COUNT)
    trigger_count = int(trigger.sum())
    return {
        "trigger_count": trigger_count,
        "abstain_count": int(trigger.numel() - trigger_count),
        "trigger_rate": float(trigger.double().mean()),
        "triggered_action_histogram": [int(x) for x in histogram],
        "used_actions": int(histogram.gt(0).sum()),
        "highest_occupancy": float(histogram.max()) / max(1, trigger_count),
    }


def condition_sha(condition: FrozenCondition) -> dict[str, Any]:
    return {
        "logits_sha256": _tensor_sha256(condition.logits_for_sha),
        "action_sha256": _tensor_sha256(condition.actions_for_sha),
        "trigger_sha256": _tensor_sha256(condition.trigger_for_sha.to(torch.uint8)),
        "swap_sha256": _tensor_sha256(condition.swap_for_sha.to(torch.uint8)),
        "swap_count": int(condition.swap_for_sha.sum()),
        "trigger_count": int(condition.trigger_for_sha.sum()),
        "soft_hard_trigger_equal": (
            condition.seen.soft_hard_trigger_equal
            and condition.unseen.soft_hard_trigger_equal
        ),
    }


def public_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        name: {key: float(value[key]) for key in ("U", "S", "H", "ZS")}
        for name, value in metrics.items()
    }


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict[str, Any]:
    config, config_sha = load_config(config_path)
    if config_sha.lower() != expected_config_sha.lower():
        raise JointSVRAPrecheckError("Joint SVRA precheck config SHA mismatch.")
    if bool(config["require_clean_tree"]):
        require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise JointSVRAPrecheckError("Joint SVRA precheck expected commit mismatch.")
    reproducibility = configure_reproducibility(
        int(config["random_seed"]),
        strict_determinism=True,
        deterministic_warn_only=False,
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise JointSVRAPrecheckError("Joint SVRA precheck必须在可用CUDA设备上运行。")

    roles, names, class_ids = load_text_tensors(config)
    features = load_official_features(config)
    checkpoints = {
        "full": load_checkpoint(config["full_checkpoint"], expected_commit=expected_commit, expected_condition=FULL_CONDITION),
        "no_joint": load_checkpoint(config["no_joint_checkpoint"], expected_commit=expected_commit, expected_condition=NO_JOINT_CONDITION),
        "sequential": load_checkpoint(config["sequential_checkpoint"], expected_commit=expected_commit, expected_condition=SEQUENTIAL_CONDITION),
    }
    models = {
        name: instantiate_model(roles, names, class_ids, checkpoint, device)
        for name, checkpoint in checkpoints.items()
    }
    batch_size = int(config["eval_batch_size"])

    full = freeze_condition("full", models["full"], features, device=device, batch_size=batch_size)
    conditions = {
        "full": full,
        "parent": parent_condition(full),
        "s_off": freeze_condition("s_off", models["full"], features, device=device, batch_size=batch_size, semantic_off=True),
        "v_off": freeze_condition("v_off", models["full"], features, device=device, batch_size=batch_size, visual_off=True),
        "i_off": freeze_condition("i_off", models["full"], features, device=device, batch_size=batch_size, interaction_off=True),
        "no_joint": freeze_condition("no_joint", models["no_joint"], features, device=device, batch_size=batch_size),
        "sequential": freeze_condition("sequential", models["sequential"], features, device=device, batch_size=batch_size),
    }

    # All condition logits/actions/triggers/swaps are frozen before this point.
    labels = load_official_labels_after_logits(config)
    metrics = {name: condition_metrics(condition, labels) for name, condition in conditions.items()}
    generator = torch.Generator().manual_seed(int(config["bootstrap_seed"]))
    seen_matrix = torch.randint(0, labels.seen_classes.numel(), (int(config["bootstrap_samples"]), labels.seen_classes.numel()), generator=generator)
    unseen_matrix = torch.randint(0, labels.unseen_classes.numel(), (int(config["bootstrap_samples"]), labels.unseen_classes.numel()), generator=generator)
    comparisons = {
        name: paired_h_comparison(metrics["full"], value, seen_matrix, unseen_matrix)
        for name, value in metrics.items()
        if name != "full"
    }
    transitions = transition_summary(metrics["parent"], metrics["full"], labels)
    group_stats = group_statistics(conditions["full"], labels, metrics["parent"], metrics["full"])
    group_safety = group_safety_gate(group_stats)
    action = action_summary(conditions["full"])
    condition_shas = {name: condition_sha(condition) for name, condition in conditions.items()}
    gates = {
        "full_vs_parent_observed_plus1H": comparisons["parent"]["observed_pp"] >= float(config["module_contract_margin"]),
        "full_vs_s_off_observed_plus1H": comparisons["s_off"]["observed_pp"] >= float(config["module_contract_margin"]),
        "full_vs_v_off_observed_plus1H": comparisons["v_off"]["observed_pp"] >= float(config["module_contract_margin"]),
        "full_vs_i_off_observed_plus1H": comparisons["i_off"]["observed_pp"] >= float(config["module_contract_margin"]),
        "full_vs_no_joint_ci_plus0_5H": comparisons["no_joint"]["observed_pp"] >= float(config["support_control_margin"]) and comparisons["no_joint"]["ci95"][0] > 0,
        "full_vs_sequential_ci_plus0_5H": comparisons["sequential"]["observed_pp"] >= float(config["support_control_margin"]) and comparisons["sequential"]["ci95"][0] > 0,
        "net_positive": transitions["net"] > 0,
        "corrections_gt_damages": transitions["corrected"] > transitions["damaged"],
        "exact_soft_hard_trigger_equivalence": all(item["soft_hard_trigger_equal"] for item in condition_shas.values()),
        "raw_image_open_count_zero": True,
        "raw_crop_encode_count_zero": True,
        "eval_all25_opened_false": True,
        "labels_loaded_after_logits": True,
        "has_trigger_and_abstain": action["trigger_count"] > 0 and action["abstain_count"] > 0,
        "used_at_least_two_actions": action["used_actions"] >= 2,
        "highest_occupancy_lte_bound": action["highest_occupancy"] <= float(config["max_action_occupancy"]),
        **group_safety["gates"],
    }
    passed = all(gates.values())
    result = {
        "schema_version": SCHEMA,
        "method": "JointSVRA",
        "experiment_id": config["experiment_id"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "metrics": public_metrics(metrics),
        "comparisons": comparisons,
        "gates": gates,
        "precheck_passed": passed,
        "decision": "continue_joint_svra_formal" if passed else "drop_joint_svra_precheck_failed",
        "transitions_vs_parent": transitions,
        "group_statistics": group_stats,
        "group_safety_gates": {
            "leader_trigger_rate": float(group_safety["leader_trigger_rate"]),
            "challenger_trigger_rate": float(group_safety["challenger_trigger_rate"]),
            "leader_damage": int(group_stats["leader"]["damaged"]),
            "challenger_corrections": int(group_stats["challenger"]["corrected"]),
        },
        "actions": action,
        "condition_identity": condition_shas,
        "b0_receipt": {
            "raw_image_open_count": 0,
            "raw_crop_encode_count": 0,
            "eval_all25_opened": False,
            "official_feature_logits_frozen_before_labels": True,
            "labels_loaded_after_logits": True,
            "opened_keys": [
                "role_tensor",
                "name_tensor",
                "test_seen_cls",
                "test_seen_patches",
                "test_unseen_cls",
                "test_unseen_patches",
                "test_seen_labels_after_logits",
                "test_unseen_labels_after_logits",
            ],
        },
        "checkpoint_identity": {
            key: {
                "path": config[config_key]["path"],
                "sha256": config[config_key]["sha256"],
                "training_commit": config[config_key]["training_commit"],
                "train_config_sha256": config[config_key]["train_config_sha256"],
                "condition_id": checkpoints[key]["condition_id"],
            }
            for key, config_key in CONDITION_TO_CONFIG_KEY.items()
        },
        "official_protocol": {
            "test_used_for_selection": True,
            "test_used_for_hyperparameter_selection": False,
            "nested_official_test_selection": True,
            "strict_blind_claim": False,
            "official_test_loaded": True,
            "unseen_images_used_for_gradient": False,
            "pclr_online_inference": False,
        },
        "reproducibility": reproducibility,
    }
    output = prepare_output_dir(output_dir)
    atomic_write_json(output / ("result.json" if passed else "failure.json"), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()
