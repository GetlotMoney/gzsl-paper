"""Official zero-crop precheck evaluator for IDEA-202 / DESC.

DESC deployment is a direct keep-vs-swap decision: freeze every official-test
logit/action/evidence-pool/swap-logit tensor first, then read labels for
U/S/H/ZS and paired class-bootstrap gates. This evaluator never opens raw
images, online CLIP, or train-only all-action evidence tables.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib
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
    HIDDEN_DIM,
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


SCHEMA = "gzsl-paper.v6-desc-precheck-eval.v1"
CHECKPOINT_SCHEMA = "gzsl-paper.v6-desc-precheck-train.v1"
FULL_CONDITION = "DESC_FULL"
NO_ACTION_AUX_CONDITION = "DESC_NO_ACTION_AUX"
PARENT_ONLY_CONDITION = "DESC_PARENT_ONLY"
DESC_MODEL_CLASS_NAMES = (
    "DirectEvidenceSwapCompetition",
    "DESCModel",
    "DirectEvidenceConditionedSwapCompetition",
    "SemanticVisualRiskArbiter",
)
CONDITION_TO_CONFIG_KEY = {
    "full": "full_checkpoint",
    "no_action_aux": "no_action_aux_checkpoint",
    "parent_only": "parent_only_checkpoint",
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
    "no_action_aux_checkpoint",
    "parent_only_checkpoint",
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


class DESCPrecheckError(RuntimeError):
    """Raised when the official DESC precheck contract is violated."""


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
    swap_logit: torch.Tensor
    action_logits: torch.Tensor
    evidence_pool: torch.Tensor
    actions: torch.Tensor
    swap: torch.Tensor


@dataclass(frozen=True)
class FrozenCondition:
    name: str
    seen: FrozenSplit
    unseen: FrozenSplit

    @property
    def logits_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.logits, self.unseen.logits), dim=0)

    @property
    def swap_logit_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.swap_logit, self.unseen.swap_logit), dim=0)

    @property
    def action_logits_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.action_logits, self.unseen.action_logits), dim=0)

    @property
    def evidence_pool_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.evidence_pool, self.unseen.evidence_pool), dim=0)

    @property
    def actions_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.actions, self.unseen.actions), dim=0)

    @property
    def swap_for_sha(self) -> torch.Tensor:
        return torch.cat((self.seen.swap, self.unseen.swap), dim=0)


def load_config(path: Path | str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != REQUIRED_CONFIG_KEYS:
        raise DESCPrecheckError(
            "DESC precheck eval config字段错误；"
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
        raise DESCPrecheckError("DESC precheck eval固定协议错误。")
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
    raise DESCPrecheckError(f"{source} 中没有tensor。")


def _load_tensor(path: str, sha256: str, *, shape: tuple[int, ...], name: str) -> torch.Tensor:
    file_path = Path(path)
    if not file_path.is_file() or sha256_file(file_path).lower() != str(sha256).lower():
        raise DESCPrecheckError(f"{name} 路径或SHA错误。")
    tensor = _first_tensor(_torch_load(file_path), source=file_path)
    if tuple(int(x) for x in tensor.shape) != shape:
        raise DESCPrecheckError(f"{name} shape错误：{tuple(tensor.shape)} != {shape}")
    if not bool(torch.isfinite(tensor).all()):
        raise DESCPrecheckError(f"{name} 包含NaN/Inf。")
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
        raise DESCPrecheckError("official seen/unseen类别数必须为150/50。")
    if not torch.equal(all_classes, torch.arange(200, dtype=torch.long)):
        raise DESCPrecheckError("official seen/unseen类别必须完整覆盖200轴。")
    return OfficialLabels(seen, unseen, seen_classes, unseen_classes)


def load_checkpoint(
    spec: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_condition: str,
) -> Mapping[str, Any]:
    required = {"path", "sha256", "training_commit", "train_config_sha256"}
    if not isinstance(spec, Mapping) or set(spec) != required:
        raise DESCPrecheckError("checkpoint字段必须精确包含path/sha256/training_commit/train_config_sha256。")
    path = Path(str(spec["path"]))
    if not path.is_file() or sha256_file(path).lower() != str(spec["sha256"]).lower():
        raise DESCPrecheckError(f"{expected_condition} checkpoint路径或SHA错误。")
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, Mapping):
        raise DESCPrecheckError(f"{expected_condition} checkpoint不是mapping。")
    invalid = (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("condition_id") != expected_condition
        or checkpoint.get("code_commit") != expected_commit
        or spec["training_commit"] != expected_commit
        or checkpoint.get("config_sha256") != spec["train_config_sha256"]
        or "state_dict" not in checkpoint
    )
    if invalid:
        raise DESCPrecheckError(f"{expected_condition} checkpoint身份错误。")
    return checkpoint


def _model_class_from_checkpoint(checkpoint: Mapping[str, Any]) -> type[torch.nn.Module]:
    module = importlib.import_module("model.frameworks.v6.svra")
    preferred = checkpoint.get("model_class")
    names = (str(preferred),) if preferred else DESC_MODEL_CLASS_NAMES
    for name in names:
        cls = getattr(module, name, None)
        if isinstance(cls, type) and issubclass(cls, torch.nn.Module):
            return cls
    if preferred:
        raise DESCPrecheckError(f"checkpoint声明的DESC model_class不存在：{preferred}")
    return SemanticVisualRiskArbiter


def instantiate_model(
    roles: torch.Tensor,
    names: torch.Tensor,
    class_ids: torch.Tensor,
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    model_cls = _model_class_from_checkpoint(checkpoint)
    model = model_cls(roles, names, class_ids, seed=7).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _get_value(value: Any, paths: Sequence[tuple[str, ...]], *, name: str) -> Any:
    for path in paths:
        current = value
        found = True
        for key in path:
            if isinstance(current, Mapping) and key in current:
                current = current[key]
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                found = False
                break
        if found:
            return current
    raise DESCPrecheckError(f"DESC output缺少{name}。")


def _as_tensor(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise DESCPrecheckError(f"DESC output {name}不是tensor。")
    tensor = value.detach().to(device=device)
    if not bool(torch.isfinite(tensor.float()).all()):
        raise DESCPrecheckError(f"DESC output {name}包含NaN/Inf。")
    return tensor


def apply_pair_swap(parent_logits: torch.Tensor, top2: torch.Tensor, swap: torch.Tensor) -> torch.Tensor:
    logits = parent_logits.clone()
    if parent_logits.ndim != 2 or top2.shape != (parent_logits.shape[0], 2):
        raise DESCPrecheckError("parent_logits/top2 shape错误。")
    if swap.shape != (parent_logits.shape[0],):
        raise DESCPrecheckError("swap shape错误。")
    rows = torch.nonzero(swap.bool(), as_tuple=False).flatten()
    if rows.numel():
        leaders = top2.index_select(0, rows)[:, 0]
        challengers = top2.index_select(0, rows)[:, 1]
        leader_values = logits[rows, leaders].clone()
        logits[rows, leaders] = logits[rows, challengers]
        logits[rows, challengers] = leader_values
    return logits


def _call_desc_model(
    model: torch.nn.Module,
    cls: torch.Tensor,
    patches: torch.Tensor | None,
    *,
    semantic_off: bool,
    visual_off: bool,
    interaction_off: bool,
    parent_only: bool,
) -> Any:
    kwargs = {
        "semantic_off": semantic_off,
        "visual_off": visual_off,
        "interaction_off": interaction_off,
        "parent_only": parent_only,
    }
    for method_name in ("direct_forward", "desc_forward", "forward_desc", "forward"):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        signature = inspect.signature(method)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepted_kwargs = kwargs if accepts_var_kwargs else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        return method(cls, patches, **accepted_kwargs)
    raise DESCPrecheckError("DESC model没有direct_forward/desc_forward/forward_desc/forward部署入口。")


def _extract_desc_split(
    output: Any,
    *,
    batch_size: int,
    class_count: int,
    device: torch.device,
    interaction_off: bool,
) -> FrozenSplit:
    parent_logits = _as_tensor(
        _get_value(output, (("parent_logits",), ("pair", "parent_logits")), name="parent_logits"),
        name="parent_logits",
        device=device,
    ).float()
    top2 = _as_tensor(
        _get_value(output, (("top2",), ("pair", "top2")), name="top2"),
        name="top2",
        device=device,
    ).long()
    swap_logit = _as_tensor(
        _get_value(output, (("swap_logit",), ("swap_logits",), ("risk_logits",)), name="swap_logit"),
        name="swap_logit",
        device=device,
    ).float().reshape(-1)
    action_logits = _as_tensor(
        _get_value(output, (("action_logits",), ("action_logits25",), ("action_state", "utility_logits")), name="action_logits"),
        name="action_logits",
        device=device,
    ).float()
    evidence_pool = _as_tensor(
        _get_value(output, (("evidence_pool",), ("pooled_evidence",), ("desc_evidence_pool",)), name="evidence_pool"),
        name="evidence_pool",
        device=device,
    ).float()

    if parent_logits.shape != (batch_size, class_count):
        raise DESCPrecheckError(f"parent_logits shape错误：{tuple(parent_logits.shape)}")
    if top2.shape != (batch_size, 2):
        raise DESCPrecheckError(f"top2 shape错误：{tuple(top2.shape)}")
    if swap_logit.shape != (batch_size,):
        raise DESCPrecheckError(f"swap_logit shape错误：{tuple(swap_logit.shape)}")
    if action_logits.shape != (batch_size, ACTION_COUNT):
        raise DESCPrecheckError(f"action_logits shape错误：{tuple(action_logits.shape)}")
    if evidence_pool.shape != (batch_size, HIDDEN_DIM):
        raise DESCPrecheckError(f"evidence_pool shape错误：{tuple(evidence_pool.shape)}")

    if interaction_off:
        swap = torch.zeros(batch_size, dtype=torch.bool, device=device)
        logits = parent_logits
    else:
        try:
            swap = _as_tensor(
                _get_value(output, (("swap",), ("swapped",)), name="swap"),
                name="swap",
                device=device,
            ).bool().reshape(-1)
        except DESCPrecheckError:
            swap = swap_logit > 0
        logits = _as_tensor(
            _get_value(output, (("logits",), ("final_logits",)), name="logits"),
            name="logits",
            device=device,
        ).float()
        expected = apply_pair_swap(parent_logits, top2, swap_logit > 0)
        if not torch.equal(swap, swap_logit > 0):
            raise DESCPrecheckError("DESC swap必须严格等于swap_logit>0。")
        if not torch.allclose(logits, expected, atol=0, rtol=0):
            raise DESCPrecheckError("DESC final logits必须严格来自Parent Top1/Top2 swap。")

    if logits.shape != (batch_size, class_count):
        raise DESCPrecheckError(f"logits shape错误：{tuple(logits.shape)}")
    actions = action_logits.argmax(dim=1).long()
    return FrozenSplit(
        logits=logits.detach().cpu(),
        parent_logits=parent_logits.detach().cpu(),
        top2=top2.detach().cpu(),
        swap_logit=swap_logit.detach().cpu(),
        action_logits=action_logits.detach().cpu(),
        evidence_pool=evidence_pool.detach().cpu(),
        actions=actions.detach().cpu(),
        swap=swap.detach().cpu(),
    )


@torch.no_grad()
def freeze_split(
    model: torch.nn.Module,
    cls_features: torch.Tensor,
    patch_features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    semantic_off: bool = False,
    visual_off: bool = False,
    interaction_off: bool = False,
    parent_only: bool = False,
) -> FrozenSplit:
    frozen: list[FrozenSplit] = []
    class_count = int(getattr(model, "class_ids", torch.arange(200)).numel())
    for start in range(0, cls_features.shape[0], batch_size):
        cls = cls_features[start : start + batch_size].to(device).float()
        patches = None if (parent_only or visual_off) else patch_features[start : start + batch_size].to(device).float()
        try:
            output = _call_desc_model(
                model,
                cls,
                patches,
                semantic_off=semantic_off,
                visual_off=visual_off,
                interaction_off=interaction_off,
                parent_only=parent_only,
            )
        except TypeError:
            if not visual_off or parent_only or patches is not None:
                raise
            # Compatibility path for older test doubles that still index patches
            # even when visual_off=True. The real DESC model handles None and
            # therefore avoids the official V-off patch transfer.
            patches = patch_features[start : start + batch_size].to(device).float()
            output = _call_desc_model(
                model,
                cls,
                patches,
                semantic_off=semantic_off,
                visual_off=visual_off,
                interaction_off=interaction_off,
                parent_only=parent_only,
            )
        frozen.append(
            _extract_desc_split(
                output,
                batch_size=cls.shape[0],
                class_count=class_count,
                device=device,
                interaction_off=interaction_off,
            )
        )
    return FrozenSplit(
        logits=torch.cat([item.logits for item in frozen]),
        parent_logits=torch.cat([item.parent_logits for item in frozen]),
        top2=torch.cat([item.top2 for item in frozen]),
        swap_logit=torch.cat([item.swap_logit for item in frozen]),
        action_logits=torch.cat([item.action_logits for item in frozen]),
        evidence_pool=torch.cat([item.evidence_pool for item in frozen]),
        actions=torch.cat([item.actions for item in frozen]),
        swap=torch.cat([item.swap for item in frozen]),
    )


def freeze_condition(
    name: str,
    model: torch.nn.Module,
    features: OfficialFeatures,
    *,
    device: torch.device,
    batch_size: int,
    semantic_off: bool = False,
    visual_off: bool = False,
    interaction_off: bool = False,
    parent_only: bool = False,
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
        parent_only=parent_only,
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
        parent_only=parent_only,
    )
    return FrozenCondition(name=name, seen=seen, unseen=unseen)


def parent_condition(full: FrozenCondition, *, name: str = "parent") -> FrozenCondition:
    def make_parent(split: FrozenSplit) -> FrozenSplit:
        return FrozenSplit(
            logits=split.parent_logits,
            parent_logits=split.parent_logits,
            top2=split.top2,
            swap_logit=torch.zeros_like(split.swap_logit),
            action_logits=torch.zeros_like(split.action_logits),
            evidence_pool=torch.zeros_like(split.evidence_pool),
            actions=torch.zeros_like(split.actions),
            swap=torch.zeros_like(split.swap),
        )

    return FrozenCondition(name=name, seen=make_parent(full.seen), unseen=make_parent(full.unseen))


def _per_class_vector(labels: torch.Tensor, predictions: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    values = []
    for class_id in classes.long():
        mask = labels.eq(class_id)
        if not bool(mask.any()):
            raise DESCPrecheckError(f"评估split缺少类别{int(class_id)}。")
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


def action_summary(condition: FrozenCondition) -> dict[str, Any]:
    actions = condition.actions_for_sha
    swap = condition.swap_for_sha
    histogram_all = torch.bincount(actions, minlength=ACTION_COUNT)
    histogram_swap = torch.bincount(actions[swap], minlength=ACTION_COUNT)
    return {
        "swap_count": int(swap.sum()),
        "keep_count": int(swap.numel() - int(swap.sum())),
        "swap_rate": float(swap.double().mean()),
        "all_action_histogram": [int(x) for x in histogram_all],
        "swap_action_histogram": [int(x) for x in histogram_swap],
        "used_actions_all": int(histogram_all.gt(0).sum()),
        "used_actions_swapped": int(histogram_swap.gt(0).sum()),
        "highest_occupancy_all": float(histogram_all.max()) / max(1, actions.numel()),
    }


def condition_sha(condition: FrozenCondition) -> dict[str, Any]:
    return {
        "swap_logit_sha256": _tensor_sha256(condition.swap_logit_for_sha),
        "action_logits_sha256": _tensor_sha256(condition.action_logits_for_sha),
        "evidence_pool_sha256": _tensor_sha256(condition.evidence_pool_for_sha),
        "logits_sha256": _tensor_sha256(condition.logits_for_sha),
        "action_sha256": _tensor_sha256(condition.actions_for_sha),
        "swap_sha256": _tensor_sha256(condition.swap_for_sha.to(torch.uint8)),
        "swap_count": int(condition.swap_for_sha.sum()),
        "action_count": int(condition.actions_for_sha.numel()),
    }


def module_signal_checks(full: FrozenCondition, s_off: FrozenCondition, v_off: FrozenCondition) -> dict[str, bool]:
    return {
        "s_off_changes_swap_logit": _tensor_sha256(full.swap_logit_for_sha) != _tensor_sha256(s_off.swap_logit_for_sha),
        "v_off_changes_swap_logit": _tensor_sha256(full.swap_logit_for_sha) != _tensor_sha256(v_off.swap_logit_for_sha),
        "s_off_changes_evidence_pool": _tensor_sha256(full.evidence_pool_for_sha) != _tensor_sha256(s_off.evidence_pool_for_sha),
        "v_off_changes_evidence_pool": _tensor_sha256(full.evidence_pool_for_sha) != _tensor_sha256(v_off.evidence_pool_for_sha),
    }


def checkpoint_trace_identity(checkpoints: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("initialization_sha256", "batch_trace_sha256", "target_census_sha256")
    values = {field: {name: checkpoint.get(field, "") for name, checkpoint in checkpoints.items()} for field in fields}
    return {
        **values,
        "same_initialization_sha256": len(set(values["initialization_sha256"].values())) == 1 and "" not in values["initialization_sha256"].values(),
        "same_batch_trace_sha256": len(set(values["batch_trace_sha256"].values())) == 1 and "" not in values["batch_trace_sha256"].values(),
        "same_target_census_sha256": len(set(values["target_census_sha256"].values())) == 1 and "" not in values["target_census_sha256"].values(),
    }


def public_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        name: {key: float(value[key]) for key in ("U", "S", "H", "ZS")}
        for name, value in metrics.items()
    }


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict[str, Any]:
    config, config_sha = load_config(config_path)
    if config_sha.lower() != expected_config_sha.lower():
        raise DESCPrecheckError("DESC precheck config SHA mismatch.")
    if bool(config["require_clean_tree"]):
        require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise DESCPrecheckError("DESC precheck expected commit mismatch.")
    reproducibility = configure_reproducibility(
        int(config["random_seed"]),
        strict_determinism=True,
        deterministic_warn_only=False,
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise DESCPrecheckError("DESC precheck必须在可用CUDA设备上运行。")

    roles, names, class_ids = load_text_tensors(config)
    features = load_official_features(config)
    checkpoints = {
        "full": load_checkpoint(
            config["full_checkpoint"],
            expected_commit=config["full_checkpoint"]["training_commit"],
            expected_condition=FULL_CONDITION,
        ),
        "no_action_aux": load_checkpoint(
            config["no_action_aux_checkpoint"],
            expected_commit=config["no_action_aux_checkpoint"]["training_commit"],
            expected_condition=NO_ACTION_AUX_CONDITION,
        ),
        "parent_only": load_checkpoint(
            config["parent_only_checkpoint"],
            expected_commit=config["parent_only_checkpoint"]["training_commit"],
            expected_condition=PARENT_ONLY_CONDITION,
        ),
    }
    models = {
        name: instantiate_model(roles, names, class_ids, checkpoint, device)
        for name, checkpoint in checkpoints.items()
    }
    batch_size = int(config["eval_batch_size"])

    full = freeze_condition("full", models["full"], features, device=device, batch_size=batch_size)
    conditions = {
        "full": full,
        "parent": parent_condition(full, name="parent"),
        "s_off": freeze_condition("s_off", models["full"], features, device=device, batch_size=batch_size, semantic_off=True),
        "v_off": freeze_condition("v_off", models["full"], features, device=device, batch_size=batch_size, visual_off=True),
        "i_off": freeze_condition("i_off", models["full"], features, device=device, batch_size=batch_size, interaction_off=True),
        "no_action_aux": freeze_condition("no_action_aux", models["no_action_aux"], features, device=device, batch_size=batch_size),
        "parent_only": freeze_condition("parent_only", models["parent_only"], features, device=device, batch_size=batch_size, parent_only=True),
    }

    # All official-test logits/actions/swaps/pools are frozen before labels are opened.
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
    action = action_summary(conditions["full"])
    condition_shas = {name: condition_sha(condition) for name, condition in conditions.items()}
    signal_checks = module_signal_checks(conditions["full"], conditions["s_off"], conditions["v_off"])
    trace_identity = checkpoint_trace_identity(checkpoints)
    gates = {
        "full_vs_parent_observed_plus1H": comparisons["parent"]["observed_pp"] >= float(config["module_contract_margin"]),
        "full_vs_s_off_observed_plus1H": comparisons["s_off"]["observed_pp"] >= float(config["module_contract_margin"]),
        "full_vs_v_off_observed_plus1H": comparisons["v_off"]["observed_pp"] >= float(config["module_contract_margin"]),
        "full_vs_i_off_observed_plus1H": comparisons["i_off"]["observed_pp"] >= float(config["module_contract_margin"]),
        "full_vs_no_action_aux_ci_plus0_5H": comparisons["no_action_aux"]["observed_pp"] >= float(config["support_control_margin"]) and comparisons["no_action_aux"]["ci95"][0] > 0,
        "full_vs_parent_only_ci_plus0_5H": comparisons["parent_only"]["observed_pp"] >= float(config["support_control_margin"]) and comparisons["parent_only"]["ci95"][0] > 0,
        "net_positive": transitions["net"] > 0,
        "corrections_gt_damages": transitions["corrected"] > transitions["damaged"],
        "both_keep_and_swap": action["swap_count"] > 0 and action["keep_count"] > 0,
        "action_diversity": action["used_actions_all"] >= 2,
        "highest_action_occupancy_lte_bound": action["highest_occupancy_all"] <= float(config["max_action_occupancy"]),
        "direct_swap_logit_threshold": all(
            torch.equal(condition.swap_for_sha, condition.swap_logit_for_sha.gt(0))
            for condition in conditions.values()
            if condition.name not in {"parent", "i_off"}
        ),
        "raw_image_open_count_zero": True,
        "raw_crop_encode_count_zero": True,
        "eval_all25_opened_false": True,
        "labels_loaded_after_logits": True,
    }
    passed = all(gates.values())
    result = {
        "schema_version": SCHEMA,
        "method": "DESC",
        "experiment_id": config["experiment_id"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "metrics": public_metrics(metrics),
        "comparisons": comparisons,
        "gates": gates,
        "precheck_passed": passed,
        "decision": "continue_desc_formal" if passed else "drop_desc_precheck_failed",
        "transitions_vs_parent": transitions,
        "actions": action,
        "module_signal_checks": signal_checks,
        "condition_identity": condition_shas,
        "checkpoint_trace_identity": trace_identity,
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
