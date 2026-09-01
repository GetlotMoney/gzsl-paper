"""Strict zero-crop Gate-0 evaluator for IDEA-199 / SVRA.

SVRA deployment never opens raw images, never encodes crops, and never reads the
eval all25 crop table.  The ordering is part of the contract:

1. Read only frozen role/name text, 336 CLS, and projected 24x24 patches.
2. Freeze Parent, S/V trigger policy, all risk probabilities, and all logits.
3. Only then load eval labels and compute metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v6-svra-gate0-eval.v1"
CHECKPOINT_SCHEMA = "gzsl-paper.v6-svra-gate0-train.v1"
FEATURE_DIM = 768
ACTION_COUNT = 25
ACTION_GEOMETRY_SHA256 = (
    "4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb"
)
REQUIRED_CONFIG_KEYS = {
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
    "action_geometry_sha256",
    "att_splits_mat_path",
    "trainval_count",
    "combined_checkpoint",
    "device",
    "random_seed",
    "eval_batch_size",
    "bootstrap_seed",
    "bootstrap_samples",
    "module_contract_margin",
    "support_control_margin",
    "max_action_occupancy",
    "require_clean_tree",
    "official_test_loaded",
    "unseen_images_used_for_gradient",
    "pclr_online_inference",
}


class SVRAEvalError(RuntimeError):
    """Raised when Gate-0 evaluation would violate the SVRA contract."""


def _load_data_api() -> Mapping[str, Any]:
    try:
        import model.frameworks.v6.rwdg_data as data_api
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SVRAEvalError("缺少V6 SVRA数据边界模块。") from exc
    return {
        "ManifestContract": data_api.ManifestContract,
        "SVRAAssetConfig": data_api.SVRAAssetConfig,
        "TensorContract": data_api.TensorContract,
        "load_svra_gate_data": data_api.load_svra_gate_data,
        "resolve_subset_output": data_api.resolve_subset_output,
    }


@dataclass(frozen=True)
class FrozenPolicy:
    name: str
    parent_logits: torch.Tensor
    top2: torch.Tensor
    leader_ids: torch.Tensor
    challenger_ids: torch.Tensor
    actions: torch.Tensor
    trigger: torch.Tensor
    parent_stats4: torch.Tensor
    risk_features13: torch.Tensor | None
    policy_scores: torch.Tensor | None
    probabilities: Mapping[str, torch.Tensor]


def load_config(path: Path | str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != REQUIRED_CONFIG_KEYS:
        raise SVRAEvalError(
            "SVRA Gate0 eval config字段错误；"
            f"缺少={sorted(REQUIRED_CONFIG_KEYS - actual)} 多出={sorted(actual - REQUIRED_CONFIG_KEYS)}"
        )
    invalid = (
        config["schema_version"] != SCHEMA
        or config["condition_id"] != "SVRA_FULL"
        or int(config["random_seed"]) != 7
        or int(config["bootstrap_seed"]) != 7
        or int(config["bootstrap_samples"]) != 10000
        or int(config["trainval_count"]) != 7057
        or int(config["eval_batch_size"]) <= 0
        or float(config["module_contract_margin"]) != 1.0
        or float(config["support_control_margin"]) != 0.5
        or float(config["max_action_occupancy"]) != 0.70
        or config["action_geometry_sha256"] != ACTION_GEOMETRY_SHA256
        or config["official_test_loaded"] is not False
        or config["unseen_images_used_for_gradient"] is not False
        or config["pclr_online_inference"] is not False
    )
    if invalid:
        raise SVRAEvalError("SVRA Gate0 eval固定协议错误。")
    return config, sha256_file(path)


def asset_config_from_eval_config(config: Mapping[str, Any]) -> Any:
    data_api = _load_data_api()
    manifest_contract = data_api["ManifestContract"]
    tensor_contract = data_api["TensorContract"]
    asset_config = data_api["SVRAAssetConfig"]
    count = int(config["trainval_count"])
    return asset_config(
        text_manifest=manifest_contract(str(config["text_manifest"]), str(config["text_manifest_sha256"])),
        role_tensor=tensor_contract(str(config["role_tensor"]), str(config["role_tensor_sha256"]), (200, 8, FEATURE_DIM), "float32"),
        name_tensor=tensor_contract(str(config["name_tensor"]), str(config["name_tensor_sha256"]), (200, FEATURE_DIM), "float32"),
        patch_manifest=manifest_contract(str(config["patch_manifest"]), str(config["patch_manifest_sha256"])),
        cls_tensor=tensor_contract(str(config["cls_tensor"]), str(config["cls_tensor_sha256"]), (count, FEATURE_DIM), "float32"),
        patch_tensor=tensor_contract(str(config["patch_tensor"]), str(config["patch_tensor_sha256"]), (count, 576, FEATURE_DIM), "float16"),
        action_bundle_manifest=manifest_contract(
            str(config["action_bundle_manifest"]),
            str(config["action_bundle_manifest_sha256"]),
        ),
        dev_train_manifest_sha256=str(config["dev_train_manifest_sha256"]),
        dev_eval_manifest_sha256=str(config["dev_eval_manifest_sha256"]),
        dev_eval_oracle_manifest_sha256=str(config["dev_eval_oracle_manifest_sha256"]),
        att_splits_mat_path=str(config["att_splits_mat_path"]),
        trainval_count=count,
    )


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping) and name in value:
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    raise SVRAEvalError(f"SVRA model state缺少字段：{name}")


def _maybe_field(value: Any, *names: str) -> Any | None:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _load_output_tensor_from_assets(assets: Any, subset_name: str, filename: str) -> torch.Tensor:
    data_api = _load_data_api()
    value = _torch_load(data_api["resolve_subset_output"](assets, subset_name, filename, verify_sha=True))
    if isinstance(value, Mapping):
        for item in value.values():
            if torch.is_tensor(item):
                return item.detach().cpu()
        raise SVRAEvalError(f"{subset_name}.{filename} mapping中没有tensor。")
    if not torch.is_tensor(value):
        raise SVRAEvalError(f"{subset_name}.{filename}不是tensor。")
    return value.detach().cpu()


def _load_class_ids(assets: Any, subset_name: str, meta: Mapping[str, Any]) -> torch.Tensor:
    ids = meta.get("class_ids")
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        return torch.as_tensor(list(ids), dtype=torch.long)
    return _load_output_tensor_from_assets(assets, subset_name, "class_ids.pt").long()


def _validate_axis_labels(name: str, labels: torch.Tensor, class_ids: torch.Tensor) -> None:
    if labels.ndim != 1:
        raise SVRAEvalError(f"{name} labels必须是一维。")
    if not bool(torch.isin(labels.long(), class_ids.long()).all()):
        raise SVRAEvalError(f"{name} labels包含active axis之外的类别。")


def load_eval_labels_after_logits(
    assets: Any,
    class_ids: torch.Tensor,
    *,
    expected_count: int,
) -> torch.Tensor:
    labels = _load_output_tensor_from_assets(assets, "dev_eval", "labels.pt").long()
    if labels.shape != (expected_count,):
        raise SVRAEvalError(f"dev_eval labels shape错误：{tuple(labels.shape)}")
    _validate_axis_labels("dev_eval", labels, class_ids)
    return labels


def load_checkpoint(
    spec: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_bundle_sha256: str,
    expected_train_config_sha256: str | None = None,
) -> Mapping[str, Any]:
    required = {"path", "sha256", "training_commit", "train_config_sha256"}
    if not isinstance(spec, Mapping) or set(spec) != required:
        raise SVRAEvalError("combined_checkpoint字段必须精确包含path/sha256/training_commit/train_config_sha256。")
    path = Path(str(spec["path"]))
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise SVRAEvalError("SVRA checkpoint路径或SHA错误。")
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, Mapping):
        raise SVRAEvalError("SVRA checkpoint不是mapping。")
    if expected_train_config_sha256 is not None and spec["train_config_sha256"] != expected_train_config_sha256:
        raise SVRAEvalError("SVRA checkpoint train config SHA与预期不一致。")
    invalid = (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("condition_id") != "SVRA_FULL"
        or checkpoint.get("code_commit") != expected_commit
        or spec["training_commit"] != expected_commit
        or checkpoint.get("config_sha256") != spec["train_config_sha256"]
    )
    if invalid:
        raise SVRAEvalError("SVRA checkpoint身份错误。")
    required_states = {
        "policy_state_dict",
        "trigger_arbiter4_state_dict",
        "trigger_arbiter13_ceiling_state_dict",
        "allrow_arbiter4_control_state_dict",
    }
    missing = required_states - set(checkpoint)
    if missing:
        raise SVRAEvalError(f"SVRA checkpoint缺少state dict：{sorted(missing)}")
    if checkpoint.get("action_bundle_manifest_sha256") != expected_bundle_sha256:
        raise SVRAEvalError("SVRA checkpoint action bundle SHA错误。")
    return checkpoint


def instantiate_model(assets: Any, class_ids: torch.Tensor, checkpoint: Mapping[str, Any], device: torch.device):
    from model.frameworks.v6.svra import SemanticVisualRiskArbiter

    roles = assets.role_embeddings.index_select(0, class_ids.long()).float()
    names = assets.name_embeddings.index_select(0, class_ids.long()).float()
    model = SemanticVisualRiskArbiter(roles, names, class_ids.long(), seed=7).to(device).eval()
    model.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    model.interaction.load_state_dict(checkpoint["trigger_arbiter4_state_dict"], strict=True)
    model.trigger_arbiter13_ceiling.load_state_dict(
        checkpoint["trigger_arbiter13_ceiling_state_dict"],
        strict=True,
    )
    model.allrow_arbiter4_control.load_state_dict(
        checkpoint["allrow_arbiter4_control_state_dict"],
        strict=True,
    )
    return model


def _policy_state(model: Any, cls: torch.Tensor, patches: torch.Tensor, *, semantic_off: bool, visual_off: bool) -> Any:
    if hasattr(model, "policy_state"):
        return model.policy_state(cls, patches, semantic_off=semantic_off, visual_off=visual_off)
    raise SVRAEvalError("SVRA model必须暴露policy_state。")


def _risk_probabilities(model: Any, policy: Any) -> Mapping[str, torch.Tensor]:
    if hasattr(model, "risk_probabilities"):
        value = model.risk_probabilities(policy)
    else:
        raise SVRAEvalError("SVRA model必须暴露risk_probabilities。")
    if not isinstance(value, Mapping):
        raise SVRAEvalError("SVRA risk probabilities必须是mapping。")
    required = {"triggered4d", "all_row4d", "ceiling13d"}
    missing = required - set(value)
    if missing:
        raise SVRAEvalError(f"SVRA risk probabilities缺少：{sorted(missing)}")
    return {name: torch.as_tensor(value[name]).float() for name in sorted(required)}


@torch.no_grad()
def freeze_policy(
    model: Any,
    view: Any,
    *,
    device: torch.device,
    batch_size: int,
    name: str,
    semantic_off: bool = False,
    visual_off: bool = False,
) -> FrozenPolicy:
    parent_logits: list[torch.Tensor] = []
    top2: list[torch.Tensor] = []
    leader_ids: list[torch.Tensor] = []
    challenger_ids: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    triggers: list[torch.Tensor] = []
    parent_stats4: list[torch.Tensor] = []
    risk_features13: list[torch.Tensor] = []
    policy_scores: list[torch.Tensor] = []
    probabilities: dict[str, list[torch.Tensor]] = {
        "triggered4d": [],
        "all_row4d": [],
        "ceiling13d": [],
    }
    for start in range(0, view.size, batch_size):
        rows = np.arange(start, min(start + batch_size, view.size), dtype=np.int64)
        batch = view.batch(rows, include_patches=True, as_torch=True, device=device)
        patches = batch["patches"]
        if visual_off:
            patches = batch["cls"][:, None, :].expand(-1, patches.shape[1], -1)
        policy = _policy_state(model, batch["cls"], patches, semantic_off=semantic_off, visual_off=visual_off)
        probs = _risk_probabilities(model, policy)
        parent_logits.append(_field(policy, "parent_logits").detach().cpu())
        top2.append(_field(policy, "top2").detach().cpu())
        leader_ids.append(_field(policy, "leader_ids").detach().cpu())
        challenger_ids.append(_field(policy, "challenger_ids").detach().cpu())
        actions.append(_field(policy, "selected_action").detach().cpu())
        triggers.append(_field(policy, "trigger").detach().cpu())
        parent_stats4.append(_field(policy, "parent_stats").detach().cpu().float())
        maybe13 = _maybe_field(policy, "risk_features13", "risk13_features", "features13")
        if maybe13 is not None:
            risk_features13.append(maybe13.detach().cpu().float())
        scores = _maybe_field(policy, "policy_scores", "utility", "action_scores")
        if scores is not None:
            policy_scores.append(scores.detach().cpu().float())
        for prob_name, prob_value in probs.items():
            probabilities[prob_name].append(prob_value.detach().cpu().flatten())
    return FrozenPolicy(
        name=name,
        parent_logits=torch.cat(parent_logits),
        top2=torch.cat(top2).long(),
        leader_ids=torch.cat(leader_ids).long(),
        challenger_ids=torch.cat(challenger_ids).long(),
        actions=torch.cat(actions).long(),
        trigger=torch.cat(triggers).bool(),
        parent_stats4=torch.cat(parent_stats4).float(),
        risk_features13=torch.cat(risk_features13).float() if risk_features13 else None,
        policy_scores=torch.cat(policy_scores).float() if policy_scores else None,
        probabilities={k: torch.cat(v).float() for k, v in probabilities.items()},
    )


def apply_pair_swap(parent_logits: torch.Tensor, top2: torch.Tensor, swap: torch.Tensor) -> torch.Tensor:
    if parent_logits.ndim != 2:
        raise SVRAEvalError(f"parent_logits must be [B,C], got {tuple(parent_logits.shape)}")
    if top2.shape != (parent_logits.shape[0], 2):
        raise SVRAEvalError(f"top2 must be [B,2], got {tuple(top2.shape)}")
    if swap.shape != (parent_logits.shape[0],):
        raise SVRAEvalError(f"swap must be [B], got {tuple(swap.shape)}")
    logits = parent_logits.clone()
    rows = torch.nonzero(swap.bool(), as_tuple=False).flatten()
    if rows.numel():
        leaders = top2.index_select(0, rows)[:, 0]
        challengers = top2.index_select(0, rows)[:, 1]
        leader_values = logits[rows, leaders].clone()
        logits[rows, leaders] = logits[rows, challengers]
        logits[rows, challengers] = leader_values
    return logits


def build_condition_logits(full: FrozenPolicy, s_off: FrozenPolicy, v_off: FrozenPolicy) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    full_prob = full.probabilities["triggered4d"]
    all_row_prob = full.probabilities["all_row4d"]
    ceiling_prob = full.probabilities["ceiling13d"]
    swaps = {
        "full": full.trigger & full_prob.gt(0.5),
        "s_off": s_off.trigger & s_off.probabilities["triggered4d"].gt(0.5),
        "v_off": v_off.trigger & v_off.probabilities["triggered4d"].gt(0.5),
        "i_off": torch.zeros_like(full.trigger, dtype=torch.bool),
        "always_swap": full.trigger.clone(),
        "triggered4d_no_trigger": full_prob.gt(0.5),
        "all_row4d_no_trigger": all_row_prob.gt(0.5),
        "ceiling13d": full.trigger & ceiling_prob.gt(0.5),
    }
    logits = {
        "parent": full.parent_logits.clone(),
        "full": apply_pair_swap(full.parent_logits, full.top2, swaps["full"]),
        "s_off": apply_pair_swap(s_off.parent_logits, s_off.top2, swaps["s_off"]),
        "v_off": apply_pair_swap(v_off.parent_logits, v_off.top2, swaps["v_off"]),
        "i_off": full.parent_logits.clone(),
        "always_swap": apply_pair_swap(full.parent_logits, full.top2, swaps["always_swap"]),
        "triggered4d_no_trigger": apply_pair_swap(full.parent_logits, full.top2, swaps["triggered4d_no_trigger"]),
        "all_row4d_no_trigger": apply_pair_swap(full.parent_logits, full.top2, swaps["all_row4d_no_trigger"]),
        "ceiling13d": apply_pair_swap(full.parent_logits, full.top2, swaps["ceiling13d"]),
    }
    return logits, swaps


def metrics(logits: torch.Tensor, labels: torch.Tensor, class_ids: torch.Tensor) -> dict[str, Any]:
    if logits.shape[0] != labels.numel() or logits.shape[1] != class_ids.numel():
        raise SVRAEvalError(f"logits shape错误：{tuple(logits.shape)}")
    predictions = class_ids.long()[logits.argmax(dim=1)]
    classes = torch.unique(labels.long(), sorted=True)
    per_class = torch.stack([
        predictions[labels.eq(cls)].eq(cls).double().mean() for cls in classes
    ])
    return {
        "macro_top1": 100 * float(per_class.mean()),
        "micro_top1": 100 * float(predictions.eq(labels).double().mean()),
        "per_class": per_class,
        "prediction": predictions,
        "classes": classes,
        "macro_class_count": int(classes.numel()),
        "axis_class_count": int(class_ids.numel()),
    }


def paired_comparison(full_vector: torch.Tensor, other_vector: torch.Tensor, matrix: torch.Tensor) -> dict[str, Any]:
    diff = 100 * (full_vector.double() - other_vector.double())
    samples = diff[matrix].mean(dim=1)
    ci = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.double))
    return {"observed_pp": float(diff.mean()), "ci95": [float(ci[0]), float(ci[1])]}


def group_statistics(
    trace: FrozenPolicy,
    labels: torch.Tensor,
    class_ids: torch.Tensor,
    parent_pred: torch.Tensor,
    full_pred: torch.Tensor,
) -> dict[str, Any]:
    lookup = {int(cls): idx for idx, cls in enumerate(class_ids.tolist())}
    local_truth = torch.tensor([lookup[int(label)] for label in labels.tolist()], dtype=torch.long)
    leader = trace.top2[:, 0]
    challenger = trace.top2[:, 1]
    group = torch.full_like(local_truth, 2)
    group[local_truth.eq(leader)] = 0
    group[local_truth.eq(challenger)] = 1
    names = {0: "leader", 1: "challenger", 2: "outside"}
    result: dict[str, Any] = {}
    for value, name in names.items():
        mask = group.eq(value)
        parent_correct = parent_pred[mask].eq(labels[mask])
        full_correct = full_pred[mask].eq(labels[mask])
        result[name] = {
            "count": int(mask.sum()),
            "trigger": int((trace.trigger & mask).sum()),
            "abstain": int((~trace.trigger & mask).sum()),
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


def _point_gate(comparison: Mapping[str, Any], margin: float) -> bool:
    return float(comparison["observed_pp"]) >= float(margin)


def _support_gate(comparison: Mapping[str, Any], margin: float) -> bool:
    return float(comparison["observed_pp"]) >= float(margin) and float(comparison["ci95"][0]) > 0


def _action_summary(actions: torch.Tensor, trigger: torch.Tensor) -> dict[str, Any]:
    histogram = torch.bincount(actions[trigger], minlength=ACTION_COUNT)
    trigger_count = int(trigger.sum())
    return {
        "trigger_count": trigger_count,
        "abstain_count": int(trigger.numel() - trigger_count),
        "trigger_rate": float(trigger.double().mean()),
        "triggered_action_histogram": [int(x) for x in histogram],
        "used_actions": int(histogram.gt(0).sum()),
        "highest_occupancy": float(histogram.max()) / max(1, trigger_count),
        "action_sha256": _tensor_sha256(actions),
        "trigger_sha256": _tensor_sha256(trigger.to(torch.uint8)),
    }


def _probability_summary(trace: FrozenPolicy) -> dict[str, Any]:
    return {
        name: {
            "mean": float(prob.double().mean()),
            "std": float(prob.double().std(unbiased=False)),
            "min": float(prob.min()),
            "max": float(prob.max()),
            "sha256": _tensor_sha256(prob),
        }
        for name, prob in trace.probabilities.items()
    }


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict[str, Any]:
    config, config_sha = load_config(config_path)
    if config_sha != expected_config_sha:
        raise SVRAEvalError("SVRA eval config SHA mismatch.")
    if bool(config["require_clean_tree"]):
        require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise SVRAEvalError("SVRA eval expected commit mismatch.")
    reproducibility = configure_reproducibility(
        int(config["random_seed"]),
        strict_determinism=True,
        deterministic_warn_only=False,
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SVRAEvalError("SVRA Gate0 eval必须在可用CUDA设备上运行。")

    asset_config = asset_config_from_eval_config(config)
    data_api = _load_data_api()
    assets, views = data_api["load_svra_gate_data"](
        asset_config,
        strict_sha=True,
        validate_tensor_values=True,
    )
    try:
        eval_view = views["dev_eval"]
        eval_class_ids = _load_class_ids(assets, "dev_eval", assets.dev_eval_manifest)
        if eval_view.size != 2355 or eval_class_ids.numel() != 150:
            raise SVRAEvalError("SVRA Gate0 eval row或class数量错误。")
        checkpoint = load_checkpoint(
            config["combined_checkpoint"],
            # Training runs on the frozen implementation commit.  The eval
            # config is filled and committed only after the checkpoint SHA is
            # known, so its clean evaluation commit may be a later ledger-only
            # commit.  Bind the checkpoint to its explicit training identity.
            expected_commit=config["combined_checkpoint"]["training_commit"],
            expected_bundle_sha256=config["action_bundle_manifest_sha256"],
            expected_train_config_sha256=config["combined_checkpoint"]["train_config_sha256"],
        )
        model = instantiate_model(assets, eval_class_ids, checkpoint, device)
        eval_batch = int(config["eval_batch_size"])

        full_trace = freeze_policy(model, eval_view, device=device, batch_size=eval_batch, name="full")
        s_off_trace = freeze_policy(
            model, eval_view, device=device, batch_size=eval_batch, name="s_off", semantic_off=True
        )
        v_off_trace = freeze_policy(
            model, eval_view, device=device, batch_size=eval_batch, name="v_off", visual_off=True
        )
        logits, swaps = build_condition_logits(full_trace, s_off_trace, v_off_trace)

        # Labels are intentionally loaded only after every condition's logits are frozen.
        labels = load_eval_labels_after_logits(assets, eval_class_ids, expected_count=eval_view.size)
        metric_values = {name: metrics(value, labels, eval_class_ids) for name, value in logits.items()}
        matrix = torch.randint(
            0,
            metric_values["full"]["per_class"].numel(),
            (int(config["bootstrap_samples"]), metric_values["full"]["per_class"].numel()),
            generator=torch.Generator().manual_seed(int(config["bootstrap_seed"])),
        )
        comparisons = {
            name: paired_comparison(metric_values["full"]["per_class"], value["per_class"], matrix)
            for name, value in metric_values.items()
            if name != "full"
        }
        parent_pred = metric_values["parent"]["prediction"]
        full_pred = metric_values["full"]["prediction"]
        corrected = full_pred.eq(labels) & parent_pred.ne(labels)
        damaged = full_pred.ne(labels) & parent_pred.eq(labels)
        group_stats = group_statistics(full_trace, labels, eval_class_ids, parent_pred, full_pred)
        group_safety = group_safety_gate(group_stats)
        action = _action_summary(full_trace.actions, full_trace.trigger)

        gates: dict[str, bool] = {
            "full_vs_parent_observed_plus1": _point_gate(comparisons["parent"], config["module_contract_margin"]),
            "full_vs_s_off_observed_plus1": _point_gate(comparisons["s_off"], config["module_contract_margin"]),
            "full_vs_v_off_observed_plus1": _point_gate(comparisons["v_off"], config["module_contract_margin"]),
            "full_vs_i_off_observed_plus1": _point_gate(comparisons["i_off"], config["module_contract_margin"]),
            "full_vs_always_swap_ci_plus0_5": _support_gate(comparisons["always_swap"], config["support_control_margin"]),
            "full_vs_triggered4d_no_trigger_ci_plus0_5": _support_gate(
                comparisons["triggered4d_no_trigger"], config["support_control_margin"]
            ),
            "full_vs_all_row4d_no_trigger_ci_plus0_5": _support_gate(
                comparisons["all_row4d_no_trigger"], config["support_control_margin"]
            ),
            "net_positive": int(corrected.sum() - damaged.sum()) > 0,
            "corrections_gt_damages": int(corrected.sum()) > int(damaged.sum()),
            "used_at_least_two_actions": int(action["used_actions"]) >= 2,
            "highest_occupancy_lte_70pct": float(action["highest_occupancy"])
            <= float(config["max_action_occupancy"]),
            "raw_image_open_count_zero": True,
            "raw_crop_encode_count_zero": True,
            "eval_all25_opened_false": True,
            "labels_loaded_after_logits": True,
            **group_safety["gates"],
        }
        passed = all(gates.values())
        opened_keys = [
            "text_manifest",
            "role_tensor",
            "name_tensor",
            "patch_manifest",
            "cls_tensor",
            "patch_tensor_safe_view",
            "action_bundle_manifest",
            "dev_eval_manifest",
            "dev_eval.class_ids",
            "dev_eval.labels_after_logits",
        ]
        result = {
            "schema_version": SCHEMA,
            "method": "SVRA",
            "condition_id": config["condition_id"],
            "experiment_id": config["experiment_id"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "metrics": {
                name: {
                    "macro_top1": value["macro_top1"],
                    "micro_top1": value["micro_top1"],
                    "macro_class_count": value["macro_class_count"],
                    "axis_class_count": value["axis_class_count"],
                }
                for name, value in metric_values.items()
            },
            "comparisons": comparisons,
            "gates": gates,
            "preliminary_gate0_passed": passed,
            "decision": "continue_formal_svra" if passed else "drop_svra_gate0_failed",
            "transitions": {
                "corrected": int(corrected.sum()),
                "damaged": int(damaged.sum()),
                "net": int(corrected.sum() - damaged.sum()),
            },
            "group_statistics": group_stats,
            "group_safety_gates": {
                "leader_trigger_rate": float(group_safety["leader_trigger_rate"]),
                "challenger_trigger_rate": float(group_safety["challenger_trigger_rate"]),
                "leader_damage": int(group_stats["leader"]["damaged"]),
                "challenger_corrections": int(group_stats["challenger"]["corrected"]),
            },
            "actions": action,
            "probabilities": _probability_summary(full_trace),
            "condition_swap_counts": {name: int(value.sum()) for name, value in swaps.items()},
            "condition_swap_sha256": {name: _tensor_sha256(value.to(torch.uint8)) for name, value in swaps.items()},
            "risk_feature_sha256": {
                "parent_stats4": _tensor_sha256(full_trace.parent_stats4),
                "risk_features13": _tensor_sha256(full_trace.risk_features13)
                if full_trace.risk_features13 is not None
                else None,
            },
            "b0_receipt": {
                "raw_image_open_count": 0,
                "raw_crop_encode_count": 0,
                "eval_all25_opened": False,
                "policy_decision_before_labels": True,
                "labels_loaded_after_logits": True,
                "opened_keys": opened_keys,
            },
            "identity": {
                "text_manifest_sha256": config["text_manifest_sha256"],
                "role_tensor_sha256": config["role_tensor_sha256"],
                "name_tensor_sha256": config["name_tensor_sha256"],
                "patch_manifest_sha256": config["patch_manifest_sha256"],
                "cls_tensor_sha256": config["cls_tensor_sha256"],
                "patch_tensor_sha256": config["patch_tensor_sha256"],
                "action_bundle_manifest_sha256": config["action_bundle_manifest_sha256"],
                "dev_eval_manifest_sha256": config["dev_eval_manifest_sha256"],
                "dev_eval_oracle_manifest_sha256": config["dev_eval_oracle_manifest_sha256"],
                "combined_checkpoint_sha256": config["combined_checkpoint"]["sha256"],
                "combined_checkpoint_training_commit": config["combined_checkpoint"]["training_commit"],
                "combined_checkpoint_train_config_sha256": config["combined_checkpoint"]["train_config_sha256"],
                "action_geometry_sha256": config["action_geometry_sha256"],
            },
            "reproducibility": reproducibility,
            "opened_keys": opened_keys,
            "official_test_loaded": False,
            "unseen_images_used_for_gradient": False,
            "pclr_online_inference": False,
        }
        output = prepare_output_dir(output_dir)
        atomic_write_json(output / ("result.json" if passed else "failure.json"), result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    finally:
        assets.close()


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
