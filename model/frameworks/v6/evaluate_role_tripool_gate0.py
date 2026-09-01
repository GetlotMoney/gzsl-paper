"""Strict Gate-0 evaluator for IDEA-197 / Role Tri-Pool.

The critical ordering is:

1. Load only safe CLS + projected-patch assets and freeze Full/S/V-off
   action/trigger decisions.
2. Build triggered control actions from safe projected-patch/train-only state.
3. Only then load eval image paths/boxes and execute selected raw crops.
4. Only after all logits are frozen, load eval labels and compute metrics.

The evaluator never opens the eval all25 crop table.  PairCropOracle25 is
accepted only as a SHA-bound external receipt produced by a separate runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from model.frameworks.v6.role_tripool import (
    ACTION_COUNT,
    ACTION_GEOMETRY_SHA256,
    FEATURE_DIM,
    PAIR_TEMPERATURE,
    WINDOW_SIZE,
    WINDOW_STARTS,
    RoleWindowDenseGlimpse,
)
from model.frameworks.v6.rwdg_data import (
    ManifestContract,
    RWDGAssetConfig,
    RWDGDataError,
    RWDGGateSubsetView,
    TensorContract,
    load_rwdg_gate_data,
    resolve_subset_output,
)
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.reproducibility import configure_reproducibility
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v6-role-tripool-gate0-eval.v1"
CHECKPOINT_SCHEMA = "gzsl-paper.v6-role-tripool-gate0-train.v1"
ORACLE_SCHEMA = "gzsl-paper.v5-rwdg-projected-pair-oracle.v1"
TEXT_HEATMAP_CONTROL = "triggered_textheatmap"
WINDOWS = tuple((row, col) for row in WINDOW_STARTS for col in WINDOW_STARTS)
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
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
    "cuav_bundle_manifest",
    "cuav_bundle_manifest_sha256",
    "dev_train_manifest_sha256",
    "dev_eval_manifest_sha256",
    "dev_eval_oracle_manifest_sha256",
    "action_geometry_sha256",
    "att_splits_mat_path",
    "trainval_count",
    "oracle_receipt",
    "oracle_receipt_sha256",
    "full_checkpoint",
    "clip_checkpoint",
    "clip_checkpoint_sha256",
    "device",
    "random_seed",
    "eval_batch_size",
    "crop_batch_size",
    "bootstrap_seed",
    "bootstrap_samples",
    "module_contract_margin",
    "support_control_margin",
    "official_test_loaded",
    "unseen_images_used_for_gradient",
    "pclr_online_inference",
}


class RWDGEvalError(RuntimeError):
    """Raised when Gate-0 evaluation would violate the Role Tri-Pool contract."""


@dataclass(frozen=True)
class FrozenDecisions:
    """CPU copy of all pre-action decisions for one condition."""

    name: str
    parent_logits: torch.Tensor
    top2: torch.Tensor
    leader_ids: torch.Tensor
    challenger_ids: torch.Tensor
    actions: torch.Tensor
    trigger: torch.Tensor
    max_utility: torch.Tensor
    utility: torch.Tensor
    tri_state_entropy: torch.Tensor
    role_statistics: torch.Tensor


@dataclass(frozen=True)
class EncodedSelectedCrops:
    """Selected raw-crop features and physical B1 counts for one condition."""

    features: torch.Tensor
    boxes: torch.Tensor
    raw_open_count: int
    selected_crop_forward_count: int
    selected_action_sha256: str
    selected_trigger_sha256: str
    selected_boxes_sha256: str


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != REQUIRED_CONFIG_KEYS:
        raise RWDGEvalError(
            "Role Tri-Pool Gate0 eval config字段错误；"
            f"缺少={sorted(REQUIRED_CONFIG_KEYS - actual)} 多出={sorted(actual - REQUIRED_CONFIG_KEYS)}"
        )
    invalid = (
        config["schema_version"] != SCHEMA
        or int(config["random_seed"]) != 7
        or int(config["bootstrap_seed"]) != 7
        or int(config["bootstrap_samples"]) != 10000
        or int(config["trainval_count"]) != 7057
        or int(config["eval_batch_size"]) <= 0
        or int(config["crop_batch_size"]) <= 0
        or float(config["module_contract_margin"]) != 1.0
        or float(config["support_control_margin"]) != 0.5
        or config["action_geometry_sha256"] != ACTION_GEOMETRY_SHA256
        or config["official_test_loaded"] is not False
        or config["unseen_images_used_for_gradient"] is not False
        or config["pclr_online_inference"] is not False
    )
    if invalid:
        raise RWDGEvalError("Role Tri-Pool Gate0 eval固定协议错误。")
    return config, sha256_file(path)


def asset_config_from_eval_config(config: Mapping[str, Any]) -> RWDGAssetConfig:
    count = int(config["trainval_count"])
    return RWDGAssetConfig(
        text_manifest=ManifestContract(
            path=str(config["text_manifest"]),
            sha256=str(config["text_manifest_sha256"]),
        ),
        role_tensor=TensorContract(
            path=str(config["role_tensor"]),
            sha256=str(config["role_tensor_sha256"]),
            shape=(200, 8, FEATURE_DIM),
            dtype="float32",
        ),
        name_tensor=TensorContract(
            path=str(config["name_tensor"]),
            sha256=str(config["name_tensor_sha256"]),
            shape=(200, FEATURE_DIM),
            dtype="float32",
        ),
        patch_manifest=ManifestContract(
            path=str(config["patch_manifest"]),
            sha256=str(config["patch_manifest_sha256"]),
        ),
        cls_tensor=TensorContract(
            path=str(config["cls_tensor"]),
            sha256=str(config["cls_tensor_sha256"]),
            shape=(count, FEATURE_DIM),
            dtype="float32",
        ),
        patch_tensor=TensorContract(
            path=str(config["patch_tensor"]),
            sha256=str(config["patch_tensor_sha256"]),
            shape=(count, 576, FEATURE_DIM),
            dtype="float16",
        ),
        cuav_bundle_manifest=ManifestContract(
            path=str(config["cuav_bundle_manifest"]),
            sha256=str(config["cuav_bundle_manifest_sha256"]),
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


def _load_output_tensor_from_assets(assets, subset_name: str, filename: str) -> torch.Tensor:
    value = _torch_load(resolve_subset_output(assets, subset_name, filename, verify_sha=True))
    if isinstance(value, Mapping):
        for item in value.values():
            if torch.is_tensor(item):
                return item.detach().cpu()
        raise RWDGEvalError(f"{subset_name}.{filename} mapping中没有tensor。")
    if not torch.is_tensor(value):
        raise RWDGEvalError(f"{subset_name}.{filename}不是tensor。")
    return value.detach().cpu()


def _load_class_ids(assets, subset_name: str, meta: Mapping[str, Any]) -> torch.Tensor:
    ids = meta.get("class_ids")
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        return torch.as_tensor(list(ids), dtype=torch.long)
    return _load_output_tensor_from_assets(assets, subset_name, "class_ids.pt").long()


def _validate_axis_labels(name: str, labels: torch.Tensor, class_ids: torch.Tensor) -> None:
    if labels.ndim != 1:
        raise RWDGEvalError(f"{name} labels必须是一维。")
    if not bool(torch.isin(labels.long(), class_ids.long()).all()):
        raise RWDGEvalError(f"{name} labels包含active axis之外的类别。")


def load_checkpoint(
    spec: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_bundle_sha256: str,
    expected_train_config_sha256: str | None = None,
) -> Mapping[str, Any]:
    required = {"path", "sha256", "training_commit", "train_config_sha256"}
    if not isinstance(spec, Mapping) or set(spec) != required:
        raise RWDGEvalError("full_checkpoint字段必须精确包含path/sha256/training_commit/train_config_sha256。")
    path = Path(str(spec["path"]))
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise RWDGEvalError("Role Tri-Pool checkpoint路径或SHA错误。")
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, Mapping):
        raise RWDGEvalError("Role Tri-Pool checkpoint不是mapping。")
    if expected_train_config_sha256 is not None and spec["train_config_sha256"] != expected_train_config_sha256:
        raise RWDGEvalError("Role Tri-Pool checkpoint train config SHA与预期不一致。")
    invalid = (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("condition_id") != "ROLE_TRIPOOL_FULL"
        or checkpoint.get("code_commit") != expected_commit
        or spec["training_commit"] != expected_commit
        or checkpoint.get("config_sha256") != spec["train_config_sha256"]
        or checkpoint.get("cuav_bundle_manifest_sha256") != expected_bundle_sha256
        or "state_dict" not in checkpoint
    )
    if invalid:
        raise RWDGEvalError("Role Tri-Pool checkpoint身份错误。")
    return checkpoint


def _nested_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise RWDGEvalError(f"oracle receipt缺少object字段：{key}")
    return item


def _sha_from_asset_identity(identity: Mapping[str, Any], key: str) -> str:
    item = identity.get(key)
    if isinstance(item, Mapping) and isinstance(item.get("sha256"), str):
        return item["sha256"]
    raise RWDGEvalError(f"oracle receipt asset_identity缺少{key}.sha256。")


def load_oracle_receipt(path: str, sha256: str, config: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt_path = Path(path)
    if not receipt_path.is_file() or sha256_file(receipt_path) != sha256:
        raise RWDGEvalError("RWDG oracle receipt路径或SHA错误。")
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RWDGEvalError("RWDG oracle receipt不是JSON object。")
    if value.get("schema_version") != ORACLE_SCHEMA:
        raise RWDGEvalError("RWDG oracle receipt schema错误。")
    required = {
        "parent_macro_top1_percent",
        "pair_crop_oracle25_macro_top1_percent",
        "oracle_gain_pp",
        "gates",
        "rows",
        "active_classes",
        "used_for_training",
        "official_test_loaded",
        "unseen_images_used_for_gradient",
        "pclr_online_inference",
        "asset_identity",
    }
    missing = [key for key in sorted(required) if key not in value]
    if missing:
        raise RWDGEvalError(f"RWDG oracle receipt缺少字段：{missing}")
    gates = _nested_mapping(value, "gates")
    failed = {str(k): v for k, v in gates.items() if v is not True}
    if failed:
        raise RWDGEvalError(f"RWDG oracle receipt gates失败：{failed}")
    if (
        int(value["rows"]) != 2355
        or int(value["active_classes"]) != 150
        or value["used_for_training"] is not False
        or value["official_test_loaded"] is not False
        or value["unseen_images_used_for_gradient"] is not False
        or value["pclr_online_inference"] is not False
    ):
        raise RWDGEvalError("RWDG oracle receipt边界字段错误。")
    identity = _nested_mapping(value, "asset_identity")
    expected_identity = {
        "text_manifest": str(config["text_manifest_sha256"]),
        "patch_manifest": str(config["patch_manifest_sha256"]),
        "bundle_manifest": str(config["cuav_bundle_manifest_sha256"]),
        "eval_manifest": str(config["dev_eval_manifest_sha256"]),
        "oracle_manifest": str(config["dev_eval_oracle_manifest_sha256"]),
    }
    mismatches = {}
    for key, expected in expected_identity.items():
        actual = _sha_from_asset_identity(identity, key)
        if actual.lower() != expected.lower():
            mismatches[key] = {"actual": actual, "expected": expected}
    action_geometry = identity.get("action_geometry_sha256")
    if action_geometry != config["action_geometry_sha256"] or action_geometry != ACTION_GEOMETRY_SHA256:
        mismatches["action_geometry_sha256"] = {
            "actual": action_geometry,
            "expected": ACTION_GEOMETRY_SHA256,
        }
    if mismatches:
        raise RWDGEvalError(f"RWDG oracle receipt身份不一致：{mismatches}")
    return value


def oracle_gate_from_receipt(receipt: Mapping[str, Any], *, min_gain: float = 1.0) -> dict[str, Any]:
    parent = float(receipt["parent_macro_top1_percent"])
    oracle = float(receipt["pair_crop_oracle25_macro_top1_percent"])
    gain = float(receipt["oracle_gain_pp"])
    if abs((oracle - parent) - gain) > 1e-6:
        raise RWDGEvalError("oracle receipt gain与parent/oracle macro不一致。")
    return {
        "parent_macro_top1_percent": parent,
        "pair_crop_oracle25_macro_top1_percent": oracle,
        "oracle_gain_pp": gain,
        "rows": int(receipt["rows"]),
        "active_classes": int(receipt["active_classes"]),
        "used_for_training": bool(receipt["used_for_training"]),
        "passed": gain >= float(min_gain),
    }


def instantiate_model(assets, class_ids: torch.Tensor, checkpoint: Mapping[str, Any], device: torch.device):
    roles = assets.role_embeddings.index_select(0, class_ids.long()).float()
    names = assets.name_embeddings.index_select(0, class_ids.long()).float()
    model = RoleWindowDenseGlimpse(roles, names, class_ids.long(), seed=7).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


@torch.no_grad()
def freeze_decisions(
    model: RoleWindowDenseGlimpse,
    view: RWDGGateSubsetView,
    *,
    device: torch.device,
    batch_size: int,
    name: str,
    semantic_off: bool = False,
    visual_off: bool = False,
) -> FrozenDecisions:
    parent_logits: list[torch.Tensor] = []
    top2: list[torch.Tensor] = []
    leader_ids: list[torch.Tensor] = []
    challenger_ids: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    triggers: list[torch.Tensor] = []
    max_utility: list[torch.Tensor] = []
    utility: list[torch.Tensor] = []
    tri_state_entropy: list[torch.Tensor] = []
    role_statistics: list[torch.Tensor] = []
    for start in range(0, view.size, batch_size):
        rows = np.arange(start, min(start + batch_size, view.size), dtype=np.int64)
        batch = view.batch(rows, include_patches=not visual_off, as_torch=True, device=device)
        patches = None if visual_off else batch["patches"]
        state = model.utility_state(
            batch["cls"],
            patches,
            semantic_off=semantic_off,
            visual_off=visual_off,
        )
        probabilities = state.tri_state_probabilities.detach().cpu().float()
        parent_logits.append(state.pair.parent_logits.detach().cpu())
        top2.append(state.pair.top2.detach().cpu())
        leader_ids.append(state.pair.leader_ids.detach().cpu())
        challenger_ids.append(state.pair.challenger_ids.detach().cpu())
        actions.append(state.selected_action.detach().cpu())
        triggers.append(state.trigger.detach().cpu())
        max_utility.append(state.max_utility.detach().cpu())
        utility.append(state.utility.detach().cpu())
        tri_state_entropy.append(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=2)
        )
        role_statistics.append(state.role_statistics.detach().cpu().float())
    return FrozenDecisions(
        name=name,
        parent_logits=torch.cat(parent_logits),
        top2=torch.cat(top2),
        leader_ids=torch.cat(leader_ids),
        challenger_ids=torch.cat(challenger_ids),
        actions=torch.cat(actions).long(),
        trigger=torch.cat(triggers).bool(),
        max_utility=torch.cat(max_utility).float(),
        utility=torch.cat(utility).float(),
        tri_state_entropy=torch.cat(tri_state_entropy).float(),
        role_statistics=torch.cat(role_statistics).float(),
    )


def raw_crop_with_box(image: Image.Image, action: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    scale = 336.0 / min(width, height)
    resized_width, resized_height = width * scale, height * scale
    offset_x, offset_y = (resized_width - 336.0) / 2.0, (resized_height - 336.0) / 2.0
    row, column = WINDOWS[int(action)]
    left = (column * 14 + offset_x) / scale
    top = (row * 14 + offset_y) / scale
    right = ((column + WINDOW_SIZE) * 14 + offset_x) / scale
    bottom = ((row + WINDOW_SIZE) * 14 + offset_y) / scale
    box = (
        max(0, min(int(math.floor(left)), width - 1)),
        max(0, min(int(math.floor(top)), height - 1)),
        max(1, min(int(math.ceil(right)), width)),
        max(1, min(int(math.ceil(bottom)), height)),
    )
    box = (box[0], box[1], max(box[0] + 1, box[2]), max(box[1] + 1, box[3]))
    return image.crop(box), box


def _load_eval_paths_and_boxes(assets) -> tuple[list[str], torch.Tensor]:
    paths_path = resolve_subset_output(assets, "dev_eval", "image_paths.json", verify_sha=True)
    paths = json.loads(paths_path.read_text(encoding="utf-8"))
    if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
        raise RWDGEvalError("image_paths.json必须是字符串列表。")
    boxes = _load_output_tensor_from_assets(assets, "dev_eval", "crop_boxes.pt").long()
    if boxes.shape != (len(paths), ACTION_COUNT, 4):
        raise RWDGEvalError(f"crop_boxes shape错误：{tuple(boxes.shape)}")
    return paths, boxes


@torch.no_grad()
def encode_selected_raw_clip(
    clip_model,
    preprocess,
    paths: Sequence[str],
    actions: torch.Tensor,
    trigger: torch.Tensor,
    expected_boxes: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> EncodedSelectedCrops:
    if actions.shape != trigger.shape or actions.ndim != 1:
        raise RWDGEvalError("actions/trigger必须是一维且shape一致。")
    if len(paths) != actions.numel() or expected_boxes.shape != (actions.numel(), ACTION_COUNT, 4):
        raise RWDGEvalError("selected crop输入行数不一致。")
    features = torch.zeros(actions.numel(), FEATURE_DIM, dtype=torch.float32)
    boxes = torch.full((actions.numel(), 4), -1, dtype=torch.long)
    pending_tensors: list[torch.Tensor] = []
    pending_rows: list[int] = []
    raw_open_count = 0
    forward_count = 0

    def flush() -> None:
        nonlocal forward_count
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device).float()
        encoded = F.normalize(clip_model.encode_image(batch).float(), dim=-1).detach().cpu()
        row_tensor = torch.as_tensor(pending_rows, dtype=torch.long)
        features.index_copy_(0, row_tensor, encoded)
        forward_count += len(pending_rows)
        pending_tensors.clear()
        pending_rows.clear()

    for row in range(actions.numel()):
        if not bool(trigger[row]):
            continue
        action = int(actions[row])
        if action < 0 or action >= ACTION_COUNT:
            raise RWDGEvalError(f"selected action越界：row={row} action={action}")
        with Image.open(paths[row]) as handle:
            crop, box = raw_crop_with_box(handle.convert("RGB"), action)
        raw_open_count += 1
        expected = tuple(int(x) for x in expected_boxes[row, action].tolist())
        if tuple(box) != expected:
            raise RWDGEvalError(
                f"selected crop box与资产geometry不一致：row={row} action={action} got={box} expected={expected}"
            )
        boxes[row] = torch.tensor(box, dtype=torch.long)
        pending_tensors.append(preprocess(crop))
        pending_rows.append(row)
        if len(pending_rows) >= batch_size:
            flush()
    flush()
    if raw_open_count != int(trigger.sum()) or forward_count != int(trigger.sum()):
        raise RWDGEvalError("B1 selected crop计数与trigger_count不一致。")
    return EncodedSelectedCrops(
        features=features,
        boxes=boxes,
        raw_open_count=raw_open_count,
        selected_crop_forward_count=forward_count,
        selected_action_sha256=_tensor_sha256(actions.long()),
        selected_trigger_sha256=_tensor_sha256(trigger.bool()),
        selected_boxes_sha256=_tensor_sha256(boxes.long()),
    )


def _load_crop_feature_memmap(assets, subset_name: str, *, expected_count: int) -> np.memmap:
    path = resolve_subset_output(assets, subset_name, "crop_features.npy", verify_sha=True)
    value = np.load(path, mmap_mode="r")
    if value.shape != (expected_count, ACTION_COUNT, FEATURE_DIM) or value.dtype != np.float16:
        raise RWDGEvalError(f"crop_features.npy shape/dtype错误：{value.shape}/{value.dtype}")
    return value


@torch.no_grad()
def train_static_best_action(
    model: RoleWindowDenseGlimpse,
    train_view: RWDGGateSubsetView,
    assets,
    train_class_ids: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[int, list[int], str, dict[str, Any]]:
    labels = _load_output_tensor_from_assets(assets, "dev_train", "labels.pt").long()
    _validate_axis_labels("dev_train", labels, train_class_ids)
    crops = _load_crop_feature_memmap(assets, "dev_train", expected_count=train_view.size)
    choice_counts = torch.zeros(ACTION_COUNT + 1, dtype=torch.long)
    target_margin_min = float("inf")
    target_margin_max = float("-inf")
    target_margin_sum = 0.0
    target_margin_count = 0
    strongest_negative_margin_min = float("inf")
    strongest_negative_margin_max = float("-inf")
    strongest_negative_margin_sum = 0.0
    strongest_negative_margin_count = 0
    group_counts = torch.zeros(3, dtype=torch.long)
    for start in range(0, train_view.size, batch_size):
        end = min(start + batch_size, train_view.size)
        rows = np.arange(start, end, dtype=np.int64)
        batch = train_view.batch(rows, include_patches=False, as_torch=True, device=device)
        crop_batch = torch.from_numpy(np.asarray(crops[start:end]).copy()).to(device).float()
        pair = model.parent_state(batch["cls"])
        names = model.name_embeddings.to(device=device)
        crop_logits = torch.einsum(
            "bad,cd->bac", F.normalize(crop_batch, dim=-1), names
        ) / PAIR_TEMPERATURE
        local_labels = model._global_to_local_indices(labels[start:end].to(device), device)
        leader = pair.top2[:, 0]
        challenger = pair.top2[:, 1]
        group = torch.full_like(local_labels, fill_value=2)
        group = torch.where(local_labels.eq(leader), torch.zeros_like(group), group)
        group = torch.where(local_labels.eq(challenger), torch.ones_like(group), group)
        group_counts += torch.bincount(group.detach().cpu(), minlength=3)[:3]

        action_axis = torch.arange(ACTION_COUNT, device=device)
        row = torch.arange(end - start, device=device)
        leader_score = crop_logits[row[:, None], action_axis[None, :], leader[:, None]]
        challenger_score = crop_logits[row[:, None], action_axis[None, :], challenger[:, None]]
        margin = leader_score - challenger_score
        strongest_margin, strongest_action = margin.min(dim=1)
        choice = torch.zeros(end - start, dtype=torch.long, device=device)
        choose_action = group.eq(1) & strongest_margin.lt(0)
        choice[choose_action] = strongest_action[choose_action] + 1
        choice_counts += torch.bincount(choice.detach().cpu(), minlength=ACTION_COUNT + 1)

        flat_margin = margin.detach().cpu().double().flatten()
        target_margin_min = min(target_margin_min, float(flat_margin.min()))
        target_margin_max = max(target_margin_max, float(flat_margin.max()))
        target_margin_sum += float(flat_margin.sum())
        target_margin_count += int(flat_margin.numel())
        chosen_negative = strongest_margin[choose_action].detach().cpu().double()
        if chosen_negative.numel() > 0:
            strongest_negative_margin_min = min(
                strongest_negative_margin_min, float(chosen_negative.min())
            )
            strongest_negative_margin_max = max(
                strongest_negative_margin_max, float(chosen_negative.max())
            )
            strongest_negative_margin_sum += float(chosen_negative.sum())
            strongest_negative_margin_count += int(chosen_negative.numel())
    non_abstain_counts = choice_counts[1:]
    action = int(torch.argmax(non_abstain_counts).item())
    choice_stats = {
        "target_policy": "per_window_damage_neutral_correction",
        "natural_train_rows": int(train_view.size),
        "abstain_count": int(choice_counts[0]),
        "non_abstain_count": int(non_abstain_counts.sum()),
        "choice_counts_26": [int(x) for x in choice_counts],
        "action_counts_25": [int(x) for x in non_abstain_counts],
        "static_best_action": action,
        "static_best_action_frequency": int(non_abstain_counts[action]),
        "static_best_expected_action12": action == 12,
        "group_counts": {
            "leader": int(group_counts[0]),
            "challenger": int(group_counts[1]),
            "outside": int(group_counts[2]),
        },
        "target_margin": {
            "min": target_margin_min,
            "mean": target_margin_sum / max(1, target_margin_count),
            "max": target_margin_max,
        },
        "strongest_negative_margin": {
            "min": None if strongest_negative_margin_count == 0 else strongest_negative_margin_min,
            "mean": None if strongest_negative_margin_count == 0 else strongest_negative_margin_sum / strongest_negative_margin_count,
            "max": None if strongest_negative_margin_count == 0 else strongest_negative_margin_max,
            "count": strongest_negative_margin_count,
        },
    }
    receipt = {
        "choice_counts_26": choice_stats["choice_counts_26"],
        "action_counts_25": choice_stats["action_counts_25"],
        "count": int(train_view.size),
        "target_policy": choice_stats["target_policy"],
        "choice_stats": choice_stats,
    }
    return action, [int(x) for x in non_abstain_counts], _json_sha256(receipt), choice_stats


def center_actions(size: int) -> torch.Tensor:
    return torch.full((size,), 12, dtype=torch.long)


def static_actions(size: int, action: int) -> torch.Tensor:
    return torch.full((size,), int(action), dtype=torch.long)


def cub_relative_image_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    marker = "CUB_200_2011/images/"
    index = normalized.find(marker)
    if index < 0:
        raise RWDGEvalError(f"HashRandom image path缺少{marker}锚点：{path}")
    return normalized[index + len(marker):]


def hash_random_actions(paths: Sequence[str], leader_ids: torch.Tensor, challenger_ids: torch.Tensor) -> tuple[torch.Tensor, str]:
    actions = []
    mapping = []
    for path, leader, challenger in zip(paths, leader_ids.tolist(), challenger_ids.tolist(), strict=True):
        rel_path = cub_relative_image_path(path)
        key = f"seed7|{rel_path}|{int(leader)}|{int(challenger)}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        action = int(digest, 16) % ACTION_COUNT
        actions.append(action)
        mapping.append({"relative_path": rel_path, "key": key, "action": action})
    return torch.tensor(actions, dtype=torch.long), _json_sha256(mapping)


@torch.no_grad()
def textheatmap_actions(
    model: RoleWindowDenseGlimpse,
    view: RWDGGateSubsetView,
    trace: FrozenDecisions,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, str]:
    actions: list[torch.Tensor] = []
    role_ids: list[torch.Tensor] = []
    for start in range(0, view.size, batch_size):
        rows = np.arange(start, min(start + batch_size, view.size), dtype=np.int64)
        batch = view.batch(rows, include_patches=True, as_torch=True, device=device)
        local_top2 = trace.top2[start : start + len(rows)].to(device)
        roles = model.semantic.role_embeddings.to(device)
        leader = local_top2[:, 0]
        challenger = local_top2[:, 1]
        leader_roles = roles.index_select(0, leader)
        challenger_roles = roles.index_select(0, challenger)
        role_distance = 1 - (leader_roles * challenger_roles).sum(dim=-1)
        chosen_role = role_distance.argmax(dim=1)
        point_scores = torch.abs(
            torch.einsum("bpd,brd->bpr", batch["patches"], leader_roles)
            - torch.einsum("bpd,brd->bpr", batch["patches"], challenger_roles)
        )
        selected_patch_scores = point_scores[
            torch.arange(len(rows), device=device), :, chosen_role
        ].view(len(rows), 24, 24)
        window_scores = []
        for row_start, col_start in WINDOWS:
            window_scores.append(
                selected_patch_scores[
                    :,
                    row_start : row_start + WINDOW_SIZE,
                    col_start : col_start + WINDOW_SIZE,
                ].mean(dim=(1, 2))
            )
        scores = torch.stack(window_scores, dim=1)
        actions.append(scores.argmax(dim=1).cpu())
        role_ids.append(chosen_role.cpu())
    action_tensor = torch.cat(actions).long()
    role_tensor = torch.cat(role_ids).long()
    return action_tensor, _json_sha256(
        {
            "control": TEXT_HEATMAP_CONTROL,
            "actions_sha256": _tensor_sha256(action_tensor),
            "roles_sha256": _tensor_sha256(role_tensor),
        }
    )


@torch.no_grad()
def apply_pair_logits(
    model: RoleWindowDenseGlimpse,
    trace: FrozenDecisions,
    crop_features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    interaction_off: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    logits: list[torch.Tensor] = []
    swaps: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for start in range(0, trace.parent_logits.shape[0], batch_size):
        end = min(start + batch_size, trace.parent_logits.shape[0])
        parent = trace.parent_logits[start:end].to(device).float()
        top2 = trace.top2[start:end].to(device).long()
        trigger = trace.trigger[start:end].to(device).bool()
        if interaction_off:
            logits.append(parent.cpu())
            swaps.append(torch.zeros(end - start, dtype=torch.bool))
            continue
        crop = crop_features[start:end].to(device).float()
        margin = model.interaction.crop_margin(crop, model.name_embeddings.to(device), top2)
        current_logits, swap = model.interaction.apply_keep_swap(parent, top2, margin, trigger)
        logits.append(current_logits.cpu())
        swaps.append(swap.cpu())
        margins.append(margin.cpu())
    return torch.cat(logits), torch.cat(swaps), torch.cat(margins) if margins else None


def load_eval_labels_after_logits(assets, class_ids: torch.Tensor, *, expected_count: int) -> torch.Tensor:
    labels = _load_output_tensor_from_assets(assets, "dev_eval", "labels.pt").long()
    if labels.shape != (expected_count,):
        raise RWDGEvalError(f"eval labels shape错误：{tuple(labels.shape)}")
    _validate_axis_labels("dev_eval", labels, class_ids)
    if torch.unique(labels).numel() != 50:
        raise RWDGEvalError("dev_eval必须包含50个unseen类。")
    return labels


def metrics(logits: torch.Tensor, labels: torch.Tensor, class_ids: torch.Tensor) -> dict[str, Any]:
    if logits.shape != (labels.numel(), class_ids.numel()):
        raise RWDGEvalError(f"logits shape错误：{tuple(logits.shape)}")
    predictions = class_ids.long()[logits.argmax(dim=1)]
    classes = torch.unique(labels.long(), sorted=True)
    vector = torch.stack([
        predictions[labels.eq(cls)].eq(cls).double().mean() for cls in classes
    ])
    return {
        "macro_top1": 100 * float(vector.mean()),
        "micro_top1": 100 * float(predictions.eq(labels).double().mean()),
        "per_class": vector,
        "prediction": predictions,
        "classes": classes,
    }


def paired_comparison(full_vector: torch.Tensor, other_vector: torch.Tensor, matrix: torch.Tensor) -> dict[str, Any]:
    diff = 100 * (full_vector.double() - other_vector.double())
    samples = diff[matrix].mean(dim=1)
    ci = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.double))
    return {"observed_pp": float(diff.mean()), "ci95": [float(ci[0]), float(ci[1])]}


def group_statistics(
    trace: FrozenDecisions,
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


def evidence_summary(trace: FrozenDecisions) -> dict[str, Any]:
    stats = trace.role_statistics.double()
    entropy = trace.tri_state_entropy.double()
    return {
        "mean_tri_state_entropy": float(entropy.mean()),
        "role_mean_evidence": [float(x) for x in stats[:, :, :, 0].mean(dim=(0, 1))],
        "role_max_top2_evidence": [float(x) for x in stats[:, :, :, 1].mean(dim=(0, 1))],
        "role_min_top1_evidence": [float(x) for x in stats[:, :, :, 2].mean(dim=(0, 1))],
        "role_statistics_sha256": _tensor_sha256(trace.role_statistics),
    }


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
        "leader_damage_lt_challenger_corrections": (
            leader["damaged"] < challenger["corrected"]
        ),
        "net_positive": total_net > 0,
        "has_trigger_and_abstain": 0 < total_trigger < total_count,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "leader_trigger_rate": leader_rate,
        "challenger_trigger_rate": challenger_rate,
    }


def _condition_gate(comparison: Mapping[str, Any], margin: float) -> bool:
    return (
        float(comparison["observed_pp"]) >= float(margin)
        and float(comparison["ci95"][0]) > 0
    )


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str) -> dict[str, Any]:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise RWDGEvalError("Role Tri-Pool eval expected commit mismatch.")
    config, config_sha = load_config(config_path)
    if config_sha != expected_config_sha:
        raise RWDGEvalError("Role Tri-Pool eval config SHA mismatch.")
    reproducibility = configure_reproducibility(
        int(config["random_seed"]),
        strict_determinism=True,
        deterministic_warn_only=False,
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RWDGEvalError("Role Tri-Pool Gate0 eval必须在可用CUDA设备上运行。")
    if not Path(config["clip_checkpoint"]).is_file() or sha256_file(config["clip_checkpoint"]) != config["clip_checkpoint_sha256"]:
        raise RWDGEvalError("Role Tri-Pool CLIP checkpoint路径或SHA错误。")

    oracle_receipt = load_oracle_receipt(
        config["oracle_receipt"],
        config["oracle_receipt_sha256"],
        config,
    )
    oracle_gate = oracle_gate_from_receipt(oracle_receipt, min_gain=float(config["module_contract_margin"]))
    asset_config = asset_config_from_eval_config(config)
    assets, views = load_rwdg_gate_data(asset_config, strict_sha=True, validate_tensor_values=True)
    eval_view = views["dev_eval"]
    train_view = views["dev_train"]
    eval_class_ids = _load_class_ids(assets, "dev_eval", assets.dev_eval_manifest)
    train_class_ids = _load_class_ids(assets, "dev_train", assets.dev_train_manifest)
    if eval_view.size != 2355 or train_view.size != 4702 or eval_class_ids.numel() != 150 or train_class_ids.numel() != 100:
        raise RWDGEvalError("Role Tri-Pool Gate0 train/eval row或class数量错误。")

    checkpoint = load_checkpoint(
        config["full_checkpoint"],
        expected_commit=expected_commit,
        expected_bundle_sha256=config["cuav_bundle_manifest_sha256"],
    )
    full_model = instantiate_model(assets, eval_class_ids, checkpoint, device)
    train_model = instantiate_model(assets, train_class_ids, checkpoint, device)
    eval_batch = int(config["eval_batch_size"])
    crop_batch = int(config["crop_batch_size"])

    # Pre-action phase: only CLS + projected patch tokens are visible.
    full_trace = freeze_decisions(
        full_model, eval_view, device=device, batch_size=eval_batch, name="full"
    )
    soff_trace = freeze_decisions(
        full_model, eval_view, device=device, batch_size=eval_batch,
        name="s_off", semantic_off=True,
    )
    voff_trace = freeze_decisions(
        full_model, eval_view, device=device, batch_size=eval_batch,
        name="v_off", visual_off=True,
    )
    text_actions, text_mapping_sha = textheatmap_actions(
        full_model, eval_view, full_trace, device=device, batch_size=eval_batch
    )

    static_action, static_action_counts, static_sha, action_choice_stats = train_static_best_action(
        train_model,
        train_view,
        assets,
        train_class_ids,
        device=device,
        batch_size=eval_batch,
    )

    # Post-freeze metadata: raw paths/boxes are opened only after all Full/S/V decisions are frozen.
    paths, crop_boxes = _load_eval_paths_and_boxes(assets)
    random_actions, random_sha = hash_random_actions(
        paths, full_trace.leader_ids, full_trace.challenger_ids
    )
    controls = {
        "triggered_center": center_actions(eval_view.size),
        "triggered_static_best": static_actions(eval_view.size, static_action),
        "triggered_random": random_actions,
        TEXT_HEATMAP_CONTROL: text_actions,
    }

    import clip

    clip_model, preprocess = clip.load(str(config["clip_checkpoint"]), device=device, jit=False)
    clip_model = clip_model.float().eval()
    encoded = {
        "full": encode_selected_raw_clip(
            clip_model, preprocess, paths, full_trace.actions, full_trace.trigger,
            crop_boxes, device=device, batch_size=crop_batch,
        ),
        "s_off": encode_selected_raw_clip(
            clip_model, preprocess, paths, soff_trace.actions, soff_trace.trigger,
            crop_boxes, device=device, batch_size=crop_batch,
        ),
        "v_off": encode_selected_raw_clip(
            clip_model, preprocess, paths, voff_trace.actions, voff_trace.trigger,
            crop_boxes, device=device, batch_size=crop_batch,
        ),
    }
    for name, actions in controls.items():
        encoded[name] = encode_selected_raw_clip(
            clip_model, preprocess, paths, actions, full_trace.trigger,
            crop_boxes, device=device, batch_size=crop_batch,
        )

    # Logit phase: no labels are loaded yet.
    logits: dict[str, torch.Tensor] = {"parent": full_trace.parent_logits.clone()}
    swaps: dict[str, torch.Tensor] = {}
    logits["full"], swaps["full"], full_margin = apply_pair_logits(
        full_model, full_trace, encoded["full"].features, device=device, batch_size=eval_batch
    )
    logits["s_off"], swaps["s_off"], _ = apply_pair_logits(
        full_model, soff_trace, encoded["s_off"].features, device=device, batch_size=eval_batch
    )
    logits["v_off"], swaps["v_off"], _ = apply_pair_logits(
        full_model, voff_trace, encoded["v_off"].features, device=device, batch_size=eval_batch
    )
    logits["i_off"], swaps["i_off"], _ = apply_pair_logits(
        full_model,
        full_trace,
        encoded["full"].features,
        device=device,
        batch_size=eval_batch,
        interaction_off=True,
    )
    for name in controls:
        control_trace = FrozenDecisions(
            name=name,
            parent_logits=full_trace.parent_logits,
            top2=full_trace.top2,
            leader_ids=full_trace.leader_ids,
            challenger_ids=full_trace.challenger_ids,
            actions=controls[name],
            trigger=full_trace.trigger,
            max_utility=full_trace.max_utility,
            utility=full_trace.utility,
            tri_state_entropy=full_trace.tri_state_entropy,
            role_statistics=full_trace.role_statistics,
        )
        logits[name], swaps[name], _ = apply_pair_logits(
            full_model,
            control_trace,
            encoded[name].features,
            device=device,
            batch_size=eval_batch,
        )

    # Label phase: metrics and group analysis are computed only after logits are frozen.
    labels = load_eval_labels_after_logits(
        assets, eval_class_ids, expected_count=eval_view.size
    )
    metric_values = {name: metrics(value, labels, eval_class_ids) for name, value in logits.items()}
    matrix = torch.randint(
        0,
        50,
        (int(config["bootstrap_samples"]), 50),
        generator=torch.Generator().manual_seed(int(config["bootstrap_seed"])),
    )
    comparisons = {
        name: paired_comparison(metric_values["full"]["per_class"], value["per_class"], matrix)
        for name, value in metric_values.items()
        if name != "full"
    }
    parent_pred = metric_values["parent"]["prediction"]
    full_pred = metric_values["full"]["prediction"]
    group_stats = group_statistics(full_trace, labels, eval_class_ids, parent_pred, full_pred)
    corrected = full_pred.eq(labels) & parent_pred.ne(labels)
    damaged = full_pred.ne(labels) & parent_pred.eq(labels)
    histogram = torch.bincount(full_trace.actions[full_trace.trigger], minlength=ACTION_COUNT)
    trigger_count = int(full_trace.trigger.sum())
    highest_occupancy = float(histogram.max()) / max(1, trigger_count)
    used_actions = int(histogram.gt(0).sum())
    b1_expected_counts = {
        "full": int(full_trace.trigger.sum()),
        "s_off": int(soff_trace.trigger.sum()),
        "v_off": int(voff_trace.trigger.sum()),
        "triggered_center": trigger_count,
        "triggered_static_best": trigger_count,
        "triggered_random": trigger_count,
        TEXT_HEATMAP_CONTROL: trigger_count,
    }
    b1_condition_gates = {
        f"b1_{name}_count": (
            encoded[name].raw_open_count == expected
            and encoded[name].selected_crop_forward_count == expected
        )
        for name, expected in b1_expected_counts.items()
    }
    opened_keys = [
        "text_manifest",
        "role_tensor",
        "name_tensor",
        "patch_manifest",
        "cls_tensor",
        "patch_tensor_safe_view",
        "cuav_bundle_manifest",
        "dev_train_manifest",
        "dev_eval_manifest",
        "oracle_receipt_json",
        "dev_train.labels",
        "dev_train.class_ids",
        "dev_train.crop_features",
        "dev_eval.class_ids",
        "dev_eval.image_paths",
        "dev_eval.crop_boxes",
        "clip_checkpoint",
        "dev_eval.labels_after_logits",
    ]
    group_safety = group_safety_gate(group_stats)

    gates: dict[str, bool] = {
        "oracle_parent_plus1": bool(oracle_gate["passed"]),
        "full_vs_parent": _condition_gate(comparisons["parent"], config["module_contract_margin"]),
        "full_vs_s_off": _condition_gate(comparisons["s_off"], config["module_contract_margin"]),
        "full_vs_v_off": _condition_gate(comparisons["v_off"], config["module_contract_margin"]),
        "full_vs_i_off": _condition_gate(comparisons["i_off"], config["module_contract_margin"]),
        "full_vs_triggered_center": _condition_gate(comparisons["triggered_center"], config["support_control_margin"]),
        "full_vs_triggered_static_best": _condition_gate(comparisons["triggered_static_best"], config["support_control_margin"]),
        "full_vs_triggered_random": _condition_gate(comparisons["triggered_random"], config["support_control_margin"]),
        "full_vs_triggered_textheatmap": _condition_gate(comparisons[TEXT_HEATMAP_CONTROL], config["support_control_margin"]),
        "net_positive": int(corrected.sum() - damaged.sum()) > 0,
        "used_at_least_two_actions": used_actions >= 2,
        "highest_occupancy_lte_70pct": highest_occupancy <= 0.70,
        "has_trigger": trigger_count > 0,
        "has_abstain": trigger_count < eval_view.size,
        **group_safety["gates"],
        **b1_condition_gates,
        "no_eval_all25": "dev_eval.crop_features" not in opened_keys,
    }
    passed = all(gates.values())

    b1_counts = {
        name: {
            "raw_open_count": value.raw_open_count,
            "selected_crop_forward_count": value.selected_crop_forward_count,
            "selected_action_sha256": value.selected_action_sha256,
            "selected_trigger_sha256": value.selected_trigger_sha256,
            "selected_boxes_sha256": value.selected_boxes_sha256,
        }
        for name, value in encoded.items()
    }
    b1_counts["i_off"] = {
        "logical_selected_crop_cost": "same_as_full",
        "logical_raw_open_count": encoded["full"].raw_open_count,
        "logical_selected_crop_forward_count": encoded["full"].selected_crop_forward_count,
        "physical_raw_open_count": 0,
        "physical_selected_crop_forward_count": 0,
        "physical_reuse_count": encoded["full"].selected_crop_forward_count,
        "reuses_full_selected_crop": True,
        "interaction_margin_computed": False,
    }
    result = {
        "schema_version": SCHEMA,
        "method": "RoleTriPool",
        "condition_id": "ROLE_TRIPOOL_FULL",
        "experiment_id": config["experiment_id"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "metrics": {
            name: {"macro_top1": value["macro_top1"], "micro_top1": value["micro_top1"]}
            for name, value in metric_values.items()
        },
        "comparisons": comparisons,
        "gates": gates,
        "preliminary_gate0_passed": passed,
        "decision": "continue_gate1_controls" if passed else "drop_role_tripool_gate0_failed",
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
        "action_choice": action_choice_stats,
        "utility": {
            "trigger_count": trigger_count,
            "abstain_count": int(eval_view.size - trigger_count),
            "trigger_rate": float(full_trace.trigger.double().mean()),
            "max_utility_mean": float(full_trace.max_utility.double().mean()),
            "triggered_action_histogram": [int(x) for x in histogram],
            "highest_occupancy": highest_occupancy,
            "used_actions": used_actions,
            "full_utility_sha256": _tensor_sha256(full_trace.utility),
            "full_crop_margin_sha256": _tensor_sha256(full_margin) if full_margin is not None else None,
        },
        "role_tripool_evidence": evidence_summary(full_trace),
        "controls": {
            "static_best_action": int(static_action),
            "static_best_action_counts": static_action_counts,
            "static_best_sha256": static_sha,
            "static_best_target_policy": "max_natural_correction_minus_damage_action",
            "hash_random_mapping_sha256": random_sha,
            "textheatmap_mapping_sha256": text_mapping_sha,
            "trigger_source": "full_trigger",
        },
        "b1_receipt": {
            "pre_action_raw_open_count": 0,
            "pre_action_all25_crop_open_count": 0,
            "all25_full_eval_encoding_count": 0,
            "oracle_all25_opened_in_eval": False,
            "policy_decision_before_raw_open": True,
            "opened_keys": opened_keys,
            "counts_by_condition": b1_counts,
        },
        "identity": {
            "text_manifest_sha256": config["text_manifest_sha256"],
            "role_tensor_sha256": config["role_tensor_sha256"],
            "name_tensor_sha256": config["name_tensor_sha256"],
            "patch_manifest_sha256": config["patch_manifest_sha256"],
            "cls_tensor_sha256": config["cls_tensor_sha256"],
            "patch_tensor_sha256": config["patch_tensor_sha256"],
            "cuav_bundle_manifest_sha256": config["cuav_bundle_manifest_sha256"],
            "dev_train_manifest_sha256": config["dev_train_manifest_sha256"],
            "dev_eval_manifest_sha256": config["dev_eval_manifest_sha256"],
            "dev_eval_oracle_manifest_sha256": config["dev_eval_oracle_manifest_sha256"],
            "oracle_receipt_sha256": config["oracle_receipt_sha256"],
            "clip_checkpoint_sha256": config["clip_checkpoint_sha256"],
            "full_checkpoint_sha256": config["full_checkpoint"]["sha256"],
            "full_checkpoint_training_commit": config["full_checkpoint"]["training_commit"],
            "full_checkpoint_train_config_sha256": config["full_checkpoint"]["train_config_sha256"],
            "action_geometry_sha256": config["action_geometry_sha256"],
        },
        "reproducibility": reproducibility,
        "oracle_gate": oracle_gate,
        "opened_keys": opened_keys,
        "official_test_loaded": False,
        "unseen_images_used_for_gradient": False,
        "pclr_online_inference": False,
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
