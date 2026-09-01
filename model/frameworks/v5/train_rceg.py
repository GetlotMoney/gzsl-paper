"""Train one frozen RCEG Gate-0 condition on the physical dev-seen asset."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from model.frameworks.v5.rceg import RCEGModel, rceg_loss
from model.frameworks.v5.rceg_data import load_rceg_subset, validate_bundle
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save, atomic_write_json, current_code_commit,
    prepare_output_dir, require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v5-rceg-dev-train.v1"
CONDITION_MODES = {
    "RCEG_FULL": "full",
    "RCEG_ABSOLUTE_ROLE": "absolute_role",
    "RCEG_REFERENCE_DIFFICULTY": "reference_difficulty",
    "RCEG_TARGET_FREE": "target_free",
    "RCEG_TARGET_SHUFFLE": "full",
    "RCEG_ROLE_SHUFFLE": "full",
}
CONFIG_KEYS = {
    "schema_version", "experiment_id", "condition_id", "train_manifest",
    "train_manifest_sha256", "bundle_manifest", "bundle_manifest_sha256",
    "asset_generation_commit", "device", "random_seed", "batch_size",
    "total_updates", "learning_rate", "weight_decay", "candidate_chunk_size",
    "unseen_images_used_for_gradient", "dev_unseen_text_used_for_gradient",
    "pclr_online_inference", "target_tensor_opened",
}


def load_config(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("RCEG train配置字段错误。")
    condition = config.get("condition_id")
    invalid = (
        config.get("schema_version") != SCHEMA
        or condition not in CONDITION_MODES
        or int(config["random_seed"]) != 7
        or int(config["batch_size"]) != 8
        or int(config["total_updates"]) != 1000
        or float(config["learning_rate"]) != 1e-3
        or float(config["weight_decay"]) != 1e-4
        or int(config["candidate_chunk_size"]) != 5
        or config["unseen_images_used_for_gradient"] is not False
        or config["dev_unseen_text_used_for_gradient"] is not False
        or config["pclr_online_inference"] is not False
        or bool(config["target_tensor_opened"]) != (condition != "RCEG_TARGET_FREE")
    )
    if invalid:
        raise ValueError("RCEG train配置身份或固定参数错误。")
    return config, sha256_file(path)


def same_class_cycle(labels: torch.Tensor) -> torch.Tensor:
    mapping = torch.empty(labels.numel(), dtype=torch.long)
    for class_id in torch.unique(labels, sorted=True):
        rows = torch.where(labels.eq(class_id))[0].sort().values
        if rows.numel() < 2:
            raise ValueError("RCEG target shuffle每类至少需要两张图。")
        mapping[rows] = rows.roll(-1)
    if bool(mapping.eq(torch.arange(labels.numel())).any()):
        raise RuntimeError("RCEG target shuffle出现固定点。")
    return mapping


def run(config_path: Path, output_path: Path, expected_commit: str, expected_config_sha: str):
    require_clean_code_tree()
    config, config_sha = load_config(config_path)
    if current_code_commit() != expected_commit or config_sha != expected_config_sha:
        raise ValueError("RCEG train commit/config SHA不匹配。")
    values, visible, manifest = load_rceg_subset(
        Path(config["train_manifest"]), config["train_manifest_sha256"],
        expected_subset="dev_train", include_target=bool(config["target_tensor_opened"]),
    )
    bundle = validate_bundle(
        Path(config["bundle_manifest"]), config["bundle_manifest_sha256"],
        subset_name="dev_train", subset_sha256=config["train_manifest_sha256"],
    )
    common = manifest["common_identity"]
    if (
        common.get("code_commit") != config["asset_generation_commit"]
        or common.get("bundle_id") != bundle.get("common_identity", {}).get("bundle_id")
        or values["class_ids"].numel() != 100
        or not bool(torch.isin(values["labels"].long(), values["class_ids"].long()).all())
    ):
        raise ValueError("RCEG train资产边界错误。")
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    roles = values["role_embeddings"].float()
    if config["condition_id"] == "RCEG_ROLE_SHUFFLE":
        roles = roles.roll(-1, dims=0)
    device = torch.device(config["device"])
    model = RCEGModel(
        values["name_embeddings"], roles, values["class_ids"],
        candidate_chunk_size=int(config["candidate_chunk_size"]),
    ).to(device)
    initial_state = copy.deepcopy(model.interaction_module.state_dict())
    optimizer = torch.optim.AdamW(
        model.interaction_module.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), foreach=False, fused=False,
    )
    class_map = torch.full((200,), -1, dtype=torch.long)
    class_map[values["class_ids"].long()] = torch.arange(values["class_ids"].numel())
    target_positions = class_map[values["labels"].long()]
    if bool(target_positions.lt(0).any()):
        raise ValueError("RCEG train label不属于100类轴。")
    target_mapping = (
        same_class_cycle(values["labels"].long())
        if config["condition_id"] == "RCEG_TARGET_SHUFFLE" else None
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    history, gradient_receipt = [], None
    train_mode = CONDITION_MODES[config["condition_id"]]
    model.train()
    for update in range(1, int(config["total_updates"]) + 1):
        ids = torch.randperm(len(target_positions), generator=generator)[: int(config["batch_size"])]
        image_cls = values["image_cls"].index_select(0, ids).to(device).float()
        masked_cls = values["masked_cls"].index_select(0, ids).to(device).float()
        visible_batch = torch.from_numpy(np.array(visible[ids.numpy()], copy=True)).to(device).float()
        target = None
        if bool(config["target_tensor_opened"]):
            target_ids = target_mapping.index_select(0, ids) if target_mapping is not None else ids
            target = values["target"].index_select(0, target_ids).to(device).float()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image_cls, masked_cls, visible_batch, target, mode=train_mode)
        losses = rceg_loss(
            outputs, target_positions.index_select(0, ids).to(device), mode=train_mode
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("RCEG loss包含NaN/Inf。")
        losses["total"].backward()
        for parameter in model.interaction_module.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("RCEG梯度包含NaN/Inf。")
        if update == 1:
            gradient_receipt = {
                name: parameter.grad is not None and bool(parameter.grad.abs().max() > 0)
                for name, parameter in model.interaction_module.named_parameters()
            }
            if not all(gradient_receipt.values()):
                raise RuntimeError(f"RCEG首步交互参数梯度门失败：{gradient_receipt}")
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == int(config["total_updates"]):
            history.append({
                "update": update,
                **{key: float(value.detach()) for key, value in losses.items()},
                "score_mean": float(outputs["score"].detach().mean()),
                "score_std": float(outputs["score"].detach().std(unbiased=False)),
            })
    output = prepare_output_dir(output_path)
    checkpoint = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "condition_id": config["condition_id"],
        "score_mode": train_mode,
        "code_commit": expected_commit,
        "config_sha256": config_sha,
        "bundle_manifest_sha256": config["bundle_manifest_sha256"],
        "bundle_id": common["bundle_id"],
        "class_ids": values["class_ids"].long(),
        "interaction_state_dict": {
            key: value.detach().cpu() for key, value in model.interaction_module.state_dict().items()
        },
        "initial_interaction_state_dict": initial_state,
        "gradient_receipt": gradient_receipt,
        "role_shuffle": config["condition_id"] == "RCEG_ROLE_SHUFFLE",
        "target_shuffle": config["condition_id"] == "RCEG_TARGET_SHUFFLE",
        "target_tensor_opened": bool(config["target_tensor_opened"]),
        "unseen_images_used_for_gradient": False,
        "dev_unseen_text_used_for_gradient": False,
        "pclr_online_inference": False,
    }
    atomic_torch_save(output / "checkpoint.pt", checkpoint)
    atomic_write_json(output / "history.json", {
        "history": history, "gradient_receipt": gradient_receipt,
        "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
    })
    return {
        "output": str(output), "checkpoint": str(output / "checkpoint.pt"),
        "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
        "final": history[-1], "gradient_receipt": gradient_receipt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args()
    print(run(args.config, args.output, args.expected_commit, args.expected_config_sha))


if __name__ == "__main__":
    main()
