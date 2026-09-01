"""Train CUAV Full or image-only policy on 100 dev-seen classes."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from model.frameworks.v5.cuav import CUAVModel, cuav_policy_loss
from model.frameworks.v5.cuav_data import load_subset, validate_bundle
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_torch_save, atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v5-cuav-dev-train.v1"
CONDITIONS = {"CUAV_FULL": False, "CUAV_IMAGE_ONLY": True}


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "condition_id", "train_manifest",
        "train_manifest_sha256", "bundle_manifest", "bundle_manifest_sha256",
        "asset_generation_commit", "device", "random_seed", "batch_size",
        "total_updates", "learning_rate", "weight_decay",
        "unseen_images_used_for_gradient", "official_test_loaded", "pclr_online_inference",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("CUAV train配置字段错误。")
    if (
        config["schema_version"] != SCHEMA or config["condition_id"] not in CONDITIONS
        or int(config["random_seed"]) != 7 or int(config["batch_size"]) != 8
        or int(config["total_updates"]) != 1000 or float(config["learning_rate"]) != 1e-3
        or float(config["weight_decay"]) != 1e-4
        or config["unseen_images_used_for_gradient"] is not False
        or config["official_test_loaded"] is not False
        or config["pclr_online_inference"] is not False
    ):
        raise ValueError("CUAV train固定协议错误。")
    return config, sha256_file(Path(path))


def run(config_path, output_path, expected_commit, expected_config_sha):
    require_clean_code_tree()
    config, config_sha = load_config(config_path)
    if current_code_commit() != expected_commit or config_sha != expected_config_sha:
        raise ValueError("CUAV train commit/config SHA错误。")
    values, crops, _, _, subset = load_subset(
        config["train_manifest"], config["train_manifest_sha256"],
        subset="dev_train", open_crops=True, open_paths=False,
    )
    bundle = validate_bundle(
        config["bundle_manifest"], config["bundle_manifest_sha256"],
        subset="dev_train", subset_sha=config["train_manifest_sha256"],
    )
    if (
        subset["common_identity"]["code_commit"] != config["asset_generation_commit"]
        or subset["common_identity"]["bundle_id"] != bundle["common_identity"]["bundle_id"]
        or values["class_ids"].numel() != 100
        or not bool(torch.isin(values["labels"].long(), values["class_ids"].long()).all())
    ):
        raise ValueError("CUAV train资产边界错误。")
    output = prepare_output_dir(output_path)
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    device = torch.device(config["device"])
    semantic_off = CONDITIONS[config["condition_id"]]
    model = CUAVModel(values["name_embeddings"], values["class_ids"]).to(device)
    optimizer = torch.optim.AdamW(
        model.visual_module.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), foreach=False, fused=False,
    )
    class_map = torch.full((200,), -1, dtype=torch.long)
    class_map[values["class_ids"].long()] = torch.arange(values["class_ids"].numel())
    targets = class_map[values["labels"].long()]
    generator = torch.Generator(device="cpu").manual_seed(7)
    initial = copy.deepcopy(model.visual_module.state_dict())
    history, receipt = [], {}
    model.train()
    for update in range(1, int(config["total_updates"]) + 1):
        ids = torch.randperm(len(targets), generator=generator)[: int(config["batch_size"])]
        crop_batch = torch.from_numpy(np.array(crops[ids.numpy()], copy=True)).to(device).float()
        optimizer.zero_grad(set_to_none=True)
        result = model.training_forward(
            values["full_cls"].index_select(0, ids).to(device).float(),
            crop_batch, semantic_off=semantic_off,
        )
        losses = cuav_policy_loss(result, targets.index_select(0, ids).to(device))
        losses["total"].backward()
        parameters = dict(model.visual_module.named_parameters())
        for parameter in parameters.values():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("CUAV梯度NaN/Inf。")
        wa = parameters["action_projection.weight"]
        if update == 1:
            receipt["step1_Wa_nonzero"] = wa.grad is not None and bool(wa.grad.abs().max() > 0)
            if not receipt["step1_Wa_nonzero"]:
                raise RuntimeError(f"CUAV step1梯度门失败：{receipt}")
        if update == 2:
            receipt["step2_Wa_nonzero"] = wa.grad is not None and bool(wa.grad.abs().max() > 0)
            receipt["step2_Wz_nonzero"] = bool(parameters["image_projection.weight"].grad.abs().max() > 0)
            if semantic_off:
                receipt["step2_Wq_Ws_not_applicable"] = True
            else:
                receipt["step2_Wq_nonzero"] = bool(parameters["query_projection.weight"].grad.abs().max() > 0)
                receipt["step2_Ws_nonzero"] = bool(parameters["stats_projection.weight"].grad.abs().max() > 0)
            if not all(receipt.values()):
                raise RuntimeError(f"CUAV step2梯度门失败：{receipt}")
        optimizer.step()
        if update in {1, 2} or update % 100 == 0 or update == 1000:
            actions = result["action"].detach().cpu()
            histogram = torch.bincount(actions, minlength=25)
            history.append({
                "update": update,
                **{key: float(value.detach()) for key, value in losses.items()},
                "policy_entropy": float((-(result["policy"] * torch.log(result["policy"].clamp_min(1e-6))).sum(1)).mean()),
                "batch_action_histogram": [int(x) for x in histogram],
            })
    checkpoint = {
        "schema_version": SCHEMA, "condition_id": config["condition_id"],
        "semantic_off": semantic_off, "code_commit": expected_commit,
        "config_sha256": config_sha, "bundle_id": subset["common_identity"]["bundle_id"],
        "class_ids": values["class_ids"].long(), "initial_policy_state": initial,
        "policy_state_dict": {key: value.detach().cpu() for key, value in model.visual_module.state_dict().items()},
        "gradient_receipt": receipt, "unseen_images_used_for_gradient": False,
        "official_test_loaded": False, "pclr_online_inference": False,
    }
    atomic_torch_save(output / "checkpoint.pt", checkpoint)
    atomic_write_json(output / "history.json", {
        "history": history, "gradient_receipt": receipt,
        "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
    })
    return {"checkpoint": str(output / "checkpoint.pt"), "checkpoint_sha256": sha256_file(output / "checkpoint.pt"), "final": history[-1], "gradient_receipt": receipt}


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
