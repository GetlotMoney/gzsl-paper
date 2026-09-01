"""Train one OREF preliminary Gate condition on physical dev-seen assets."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from model.frameworks.v5.oref import OREFModel, oref_loss
from model.frameworks.v5.oref_data import load_subset, validate_bundle
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save, atomic_write_json, current_code_commit,
    prepare_output_dir, require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v5-oref-dev-train.v1"
CONDITIONS = {
    "OREF_FULL": "full", "OREF_LEDGER_MLP": "ledger_mlp",
    "OREF_FILIP": "filip", "OREF_SIGNED_LEDGER": "signed_ledger",
}
KEYS = {
    "schema_version", "experiment_id", "condition_id", "train_manifest",
    "train_manifest_sha256", "bundle_manifest", "bundle_manifest_sha256",
    "asset_generation_commit", "device", "random_seed", "batch_size",
    "total_updates", "learning_rate", "weight_decay", "candidate_chunk_size",
    "unseen_images_used_for_gradient", "dev_unseen_text_used_for_gradient",
    "official_test_loaded", "pclr_online_inference",
}


def load_config(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError("OREF train配置字段错误。")
    if (
        config["schema_version"] != SCHEMA
        or config["condition_id"] not in CONDITIONS
        or int(config["random_seed"]) != 7
        or int(config["batch_size"]) != 8
        or int(config["total_updates"]) != 1000
        or float(config["learning_rate"]) != 1e-3
        or float(config["weight_decay"]) != 1e-4
        or int(config["candidate_chunk_size"]) != 5
        or config["unseen_images_used_for_gradient"] is not False
        or config["dev_unseen_text_used_for_gradient"] is not False
        or config["official_test_loaded"] is not False
        or config["pclr_online_inference"] is not False
    ):
        raise ValueError("OREF train固定协议错误。")
    return config, sha256_file(path)


def run(config_path, output_path, expected_commit, expected_config_sha):
    require_clean_code_tree()
    config, config_sha = load_config(config_path)
    if current_code_commit() != expected_commit or config_sha != expected_config_sha:
        raise ValueError("OREF train commit/config SHA不匹配。")
    values, patches, subset = load_subset(
        Path(config["train_manifest"]), config["train_manifest_sha256"],
        subset="dev_train", open_patches=True, open_roles=True,
    )
    bundle = validate_bundle(
        Path(config["bundle_manifest"]), config["bundle_manifest_sha256"],
        subset="dev_train", subset_sha=config["train_manifest_sha256"],
    )
    common = subset["common_identity"]
    if (
        common.get("code_commit") != config["asset_generation_commit"]
        or common.get("bundle_id") != bundle.get("common_identity", {}).get("bundle_id")
        or values["class_ids"].numel() != 100
        or not bool(torch.isin(values["labels"].long(), values["class_ids"].long()).all())
    ):
        raise ValueError("OREF train资产边界错误。")
    output = prepare_output_dir(output_path)
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    device = torch.device(config["device"])
    model = OREFModel(
        values["name_embeddings"], values["role_embeddings"], values["class_ids"],
        candidate_chunk_size=int(config["candidate_chunk_size"]),
    ).to(device)
    mode = CONDITIONS[config["condition_id"]]
    parameters = list(model.visual_module.parameters())
    if mode == "ledger_mlp":
        parameters += list(model.interaction_module.ledger_mlp.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), foreach=False, fused=False,
    )
    class_map = torch.full((200,), -1, dtype=torch.long)
    class_map[values["class_ids"].long()] = torch.arange(values["class_ids"].numel())
    targets = class_map[values["labels"].long()]
    if bool(targets.lt(0).any()):
        raise ValueError("OREF train label不属于100类轴。")
    generator = torch.Generator(device="cpu").manual_seed(7)
    initial = {
        "visual": copy.deepcopy(model.visual_module.state_dict()),
        "ledger_mlp": copy.deepcopy(model.interaction_module.ledger_mlp.state_dict()),
    }
    history, gradient_receipt = [], {}
    model.train()
    for update in range(1, int(config["total_updates"]) + 1):
        ids = torch.randperm(len(targets), generator=generator)[: int(config["batch_size"])]
        patch_batch = torch.from_numpy(np.array(patches[ids.numpy()], copy=True)).to(device).float()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            values["image_cls"].index_select(0, ids).to(device).float(),
            patch_batch, mode=mode,
        )
        losses = oref_loss(outputs, targets.index_select(0, ids).to(device))
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("OREF loss包含NaN/Inf。")
        losses["total"].backward()
        for parameter in parameters:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("OREF梯度包含NaN/Inf。")
        wi = model.visual_module.input_projection.weight
        wo = model.visual_module.output_projection.weight
        if update == 1:
            gradient_receipt["step1_Wo_nonzero"] = wo.grad is not None and bool(wo.grad.abs().max() > 0)
            gradient_receipt["step1_Wi_zero_allowed"] = wi.grad is not None and bool(wi.grad.abs().max() == 0)
            if not gradient_receipt["step1_Wo_nonzero"]:
                raise RuntimeError(f"OREF step1梯度门失败：{gradient_receipt}")
        if update == 2:
            gradient_receipt["step2_Wo_nonzero"] = wo.grad is not None and bool(wo.grad.abs().max() > 0)
            gradient_receipt["step2_Wi_nonzero"] = wi.grad is not None and bool(wi.grad.abs().max() > 0)
            if not gradient_receipt["step2_Wo_nonzero"] or not gradient_receipt["step2_Wi_nonzero"]:
                raise RuntimeError(f"OREF step2梯度门失败：{gradient_receipt}")
        optimizer.step()
        if update in {1, 2} or update % 100 == 0 or update == int(config["total_updates"]):
            history.append({
                "update": update,
                **{key: float(value.detach()) for key, value in losses.items()},
                "score_mean": float(outputs["score"].detach().mean()),
                "score_std": float(outputs["score"].detach().std(unbiased=False)),
            })
    checkpoint = {
        "schema_version": SCHEMA, "experiment_id": config["experiment_id"],
        "condition_id": config["condition_id"], "score_mode": mode,
        "code_commit": expected_commit, "config_sha256": config_sha,
        "bundle_manifest_sha256": config["bundle_manifest_sha256"],
        "bundle_id": common["bundle_id"], "class_ids": values["class_ids"].long(),
        "visual_state_dict": {k: v.detach().cpu() for k, v in model.visual_module.state_dict().items()},
        "ledger_mlp_state_dict": {k: v.detach().cpu() for k, v in model.interaction_module.ledger_mlp.state_dict().items()},
        "initial_state": initial, "gradient_receipt": gradient_receipt,
        "opened_asset_keys": ["image_cls", "patch_tokens", "name_embeddings", "role_embeddings", "labels"],
        "unseen_images_used_for_gradient": False,
        "dev_unseen_text_used_for_gradient": False,
        "official_test_loaded": False, "pclr_online_inference": False,
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
