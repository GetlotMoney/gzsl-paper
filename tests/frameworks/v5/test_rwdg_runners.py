import dataclasses
import json

import torch

from model.frameworks.v5.evaluate_rwdg_gate0 import (
    cub_relative_image_path,
    hash_random_actions,
    load_oracle_receipt as load_eval_oracle,
    oracle_gate_from_receipt as eval_oracle_gate,
    paired_comparison,
)
from model.frameworks.v5.train_rwdg_gate0 import (
    load_and_validate_oracle_receipt,
    load_strict_config,
    validate_config,
)
from tools.runtime import sha256_file


def _receipt(config):
    return {
        "schema_version": "gzsl-paper.v5-rwdg-projected-pair-oracle.v1",
        "rows": 2355,
        "active_classes": 150,
        "parent_macro_top1_percent": 66.69292303042761,
        "pair_crop_oracle25_macro_top1_percent": 82.76147019736808,
        "oracle_gain_pp": 16.068547166940476,
        "used_for_training": False,
        "official_test_loaded": False,
        "unseen_images_used_for_gradient": False,
        "pclr_online_inference": False,
        "gates": {
            "axis_150": True,
            "damage_zero": True,
            "oracle_gain_at_least_1pp": True,
            "rows_2355": True,
        },
        "asset_identity": {
            "text_manifest": {"sha256": config.text_manifest_sha256},
            "patch_manifest": {"sha256": config.patch_manifest_sha256},
            "bundle_manifest": {"sha256": config.cuav_bundle_manifest_sha256},
            "eval_manifest": {"sha256": config.dev_eval_manifest_sha256},
            "oracle_manifest": {"sha256": config.dev_eval_oracle_manifest_sha256},
            "action_geometry_sha256": config.action_geometry_sha256,
        },
    }


def test_train_config_and_exact_oracle_receipt_contract(tmp_path):
    config, _ = load_strict_config("config/tries/v5_try_007_rwdg_gate0_train.json")
    validate_config(config)
    receipt_path = tmp_path / "oracle.json"
    receipt_path.write_text(json.dumps(_receipt(config)), encoding="utf-8")
    bound = dataclasses.replace(
        config,
        oracle_receipt=str(receipt_path),
        oracle_receipt_sha256=sha256_file(receipt_path),
    )
    validated = load_and_validate_oracle_receipt(bound)
    assert validated["gate"]["passed"]
    assert validated["gate"]["oracle_gain_pp"] == 16.068547166940476


def test_eval_oracle_parser_binds_every_identity(tmp_path):
    config, _ = load_strict_config("config/tries/v5_try_007_rwdg_gate0_train.json")
    receipt_path = tmp_path / "oracle.json"
    receipt = _receipt(config)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    eval_config = {
        "text_manifest_sha256": config.text_manifest_sha256,
        "patch_manifest_sha256": config.patch_manifest_sha256,
        "cuav_bundle_manifest_sha256": config.cuav_bundle_manifest_sha256,
        "dev_eval_manifest_sha256": config.dev_eval_manifest_sha256,
        "dev_eval_oracle_manifest_sha256": config.dev_eval_oracle_manifest_sha256,
        "action_geometry_sha256": config.action_geometry_sha256,
    }
    loaded = load_eval_oracle(str(receipt_path), sha256_file(receipt_path), eval_config)
    assert eval_oracle_gate(loaded)["passed"]


def test_hash_random_uses_normalized_cub_relative_path():
    absolute = "/warehouse/CUB_200_2011/images/005.Crested_Auklet/a.jpg"
    windows = r"C:\warehouse\CUB_200_2011\images\005.Crested_Auklet\a.jpg"
    assert cub_relative_image_path(absolute) == "005.Crested_Auklet/a.jpg"
    assert cub_relative_image_path(windows) == "005.Crested_Auklet/a.jpg"
    first, first_sha = hash_random_actions(
        [absolute], torch.tensor([4]), torch.tensor([9])
    )
    second, second_sha = hash_random_actions(
        [windows], torch.tensor([4]), torch.tensor([9])
    )
    assert torch.equal(first, second)
    assert first_sha == second_sha
    assert 0 <= int(first.item()) < 25


def test_paired_bootstrap_uses_classwise_differences():
    full = torch.ones(50)
    other = torch.zeros(50)
    matrix = torch.randint(0, 50, (100, 50), generator=torch.Generator().manual_seed(7))
    result = paired_comparison(full, other, matrix)
    assert result["observed_pp"] == 100.0
    assert result["ci95"] == [100.0, 100.0]
