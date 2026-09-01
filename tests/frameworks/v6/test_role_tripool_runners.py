import dataclasses
import json

import torch

from model.frameworks.v6 import evaluate_role_tripool_gate0 as tripool_eval
from model.frameworks.v6 import train_role_tripool_gate0 as tripool_train
from model.frameworks.v6.role_tripool import ACTION_COUNT, FEATURE_DIM, RoleTriPoolGlimpse
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
        "oracle_all25_opened": True,
        "official_test_loaded": False,
        "unseen_images_used_for_gradient": False,
        "pclr_online_inference": False,
        "gates": {"axis_150": True, "damage_zero": True, "oracle_gain_at_least_1pp": True, "rows_2355": True},
        "asset_identity": {
            "text_manifest": {"sha256": config.text_manifest_sha256},
            "patch_manifest": {"sha256": config.patch_manifest_sha256},
            "bundle_manifest": {"sha256": config.cuav_bundle_manifest_sha256},
            "eval_manifest": {"sha256": config.dev_eval_manifest_sha256},
            "oracle_manifest": {"sha256": config.dev_eval_oracle_manifest_sha256},
            "action_geometry_sha256": config.action_geometry_sha256,
        },
    }


def _toy_model():
    names = torch.zeros(3, FEATURE_DIM)
    names[0, 0], names[1, 1], names[2, 2] = 1, 1, 1
    roles = names[:, None, :].expand(-1, 8, -1).clone()
    return RoleTriPoolGlimpse(roles, names, torch.tensor([10, 20, 30]), seed=7)


def test_runner_schemas_and_config():
    assert tripool_train.TRAIN_SCHEMA == "gzsl-paper.v6-role-tripool-gate0-train.v1"
    assert tripool_eval.SCHEMA == "gzsl-paper.v6-role-tripool-gate0-eval.v1"
    config, _ = tripool_train.load_strict_config("config/tries/v6_try_001_role_tripool_gate0_train.json")
    tripool_train.validate_config(config)


def test_oracle_receipt_identity_contract(tmp_path):
    config, _ = tripool_train.load_strict_config("config/tries/v6_try_001_role_tripool_gate0_train.json")
    receipt_path = tmp_path / "oracle.json"
    receipt_path.write_text(json.dumps(_receipt(config)), encoding="utf-8")
    bound = dataclasses.replace(config, oracle_receipt=str(receipt_path), oracle_receipt_sha256=sha256_file(receipt_path))
    assert tripool_train.load_and_validate_oracle_receipt(bound)["gate"]["passed"]


def test_target_helper_returns_all_25_signed_labels():
    model = _toy_model()
    full = torch.zeros(3, FEATURE_DIM)
    full[:, 0], full[:, 1] = 1.0, 0.4
    crops = torch.zeros(3, ACTION_COUNT, FEATURE_DIM)
    crops[:, :, 0] = 1.0
    crops[0, 2, 0], crops[0, 2, 1] = -0.1, 1.0
    crops[1, 4, 0], crops[1, 4, 1] = -0.4, 1.0
    targets, groups, detail = tripool_train.tri_pool_action_targets(
        model, full, crops, torch.tensor([10, 20, 30])
    )
    assert targets.shape == (3, ACTION_COUNT)
    assert groups.tolist() == [0, 1, 2]
    assert set(targets.unique().tolist()) == {0, 1, 2}
    assert detail["target_policy"] == "per_window_executable_damage_neutral_correction"


def test_natural_sampler_is_deterministic_without_4to4_balancing():
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long)
    left = tripool_train.NaturalRowSampler(groups, batch_size=4, seed=7)
    right = tripool_train.NaturalRowSampler(groups, batch_size=4, seed=7)
    for _ in range(6):
        assert torch.equal(left.sample(), right.sample())
    assert left.state_dict()["sampling_distribution"] == "natural_uniform_rows"


def test_static_best_maximizes_natural_net_action():
    targets = torch.ones(6, ACTION_COUNT, dtype=torch.long)
    targets[:3, 2] = 2
    targets[3, 2] = 0
    targets[:2, 4] = 2
    action, summary = tripool_train.static_best_action_from_natural_targets(targets)
    assert action == 2
    assert summary["net_histogram"][2] == 2
    assert summary["static_best_rule"] == "max_natural_net_action"


def test_group_safety_gate_and_bootstrap_helpers():
    groups = {
        "leader": {"count": 100, "trigger": 10, "abstain": 90, "corrected": 0, "damaged": 2, "net": -2},
        "challenger": {"count": 50, "trigger": 30, "abstain": 20, "corrected": 12, "damaged": 0, "net": 12},
        "outside": {"count": 20, "trigger": 2, "abstain": 18, "corrected": 0, "damaged": 1, "net": -1},
    }
    assert tripool_eval.group_safety_gate(groups)["passed"]
    matrix = torch.randint(0, 50, (100, 50), generator=torch.Generator().manual_seed(7))
    assert tripool_eval.paired_comparison(torch.ones(50), torch.zeros(50), matrix)["ci95"] == [100.0, 100.0]
