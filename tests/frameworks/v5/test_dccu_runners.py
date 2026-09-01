import dataclasses
import json

import torch

from model.frameworks.v5 import evaluate_dccu_gate0 as dccu_eval
from model.frameworks.v5 import train_dccu_gate0 as dccu_train
from model.frameworks.v5.rwdg import ACTION_COUNT, FEATURE_DIM, RoleWindowDenseGlimpse
from tools.runtime import sha256_file


EXPECTED_TRAIN_SCHEMA = "gzsl-paper.v5-dccu-gate0-train.v1"
EXPECTED_EVAL_SCHEMA = "gzsl-paper.v5-dccu-gate0-eval.v1"


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


def _require_helper(module, name):
    assert hasattr(module, name), f"{module.__name__}.{name} helper is required"
    return getattr(module, name)


def _toy_model() -> RoleWindowDenseGlimpse:
    class_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    names = torch.zeros(3, FEATURE_DIM)
    names[0, 0] = 1.0
    names[1, 1] = 1.0
    names[2, 2] = 1.0
    roles = torch.zeros(3, 8, FEATURE_DIM)
    return RoleWindowDenseGlimpse(roles, names, class_ids, seed=7)


def test_dccu_runner_schemas_are_not_rwdg_schemas():
    assert dccu_train.TRAIN_SCHEMA == EXPECTED_TRAIN_SCHEMA
    assert dccu_eval.SCHEMA == EXPECTED_EVAL_SCHEMA


def test_train_config_and_exact_oracle_receipt_contract(tmp_path):
    config, _ = dccu_train.load_strict_config("config/tries/v5_try_008_dccu_gate0_train.json")
    dccu_train.validate_config(config)
    receipt_path = tmp_path / "oracle.json"
    receipt_path.write_text(json.dumps(_receipt(config)), encoding="utf-8")
    bound = dataclasses.replace(
        config,
        oracle_receipt=str(receipt_path),
        oracle_receipt_sha256=sha256_file(receipt_path),
    )
    validated = dccu_train.load_and_validate_oracle_receipt(bound)
    assert validated["gate"]["passed"]
    assert validated["gate"]["oracle_gain_pp"] == 16.068547166940476


def test_eval_oracle_parser_binds_every_identity(tmp_path):
    config, _ = dccu_train.load_strict_config("config/tries/v5_try_008_dccu_gate0_train.json")
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
    loaded = dccu_eval.load_oracle_receipt(str(receipt_path), sha256_file(receipt_path), eval_config)
    assert dccu_eval.oracle_gate_from_receipt(loaded)["passed"]


def test_correction_target_masks_leader_and_outside_rows():
    dccu_targets = _require_helper(dccu_train, "dccu_correction_targets")
    model = _toy_model()

    full_cls = torch.zeros(3, FEATURE_DIM)
    full_cls[:, 0] = 1.0
    full_cls[:, 1] = 0.4
    full_cls[:, 2] = -0.2
    labels = torch.tensor([10, 20, 30], dtype=torch.long)

    crops = torch.zeros(3, ACTION_COUNT, FEATURE_DIM)
    crops[:, :, 0] = 1.0
    crops[1, 4, 0] = -0.1
    crops[1, 4, 1] = 1.0
    crops[2, 7, 0] = -0.1
    crops[2, 7, 1] = 1.0

    targets, groups, _ = dccu_targets(model, full_cls, crops, labels)

    assert groups.tolist() == [0, 1, 2]
    assert torch.equal(targets[0], torch.zeros(ACTION_COUNT))
    assert torch.equal(targets[2], torch.zeros(ACTION_COUNT))
    assert int(targets[1].sum().item()) == 1
    assert targets[1, 4].item() == 1


def test_balanced_group_sampler_returns_four_challenger_and_four_other_rows():
    sampler_cls = _require_helper(dccu_train, "BalancedGroupSampler")
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    first = sampler_cls(groups=groups, seed=7, batch_size=8, challenger_per_batch=4)
    second = sampler_cls(groups=groups, seed=7, batch_size=8, challenger_per_batch=4)
    first_batches = [first.next_batch() for _ in range(6)]
    second_batches = [second.next_batch() for _ in range(6)]

    for left, right in zip(first_batches, second_batches, strict=True):
        assert torch.equal(left, right)
        assert left.shape == (8,)
        batch_groups = groups.index_select(0, left)
        assert int(batch_groups.eq(1).sum().item()) == 4
        assert int(batch_groups.ne(1).sum().item()) == 4


def test_static_best_uses_natural_distribution_targets_not_sampled_stream():
    static_best = _require_helper(dccu_train, "static_best_action_from_natural_targets")
    targets = torch.zeros(6, ACTION_COUNT)
    targets[0:2, 2] = 1
    targets[2:5, 5] = 1

    action, summary = static_best(targets)

    assert action == 5
    assert summary["count"] == 6
    assert summary["mean_target_utility"][5] > summary["mean_target_utility"][2]


def test_group_safety_gate_requires_challenger_trigger_and_low_leader_damage():
    group_safety_gate = _require_helper(dccu_eval, "group_safety_gate")
    passing = {
        "leader": {"count": 100, "trigger": 10, "abstain": 90, "corrected": 0, "damaged": 2, "net": -2},
        "challenger": {"count": 50, "trigger": 30, "abstain": 20, "corrected": 12, "damaged": 0, "net": 12},
        "outside": {"count": 20, "trigger": 2, "abstain": 18, "corrected": 0, "damaged": 1, "net": -1},
    }
    result = group_safety_gate(passing)
    assert result["passed"] is True
    assert result["gates"]["challenger_trigger_gt_leader_trigger"] is True
    assert result["gates"]["leader_damage_lt_challenger_corrections"] is True
    assert result["gates"]["net_positive"] is True
    assert result["gates"]["has_trigger_and_abstain"] is True

    failing = dict(passing)
    failing["leader"] = dict(passing["leader"], trigger=80, damaged=20)
    result = group_safety_gate(failing)
    assert result["passed"] is False
    assert result["gates"]["challenger_trigger_gt_leader_trigger"] is False
    assert result["gates"]["leader_damage_lt_challenger_corrections"] is False


def test_hash_random_uses_normalized_cub_relative_path():
    absolute = "/warehouse/CUB_200_2011/images/005.Crested_Auklet/a.jpg"
    windows = r"C:\warehouse\CUB_200_2011\images\005.Crested_Auklet\a.jpg"
    assert dccu_eval.cub_relative_image_path(absolute) == "005.Crested_Auklet/a.jpg"
    assert dccu_eval.cub_relative_image_path(windows) == "005.Crested_Auklet/a.jpg"
    first, first_sha = dccu_eval.hash_random_actions(
        [absolute], torch.tensor([4]), torch.tensor([9])
    )
    second, second_sha = dccu_eval.hash_random_actions(
        [windows], torch.tensor([4]), torch.tensor([9])
    )
    assert torch.equal(first, second)
    assert first_sha == second_sha
    assert 0 <= int(first.item()) < ACTION_COUNT


def test_paired_bootstrap_uses_classwise_differences():
    full = torch.ones(50)
    other = torch.zeros(50)
    matrix = torch.randint(0, 50, (100, 50), generator=torch.Generator().manual_seed(7))
    result = dccu_eval.paired_comparison(full, other, matrix)
    assert result["observed_pp"] == 100.0
    assert result["ci95"] == [100.0, 100.0]
