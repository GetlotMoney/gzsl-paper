import dataclasses
import json

import torch
import torch.nn.functional as F

from model.frameworks.v5 import evaluate_scaa_gate0 as scaa_eval
from model.frameworks.v5 import train_scaa_gate0 as scaa_train
from model.frameworks.v5.rwdg import ACTION_COUNT, FEATURE_DIM, RoleWindowDenseGlimpse
from tools.runtime import sha256_file


EXPECTED_TRAIN_SCHEMA = "gzsl-paper.v5-scaa-gate0-train.v1"
EXPECTED_EVAL_SCHEMA = "gzsl-paper.v5-scaa-gate0-eval.v1"


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


def test_scaa_runner_schemas_are_not_rwdg_or_dccu_schemas():
    assert scaa_train.TRAIN_SCHEMA == EXPECTED_TRAIN_SCHEMA
    assert scaa_eval.SCHEMA == EXPECTED_EVAL_SCHEMA


def test_train_config_and_exact_oracle_receipt_contract(tmp_path):
    config, _ = scaa_train.load_strict_config("config/tries/v5_try_009_scaa_gate0_train.json")
    scaa_train.validate_config(config)
    receipt_path = tmp_path / "oracle.json"
    receipt_path.write_text(json.dumps(_receipt(config)), encoding="utf-8")
    bound = dataclasses.replace(
        config,
        oracle_receipt=str(receipt_path),
        oracle_receipt_sha256=sha256_file(receipt_path),
    )
    validated = scaa_train.load_and_validate_oracle_receipt(bound)
    assert validated["gate"]["passed"]
    assert validated["gate"]["oracle_gain_pp"] == 16.068547166940476


def test_eval_oracle_parser_binds_every_identity(tmp_path):
    config, _ = scaa_train.load_strict_config("config/tries/v5_try_009_scaa_gate0_train.json")
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
    loaded = scaa_eval.load_oracle_receipt(str(receipt_path), sha256_file(receipt_path), eval_config)
    assert scaa_eval.oracle_gate_from_receipt(loaded)["passed"]


def test_scaa_signed_targets_encode_damage_correction_and_neutral_rows():
    scaa_targets = _require_helper(scaa_train, "scaa_signed_targets")
    model = _toy_model()

    full_cls = torch.zeros(3, FEATURE_DIM)
    full_cls[:, 0] = 1.0
    full_cls[:, 1] = 0.4
    full_cls[:, 2] = -0.2
    labels = torch.tensor([10, 20, 30], dtype=torch.long)

    crops = torch.zeros(3, ACTION_COUNT, FEATURE_DIM)
    crops[:, :, 0] = 1.0
    crops[0, 4, 0] = -0.1
    crops[0, 4, 1] = 1.0
    crops[1, 4, 0] = -0.1
    crops[1, 4, 1] = 1.0
    crops[2, 7, 0] = -0.1
    crops[2, 7, 1] = 1.0

    targets, groups, _ = scaa_targets(model, full_cls, crops, labels)

    assert groups.tolist() == [0, 1, 2]
    assert targets[0, 4].item() == -1
    assert int(targets[0].eq(-1).sum().item()) == 1
    assert int(targets[0].eq(1).sum().item()) == 0
    assert torch.equal(targets[2], torch.zeros(ACTION_COUNT))
    assert int(targets[1].sum().item()) == 1
    assert targets[1, 4].item() == 1


def test_signed_advantage_loss_is_tanh_mse_not_bce():
    loss_fn = _require_helper(scaa_train, "signed_advantage_loss")
    logits = torch.zeros(2, ACTION_COUNT)
    targets = torch.zeros(2, ACTION_COUNT)
    logits[0, 1] = 0.5
    logits[0, 2] = -0.5
    logits[1, 0] = 1.0
    logits[1, 1] = -1.0
    logits[1, 2] = 0.25
    targets[0, 1] = 1.0
    targets[0, 2] = -1.0
    targets[1, 0] = 1.0
    targets[1, 1] = -1.0

    expected = F.mse_loss(torch.tanh(logits), targets.float())

    assert torch.allclose(loss_fn(logits, targets), expected)
    assert loss_fn(torch.zeros(1, ACTION_COUNT), torch.zeros(1, ACTION_COUNT)).item() == 0.0


def test_scaa_loss_reaches_registered_projections_after_zero_head_moves():
    loss_fn = _require_helper(scaa_train, "signed_advantage_loss")
    collect_gradient_report = _require_helper(scaa_train, "collect_gradient_report")
    roles = torch.zeros(5, 8, FEATURE_DIM)
    names = torch.zeros(5, FEATURE_DIM)
    for index in range(5):
        names[index, index] = 1.0
        roles[index, :, index] = 1.0
    model = RoleWindowDenseGlimpse(roles, names, torch.arange(5), seed=7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    full = torch.zeros(4, FEATURE_DIM)
    full[:, 0] = 1.0
    patches = torch.zeros(4, 576, FEATURE_DIM)
    patches[:, :, 0] = 1.0
    target = torch.zeros(4, ACTION_COUNT)
    target[:, 0] = 1.0
    target[:, 1] = -1.0

    first = model.utility_state(full, patches)
    loss_fn(first.utility_logits, target).backward()
    first_report = collect_gradient_report(model)
    assert first_report["w_u"].finite and first_report["w_u"].nonzero
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = model.utility_state(full, patches)
    loss_fn(second.utility_logits, target).backward()
    second_report = collect_gradient_report(model)

    for name in ("W_r", "W_n", "W_x", "W_vx", "W_vr", "W_h", "w_u"):
        assert second_report[name].finite, name
        assert second_report[name].nonzero, name


def test_balanced_group_sampler_returns_four_challenger_and_four_other_rows_with_signed_trace():
    sampler_cls = _require_helper(scaa_train, "BalancedGroupSampler")
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    signed_targets = torch.zeros(groups.numel(), ACTION_COUNT)
    signed_targets[0, 1] = -1
    signed_targets[1, 2] = -1
    signed_targets[3, 3] = 1
    signed_targets[5, 4] = 1

    first = sampler_cls(groups=groups, seed=7, batch_size=8, challenger_per_batch=4)
    second = sampler_cls(groups=groups, seed=7, batch_size=8, challenger_per_batch=4)
    first_batches = [first.next_batch() for _ in range(12)]
    second_batches = [second.next_batch() for _ in range(12)]
    sampled = []

    for left, right in zip(first_batches, second_batches, strict=True):
        assert torch.equal(left, right)
        assert left.shape == (8,)
        batch_groups = groups.index_select(0, left)
        assert int(batch_groups.eq(1).sum().item()) == 4
        assert int(batch_groups.ne(1).sum().item()) == 4
        sampled.append(signed_targets.index_select(0, left))

    trace = torch.cat(sampled)
    assert int(trace.eq(1).sum().item()) > 0
    assert int(trace.eq(-1).sum().item()) > 0


def test_signed_static_best_uses_natural_signed_mean_not_positive_count():
    static_best = _require_helper(scaa_train, "static_best_action_from_natural_targets")
    targets = torch.zeros(4, ACTION_COUNT)
    targets[0:2, 2] = 1
    targets[2:4, 2] = -1
    targets[0, 5] = 1

    action, summary = static_best(targets)

    assert action == 5
    assert summary["count"] == 4
    assert summary["mean_signed_target"][2] == 0.0
    assert summary["mean_signed_target"][5] > summary["mean_signed_target"][2]


def test_group_safety_gate_requires_challenger_trigger_and_low_leader_damage():
    group_safety_gate = _require_helper(scaa_eval, "group_safety_gate")
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
    assert scaa_eval.cub_relative_image_path(absolute) == "005.Crested_Auklet/a.jpg"
    assert scaa_eval.cub_relative_image_path(windows) == "005.Crested_Auklet/a.jpg"
    first, first_sha = scaa_eval.hash_random_actions(
        [absolute], torch.tensor([4]), torch.tensor([9])
    )
    second, second_sha = scaa_eval.hash_random_actions(
        [windows], torch.tensor([4]), torch.tensor([9])
    )
    assert torch.equal(first, second)
    assert first_sha == second_sha
    assert 0 <= int(first.item()) < ACTION_COUNT


def test_paired_bootstrap_uses_classwise_differences():
    full = torch.ones(50)
    other = torch.zeros(50)
    matrix = torch.randint(0, 50, (100, 50), generator=torch.Generator().manual_seed(7))
    result = scaa_eval.paired_comparison(full, other, matrix)
    assert result["observed_pp"] == 100.0
    assert result["ci95"] == [100.0, 100.0]
