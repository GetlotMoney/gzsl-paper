import inspect

import pytest
import torch

from model.frameworks.v6 import evaluate_joint_svra_precheck as joint_eval
from model.frameworks.v6.svra import FEATURE_DIM, SemanticVisualRiskArbiter


def test_joint_precheck_config_binds_nested_official_protocol():
    config, _ = joint_eval.load_config("config/tries/v6_try_004_joint_svra_precheck_eval.yaml")

    assert joint_eval.SCHEMA == "gzsl-paper.v6-joint-svra-precheck-eval.v1"
    assert config["test_used_for_selection"] is True
    assert config["test_used_for_hyperparameter_selection"] is False
    assert config["nested_official_test_selection"] is True
    assert config["strict_blind_claim"] is False
    assert config["official_test_loaded"] is True
    assert "clip_checkpoint" not in config
    assert "oracle_receipt" not in config


def test_joint_precheck_source_has_no_crop_or_online_clip_path():
    source = inspect.getsource(joint_eval)

    forbidden = [
        "from PIL",
        "import clip",
        "Image.open",
        "encode_image",
        "image_paths",
        "crop_boxes",
        "all25_crop",
    ]
    for token in forbidden:
        assert token not in source


def test_apply_pair_swap_only_exchanges_parent_top2_logits():
    parent = torch.tensor([[9.0, 8.0, 0.0], [1.0, 7.0, 6.0]])
    top2 = torch.tensor([[0, 1], [1, 2]])
    swap = torch.tensor([False, True])

    logits = joint_eval.apply_pair_swap(parent, top2, swap)

    assert logits.tolist() == [[9.0, 8.0, 0.0], [1.0, 6.0, 7.0]]
    assert parent.tolist() == [[9.0, 8.0, 0.0], [1.0, 7.0, 6.0]]


def test_real_svra_model_freeze_split_is_zero_crop_and_exact_trigger_equivalent():
    class_count = 4
    roles = torch.zeros(class_count, 8, FEATURE_DIM)
    names = torch.zeros(class_count, FEATURE_DIM)
    for idx in range(class_count):
        names[idx, idx] = 1.0
        roles[idx, :, idx] = 1.0
    class_ids = torch.arange(class_count)
    model = SemanticVisualRiskArbiter(roles, names, class_ids, seed=7)
    features = torch.zeros(3, FEATURE_DIM)
    features[:, 0] = 1.0
    patches = features[:, None, :].expand(-1, 576, -1).clone()

    frozen = joint_eval.freeze_split(
        model,
        features,
        patches,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert frozen.logits.shape == (3, class_count)
    assert frozen.actions.shape == (3,)
    assert frozen.trigger.shape == (3,)
    assert frozen.swap.shape == (3,)
    assert frozen.soft_hard_trigger_equal is True


def test_paired_h_comparison_uses_seen_and_unseen_class_vectors():
    full = {
        "S": 80.0,
        "U": 60.0,
        "H": 68.57142857142857,
        "seen_per_class": torch.tensor([1.0, 0.6]),
        "unseen_per_class": torch.tensor([0.5, 0.7]),
    }
    other = {
        "S": 60.0,
        "U": 40.0,
        "H": 48.0,
        "seen_per_class": torch.tensor([0.6, 0.6]),
        "unseen_per_class": torch.tensor([0.4, 0.4]),
    }
    seen_matrix = torch.tensor([[0, 1], [0, 0]])
    unseen_matrix = torch.tensor([[0, 1], [1, 1]])

    result = joint_eval.paired_h_comparison(full, other, seen_matrix, unseen_matrix)

    assert result["observed_pp"] == pytest.approx(20.57142857142857)
    assert result["ci95"][0] > 0


def test_load_checkpoint_requires_expected_condition_and_schema(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "schema_version": joint_eval.CHECKPOINT_SCHEMA,
            "condition_id": joint_eval.FULL_CONDITION,
            "code_commit": "abc",
            "config_sha256": "cfg",
            "state_dict": {},
        },
        path,
    )
    monkeypatch.setattr(joint_eval, "sha256_file", lambda _: "ckpt")

    loaded = joint_eval.load_checkpoint(
        {
            "path": str(path),
            "sha256": "ckpt",
            "training_commit": "abc",
            "train_config_sha256": "cfg",
        },
        expected_commit="abc",
        expected_condition=joint_eval.FULL_CONDITION,
    )

    assert loaded["condition_id"] == joint_eval.FULL_CONDITION

    with pytest.raises(joint_eval.JointSVRAPrecheckError):
        joint_eval.load_checkpoint(
            {
                "path": str(path),
                "sha256": "ckpt",
                "training_commit": "abc",
                "train_config_sha256": "cfg",
            },
            expected_commit="abc",
            expected_condition=joint_eval.NO_JOINT_CONDITION,
        )


def test_group_safety_gate_requires_challenger_trigger_and_positive_net():
    passing = {
        "leader": {"count": 100, "trigger": 10, "abstain": 90, "corrected": 0, "damaged": 2, "net": -2},
        "challenger": {"count": 50, "trigger": 30, "abstain": 20, "corrected": 12, "damaged": 0, "net": 12},
        "outside": {"count": 20, "trigger": 2, "abstain": 18, "corrected": 0, "damaged": 1, "net": -1},
    }
    result = joint_eval.group_safety_gate(passing)
    assert result["passed"] is True

    failing = dict(passing)
    failing["challenger"] = dict(passing["challenger"], trigger=1, corrected=0)
    result = joint_eval.group_safety_gate(failing)
    assert result["passed"] is False
