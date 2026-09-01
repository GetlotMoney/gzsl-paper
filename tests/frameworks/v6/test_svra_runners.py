import inspect
from dataclasses import dataclass

import pytest
import torch

from model.frameworks.v6 import evaluate_svra_gate0 as svra_eval
from model.frameworks.v6 import train_svra_gate0 as svra_train
from model.frameworks.v6.svra import (
    ACTION_COUNT,
    FEATURE_DIM,
    ParentRiskArbiter,
    ParentRiskCeilingArbiter,
    SemanticVisualRiskArbiter,
)


EXPECTED_EVAL_SCHEMA = "gzsl-paper.v6-svra-gate0-eval.v1"


def test_svra_eval_schema_and_config_are_zero_crop():
    config, _ = svra_eval.load_config("config/tries/v6_try_003_svra_gate0_eval.yaml")

    assert svra_eval.SCHEMA == EXPECTED_EVAL_SCHEMA
    assert "clip_checkpoint" not in config
    assert "crop_batch_size" not in config
    assert "oracle_receipt" not in config
    assert config["official_test_loaded"] is False
    assert config["unseen_images_used_for_gradient"] is False
    assert config["pclr_online_inference"] is False


def test_svra_eval_source_has_no_raw_crop_or_clip_encode_path():
    source = inspect.getsource(svra_eval)

    forbidden = [
        "from PIL",
        "import clip",
        "Image.open",
        "encode_image",
        "image_paths",
        "crop_boxes",
        "crop_features",
        "all25_crop",
    ]
    for token in forbidden:
        assert token not in source


def test_apply_pair_swap_only_exchanges_parent_top2_logits():
    parent = torch.tensor(
        [
            [9.0, 8.0, 0.0],
            [1.0, 7.0, 6.0],
            [5.0, 4.0, 3.0],
        ]
    )
    top2 = torch.tensor([[0, 1], [1, 2], [0, 1]])
    swap = torch.tensor([False, True, True])

    logits = svra_eval.apply_pair_swap(parent, top2, swap)

    assert logits.tolist() == [
        [9.0, 8.0, 0.0],
        [1.0, 6.0, 7.0],
        [4.0, 5.0, 3.0],
    ]
    assert parent.tolist() == [
        [9.0, 8.0, 0.0],
        [1.0, 7.0, 6.0],
        [5.0, 4.0, 3.0],
    ]


@dataclass(frozen=True)
class _Trace:
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
    probabilities: dict[str, torch.Tensor]


def test_build_condition_logits_matches_svra_controls():
    parent = torch.tensor(
        [
            [4.0, 3.0, 1.0],
            [5.0, 4.0, 0.0],
            [6.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ]
    )
    top2 = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    base = dict(
        name="full",
        parent_logits=parent,
        top2=top2,
        leader_ids=torch.tensor([0, 0, 0, 0]),
        challenger_ids=torch.tensor([1, 1, 1, 1]),
        actions=torch.tensor([0, 1, 2, 3]),
        trigger=torch.tensor([True, False, True, False]),
        parent_stats4=torch.zeros(4, 4),
        risk_features13=torch.zeros(4, 13),
        policy_scores=None,
    )
    full = _Trace(
        **base,
        probabilities={
            "triggered4d": torch.tensor([0.6, 0.8, 0.4, 0.9]),
            "all_row4d": torch.tensor([0.4, 0.7, 0.7, 0.2]),
            "ceiling13d": torch.tensor([0.9, 0.9, 0.9, 0.9]),
        },
    )
    s_off = _Trace(**{**base, "name": "s_off", "trigger": torch.zeros(4, dtype=torch.bool), "probabilities": full.probabilities})
    v_off = _Trace(**{**base, "name": "v_off", "trigger": torch.ones(4, dtype=torch.bool), "probabilities": full.probabilities})

    logits, swaps = svra_eval.build_condition_logits(full, s_off, v_off)

    assert swaps["full"].tolist() == [True, False, False, False]
    assert swaps["always_swap"].tolist() == [True, False, True, False]
    assert swaps["triggered4d_no_trigger"].tolist() == [True, True, False, True]
    assert swaps["all_row4d_no_trigger"].tolist() == [False, True, True, False]
    assert swaps["ceiling13d"].tolist() == [True, False, True, False]
    assert torch.equal(logits["i_off"], parent)


def test_group_safety_gate_requires_challenger_trigger_and_low_leader_damage():
    passing = {
        "leader": {"count": 100, "trigger": 10, "abstain": 90, "corrected": 0, "damaged": 2, "net": -2},
        "challenger": {"count": 50, "trigger": 30, "abstain": 20, "corrected": 12, "damaged": 0, "net": 12},
        "outside": {"count": 20, "trigger": 2, "abstain": 18, "corrected": 0, "damaged": 1, "net": -1},
    }

    result = svra_eval.group_safety_gate(passing)

    assert result["passed"] is True
    assert result["gates"]["challenger_trigger_gt_leader_trigger"] is True
    assert result["gates"]["leader_damage_lt_challenger_corrections"] is True
    assert result["gates"]["net_positive"] is True
    assert result["gates"]["has_trigger_and_abstain"] is True

    failing = dict(passing)
    failing["leader"] = dict(passing["leader"], trigger=80, damaged=20)
    result = svra_eval.group_safety_gate(failing)
    assert result["passed"] is False
    assert result["gates"]["challenger_trigger_gt_leader_trigger"] is False
    assert result["gates"]["leader_damage_lt_challenger_corrections"] is False


def test_paired_bootstrap_uses_classwise_differences():
    full = torch.ones(50)
    other = torch.zeros(50)
    matrix = torch.randint(0, 50, (100, 50), generator=torch.Generator().manual_seed(7))

    result = svra_eval.paired_comparison(full, other, matrix)

    assert result["observed_pp"] == 100.0
    assert result["ci95"] == [100.0, 100.0]


def test_load_checkpoint_requires_svra_combined_identity(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    payload = {
        "schema_version": svra_eval.CHECKPOINT_SCHEMA,
        "condition_id": "SVRA_FULL",
        "code_commit": "abc",
        "config_sha256": "cfg",
        "action_bundle_manifest_sha256": "bundle",
        "policy_state_dict": {},
        "trigger_arbiter4_state_dict": {},
        "trigger_arbiter13_ceiling_state_dict": {},
        "allrow_arbiter4_control_state_dict": {},
    }
    torch.save(payload, checkpoint_path)
    monkeypatch.setattr(svra_eval, "sha256_file", lambda path: "ckpt")

    loaded = svra_eval.load_checkpoint(
        {
            "path": str(checkpoint_path),
            "sha256": "ckpt",
            "training_commit": "abc",
            "train_config_sha256": "cfg",
        },
        expected_commit="abc",
        expected_bundle_sha256="bundle",
        expected_train_config_sha256="cfg",
    )

    assert loaded["condition_id"] == "SVRA_FULL"


def test_load_checkpoint_rejects_non_svra_schema(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "schema_version": "gzsl-paper.v6-role-tripool-gate0-train.v1",
            "condition_id": "ROLE_TRIPOOL_FULL",
            "code_commit": "abc",
            "config_sha256": "cfg",
            "action_bundle_manifest_sha256": "bundle",
            "policy_state_dict": {},
            "trigger_arbiter4_state_dict": {},
            "trigger_arbiter13_ceiling_state_dict": {},
            "allrow_arbiter4_control_state_dict": {},
        },
        checkpoint_path,
    )
    monkeypatch.setattr(svra_eval, "sha256_file", lambda path: "ckpt")

    with pytest.raises(svra_eval.SVRAEvalError):
        svra_eval.load_checkpoint(
            {
                "path": str(checkpoint_path),
                "sha256": "ckpt",
                "training_commit": "abc",
                "train_config_sha256": "cfg",
            },
            expected_commit="abc",
            expected_bundle_sha256="bundle",
            expected_train_config_sha256="cfg",
        )


class _Assets:
    def __init__(self, role_embeddings: torch.Tensor, name_embeddings: torch.Tensor) -> None:
        self.role_embeddings = role_embeddings
        self.name_embeddings = name_embeddings


class _EvalView:
    def __init__(self, cls: torch.Tensor, patches: torch.Tensor) -> None:
        self.cls = cls
        self.patches = patches
        self.size = int(cls.shape[0])
        self.include_patches_calls: list[bool] = []

    def batch(self, rows, *, include_patches: bool, as_torch: bool, device: torch.device):
        assert as_torch is True
        self.include_patches_calls.append(bool(include_patches))
        index = torch.as_tensor(rows, dtype=torch.long)
        result = {
            "cls": self.cls.index_select(0, index).to(device),
        }
        if include_patches:
            result["patches"] = self.patches.index_select(0, index).to(device)
        return result


def _train_config() -> svra_train.Gate0TrainConfig:
    return svra_train.Gate0TrainConfig(
        schema_version=svra_train.TRAIN_SCHEMA,
        experiment_id="V6-TRY-003-SVRA-GATE0-FULL",
        condition_id="SVRA_FULL",
        text_manifest="text.json",
        text_manifest_sha256="0" * 64,
        role_tensor="role.pt",
        role_tensor_sha256="1" * 64,
        name_tensor="name.pt",
        name_tensor_sha256="2" * 64,
        patch_manifest="patch.json",
        patch_manifest_sha256="3" * 64,
        cls_tensor="cls.pt",
        cls_tensor_sha256="4" * 64,
        patch_tensor="patch.npy",
        patch_tensor_sha256="5" * 64,
        action_bundle_manifest="bundle.json",
        action_bundle_manifest_sha256="6" * 64,
        dev_train_manifest_sha256="7" * 64,
        dev_eval_manifest_sha256="8" * 64,
        dev_eval_oracle_manifest_sha256="9" * 64,
        att_splits_mat_path="att_splits.mat",
        trainval_count=7057,
        oracle_receipt="oracle.json",
        oracle_receipt_sha256="a" * 64,
        action_geometry_sha256=svra_train.ACTION_GEOMETRY_SHA256,
        output_dir="/tmp/svra",
        stage1_updates=2,
        stage2_updates=2,
    )


def test_train_payload_loads_into_eval_core_and_freezes_all_condition_logits(tmp_path):
    generator = torch.Generator().manual_seed(7)
    class_ids = torch.arange(4)
    names = torch.nn.functional.normalize(
        torch.randn(4, FEATURE_DIM, generator=generator), dim=-1
    )
    roles = torch.nn.functional.normalize(
        torch.randn(4, 8, FEATURE_DIM, generator=generator), dim=-1
    )
    model = SemanticVisualRiskArbiter(roles, names, class_ids, seed=7)
    trig4 = ParentRiskArbiter()
    ceiling13 = ParentRiskCeilingArbiter()
    allrow4 = ParentRiskArbiter()
    trig4.output.bias.data.fill_(1.0)
    ceiling13.output.bias.data.fill_(2.0)
    allrow4.output.bias.data.fill_(-1.0)
    config = _train_config()
    grad = {
        "output.weight": svra_train.GradientGateReport(True, True, 1.0, 1.0),
    }
    target_plan = svra_train.EAACTargetPlan(
        targets26=torch.zeros(4, dtype=torch.long),
        groups=torch.tensor([0, 1, 1, 2]),
        margins=torch.zeros(4, ACTION_COUNT),
        dense_targets=torch.zeros(4, ACTION_COUNT, dtype=torch.bool),
        stats={"target_sha256": "target"},
    )
    trigger_plan = svra_train.TriggerPlan(
        triggered_rows=torch.tensor([1, 2]),
        labels=torch.tensor([1.0, 1.0]),
        selected_actions=torch.tensor([0, 1]),
        features4=torch.zeros(2, 4),
        features13=torch.zeros(2, 13),
        allrow_rows=torch.arange(4),
        allrow_labels=torch.tensor([0.0, 1.0, 1.0, 0.0]),
        allrow_features4=torch.zeros(4, 4),
        stats={
            "trigger_rows_sha256": "rows",
            "trigger_label_sha256": "labels",
        },
    )
    train_targets = svra_train.TrainSubsetTargets(
        labels=torch.tensor([0, 1, 2, 3]),
        class_ids=class_ids,
        crop_features=torch.zeros(4, ACTION_COUNT, FEATURE_DIM),
        labels_path="labels.pt",
        class_ids_path="class_ids.pt",
        crop_features_path="crop_features.pt",
        labels_sha256="labels",
        class_ids_sha256="classes",
        crop_features_sha256="crops",
    )
    sampler = svra_train.stage1_sampler_from_groups(
        torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]),
        seed=7,
    )
    payload = svra_train.build_checkpoint_payload(
        model,
        {
            "arbiter4": trig4,
            "arbiter13": ceiling13,
            "allrow4": allrow4,
            "loss_summary": {},
            "gradient_gates": {},
            "samplers": {
                "triggered_shared": {"batch_trace_sha256": "trace"},
                "allrow_4d": {"batch_trace_sha256": "alltrace"},
            },
        },
        config=config,
        config_sha256="cfg",
        commit="abc",
        train_targets=train_targets,
        target_plan=target_plan,
        trigger_plan=trigger_plan,
        stage1_losses=[1.0, 0.5],
        stage1_grad1=grad,
        stage1_grad2=grad,
        stage1_sampler=sampler,
        stage1_sampled_stats={},
        asset_receipt={},
        oracle_receipt={"path": "oracle.json", "sha256": "a" * 64},
    )
    checkpoint_path = tmp_path / "svra_gate0_combined.pt"
    torch.save(payload, checkpoint_path)
    checkpoint_sha = svra_eval.sha256_file(checkpoint_path)

    checkpoint = svra_eval.load_checkpoint(
        {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "training_commit": "abc",
            "train_config_sha256": "cfg",
        },
        expected_commit="abc",
        expected_bundle_sha256="6" * 64,
        expected_train_config_sha256="cfg",
    )
    loaded = svra_eval.instantiate_model(
        _Assets(roles, names),
        class_ids,
        checkpoint,
        torch.device("cpu"),
    )
    cls = torch.nn.functional.normalize(
        torch.randn(3, FEATURE_DIM, generator=generator), dim=-1
    )
    patches = torch.nn.functional.normalize(
        torch.randn(3, 576, FEATURE_DIM, generator=generator), dim=-1
    )
    view = _EvalView(cls, patches)

    full = svra_eval.freeze_policy(
        loaded,
        view,
        device=torch.device("cpu"),
        batch_size=2,
        name="full",
    )
    s_off = svra_eval.freeze_policy(
        loaded,
        view,
        device=torch.device("cpu"),
        batch_size=2,
        name="s_off",
        semantic_off=True,
    )
    v_off = svra_eval.freeze_policy(
        loaded,
        view,
        device=torch.device("cpu"),
        batch_size=2,
        name="v_off",
        visual_off=True,
    )
    logits, swaps = svra_eval.build_condition_logits(full, s_off, v_off)

    assert False in view.include_patches_calls
    assert set(logits) == {
        "parent",
        "full",
        "s_off",
        "v_off",
        "i_off",
        "always_swap",
        "triggered4d_no_trigger",
        "all_row4d_no_trigger",
        "ceiling13d",
    }
    assert set(swaps) == set(logits) - {"parent"}
    assert all(value.shape == (3, 4) for value in logits.values())
    assert full.parent_stats4.shape == (3, 4)
    assert full.risk_features13 is not None
    assert full.risk_features13.shape == (3, 13)
    assert checkpoint_path.name == "svra_gate0_combined.pt"
