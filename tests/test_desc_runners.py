import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from model.frameworks.v6 import evaluate_desc_precheck as desc_eval
from model.frameworks.v6.svra import (
    ACTION_COUNT,
    FEATURE_DIM,
    HIDDEN_DIM,
    SemanticVisualRiskArbiter,
    stable_top2_by_logit_then_class_id,
)
from tools.runtime import sha256_file


class FakeDESCModel(torch.nn.Module):
    def __init__(self, class_count: int = 4) -> None:
        super().__init__()
        self.register_buffer("class_ids", torch.arange(class_count), persistent=False)
        self.class_count = class_count

    def desc_forward(
        self,
        full_cls,
        patch_tokens,
        *,
        semantic_off=False,
        visual_off=False,
        interaction_off=False,
    ):
        parent_logits = full_cls[:, : self.class_count].float().clone()
        top2 = stable_top2_by_logit_then_class_id(parent_logits, self.class_ids)
        patch_scalar = patch_tokens[:, :, 0].mean(dim=1)
        semantic_delta = torch.full_like(patch_scalar, -0.25 if semantic_off else 0.25)
        visual_delta = torch.full_like(patch_scalar, -0.50 if visual_off else 0.50)
        swap_logit = patch_scalar + semantic_delta + visual_delta
        evidence_pool = torch.zeros(full_cls.shape[0], HIDDEN_DIM, device=full_cls.device)
        evidence_pool[:, 0] = swap_logit
        action_logits = torch.zeros(full_cls.shape[0], ACTION_COUNT, device=full_cls.device)
        action_logits[:, 3] = swap_logit
        action_logits[:, 7] = -swap_logit
        swap = torch.zeros_like(swap_logit, dtype=torch.bool) if interaction_off else swap_logit > 0
        logits = desc_eval.apply_pair_swap(parent_logits, top2, swap)
        return SimpleNamespace(
            logits=logits,
            parent_logits=parent_logits,
            top2=top2,
            swap_logit=swap_logit,
            action_logits=action_logits,
            evidence_pool=evidence_pool,
            swapped=swap,
        )


def test_desc_eval_config_binds_official_nested_protocol():
    config, _ = desc_eval.load_config("config/tries/v6_try_005_desc_precheck_eval.yaml")

    assert config["schema_version"] == desc_eval.SCHEMA
    assert config["experiment_id"] == "V6-TRY-005-DESC-PRECHECK-EVAL"
    assert config["test_used_for_selection"] is True
    assert config["test_used_for_hyperparameter_selection"] is False
    assert config["nested_official_test_selection"] is True
    assert config["strict_blind_claim"] is False
    assert config["unseen_images_used_for_gradient"] is False
    assert config["pclr_online_inference"] is False
    assert config["module_contract_margin"] == 1.0
    assert config["support_control_margin"] == 0.5
    assert set(desc_eval.CONDITION_TO_CONFIG_KEY) == {"full", "no_action_aux", "parent_only"}


def test_desc_eval_source_has_no_runtime_image_or_online_clip_path():
    source = open("model/frameworks/v6/evaluate_desc_precheck.py", encoding="utf-8").read()

    forbidden = [
        "from PIL",
        "import clip",
        "Image.open",
        "encode_image",
        "image_paths",
        "crop_boxes",
        "all25_crop",
        "eval_oracle",
    ]
    for token in forbidden:
        assert token not in source


def test_freeze_split_accepts_real_desc_payload_and_applies_swap_logit_threshold():
    model = FakeDESCModel(class_count=4)
    cls = torch.tensor(
        [
            [4.0, 3.0, 1.0, 0.0],
            [1.0, 4.0, 3.0, 0.0],
            [4.0, 2.0, 1.0, 0.0],
        ]
    )
    cls = F.pad(cls, (0, FEATURE_DIM - cls.shape[1]))
    patches = torch.zeros(3, 576, FEATURE_DIM)
    patches[0, :, 0] = 0.10
    patches[1, :, 0] = -2.00
    patches[2, :, 0] = 0.20

    frozen = desc_eval.freeze_split(model, cls, patches, device=torch.device("cpu"), batch_size=2)

    assert frozen.logits.shape == (3, 4)
    assert frozen.swap_logit.shape == (3,)
    assert frozen.action_logits.shape == (3, ACTION_COUNT)
    assert frozen.evidence_pool.shape == (3, HIDDEN_DIM)
    assert torch.equal(frozen.swap, frozen.swap_logit > 0)
    assert frozen.logits.argmax(dim=1).tolist() == [1, 1, 1]
    assert frozen.actions.tolist() == [3, 7, 3]


def test_freeze_split_accepts_real_semantic_visual_risk_arbiter_direct_payload():
    generator = torch.Generator().manual_seed(11)
    names = F.normalize(torch.randn(4, FEATURE_DIM, generator=generator), dim=-1)
    roles = F.normalize(torch.randn(4, 8, FEATURE_DIM, generator=generator), dim=-1)
    model = SemanticVisualRiskArbiter(roles, names, torch.arange(4), seed=7)
    cls = F.normalize(torch.randn(3, FEATURE_DIM, generator=generator), dim=-1)
    patches = F.normalize(torch.randn(3, 576, FEATURE_DIM, generator=generator), dim=-1)

    frozen = desc_eval.freeze_split(model, cls, patches, device=torch.device("cpu"), batch_size=2)
    parent_only = desc_eval.freeze_split(model, cls, patches, device=torch.device("cpu"), batch_size=2, parent_only=True)

    assert frozen.logits.shape == (3, 4)
    assert frozen.action_logits.shape == (3, ACTION_COUNT)
    assert frozen.evidence_pool.shape == (3, HIDDEN_DIM)
    assert torch.equal(frozen.swap, frozen.swap_logit > 0)
    assert parent_only.evidence_pool.eq(0).all()
    assert parent_only.action_logits.eq(0).all()
    assert torch.equal(parent_only.swap, parent_only.swap_logit > 0)


def test_module_off_changes_desc_payload_without_changing_parent_shape():
    model = FakeDESCModel(class_count=4)
    cls = F.pad(torch.tensor([[4.0, 3.0, 1.0, 0.0], [1.0, 4.0, 3.0, 0.0]]), (0, FEATURE_DIM - 4))
    patches = torch.zeros(2, 576, FEATURE_DIM)
    patches[:, :, 0] = 0.10

    full = desc_eval.freeze_split(model, cls, patches, device=torch.device("cpu"), batch_size=2)
    s_off = desc_eval.freeze_split(model, cls, patches, device=torch.device("cpu"), batch_size=2, semantic_off=True)
    v_off = desc_eval.freeze_split(model, cls, patches, device=torch.device("cpu"), batch_size=2, visual_off=True)
    i_off = desc_eval.freeze_split(model, cls, patches, device=torch.device("cpu"), batch_size=2, interaction_off=True)

    assert not torch.equal(full.swap_logit, s_off.swap_logit)
    assert not torch.equal(full.swap_logit, v_off.swap_logit)
    assert torch.equal(i_off.logits, i_off.parent_logits)
    assert not bool(i_off.swap.any())


def test_metrics_and_paired_h_comparison_use_200_axis_style_predictions():
    labels = desc_eval.OfficialLabels(
        seen=torch.tensor([0, 0, 1, 1]),
        unseen=torch.tensor([2, 2, 3, 3]),
        seen_classes=torch.tensor([0, 1]),
        unseen_classes=torch.tensor([2, 3]),
    )
    full_logits = torch.tensor(
        [
            [9.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 9.0],
            [0.0, 0.0, 0.0, 8.0],
        ]
    )
    other_logits = full_logits.clone()
    other_logits[4] = torch.tensor([9.0, 0.0, 0.0, 0.0])
    common = {
        "parent_logits": torch.zeros(8, 4),
        "top2": torch.tensor([[0, 1]] * 8),
        "swap_logit": torch.zeros(8),
        "action_logits": torch.zeros(8, ACTION_COUNT),
        "evidence_pool": torch.zeros(8, HIDDEN_DIM),
        "actions": torch.zeros(8, dtype=torch.long),
        "swap": torch.zeros(8, dtype=torch.bool),
    }
    full = desc_eval.FrozenCondition(
        "full",
        desc_eval.FrozenSplit(logits=full_logits[:4], **{key: value[:4] for key, value in common.items()}),
        desc_eval.FrozenSplit(logits=full_logits[4:], **{key: value[4:] for key, value in common.items()}),
    )
    other = desc_eval.FrozenCondition(
        "other",
        desc_eval.FrozenSplit(logits=other_logits[:4], **{key: value[:4] for key, value in common.items()}),
        desc_eval.FrozenSplit(logits=other_logits[4:], **{key: value[4:] for key, value in common.items()}),
    )

    full_metrics = desc_eval.condition_metrics(full, labels)
    other_metrics = desc_eval.condition_metrics(other, labels)
    seen_matrix = torch.tensor([[0, 1], [1, 0]])
    unseen_matrix = torch.tensor([[0, 1], [1, 0]])
    comparison = desc_eval.paired_h_comparison(full_metrics, other_metrics, seen_matrix, unseen_matrix)

    assert full_metrics["H"] > other_metrics["H"]
    assert comparison["observed_pp"] > 0


def test_load_checkpoint_requires_desc_schema_condition_and_identity(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    payload = {
        "schema_version": desc_eval.CHECKPOINT_SCHEMA,
        "condition_id": desc_eval.FULL_CONDITION,
        "code_commit": "abc123",
        "config_sha256": "cfgsha",
        "state_dict": {},
        "model_class": "SemanticVisualRiskArbiter",
    }
    torch.save(payload, checkpoint_path)
    spec = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "training_commit": "abc123",
        "train_config_sha256": "cfgsha",
    }

    assert desc_eval.load_checkpoint(spec, expected_commit="abc123", expected_condition=desc_eval.FULL_CONDITION)["condition_id"] == desc_eval.FULL_CONDITION
    with pytest.raises(desc_eval.DESCPrecheckError):
        desc_eval.load_checkpoint(spec, expected_commit="abc123", expected_condition=desc_eval.NO_ACTION_AUX_CONDITION)


def test_action_summary_requires_both_keep_swap_and_diverse_actions():
    split = desc_eval.FrozenSplit(
        logits=torch.zeros(4, 4),
        parent_logits=torch.zeros(4, 4),
        top2=torch.tensor([[0, 1]] * 4),
        swap_logit=torch.tensor([1.0, -1.0, 2.0, -2.0]),
        action_logits=torch.zeros(4, ACTION_COUNT),
        evidence_pool=torch.zeros(4, HIDDEN_DIM),
        actions=torch.tensor([3, 7, 3, 7]),
        swap=torch.tensor([True, False, True, False]),
    )
    condition = desc_eval.FrozenCondition("full", split, split)

    summary = desc_eval.action_summary(condition)

    assert summary["swap_count"] == 4
    assert summary["keep_count"] == 4
    assert summary["used_actions_all"] == 2
    assert summary["highest_occupancy_all"] == 0.5


def test_eval_ledger_commit_can_differ_from_training_commit():
    source = inspect.getsource(desc_eval.run)
    for condition in ("full_checkpoint", "no_action_aux_checkpoint", "parent_only_checkpoint"):
        assert f'expected_commit=config["{condition}"]["training_commit"]' in source
