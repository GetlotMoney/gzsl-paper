import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from model.frameworks.v6 import train_svra_gate0 as svra_train
from model.frameworks.v6.svra import ParentRiskArbiter, ParentRiskCeilingArbiter


def _config(**overrides):
    data = {
        "schema_version": svra_train.TRAIN_SCHEMA,
        "experiment_id": "V6-TRY-003-SVRA-GATE0-FULL",
        "condition_id": "SVRA_FULL",
        "text_manifest": "text.json",
        "text_manifest_sha256": "0" * 64,
        "role_tensor": "role.pt",
        "role_tensor_sha256": "1" * 64,
        "name_tensor": "name.pt",
        "name_tensor_sha256": "2" * 64,
        "patch_manifest": "patch.json",
        "patch_manifest_sha256": "3" * 64,
        "cls_tensor": "cls.pt",
        "cls_tensor_sha256": "4" * 64,
        "patch_tensor": "patch.npy",
        "patch_tensor_sha256": "5" * 64,
        "action_bundle_manifest": "bundle.json",
        "action_bundle_manifest_sha256": "6" * 64,
        "dev_train_manifest_sha256": "7" * 64,
        "dev_eval_manifest_sha256": "8" * 64,
        "dev_eval_oracle_manifest_sha256": "9" * 64,
        "att_splits_mat_path": "att_splits.mat",
        "trainval_count": 7057,
        "oracle_receipt": "oracle.json",
        "oracle_receipt_sha256": "a" * 64,
        "action_geometry_sha256": svra_train.ACTION_GEOMETRY_SHA256,
        "output_dir": "/tmp/svra",
        "device": "cuda:0",
        "seed": 7,
        "stage1_batch_size": 8,
        "stage1_updates": 1000,
        "stage1_lr": 0.001,
        "stage1_weight_decay": 0.0001,
        "stage1_challenger_per_batch": 4,
        "stage2_batch_size": 32,
        "stage2_updates": 1000,
        "stage2_lr": 0.001,
        "stage2_weight_decay": 0.0001,
        "stage2_positive_per_batch": 16,
        "stage2_threshold": 0.5,
        "expected_stage1_abstain_rows": 4107,
        "expected_stage1_action_rows": 595,
        "expected_stage2_triggered_rows": 574,
        "expected_stage2_positive_rows": 300,
        "expected_stage2_negative_rows": 274,
        "strict_sha": True,
        "validate_tensor_values": True,
        "require_clean_tree": True,
        "allow_cpu": False,
        "official_test_loaded": False,
        "unseen_images_used_for_gradient": False,
        "pclr_online_inference": False,
    }
    data.update(overrides)
    return svra_train.Gate0TrainConfig(**data)


class Pair:
    def __init__(self, top2):
        self.top2 = top2


class ToySVRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name_embeddings = torch.zeros(3, svra_train.FEATURE_DIM)
        self.name_embeddings[0, 0] = 1.0
        self.name_embeddings[1, 1] = 1.0
        self.name_embeddings[2, 2] = 1.0
        self.policy = nn.Linear(2, svra_train.ACTION_COUNT, bias=False)
        nn.init.zeros_(self.policy.weight)

    def dense_utility_targets(self, full_cls, all_crop_cls, target_class_ids, *, semantic_off=False):
        del semantic_off
        top2 = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], device=full_cls.device)
        groups = torch.tensor([0, 1, 1, 2], device=full_cls.device)
        dense = torch.zeros(4, svra_train.ACTION_COUNT, dtype=torch.bool, device=full_cls.device)
        dense[1, 4] = True
        dense[1, 6] = True
        return dense, groups, Pair(top2)

    def utility_state(self, full_cls, patches):
        del patches
        return self.policy(full_cls[:, :2])

    def policy_state(self, full_cls, patches, *, semantic_off=False, visual_off=False):
        del semantic_off, visual_off
        return self.utility_state(full_cls, patches)


def _toy_eaac_inputs():
    full = torch.zeros(4, svra_train.FEATURE_DIM)
    full[:, 0] = 1.0
    full[:, 1] = 0.5
    crops = torch.zeros(4, svra_train.ACTION_COUNT, svra_train.FEATURE_DIM)
    crops[:, :, 0] = 1.0
    crops[:, :, 1] = 0.1
    crops[1, 4, 0] = -0.5
    crops[1, 4, 1] = 1.0
    crops[1, 6, 0] = -0.1
    crops[1, 6, 1] = 0.6
    labels = torch.tensor([10, 20, 20, 30])
    return full, crops, labels


def test_schema_config_and_protocol_flags_are_strict():
    config = _config()
    svra_train.validate_config(config)
    assert svra_train.TRAIN_SCHEMA == "gzsl-paper.v6-svra-gate0-train.v1"
    assert svra_train.SVRA_PREREGISTERED_STAGE1_TARGET_COUNTS["abstain_rows"] == 4107
    assert svra_train.SVRA_PREREGISTERED_TRIGGER_COUNTS["triggered_rows"] == 574

    with pytest.raises(RuntimeError, match="Stage2 threshold"):
        svra_train.validate_config(dataclasses.replace(config, stage2_threshold=0.4))
    with pytest.raises(RuntimeError, match="official_test_loaded"):
        svra_train.validate_config(dataclasses.replace(config, official_test_loaded=True))


def test_train_asset_config_uses_svra_action_bundle_name():
    config = _config()

    asset_config = svra_train.asset_config_from_train_config(config)

    assert asset_config.action_bundle_manifest.path == "bundle.json"
    assert asset_config.action_bundle_manifest.sha256 == "6" * 64


def test_build_svra_model_slices_200_class_assets_to_active_axis():
    generator = torch.Generator().manual_seed(7)
    role_embeddings = F.normalize(
        torch.randn(200, 8, svra_train.FEATURE_DIM, generator=generator),
        dim=-1,
    )
    name_embeddings = F.normalize(
        torch.randn(200, svra_train.FEATURE_DIM, generator=generator),
        dim=-1,
    )
    class_ids = torch.tensor([3, 9, 42, 199])

    model = svra_train.build_svra_model(
        role_embeddings,
        name_embeddings,
        class_ids,
        device=torch.device("cpu"),
        seed=7,
    )
    state = model.policy_state(
        F.normalize(torch.randn(2, svra_train.FEATURE_DIM, generator=generator), dim=-1),
        F.normalize(
            torch.randn(2, 576, svra_train.FEATURE_DIM, generator=generator),
            dim=-1,
        ),
    )

    assert model.semantic.role_embeddings.shape == (4, 8, svra_train.FEATURE_DIM)
    assert torch.equal(model.class_ids, class_ids)
    assert state.parent_logits.shape == (2, 4)


def test_oracle_receipt_is_verified_before_training_identity(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"ok": true}\n', encoding="utf-8")
    sha = svra_train.sha256_file(receipt)

    verified = svra_train.verify_oracle_receipt(
        _config(oracle_receipt=str(receipt), oracle_receipt_sha256=sha)
    )

    assert verified == {"path": str(receipt), "sha256": sha}
    with pytest.raises(RuntimeError, match="oracle_receipt_sha256 mismatch"):
        svra_train.verify_oracle_receipt(
            _config(oracle_receipt=str(receipt), oracle_receipt_sha256="0" * 64)
        )


def test_load_dev_train_targets_reuses_verified_manifest_sha_without_rehash(
    tmp_path,
    monkeypatch,
):
    labels_path = tmp_path / "labels.pt"
    class_ids_path = tmp_path / "class_ids.pt"
    crop_path = tmp_path / "crop_features.pt"
    torch.save(torch.tensor([0, 1, 2, 3]), labels_path)
    torch.save(torch.tensor([0, 1, 2]), class_ids_path)
    torch.save(torch.zeros(4, svra_train.ACTION_COUNT, svra_train.FEATURE_DIM), crop_path)
    paths = {
        "labels.pt": labels_path,
        "class_ids.pt": class_ids_path,
        "crop_features.pt": crop_path,
    }

    def resolve_subset_output(assets, subset_name, filename, *, verify_sha):
        del assets
        assert subset_name == "dev_train"
        assert verify_sha is False
        if filename not in paths:
            raise RuntimeError(filename)
        return paths[filename]

    monkeypatch.setattr(
        svra_train,
        "_load_data_api",
        lambda: {"resolve_subset_output": resolve_subset_output},
    )
    monkeypatch.setattr(
        svra_train,
        "sha256_file",
        lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected rehash: {path}")),
    )
    assets = SimpleNamespace(
        dev_train_manifest={
            "outputs_sha256": {
                "labels.pt": "labels_manifest_sha",
                "class_ids.pt": "class_ids_manifest_sha",
                "crop_features.pt": "crop_manifest_sha",
            }
        }
    )

    loaded = svra_train.load_dev_train_targets(assets)

    assert loaded.labels_sha256 == "labels_manifest_sha"
    assert loaded.class_ids_sha256 == "class_ids_manifest_sha"
    assert loaded.crop_features_sha256 == "crop_manifest_sha"


def test_stage1_eaac_targets_pick_strongest_corrective_action_else_abstain():
    model = ToySVRAModel()
    full, crops, labels = _toy_eaac_inputs()
    targets26, groups, margins, _, dense = svra_train.eaac_action_targets(
        model, full, crops, labels
    )
    assert groups.tolist() == [0, 1, 1, 2]
    assert targets26.tolist() == [0, 5, 0, 0]
    assert dense[1, 4]
    assert margins.shape == (4, svra_train.ACTION_COUNT)


def test_stage1_sampler_is_fixed_4_to_4_and_deterministic():
    groups = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 1], dtype=torch.long)
    left = svra_train.stage1_sampler_from_groups(groups, seed=7)
    right = svra_train.stage1_sampler_from_groups(groups, seed=7)
    for _ in range(5):
        lrows = left.sample()
        rrows = right.sample()
        assert torch.equal(lrows, rrows)
        assert int(groups.index_select(0, lrows).eq(1).sum().item()) == 4
        assert int(groups.index_select(0, lrows).ne(1).sum().item()) == 4


def test_stage2_trigger_features_and_shared_batch_trace_contract():
    policy_logits = torch.zeros(6, svra_train.EAAC_CLASS_COUNT)
    policy_logits[0, 1] = 4
    policy_logits[1, 0] = 5
    policy_logits[2, 3] = 4
    policy_logits[3, 0] = 5
    policy_logits[4, 5] = 4
    policy_logits[5, 0] = 5
    probabilities = torch.softmax(policy_logits, dim=1)
    selected26 = torch.argmax(policy_logits, dim=1)
    confidence = probabilities[torch.arange(6), selected26]
    selected_action = (selected26 - 1).clamp(min=0, max=svra_train.ACTION_COUNT - 1)
    parent_stats = torch.tensor(
        [
            [0.4, 1.1, 0.2, 0.3],
            [0.5, 1.2, 0.3, 0.4],
            [0.6, 1.3, 0.4, 0.5],
            [0.7, 1.4, 0.5, 0.6],
            [0.8, 1.5, 0.6, 0.7],
            [0.9, 1.6, 0.7, 0.8],
        ]
    )
    target_plan = svra_train.EAACTargetPlan(
        targets26=torch.tensor([1, 0, 0, 0, 5, 0]),
        groups=torch.tensor([1, 0, 1, 0, 1, 2]),
        margins=torch.randn(6, svra_train.ACTION_COUNT, generator=torch.Generator().manual_seed(7)),
        dense_targets=torch.zeros(6, svra_train.ACTION_COUNT, dtype=torch.bool),
        stats={},
    )

    features4, features13 = svra_train.build_trigger_features(
        parent_stats=parent_stats,
        selected_policy_confidence=confidence,
        selected_action=selected_action,
    )

    assert features4.shape == (6, 4)
    assert features13.shape == (6, 13)
    assert torch.equal(features4, parent_stats)
    assert torch.equal(features13[:, :4], parent_stats)
    assert torch.equal(features13[:, 4], confidence)

    labels = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)
    trace4, receipt4 = svra_train.make_balanced_batch_trace(
        labels,
        updates=4,
        batch_size=4,
        positive_per_batch=2,
        seed=7,
        name="shared",
    )
    trace13, receipt13 = svra_train.make_balanced_batch_trace(
        labels,
        updates=4,
        batch_size=4,
        positive_per_batch=2,
        seed=7,
        name="shared",
    )
    assert receipt4["batch_trace_sha256"] == receipt13["batch_trace_sha256"]
    for left, right in zip(trace4, trace13, strict=True):
        assert torch.equal(left, right)


def test_train_arbiter_reports_nonzero_gradient():
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    features = torch.tensor(
        [[2.0, 0.8, 0.0, 0.1], [0.1, 0.2, 0.0, 0.7], [1.5, 0.7, 0.0, 0.2], [0.2, 0.3, 0.0, 0.6]]
    )
    trace = [torch.tensor([0, 1, 2, 3])]
    arbiter = svra_train.LinearTriggerArbiter(4, seed=7)
    assert torch.equal(arbiter.output.weight, torch.zeros_like(arbiter.output.weight))
    assert torch.equal(arbiter.output.bias, torch.zeros_like(arbiter.output.bias))
    _, grad, summary = svra_train.train_arbiter_with_trace(
        arbiter,
        features,
        labels,
        trace,
        lr=0.001,
        weight_decay=0.0001,
        device=torch.device("cpu"),
    )
    assert summary["num_updates"] == 1
    step1 = svra_train._gradient_report_to_json(grad["step1"])
    assert step1["output.weight"]["nonzero"]
    assert not step1["hidden.weight"]["nonzero"]


def test_train_arbiter_step2_reaches_hidden_gradient_after_zero_head_update():
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    features = torch.tensor(
        [[2.0, 0.8, 0.0, 0.1], [0.1, 0.2, 0.0, 0.7], [1.5, 0.7, 0.0, 0.2], [0.2, 0.3, 0.0, 0.6]]
    )
    trace = [torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3])]
    arbiter = svra_train.LinearTriggerArbiter(4, seed=7)

    _, grad, summary = svra_train.train_arbiter_with_trace(
        arbiter,
        features,
        labels,
        trace,
        lr=0.001,
        weight_decay=0.0001,
        device=torch.device("cpu"),
    )

    assert summary["num_updates"] == 2
    step2 = svra_train._gradient_report_to_json(grad["step2"])
    assert step2["hidden.weight"]["finite"]
    assert step2["hidden.weight"]["nonzero"]
    assert step2["output.weight"]["nonzero"]


class _Stage1View:
    size = 4

    def __init__(self, cls: torch.Tensor, patches: torch.Tensor) -> None:
        self.cls = cls
        self.patches = patches

    def batch(self, rows, *, include_patches: bool, as_torch: bool, device: torch.device):
        assert include_patches is True
        assert as_torch is True
        index = torch.as_tensor(rows, dtype=torch.long)
        return {
            "cls": self.cls.index_select(0, index).to(device),
            "patches": self.patches.index_select(0, index).to(device),
        }


def test_stage1_zero_head_step2_reaches_semantic_and_visual_upstream():
    generator = torch.Generator().manual_seed(11)
    class_ids = torch.arange(4)
    roles = F.normalize(
        torch.randn(4, 8, svra_train.FEATURE_DIM, generator=generator),
        dim=-1,
    )
    names = F.normalize(
        torch.randn(4, svra_train.FEATURE_DIM, generator=generator),
        dim=-1,
    )
    model = svra_train.build_svra_model(
        roles,
        names,
        class_ids,
        device=torch.device("cpu"),
        seed=7,
    )
    svra_train.configure_stage1_trainable(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=0.001,
        weight_decay=0.0001,
        foreach=False,
        fused=False,
    )
    view = _Stage1View(
        F.normalize(torch.randn(4, svra_train.FEATURE_DIM, generator=generator), dim=-1),
        F.normalize(
            torch.randn(4, 576, svra_train.FEATURE_DIM, generator=generator),
            dim=-1,
        ),
    )
    targets = svra_train.TrainSubsetTargets(
        labels=torch.arange(4),
        class_ids=class_ids,
        crop_features=torch.zeros(4, svra_train.ACTION_COUNT, svra_train.FEATURE_DIM),
        labels_path="labels.pt",
        class_ids_path="class_ids.pt",
        crop_features_path="crop_features.pt",
        labels_sha256="labels",
        class_ids_sha256="classes",
        crop_features_sha256="crops",
    )
    target_plan = svra_train.EAACTargetPlan(
        targets26=torch.tensor([1, 2, 0, 3]),
        groups=torch.tensor([1, 1, 0, 1]),
        margins=torch.zeros(4, svra_train.ACTION_COUNT),
        dense_targets=torch.zeros(4, svra_train.ACTION_COUNT, dtype=torch.bool),
        stats={},
    )
    rows = torch.arange(4)

    _, step1 = svra_train.train_stage1_step(
        model,
        optimizer,
        view,
        targets,
        rows,
        target_plan,
        device=torch.device("cpu"),
    )
    assert any(
        name.startswith("visual.utility_output.") and gate.nonzero
        for name, gate in step1.items()
    )
    _, step2 = svra_train.train_stage1_step(
        model,
        optimizer,
        view,
        targets,
        rows,
        target_plan,
        device=torch.device("cpu"),
    )

    svra_train.assert_stage1_second_step_contract(step2)


def test_stage2_uses_core_arbiter_architecture():
    arbiter4 = ParentRiskArbiter()
    arbiter13 = ParentRiskCeilingArbiter()

    assert set(arbiter4.state_dict()) == set(svra_train.LinearTriggerArbiter(4).state_dict())
    assert set(arbiter13.state_dict()) == set(svra_train.LinearTriggerArbiter(13).state_dict())


def test_stage1_trainable_scope_excludes_interaction_when_present():
    class ModelWithSvi(nn.Module):
        def __init__(self):
            super().__init__()
            self.semantic = nn.Linear(2, 2)
            self.visual = nn.Linear(2, 2)
            self.interaction = nn.Linear(2, 2)

    model = ModelWithSvi()
    selected = svra_train.configure_stage1_trainable(model)

    assert selected
    assert all(param.requires_grad for param in model.semantic.parameters())
    assert all(param.requires_grad for param in model.visual.parameters())
    assert not any(param.requires_grad for param in model.interaction.parameters())
