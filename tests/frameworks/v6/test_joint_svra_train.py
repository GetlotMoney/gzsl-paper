import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from model.frameworks.v6 import evaluate_joint_svra_precheck as joint_eval
from model.frameworks.v6.svra import (
    ACTION_COUNT,
    FEATURE_DIM,
    JSVRA_ACTION_POS_WEIGHT,
    JSVRA_RISK_POS_WEIGHT,
    SemanticVisualRiskArbiter,
    joint_action_targets_from_logits,
    joint_svra_loss,
)
from model.frameworks.v6.train_joint_svra_precheck import (
    JointPrecheckConfig,
    JointTrainTable,
    build_model,
    generate_batch_trace,
    load_config,
    train_precheck,
)


def _basis(index: int) -> torch.Tensor:
    out = torch.zeros(FEATURE_DIM)
    out[index] = 1.0
    return out


def _synthetic_table(rows: int = 8, classes: int = 200) -> JointTrainTable:
    names = torch.zeros(classes, FEATURE_DIM)
    for class_id in range(classes):
        names[class_id, class_id % FEATURE_DIM] = 1.0
    names = F.normalize(names, dim=-1)
    roles = names[:, None, :].expand(-1, 8, -1).clone()
    class_ids = torch.arange(classes, dtype=torch.long)

    full_cls = torch.stack([_basis(0) for _ in range(rows)])
    generator = torch.Generator().manual_seed(11)
    patch_tokens = F.normalize(torch.randn(rows, 576, FEATURE_DIM, generator=generator), dim=-1)
    all_crop_cls = torch.stack([_basis(0).repeat(ACTION_COUNT, 1) for _ in range(rows)])
    target_class_ids = torch.tensor([0, 1, 1, 2, 1, 0, 2, 1], dtype=torch.long)[:rows]

    # Top2 is class0/class1 for all rows.  Rows with truth=1 are challenger
    # rows; all except row2 get at least one crop that makes class1 beat class0.
    for row in [1, 4, 7]:
        if row < rows:
            all_crop_cls[row, 3] = _basis(1)
    return JointTrainTable(
        role_embeddings=roles,
        name_embeddings=names,
        class_ids=class_ids,
        full_cls=full_cls,
        patch_tokens=patch_tokens,
        all_crop_cls=all_crop_cls,
        target_class_ids=target_class_ids,
        raw_indices=torch.arange(rows, dtype=torch.long),
        source_splits=tuple("dev_train" if i < rows // 2 else "dev_eval_oracle" for i in range(rows)),
    )


def test_joint_targets_use_full200_axis_and_preserve_conflict_rows():
    table = _synthetic_table()
    model = SemanticVisualRiskArbiter(
        table.role_embeddings,
        table.name_embeddings,
        table.class_ids,
        seed=7,
    )
    pair = model.parent_state(table.full_cls)
    targets = joint_action_targets_from_logits(
        pair,
        table.all_crop_cls,
        model.name_embeddings,
        table.target_class_ids,
        model.class_ids,
    )

    assert pair.parent_logits.shape == (table.rows, 200)
    assert targets.action_targets26.tolist() == [0, 4, 0, 0, 4, 0, 0, 4]
    assert targets.top2_group.tolist() == [0, 1, 1, 2, 1, 0, 2, 1]
    assert targets.risk_targets.tolist() == [0, 1, 1, 0, 1, 0, 0, 1]
    assert targets.joint_targets.tolist() == [0, 1, 0, 0, 1, 0, 0, 1]
    assert int(targets.conflict_mask.sum().item()) == 1


def test_joint_loss_has_fixed_weights_and_step2_gradients_reach_s_v_i():
    table = _synthetic_table()
    model = build_model(table, seed=7, device=torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    rows = torch.arange(table.rows)

    step2_norms = None
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        batch = table.batch(rows, device=torch.device("cpu"))
        output = model.joint_forward(batch["full_cls"], batch["patch_tokens"])
        targets = joint_action_targets_from_logits(
            output.pair,
            batch["all_crop_cls"],
            model.name_embeddings,
            batch["target_class_ids"],
            model.class_ids,
        )
        loss = joint_svra_loss(output, targets)
        assert loss.soft_hard_trigger_equal
        assert loss.weights["action_positive"] == JSVRA_ACTION_POS_WEIGHT
        assert loss.weights["risk_positive"] == JSVRA_RISK_POS_WEIGHT
        assert loss.weights["joint_positive"] == JSVRA_ACTION_POS_WEIGHT
        loss.total.backward()
        if step == 2:
            step2_norms = {
                "S": sum(
                    float(p.grad.detach().norm().item())
                    for p in model.semantic.parameters()
                    if p.grad is not None
                ),
                "V": sum(
                    float(p.grad.detach().norm().item())
                    for p in model.visual.parameters()
                    if p.grad is not None
                ),
                "I": sum(
                    float(p.grad.detach().norm().item())
                    for p in model.interaction.parameters()
                    if p.grad is not None
                ),
            }
        optimizer.step()

    assert step2_norms is not None
    assert step2_norms["S"] > 0
    assert step2_norms["V"] > 0
    assert step2_norms["I"] > 0


def test_fixed_batch_trace_is_independent_randperm_and_reproducible():
    trace_a = generate_batch_trace(rows=8, updates=4, batch_size=3, seed=7)
    trace_b = generate_batch_trace(rows=8, updates=4, batch_size=3, seed=7)

    assert [x.tolist() for x in trace_a] == [x.tolist() for x in trace_b]
    assert all(len(set(x.tolist())) == 3 for x in trace_a)
    assert len({tuple(x.tolist()) for x in trace_a}) > 1


def test_precheck_trains_three_real_conditions_and_saves_receipts(tmp_path: Path):
    table = _synthetic_table()
    config = JointPrecheckConfig(
        output_dir=str(tmp_path / "run"),
        device="cpu",
        batch_size=4,
        updates=3,
        sequential_policy_updates=1,
        lr=0.01,
        expected_target_census={
            "rows": 8,
            "abstain": 5,
            "action": 3,
            "leader": 2,
            "challenger": 4,
            "outside": 2,
            "conflict": 1,
        },
    )

    receipt = train_precheck(table, config, code_commit="abc123", config_sha256="cfg456")
    receipt_path = Path(receipt["receipt_path"])
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert saved["official_test_loaded"] is False
    assert saved["unseen_images_used_for_gradient"] is False
    assert saved["full_axis_classes"] == 200
    assert saved["code_commit"] == "abc123"
    assert saved["config_sha256"] == "cfg456"
    assert Path(saved["target_census_path"]).is_file()
    assert saved["action_target_histogram26"][0] == 5
    assert saved["action_target_histogram26"][4] == 3
    assert set(saved["conditions"]) == {"full_joint", "no_joint", "sequential"}
    assert saved["conditions"]["full_joint"]["objective"] == "L_action+L_risk+L_joint"
    assert saved["conditions"]["no_joint"]["objective"] == "L_action+L_risk"
    assert saved["conditions"]["sequential"]["policy_updates"] == 1
    assert set(saved["checkpoint_specs"]) == {"full_joint", "no_joint", "sequential"}
    for condition in saved["conditions"].values():
        assert Path(condition["checkpoint_path"]).is_file()
        assert condition["condition_id"].startswith("JOINT_SVRA_")
        assert condition["digests"]["soft_hard_trigger_equal"] is True
        assert "batch_sha256" in condition["digests"]
        assert "logit_sha256" in condition["digests"]
        assert "action_sha256" in condition["digests"]
        assert "trigger_sha256" in condition["digests"]
        assert "step1" in condition["gradients"]
        assert "step2" in condition["gradients"]
        assert "step3" in condition["gradients"]


def test_precheck_rejects_existing_output_dir(tmp_path: Path):
    table = _synthetic_table()
    config = JointPrecheckConfig(
        output_dir=str(tmp_path),
        device="cpu",
        batch_size=4,
        updates=3,
        sequential_policy_updates=1,
        lr=0.01,
        expected_target_census=None,
    )

    with pytest.raises(RuntimeError, match="output_dir already exists"):
        train_precheck(table, config)


def test_train_payload_loads_in_eval_and_freezes_real_svra(tmp_path: Path):
    table = _synthetic_table()
    config = JointPrecheckConfig(
        output_dir=str(tmp_path / "run"),
        device="cpu",
        batch_size=4,
        updates=3,
        sequential_policy_updates=1,
        lr=0.01,
        expected_target_census={
            "rows": 8,
            "abstain": 5,
            "action": 3,
            "leader": 2,
            "challenger": 4,
            "outside": 2,
            "conflict": 1,
        },
    )
    receipt = train_precheck(table, config, code_commit="abc123", config_sha256="cfg456")
    spec = receipt["checkpoint_specs"]["full_joint"]

    checkpoint = joint_eval.load_checkpoint(
        spec,
        expected_commit="abc123",
        expected_condition=joint_eval.FULL_CONDITION,
    )
    model = joint_eval.instantiate_model(
        table.role_embeddings,
        table.name_embeddings,
        table.class_ids,
        checkpoint,
        torch.device("cpu"),
    )
    frozen = joint_eval.freeze_split(
        model,
        table.full_cls[:4],
        table.patch_tokens[:4].half(),
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert frozen.logits.shape == (4, 200)
    assert frozen.parent_logits.shape == (4, 200)
    assert frozen.actions.shape == (4,)
    assert frozen.trigger.shape == (4,)
    assert frozen.swap.shape == (4,)
    assert frozen.soft_hard_trigger_equal is True


def test_formal_config_binds_idea200_train_contract():
    train_cfg, asset_cfg = load_config("config/tries/v6_try_004_joint_svra_precheck.json")

    assert train_cfg.schema_version == "gzsl-paper.v6-joint-svra-precheck-train-config.v1"
    assert train_cfg.seed == 7
    assert train_cfg.batch_size == 50
    assert train_cfg.updates == 1000
    assert train_cfg.sequential_policy_updates == 500
    assert train_cfg.strict_fixed_contract is True
    assert train_cfg.require_clean_tree is True
    assert str(train_cfg.device).startswith("cuda")
    assert train_cfg.expected_target_census == {
        "rows": 7057,
        "abstain": 6065,
        "action": 992,
        "leader": 4485,
        "challenger": 1022,
        "outside": 1550,
        "conflict": 30,
    }
    assert asset_cfg.name_tensor.shape == (200, 768)
    assert asset_cfg.role_tensor.shape == (200, 8, 768)
