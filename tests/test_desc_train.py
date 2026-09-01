import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from model.frameworks.v6.svra import (
    ACTION_COUNT,
    DIRECT_SWAP_INPUT_DIM,
    FEATURE_DIM,
    HIDDEN_DIM,
    DirectSwapInteraction,
    SemanticVisualRiskArbiter,
    direct_svra_loss,
    joint_action_targets_from_logits,
)
from model.frameworks.v6.train_desc_precheck import (
    CHECKPOINT_SCHEMA,
    DESCPrecheckConfig,
    FULL_CONDITION,
    JointTrainTable,
    NO_ACTION_AUX_CONDITION,
    PARENT_ONLY_CONDITION,
    load_config,
    train_desc_precheck,
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
    full_cls = torch.stack([_basis(0) for _ in range(rows)])
    generator = torch.Generator().manual_seed(13)
    patch_tokens = F.normalize(torch.randn(rows, 576, FEATURE_DIM, generator=generator), dim=-1)
    all_crop_cls = torch.stack([_basis(0).repeat(ACTION_COUNT, 1) for _ in range(rows)])
    labels = torch.tensor([0, 1, 1, 2, 1, 0, 2, 1], dtype=torch.long)[:rows]
    for row in [1, 4, 7]:
        if row < rows:
            all_crop_cls[row, 3] = _basis(1)
    return JointTrainTable(
        role_embeddings=roles,
        name_embeddings=names,
        class_ids=torch.arange(classes, dtype=torch.long),
        full_cls=full_cls,
        patch_tokens=patch_tokens,
        all_crop_cls=all_crop_cls,
        target_class_ids=labels,
        raw_indices=torch.arange(rows, dtype=torch.long),
        source_splits=tuple("dev_train" if i < rows // 2 else "dev_eval_oracle" for i in range(rows)),
    )


def test_desc_core_has_action_hidden_and_direct_69d_zero_head():
    table = _synthetic_table()
    model = SemanticVisualRiskArbiter(
        table.role_embeddings,
        table.name_embeddings,
        table.class_ids,
        seed=7,
    )
    batch = table.batch(torch.arange(4), device=torch.device("cpu"))
    output = model.direct_forward(batch["full_cls"], batch["patch_tokens"])

    assert isinstance(model.direct_interaction, DirectSwapInteraction)
    assert model.direct_interaction.hidden.in_features == DIRECT_SWAP_INPUT_DIM
    assert model.direct_interaction.hidden.out_features == HIDDEN_DIM
    assert model.direct_interaction.output.out_features == 1
    assert torch.equal(
        model.direct_interaction.output.weight,
        torch.zeros_like(model.direct_interaction.output.weight),
    )
    assert output.action_state is not None
    assert output.action_state.action_hidden.shape == (4, ACTION_COUNT, HIDDEN_DIM)
    assert output.evidence_pool.shape == (4, HIDDEN_DIM)
    assert output.direct_input.shape == (4, DIRECT_SWAP_INPUT_DIM)
    assert output.swap_logits.shape == (4,)
    assert not bool(output.swapped.any())


def test_desc_loss_is_unweighted_swap_plus_optional_action_and_reaches_svi_step2():
    table = _synthetic_table()
    model = SemanticVisualRiskArbiter(
        table.role_embeddings,
        table.name_embeddings,
        table.class_ids,
        seed=7,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    rows = torch.arange(table.rows)

    norms = None
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        batch = table.batch(rows, device=torch.device("cpu"))
        output = model.direct_forward(batch["full_cls"], batch["patch_tokens"])
        targets = joint_action_targets_from_logits(
            output.pair,
            batch["all_crop_cls"],
            model.name_embeddings,
            batch["target_class_ids"],
            model.class_ids,
        )
        loss = direct_svra_loss(output, targets, include_action=True)
        assert loss.include_action is True
        loss.total.backward()
        if step == 2:
            norms = {
                "S": sum(float(p.grad.norm()) for p in model.semantic.parameters() if p.grad is not None),
                "V": sum(float(p.grad.norm()) for p in model.visual.parameters() if p.grad is not None),
                "I": sum(float(p.grad.norm()) for p in model.direct_interaction.parameters() if p.grad is not None),
            }
        optimizer.step()

    assert norms is not None
    assert norms["S"] > 0
    assert norms["V"] > 0
    assert norms["I"] > 0


def test_desc_parent_only_uses_parent4_plus_65_zeros():
    table = _synthetic_table()
    model = SemanticVisualRiskArbiter(
        table.role_embeddings,
        table.name_embeddings,
        table.class_ids,
        seed=7,
    )
    batch = table.batch(torch.arange(3), device=torch.device("cpu"))
    output = model.direct_forward(batch["full_cls"], None, parent_only=True)

    assert output.action_state is None
    assert torch.equal(output.direct_input[:, 4:], torch.zeros_like(output.direct_input[:, 4:]))
    assert output.direct_input.shape == (3, DIRECT_SWAP_INPUT_DIM)


def test_desc_precheck_trains_three_conditions_and_saves_checkpoint_contract(tmp_path: Path):
    table = _synthetic_table()
    config = DESCPrecheckConfig(
        output_dir=str(tmp_path / "desc"),
        device="cpu",
        batch_size=4,
        updates=3,
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

    receipt = train_desc_precheck(table, config, code_commit="commit-desc", config_sha256="cfg-desc")
    saved = json.loads(Path(receipt["receipt_path"]).read_text(encoding="utf-8"))

    assert saved["official_test_loaded"] is False
    assert saved["initialization_sha256"]
    assert saved["batch_trace_sha256"]
    assert saved["target_census_sha256"]
    assert saved["action_target_histogram26"][0] == 5
    assert saved["action_target_histogram26"][4] == 3
    assert set(saved["conditions"]) == {"full", "no_action_aux", "parent_only"}
    assert saved["conditions"]["full"]["objective"] == "L_swap+L_action"
    assert saved["conditions"]["no_action_aux"]["objective"] == "L_swap"
    assert saved["conditions"]["parent_only"]["parent_only"] is True
    assert saved["conditions"]["full"]["condition_id"] == FULL_CONDITION
    assert saved["conditions"]["no_action_aux"]["condition_id"] == NO_ACTION_AUX_CONDITION
    assert saved["conditions"]["parent_only"]["condition_id"] == PARENT_ONLY_CONDITION

    for name, spec in saved["checkpoint_specs"].items():
        checkpoint = torch.load(spec["path"], map_location="cpu", weights_only=True)
        assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA
        assert checkpoint["condition_id"] == saved["conditions"][name]["condition_id"]
        assert checkpoint["code_commit"] == "commit-desc"
        assert checkpoint["config_sha256"] == "cfg-desc"
        assert Path(spec["path"]).name == f"{name}_final.pt"
        assert saved["conditions"][name]["digests"]["swap_logit_sha256"]
        assert saved["conditions"][name]["digests"]["action_logits_sha256"]
        assert saved["conditions"][name]["digests"]["evidence_pool_sha256"]


def test_desc_precheck_rejects_existing_output_dir(tmp_path: Path):
    with pytest.raises(RuntimeError, match="output_dir already exists"):
        train_desc_precheck(
            _synthetic_table(),
            DESCPrecheckConfig(
                output_dir=str(tmp_path),
                device="cpu",
                batch_size=4,
                updates=3,
                expected_target_census=None,
            ),
        )


def test_desc_formal_config_binds_fixed_precheck_contract():
    train_cfg, asset_cfg = load_config("config/tries/v6_try_005_desc_precheck.json")

    assert train_cfg.schema_version == "gzsl-paper.v6-desc-precheck-train-config.v1"
    assert train_cfg.seed == 7
    assert train_cfg.batch_size == 50
    assert train_cfg.updates == 1000
    assert train_cfg.lr == 0.001
    assert train_cfg.weight_decay == 0.0
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
