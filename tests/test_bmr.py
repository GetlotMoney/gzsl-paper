from __future__ import annotations

from pathlib import Path

import torch
from torch.func import functional_call

from model.candidates.v2.trainers.train_elpt import (
    _first_order_adapted_parameters,
    _pcgrad_merge,
    _symmetric_pcgrad_merge,
    load_config,
)
from model.frameworks.v4.tg import semantic_pca_folds
from model.frameworks.v4.tst import SummaryResidualGate, TangentStepGate


ROOT = Path(__file__).resolve().parents[1]


def test_first_order_inner_update_keeps_outer_gradient_path():
    gate = TangentStepGate(input_dim=4)
    inner_features = torch.randn(12, 4, generator=torch.Generator().manual_seed(81))
    outer_features = torch.randn(12, 4, generator=torch.Generator().manual_seed(82))
    inner_loss = gate(inner_features).square().mean()
    adapted = _first_order_adapted_parameters(gate, inner_loss, 0.01)
    outer_loss = functional_call(gate, adapted, (outer_features,)).mean()
    outer_loss.backward()
    assert all(parameter.grad is not None for parameter in gate.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in gate.parameters())


def test_bmr_config_and_official_test_boundary():
    config, _ = load_config(ROOT / "config/tries/v2_try_037_bmr_seed7.yaml")
    assert config["idea_id"] == "IDEA-011"
    assert config["gate_training_mode"] == "bilevel_first_order"
    assert config["inner_lr"] == 0.01
    source = (ROOT / "model/candidates/v2/trainers/train_elpt.py").read_text(encoding="utf-8")
    assert source.index("gate = _train_gate") < source.index("# official test只在所有训练完成后加载。")


def test_pcgrad_removes_conflict_and_rescue_config():
    primary = (torch.tensor([-1.0, 0.0]),)
    anchor = (torch.tensor([1.0, 0.0]),)
    merged, conflict = _pcgrad_merge(primary, anchor, anchor_weight=1.0)
    assert conflict is True
    assert float((merged[0] * anchor[0]).sum()) >= 0.0
    config, _ = load_config(
        ROOT / "config/tries/v2_try_038_bmr_rescue1_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-038"
    assert config["meta_gradient_mode"] == "pcgrad_seen_outer"
    assert config["seen_gradient_weight"] == 1.0


def test_bmr_residual_starts_at_tst_and_rescue2_config():
    base = TangentStepGate(input_dim=4)
    residual = SummaryResidualGate(base, max_delta=0.1)
    features = torch.randn(10, 4, generator=torch.Generator().manual_seed(83))
    assert torch.equal(residual(features), base(features))
    residual(features).mean().backward()
    assert all(parameter.grad is None for parameter in residual.base_gate.parameters())
    assert any(parameter.grad is not None for parameter in residual.residual.parameters())
    config, _ = load_config(
        ROOT / "config/tries/v2_try_039_bmr_rescue2_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-039"
    assert config["gate_architecture"] == "bilevel_residual"
    assert config["max_residual_step"] == 0.1


def test_first_order_update_ignores_frozen_parent_gate():
    gate = SummaryResidualGate(TangentStepGate(input_dim=4), max_delta=0.1)
    features = torch.randn(7, 4, generator=torch.Generator().manual_seed(84))
    adapted = _first_order_adapted_parameters(
        gate, gate(features).square().mean(), 0.01
    )
    assert adapted
    assert all(name.startswith("residual.") for name in adapted)


def test_semantic_hard_folds_and_final_rescue_config():
    generator = torch.Generator().manual_seed(85)
    sentences = torch.randn(200, 8, 768, generator=generator)
    seen = torch.tensor([class_id for class_id in range(200) if class_id % 4 != 0])
    folds = semantic_pca_folds(seen, sentences)
    assert len(folds) == 3
    assert all(pseudo_seen.numel() == 100 for pseudo_seen, _ in folds)
    assert all(pseudo_unseen.numel() == 50 for _, pseudo_unseen in folds)
    assert torch.equal(
        torch.cat([pseudo_unseen for _, pseudo_unseen in folds]).sort().values,
        seen.sort().values,
    )
    config, _ = load_config(
        ROOT / "config/tries/v2_try_040_bmr_rescue3_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-040"
    assert config["fold_strategy"] == "semantic_pca_blocks"
    assert config["fold_checkpoint_dir"] is None


def test_symmetric_pcgrad_and_pgo_config():
    first = (torch.tensor([-1.0, 0.0]),)
    second = (torch.tensor([1.0, 0.0]),)
    merged, conflict = _symmetric_pcgrad_merge(first, second)
    assert conflict is True
    assert torch.isfinite(merged[0]).all()
    config, _ = load_config(ROOT / "config/tries/v2_try_048_pgo_seed7.yaml")
    assert config["idea_id"] == "IDEA-015"
    assert config["gate_training_mode"] == "pcgrad_joint_residual"


def test_unit_normalized_pcgrad_and_pgo_rescue_config():
    first = (torch.tensor([-10.0, 0.0]),)
    second = (torch.tensor([1.0, 0.0]),)
    merged, conflict = _symmetric_pcgrad_merge(first, second, normalize=True)
    assert conflict is True
    assert torch.isfinite(merged[0]).all()
    config, _ = load_config(
        ROOT / "config/tries/v2_try_049_pgo_rescue1_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-049"
    assert config["gradient_normalization"] == "unit_global_norm"
