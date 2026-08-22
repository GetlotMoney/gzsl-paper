from __future__ import annotations

from pathlib import Path

import torch
from torch.func import functional_call

from model.innovations.train_elpt import (
    _first_order_adapted_parameters,
    _pcgrad_merge,
    load_config,
)
from model.innovations.tst import SummaryResidualGate, TangentStepGate


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
    source = (ROOT / "model/innovations/train_elpt.py").read_text(encoding="utf-8")
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
