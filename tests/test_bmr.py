from __future__ import annotations

from pathlib import Path

import torch
from torch.func import functional_call

from model.innovations.train_elpt import (
    _first_order_adapted_parameters,
    load_config,
)
from model.innovations.tst import TangentStepGate


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
