from __future__ import annotations

from pathlib import Path

import torch

from model.innovations.train_elpt import load_config
from model.innovations.tst import (
    TangentStepGate,
    centroid_alignment_loss,
    centroid_contrastive_loss,
    tangent_transport,
)


ROOT = Path(__file__).resolve().parents[1]


def test_tangent_step_gate_initialization_and_gradient():
    gate = TangentStepGate()
    features = torch.randn(12, 4, generator=torch.Generator().manual_seed(51))
    step = gate(features)
    assert torch.allclose(step, torch.full_like(step, 0.1), atol=1e-7)
    step.square().mean().backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in gate.parameters()
    )


def test_tangent_transport_is_normalized_and_orthogonal_direction():
    generator = torch.Generator().manual_seed(52)
    base = torch.nn.functional.normalize(torch.randn(8, 768, generator=generator), dim=-1)
    value = torch.nn.functional.normalize(torch.randn(8, 768, generator=generator), dim=-1)
    step = torch.linspace(0.1, 1.2, 8)
    moved = tangent_transport(base, value, step)
    tangent = value - (value * base).sum(dim=-1, keepdim=True) * base
    assert torch.allclose((tangent * base).sum(dim=-1), torch.zeros(8), atol=1e-6)
    assert torch.allclose(moved.norm(dim=-1), torch.ones(8), atol=1e-6)
    assert torch.isfinite(moved).all()


def test_tst_config_contract():
    config, _ = load_config(ROOT / "config/tries/v2_try_015_tst_seed7.yaml")
    assert config["idea_id"] == "IDEA-005"
    assert config["transport_mode"] == "tangent"
    assert config["gate_max_step"] == 1.5
    assert config["fold_checkpoint_dir"].endswith("V2-TRY-006")


def test_tst_multiseed_configs_train_own_folds():
    expected = {"V2-TRY-016": 5, "V2-TRY-017": 6, "V2-TRY-018": 8}
    for attempt_id, seed in expected.items():
        config, _ = load_config(
            ROOT / "config/tries" / f"v2_try_{attempt_id[-3:]}_tst_seed{seed}.yaml"
        )
        assert config["attempt_id"] == attempt_id
        assert config["seed"] == seed
        assert config["fold_checkpoint_dir"] is None


def test_centroid_alignment_and_cata_config():
    generator = torch.Generator().manual_seed(53)
    prototypes = torch.randn(8, 768, generator=generator, requires_grad=True)
    centroids = torch.randn(8, 768, generator=generator)
    loss = centroid_alignment_loss(prototypes, centroids)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prototypes.grad).all()
    config, _ = load_config(ROOT / "config/tries/v2_try_021_cata_seed7.yaml")
    assert config["idea_id"] == "IDEA-007"
    assert config["centroid_alignment_weight"] == 0.1
    assert config["parent_metrics_percent"]["H"] == 76.98454484002713


def test_centroid_contrastive_alignment_and_rescue_config():
    generator = torch.Generator().manual_seed(54)
    prototypes = torch.randn(12, 768, generator=generator, requires_grad=True)
    centroids = torch.randn(12, 768, generator=generator)
    loss = centroid_contrastive_loss(prototypes, centroids)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prototypes.grad).all()
    config, _ = load_config(
        ROOT / "config/tries/v2_try_022_cata_rescue1_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-022"
    assert config["centroid_alignment_mode"] == "contrastive"
