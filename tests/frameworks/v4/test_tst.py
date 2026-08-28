from __future__ import annotations

from pathlib import Path

import torch

from model.candidates.v2.trainers.train_elpt import _pseudo_unseen_risk, load_config
from model.frameworks.v4.tst import (
    TangentStepGate,
    NeighborhoodResidualGate,
    bidirectional_centroid_contrastive_loss,
    centroid_alignment_loss,
    centroid_contrastive_loss,
    tangent_transport,
)


ROOT = Path(__file__).resolve().parents[3]


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


def test_tangent_step_gate_accepts_neighborhood_vector():
    gate = TangentStepGate(input_dim=8)
    features = torch.randn(9, 8, generator=torch.Generator().manual_seed(56))
    assert torch.allclose(
        gate(features), torch.full((9,), 0.1), atol=1e-7
    )


def test_tangent_step_gate_accepts_dispersion_summary():
    gate = TangentStepGate(input_dim=5)
    features = torch.randn(9, 5, generator=torch.Generator().manual_seed(58))
    assert torch.allclose(gate(features), torch.full((9,), 0.1), atol=1e-7)


def test_neighborhood_residual_starts_at_frozen_tst_step():
    base = TangentStepGate(input_dim=4)
    gate = NeighborhoodResidualGate(base, max_delta=0.1)
    features = torch.randn(11, 8, generator=torch.Generator().manual_seed(57))
    summary = torch.stack(
        (features[:, 0], features[:, 1], features[:, 2], features[:, 3]), dim=1
    )
    assert torch.equal(gate(features), base(summary))
    gate(features).square().mean().backward()
    assert all(parameter.grad is None for parameter in gate.base_gate.parameters())
    assert any(parameter.grad is not None for parameter in gate.residual.parameters())


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


def test_tst_seed9_config_trains_own_folds():
    config, _ = load_config(ROOT / "config/tries/v2_try_051_tst_seed9.yaml")
    assert config["attempt_id"] == "V2-TRY-051"
    assert config["seed"] == 9
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


def test_bidirectional_centroid_contrastive_and_rescue2_config():
    generator = torch.Generator().manual_seed(55)
    prototypes = torch.randn(10, 768, generator=generator, requires_grad=True)
    centroids = torch.randn(10, 768, generator=generator)
    loss = bidirectional_centroid_contrastive_loss(prototypes, centroids)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prototypes.grad).all()
    config, _ = load_config(
        ROOT / "config/tries/v2_try_023_cata_rescue2_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-023"
    assert config["centroid_alignment_mode"] == "bidirectional_contrastive"


def test_cata_final_rescue_uses_three_initializations():
    config, _ = load_config(
        ROOT / "config/tries/v2_try_024_cata_rescue3_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-024"
    assert config["centroid_alignment_mode"] == "contrastive"
    assert config["gate_initialization_ensemble"] == 3


def test_purl_config_reweights_pseudo_unseen_risk():
    config, _ = load_config(ROOT / "config/tries/v2_try_026_purl_seed7.yaml")
    assert config["idea_id"] == "IDEA-009"
    assert config["pseudo_unseen_ce_weight"] == 1.0
    assert config["centroid_alignment_weight"] == 0.0


def test_purl_focal_risk_and_rescue_config():
    logits = torch.tensor([[3.0, 0.0], [0.2, 0.0]], requires_grad=True)
    targets = torch.tensor([0, 0])
    focal = _pseudo_unseen_risk(logits, targets, "focal_gamma2")
    ce = _pseudo_unseen_risk(logits, targets, "cross_entropy")
    assert float(focal.detach()) < float(ce.detach())
    focal.backward()
    assert torch.isfinite(logits.grad).all()
    config, _ = load_config(
        ROOT / "config/tries/v2_try_027_purl_rescue1_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-027"
    assert config["pseudo_unseen_loss_mode"] == "focal_gamma2"


def test_ntr_config_uses_full_top5_neighborhood():
    config, _ = load_config(ROOT / "config/tries/v2_try_028_ntr_seed7.yaml")
    assert config["idea_id"] == "IDEA-010"
    assert config["gate_feature_mode"] == "top5_vector"
    assert config["centroid_alignment_weight"] == 0.0
    assert config["pseudo_unseen_ce_weight"] == 0.0


def test_ntr_seed9_config_reuses_seed9_folds():
    config, _ = load_config(ROOT / "config/tries/v2_try_052_ntr_seed9.yaml")
    assert config["attempt_id"] == "V2-TRY-052"
    assert config["seed"] == 9
    assert config["gate_feature_mode"] == "top5_vector"
    assert config["fold_checkpoint_dir"].endswith("V2-TRY-051")


def test_ntr_multiseed_configs_train_own_folds():
    expected = {"V2-TRY-029": 5, "V2-TRY-030": 6, "V2-TRY-031": 8}
    for attempt, seed in expected.items():
        config, _ = load_config(
            ROOT / "config/tries" / f"v2_try_{attempt[-3:]}_ntr_seed{seed}.yaml"
        )
        assert config["attempt_id"] == attempt
        assert config["seed"] == seed
        assert config["fold_checkpoint_dir"] is None
        assert config["gate_feature_mode"] == "top5_vector"


def test_ntr_residual_rescue_config():
    config, _ = load_config(
        ROOT / "config/tries/v2_try_032_ntr_rescue1_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-032"
    assert config["gate_architecture"] == "neighborhood_residual"
    assert config["max_residual_step"] == 0.1


def test_ntr_dispersion_rescue_config():
    config, _ = load_config(
        ROOT / "config/tries/v2_try_036_ntr_rescue2_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-036"
    assert config["gate_feature_mode"] == "summary_std"
    assert config["gate_architecture"] == "direct"


def test_residual_ntr_multiseed_configs_bind_own_tst_gate():
    expected = {"V2-TRY-033": 5, "V2-TRY-034": 6, "V2-TRY-035": 8}
    for attempt, seed in expected.items():
        config, _ = load_config(
            ROOT / "config/tries" / f"v2_try_{attempt[-3:]}_ntr_residual_seed{seed}.yaml"
        )
        assert config["attempt_id"] == attempt
        assert config["seed"] == seed
        assert config["gate_architecture"] == "neighborhood_residual"
        assert config["parent_gate_model"]
        assert config["fold_checkpoint_dir"]
