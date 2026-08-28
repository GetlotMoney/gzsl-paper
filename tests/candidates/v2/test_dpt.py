from __future__ import annotations

from pathlib import Path

import torch

from model.candidates.v2.modules.dpt import (
    AdaptiveDistributionalPrototypeClassifier,
    DistributionalPrototypeClassifier,
    text_resultant_lengths,
    text_uncertainty_features,
)
from model.candidates.v2.trainers.train_dpt import load_config


ROOT = Path(__file__).resolve().parents[3]


def make_model():
    generator = torch.Generator().manual_seed(91)
    sentences = torch.randn(200, 8, 768, generator=generator)
    prototypes = torch.randn(200, 768, generator=generator)
    seen = torch.tensor([class_id for class_id in range(200) if class_id % 4 != 0])
    return DistributionalPrototypeClassifier(
        prototypes, text_resultant_lengths(sentences), seen, torch.tensor(10.0)
    )


def test_dpt_off_reproduces_parent_and_confidence_is_centered():
    model = make_model()
    assert torch.equal(model.prototypes(enabled=False), model.parent_prototypes)
    confidence = model.class_confidence()
    seen_log_mean = confidence.index_select(0, model.seenclasses).log().mean()
    assert abs(float(seen_log_mean.detach())) < 1e-6
    assert torch.isfinite(confidence).all()


def test_dpt_gamma_is_bounded_and_trainable():
    model = make_model()
    images = torch.randn(6, 768, generator=torch.Generator().manual_seed(92))
    torch.nn.functional.cross_entropy(
        model.logits(images, model.seenclasses), torch.arange(6)
    ).backward()
    assert model.raw_gamma.grad is not None and torch.isfinite(model.raw_gamma.grad)
    model.raw_gamma.data.fill_(100.0)
    assert float(model.gamma().detach()) <= 2.000001


def test_dpt_config_and_training_boundary():
    config, _ = load_config(ROOT / "config/tries/v2_try_041_dpt_seed7.yaml")
    assert config["idea_id"] == "IDEA-012"
    assert config["max_gamma"] == 2.0
    source = (ROOT / "model/candidates/v2/trainers/train_dpt.py").read_text(encoding="utf-8")
    assert source.index("for epoch in range") < source.index("# official test严格在DPT训练结束后加载。")


def test_adaptive_dpt_starts_off_and_rescue_config():
    generator = torch.Generator().manual_seed(93)
    sentences = torch.randn(200, 8, 768, generator=generator)
    parent = torch.randn(200, 768, generator=generator)
    model = AdaptiveDistributionalPrototypeClassifier(
        parent, text_uncertainty_features(sentences), torch.tensor(10.0)
    )
    assert torch.equal(model.class_confidence(), torch.ones(200))
    images = torch.randn(6, 768, generator=generator)
    torch.nn.functional.cross_entropy(model.logits(images), torch.arange(6)).backward()
    assert any(parameter.grad is not None for parameter in model.gate.parameters())
    config, _ = load_config(
        ROOT / "config/tries/v2_try_042_dpt_rescue1_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-042"
    assert config["confidence_mode"] == "adaptive_gate"


def test_centered_adaptive_dpt_removes_common_mode():
    generator = torch.Generator().manual_seed(94)
    sentences = torch.randn(200, 8, 768, generator=generator)
    parent = torch.randn(200, 768, generator=generator)
    seen = torch.tensor([class_id for class_id in range(200) if class_id % 4 != 0])
    model = AdaptiveDistributionalPrototypeClassifier(
        parent,
        text_uncertainty_features(sentences),
        torch.tensor(10.0),
        seenclasses=seen,
        center_seen_log_scale=True,
    )
    model.gate[-1].bias.data.fill_(3.0)
    confidence = model.class_confidence()
    assert abs(float(confidence.index_select(0, seen).log().mean().detach())) < 1e-6
    config, _ = load_config(
        ROOT / "config/tries/v2_try_043_dpt_rescue2_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-043"
    assert config["confidence_mode"] == "centered_adaptive_gate"
