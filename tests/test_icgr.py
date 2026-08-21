from __future__ import annotations

from pathlib import Path

import torch

from model.innovations.icgr import ICGRClassifier, ICGRRouter
from model.innovations.train_icgr import load_config
from tests.test_tg_vpr_h1 import make_model


ROOT = Path(__file__).resolve().parents[1]


def test_router_initial_weights_are_strict_uniform():
    router = ICGRRouter()
    inputs = torch.randn(5, 768, generator=torch.Generator().manual_seed(31))
    expected = torch.full((5, 3), 1.0 / 3.0)
    assert torch.equal(router(inputs), expected)


def test_initial_and_disabled_logits_reproduce_parent():
    parent = make_model().eval()
    classifier = ICGRClassifier(parent).eval()
    images = torch.randn(7, 768, generator=torch.Generator().manual_seed(32))
    expected = parent.logits(images)
    assert torch.allclose(classifier.logits(images), expected, atol=2e-5)
    classifier.router.network[-1].bias.data.copy_(torch.tensor((1.0, -1.0, 0.5)))
    assert torch.allclose(
        classifier.logits(images, enabled=False), expected, atol=2e-5
    )


def test_only_router_receives_gradients():
    parent = make_model().eval()
    classifier = ICGRClassifier(parent)
    images = torch.randn(6, 768, generator=torch.Generator().manual_seed(33))
    loss = torch.nn.functional.cross_entropy(
        classifier.logits(images, parent.adapted_classes), torch.arange(6)
    )
    loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in classifier.router.parameters()
    )
    assert all(parameter.grad is None for parameter in classifier.parent.parameters())


def test_config_and_training_boundary_contract():
    config, _ = load_config(ROOT / "config/tries/v2_try_010_icgr_seed7.yaml")
    assert config["idea_id"] == "IDEA-003"
    assert config["epochs"] == 10
    assert config["hidden_dim"] == 64
    source = (ROOT / "model/innovations/train_icgr.py").read_text(encoding="utf-8")
    assert source.index("for epoch in range") < source.index("# official test严格在路由训练结束后加载。")
    assert 'for name in ("sentence_embeds", "train_features", "train_labels")' in source
