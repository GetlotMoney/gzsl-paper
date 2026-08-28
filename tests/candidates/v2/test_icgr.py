from __future__ import annotations

from pathlib import Path

import torch

from model.candidates.v2.modules.icgr import ICGRClassifier, ICGRRouter
from model.candidates.v2.trainers.train_icgr import load_config, uniform_kl
from tests.frameworks.v2.test_tg_vpr_h1 import make_model


ROOT = Path(__file__).resolve().parents[3]


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


def test_role_cosine_inputs_have_771_dimensions_and_keep_uniform_start():
    parent = make_model().eval()
    classifier = ICGRClassifier(parent, router_input_mode="image_cls_role_cosine")
    images = torch.randn(5, 768, generator=torch.Generator().manual_seed(34))
    inputs = classifier.router_inputs(images)
    assert inputs.shape == (5, 771)
    assert torch.isfinite(inputs).all()
    assert torch.equal(
        classifier.route_weights(images), torch.full((5, 3), 1.0 / 3.0)
    )
    assert torch.allclose(classifier.logits(images), parent.logits(images), atol=2e-5)


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
    source = (ROOT / "model/candidates/v2/trainers/train_icgr.py").read_text(encoding="utf-8")
    assert source.index("for epoch in range") < source.index("# official test严格在路由训练结束后加载。")
    assert 'for name in ("sentence_embeds", "train_features", "train_labels")' in source


def test_rescue2_config_contract():
    config, _ = load_config(
        ROOT / "config/tries/v2_try_011_icgr_rescue2_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-011"
    assert config["router_input_mode"] == "image_cls_role_cosine"


def test_uniform_kl_and_rescue1_config_contract():
    uniform = torch.full((4, 3), 1.0 / 3.0)
    collapsed = torch.tensor([[0.98, 0.01, 0.01]])
    assert abs(float(uniform_kl(uniform))) < 1e-7
    assert float(uniform_kl(collapsed)) > 0.0
    config, _ = load_config(
        ROOT / "config/tries/v2_try_012_icgr_rescue1_seed7.yaml"
    )
    assert config["attempt_id"] == "V2-TRY-012"
    assert config["router_input_mode"] == "image_cls_role_cosine"
    assert config["kl_weight"] == 0.01
