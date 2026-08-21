from __future__ import annotations

from pathlib import Path

import torch

from model.innovations.acgr import AllClassCenteredGroupRouter
from model.innovations.train_icgr import load_config
from tests.test_tg_vpr_h1 import make_model


ROOT = Path(__file__).resolve().parents[1]


def test_uniform_and_disabled_acgr_reproduce_parent():
    parent = make_model().eval()
    model = AllClassCenteredGroupRouter(parent).eval()
    images = torch.randn(7, 768, generator=torch.Generator().manual_seed(41))
    expected = parent.logits(images)
    assert torch.equal(
        model.route_weights(images), torch.full((7, 3), 1.0 / 3.0)
    )
    assert torch.allclose(model.logits(images), expected, atol=2e-5)
    assert torch.allclose(model.logits(images, enabled=False), expected, atol=2e-5)


def test_centered_roles_cover_true_unseen_and_have_zero_group_mean():
    parent = make_model().eval()
    model = AllClassCenteredGroupRouter(parent).eval()
    images = torch.randn(4, 768, generator=torch.Generator().manual_seed(42))
    unseen = torch.arange(200)[~torch.isin(torch.arange(200), parent.adapted_classes)]
    _, roles = model.component_logits(images, unseen)
    assert roles.shape == (4, 50, 3)
    assert float(roles.abs().sum()) > 0.0
    assert torch.allclose(roles.mean(dim=-1), torch.zeros(4, 50), atol=2e-6)


def test_acgr_only_router_receives_gradients():
    parent = make_model().eval()
    model = AllClassCenteredGroupRouter(parent)
    images = torch.randn(6, 768, generator=torch.Generator().manual_seed(43))
    loss = torch.nn.functional.cross_entropy(
        model.logits(images, parent.adapted_classes), torch.arange(6)
    )
    loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in model.router.parameters()
    )
    assert all(parameter.grad is None for parameter in parent.parameters())


def test_acgr_config_contract():
    config, _ = load_config(ROOT / "config/tries/v2_try_013_acgr_seed7.yaml")
    assert config["idea_id"] == "IDEA-004"
    assert config["routing_semantics"] == "all_class_centered_roles"
    assert config["role_scale"] == 0.65
