from __future__ import annotations

from pathlib import Path

import torch

from model.innovations.spa import SeenPrototypeAnchor
from model.innovations.train_spa import load_config


ROOT = Path(__file__).resolve().parents[1]


def make_spa():
    generator = torch.Generator().manual_seed(71)
    parent = torch.randn(200, 768, generator=generator)
    seen = torch.tensor([i for i in range(200) if i % 4 != 0])
    centroids = torch.randn(150, 768, generator=generator)
    return SeenPrototypeAnchor(parent, seen, centroids, torch.tensor(10.0))


def test_spa_off_is_parent_and_unseen_never_changes():
    model = make_spa()
    off = model.prototypes(enabled=False)
    on = model.prototypes()
    unseen = torch.arange(200)[~torch.isin(torch.arange(200), model.seenclasses)]
    assert torch.equal(off, model.parent_prototypes)
    assert torch.equal(on.index_select(0, unseen), off.index_select(0, unseen))
    assert not torch.equal(on.index_select(0, model.seenclasses), off.index_select(0, model.seenclasses))


def test_spa_strength_is_bounded_and_trainable():
    model = make_spa()
    images = torch.randn(6, 768, generator=torch.Generator().manual_seed(72))
    loss = torch.nn.functional.cross_entropy(model.logits(images, model.seenclasses), torch.arange(6)) + 0.1 * model.topology_loss()
    loss.backward()
    assert model.raw_strength.grad is not None and torch.isfinite(model.raw_strength.grad)
    model.raw_strength.data.fill_(100.0)
    assert float(model.strength().detach()) <= 0.10000001


def test_spa_config_and_training_boundary():
    config, _ = load_config(ROOT / "config/tries/v2_try_025_spa_seed7.yaml")
    assert config["idea_id"] == "IDEA-008"
    assert config["max_strength"] == 0.1
    source = (ROOT / "model/innovations/train_spa.py").read_text(encoding="utf-8")
    assert source.index("for epoch in range") < source.index("# official test严格在SPA训练结束后加载。")
