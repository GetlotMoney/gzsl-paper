from __future__ import annotations

from pathlib import Path

import torch

from model.innovations.epc import EpisodicPriorCalibration
from model.innovations.train_epc import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_epc_initially_reproduces_parent_and_targets_only_selected_classes():
    module = EpisodicPriorCalibration(max_margin=0.5)
    logits = torch.randn(3, 6, generator=torch.Generator().manual_seed(61))
    classes = torch.tensor([1, 2, 4, 7, 8, 9])
    target = torch.tensor([2, 8])
    assert torch.equal(module(logits, classes, target), logits)
    module.raw_margin.data.fill_(0.4)
    changed = module(logits, classes, target)
    delta = changed - logits
    assert torch.all(delta[:, torch.tensor([0, 2, 3, 5])] == 0)
    assert torch.all(delta[:, torch.tensor([1, 4])] > 0)


def test_epc_margin_is_bounded_and_receives_gradient():
    module = EpisodicPriorCalibration(max_margin=0.5)
    logits = torch.randn(4, 5, generator=torch.Generator().manual_seed(62))
    output = module(logits, torch.arange(5), torch.tensor([3, 4]))
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1, 3, 4])).backward()
    assert module.raw_margin.grad is not None
    module.raw_margin.data.fill_(100.0)
    assert float(module.margin().detach()) <= 0.5


def test_epc_config_and_training_boundary():
    config, _ = load_config(ROOT / "config/tries/v2_try_019_epc_seed7.yaml")
    assert config["idea_id"] == "IDEA-006"
    assert config["max_margin"] == 0.5
    source = (ROOT / "model/innovations/train_epc.py").read_text(encoding="utf-8")
    assert source.index("for epoch in range") < source.index("# official test严格在EPC训练结束后加载。")
