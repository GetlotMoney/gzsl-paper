from __future__ import annotations

from pathlib import Path

import torch

from model.innovations.sgt import GraphResidualClassifier, GraphTransportStrength, semantic_graph_residual
from model.innovations.train_sgt import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_graph_residual_is_tangent_and_off_is_parent():
    generator = torch.Generator().manual_seed(101)
    base = torch.randn(200, 768, generator=generator); source_classes = torch.arange(100); target_classes = torch.arange(100, 150); source = torch.randn(100, 768, generator=generator); residual = semantic_graph_residual(base, source, source_classes, target_classes); target_base = torch.nn.functional.normalize(base, dim=-1).index_select(0, target_classes); assert residual.shape == (50, 768); assert torch.allclose((residual*target_base).sum(dim=-1), torch.zeros(50), atol=1e-5)
    strength = GraphTransportStrength(); classifier = GraphResidualClassifier(torch.nn.functional.normalize(base, dim=-1), target_classes, residual, strength, torch.tensor(10.0)); assert torch.equal(classifier.prototypes(enabled=False), classifier.parent_prototypes)


def test_graph_strength_is_trainable_and_config_is_frozen():
    strength = GraphTransportStrength(); strength().square().backward(); assert strength.raw_strength.grad is not None
    config, _ = load_config(ROOT / "config/tries/v2_try_044_sgt_seed7.yaml"); assert config["idea_id"] == "IDEA-013"; assert config["top_k"] == 5
    source = (ROOT / "model/innovations/train_sgt.py").read_text(encoding="utf-8"); assert source.index("for epoch in range") < source.index("# official test严格在SGT训练结束后加载。")
