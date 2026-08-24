from __future__ import annotations

from pathlib import Path
import unittest

import torch

from model.innovations.elpt import (
    ELPTGate,
    VariableClassTGVPR,
    blend_prototypes,
    fixed_class_folds,
    semantic_balanced_class_folds,
    class_fold_sha256,
    gate_features,
)
from model.innovations.train_elpt import load_config
from model.tg_vpr_h1 import TGVPRH1FixedEqual


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    generator = torch.Generator().manual_seed(2406)
    sentences = torch.randn(200, 8, 768, generator=generator)
    seen = torch.tensor([i for i in range(200) if i % 4 != 0])
    centroids = torch.randn(150, 768, generator=generator)
    return sentences, seen, centroids


def test_variable_150_path_is_bitwise_equal_to_v2():
    sentences, seen, centroids = _inputs()
    baseline = TGVPRH1FixedEqual(
        sentences, seen, centroids, dropout=0.0
    ).eval()
    variable = VariableClassTGVPR(
        sentences, seen, centroids, dropout=0.0
    ).eval()
    variable.load_state_dict(baseline.state_dict(), strict=True)
    assert baseline.state_dict().keys() == variable.state_dict().keys()
    assert torch.equal(baseline.prototypes(), variable.prototypes())
    images = torch.randn(4, 768, generator=torch.Generator().manual_seed(3))
    assert torch.equal(baseline.logits(images), variable.logits(images))
    assert torch.equal(baseline.topology_loss(), variable.topology_loss())


def test_fixed_folds_are_disjoint_and_cover_seen_classes():
    _, seen, _ = _inputs()
    folds = fixed_class_folds(seen)
    assert len(folds) == 3
    pseudo_unseen = [fold[1] for fold in folds]
    assert all(x.numel() == 50 for x in pseudo_unseen)
    assert all(fold[0].numel() == 100 for fold in folds)
    merged = torch.cat(pseudo_unseen).sort().values
    assert torch.equal(merged, seen.sort().values)
    assert all(
        not torch.isin(pseudo_unseen[i], pseudo_unseen[j]).any()
        for i in range(3)
        for j in range(i + 1, 3)
    )


def test_semantic_balanced_folds_cover_outer_train_classes_deterministically():
    generator = torch.Generator().manual_seed(2407)
    sentences = torch.randn(200, 8, 768, generator=generator)
    classes = torch.arange(100)
    counts = torch.arange(1, 101)
    first = semantic_balanced_class_folds(classes, sentences, counts)
    second = semantic_balanced_class_folds(classes, sentences, counts)
    assert [fold[1].numel() for fold in first] == [34, 33, 33]
    assert torch.equal(
        torch.cat([fold[1] for fold in first]).sort().values,
        classes,
    )
    assert class_fold_sha256(first) == class_fold_sha256(second)
    assert all(
        not torch.isin(first[i][1], first[j][1]).any()
        for i in range(3)
        for j in range(i + 1, 3)
    )


def test_gate_initializes_at_point_one_and_receives_gradients():
    gate = ELPTGate()
    features = torch.randn(50, 4)
    alpha = gate(features)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.1), atol=1e-7)
    loss = alpha.square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in gate.parameters()
    )


def test_bounded_gate_keeps_point_one_initialization_and_caps_output():
    gate = ELPTGate(max_alpha=0.25)
    features = torch.randn(50, 4)
    alpha = gate(features)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.1), atol=1e-7)
    assert float(alpha.detach().max()) <= 0.25


def test_gate_features_and_blend_are_finite_and_normalized():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(50, 768, generator=generator)
    value = torch.randn(50, 768, generator=generator)
    support = torch.randn(100, 768, generator=generator)
    features = gate_features(base, value, support)
    assert features.shape == (50, 4)
    alpha = ELPTGate()(features)
    blended = blend_prototypes(base, value, alpha)
    assert torch.isfinite(features).all() and torch.isfinite(blended).all()
    assert torch.allclose(blended.norm(dim=-1), torch.ones(50), atol=1e-6)


def test_top5_vector_features_have_eight_dimensions():
    generator = torch.Generator().manual_seed(12)
    base = torch.randn(50, 768, generator=generator)
    value = torch.randn(50, 768, generator=generator)
    support = torch.randn(100, 768, generator=generator)
    features = gate_features(base, value, support, mode="top5_vector")
    assert features.shape == (50, 8)
    alpha = ELPTGate(input_dim=8, max_alpha=0.25)(features)
    assert alpha.shape == (50,)


def test_summary_std_features_have_five_dimensions():
    generator = torch.Generator().manual_seed(13)
    base = torch.randn(50, 768, generator=generator)
    value = torch.randn(50, 768, generator=generator)
    support = torch.randn(100, 768, generator=generator)
    features = gate_features(base, value, support, mode="summary_std")
    assert features.shape == (50, 5)
    assert torch.isfinite(features).all()


def test_try_config_is_frozen():
    config, digest = load_config(ROOT / "config" / "tries" / "v2_try_006_elpt_seed7.yaml")
    assert len(digest) == 64
    assert config["fold_count"] == 3
    assert config["fold_epochs"] == 50
    assert config["gate_epochs"] == 20
    assert config["gate_batch_half"] == 32


def test_rescue_config_reuses_fold_checkpoints():
    config, _ = load_config(
        ROOT / "config" / "tries" / "v2_try_007_elpt_rescue1_seed7.yaml"
    )
    assert config["gate_max_alpha"] == 0.25
    assert config["alpha_penalty"] == 0.01
    assert config["fold_checkpoint_dir"].endswith("V2-TRY-006")


def test_rescue2_config_uses_top5_vector():
    config, _ = load_config(
        ROOT / "config" / "tries" / "v2_try_008_elpt_rescue2_seed7.yaml"
    )
    assert config["gate_feature_mode"] == "top5_vector"
    assert config["gate_max_alpha"] == 0.25


def test_rescue3_config_uses_gate_ensemble():
    config, _ = load_config(
        ROOT / "config" / "tries" / "v2_try_009_elpt_rescue3_seed7.yaml"
    )
    assert config["gate_ensemble"] is True
    assert config["gate_feature_mode"] == "top5_vector"


class ELPTTest(unittest.TestCase):
    def test_equivalence(self):
        test_variable_150_path_is_bitwise_equal_to_v2()

    def test_folds(self):
        test_fixed_folds_are_disjoint_and_cover_seen_classes()

    def test_semantic_balanced_folds(self):
        test_semantic_balanced_folds_cover_outer_train_classes_deterministically()

    def test_gate_gradient(self):
        test_gate_initializes_at_point_one_and_receives_gradients()

    def test_bounded_gate(self):
        test_bounded_gate_keeps_point_one_initialization_and_caps_output()

    def test_features(self):
        test_gate_features_and_blend_are_finite_and_normalized()

    def test_top5_features(self):
        test_top5_vector_features_have_eight_dimensions()

    def test_config(self):
        test_try_config_is_frozen()

    def test_rescue_config(self):
        test_rescue_config_reuses_fold_checkpoints()

    def test_rescue2_config(self):
        test_rescue2_config_uses_top5_vector()

    def test_rescue3_config(self):
        test_rescue3_config_uses_gate_ensemble()


if __name__ == "__main__":
    unittest.main()
