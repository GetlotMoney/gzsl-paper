from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model.innovations.rsdm import (
    FullSemanticSymmetricDiagonalMetric,
    ResidualSymmetricDiagonalMetric,
)
from model.innovations.train_rsdm import load_config


ROOT = Path(__file__).resolve().parents[1]


class RSDMTest(unittest.TestCase):
    def test_identity_metric_reproduces_fixed_residual_branch(self):
        generator = torch.Generator().manual_seed(787)
        prototypes = torch.randn(200, 768, generator=generator)
        images = torch.randn(4, 768, generator=generator)
        parent = torch.randn(4, 200, generator=generator)
        model = ResidualSymmetricDiagonalMetric(prototypes, 13.0, 0.1)
        expected = parent + 13.0 * (
            F.normalize(images, dim=-1)
            @ F.normalize(prototypes, dim=-1).T
        )
        self.assertTrue(torch.equal(model(parent, images), expected))

    def test_metric_weights_receive_gradient_and_are_bounded(self):
        generator = torch.Generator().manual_seed(797)
        model = ResidualSymmetricDiagonalMetric(
            torch.randn(200, 768, generator=generator), 13.0, 0.1
        )
        logits = model(
            torch.zeros(3, 200), torch.randn(3, 768, generator=generator)
        )
        logits.sum().backward()
        self.assertIsNotNone(model.metric.raw_log_weight.grad)
        weights = model.metric.weight().detach()
        self.assertGreaterEqual(float(weights.min()), float(torch.exp(torch.tensor(-0.2))))
        self.assertLessEqual(float(weights.max()), float(torch.exp(torch.tensor(0.2))))

    def test_config_binds_sdcr_and_no_expert_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-048_rsdm/configs/RUN-001.yaml"
        )
        self.assertEqual(config["max_log_weight"], 0.1)
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_full_metric_identity_reproduces_three_branch_parent(self):
        generator = torch.Generator().manual_seed(809)
        images = torch.randn(3, 768, generator=generator)
        parent_prototypes = torch.randn(200, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        residual = torch.randn(200, 768, generator=generator)
        class_beta = torch.randn(200, generator=generator)
        seen = torch.arange(150)
        scale, residual_beta, seen_gamma = 11.0, 13.0, 0.15
        normalized = F.normalize(images, dim=-1)
        parent_logits = (
            normalized @ F.normalize(parent_prototypes, dim=-1).T * scale
            + (normalized @ F.normalize(names, dim=-1).T)
            * class_beta.unsqueeze(0)
        )
        parent_logits[:, seen] -= seen_gamma
        expected = parent_logits + residual_beta * (
            normalized @ F.normalize(residual, dim=-1).T
        )
        model = FullSemanticSymmetricDiagonalMetric(
            parent_prototypes, scale, names, class_beta, residual,
            residual_beta, seen, seen_gamma, 0.1
        )
        self.assertTrue(
            torch.allclose(
                model(parent_logits, images), expected, atol=1e-6, rtol=1e-6
            )
        )

    def test_full_metric_config_uses_same_bounds(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-049_fsdm/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.fsdm.v1")
        self.assertEqual(config["max_log_weight"], 0.1)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
