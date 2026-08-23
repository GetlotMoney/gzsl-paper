from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model.innovations.rsdm import ResidualSymmetricDiagonalMetric
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


if __name__ == "__main__":
    unittest.main()
