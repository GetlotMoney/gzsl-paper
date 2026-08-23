from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model.innovations.semantic_orthogonal import classwise_bi_orthogonal_residual
from model.innovations.train_clre import load_config


ROOT = Path(__file__).resolve().parents[1]


class BOCRTest(unittest.TestCase):
    def test_residual_is_orthogonal_to_two_classwise_directions(self):
        generator = torch.Generator().manual_seed(647)
        source = torch.randn(5, 8, generator=generator)
        first = torch.randn(5, 8, generator=generator)
        second = torch.randn(5, 8, generator=generator)
        residual = classwise_bi_orthogonal_residual(source, first, second)
        q1 = F.normalize(first, dim=-1)
        q2 = F.normalize(second - (second * q1).sum(-1, keepdim=True) * q1, dim=-1)
        self.assertLess(float((residual * q1).sum(-1).abs().max()), 1e-5)
        self.assertLess(float((residual * q2).sum(-1).abs().max()), 1e-5)

    def test_bocr_config_identity_and_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-032_bocr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.bocr.v1")
        self.assertAlmostEqual(config["comparison_H"], 78.0721851209539)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
