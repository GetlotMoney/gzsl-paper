from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.ccpe import DualScalePatchEvidence
from model.candidates.v2.trainers.train_ccpe import load_config


ROOT = Path(__file__).resolve().parents[3]


class DSPETest(unittest.TestCase):
    def test_zero_dual_beta_exact_parent_and_both_gradients(self):
        model = DualScalePatchEvidence(10.0, 2.0)
        parent = torch.randn(3, 7)
        scores = torch.randn(3, 400)
        ids = torch.arange(7)
        self.assertTrue(torch.equal(model(parent, scores, ids), parent))
        model(parent, scores, ids).sum().backward()
        self.assertIsNotNone(model.raw_absolute_beta.grad)
        self.assertIsNotNone(model.raw_normalized_beta.grad)

    def test_dspe_config_identity_and_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-019_dspe/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.dspe.v1")
        self.assertEqual(config["max_beta"], 10.0)
        self.assertEqual(config["normalized_max_beta"], 2.0)
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_stagewise_dspe_binds_ccpe_parent(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-019_dspe/configs/RUN-003.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.dspe.v2")
        self.assertEqual(
            config["ccpe_model_sha256"],
            "e3b2685b07883b976962c38804825e4043c500679003a869b4bc6997f60cfaf9",
        )


if __name__ == "__main__":
    unittest.main()
