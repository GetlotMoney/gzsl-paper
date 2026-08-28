from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.orer import GammaResidualCalibration
from model.candidates.v2.trainers.train_orer import load_config


ROOT = Path(__file__).resolve().parents[1]


class ORERTest(unittest.TestCase):
    def test_zero_residual_matches_parent_gamma(self):
        model = GammaResidualCalibration(0.15, 0.1)
        logits = torch.randn(3, 5)
        mask = torch.tensor([True, True, False, False, False])
        self.assertTrue(torch.equal(model(logits, mask), model(logits, mask, enabled=False)))
        model(logits, mask).sum().backward()
        self.assertIsNotNone(model.raw_residual.grad)

    def test_orer_config_binds_folds_and_oclr(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-034_orer/configs/RUN-001.yaml"
        )
        self.assertEqual(set(config["fold_model_sha256"]), {"0", "1", "2"})
        self.assertEqual(config["max_gamma_residual"], 0.1)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
