from pathlib import Path
import unittest

import torch

from model.candidates.v2.trainers.train_sdcr import load_config, sample_importance_mask


ROOT = Path(__file__).resolve().parents[1]


class IADRTest(unittest.TestCase):
    def test_importance_sampler_follows_weight_support(self):
        weights = torch.zeros(8, requires_grad=True)
        weights = weights.clone()
        weights[3] = 1.0
        generator = torch.Generator().manual_seed(751)
        self.assertEqual(sample_importance_mask(weights, generator), 3)

    def test_iadr_config_binds_weight_proportional_sampling(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-044_iadr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.iadr.v1")
        self.assertEqual(
            config["sampling_strategy"], "current_weight_proportional"
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
