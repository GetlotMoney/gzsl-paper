from pathlib import Path
import unittest

import torch

from model.innovations.train_pgsd import (
    centered_patch_reliability_weights,
    load_config,
    patch_reliability_weights,
)


ROOT = Path(__file__).resolve().parents[1]


class PGSDTest(unittest.TestCase):
    def test_patch_reliability_is_samplewise_and_bounded(self):
        scores = torch.zeros(3, 200)
        labels = torch.tensor([0, 1, 2])
        scores[0, 0] = 3.0
        scores[1, 1] = -3.0
        weights = patch_reliability_weights(scores, labels, 0.25)
        self.assertGreater(float(weights[0]), 1.0)
        self.assertLess(float(weights[1]), 1.0)
        self.assertGreaterEqual(float(weights.min()), 0.75)
        self.assertLessEqual(float(weights.max()), 1.25)

    def test_config_uses_train_patch_only_and_patch_free_inference(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-053_pgsd/configs/RUN-001.yaml"
        )
        self.assertEqual(config["reliability_amplitude"], 0.25)
        self.assertEqual(config["patch_top_k"], 2)
        self.assertIn("train_patch", config)
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_centered_weights_have_unit_mean_and_fixed_bounds(self):
        generator = torch.Generator().manual_seed(839)
        scores = torch.randn(32, 200, generator=generator)
        labels = torch.arange(32) % 200
        weights = centered_patch_reliability_weights(scores, labels, 0.25)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreaterEqual(float(weights.min()), 0.75)
        self.assertLessEqual(float(weights.max()), 1.25)
        self.assertGreater(float(weights.std(unbiased=False)), 0.0)

    def test_centered_config_binds_new_experiment(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-054_cpgsd/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.cpgsd.v1")
        self.assertTrue(config["center_weights"])


if __name__ == "__main__":
    unittest.main()
