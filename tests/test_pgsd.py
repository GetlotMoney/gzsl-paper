from pathlib import Path
import unittest

import torch

from model.innovations.train_pgsd import load_config, patch_reliability_weights


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


if __name__ == "__main__":
    unittest.main()
