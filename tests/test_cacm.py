from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.aclm import ClassAdaptiveCrossLLMMixture
from model.candidates.v2.trainers.train_aclm import load_config


ROOT = Path(__file__).resolve().parents[1]


class CACMTest(unittest.TestCase):
    def test_initial_weights_are_half_then_class_adaptive(self):
        generator = torch.Generator().manual_seed(631)
        claude = torch.randn(200, 768, generator=generator)
        merge = torch.randn(200, 768, generator=generator)
        model = ClassAdaptiveCrossLLMMixture(
            claude, merge, torch.arange(150), 19.0, 16.0
        )
        initial = model.weight_stats()
        self.assertEqual(initial["mean"], 0.5)
        self.assertEqual(initial["std"], 0.0)
        parent = torch.randn(2, 7, generator=generator)
        images = torch.randn(2, 768, generator=generator)
        model(parent, images, torch.arange(7)).sum().backward()
        self.assertIsNotNone(model.raw_bias.grad)
        self.assertIsNotNone(model.raw_slope.grad)

    def test_cacm_config_identity(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-028_cacm/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.cacm.v1")
        self.assertEqual(config["experiment_id"], "V2-INNOVATION-028")
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
