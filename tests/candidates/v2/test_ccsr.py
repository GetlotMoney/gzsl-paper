from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.ccsr import ClassConditionedSentenceRouting
from model.candidates.v2.trainers.train_ccsr import load_config


ROOT = Path(__file__).resolve().parents[3]


class CCSRTest(unittest.TestCase):
    def test_zero_delta_uses_same_weights_for_all_classes(self):
        generator = torch.Generator().manual_seed(691)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        base = torch.softmax(torch.randn(8, generator=generator), dim=0)
        model = ClassConditionedSentenceRouting(
            sentences, names, torch.arange(150), base, 13.0, 2.0
        )
        weights = model.class_sentence_weights().detach()
        self.assertTrue(torch.allclose(weights, base.unsqueeze(0).expand_as(weights)))
        model.prototypes().sum().backward()
        self.assertIsNotNone(model.raw_delta.grad)

    def test_ccsr_config_binds_casr(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-039_ccsr/configs/RUN-001.yaml"
        )
        self.assertEqual(
            config["casr_model_sha256"],
            "6056345e17786ee84e62d9489368ade4e1616b03b26f33d9d9741d77af6d2be5",
        )
        self.assertEqual(config["max_delta"], 2.0)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
