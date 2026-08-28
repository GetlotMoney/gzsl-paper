from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.aosr import AdaptiveOrthogonalSentenceResidual
from model.candidates.v2.trainers.train_aosr import load_config


ROOT = Path(__file__).resolve().parents[3]


class AOSRTest(unittest.TestCase):
    def test_initial_sentence_weights_are_equal_and_trainable(self):
        generator = torch.Generator().manual_seed(677)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        model = AdaptiveOrthogonalSentenceResidual(sentences, names, 13.0)
        weights = model.sentence_weights().detach()
        self.assertTrue(torch.equal(weights, torch.full((8,), 1.0 / 8.0)))
        model.prototypes().sum().backward()
        self.assertIsNotNone(model.raw_sentence_weights.grad)

    def test_aosr_config_binds_oesr(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-037_aosr/configs/RUN-001.yaml"
        )
        self.assertEqual(
            config["oesr_model_sha256"],
            "74bd92c84278c4f623e2ae357358a34bc07a810714b1f03c236065fc77a9a8e1",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_seed7_chain_binds_seed7_oesr(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-037_aosr/configs/RUN-002.yaml"
        )
        self.assertEqual(config["random_seed"], 7)
        self.assertEqual(
            config["oesr_model_sha256"],
            "2cbc9d6860c398ce938833e345fffaae6bc71b5cff6accc7dd304132c15a7bb2",
        )


if __name__ == "__main__":
    unittest.main()
