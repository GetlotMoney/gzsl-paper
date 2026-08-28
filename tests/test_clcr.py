from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model.candidates.v2.modules.clcr import CrossLLMComplementaryResidual
from model.candidates.v2.trainers.train_clcr import load_config


ROOT = Path(__file__).resolve().parents[1]


class CLCRTest(unittest.TestCase):
    def test_zero_claude_beta_reproduces_sdcr_parent(self):
        generator = torch.Generator().manual_seed(821)
        sdcr = torch.randn(200, 768, generator=generator)
        claude = torch.randn(200, 768, generator=generator)
        images = torch.randn(3, 768, generator=generator)
        parent = torch.randn(3, 200, generator=generator)
        model = CrossLLMComplementaryResidual(sdcr, 13.0, claude, 5.0)
        expected = parent + 13.0 * (
            F.normalize(images, dim=-1) @ F.normalize(sdcr, dim=-1).T
        )
        self.assertTrue(torch.equal(model(parent, images), expected))

    def test_claude_beta_is_bounded_and_trainable(self):
        generator = torch.Generator().manual_seed(823)
        model = CrossLLMComplementaryResidual(
            torch.randn(200, 768, generator=generator),
            13.0,
            torch.randn(200, 768, generator=generator),
            5.0,
        )
        logits = model(
            torch.zeros(2, 200), torch.randn(2, 768, generator=generator)
        )
        logits.sum().backward()
        self.assertIsNotNone(model.raw_beta.grad)
        self.assertLessEqual(abs(float(model.beta().detach())), 5.0)

    def test_config_binds_two_text_sources_and_seen_only_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-051_clcr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["max_beta"], 5.0)
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertTrue(config["oclr_model_sha256"])
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
