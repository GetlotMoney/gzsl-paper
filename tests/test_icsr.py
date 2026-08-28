from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.icsr import ImageConditionedSentenceRouting
from model.candidates.v2.trainers.train_icsr import load_config


ROOT = Path(__file__).resolve().parents[1]


class ICSRTest(unittest.TestCase):
    def test_zero_gate_returns_base_weights_and_has_gradients(self):
        generator = torch.Generator().manual_seed(709)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        base = torch.softmax(torch.randn(8, generator=generator), dim=0)
        model = ImageConditionedSentenceRouting(sentences, names, base, 13.0, 32, 0.5)
        images = torch.randn(3, 768, generator=generator)
        weights = model.sentence_weights(images).detach()
        self.assertTrue(torch.allclose(weights, base.unsqueeze(0).expand_as(weights)))
        parent = torch.randn(3, 7, generator=generator)
        model(parent, images, torch.arange(7)).sum().backward()
        self.assertIsNotNone(model.gate[-1].weight.grad)

    def test_icsr_config_has_chunking_and_kl(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-040_icsr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["eval_batch_size"], 128)
        self.assertEqual(config["kl_weight"], 0.01)
        self.assertEqual(
            config["casr_model_sha256"],
            "6056345e17786ee84e62d9489368ade4e1616b03b26f33d9d9741d77af6d2be5",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
