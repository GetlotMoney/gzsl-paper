from pathlib import Path
import unittest

import torch

from model.innovations.sdcr import SentenceDropoutConservativeRouting
from model.innovations.train_sdcr import load_config


ROOT = Path(__file__).resolve().parents[1]


class SDCRTest(unittest.TestCase):
    def test_train_masks_one_sentence_and_eval_restores_all(self):
        generator = torch.Generator().manual_seed(727)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        base = torch.softmax(torch.randn(8, generator=generator), dim=0)
        model = SentenceDropoutConservativeRouting(sentences, names, base, 13.0, 0.5)
        model.train()
        train_weights = model.active_sentence_weights()
        self.assertEqual(int((train_weights == 0).sum()), 1)
        self.assertGreaterEqual(model.last_masked_role, 0)
        model.eval()
        self.assertTrue(torch.allclose(model.active_sentence_weights(), base))
        self.assertEqual(model.last_masked_role, -1)

    def test_sdcr_config_binds_casr(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-041_sdcr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["kl_weight"], 0.01)
        self.assertEqual(
            config["casr_model_sha256"],
            "6056345e17786ee84e62d9489368ade4e1616b03b26f33d9d9741d77af6d2be5",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_seed5_reliability_binds_seed5_casr(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-041_sdcr/configs/RUN-002.yaml"
        )
        self.assertEqual(config["random_seed"], 5)
        self.assertEqual(
            config["casr_model_sha256"],
            "281658e2e01aef4ad5d2fa598af0039cc70774a76ec43328399aa2545b6515a7",
        )


if __name__ == "__main__":
    unittest.main()
