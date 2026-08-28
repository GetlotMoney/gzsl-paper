from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.sdcr import SentenceDropoutConservativeRouting
from model.candidates.v2.trainers.train_sdcr import load_config


ROOT = Path(__file__).resolve().parents[3]


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

    def test_two_sentence_dropout(self):
        generator = torch.Generator().manual_seed(733)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        base = torch.full((8,), 1.0 / 8.0)
        model = SentenceDropoutConservativeRouting(
            sentences, names, base, 13.0, 0.5, drop_count=2
        ).train()
        weights = model.active_sentence_weights()
        self.assertEqual(int((weights == 0).sum()), 2)
        self.assertEqual(len(model.last_masked_roles), 2)

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

    def test_drop2_config(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-041_sdcr/configs/RUN-003.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.sdcr.v2")
        self.assertEqual(config["drop_count"], 2)


if __name__ == "__main__":
    unittest.main()
