from pathlib import Path
import unittest

import torch

from model.innovations.sdcr import SentenceDropoutConservativeRouting
from model.innovations.train_sdcr import load_config


ROOT = Path(__file__).resolve().parents[1]


class WSDRTest(unittest.TestCase):
    def test_explicit_mask_role(self):
        generator = torch.Generator().manual_seed(743)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        base = torch.full((8,), 1.0 / 8.0)
        model = SentenceDropoutConservativeRouting(
            sentences, names, base, 13.0, 0.5, drop_count=1
        ).train()
        weights = model.active_sentence_weights(mask_roles=[3])
        self.assertEqual(float(weights[3].detach()), 0.0)
        self.assertEqual(model.last_masked_roles, [3])

    def test_wsdr_config_uses_two_candidates(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-043_wsdr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.wsdr.v1")
        self.assertEqual(config["candidate_masks"], 2)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
