from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.sdrs import SemanticDisagreementResidualScaling
from model.candidates.v2.trainers.train_sdrs import load_config


ROOT = Path(__file__).resolve().parents[3]


class SDRSTest(unittest.TestCase):
    def test_zero_slope_is_exact_ncra_parent(self):
        generator = torch.Generator().manual_seed(461)
        parent_prototypes = torch.randn(8, 6, generator=generator)
        names = torch.randn(8, 6, generator=generator)
        seen = torch.arange(6)
        model = SemanticDisagreementResidualScaling(
            parent_prototypes, names, seen, base_beta=17.0, max_delta=5.0
        )
        images = torch.randn(4, 6, generator=generator)
        parent_logits = torch.randn(4, 8, generator=generator)
        expected = parent_logits + 17.0 * model.residual_logits(images)
        self.assertTrue(torch.equal(model(parent_logits, images), expected))
        model(parent_logits, images).sum().backward()
        self.assertIsNotNone(model.raw_slope.grad)

    def test_config_is_seen_only_no_expert(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-012_sdrs/configs/RUN-001.yaml"
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])
        source = (ROOT / "model/candidates/v2/trainers/train_sdrs.py").read_text(encoding="utf-8")
        self.assertNotIn('["att"]', source)

    def test_conservative_rescue_config_is_accepted(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-012_sdrs/configs/RUN-002.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.sdrs.v2")
        self.assertEqual(config["max_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
