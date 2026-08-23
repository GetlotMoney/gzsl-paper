from pathlib import Path
import unittest

import torch

from model.innovations.agct import AmbiguityGatedCrossLLMTieBreaker
from model.innovations.train_agct import load_config, select_margin_threshold


ROOT = Path(__file__).resolve().parents[1]


class AGCTTest(unittest.TestCase):
    def _model(self):
        generator = torch.Generator().manual_seed(877)
        groups = torch.full((200,), -1, dtype=torch.long)
        groups[:4] = torch.tensor([0, 0, 1, 1])
        return AmbiguityGatedCrossLLMTieBreaker(
            torch.randn(200, 768, generator=generator),
            13.0,
            torch.randn(200, 768, generator=generator),
            groups,
            0.5,
            0.1,
            5.0,
        )

    def test_gate_only_activates_for_low_margin_same_group_top2(self):
        model = self._model()
        logits = torch.full((3, 200), -10.0)
        logits[0, 0], logits[0, 1] = 1.0, 0.8
        logits[1, 0], logits[1, 2] = 1.0, 0.8
        logits[2, 0], logits[2, 1] = 2.0, 0.0
        gate, same, _ = model.gate_values(logits)
        self.assertTrue(bool(same[0]))
        self.assertFalse(bool(same[1]))
        self.assertGreater(float(gate[0]), 0.5)
        self.assertEqual(float(gate[1]), 0.0)
        self.assertLess(float(gate[2]), 1e-5)

    def test_zero_beta_reproduces_sdcr_parent(self):
        model = self._model()
        images = torch.randn(2, 768)
        parent = torch.randn(2, 200)
        expected = parent + model.sdcr_beta * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.sdcr_prototypes.T
        )
        self.assertTrue(torch.equal(model(parent, images), expected))

    def test_threshold_uses_wrong_same_group_median(self):
        margins = torch.tensor([0.1, 0.2, 0.4, 0.8])
        same = torch.tensor([True, True, True, False])
        wrong = torch.tensor([True, False, True, True])
        threshold, stats = select_margin_threshold(margins, same, wrong, 0.5)
        self.assertAlmostEqual(threshold, 0.25)
        self.assertEqual(stats["candidate_count"], 2)
        self.assertEqual(stats["source"], "train_wrong_same_group")

    def test_config_binds_train_only_threshold_and_claude(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-058_agct/configs/RUN-001.yaml"
        )
        self.assertEqual(config["threshold_source"], "train_wrong_same_group_margin")
        self.assertEqual(config["margin_temperature"], 0.1)
        self.assertTrue(config["claude_embeddings_sha256"])
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
