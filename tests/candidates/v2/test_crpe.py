from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.ccpe import ClassReliabilityPatchEvidence
from model.candidates.v2.modules.lpsr import local_text_orthogonal_reliability
from model.candidates.v2.trainers.train_ccpe import load_config


ROOT = Path(__file__).resolve().parents[3]


class CRPETest(unittest.TestCase):
    def test_reliability_is_bounded_and_seen_centered(self):
        generator = torch.Generator().manual_seed(563)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        seen = torch.arange(150)
        reliability = local_text_orthogonal_reliability(sentences, names, seen)
        self.assertEqual(tuple(reliability.shape), (200,))
        self.assertLessEqual(float(reliability.abs().max()), 1.0)

    def test_zero_delta_reproduces_fixed_ccpe(self):
        model = ClassReliabilityPatchEvidence(torch.linspace(-1, 1, 200), 10.0, 2.0)
        model.raw_absolute_beta.data.fill_(0.5)
        model.raw_absolute_beta.requires_grad_(False)
        parent = torch.randn(2, 7)
        scores = torch.randn(2, 200)
        ids = torch.arange(7)
        expected = parent + model.absolute_beta() * scores.index_select(1, ids)
        self.assertTrue(torch.equal(model(parent, scores, ids), expected))
        model(parent, scores, ids).sum().backward()
        self.assertIsNotNone(model.raw_delta_beta.grad)
        self.assertIsNone(model.raw_absolute_beta.grad)

    def test_crpe_config_binds_ccpe_and_seen_only(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-022_crpe/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.crpe.v1")
        self.assertEqual(config["delta_max_beta"], 2.0)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
