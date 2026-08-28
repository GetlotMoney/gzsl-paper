from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.ccpe import (
    PatchConsensusMarginEvidence,
    class_conditioned_patch_mean_gap_scores,
)
from model.candidates.v2.trainers.train_ccpe import load_config


ROOT = Path(__file__).resolve().parents[1]


class PCMETest(unittest.TestCase):
    def test_mean_and_gap_are_separate_channels(self):
        patches = torch.zeros(1, 576, 768)
        patches[0, 0, 0] = 1.0
        patches[0, 1, 0] = 0.5
        text = torch.zeros(1, 768)
        text[0, 0] = 1.0
        scores = class_conditioned_patch_mean_gap_scores(
            patches, text, torch.device("cpu"), chunk_size=1
        )
        self.assertEqual(tuple(scores.shape), (1, 2))
        self.assertGreaterEqual(float(scores[0, 0]), 0.0)
        self.assertGreaterEqual(float(scores[0, 1]), 0.0)

    def test_zero_gap_beta_reproduces_fixed_ccpe(self):
        model = PatchConsensusMarginEvidence(10.0, 5.0)
        model.raw_absolute_beta.data.fill_(0.5)
        model.raw_absolute_beta.requires_grad_(False)
        parent = torch.randn(2, 7)
        scores = torch.randn(2, 400)
        ids = torch.arange(7)
        expected = parent + model.absolute_beta() * scores[:, :200].index_select(1, ids)
        self.assertTrue(torch.equal(model(parent, scores, ids), expected))
        model(parent, scores, ids).sum().backward()
        self.assertIsNotNone(model.raw_gap_beta.grad)
        self.assertIsNone(model.raw_absolute_beta.grad)

    def test_pcme_config_binds_ccpe_and_seen_only(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-020_pcme/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.pcme.v1")
        self.assertEqual(config["gap_max_beta"], 5.0)
        self.assertEqual(
            config["ccpe_model_sha256"],
            "e3b2685b07883b976962c38804825e4043c500679003a869b4bc6997f60cfaf9",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
