from pathlib import Path
import unittest

import torch

from model.innovations.ccpe import (
    ClassConditionedPatchEvidence,
    class_conditioned_patch_scores,
)
from model.innovations.train_ccpe import load_config


ROOT = Path(__file__).resolve().parents[1]


class CCPETest(unittest.TestCase):
    def test_zero_beta_exact_parent_and_gradient(self):
        model = ClassConditionedPatchEvidence(max_beta=10.0)
        parent = torch.randn(3, 7)
        scores = torch.randn(3, 7)
        self.assertTrue(torch.equal(model(parent, scores), parent))
        model(parent, scores).sum().backward()
        self.assertIsNotNone(model.raw_beta.grad)

    def test_each_class_selects_its_own_patches(self):
        patches = torch.zeros(1, 576, 768)
        patches[0, 0, 0] = 1.0
        patches[0, 1, 1] = 1.0
        texts = torch.zeros(2, 768)
        texts[0, 0] = 1.0
        texts[1, 1] = 1.0
        scores = class_conditioned_patch_scores(
            patches, texts, 1, torch.device("cpu"), chunk_size=1
        )
        self.assertEqual(tuple(scores.shape), (1, 2))
        self.assertTrue(torch.equal(scores, torch.ones_like(scores)))

    def test_config_is_seen_only_and_patch_bound(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-015_ccpe/configs/RUN-001.yaml"
        )
        self.assertEqual(config["patch_top_k"], 8)
        self.assertFalse(config["unseen_images_used_for_gradient"])
        self.assertFalse(config["feature_provenance_complete"])


if __name__ == "__main__":
    unittest.main()
