from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.ccpe import ClassConditionedPatchEvidence
from model.candidates.v2.trainers.train_spcr import compose_logits, load_config


ROOT = Path(__file__).resolve().parents[1]


class DummySDCR(torch.nn.Module):
    def forward(self, logits, images, class_ids):
        return logits + 0.25


class SPCRTest(unittest.TestCase):
    def test_zero_patch_beta_reproduces_sdcr_parent(self):
        logits = torch.randn(3, 5)
        scores = torch.randn(3, 5)
        patch_model = ClassConditionedPatchEvidence(max_beta=5.0)
        result = compose_logits(
            logits, DummySDCR(), patch_model, torch.randn(3, 4), scores, None
        )
        self.assertTrue(torch.equal(result, logits + 0.25))

    def test_config_binds_top2_patch_and_sdcr_parent(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-052_spcr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["patch_top_k"], 2)
        self.assertEqual(config["max_beta"], 5.0)
        self.assertTrue(config["sdcr_model_sha256"])
        self.assertFalse(config["unseen_images_used_for_gradient"])
        self.assertFalse(config["feature_provenance_complete"])


if __name__ == "__main__":
    unittest.main()
