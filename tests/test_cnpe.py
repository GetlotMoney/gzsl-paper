from pathlib import Path
import unittest

import torch

from model.innovations.ccpe import normalize_patch_scores_by_seen_reference
from model.innovations.train_ccpe import load_config


ROOT = Path(__file__).resolve().parents[1]


class CNPETest(unittest.TestCase):
    def test_seen_reference_normalization_removes_class_bias(self):
        train = torch.tensor([[1.0, 10.0], [3.0, 14.0]])
        scores = {"train": train, "seen": train.clone(), "unseen": train.clone()}
        normalized, mean, std = normalize_patch_scores_by_seen_reference(scores)
        self.assertTrue(torch.allclose(mean, torch.tensor([2.0, 12.0])))
        self.assertTrue(torch.allclose(std, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(normalized["train"].mean(dim=0), torch.zeros(2)))

    def test_cnpe_config_identity_and_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-018_cnpe/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.cnpe.v1")
        self.assertEqual(config["max_beta"], 2.0)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
