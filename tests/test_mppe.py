from pathlib import Path
import unittest

import torch

from model.innovations.ccpe import multi_part_patch_scores
from model.innovations.train_ccpe import load_config


ROOT = Path(__file__).resolve().parents[1]


class MPPETest(unittest.TestCase):
    def test_each_part_can_select_a_different_patch(self):
        patches = torch.zeros(1, 576, 768)
        patches[0, 0, 0] = 1.0
        patches[0, 575, 1] = 1.0
        parts = torch.zeros(1, 2, 768)
        parts[0, 0, 0] = 1.0
        parts[0, 1, 1] = 1.0
        scores = multi_part_patch_scores(
            patches, parts, torch.device("cpu"), chunk_size=1
        )
        self.assertTrue(torch.equal(scores, torch.ones_like(scores)))

    def test_mppe_config_identity_and_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-017_mppe/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.mppe.v1")
        self.assertEqual(config["patch_top_k"], 1)
        self.assertEqual(config["patch_chunk_size"], 4)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
