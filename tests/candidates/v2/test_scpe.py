from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.ccpe import spatially_coherent_patch_scores
from model.candidates.v2.trainers.train_ccpe import load_config


ROOT = Path(__file__).resolve().parents[3]


class SCPETest(unittest.TestCase):
    def test_nearby_top2_receive_more_evidence_than_distant_top2(self):
        texts = torch.zeros(1, 768)
        texts[0, 0] = 1.0
        patches = torch.zeros(2, 576, 768)
        patches[0, 0, 0] = 1.0
        patches[0, 1, 0] = 1.0
        patches[1, 0, 0] = 1.0
        patches[1, 575, 0] = 1.0
        scores = spatially_coherent_patch_scores(
            patches, texts, torch.device("cpu"), chunk_size=1
        )
        self.assertGreater(float(scores[0, 0]), float(scores[1, 0]))

    def test_scpe_config_identity_and_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-016_scpe/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.scpe.v1")
        self.assertEqual(config["patch_top_k"], 2)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
