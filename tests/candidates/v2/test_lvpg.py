from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.lvpg import ridge_predict_local_visual
from model.candidates.v2.trainers.train_ccpe import load_config


ROOT = Path(__file__).resolve().parents[3]


class LVPGTest(unittest.TestCase):
    def test_ridge_predicts_all_classes_from_seen_centroids(self):
        generator = torch.Generator().manual_seed(587)
        semantics = torch.randn(5, 6, generator=generator)
        seen = torch.tensor([0, 2, 4])
        centroids = torch.randn(3, 6, generator=generator)
        predicted = ridge_predict_local_visual(semantics, seen, centroids, 0.1)
        self.assertEqual(tuple(predicted.shape), (5, 6))
        self.assertTrue(torch.allclose(predicted.norm(dim=-1), torch.ones(5), atol=1e-5))

    def test_lvpg_config_is_seen_only_and_patch_bound(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-023_lvpg/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.lvpg.v1")
        self.assertEqual(config["ridge"], 0.1)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
