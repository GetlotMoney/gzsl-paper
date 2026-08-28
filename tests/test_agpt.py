from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.agpt import AmbiguityGatedPatchTieBreaker
from model.candidates.v2.trainers.train_agpt import load_config


ROOT = Path(__file__).resolve().parents[1]


class AGPTTest(unittest.TestCase):
    def _model(self):
        groups = torch.full((200,), -1, dtype=torch.long)
        groups[:2] = 0
        return AmbiguityGatedPatchTieBreaker(
            torch.randn(200, 768), 13.0, groups, 0.25, 0.1, 5.0
        )

    def test_zero_beta_reproduces_sdcr_and_patch_only_changes_top2(self):
        model = self._model()
        images = torch.randn(1, 768)
        parent = torch.full((1, 200), -10.0)
        parent[0, 0], parent[0, 1] = 1.0, 0.9
        scores = torch.randn(1, 200)
        baseline = model(parent, images, scores)
        with torch.no_grad():
            model.raw_beta.fill_(0.2)
        changed = model(parent, images, scores)
        changed_positions = (changed - baseline).abs().gt(1e-8).nonzero()[:, 1]
        self.assertTrue(set(changed_positions.tolist()).issubset({0, 1}))
        self.assertTrue(
            torch.allclose(
                baseline[:, :2].mean(dim=1),
                changed[:, :2].mean(dim=1),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_config_binds_patch_and_agct_gate(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-061_agpt/configs/RUN-001.yaml"
        )
        self.assertEqual(config["threshold_quantile"], 0.25)
        self.assertEqual(config["patch_top_k"], 2)
        self.assertFalse(config["feature_provenance_complete"])
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
