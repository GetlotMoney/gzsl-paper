from pathlib import Path
import unittest

import torch

from model.innovations.ncsr import NeighborhoodContrastiveSemanticResidual
from model.innovations.train_ncsr import load_config


ROOT = Path(__file__).resolve().parents[1]


class NCSRTest(unittest.TestCase):
    def test_neighbors_exclude_self_and_contrastive_is_orthogonal(self):
        generator = torch.Generator().manual_seed(773)
        prototypes = torch.randn(200, 768, generator=generator)
        model = NeighborhoodContrastiveSemanticResidual(prototypes, 13.0, 5, 5.0)
        own = torch.arange(200).unsqueeze(1)
        self.assertFalse(bool((model.neighbor_indices.cpu() == own).any()))
        cosine = (
            model.base_prototypes * model.contrastive_prototypes
        ).sum(dim=-1).abs()
        self.assertLess(float(cosine.max()), 1e-5)

    def test_zero_gamma_adds_only_fixed_sdcr_parent(self):
        generator = torch.Generator().manual_seed(779)
        prototypes = torch.randn(200, 768, generator=generator)
        images = torch.randn(3, 768, generator=generator)
        parent = torch.randn(3, 200, generator=generator)
        model = NeighborhoodContrastiveSemanticResidual(prototypes, 13.0, 5, 5.0)
        expected = parent + 13.0 * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.base_prototypes.T
        )
        self.assertTrue(torch.allclose(model(parent, images), expected))

    def test_config_binds_sdcr_and_seen_only_training(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-047_ncsr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["neighbor_k"], 5)
        self.assertEqual(config["max_gamma"], 5.0)
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_rescue_config_only_lowers_learning_rate(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-047_ncsr/configs/RUN-002.yaml"
        )
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertEqual(config["neighbor_k"], 5)
        self.assertEqual(config["max_gamma"], 5.0)


if __name__ == "__main__":
    unittest.main()
