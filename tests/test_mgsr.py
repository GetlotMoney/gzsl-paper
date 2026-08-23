from pathlib import Path
import unittest

import torch

from model.innovations.mgsr import MultiGeometrySentenceRouting
from model.innovations.train_mgsr import load_config


ROOT = Path(__file__).resolve().parents[1]


class MGSRTest(unittest.TestCase):
    def _model(self):
        generator = torch.Generator().manual_seed(761)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        parents = torch.randn(200, 768, generator=generator)
        seen = torch.arange(150)
        base = torch.softmax(torch.randn(8, generator=generator), dim=0)
        return MultiGeometrySentenceRouting(
            sentences, names, parents, seen, base, 13.0, 0.25
        ), base

    def test_zero_initialization_reproduces_global_parent(self):
        model, base = self._model()
        weights = model.class_sentence_weights()
        self.assertTrue(torch.allclose(weights, base.unsqueeze(0).expand_as(weights)))
        self.assertEqual(model.routing_stats()["class_variation"], 0.0)

    def test_shared_geometry_coefficients_create_class_variation(self):
        model, _ = self._model()
        with torch.no_grad():
            model.raw_geometry_coefficients[0] = 0.5
        self.assertGreater(model.routing_stats()["class_variation"], 0.0)

    def test_config_binds_sdcr_parent_and_seen_only_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-045_mgsr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.mgsr.v1")
        self.assertEqual(
            config["sdcr_model_sha256"],
            "53f9065ddd5f32bc02ff4be3ce5db3c7a4eadf5117282b55a672780acec001ae",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_rescue_config_only_tightens_residual_limit(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-045_mgsr/configs/RUN-002.yaml"
        )
        self.assertEqual(config["max_logit_residual"], 0.1)
        self.assertEqual(config["random_seed"], 5)


if __name__ == "__main__":
    unittest.main()
