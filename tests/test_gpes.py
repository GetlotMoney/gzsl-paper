from pathlib import Path
import unittest

import torch

from model.innovations.gpes import GatedPairEvidenceSelector
from model.innovations.train_gpes import extract_pair_examples, load_config


ROOT = Path(__file__).resolve().parents[1]


class GPESTest(unittest.TestCase):
    def _model(self):
        groups = torch.arange(200) // 2
        return GatedPairEvidenceSelector(
            torch.randn(200, 768),
            13.0,
            torch.randn(200, 768),
            torch.randn(200, 768),
            groups,
            0.25,
            0.1,
            torch.zeros(4),
            torch.ones(4),
            0.5,
        )

    def test_zero_selector_reproduces_pair_and_full_parent(self):
        model = self._model()
        pair = torch.tensor([[1.0, 0.9], [0.7, 0.6]])
        features = torch.randn(2, 4)
        self.assertTrue(torch.equal(model.corrected_pair_logits(pair, features), pair))
        images = torch.randn(2, 768)
        parent = torch.randn(2, 200)
        patch = torch.randn(2, 200)
        expected = parent + model.sdcr_beta * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.sdcr_prototypes.T
        )
        self.assertTrue(torch.equal(model(parent, images, patch), expected))

    def test_pair_correction_preserves_pair_mean_and_has_gradients(self):
        model = self._model()
        with torch.no_grad():
            model.selector_weight[1] = 0.5
        pair = torch.tensor([[1.0, 0.9], [0.7, 0.6]])
        features = torch.randn(2, 4)
        corrected = model.corrected_pair_logits(pair, features)
        self.assertTrue(
            torch.allclose(pair.mean(dim=1), corrected.mean(dim=1), atol=1e-6)
        )
        corrected.sum().backward()
        self.assertIsNotNone(model.selector_weight.grad)

    def test_extract_pair_examples_keeps_only_gated_top2_targets(self):
        logits = torch.tensor(
            [[1.0, 0.9, 0.0], [1.0, 0.9, 0.0]], requires_grad=True
        )
        images = torch.randn(2, 4)
        patch = torch.randn(2, 3)
        targets = torch.tensor([1, 2])
        ids = torch.tensor([0, 1, 2])
        groups = torch.tensor([0, 0, 1])
        claude = torch.randn(3, 4)
        merge = torch.randn(3, 4)
        package = extract_pair_examples(
            logits, images, patch, targets, ids, groups,
            claude, merge, threshold=0.2
        )
        self.assertEqual(package[3], 1)
        self.assertEqual(int(package[2][0]), 1)
        self.assertFalse(package[0].requires_grad)
        self.assertFalse(package[1].requires_grad)

    def test_config_binds_pair_ce_and_four_features(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-062_gpes/configs/RUN-001.yaml"
        )
        self.assertEqual(config["max_delta"], 0.5)
        self.assertEqual(config["threshold_quantile"], 0.25)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
