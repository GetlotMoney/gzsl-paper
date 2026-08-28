from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.clec import CrossLLMLocalEvidenceComposition
from model.candidates.v2.trainers.train_clec import load_config


ROOT = Path(__file__).resolve().parents[3]


class CLECTest(unittest.TestCase):
    def test_initial_scale_is_one_and_only_scale_has_gradient(self):
        generator = torch.Generator().manual_seed(601)
        claude = torch.randn(200, 768, generator=generator)
        model = CrossLLMLocalEvidenceComposition(claude, 19.0, 9.0, 0.25)
        parent = torch.randn(2, 7, generator=generator)
        images = torch.randn(2, 768, generator=generator)
        patch = torch.randn(2, 200, generator=generator)
        ids = torch.arange(7)
        self.assertEqual(float(model.patch_scale().detach()), 1.0)
        model(parent, images, patch, ids).sum().backward()
        self.assertIsNotNone(model.raw_patch_scale.grad)

    def test_clec_binds_two_supported_parents(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-025_clec/configs/RUN-001.yaml"
        )
        self.assertEqual(
            config["ccpe_model_sha256"],
            "e3b2685b07883b976962c38804825e4043c500679003a869b4bc6997f60cfaf9",
        )
        self.assertEqual(
            config["clre_model_sha256"],
            "03db81c9e42080eba45f788087ad7c3845ee0c0128135b2fbc7a91a1d2cf8538",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
