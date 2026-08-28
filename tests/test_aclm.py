from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.aclm import AdaptiveCrossLLMMixture
from model.candidates.v2.trainers.train_aclm import load_config


ROOT = Path(__file__).resolve().parents[1]


class ACLMTest(unittest.TestCase):
    def test_initial_mixture_is_balanced_and_trainable(self):
        generator = torch.Generator().manual_seed(619)
        claude = torch.randn(200, 768, generator=generator)
        merge = torch.randn(200, 768, generator=generator)
        model = AdaptiveCrossLLMMixture(claude, merge, 19.0, 16.0)
        self.assertEqual(float(model.claude_weight().detach()), 0.5)
        parent = torch.randn(2, 7, generator=generator)
        images = torch.randn(2, 768, generator=generator)
        model(parent, images, torch.arange(7)).sum().backward()
        self.assertIsNotNone(model.raw_mix.grad)

    def test_aclm_binds_clre_and_mlre_parents(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-027_aclm/configs/RUN-001.yaml"
        )
        self.assertEqual(
            config["clre_model_sha256"],
            "03db81c9e42080eba45f788087ad7c3845ee0c0128135b2fbc7a91a1d2cf8538",
        )
        self.assertEqual(
            config["mlre_model_sha256"],
            "2b2bc27b73bf57732cb9d8748efa5b2c267f8a4ef0ae559eb32d7cc706e5cddf",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
