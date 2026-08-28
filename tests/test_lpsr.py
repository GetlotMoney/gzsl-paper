from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.lpsr import LocalPatchSemanticResidual, pool_fgvd_local_features
from model.candidates.v2.trainers.train_lpsr import load_config


ROOT = Path(__file__).resolve().parents[1]


class LPSRTest(unittest.TestCase):
    def test_zero_beta_exactly_returns_parent_and_has_gradient(self):
        generator = torch.Generator().manual_seed(487)
        sentences = torch.randn(200, 8, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        model = LocalPatchSemanticResidual(sentences, names, max_beta=10.0)
        parent = torch.randn(3, 7, generator=generator)
        local = torch.randn(3, 768, generator=generator)
        ids = torch.arange(7)
        self.assertTrue(torch.equal(model(parent, local, ids), parent))
        model(parent, local, ids).sum().backward()
        self.assertIsNotNone(model.raw_beta.grad)

    def test_pooling_contract(self):
        patches = torch.randn(2, 576, 768)
        pooled = pool_fgvd_local_features(patches, 64, torch.device("cpu"), 1)
        self.assertEqual(tuple(pooled.shape), (2, 768))
        self.assertTrue(torch.allclose(pooled.norm(dim=-1), torch.ones(2), atol=1e-6))

    def test_config_binds_real_patch_sha_and_no_expert(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-014_lpsr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["pool_top_k"], 64)
        self.assertFalse(config["unseen_images_used_for_gradient"])
        self.assertFalse(config["feature_provenance_complete"])
        source = (ROOT / "model/candidates/v2/trainers/train_lpsr.py").read_text(encoding="utf-8")
        self.assertNotIn('["att"]', source)


if __name__ == "__main__":
    unittest.main()
