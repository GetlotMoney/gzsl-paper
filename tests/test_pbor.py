from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model.candidates.v2.modules.pbor import PartialBiOrthogonalResidual
from model.candidates.v2.trainers.train_pbor import load_config


ROOT = Path(__file__).resolve().parents[1]


class PBORTest(unittest.TestCase):
    def test_zero_projection_matches_oclr_formula(self):
        generator = torch.Generator().manual_seed(661)
        claude = torch.randn(200, 768, generator=generator)
        names = torch.randn(200, 768, generator=generator)
        parent_proto = torch.randn(200, 768, generator=generator)
        model = PartialBiOrthogonalResidual(claude, names, parent_proto, 3.0, 1.0)
        expected = F.normalize(
            F.normalize(claude, dim=-1)
            - (
                F.normalize(claude, dim=-1) * F.normalize(names, dim=-1)
            ).sum(-1, keepdim=True) * F.normalize(names, dim=-1),
            dim=-1,
        )
        self.assertTrue(torch.allclose(model.prototypes(), expected, atol=1e-6))
        model.prototypes().sum().backward()
        self.assertIsNotNone(model.raw_parent_projection.grad)

    def test_pbor_config_binds_oclr(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-033_pbor/configs/RUN-001.yaml"
        )
        self.assertEqual(
            config["oclr_model_sha256"],
            "f27b5c10fddb570a0aa78aca64e61683ada35eb81ca2a7cd52580b6c36f6a19c",
        )
        self.assertEqual(config["max_parent_projection"], 1.0)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
