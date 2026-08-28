from pathlib import Path
import unittest

from model.candidates.v2.trainers.train_clec import load_config


ROOT = Path(__file__).resolve().parents[1]


class OGLCTest(unittest.TestCase):
    def test_oglc_binds_oclr_and_ccpe(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-030_oglc/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.oglc.v1")
        self.assertEqual(
            config["oclr_model_sha256"],
            "f27b5c10fddb570a0aa78aca64e61683ada35eb81ca2a7cd52580b6c36f6a19c",
        )
        self.assertEqual(
            config["ccpe_model_sha256"],
            "e3b2685b07883b976962c38804825e4043c500679003a869b4bc6997f60cfaf9",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
