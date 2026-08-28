from pathlib import Path
import unittest

from model.candidates.v2.trainers.train_ecpe import load_config


ROOT = Path(__file__).resolve().parents[1]


class ECPETest(unittest.TestCase):
    def test_ecpe_binds_fold_and_patch_evidence(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-021_ecpe/configs/RUN-001.yaml"
        )
        self.assertEqual(set(config["fold_model_sha256"]), {"0", "1", "2"})
        self.assertEqual(config["patch_top_k"], 2)
        self.assertEqual(config["epochs"], 20)
        self.assertFalse(config["unseen_images_used_for_gradient"])
        source = (ROOT / "model/candidates/v2/trainers/train_ecpe.py").read_text(encoding="utf-8")
        self.assertIn("balanced_fold_batch", source)
        self.assertNotIn('["att"]', source)


if __name__ == "__main__":
    unittest.main()
