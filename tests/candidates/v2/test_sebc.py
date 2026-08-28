from pathlib import Path
import unittest

from model.candidates.v2.trainers.train_sebc import load_config


ROOT = Path(__file__).resolve().parents[3]


class SEBCTest(unittest.TestCase):
    def test_config_binds_class_exclusive_folds_and_no_expert_parent(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-013_sebc/configs/RUN-001.yaml"
        )
        self.assertEqual(set(config["fold_model_sha256"]), {"0", "1", "2"})
        self.assertFalse(config["unseen_images_used_for_gradient"])
        source = (ROOT / "model/candidates/v2/trainers/train_sebc.py").read_text(encoding="utf-8")
        self.assertNotIn('["att"]', source)
        self.assertIn("balanced_fold_batch", source)

    def test_conservative_gamma_rescue_is_accepted(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-013_sebc/configs/RUN-002.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.sebc.v2")
        self.assertEqual(config["max_gamma"], 0.2)


if __name__ == "__main__":
    unittest.main()
