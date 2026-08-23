from pathlib import Path
import unittest

from model.innovations.train_sebc import load_config


ROOT = Path(__file__).resolve().parents[1]


class SEBCTest(unittest.TestCase):
    def test_config_binds_class_exclusive_folds_and_no_expert_parent(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-013_sebc/configs/RUN-001.yaml"
        )
        self.assertEqual(set(config["fold_model_sha256"]), {"0", "1", "2"})
        self.assertFalse(config["unseen_images_used_for_gradient"])
        source = (ROOT / "model/innovations/train_sebc.py").read_text(encoding="utf-8")
        self.assertNotIn('["att"]', source)
        self.assertIn("balanced_fold_batch", source)


if __name__ == "__main__":
    unittest.main()
