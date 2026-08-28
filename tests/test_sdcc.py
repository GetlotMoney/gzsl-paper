from pathlib import Path
import unittest

from model.candidates.v2.trainers.train_sdcr import load_config


ROOT = Path(__file__).resolve().parents[1]


class SDCCTest(unittest.TestCase):
    def test_sdcc_config_has_consistency_distillation(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-042_sdcc/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.sdcc.v1")
        self.assertEqual(config["consistency_weight"], 0.1)
        self.assertEqual(config["distill_temperature"], 1.0)
        self.assertFalse(config["unseen_images_used_for_gradient"])
        source = (ROOT / "model/candidates/v2/trainers/train_sdcr.py").read_text(encoding="utf-8")
        self.assertIn("teacher_logits", source)
        self.assertIn("consistency_loss", source)


if __name__ == "__main__":
    unittest.main()
