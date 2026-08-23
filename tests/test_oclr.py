from pathlib import Path
import unittest

from model.innovations.train_clre import load_config


ROOT = Path(__file__).resolve().parents[1]


class OCLRTest(unittest.TestCase):
    def test_oclr_identity_and_threshold(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-029_oclr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.oclr.v1")
        self.assertAlmostEqual(config["comparison_H"], 77.82913952565472)
        source = (ROOT / "model/innovations/train_clre.py").read_text(encoding="utf-8")
        self.assertIn("normalized_residual", source)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
