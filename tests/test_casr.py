from pathlib import Path
import unittest

from model.innovations.train_aosr import load_config


ROOT = Path(__file__).resolve().parents[1]


class CASRTest(unittest.TestCase):
    def test_casr_config_has_conservative_kl(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-038_casr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.casr.v1")
        self.assertEqual(config["kl_weight"], 0.1)
        self.assertEqual(config["random_seed"], 7)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
