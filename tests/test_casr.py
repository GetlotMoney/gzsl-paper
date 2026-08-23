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

    def test_rescue_uses_weaker_kl(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-038_casr/configs/RUN-002.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.casr.v2")
        self.assertEqual(config["kl_weight"], 0.01)

    def test_seed5_reliability_uses_seed5_parent(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-038_casr/configs/RUN-003.yaml"
        )
        self.assertEqual(config["random_seed"], 5)
        self.assertEqual(
            config["oesr_model_sha256"],
            "74bd92c84278c4f623e2ae357358a34bc07a810714b1f03c236065fc77a9a8e1",
        )


if __name__ == "__main__":
    unittest.main()
