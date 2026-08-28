from pathlib import Path
import unittest

from model.candidates.v2.trainers.train_clre import load_config


ROOT = Path(__file__).resolve().parents[1]


class OESRTest(unittest.TestCase):
    def test_oesr_identity_cache_and_threshold(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-036_oesr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.oesr.v1")
        self.assertEqual(
            config["eight_sentence_embeddings_sha256"],
            "8c1a8e27a70681759b22e87412c424b6c9c3a7991ed391b3acc244bbc3a6bca3",
        )
        self.assertAlmostEqual(config["comparison_H"], 78.0721851209539)
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_seed7_reliability_config(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-036_oesr/configs/RUN-002.yaml"
        )
        self.assertEqual(config["random_seed"], 7)


if __name__ == "__main__":
    unittest.main()
