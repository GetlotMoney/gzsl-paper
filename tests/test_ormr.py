from pathlib import Path
import unittest

from model.candidates.v2.trainers.train_clre import load_config


ROOT = Path(__file__).resolve().parents[1]


class ORMRTest(unittest.TestCase):
    def test_ormr_identity_cache_and_threshold(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-035_ormr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.ormr.v1")
        self.assertEqual(
            config["rolematched_sentence_embeddings_sha256"],
            "71a6ccecff967a5a362b1bd49a1e821b3374c635bceb9287c81a80449011c1b3",
        )
        self.assertAlmostEqual(config["comparison_H"], 78.0721851209539)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
