from pathlib import Path
import unittest

from model.innovations.train_clre import load_config


ROOT = Path(__file__).resolve().parents[1]


class OMLRTest(unittest.TestCase):
    def test_omlr_identity_cache_and_threshold(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-031_omlr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.omlr.v1")
        self.assertEqual(
            config["merge_embeddings_sha256"],
            "304ad6834cb40bd220a066764a3bf39279c072679a83d5aef74012b2b9cfb9f0",
        )
        self.assertAlmostEqual(config["comparison_H"], 78.0721851209539)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
