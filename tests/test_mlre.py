from pathlib import Path
import unittest

from model.innovations.train_clre import load_config


ROOT = Path(__file__).resolve().parents[1]


class MLRETest(unittest.TestCase):
    def test_mlre_binds_merge_cache_and_clre_threshold(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-026_mlre/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.mlre.v1")
        self.assertEqual(
            config["merge_embeddings_sha256"],
            "304ad6834cb40bd220a066764a3bf39279c072679a83d5aef74012b2b9cfb9f0",
        )
        self.assertAlmostEqual(config["comparison_H"], 77.80809298394227)
        self.assertFalse(config["unseen_images_used_for_gradient"])


if __name__ == "__main__":
    unittest.main()
