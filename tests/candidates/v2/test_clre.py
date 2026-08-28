from pathlib import Path
import unittest

from model.candidates.v2.trainers.train_clre import load_config


ROOT = Path(__file__).resolve().parents[3]


class CLRETest(unittest.TestCase):
    def test_clre_binds_independent_text_cache_and_no_expert(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-024_clre/configs/RUN-001.yaml"
        )
        self.assertEqual(
            config["claude_embeddings_sha256"],
            "c549c0c0cc437c05aa42ade516ef8488ded698752e98f1f7888d6a540563a043",
        )
        self.assertFalse(config["unseen_images_used_for_gradient"])
        self.assertFalse(config["text_cache_provenance_complete"])
        source = (ROOT / "model/candidates/v2/trainers/train_clre.py").read_text(encoding="utf-8")
        self.assertNotIn('["att"]', source)


if __name__ == "__main__":
    unittest.main()
