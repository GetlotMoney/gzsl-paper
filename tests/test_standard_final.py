from pathlib import Path
import unittest

from model.innovations.train_standard_final import load_config


ROOT = Path(__file__).resolve().parents[1]


class StandardFinalTest(unittest.TestCase):
    def test_configs_bind_validation_selection(self):
        root = ROOT / "experiments/v2/confirmation/CONFIRM-003_standard_clip_final/configs"
        no_expert, _ = load_config(root / "RUN-001.yaml")
        expert, _ = load_config(root / "RUN-002.yaml")
        self.assertEqual(no_expert["epochs"], 24)
        self.assertEqual(no_expert["validation_selection"]["run_id"], "RUN-001")
        self.assertFalse(no_expert["expert_attributes_used"])
        self.assertEqual(expert["epochs"], 22)
        self.assertEqual(expert["validation_selection"]["run_id"], "RUN-006")
        self.assertTrue(expert["expert_attributes_used"])
        for config in (no_expert, expert):
            self.assertFalse(config["test_used_for_selection"])
            self.assertEqual(config["official_test_evaluations"], 1)
            self.assertFalse(config["strict_blind_claim_eligible"])

    def test_official_data_load_occurs_after_training_and_checkpoint(self):
        source = (ROOT / "model/innovations/train_standard_final.py").read_text(encoding="utf-8")
        loop = source.index("for epoch in range")
        checkpoint = source.index('atomic_torch_save(output_dir / "model_best.pth"')
        official = source.index("official test只在validation冻结")
        self.assertLess(loop, checkpoint)
        self.assertLess(checkpoint, official)


if __name__ == "__main__":
    unittest.main()
