from pathlib import Path
import unittest

import torch

from model.candidates.v2.trainers.train_chen_style import load_config, random_batch_indices


ROOT = Path(__file__).resolve().parents[3]


class ChenStyleTrainingTest(unittest.TestCase):
    def test_configs_match_public_code_training_contract(self):
        root = ROOT / "experiments/v2/confirmation/CONFIRM-004_chen_style_end_to_end/configs"
        no_expert, _ = load_config(root / "RUN-001.yaml")
        expert, _ = load_config(root / "RUN-002.yaml")
        for config in (no_expert, expert):
            self.assertEqual(config["batch_size"], 50)
            self.assertEqual(config["epochs"], 200)
            self.assertEqual(config["niters"], 28228)
            self.assertEqual(config["report_interval"], 141)
            self.assertEqual(config["random_seed"], 5)
            self.assertEqual(config["optimizer"], "Adam")
            self.assertEqual(config["learning_rate"], 0.0001)
            self.assertTrue(config["test_used_for_selection"])
            self.assertFalse(config["unseen_images_used_for_gradient"])
            self.assertEqual(config["selection_scope"], "whole_model_only")
        self.assertFalse(no_expert["expert_attributes_used"])
        self.assertTrue(expert["expert_attributes_used"])

    def test_random_batch_is_unique_inside_batch_but_resampled_each_step(self):
        generator = torch.Generator().manual_seed(5)
        first = random_batch_indices(7057, 50, generator)
        second = random_batch_indices(7057, 50, generator)
        self.assertEqual(first.unique().numel(), 50)
        self.assertEqual(second.unique().numel(), 50)
        self.assertFalse(torch.equal(first, second))

    def test_exact_public_code_evaluation_count(self):
        points = [iteration for iteration in range(28228) if iteration % 141 == 0]
        self.assertEqual(len(points), 201)
        self.assertEqual(points[0], 0)
        self.assertEqual(points[-1], 28200)

    def test_source_selects_only_whole_model_h(self):
        source = (ROOT / "model/candidates/v2/trainers/train_chen_style.py").read_text(encoding="utf-8")
        self.assertIn('if metrics["H"] > best_h:', source)
        self.assertNotIn("best_tg_vpr", source)
        self.assertNotIn("best_ccgr", source)
        self.assertIn('"training_strategy": "end_to_end_joint"', source)


if __name__ == "__main__":
    unittest.main()
