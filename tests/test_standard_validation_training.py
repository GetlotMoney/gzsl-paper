from pathlib import Path
import unittest

import torch

from model.innovations.train_standard_validation import (
    evaluate_validation,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


class _IdentityPrototypeModel:
    def __init__(self):
        self._prototypes = torch.zeros(200, 768)
        self._prototypes[:4, :4] = torch.eye(4)

    def eval(self):
        return self

    def prototypes(self):
        return self._prototypes

    def scale(self):
        return torch.tensor(10.0)


class StandardValidationTrainingTest(unittest.TestCase):
    def test_both_configs_exclude_official_inputs(self):
        root = ROOT / "experiments/v2/tune/TUNE-001_standard_clip_validation/configs"
        no_expert, _ = load_config(root / "RUN-001.yaml")
        expert, _ = load_config(root / "RUN-002.yaml")
        self.assertFalse(no_expert["expert_attributes_used"])
        self.assertTrue(expert["expert_attributes_used"])
        for config in (no_expert, expert):
            self.assertFalse(config["official_test_loaded"])
            self.assertTrue(config["validation_used_for_selection"])
            self.assertFalse(config["test_used_for_selection"])
            self.assertNotIn("seen_features", config["inputs"])
            self.assertNotIn("unseen_features", config["inputs"])

    def test_validation_metrics_use_joint_active_competition(self):
        features = torch.zeros(4, 768)
        features[:, :4] = torch.eye(4)
        labels = torch.tensor([0, 1, 2, 3])
        split = {
            "dev_seen_classes": torch.tensor([0, 1]),
            "dev_unseen_classes": torch.tensor([2, 3]),
            "val_seen_positions": torch.tensor([0, 1]),
            "val_unseen_positions": torch.tensor([2, 3]),
        }
        metrics = evaluate_validation(
            _IdentityPrototypeModel(), features, labels, split, torch.device("cpu")
        )
        self.assertEqual(metrics, {"U": 100.0, "S": 100.0, "H": 100.0, "ZS": 100.0})

    def test_training_source_has_no_official_cache_names(self):
        source = (ROOT / "model/innovations/train_standard_validation.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("CUB_test_seen", source)
        self.assertNotIn("CUB_test_unseen", source)
        self.assertNotIn("test_seen_loc", source)
        self.assertNotIn("test_unseen_loc", source)


if __name__ == "__main__":
    unittest.main()
