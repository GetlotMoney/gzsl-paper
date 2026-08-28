from pathlib import Path, PurePosixPath
import unittest

import torch

from model.frameworks.v4.tg import fixed_class_folds
from model.candidates.v2.trainers.train_chen_class_exclusive import (
    balanced_fold_batch,
    load_config,
)


ROOT = Path(__file__).resolve().parents[3]


class ChenClassExclusiveTest(unittest.TestCase):
    def test_config_requires_true_fold_parents(self):
        config, _ = load_config(
            ROOT / "experiments/v2/confirmation/CONFIRM-007_chen_class_exclusive/configs/RUN-001.yaml"
        )
        self.assertEqual(config["fold_count"], 3)
        self.assertEqual(config["fold_tg_epochs"], 50)
        self.assertEqual(config["max_transport_step"], 0.5)
        self.assertFalse(config["nested_official_test_selection"])
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_folds_are_disjoint_and_cover_150_seen_classes(self):
        seen = torch.arange(150)
        folds = fixed_class_folds(seen)
        heldout = []
        for pseudo_seen, pseudo_unseen in folds:
            self.assertEqual(pseudo_seen.numel(), 100)
            self.assertEqual(pseudo_unseen.numel(), 50)
            self.assertFalse(torch.isin(pseudo_seen, pseudo_unseen).any())
            heldout.append(pseudo_unseen)
        joined = torch.cat(heldout)
        self.assertTrue(torch.equal(joined.sort().values, seen))

    def test_balanced_batch_has_25_seen_and_25_class_exclusive_samples(self):
        labels = torch.arange(150).repeat_interleave(4)
        pseudo_seen, pseudo_unseen = fixed_class_folds(torch.arange(150))[0]
        indices = balanced_fold_batch(
            labels, pseudo_seen, pseudo_unseen, 25, torch.Generator().manual_seed(7)
        )
        self.assertEqual(indices.numel(), 50)
        self.assertTrue(torch.isin(labels[indices[:25]], pseudo_seen).all())
        self.assertTrue(torch.isin(labels[indices[25:]], pseudo_unseen).all())

    def test_fold_training_source_never_evaluates_fold_on_official_test(self):
        source = (ROOT / "model/candidates/v2/trainers/train_chen_class_exclusive.py").read_text(encoding="utf-8")
        fold_start = source.index("for fold_id")
        fold_end = source.index("def evaluate_and_track")
        fold_section = source[fold_start:fold_end]
        self.assertNotIn("evaluate", fold_section)
        self.assertIn("prototype_stages_from_tg", source)

    def test_reuse_config_binds_parent_artifacts_and_only_changes_transport_cap(self):
        root = ROOT / "experiments/v2/confirmation/CONFIRM-007_chen_class_exclusive/configs"
        parent, _ = load_config(root / "RUN-001.yaml")
        rescue, _ = load_config(root / "RUN-002.yaml")
        self.assertEqual(parent["max_transport_step"], 0.5)
        self.assertEqual(rescue["max_transport_step"], 1.5)
        self.assertTrue(PurePosixPath(rescue["reuse_parent_dir"]).is_absolute())
        self.assertEqual(set(rescue["fold_model_sha256"]), {"0", "1", "2"})
        self.assertIsNone(parent["reuse_parent_dir"])


if __name__ == "__main__":
    unittest.main()
