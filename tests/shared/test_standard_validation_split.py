from pathlib import Path
import unittest

import torch

from tools.prepare_cub_standard_validation import (
    _per_class_fit_and_seen_validation,
)


ROOT = Path(__file__).resolve().parents[2]


class StandardValidationSplitTest(unittest.TestCase):
    def test_per_class_holdout_is_deterministic_and_disjoint(self):
        labels = torch.tensor([0] * 10 + [1] * 15)
        indices = torch.arange(labels.numel())
        classes = torch.tensor([0, 1])
        first = _per_class_fit_and_seen_validation(indices, labels, classes)
        second = _per_class_fit_and_seen_validation(indices, labels, classes)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertFalse(torch.isin(first[0], first[1]).any())
        self.assertEqual(first[0].numel() + first[1].numel(), 25)
        self.assertEqual(first[1].numel(), 5)

    def test_split_builder_does_not_reference_official_test(self):
        source = (
            ROOT / "tools/prepare_cub_standard_validation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("test_seen_loc", source)
        self.assertNotIn("test_unseen_loc", source)
        self.assertNotIn("CUB_test_seen", source)
        self.assertNotIn("CUB_test_unseen", source)


if __name__ == "__main__":
    unittest.main()
