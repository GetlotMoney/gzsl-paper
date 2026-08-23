import unittest

import torch

from tools.diagnose_agct_sources import (
    apply_hard_tie_break,
    transition_stats,
)


class AGCTSourceOracleTest(unittest.TestCase):
    def test_tie_break_and_transition_counts(self):
        baseline = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 1, 1, 0])
        top_ids = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
        scores = torch.tensor([[2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0]])
        gate = torch.tensor([True, True, True, False])
        candidate = apply_hard_tie_break(
            baseline, top_ids, scores, gate, choose_max=True
        )
        stats = transition_stats(labels, baseline, candidate, gate, top_ids)
        self.assertEqual(stats["corrected_count"], 1)
        self.assertEqual(stats["broken_count"], 0)
        self.assertEqual(stats["net_correct_count"], 1)


if __name__ == "__main__":
    unittest.main()
