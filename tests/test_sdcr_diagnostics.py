import unittest

import torch

from tools.diagnose_sdcr_errors import summarize_predictions


class SDCRDiagnosticsTest(unittest.TestCase):
    def test_summary_tracks_domain_crossing_and_confusion(self):
        labels = torch.tensor([0, 0, 2, 2])
        predictions = torch.tensor([0, 2, 1, 2])
        logits = torch.tensor(
            [
                [3.0, 1.0, 0.0],
                [0.5, 0.1, 1.0],
                [0.2, 1.2, 0.5],
                [0.1, 0.2, 2.0],
            ]
        )
        names = ["seen-0", "seen-1", "unseen-2"]
        seen_mask = torch.tensor([True, True, False])
        summary = summarize_predictions(
            labels, predictions, logits, names, seen_mask, top_confusions=5
        )
        self.assertEqual(summary["wrong_count"], 2)
        self.assertAlmostEqual(
            summary["true_seen_predicted_unseen_rate_percent"], 25.0
        )
        self.assertAlmostEqual(
            summary["true_unseen_predicted_seen_rate_percent"], 25.0
        )
        self.assertEqual(summary["top_confusions"][0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
