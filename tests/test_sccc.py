import unittest

import torch

from model.innovations.sccc import (
    SampleConditionedCompetitionCalibration,
    competition_confidence_features,
)


class SCCCTest(unittest.TestCase):
    def test_features_and_zero_initialization(self):
        logits = torch.randn(8, 20, generator=torch.Generator().manual_seed(601))
        mask = torch.zeros(20, dtype=torch.bool)
        mask[:12] = True
        self.assertEqual(tuple(competition_confidence_features(logits, mask).shape), (8, 6))
        model = SampleConditionedCompetitionCalibration()
        self.assertTrue(torch.equal(model(logits, mask), logits))
        self.assertEqual(float(model.gamma(logits, mask).abs().max()), 0.0)

    def test_gate_receives_gradient_and_is_bounded(self):
        logits = torch.randn(10, 30, generator=torch.Generator().manual_seed(602))
        mask = torch.zeros(30, dtype=torch.bool)
        mask[:20] = True
        model = SampleConditionedCompetitionCalibration(max_gamma=2.0)
        adjusted = model(logits, mask)
        adjusted.square().mean().backward()
        self.assertIsNotNone(model.network[-1].weight.grad)
        self.assertGreater(float(model.network[-1].weight.grad.abs().sum()), 0.0)
        with torch.no_grad():
            model.network[-1].bias.fill_(100.0)
        self.assertLessEqual(float(model.gamma(logits, mask).max()), 2.0)

    def test_invalid_partition_is_rejected(self):
        logits = torch.randn(2, 5)
        with self.assertRaises(ValueError):
            competition_confidence_features(logits, torch.ones(5, dtype=torch.bool))

    def test_nonnegative_mode_starts_exactly_zero_and_keeps_gradient(self):
        logits = torch.randn(8, 20, generator=torch.Generator().manual_seed(603))
        mask = torch.zeros(20, dtype=torch.bool); mask[:12] = True
        model = SampleConditionedCompetitionCalibration(max_gamma=0.5, gamma_mode="nonnegative")
        value = model.gamma(logits, mask)
        self.assertEqual(float(value.detach().abs().max()), 0.0)
        (-model(logits, mask)[:, :12].mean()).backward()
        self.assertGreater(float(model.network[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreaterEqual(float(model.gamma(logits, mask).min()), 0.0)


if __name__ == "__main__":
    unittest.main()
