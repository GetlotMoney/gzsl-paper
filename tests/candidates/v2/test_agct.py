from pathlib import Path
import unittest

import torch

from model.candidates.v2.modules.agct import (
    AmbiguityGatedCrossLLMTieBreaker,
    MultiSourceAmbiguityGatedTieBreaker,
)
from model.candidates.v2.trainers.train_agct import load_config, select_margin_threshold


ROOT = Path(__file__).resolve().parents[3]


class AGCTTest(unittest.TestCase):
    def _model(self):
        generator = torch.Generator().manual_seed(877)
        groups = torch.full((200,), -1, dtype=torch.long)
        groups[:4] = torch.tensor([0, 0, 1, 1])
        return AmbiguityGatedCrossLLMTieBreaker(
            torch.randn(200, 768, generator=generator),
            13.0,
            torch.randn(200, 768, generator=generator),
            groups,
            0.5,
            0.1,
            5.0,
        )

    def test_gate_only_activates_for_low_margin_same_group_top2(self):
        model = self._model()
        logits = torch.full((3, 200), -10.0)
        logits[0, 0], logits[0, 1] = 1.0, 0.8
        logits[1, 0], logits[1, 2] = 1.0, 0.8
        logits[2, 0], logits[2, 1] = 2.0, 0.0
        gate, same, _ = model.gate_values(logits)
        self.assertTrue(bool(same[0]))
        self.assertFalse(bool(same[1]))
        self.assertGreater(float(gate[0]), 0.5)
        self.assertEqual(float(gate[1]), 0.0)
        self.assertLess(float(gate[2]), 1e-5)

    def test_zero_beta_reproduces_sdcr_parent(self):
        model = self._model()
        images = torch.randn(2, 768)
        parent = torch.randn(2, 200)
        expected = parent + model.sdcr_beta * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.sdcr_prototypes.T
        )
        self.assertTrue(torch.equal(model(parent, images), expected))

    def test_threshold_uses_wrong_same_group_median(self):
        margins = torch.tensor([0.1, 0.2, 0.4, 0.8])
        same = torch.tensor([True, True, True, False])
        wrong = torch.tensor([True, False, True, True])
        threshold, stats = select_margin_threshold(margins, same, wrong, 0.5)
        self.assertAlmostEqual(threshold, 0.25)
        self.assertEqual(stats["candidate_count"], 2)
        self.assertEqual(stats["source"], "train_wrong_same_group")

    def test_config_binds_train_only_threshold_and_claude(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-058_agct/configs/RUN-001.yaml"
        )
        self.assertEqual(config["threshold_source"], "train_wrong_same_group_margin")
        self.assertEqual(config["margin_temperature"], 0.1)
        self.assertTrue(config["claude_embeddings_sha256"])
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_consensus_gate_requires_claude_to_agree_with_sdcr_top1(self):
        generator = torch.Generator().manual_seed(881)
        groups = torch.full((200,), -1, dtype=torch.long)
        groups[:2] = 0
        sdcr = torch.randn(200, 768, generator=generator)
        claude = torch.randn(200, 768, generator=generator)
        image = torch.zeros(1, 768)
        image[0, 0] = 1.0
        claude[0].zero_(); claude[0, 0] = 1.0
        claude[1].zero_(); claude[1, 0] = -1.0
        model = AmbiguityGatedCrossLLMTieBreaker(
            sdcr, 13.0, claude, groups, 0.5, 0.1, 5.0, consensus_only=True
        )
        logits = torch.full((1, 200), -10.0)
        logits[0, 0], logits[0, 1] = 1.0, 0.8
        gate, _, _ = model.gate_values(logits, images=image)
        self.assertGreater(float(gate[0]), 0.5)
        model.claude_prototypes[[0, 1]] = model.claude_prototypes[[1, 0]]
        gate, _, _ = model.gate_values(logits, images=image)
        self.assertEqual(float(gate[0]), 0.0)

    def test_cctb_config_binds_consensus_only_gate(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-059_cctb/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.cctb.v1")
        self.assertTrue(config["consensus_only"])

    def test_agct_coverage_rescue_uses_75th_percentile(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-058_agct/configs/RUN-003.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.agct.v1")
        self.assertEqual(config["threshold_quantile"], 0.75)
        self.assertEqual(config["random_seed"], 5)

    def test_agct_precision_rescue_uses_25th_percentile(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-058_agct/configs/RUN-004.yaml"
        )
        self.assertEqual(config["threshold_quantile"], 0.25)
        self.assertEqual(config["random_seed"], 5)

    def test_agct_final_rescue_uses_sharper_gate(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-058_agct/configs/RUN-006.yaml"
        )
        self.assertEqual(config["threshold_quantile"], 0.25)
        self.assertEqual(config["margin_temperature"], 0.05)

    def test_magt_has_two_trainable_source_betas_and_exact_parent(self):
        generator = torch.Generator().manual_seed(887)
        groups = torch.arange(200) // 2
        model = MultiSourceAmbiguityGatedTieBreaker(
            torch.randn(200, 768, generator=generator),
            13.0,
            torch.randn(2, 200, 768, generator=generator),
            groups,
            0.25,
            0.1,
            5.0,
        )
        images = torch.randn(2, 768, generator=generator)
        parent = torch.randn(2, 200, generator=generator)
        baseline = model(parent, images)
        expected = parent + model.sdcr_beta * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.sdcr_prototypes.T
        )
        self.assertTrue(torch.equal(baseline, expected))
        baseline.sum().backward()
        self.assertEqual(model.raw_betas.numel(), 2)
        self.assertIsNotNone(model.raw_betas.grad)

    def test_magt_config_binds_merge_source(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-060_magt/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.magt.v1")
        self.assertTrue(config["merge_embeddings_sha256"])
        self.assertTrue(config["omlr_model_sha256"])


if __name__ == "__main__":
    unittest.main()
