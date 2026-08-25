from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from model.paper_v2 import PaperV2ThreeModuleModel
from model.train_paper_v2 import _active_groups
from model.visual_evidence import PaperV2VisualModel, VISUAL_MODES


class VisualEvidenceContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        sentences = F.normalize(torch.randn(10, 8, 768), dim=-1)
        seen = torch.tensor([0, 1, 2, 3, 4, 5])
        centroids = F.normalize(torch.randn(6, 768), dim=-1)
        self.parent = PaperV2ThreeModuleModel(
            sentences,
            seen,
            centroids,
            dropout=0.0,
        )
        self.images = F.normalize(torch.randn(2, 768), dim=-1)
        self.patches = F.normalize(torch.randn(2, 576, 768), dim=-1)
        self.targets = torch.tensor([1, 4])

    def build(self, mode: str) -> PaperV2VisualModel:
        return PaperV2VisualModel(
            self.parent,
            visual_mode=mode,
            hidden_dim=16,
            max_beta=0.3,
            confusion_topk=5,
            visual_scales=(24, 12, 6),
        )

    def test_registered_modes_are_exact(self):
        self.assertEqual(
            VISUAL_MODES,
            {
                "off",
                "spatial_rgve",
                "semantic_part_tokens",
                "confusion_local_refiner",
                "multiscale_part_tokens",
            },
        )

    def test_off_and_enabled_initialization_are_exact_parent(self):
        expected = self.parent.logits(self.images)
        off = self.build("off")
        self.assertTrue(torch.equal(off.logits(self.images, self.patches), expected))
        self.assertEqual(off.parameter_groups()["visual"], [])
        for mode in VISUAL_MODES - {"off"}:
            model = self.build(mode)
            actual = model.logits(self.images, self.patches)
            self.assertTrue(torch.equal(actual, expected), mode)

    def test_all_enabled_modes_have_finite_shapes_and_active_stage_groups(self):
        for mode in VISUAL_MODES - {"off"}:
            model = self.build(mode)
            components = model.score_components(
                self.images,
                self.patches,
                target_class_ids=self.targets,
            )
            self.assertEqual(tuple(components["final_scores"].shape), (2, 10))
            self.assertEqual(tuple(components["part_scores"].shape), (2, 10, 3))
            self.assertTrue(torch.isfinite(components["final_scores"]).all(), mode)
            self.assertTrue(torch.isfinite(components["diversity_loss"]), mode)
            self.assertTrue(torch.isfinite(components["anchor_loss"]), mode)
            self.assertIn(
                "visual",
                _active_groups(model, "stagewise_50_100_50", "TRANSFER_CCGR"),
            )
            self.assertNotIn(
                "visual", _active_groups(model, "stagewise_50_100_50", "TG_ONLY")
            )

    def test_part_loss_trains_visual_branch_while_beta_is_zero(self):
        model = self.build("semantic_part_tokens")
        components = model.score_components(
            self.images,
            self.patches,
            target_class_ids=self.targets,
        )
        self.assertEqual(float(components["beta"].detach().abs().max()), 0.0)
        losses = model.visual_losses(
            components,
            self.parent.seen_classes,
            torch.tensor([1, 4]),
            self.targets,
            hard_margin=0.1,
        )
        losses["part"].backward()
        gradient = model.visual.adapter.up.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.norm()), 0.0)

    def test_confusion_inference_uses_only_model_topk(self):
        model = self.build("confusion_local_refiner")
        components = model.score_components(self.images, self.patches)
        expected = components["global_scores"].topk(5, dim=1).indices
        self.assertTrue(torch.equal(components["candidate_ids"], expected))
        self.assertIsNone(components.get("target_class_ids"))

    def test_confusion_training_inserts_target_and_has_hard_loss(self):
        model = self.build("confusion_local_refiner")
        components = model.score_components(
            self.images,
            self.patches,
            target_class_ids=self.targets,
        )
        self.assertTrue(
            components["candidate_ids"].eq(self.targets[:, None]).any(dim=1).all()
        )
        losses = model.visual_losses(
            components,
            self.parent.seen_classes,
            torch.tensor([1, 4]),
            self.targets,
            hard_margin=0.1,
        )
        self.assertTrue(torch.isfinite(losses["part"]))
        self.assertTrue(torch.isfinite(losses["hard"]))


if __name__ == "__main__":
    unittest.main()
