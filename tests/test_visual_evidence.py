from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from model.paper_v2 import PaperV2ThreeModuleModel
from model.train_paper_v2 import _active_groups, _load_patch_batch, load_config
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

    def test_ten_prerun_configs_cover_five_modes_and_two_strategies(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        files = sorted(
            path
            for path in root.glob("v2_try_1*.yaml")
            if 148 <= int(path.name.split("_")[2]) <= 157
        )
        self.assertEqual(len(files), 10)
        configs = [load_config(path)[0] for path in files]
        self.assertEqual({value["visual_mode"] for value in configs}, VISUAL_MODES)
        self.assertEqual(
            {value["training_strategy"] for value in configs},
            {"end_to_end_joint", "stagewise_50_100_50"},
        )
        self.assertTrue(all(value["topology_weight"] == 0.1 for value in configs))
        confusion = [
            value for value in configs if value["visual_mode"] == "confusion_local_refiner"
        ]
        self.assertEqual({value["visual_hard_weight"] for value in confusion}, {0.1})
        enabled = [value for value in configs if value["visual_mode"] != "off"]
        self.assertTrue(all(value["patch_cache_mode"] == "gpu_fp16" for value in enabled))

    def test_runtime_off_configs_skip_patch_cache(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        files = sorted(root.glob("v2_try_15[89]_off-runtime-*.yaml"))
        self.assertEqual(len(files), 2)
        configs = [load_config(path)[0] for path in files]
        self.assertTrue(all(value["schema_version"].endswith("visual-run.v2") for value in configs))
        self.assertTrue(all(value["visual_mode"] == "off" for value in configs))
        self.assertTrue(all(value["patch_cache_mode"] == "none" for value in configs))

    def test_cached_half_patch_batch_is_selected_and_promoted_to_float(self):
        cached = torch.arange(4 * 3 * 2, dtype=torch.float16).reshape(4, 3, 2)
        actual = _load_patch_batch(cached, torch.tensor([3, 1]), torch.device("cpu"))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, cached[[3, 1]].float()))

    def test_multiscale_pool_is_exact_block_mean(self):
        model = self.build("multiscale_part_tokens")
        grid = torch.arange(24 * 24, dtype=torch.float32).reshape(1, 1, 24, 24)
        pooled = model.visual._deterministic_grid_pool(grid, 12)
        expected = grid.reshape(1, 1, 12, 2, 12, 2).mean(dim=(3, 5))
        self.assertTrue(torch.equal(pooled, expected))

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
            self.assertGreaterEqual(float(components["anchor_loss"].detach()), 0.0)
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
