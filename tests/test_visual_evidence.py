from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.paper_v2 import PaperV2ThreeModuleModel
from model.train_paper_v2 import (
    _active_groups,
    _load_patch_batch,
    best_handoff_stage_for_iteration,
    full_model_eligible_stages,
    load_config,
    modulewise_stage_for_iteration,
    restore_stage_best,
    sequential_stage_for_iteration,
    short_modulewise_stage_for_iteration,
    state_dict_sha256,
)
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

    def test_modulewise_boundaries_and_groups_are_isolated(self):
        model = self.build("spatial_rgve")
        ntrain = 100
        self.assertEqual(modulewise_stage_for_iteration(ntrain, 0)[0], "TG_ONLY")
        self.assertEqual(modulewise_stage_for_iteration(ntrain, 100)[0], "TST_NTR_ONLY")
        self.assertEqual(modulewise_stage_for_iteration(ntrain, 200)[0], "CCGR_ONLY")
        self.assertEqual(modulewise_stage_for_iteration(ntrain, 300)[0], "VISUAL_ONLY")
        self.assertEqual(
            _active_groups(model, "modulewise_50_50_50_50", "TG_ONLY"),
            ["tg_vpr"],
        )
        self.assertEqual(
            _active_groups(model, "modulewise_50_50_50_50", "TST_NTR_ONLY"),
            ["ntr", "transport"],
        )
        self.assertEqual(
            _active_groups(model, "modulewise_50_50_50_50", "CCGR_ONLY"),
            ["ccgr_class"],
        )
        self.assertEqual(
            _active_groups(model, "modulewise_50_50_50_50", "VISUAL_ONLY"),
            ["visual"],
        )
        off = self.build("off")
        self.assertEqual(
            _active_groups(off, "modulewise_50_50_50_50", "VISUAL_ONLY"), []
        )

    def test_short_modulewise_boundaries_add_joint_only_after_visual(self):
        ntrain = 100
        self.assertEqual(
            short_modulewise_stage_for_iteration(ntrain, 0, 5)[0], "TG_ONLY"
        )
        self.assertEqual(
            short_modulewise_stage_for_iteration(ntrain, 100, 5)[0], "TST_NTR_ONLY"
        )
        self.assertEqual(
            short_modulewise_stage_for_iteration(ntrain, 110, 5)[0], "CCGR_ONLY"
        )
        self.assertEqual(
            short_modulewise_stage_for_iteration(ntrain, 120, 5)[0], "VISUAL_ONLY"
        )
        self.assertEqual(
            short_modulewise_stage_for_iteration(ntrain, 130, 5)[0], "JOINT_FINETUNE"
        )
        self.assertEqual(
            short_modulewise_stage_for_iteration(ntrain, 139, 10)[0], "VISUAL_ONLY"
        )
        self.assertEqual(
            short_modulewise_stage_for_iteration(ntrain, 140, 10)[0], "JOINT_FINETUNE"
        )

    def test_balanced_sequential_boundaries_and_groups_are_exact(self):
        ntrain = 100
        epochs = {
            "tg_vpr": 50,
            "tst": 25,
            "ntr": 25,
            "ccgr": 25,
            "visual": 25,
            "joint": 50,
        }
        expected = {
            0: "TG_ONLY",
            99: "TG_ONLY",
            100: "TST_ONLY",
            149: "TST_ONLY",
            150: "NTR_ONLY",
            199: "NTR_ONLY",
            200: "CCGR_ONLY",
            249: "CCGR_ONLY",
            250: "VISUAL_ONLY",
            299: "VISUAL_ONLY",
            300: "JOINT_FINETUNE",
            399: "JOINT_FINETUNE",
        }
        self.assertEqual(
            {iteration: sequential_stage_for_iteration(ntrain, iteration, epochs)[0] for iteration in expected},
            expected,
        )
        model = self.build("spatial_rgve")
        self.assertEqual(
            _active_groups(model, "modulewise_sequential_joint", "TST_ONLY"),
            ["transport"],
        )
        self.assertEqual(
            _active_groups(model, "modulewise_sequential_joint", "NTR_ONLY"),
            ["ntr"],
        )
        self.assertEqual(
            _active_groups(model, "modulewise_sequential_joint", "JOINT_FINETUNE"),
            ["ccgr_class", "ntr", "tg_vpr", "transport", "visual"],
        )

    def test_best_handoff_boundaries_composite_groups_and_restore_are_exact(self):
        ntrain = 100
        epochs = {"tg_vpr": 50, "tst_ntr": 50, "ccgr": 25, "visual": 25, "joint": 50}
        expected = {
            0: "TG_ONLY",
            99: "TG_ONLY",
            100: "TST_NTR_ONLY",
            199: "TST_NTR_ONLY",
            200: "CCGR_ONLY",
            249: "CCGR_ONLY",
            250: "VISUAL_ONLY",
            299: "VISUAL_ONLY",
            300: "JOINT_FINETUNE",
            399: "JOINT_FINETUNE",
        }
        self.assertEqual(
            {iteration: best_handoff_stage_for_iteration(ntrain, iteration, epochs)[0] for iteration in expected},
            expected,
        )
        model = self.build("spatial_rgve")
        self.assertEqual(
            _active_groups(model, "modulewise_best_handoff", "TST_NTR_ONLY"),
            ["ntr", "transport"],
        )
        self.assertEqual(full_model_eligible_stages(model), ("VISUAL_ONLY", "JOINT_FINETUNE"))
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        expected_sha = state_dict_sha256(state)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        self.assertNotEqual(state_dict_sha256(model.state_dict()), expected_sha)
        self.assertEqual(restore_stage_best(model, state, expected_sha), expected_sha)
        self.assertEqual(state_dict_sha256(model.state_dict()), expected_sha)

    def test_module_strategy_matrix_configs_are_no_annotation_and_paired(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        files = sorted(
            path
            for path in root.glob("v2_try_1*.yaml")
            if 174 <= int(path.name.split("_")[2]) <= 184
        )
        self.assertEqual(len(files), 11)
        configs = [load_config(path)[0] for path in files]
        self.assertTrue(all(value["human_annotations_used"] is False for value in configs))
        baseline = [value for value in configs if value["training_strategy"] == "no_training"]
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["condition_id"], "M0_MEAN8")
        trained = [value for value in configs if value["training_strategy"] != "no_training"]
        self.assertEqual(len(trained), 10)
        self.assertEqual(
            {value["training_strategy"] for value in trained},
            {"end_to_end_joint", "modulewise_sequential_joint"},
        )
        for condition in ("M1_TG_VPR", "M2_TST", "M3_NTR", "M4_CCGR", "M5_VISUAL"):
            pair = [value for value in trained if value["condition_id"] == condition]
            self.assertEqual(len(pair), 2)
        expected_modes = {
            "M1_TG_VPR": ("off", "off", "off"),
            "M2_TST": ("tangent", "off", "off"),
            "M3_NTR": ("tangent_ntr", "off", "off"),
            "M4_CCGR": ("tangent_ntr", "class_conditioned_four", "off"),
            "M5_VISUAL": (
                "tangent_ntr",
                "class_conditioned_four",
                "spatial_rgve",
            ),
        }
        for value in trained:
            self.assertEqual(
                (
                    value["transport_mode"],
                    value["ccgr_mode"],
                    value["visual_mode"],
                ),
                expected_modes[value["condition_id"]],
            )
        sequential = [
            value for value in trained if value["training_strategy"] == "modulewise_sequential_joint"
        ]
        self.assertTrue(all(sum(value["stage_epochs"].values()) == 200 for value in sequential))
        self.assertTrue(all(value["nominal_epochs"] == 200 for value in trained))

    def test_v3_config_rejects_human_annotations_and_bad_stage_total(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        source = root / "v2_try_184_module-matrix_m5-visual-stagewise.yaml"
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            payload["human_annotations_used"] = True
            path.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "禁止人工属性"):
                load_config(path)
            payload["human_annotations_used"] = False
            payload["stage_epochs"]["joint"] = 49
            path.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "之和必须为200"):
                load_config(path)

    def test_strategy_seed_and_bounded_tune_configs_are_exact(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        files = sorted(
            path
            for path in root.glob("v2_try_1*.yaml")
            if 185 <= int(path.name.split("_")[2]) <= 192
        )
        self.assertEqual(len(files), 8)
        configs = {value["experiment_id"]: value for value in (load_config(path)[0] for path in files)}
        self.assertTrue(all(value["human_annotations_used"] is False for value in configs.values()))
        self.assertEqual(
            {
                (configs[attempt]["random_seed"], configs[attempt]["training_strategy"])
                for attempt in ("V2-TRY-185", "V2-TRY-186", "V2-TRY-187", "V2-TRY-188")
            },
            {
                (5, "end_to_end_joint"),
                (5, "modulewise_sequential_joint"),
                (8, "end_to_end_joint"),
                (8, "modulewise_sequential_joint"),
            },
        )
        self.assertEqual(configs["V2-TRY-189"]["max_ntr_delta"], 0.05)
        self.assertEqual(configs["V2-TRY-190"]["max_generator_magnitude"], 0.1)
        self.assertEqual(
            {
                configs["V2-TRY-191"]["visual_diversity_weight"],
                configs["V2-TRY-192"]["visual_diversity_weight"],
            },
            {0.0, 0.05},
        )

    def test_hard1_four_module_matrix_configs_are_exact(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        files = sorted(
            path
            for path in root.glob("v2_try_*.yaml")
            if 193 <= int(path.name.split("_")[2]) <= 205
        )
        self.assertEqual(len(files), 13)
        configs = {value["experiment_id"]: value for value in (load_config(path)[0] for path in files)}
        self.assertTrue(all(value["random_seed"] == 7 for value in configs.values()))
        self.assertTrue(all(value["nominal_epochs"] == 200 for value in configs.values()))
        self.assertTrue(all(value["human_annotations_used"] is False for value in configs.values()))
        staged = [value for value in configs.values() if value["training_strategy"] == "modulewise_best_handoff"]
        self.assertEqual(len(staged), 5)
        self.assertTrue(all(value["nested_official_test_selection"] is True for value in staged))
        self.assertTrue(
            all(value["selection_scope"] == "stage_best_handoff_full_model_only" for value in staged)
        )
        self.assertTrue(
            all(
                value["stage_epochs"]
                == {"tg_vpr": 50, "tst_ntr": 50, "ccgr": 25, "visual": 25, "joint": 50}
                for value in staged
            )
        )
        self.assertEqual(configs["V2-TRY-194"]["transport_mode"], "tangent_ntr")
        self.assertEqual(configs["V2-TRY-199"]["transport_mode"], "off")
        self.assertEqual(configs["V2-TRY-203"]["transport_mode"], "off")
        self.assertEqual(configs["V2-TRY-198"]["tg_vpr_mode"], "off")
        self.assertEqual(configs["V2-TRY-202"]["tg_vpr_mode"], "off")
        self.assertEqual(configs["V2-TRY-200"]["ccgr_mode"], "off")
        self.assertEqual(configs["V2-TRY-204"]["ccgr_mode"], "off")
        self.assertEqual(configs["V2-TRY-201"]["visual_mode"], "off")
        self.assertEqual(configs["V2-TRY-205"]["visual_mode"], "off")

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

    def test_deterministic_multiscale_rerun_configs(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        files = sorted(root.glob("v2_try_16[01]_multiscale-deterministic_*.yaml"))
        self.assertEqual(len(files), 2)
        configs = [load_config(path)[0] for path in files]
        self.assertTrue(all(value["visual_mode"] == "multiscale_part_tokens" for value in configs))
        self.assertTrue(all(value["patch_cache_mode"] == "gpu_fp16" for value in configs))

    def test_modulewise_visual_pair_configs(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        files = sorted(root.glob("v2_try_16[89]_modulewise-*.yaml"))
        self.assertEqual(len(files), 2)
        configs = [load_config(path)[0] for path in files]
        self.assertTrue(
            all(value["training_strategy"] == "modulewise_50_50_50_50" for value in configs)
        )
        self.assertEqual({value["visual_mode"] for value in configs}, {"off", "spatial_rgve"})

    def test_short_modulewise_joint150_pair_configs(self):
        root = Path(__file__).resolve().parents[1] / "config/tries"
        files = sorted(root.glob("v2_try_17[0-3]_short-modulewise-*.yaml"))
        self.assertEqual(len(files), 4)
        configs = [load_config(path)[0] for path in files]
        self.assertEqual(
            {value["training_strategy"] for value in configs},
            {"modulewise_short_v5_joint150", "modulewise_short_v10_joint150"},
        )
        self.assertTrue(all(value["nominal_epochs"] == 150 for value in configs))

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
