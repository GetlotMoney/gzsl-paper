from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model.innovations.train_chen_stagewise import (
    load_config,
    set_trainable_stage,
    stage_for_iteration,
)
from model.innovations.unified_seen import UnifiedSeenPrototypeModel


ROOT = Path(__file__).resolve().parents[1]


class ChenStagewiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(501)
        cls.model = UnifiedSeenPrototypeModel(
            torch.randn(200, 8, 768, generator=generator),
            torch.arange(150),
            F.normalize(torch.randn(150, 768, generator=generator), dim=-1),
            dropout=0.0,
        )

    def test_fixed_stage_boundaries(self):
        config, _ = load_config(
            ROOT / "experiments/v2/confirmation/CONFIRM-005_chen_style_stagewise/configs/RUN-001.yaml"
        )
        self.assertEqual(stage_for_iteration(config, 0)["name"], "TG_ONLY")
        self.assertEqual(stage_for_iteration(config, 7049)["name"], "TG_ONLY")
        self.assertEqual(stage_for_iteration(config, 7050)["name"], "TRANSFER_CCGR")
        self.assertEqual(stage_for_iteration(config, 21149)["name"], "TRANSFER_CCGR")
        self.assertEqual(stage_for_iteration(config, 21150)["name"], "JOINT_FINETUNE")
        self.assertEqual(stage_for_iteration(config, 28227)["name"], "JOINT_FINETUNE")
        self.assertFalse(config["nested_official_test_selection"])

    def test_rescue_config_only_caps_transport(self):
        root = ROOT / "experiments/v2/confirmation/CONFIRM-005_chen_style_stagewise/configs"
        parent, _ = load_config(root / "RUN-001.yaml")
        rescue, _ = load_config(root / "RUN-002.yaml")
        rescue2, _ = load_config(root / "RUN-003.yaml")
        self.assertEqual(parent["max_transport_step"], 1.5)
        self.assertEqual(rescue["max_transport_step"], 0.5)
        self.assertEqual(rescue2["max_transport_step"], 0.75)
        ignored = {"max_transport_step"}
        self.assertEqual(
            {key: value for key, value in parent.items() if key not in ignored},
            {key: value for key, value in rescue.items() if key not in ignored},
        )
        self.assertEqual(
            {key: value for key, value in parent.items() if key not in ignored},
            {key: value for key, value in rescue2.items() if key not in ignored},
        )

    def test_trainable_groups_match_stage_contract(self):
        set_trainable_stage(self.model, "TG_ONLY")
        self.assertTrue(any(p.requires_grad for p in self.model.tg_vpr.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.model.transport_head.parameters()))
        for parameter in self.model.tg_vpr.parameters():
            parameter.grad = torch.ones_like(parameter)
        set_trainable_stage(self.model, "TRANSFER_CCGR")
        self.assertFalse(any(p.requires_grad for p in self.model.tg_vpr.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.tg_vpr.parameters()))
        self.assertTrue(any(p.requires_grad for p in self.model.transport_head.parameters()))
        self.assertTrue(any(p.requires_grad for p in self.model.generator_magnitude_head.parameters()))
        set_trainable_stage(self.model, "JOINT_FINETUNE")
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_source_has_one_global_best_not_per_stage(self):
        source = (ROOT / "model/innovations/train_chen_stagewise.py").read_text(encoding="utf-8")
        self.assertIn('if metrics["H"] > best_h:', source)
        self.assertNotIn("best_h_by_stage", source)
        self.assertIn('"nested_official_test_selection": False', source)


if __name__ == "__main__":
    unittest.main()
