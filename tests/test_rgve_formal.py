import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from model.frameworks.v4.model import PaperV2RGVEModel, PaperV2ThreeModuleModel
from model.candidates.v2.trainers.paper_v2 import _active_groups, load_config, stage_for_iteration


ROOT = Path(__file__).resolve().parents[1]


class FormalRGVEContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        sentence_embeds = F.normalize(torch.randn(10, 8, 768), dim=-1)
        seen_classes = torch.tensor([0, 1, 2, 3, 4, 5])
        centroids = F.normalize(torch.randn(6, 768), dim=-1)
        self.parent = PaperV2ThreeModuleModel(
            sentence_embeds,
            seen_classes,
            centroids,
            dropout=0.0,
        )
        self.images = F.normalize(torch.randn(3, 768), dim=-1)
        self.patches = F.normalize(torch.randn(3, 12, 768), dim=-1)

    def test_off_path_is_exact_parent(self):
        model = PaperV2RGVEModel(self.parent, rgve_mode="off")
        expected = self.parent.logits(self.images)
        actual = model.logits(self.images, self.patches)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(model.parameter_groups()["rgve"], [])

    def test_enabled_initialization_is_exact_parent(self):
        model = PaperV2RGVEModel(
            self.parent,
            rgve_mode="soft_attention_calibrated",
        )
        self.assertEqual(float(model.rgve.beta().detach()), 0.0)
        expected = self.parent.logits(self.images)
        actual = model.logits(self.images, self.patches)
        self.assertTrue(torch.equal(expected, actual))
        self.assertGreater(len(model.parameter_groups()["rgve"]), 0)

    def test_stagewise_activation_is_fixed_and_includes_rgve(self):
        model = PaperV2RGVEModel(
            self.parent,
            rgve_mode="soft_attention_calibrated",
        )
        self.assertEqual(stage_for_iteration(100, 0)[0], "TG_ONLY")
        self.assertEqual(stage_for_iteration(100, 100)[0], "TRANSFER_CCGR")
        self.assertEqual(stage_for_iteration(100, 300)[0], "JOINT_FINETUNE")
        self.assertNotIn("rgve", _active_groups(model, "stagewise_50_100_50", "TG_ONLY"))
        self.assertIn("rgve", _active_groups(model, "stagewise_50_100_50", "TRANSFER_CCGR"))
        self.assertIn("rgve", _active_groups(model, "stagewise_50_100_50", "JOINT_FINETUNE"))
        self.assertIn("rgve", _active_groups(model, "end_to_end_joint", "END_TO_END"))

    def test_four_formal_configs_share_the_chen_contract(self):
        config_root = ROOT / "experiments/v2/confirmation/CONFIRM-011_rgve_formal/configs"
        configs = [load_config(path)[0] for path in sorted(config_root.glob("RUN-*.yaml"))]
        self.assertEqual(len(configs), 4)
        self.assertEqual({config["batch_size"] for config in configs}, {50})
        self.assertEqual({config["nominal_epochs"] for config in configs}, {200})
        self.assertEqual({config["random_seed"] for config in configs}, {7})
        self.assertEqual(
            {config["training_strategy"] for config in configs},
            {"end_to_end_joint", "stagewise_50_100_50"},
        )
        self.assertEqual(
            {config["rgve_mode"] for config in configs},
            {"off", "soft_attention_calibrated"},
        )
        self.assertTrue(all(config["nested_official_test_selection"] is False for config in configs))
        self.assertTrue(all(config["unseen_images_used_for_gradient"] is False for config in configs))


if __name__ == "__main__":
    unittest.main()
