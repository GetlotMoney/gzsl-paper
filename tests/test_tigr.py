from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model.innovations.tigr import (
    TaxonomicIntraGroupResidual,
    TaxonomicWithinGroupLogitSharpening,
    taxonomic_suffix_group_ids,
)
from model.innovations.train_tigr import load_config


ROOT = Path(__file__).resolve().parents[1]


class TIGRTest(unittest.TestCase):
    def test_suffix_groups_join_same_taxonomic_family(self):
        names = [f"{index:03d}.Unique_{index}" for index in range(200)]
        names[0] = "001.Blue_winged_Warbler"
        names[1] = "002.Wilson_Warbler"
        names[2] = "003.Tree_Sparrow"
        names[3] = "004.Baird_Sparrow"
        groups = taxonomic_suffix_group_ids(names)
        self.assertEqual(int(groups[0]), int(groups[1]))
        self.assertEqual(int(groups[2]), int(groups[3]))
        self.assertNotEqual(int(groups[0]), int(groups[2]))

    def test_zero_identity_beta_reproduces_sdcr_parent(self):
        generator = torch.Generator().manual_seed(853)
        prototypes = torch.randn(200, 768, generator=generator)
        groups = torch.arange(200) // 2
        model = TaxonomicIntraGroupResidual(prototypes, 13.0, groups, 5.0)
        images = torch.randn(3, 768, generator=generator)
        parent = torch.randn(3, 200, generator=generator)
        expected = parent + 13.0 * (
            F.normalize(images, dim=-1) @ F.normalize(prototypes, dim=-1).T
        )
        self.assertTrue(torch.equal(model(parent, images), expected))

    def test_config_binds_taxonomic_rule_and_seen_only_training(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-055_tigr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["group_rule"], "class_name_last_token_min2")
        self.assertEqual(config["max_beta"], 5.0)
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_twls_zero_alpha_is_identity_and_group_mean_is_preserved(self):
        generator = torch.Generator().manual_seed(857)
        prototypes = torch.randn(200, 768, generator=generator)
        groups = torch.arange(200) // 2
        model = TaxonomicWithinGroupLogitSharpening(
            prototypes, 13.0, groups, 1.0
        )
        images = torch.randn(4, 768, generator=generator)
        parent = torch.randn(4, 200, generator=generator)
        baseline = model(parent, images)
        with torch.no_grad():
            model.raw_alpha.fill_(0.4)
        sharpened = model(parent, images)
        self.assertTrue(
            torch.allclose(
                baseline[:, :2].mean(dim=1),
                sharpened[:, :2].mean(dim=1),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        model.raw_alpha.data.zero_()
        self.assertTrue(torch.equal(model(parent, images), baseline))

    def test_twls_config_binds_logit_space_operation(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-056_twls/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.twls.v1")
        self.assertEqual(config["max_alpha"], 1.0)


if __name__ == "__main__":
    unittest.main()
