from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model.innovations.train_unified_seen import full_epoch_batches, load_config
from model.innovations.unified_seen import UnifiedSeenPrototypeModel


ROOT = Path(__file__).resolve().parents[1]


class UnifiedSeenTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(301)
        cls.sentences = torch.randn(200, 8, 768, generator=generator)
        cls.seenclasses = torch.arange(150)
        cls.centroids = F.normalize(
            torch.randn(150, 768, generator=generator), dim=-1
        )

    def make_model(self):
        return UnifiedSeenPrototypeModel(
            self.sentences,
            self.seenclasses,
            self.centroids,
            dropout=0.0,
        )

    def test_development_mode_accepts_100_seen_and_150_active_classes(self):
        model = UnifiedSeenPrototypeModel(
            self.sentences,
            torch.arange(100),
            self.centroids[:100],
            active_classes=torch.arange(150),
            dropout=0.0,
        )
        self.assertEqual(model.seenclasses.numel(), 100)
        self.assertEqual(model.active_classes.numel(), 150)
        self.assertTrue(torch.isfinite(model.topology_loss()))

    def test_external_fold_parent_reuses_shared_transfer_generator(self):
        model = self.make_model().eval()
        with torch.no_grad():
            direct = model.prototype_stages()
            external = model.prototype_stages_from_tg(model.tg_vpr, model.seenclasses)
        for key in direct:
            self.assertTrue(torch.equal(direct[key], external[key]), key)

        fold_parent = UnifiedSeenPrototypeModel(
            self.sentences,
            torch.arange(100),
            self.centroids[:100],
            active_classes=torch.arange(150),
            dropout=0.0,
        ).tg_vpr
        stages = model.prototype_stages_from_tg(fold_parent, torch.arange(100))
        self.assertEqual(tuple(stages["final"].shape), (200, 768))

    def test_full_epoch_batches_cover_every_sample_once(self):
        batches = full_epoch_batches(7057, 64, torch.Generator().manual_seed(7))
        joined = torch.cat(batches)
        self.assertEqual(len(batches), 111)
        self.assertEqual(batches[-1].numel(), 17)
        self.assertEqual(joined.numel(), 7057)
        self.assertEqual(joined.unique().numel(), 7057)
        self.assertTrue(torch.equal(joined.sort().values, torch.arange(7057)))

    def test_zero_residual_initialization_matches_tg_vpr(self):
        model = self.make_model().eval()
        with torch.no_grad():
            stages = model.prototype_stages()
        self.assertTrue(torch.allclose(stages["tg_vpr"], stages["transported"], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(stages["tg_vpr"], stages["final"], atol=1e-6, rtol=1e-6))
        self.assertEqual(float(stages["transport_step"].abs().max()), 0.0)
        self.assertEqual(float(stages["generator_magnitude"].abs().max()), 0.0)

    def test_all_three_module_groups_receive_gradient(self):
        model = self.make_model().train()
        images = torch.randn(8, 768, generator=torch.Generator().manual_seed(302))
        targets = torch.arange(8)
        loss = F.cross_entropy(model.logits(images, self.seenclasses), targets)
        loss = loss + 0.1 * model.topology_loss()
        loss.backward()
        groups = {
            "tg_vpr": model.tg_vpr.parameters(),
            "transport": list(model.transport_trunk.parameters()) + list(model.transport_head.parameters()),
            "generator": list(model.generator_trunk.parameters()) + list(model.generator_weight_head.parameters()) + list(model.generator_magnitude_head.parameters()),
        }
        for name, parameters in groups.items():
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            self.assertTrue(gradients, name)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0, name)

    def test_config_and_official_test_boundary(self):
        config, _ = load_config(
            ROOT
            / "experiments/v2/confirmation/CONFIRM-002_unified_seen_training/configs/RUN-001.yaml"
        )
        self.assertEqual(config["epochs"], 50)
        self.assertFalse(config["test_used_for_selection"])
        self.assertEqual(config["official_test_load_epoch"], "after_epoch_50")
        source = (ROOT / "model/innovations/train_unified_seen.py").read_text(encoding="utf-8")
        self.assertNotIn("fixed_class_folds", source)
        self.assertLess(source.index("for epoch in range"), source.index("official test只在全部50轮"))

    def test_tg_vpr_only_control_config(self):
        config, _ = load_config(
            ROOT
            / "experiments/v2/confirmation/CONFIRM-002_unified_seen_training/configs/RUN-002.yaml"
        )
        self.assertEqual(config["model_variant"], "tg_vpr_only")
        self.assertEqual(config["random_seed"], 7)
        self.assertEqual(config["epochs"], 50)
        self.assertFalse(config["test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
