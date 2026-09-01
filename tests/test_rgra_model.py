import unittest

import torch
import torch.nn.functional as F

from model.frameworks.v6.rgra import RGRAModel, raw_role_groups


class RGRAModelTest(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(7)
        classes = 6
        roles = F.normalize(torch.randn(classes, 8, 768), dim=-1)
        relations = F.normalize(torch.randn(7, 2, 768), dim=-1)
        edges = torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5]])
        seen = torch.tensor([0, 1, 2, 3])
        p_v5 = F.normalize(torch.randn(classes, 768), dim=-1)
        return RGRAModel(roles, seen, relations, edges, p_v5=p_v5, class_count=classes, hidden_dim=16)

    def test_raw_groups_are_six_plus_one_plus_one(self):
        roles = torch.randn(5, 8, 768)
        groups = raw_role_groups(roles)
        self.assertEqual(tuple(groups.shape), (5, 3, 768))
        self.assertTrue(torch.allclose(groups.norm(dim=-1), torch.ones(5, 3), atol=1e-5))

    def test_cls_loss_reaches_all_three_modules(self):
        model = self.make_model()
        cls = torch.randn(4, 768)
        patches = torch.randn(4, 36, 768)
        targets = torch.tensor([0, 1, 2, 3])
        seen = model.seen_classes
        loss = F.cross_entropy(model(cls, patches).index_select(1, seen), torch.arange(4))
        loss.backward()
        norms = {}
        for name, params in model.training_parameter_groups().items():
            norms[name] = sum(
                float(param.grad.norm()) for param in params if param.grad is not None
            )
        self.assertGreater(norms["semantic"], 0.0)
        self.assertGreater(norms["visual"], 0.0)
        self.assertGreater(norms["interaction"], 0.0)

    def test_alpha_zero_matches_i_off(self):
        model = self.make_model()
        cls = torch.randn(3, 768)
        patches = torch.randn(3, 36, 768)
        with torch.no_grad():
            model.raw_alpha.fill_(-80.0)
            full = model.logits(cls, patches, mode="full")
            off = model.logits(cls, patches, mode="i_off")
        self.assertLessEqual(float((full - off).abs().max()), 1e-6)

    def test_modes_preserve_shape_and_export(self):
        model = self.make_model()
        cls = torch.randn(2, 768)
        patches = torch.randn(2, 36, 768)
        for mode in ("full", "s_off", "v_off", "i_off", "additive", "shuffled"):
            logits = model.logits(cls, patches, mode=mode)
            self.assertEqual(tuple(logits.shape), (2, 6))
            self.assertTrue(torch.isfinite(logits).all())
        exported = model.export_classifier()
        self.assertEqual(tuple(exported["prototypes"].shape), (6, 768))
        self.assertEqual(tuple(exported["relation_field"].shape), (6, 768))


if __name__ == "__main__":
    unittest.main()

