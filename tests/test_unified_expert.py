import unittest

import torch
import torch.nn.functional as F

from model.candidates.v2.modules.unified_expert import ExpertAttributeUnifiedModel
from model.candidates.v2.modules.unified_seen import UnifiedSeenPrototypeModel


class UnifiedExpertTest(unittest.TestCase):
    def make_models(self):
        generator = torch.Generator().manual_seed(401)
        text = UnifiedSeenPrototypeModel(
            torch.randn(200, 8, 768, generator=generator),
            torch.arange(100),
            F.normalize(torch.randn(100, 768, generator=generator), dim=-1),
            active_classes=torch.arange(150),
            dropout=0.0,
        )
        expert = ExpertAttributeUnifiedModel(
            text, torch.rand(200, 312, generator=generator)
        )
        return text, expert

    def test_expert_starts_from_no_expert_parent(self):
        text, expert = self.make_models()
        text.eval()
        expert.eval()
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(text.prototypes(), expert.prototypes(), atol=1e-6, rtol=1e-6)
            )
            self.assertEqual(float(expert.attribute_residual()), 0.0)

    def test_expert_branch_receives_gradient(self):
        _, expert = self.make_models()
        images = torch.randn(8, 768, generator=torch.Generator().manual_seed(402))
        loss = F.cross_entropy(expert.logits(images, torch.arange(100)), torch.arange(8))
        loss.backward()
        self.assertIsNotNone(expert.raw_attribute_residual.grad)
        self.assertGreater(float(expert.raw_attribute_residual.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
