import unittest

import torch

from model.innovations.lver import LocalViewEvidenceRouter


class LVERTest(unittest.TestCase):
    def _inputs(self, batch: int = 3, classes: int = 11):
        generator = torch.Generator().manual_seed(17)
        parent = torch.randn(batch, classes, generator=generator)
        local = torch.randn(batch, 4, 768, generator=generator)
        prototypes = torch.randn(classes, 768, generator=generator)
        global_features = torch.randn(batch, 768, generator=generator)
        return parent, local, prototypes, global_features

    def test_shape_finite_and_zero_or_off_is_exact(self):
        model = LocalViewEvidenceRouter()
        inputs = self._inputs()
        zero = model(*inputs)
        off = model(*inputs, enabled=False)
        self.assertEqual(zero.shape, inputs[0].shape)
        self.assertTrue(torch.isfinite(zero).all())
        self.assertTrue(torch.equal(zero, inputs[0]))
        self.assertTrue(torch.equal(off, inputs[0]))

    def test_only_parent_top3_changes_and_candidate_delta_is_zero_sum(self):
        model = LocalViewEvidenceRouter()
        parent, local, prototypes, global_features = self._inputs()
        with torch.no_grad():
            model.raw_strength.fill_(0.4)
        output = model(parent, local, prototypes, global_features)
        parts = model.components(parent, local, prototypes, global_features)
        delta = output - parent
        candidate_delta = delta.gather(1, parts["candidate_indices"])
        candidate_mask = torch.zeros_like(parent, dtype=torch.bool)
        candidate_mask.scatter_(1, parts["candidate_indices"], True)
        self.assertTrue(torch.equal(delta.masked_select(~candidate_mask), torch.zeros_like(delta.masked_select(~candidate_mask))))
        self.assertTrue(
            torch.allclose(
                candidate_delta.sum(dim=1),
                torch.zeros(parent.shape[0]),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_shared_parameters_receive_gradients(self):
        model = LocalViewEvidenceRouter()
        parent, local, prototypes, global_features = self._inputs()
        parent.requires_grad_()
        local.requires_grad_()
        with torch.no_grad():
            model.raw_strength.fill_(0.2)
        output = model(parent, local, prototypes, global_features)
        weights = torch.linspace(-1.0, 1.0, parent.shape[1]).unsqueeze(0)
        loss = (output * weights).sum()
        loss.backward()
        parameters = list(model.parameters())
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        self.assertTrue(any(float(parameter.grad.abs().sum()) > 0.0 for parameter in parameters))
        self.assertIsNotNone(local.grad)
        self.assertTrue(torch.isfinite(local.grad).all())


if __name__ == "__main__":
    unittest.main()
