from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from model.frameworks.v6.arra import (
    CLASS_COUNT,
    EDGE_COUNT,
    EMBED_DIM,
    PATCH_COUNT,
    ROLE_COUNT,
    ARRAClassifier,
    ARRAGraphFreeClassifier,
    compile_relation_field,
)


def _edges() -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for class_id in range(150):
        pair = tuple(sorted((class_id, (class_id + 1) % 150)))
        if pair not in used:
            used.add(pair)
            pairs.append(pair)
    for a_id in range(CLASS_COUNT):
        for b_id in range(a_id + 1, CLASS_COUNT):
            pair = (a_id, b_id)
            if pair not in used:
                used.add(pair)
                pairs.append(pair)
            if len(pairs) == EDGE_COUNT:
                return torch.tensor(pairs, dtype=torch.long)
    raise AssertionError("edge fixture incomplete")


def _fixture() -> tuple[ARRAClassifier, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(9206)
    prototypes = F.normalize(
        torch.randn(CLASS_COUNT, EMBED_DIM, generator=generator), dim=-1
    )
    roles = F.normalize(
        torch.randn(CLASS_COUNT, ROLE_COUNT, EMBED_DIM, generator=generator),
        dim=-1,
    )
    relations = F.normalize(
        torch.randn(EDGE_COUNT, 2, EMBED_DIM, generator=generator),
        dim=-1,
    )
    cls = torch.randn(4, EMBED_DIM, generator=generator)
    patches = torch.randn(4, PATCH_COUNT, EMBED_DIM, generator=generator)
    model = ARRAClassifier(
        prototypes,
        roles,
        relations,
        _edges(),
        torch.arange(150),
        torch.tensor(14.2),
    )
    return model, cls, patches, torch.tensor([0, 1, 2, 149])


class ARRATest(unittest.TestCase):
    def test_constructor_freezes_eval_mode_anchor_scale_and_compiles_relations(self) -> None:
        model, _, _, _ = _fixture()
        self.assertFalse(model.class_prototypes.requires_grad)
        self.assertFalse(model.source_scale.requires_grad)
        self.assertFalse(model.edge_index.requires_grad)
        weights = model.role_weights().detach()
        self.assertAlmostEqual(float(weights[0]), 0.16, places=6)
        self.assertAlmostEqual(float(weights[6]), 0.36, places=6)
        self.assertAlmostEqual(float(weights[1]), 0.0, places=6)
        self.assertAlmostEqual(float(model.beta().detach()), 0.10, places=6)
        self.assertAlmostEqual(float(model.alpha().detach()), 1.0, places=6)
        self.assertAlmostEqual(float(model.delta().detach()), 0.0, places=6)
        expected = compile_relation_field(
            model.relation_sentence_embeds,
            model.edge_index,
            ridge_lambda=model.ridge_lambda,
        )
        self.assertTrue(torch.equal(model.compiled_relation_field, expected))
        incidence = torch.zeros(EDGE_COUNT, CLASS_COUNT)
        rows = torch.arange(EDGE_COUNT)
        incidence[rows, model.edge_index[:, 0]] = 1.0
        incidence[rows, model.edge_index[:, 1]] = -1.0
        mapping = torch.linalg.solve(
            incidence.T @ incidence + model.ridge_lambda * torch.eye(CLASS_COUNT),
            incidence.T,
        )
        raw_difference = (
            model.relation_sentence_embeds[:, 0]
            - model.relation_sentence_embeds[:, 1]
        )
        receipt_field = F.normalize(mapping @ raw_difference, dim=-1)
        normalized_edge_field = F.normalize(
            mapping @ F.normalize(raw_difference, dim=-1), dim=-1
        )
        self.assertTrue(torch.equal(model.compiled_relation_field, receipt_field))
        self.assertFalse(torch.allclose(receipt_field, normalized_edge_field, atol=1e-6))

    def test_full_formula_and_controls_are_explicit(self) -> None:
        model, cls, patches, _ = _fixture()
        with torch.no_grad():
            model.raw_delta.fill_(0.5 * math.log((1.0 + 0.4) / (1.0 - 0.4)))
        full = model.components(cls, patches)
        expected = (
            full.semantic_logits
            + full.beta * full.visual_logits
            + full.relation_logits
            + full.calibrated_bias
        )
        self.assertTrue(torch.equal(full.logits, expected))
        self.assertLess(
            float((full.attention.sum(dim=-1) - 1.0).abs().max().detach()),
            1e-6,
        )
        self.assertGreater(float(full.attention.std().detach()), 0.0)

        i_off = model.components(cls, patches, condition="i_off")
        alpha0 = model.components(cls, patches, alpha_override=0.0)
        self.assertTrue(torch.equal(i_off.logits, alpha0.logits))
        self.assertTrue(torch.equal(i_off.relation_logits, torch.zeros_like(i_off.relation_logits)))

        additive = model.components(cls, patches, condition="additive")
        explicit_additive = model.components(cls, patches, delta_override=0.0)
        self.assertTrue(torch.equal(additive.logits, explicit_additive.logits))

        v_off = model.components(cls, patches, condition="v_off")
        self.assertTrue(torch.equal(v_off.visual_logits, torch.zeros_like(v_off.visual_logits)))
        self.assertTrue(torch.equal(v_off.g, torch.full_like(v_off.g, 0.5)))
        self.assertTrue(torch.allclose(v_off.z, F.normalize(cls.float(), dim=-1)))

        s_off = model.components(cls, patches, condition="s_off")
        self.assertFalse(torch.equal(s_off.logits, full.logits))
        self.assertTrue(torch.equal(s_off.visual_logits, full.visual_logits))

    def test_losses_match_contract_and_cls_gradients_reach_every_module(self) -> None:
        model, cls, patches, targets = _fixture()
        loss = model.classification_loss(cls, patches, targets)
        loss.backward()
        self.assertGreater(float(model.raw_role_weights.grad.abs().sum().detach()), 0.0)
        self.assertGreater(float(model.raw_beta.grad.abs().detach()), 0.0)
        self.assertGreater(float(model.raw_alpha.grad.abs().detach()), 0.0)
        self.assertGreater(float(model.raw_delta.grad.abs().detach()), 0.0)
        self.assertTrue(any(
            p.grad is not None and float(p.grad.abs().sum().detach()) > 0.0
            for p in model.visual_adapter.parameters()
        ))
        self.assertGreater(float(model.patch_query.weight.grad.abs().sum().detach()), 0.0)
        self.assertGreater(float(model.patch_key.weight.grad.abs().sum().detach()), 0.0)
        self.assertTrue(any(
            p.grad is not None and float(p.grad.abs().sum().detach()) > 0.0
            for p in model.relation_reader.parameters()
        ))

        values = model.losses(cls, patches, targets)
        expected = values["cls"] + 0.3 * values["topology"] + 0.1 * values["direction"]
        self.assertTrue(torch.allclose(values["total"], expected, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.isfinite(values["topology"]))
        self.assertTrue(torch.isfinite(values["direction"]))
        with self.assertRaisesRegex(ValueError, "seen"):
            model.classification_loss(cls[:1], patches[:1], torch.tensor([175]))

    def test_optimizer_groups_are_the_two_preregistered_groups(self) -> None:
        model, _, _, _ = _fixture()
        groups = model.optimizer_parameter_groups()
        self.assertEqual([group["name"] for group in groups], ["role_relation", "visual_interaction"])
        self.assertEqual([group["lr"] for group in groups], [3e-6, 3e-5])
        self.assertTrue(all(group["weight_decay"] == 1e-3 for group in groups))
        ids = [id(param) for group in groups for param in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(id(model.raw_delta), ids)

    def test_affine_reference_matches_beta0_delta0_full_at_initialization(self) -> None:
        model, cls, patches, _ = _fixture()
        model.eval()
        with torch.no_grad():
            full = model(
                cls,
                patches,
                beta_override=0.0,
                delta_override=0.0,
            )
            reference = model.affine_reference_logits(cls)
        self.assertTrue(torch.allclose(full, reference, atol=1e-6, rtol=0.0))

    def test_graph_free_export_reload_preserves_logits_without_graph_assets(self) -> None:
        model, cls, patches, _ = _fixture()
        with torch.no_grad():
            model.raw_delta.fill_(0.25)
            model.raw_beta.fill_(-1.5)
        payload = model.export_graph_free()
        self.assertNotIn("edge_index", payload)
        self.assertNotIn("relation_sentence_embeds", payload)
        self.assertNotIn("relation_embeddings", payload)
        restored = ARRAGraphFreeClassifier.from_export(payload)
        shuffle = torch.stack([torch.arange(CLASS_COUNT - 1, -1, -1) for _ in range(cls.size(0))])
        for condition in ("full", "s_off", "v_off", "i_off", "additive", "shuffled"):
            kwargs = {"shuffle_indices": shuffle} if condition == "shuffled" else {}
            expected = model(cls, patches, condition=condition, **kwargs)
            actual = restored(cls, patches, condition=condition, **kwargs)
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
