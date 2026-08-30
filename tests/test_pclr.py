from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.innovations.gtd_tst import GTDTSTModel
from model.innovations.pclr import PCLRModel
from model.innovations.evaluate_pclr_inference_tuned import (
    load_inference_config,
    tuned_inference_logits,
)
from model.innovations.train_gtd_tst import (
    evaluation_updates,
    load_config,
    load_pclr_parent_history,
    validate_pclr_off_history,
    validate_tune_run_identity,
)


class _FakeVPR(nn.Module):
    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        prototypes = F.normalize(
            torch.randn(200, 768, generator=generator), dim=-1
        )
        self.prototype_parameter = nn.Parameter(prototypes)
        self.register_buffer("sentence_embeds", torch.empty(200, 8, 768))

    def base_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototype_parameter, dim=-1)

    def value_candidate(self, class_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_prototypes().index_select(0, class_ids.to(self.prototype_parameter.device))
        shifted = base + 0.05 * base.roll(1, dims=-1)
        return F.normalize(shifted, dim=-1)


class _FakeTG(nn.Module):
    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.tg_vpr = _FakeVPR(generator)
        self.raw_scale = nn.Parameter(torch.tensor(2.0))

    def prototypes(self) -> torch.Tensor:
        return self.tg_vpr.base_prototypes()

    def scale(self) -> torch.Tensor:
        return self.raw_scale.exp()

    def topology_loss(self) -> torch.Tensor:
        return self.tg_vpr.prototype_parameter.square().mean()


def _edges() -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    # A seen-only ring guarantees at least one eligible edge for every seen class.
    for class_id in range(150):
        pair = tuple(sorted((class_id, (class_id + 1) % 150)))
        if pair not in used:
            used.add(pair)
            pairs.append(pair)
    for a_id in range(200):
        for b_id in range(a_id + 1, 200):
            pair = (a_id, b_id)
            if pair not in used:
                used.add(pair)
                pairs.append(pair)
            if len(pairs) == 438:
                return torch.tensor(pairs, dtype=torch.long)
    raise AssertionError("edge fixture incomplete")


def _fixture() -> tuple[_FakeTG, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(186)
    parent = _FakeTG(generator)
    relations = F.normalize(
        torch.randn(438, 2, 768, generator=generator), dim=-1
    )
    edges = _edges()
    seen = torch.arange(150)
    return parent, seen, relations, edges


class PCLRTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        parent, seen, relations, edges = _fixture()
        cls.model = PCLRModel(parent, seen, relations, edges)
        cls.generator = torch.Generator().manual_seed(187)
        cls.images = torch.randn(4, 768, generator=cls.generator)
        cls.targets = torch.tensor([0, 1, 2, 149])

    def _zero_grad(self) -> None:
        for parameter in self.model.parameters():
            parameter.grad = None

    def _local_model(
        self,
        *,
        candidate_top_k: int = 20,
        correction_scale: float = 1.25,
        ridge_lambda: float = 1.0,
        seen_logit_gamma: float = 0.0,
    ) -> PCLRModel:
        parent, seen, relations, edges = _fixture()
        return PCLRModel(
            parent,
            seen,
            relations,
            edges,
            candidate_top_k=candidate_top_k,
            correction_scale=correction_scale,
            ridge_lambda=ridge_lambda,
            seen_logit_gamma=seen_logit_gamma,
        )

    def test_constructor_matches_parent_rng_and_reader_initialization_is_fixed(self):
        parent, seen, relations, edges = _fixture()
        torch.manual_seed(9281)
        initial_state = torch.random.get_rng_state().clone()
        GTDTSTModel(parent, seen)
        expected_state = torch.random.get_rng_state().clone()

        torch.random.set_rng_state(initial_state)
        model = PCLRModel(parent, seen, relations, edges)
        actual_state = torch.random.get_rng_state()
        self.assertTrue(torch.equal(expected_state, actual_state))

        generator = torch.Generator(device="cpu").manual_seed(18601)
        expected = torch.empty(64, 768)
        nn.init.xavier_uniform_(expected, generator=generator)
        self.assertTrue(torch.equal(model.reader_in.weight, expected))
        self.assertTrue(torch.equal(model.reader_in.bias, torch.zeros(64)))
        self.assertTrue(torch.equal(model.reader_out.weight, torch.zeros(768, 64)))
        self.assertTrue(torch.equal(model.reader_out.bias, torch.zeros(768)))
        self.assertFalse(model.relation_embeddings.requires_grad)
        self.assertFalse(model.edge_index.requires_grad)

    def test_beta_initializes_to_point_zero_five(self):
        self.assertAlmostEqual(float(self.model.beta().detach()), 0.05, places=7)
        groups = self.model.training_parameter_groups()
        self.assertEqual(set(groups), {"parent", "relation", "beta"})
        self.assertEqual(groups["beta"], (self.model.raw_beta,))

    def test_positive_single_edge_favors_a_endpoint(self):
        scores = torch.zeros(1, 438, 2)
        scores[0, 0, 0] = 1.0
        potential = self.model.potentials_from_scores(scores)
        a_id, b_id = self.model.edge_index[0].tolist()
        self.assertGreater(float(potential[0, a_id]), float(potential[0, b_id]))

    def test_potential_is_centered_bounded_and_class_subset_is_late_slice(self):
        potential = self.model.potentials(self.images)
        self.assertTrue(
            torch.allclose(
                potential.mean(dim=1), torch.zeros(4), atol=2e-7, rtol=0.0
            )
        )
        self.assertLessEqual(float(potential.detach().abs().max()), 0.5 + 1e-7)
        ids = torch.tensor([7, 2, 180, 151])
        full = self.model.pclr_logits(self.images)
        subset = self.model.pclr_logits(self.images, ids)
        self.assertTrue(torch.equal(subset, full.index_select(1, ids)))

    def test_off_path_is_bitwise_inherited_gtd_logits(self):
        ids = torch.tensor([0, 4, 151, 199])
        expected = GTDTSTModel.logits(self.model, self.images, ids)
        actual = self.model.pclr_logits(self.images, ids, enabled=False)
        self.assertTrue(torch.equal(actual, expected))

    def test_three_loss_gradient_firewalls(self):
        parent_parameters = self.model.parent_parameters()
        relation_parameters = self.model.relation_parameters()

        self._zero_grad()
        parent_loss = F.cross_entropy(
            self.model.pclr_logits(self.images), self.targets
        ) + self.model.topology_loss()
        parent_loss.backward()
        self.assertTrue(any(p.grad is not None for p in parent_parameters))
        self.assertTrue(all(p.grad is None for p in relation_parameters))
        self.assertIsNone(self.model.raw_beta.grad)

        self._zero_grad()
        relation_loss = self.model.relation_loss(self.images, self.targets)
        relation_loss.backward()
        self.assertTrue(any(p.grad is not None for p in relation_parameters))
        self.assertTrue(all(p.grad is None for p in parent_parameters))
        self.assertIsNone(self.model.raw_beta.grad)

        self._zero_grad()
        beta_loss = self.model.beta_loss(self.images, self.targets)
        beta_loss.backward()
        self.assertIsNotNone(self.model.raw_beta.grad)
        self.assertTrue(all(p.grad is None for p in parent_parameters))
        self.assertTrue(all(p.grad is None for p in relation_parameters))

    def test_relation_loss_is_mean_of_each_images_incident_edges(self):
        scores = self.model.relation_scores(self.images)
        edges = self.model.edge_index
        terms = []
        for row, target in enumerate(self.targets.tolist()):
            incident = self.model.seen_edge_mask & (
                edges[:, 0].eq(target) | edges[:, 1].eq(target)
            )
            labels = edges[incident, 1].eq(target).long()
            terms.append(F.cross_entropy(scores[row, incident], labels))
        expected = torch.stack(terms).mean()
        self.assertTrue(
            torch.allclose(
                self.model.relation_loss(self.images, self.targets),
                expected,
                atol=1e-7,
                rtol=0.0,
            )
        )

    def test_local_config_fixes_the_only_rescue_condition(self):
        config, digest = load_config(
            Path("config/tries/v4_try_023_r1_local_pclr_rescue.yaml")
        )
        self.assertEqual(config["experiment_id"], "V4-TRY-023-R1")
        self.assertEqual(config["candidate_top_k"], 20)
        self.assertEqual(config["edge_selection"], "both_endpoints_in_parent_topk")
        self.assertEqual(config["correction_scale"], 1.25)
        self.assertEqual(len(digest), 64)

    def test_local_mask_requires_both_edge_endpoints_in_parent_top20(self):
        model = self._local_model()
        logits = torch.full((2, 200), -100.0)
        selected = torch.arange(20)
        logits[0, selected] = torch.arange(20).float()
        logits[1, 180:] = torch.arange(20).float()
        mask = model.candidate_edge_mask(logits)
        expected0 = torch.isin(model.edge_index[:, 0], selected) & torch.isin(
            model.edge_index[:, 1], selected
        )
        expected1 = torch.isin(model.edge_index[:, 0], torch.arange(180, 200)) & torch.isin(
            model.edge_index[:, 1], torch.arange(180, 200)
        )
        self.assertTrue(torch.equal(mask[0], expected0))
        self.assertTrue(torch.equal(mask[1], expected1))

    def test_local_potential_is_exact_masked_fixed_laplacian_solution(self):
        model = self._local_model()
        scores = torch.randn(2, 438, 2, generator=self.generator)
        parent_logits = torch.randn(2, 200, generator=self.generator)
        actual = model.potentials_from_scores(scores, parent_logits)
        difference = scores[..., 0] - scores[..., 1]
        difference = difference * model.candidate_edge_mask(parent_logits)
        expected = difference @ model.laplacian_map.T
        expected = expected - expected.mean(dim=1, keepdim=True)
        norm = expected.abs().amax(dim=1, keepdim=True)
        expected = 0.5 * expected / torch.maximum(norm, torch.full_like(norm, 0.5))
        self.assertTrue(torch.equal(actual, expected))

    def test_local_off_uses_canonical_normalized_prototype_evaluation(self):
        model = self._local_model().eval()
        actual = model.pclr_logits(self.images, enabled=False)
        expected = (
            F.normalize(self.images.float(), dim=-1)
            @ F.normalize(model.prototypes().float(), dim=-1).T
            * model.scale()
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_local_full_is_exact_parent_plus_scaled_masked_correction(self):
        model = self._local_model().eval()
        parent = model.deployed_parent_logits(self.images)
        potential = model.potentials(self.images, parent)
        expected = (
            parent
            + 1.25
            * model.beta().detach()
            * parent.std(dim=1, unbiased=False, keepdim=True)
            * potential
        )
        self.assertTrue(torch.equal(model.pclr_logits(self.images), expected))

    def test_local_scale_and_topk_do_not_break_gradient_firewalls(self):
        model = self._local_model()
        relation_parameters = model.relation_parameters()
        loss = model.beta_loss(self.images, self.targets)
        loss.backward()
        self.assertIsNotNone(model.raw_beta.grad)
        self.assertTrue(all(parameter.grad is None for parameter in relation_parameters))
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parent_parameters())
        )

    def test_local_parent_history_is_sha_bound_and_every_point_is_compared(self):
        config, _ = load_config(
            Path("config/tries/v4_try_023_r1_local_pclr_rescue.yaml")
        )
        updates = [0, *evaluation_updates()]
        rows = [
            {
                "evaluation_index": index,
                "update": update,
                "U": 70.0 + index / 1000,
                "S": 80.0,
                "H": 75.0,
                "ZS": 85.0,
            }
            for index, update in enumerate(updates)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_history.json"
            path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            config["parent_evaluation_history"] = str(path.resolve())
            config["parent_evaluation_history_sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            parent = load_pclr_parent_history(config)
            history = [
                {
                    "evaluation_index": row["evaluation_index"],
                    "update": row["update"],
                    "module_off_metrics": {
                        metric: row[metric] for metric in ("U", "S", "H", "ZS")
                    },
                }
                for row in parent
            ]
            validate_pclr_off_history(history, parent)
            history[131]["module_off_metrics"]["ZS"] -= 0.01
            with self.assertRaisesRegex(RuntimeError, "evaluation_index=131"):
                validate_pclr_off_history(history, parent)

    def test_local_output_dir_and_off_subset_contracts_are_strict(self):
        config, digest = load_config(
            Path("config/tries/v4_try_023_r1_local_pclr_rescue.yaml")
        )
        validate_tune_run_identity(
            config, digest, digest, Path("V4-TRY-023-R1")
        )
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_tune_run_identity(config, digest, digest, Path("wrong"))
        model = self._local_model().eval()
        with self.assertRaisesRegex(ValueError, "class_ids"):
            model.pclr_logits(
                self.images,
                torch.tensor([0, 0, 151]),
                enabled=False,
            )
        diagnostics = model.pclr_diagnostics(self.images)
        self.assertAlmostEqual(
            diagnostics["effective_beta"],
            1.25 * diagnostics["beta"],
        )
        self.assertEqual(diagnostics["effective_beta_max"], 0.3125)

    def test_tuned_local_config_and_calibrated_off_are_explicit(self):
        config, digest = load_config(
            Path("config/tries/v4_try_023_r2_tuned_local_pclr.yaml")
        )
        self.assertEqual(config["experiment_id"], "V4-TRY-023-R2")
        self.assertEqual(config["candidate_top_k"], 15)
        self.assertEqual(config["ridge_lambda"], 0.03)
        self.assertEqual(config["correction_scale"], 2.38)
        self.assertEqual(config["seen_logit_gamma"], 0.525)
        validate_tune_run_identity(
            config, digest, digest, Path("V4-TRY-023-R2")
        )
        model = self._local_model(
            candidate_top_k=15,
            correction_scale=2.38,
            ridge_lambda=0.03,
            seen_logit_gamma=0.525,
        ).eval()
        raw = model.pclr_logits(self.images, enabled=False)
        calibrated = model.pclr_logits(
            self.images, enabled=False, calibrated=True
        )
        expected = raw.clone()
        expected[:, model.seen_classes] -= 0.525
        self.assertTrue(torch.equal(calibrated, expected))
        zs = torch.arange(150, 200)
        self.assertTrue(
            torch.equal(
                model.pclr_logits(
                    self.images, zs, enabled=False, calibrated=True
                ),
                raw.index_select(1, zs),
            )
        )

    def test_tuned_local_beta_loss_does_not_train_on_test_selected_gamma(self):
        model = self._local_model(
            candidate_top_k=15,
            correction_scale=2.38,
            ridge_lambda=0.03,
            seen_logit_gamma=0.525,
        ).eval()
        parent = model.deployed_parent_logits(self.images).detach()
        potential = model.potentials(self.images, parent).detach()
        expected_logits = (
            parent
            + 2.38
            * model.beta()
            * parent.std(dim=1, unbiased=False, keepdim=True)
            * potential
        )
        self.assertTrue(
            torch.equal(
                model.beta_loss(self.images, self.targets),
                F.cross_entropy(expected_logits, self.targets),
            )
        )
        self.assertEqual(model.pclr_diagnostics(self.images)["seen_logit_gamma"], 0.525)

    def test_r3_inference_config_and_three_paths_are_explicit(self):
        config, digest = load_inference_config(
            Path("config/tries/v4_try_023_r3_pclr_inference_tune.yaml")
        )
        self.assertEqual(config["candidate_top_k"], 17)
        self.assertEqual(config["ridge_lambda"], 0.3)
        self.assertEqual(config["inference_relation_temperature"], 0.2)
        self.assertEqual(config["correction_scale"], 6.95)
        self.assertEqual(config["seen_logit_gamma"], 0.575)
        self.assertTrue(config["nested_official_test_selection"])
        self.assertEqual(len(digest), 64)
        model = self._local_model(
            candidate_top_k=15,
            correction_scale=2.38,
            ridge_lambda=0.03,
            seen_logit_gamma=0.525,
        ).eval()
        raw, calibrated, full, active = tuned_inference_logits(
            model,
            self.images,
            candidate_top_k=17,
            ridge_lambda=0.3,
            potential_cap=0.5,
            inference_relation_temperature=0.2,
            correction_scale=6.95,
            seen_logit_gamma=0.575,
        )
        self.assertTrue(torch.equal(raw, model.deployed_parent_logits(self.images)))
        expected = raw.clone()
        expected[:, model.seen_classes] -= 0.575
        self.assertTrue(torch.equal(calibrated, expected))
        self.assertTrue(torch.equal(full[:, model.unseen_classes], (
            full.index_select(1, model.unseen_classes)
        )))
        self.assertTrue(torch.isfinite(full).all())
        self.assertGreaterEqual(active, 0.0)
        self.assertLessEqual(active, 1.0)


if __name__ == "__main__":
    unittest.main()
