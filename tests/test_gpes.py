from pathlib import Path
import unittest

import torch

from model.innovations.gpes import (
    CenteredRoleGatedPairSelector,
    GatedPairEvidenceSelector,
    NonlinearGatedPairSelector,
    PairDiscriminativeRoleSelector,
    ReciprocalSemanticNeighborPairSelector,
    RoleAwareGatedPairSelector,
    SemanticNeighborPairSelector,
    SemanticGatedPairSelector,
    TextOnlyGatedPairSelector,
    TriadicCompetitionPairSelector,
    semantic_neighbor_adjacency,
    reciprocal_neighbor_confidence,
    pair_role_distance_weights,
)
from model.innovations.train_gpes import (
    class_balanced_pair_weights,
    extract_pair_examples,
    hard_margin_only_for_schema,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


class GPESTest(unittest.TestCase):
    def _model(self):
        groups = torch.arange(200) // 2
        return GatedPairEvidenceSelector(
            torch.randn(200, 768),
            13.0,
            torch.randn(200, 768),
            torch.randn(200, 768),
            groups,
            0.25,
            0.1,
            torch.zeros(4),
            torch.ones(4),
            0.5,
        )

    def test_zero_selector_reproduces_pair_and_full_parent(self):
        model = self._model()
        pair = torch.tensor([[1.0, 0.9], [0.7, 0.6]])
        features = torch.randn(2, 4)
        self.assertTrue(torch.equal(model.corrected_pair_logits(pair, features), pair))
        images = torch.randn(2, 768)
        parent = torch.randn(2, 200)
        patch = torch.randn(2, 200)
        expected = parent + model.sdcr_beta * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.sdcr_prototypes.T
        )
        self.assertTrue(torch.equal(model(parent, images, patch), expected))

    def test_pair_correction_preserves_pair_mean_and_has_gradients(self):
        model = self._model()
        with torch.no_grad():
            model.selector_weight[1] = 0.5
        pair = torch.tensor([[1.0, 0.9], [0.7, 0.6]])
        features = torch.randn(2, 4)
        corrected = model.corrected_pair_logits(pair, features)
        self.assertTrue(
            torch.allclose(pair.mean(dim=1), corrected.mean(dim=1), atol=1e-6)
        )
        corrected.sum().backward()
        self.assertIsNotNone(model.selector_weight.grad)

    def test_extract_pair_examples_keeps_only_gated_top2_targets(self):
        logits = torch.tensor(
            [[1.0, 0.9, 0.0], [1.0, 0.9, 0.0]], requires_grad=True
        )
        images = torch.randn(2, 4)
        patch = torch.randn(2, 3)
        targets = torch.tensor([1, 2])
        ids = torch.tensor([0, 1, 2])
        groups = torch.tensor([0, 0, 1])
        claude = torch.randn(3, 4)
        merge = torch.randn(3, 4)
        package = extract_pair_examples(
            logits, images, patch, targets, ids, groups,
            claude, merge, threshold=0.2
        )
        self.assertEqual(package[3], 1)
        self.assertEqual(int(package[2][0]), 1)
        self.assertFalse(package[0].requires_grad)
        self.assertFalse(package[1].requires_grad)

    def test_config_binds_pair_ce_and_four_features(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-062_gpes/configs/RUN-001.yaml"
        )
        self.assertEqual(config["max_delta"], 0.5)
        self.assertEqual(config["threshold_quantile"], 0.25)
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_gwps_expands_pair_scope_and_returns_soft_weights(self):
        logits = torch.tensor([[1.0, 0.9, 0.0], [2.0, 0.0, -1.0]])
        images = torch.randn(2, 4)
        patch = torch.randn(2, 3)
        targets = torch.tensor([1, 1])
        ids = torch.tensor([0, 1, 2])
        groups = torch.tensor([0, 0, 1])
        claude = torch.randn(3, 4)
        merge = torch.randn(3, 4)
        narrow = extract_pair_examples(
            logits, images, patch, targets, ids, groups,
            claude, merge, threshold=0.2, hard_margin_only=True
        )
        expanded = extract_pair_examples(
            logits, images, patch, targets, ids, groups,
            claude, merge, threshold=0.2, hard_margin_only=False
        )
        self.assertEqual(narrow[3], 1)
        self.assertEqual(expanded[3], 2)
        self.assertEqual(expanded[4].numel(), 2)
        self.assertTrue(bool((expanded[4] > 0).all()))

    def test_gwps_config_binds_soft_gate_scope(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-063_gwps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.gwps.v1")
        self.assertEqual(
            config["pair_training_scope"], "all_same_group_top2_soft_gate"
        )

    def test_balanced_pair_weights_equalize_label_mass(self):
        targets = torch.tensor([0, 0, 0, 1])
        weights, class_weights = class_balanced_pair_weights(
            targets, torch.ones(4)
        )
        self.assertGreater(float(class_weights[1]), float(class_weights[0]))
        self.assertAlmostEqual(
            float(weights[targets == 0].sum()),
            float(weights[targets == 1].sum()),
            places=6,
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_bgwps_config_binds_inverse_frequency_balance(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-064_bgwps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.bgwps.v1")
        self.assertEqual(config["pair_class_balance"], "inverse_frequency")

    def test_sqrt_balance_is_milder_than_full_inverse(self):
        targets = torch.tensor([0, 0, 0, 1])
        _, full = class_balanced_pair_weights(targets, torch.ones(4), exponent=1.0)
        weights, sqrt = class_balanced_pair_weights(
            targets, torch.ones(4), exponent=0.5
        )
        self.assertLess(float(sqrt[1] / sqrt[0]), float(full[1] / full[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_mbgwps_config_binds_sqrt_balance(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-065_mbgwps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.mbgwps.v1")
        self.assertEqual(
            config["pair_class_balance"], "sqrt_inverse_frequency"
        )

    def test_egpes_config_separates_train_and_inference_quantiles(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-066_egpes/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.egpes.v1")
        self.assertEqual(config["threshold_quantile"], 0.25)
        self.assertEqual(config["pair_training_quantile"], 0.5)

    def test_nonlinear_selector_starts_at_exact_zero_output(self):
        groups = torch.arange(200) // 2
        model = NonlinearGatedPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(4), torch.ones(4), 0.5,
            hidden_dim=8,
        )
        pair = torch.tensor([[1.0, 0.9], [0.7, 0.6]])
        features = torch.randn(2, 4)
        self.assertTrue(torch.equal(model.corrected_pair_logits(pair, features), pair))
        self.assertEqual(model.hidden_dim, 8)
        self.assertEqual(float(model.selector[-1].weight.detach().abs().sum()), 0.0)

    def test_nps_config_binds_hidden_dim(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-067_nps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.nps.v1")
        self.assertEqual(config["selector_hidden_dim"], 8)

    def test_text_only_selector_uses_three_features_and_no_patch(self):
        groups = torch.arange(200) // 2
        model = TextOnlyGatedPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(3), torch.ones(3), 0.5,
        )
        self.assertEqual(model.selector_weight.numel(), 3)
        images = torch.randn(2, 768)
        parent = torch.randn(2, 200)
        expected = parent + model.sdcr_beta * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.sdcr_prototypes.T
        )
        self.assertTrue(torch.equal(model(parent, images, None), expected))

    def test_tgwps_config_has_no_patch_fields(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-068_tgwps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.tgwps.v1")
        self.assertNotIn("patch_inputs", config)
        self.assertNotIn("feature_provenance_complete", config)
        self.assertFalse(hard_margin_only_for_schema(config["schema_version"]))

    def test_semantic_selector_adds_class_name_feature_without_patch(self):
        groups = torch.arange(200) // 2
        model = SemanticGatedPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(4), torch.ones(4), 0.5,
            class_name_prototypes=torch.randn(200, 768),
        )
        self.assertEqual(model.selector_weight.numel(), 4)
        self.assertTrue(hasattr(model, "class_name_prototypes"))

    def test_sgwps_config_is_patch_free(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-069_sgwps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.sgwps.v1")
        self.assertNotIn("patch_inputs", config)

    def test_role_aware_selector_adds_eight_sentence_differences(self):
        groups = torch.arange(200) // 2
        model = RoleAwareGatedPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(12), torch.ones(12), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
        )
        self.assertEqual(model.selector_weight.numel(), 12)
        top, _, _, features = model._top2_context(
            torch.randn(2, 200), torch.randn(2, 768), None, torch.arange(200)
        )
        self.assertEqual(tuple(top.indices.shape), (2, 2))
        self.assertEqual(tuple(features.shape), (2, 12))

    def test_rgwps_config_is_patch_free(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-070_rgwps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.rgwps.v1")
        self.assertNotIn("patch_inputs", config)

    def test_centered_role_selector_removes_common_identity(self):
        groups = torch.arange(200) // 2
        model = CenteredRoleGatedPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(12), torch.ones(12), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
        )
        _, _, _, features = model._top2_context(
            torch.randn(3, 200), torch.randn(3, 768), None, torch.arange(200)
        )
        roles = features[:, -8:]
        self.assertTrue(torch.allclose(roles.mean(dim=1), torch.zeros(3), atol=1e-5))
        self.assertTrue(torch.allclose(
            roles.std(dim=1, unbiased=False), torch.ones(3), atol=1e-4
        ))

    def test_crgwps_config_is_patch_free(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-071_crgwps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.crgwps.v1")
        self.assertNotIn("patch_inputs", config)

    def test_semantic_neighbor_adjacency_is_symmetric_without_self_edges(self):
        prototypes = torch.randn(200, 32)
        adjacency = semantic_neighbor_adjacency(prototypes, 5)
        mutual = semantic_neighbor_adjacency(prototypes, 5, mutual_only=True)
        self.assertEqual(tuple(adjacency.shape), (200, 200))
        self.assertTrue(torch.equal(adjacency, adjacency.T))
        self.assertFalse(bool(adjacency.diagonal().any()))
        self.assertTrue(bool(adjacency.sum(dim=1).ge(5).all()))
        self.assertTrue(torch.equal(mutual, mutual.T))
        self.assertFalse(bool(mutual.diagonal().any()))
        self.assertTrue(bool((mutual <= adjacency).all()))

    def test_semantic_neighbor_selector_expands_suffix_gate(self):
        groups = torch.full((200,), -1)
        adjacency = torch.zeros(200, 200, dtype=torch.bool)
        adjacency[0, 1] = adjacency[1, 0] = True
        model = SemanticNeighborPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(12), torch.ones(12), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        logits = torch.zeros(1, 200)
        logits[0, 0], logits[0, 1] = 2.0, 1.0
        _, _, related, _ = model._top2_context(
            logits, torch.randn(1, 768), None, torch.arange(200)
        )
        self.assertTrue(bool(related.item()))

    def test_snps_config_is_patch_free_and_uses_top5(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-072_snps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.snps.v1")
        self.assertEqual(config["semantic_neighbor_k"], 5)
        self.assertNotIn("patch_inputs", config)

    def test_snps_top3_rescue_config_is_bound(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-072_snps/configs/RUN-003.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.snps.v1")
        self.assertEqual(config["semantic_neighbor_k"], 3)
        self.assertEqual(
            config["pair_training_scope"], "suffix_or_semantic_top3_soft_gate"
        )

    def test_msnps_config_uses_mutual_top5(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-073_msnps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.msnps.v1")
        self.assertEqual(config["semantic_neighbor_rule"], "mutual_top5")
        self.assertNotIn("patch_inputs", config)

    def test_reciprocal_neighbor_confidence_matches_union_graph(self):
        prototypes = torch.randn(200, 32)
        confidence = reciprocal_neighbor_confidence(prototypes, 5)
        union = semantic_neighbor_adjacency(prototypes, 5)
        self.assertTrue(torch.allclose(confidence, confidence.T))
        self.assertTrue(torch.equal(confidence.gt(0), union))
        self.assertTrue(set(confidence.unique().tolist()).issubset({0.0, 0.5, 1.0}))

    def test_reciprocal_selector_disabled_returns_parent_chain(self):
        groups = torch.arange(200) // 2
        model = ReciprocalSemanticNeighborPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(12), torch.ones(12), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_confidence=torch.zeros(200, 200),
        )
        images = torch.randn(2, 768)
        parent = torch.randn(2, 200)
        expected = parent + model.sdcr_beta * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.sdcr_prototypes.T
        )
        self.assertTrue(torch.equal(model(parent, images, None, enabled=False), expected))

    def test_rsnps_config_uses_reciprocity_weighting(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-074_rsnps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.rsnps.v1")
        self.assertEqual(
            config["semantic_neighbor_rule"], "reciprocity_weighted_top5"
        )
        self.assertNotIn("patch_inputs", config)

    def test_triadic_selector_adds_third_class_gap(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = TriadicCompetitionPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(13), torch.ones(13), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        logits = torch.randn(2, 200)
        _, _, _, features = model._top2_context(
            logits, torch.randn(2, 768), None, torch.arange(200)
        )
        expected_gap = logits.topk(3, dim=1).values[:, 1] - logits.topk(
            3, dim=1
        ).values[:, 2]
        self.assertEqual(tuple(features.shape), (2, 13))
        self.assertTrue(torch.equal(features[:, -1], expected_gap))

    def test_tcps_config_binds_top3_context(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-075_tcps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.tcps.v1")
        self.assertEqual(config["semantic_neighbor_k"], 3)
        self.assertEqual(config["context_feature"], "top2_minus_top3_margin")

    def test_pair_role_distance_weights_have_pairwise_mean_one(self):
        prototypes = torch.randn(200, 8, 768)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
        weights = pair_role_distance_weights(prototypes, pairs)
        self.assertEqual(tuple(weights.shape), (3, 8))
        self.assertTrue(torch.allclose(
            weights.mean(dim=1), torch.ones(3), atol=1e-5
        ))
        self.assertTrue(bool((weights >= 0).all()))

    def test_pair_discriminative_selector_keeps_twelve_features(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = PairDiscriminativeRoleSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(12), torch.ones(12), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        _, _, _, features = model._top2_context(
            torch.randn(2, 200), torch.randn(2, 768), None, torch.arange(200)
        )
        self.assertEqual(tuple(features.shape), (2, 12))

    def test_pdrs_config_binds_pair_role_weighting(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-076_pdrs/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.pdrs.v1")
        self.assertEqual(config["semantic_neighbor_k"], 3)
        self.assertEqual(config["pair_role_weighting"], "cosine_distance_mean1")


if __name__ == "__main__":
    unittest.main()
