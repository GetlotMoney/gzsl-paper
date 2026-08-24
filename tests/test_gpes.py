from pathlib import Path
import unittest

import torch

from model.innovations.gpes import (
    AntisymmetricPairSelector,
    BiasFreeSemanticNeighborSelector,
    CenteredRoleGatedPairSelector,
    CrossSourceDisagreementSelector,
    GatedPairEvidenceSelector,
    LocalSemanticCompetitionResolver,
    NonlinearGatedPairSelector,
    NeighborhoodDegreePairSelector,
    PairDiscriminativeRoleSelector,
    ReciprocalSemanticNeighborPairSelector,
    RoleDisagreementScaleSelector,
    RoleVotePairSelector,
    RoleUncertaintyGatedSelector,
    RoleAwareGatedPairSelector,
    SemanticNeighborPairSelector,
    StagedRoleDisagreementScaleSelector,
    SemanticGatedPairSelector,
    TextOnlyGatedPairSelector,
    TriadicCompetitionPairSelector,
    TrustRegionRoleDisagreementScaleSelector,
    semantic_neighbor_adjacency,
    reciprocal_neighbor_confidence,
    pair_role_distance_weights,
)
from model.innovations.train_gpes import (
    class_balanced_pair_weights,
    antisymmetric_pair_augmentation,
    extract_pair_examples,
    extract_triplet_examples,
    extract_teacher_forced_pairs,
    focal_pair_losses,
    hard_margin_only_for_schema,
    load_config,
    matched_hard_pair_indices,
    true_class_balancing_weights,
    minimal_flip_delta_targets,
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

    def test_minimal_flip_targets_leave_correct_pairs_unchanged(self):
        logits = torch.tensor([[2.0, 1.0], [0.4, 0.2], [3.0, 0.0]])
        targets = minimal_flip_delta_targets(
            logits, torch.tensor([0, 1, 1]), max_delta=0.5
        )
        self.assertTrue(torch.equal(targets, torch.tensor([0.0, -0.1, -0.5])))

    def test_etpc_config_binds_minimal_flip_regression(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-077_etpc/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.etpc.v1")
        self.assertEqual(config["training_objective"], "minimal_flip_regression")
        self.assertEqual(config["semantic_neighbor_k"], 3)

    def test_role_disagreement_scale_selector_adds_raw_std(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = RoleDisagreementScaleSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(13), torch.ones(13), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        _, _, _, features = model._top2_context(
            torch.randn(2, 200), torch.randn(2, 768), None, torch.arange(200)
        )
        self.assertEqual(tuple(features.shape), (2, 13))
        self.assertTrue(bool((features[:, -1] >= 0).all()))

    def test_rdss_config_binds_raw_role_scale(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-078_rdss/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.rdss.v1")
        self.assertEqual(config["context_feature"], "raw_role_difference_std")
        self.assertEqual(config["semantic_neighbor_k"], 3)

    def test_staged_rdss_trains_only_scale_and_reproduces_base_delta(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        base_weight = torch.randn(12)
        base_bias = torch.randn(())
        model = StagedRoleDisagreementScaleSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(13), torch.ones(13), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
            base_selector_weight=base_weight,
            base_selector_bias=base_bias,
            base_feature_mean=torch.zeros(12),
            base_feature_std=torch.ones(12),
        )
        features = torch.randn(4, 13)
        expected = 0.5 * torch.tanh(features[:, :12] @ base_weight + base_bias)
        self.assertTrue(torch.allclose(model.pair_delta(features), expected))
        self.assertEqual(
            [name for name, _ in model.named_parameters()], ["scale_weight"]
        )

    def test_srdss_config_binds_frozen_snps_parent(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-079_srdss/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.srdss.v1")
        self.assertEqual(config["training_scope"], "freeze_snps_train_scale_only")
        self.assertIn("snps_model_sha256", config)

    def test_trust_region_rdss_initializes_parent_and_penalizes_drift(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        base_weight = torch.randn(12)
        base_bias = torch.randn(())
        model = TrustRegionRoleDisagreementScaleSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(13), torch.ones(13), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
            base_selector_weight=base_weight,
            base_selector_bias=base_bias,
            base_feature_mean=torch.zeros(12),
            base_feature_std=torch.ones(12),
        )
        features = torch.randn(4, 13)
        expected = 0.5 * torch.tanh(features[:, :12] @ base_weight + base_bias)
        self.assertTrue(torch.allclose(model.pair_delta(features), expected))
        self.assertEqual(float(model.trust_region_loss().detach()), 0.0)
        with torch.no_grad():
            model.selector_weight[0].add_(0.1)
        self.assertGreater(float(model.trust_region_loss().detach()), 0.0)

    def test_trdss_config_binds_trust_region(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-080_trdss/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.trdss.v1")
        self.assertEqual(config["training_scope"], "snps_initialized_joint_trust_region")
        self.assertEqual(config["trust_region_weight"], 0.1)

    def test_role_vote_selector_adds_bounded_signed_vote(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = RoleVotePairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(13), torch.ones(13), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        _, _, _, features = model._top2_context(
            torch.randn(3, 200), torch.randn(3, 768), None, torch.arange(200)
        )
        self.assertEqual(tuple(features.shape), (3, 13))
        self.assertTrue(bool((features[:, -1].abs() <= 1).all()))
        self.assertTrue(bool(((features[:, -1] * 4).round() == features[:, -1] * 4).all()))

    def test_rvps_config_binds_signed_vote(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-081_rvps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.rvps.v1")
        self.assertEqual(config["context_feature"], "signed_role_vote_mean")
        self.assertEqual(config["semantic_neighbor_k"], 3)

    def test_cross_source_selector_adds_absolute_gap(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = CrossSourceDisagreementSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(13), torch.ones(13), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        _, _, _, features = model._top2_context(
            torch.randn(3, 200), torch.randn(3, 768), None, torch.arange(200)
        )
        self.assertEqual(tuple(features.shape), (3, 13))
        self.assertTrue(torch.equal(
            features[:, -1], (features[:, 1] - features[:, 2]).abs()
        ))

    def test_csds_config_binds_absolute_source_gap(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-082_csds/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.csds.v1")
        self.assertEqual(
            config["context_feature"], "absolute_claude_merge_pair_gap"
        )
        self.assertEqual(config["semantic_neighbor_k"], 3)

    def test_role_uncertainty_gate_starts_at_parent_and_projects_gamma(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        base_weight = torch.randn(12)
        base_bias = torch.randn(())
        feature_mean = torch.cat((torch.zeros(12), torch.tensor([0.01])))
        model = RoleUncertaintyGatedSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, feature_mean, torch.ones(13), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
            base_selector_weight=base_weight,
            base_selector_bias=base_bias,
            base_feature_mean=torch.zeros(12),
            base_feature_std=torch.ones(12),
            max_gamma=1.0,
        )
        features = torch.randn(2, 13)
        features[:, 12] = torch.tensor([0.01, 0.02])
        expected = 0.5 * torch.tanh(features[:, :12] @ base_weight + base_bias)
        self.assertTrue(torch.allclose(model.pair_delta(features), expected))
        self.assertEqual([name for name, _ in model.named_parameters()], ["gamma"])
        with torch.no_grad():
            model.gamma.fill_(2.0)
        model.project_parameters()
        self.assertEqual(float(model.gamma.detach()), 1.0)
        attenuated = model.pair_delta(features).abs()
        self.assertLessEqual(
            float(attenuated[1].detach()), float(expected[1].abs())
        )

    def test_rugs_config_binds_multiplicative_uncertainty_gate(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-083_rugs/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.rugs.v1")
        self.assertEqual(config["training_scope"], "freeze_snps_train_gamma_only")
        self.assertEqual(config["max_gamma"], 1.0)

    def test_neighborhood_degree_selector_adds_log_degree_difference(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = NeighborhoodDegreePairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(13), torch.ones(13), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        _, global_ids, _, features = model._top2_context(
            torch.randn(3, 200), torch.randn(3, 768), None, torch.arange(200)
        )
        expected = model.semantic_log_degree[global_ids[:, 0]] - model.semantic_log_degree[
            global_ids[:, 1]
        ]
        self.assertEqual(tuple(features.shape), (3, 13))
        self.assertTrue(torch.equal(features[:, -1], expected))

    def test_ndps_config_binds_degree_difference(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-084_ndps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.ndps.v1")
        self.assertEqual(
            config["context_feature"], "semantic_log_degree_difference"
        )
        self.assertEqual(config["semantic_neighbor_k"], 3)

    def test_local_competition_resolver_starts_as_exact_parent(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = LocalSemanticCompetitionResolver(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768),
            torch.randn(200, 768), torch.randn(200, 8, 768),
            groups, adjacency, 0.25, 0.1,
            torch.zeros(11), torch.ones(11), 0.5,
        )
        images = torch.randn(2, 768)
        parent = torch.randn(2, 200)
        expected = parent + model.sdcr_beta * (
            torch.nn.functional.normalize(images, dim=-1)
            @ model.sdcr_prototypes.T
        )
        self.assertTrue(torch.equal(model(parent, images, None), expected))
        top, _, _, features = model.candidate_context(
            expected, images, torch.arange(200)
        )
        self.assertEqual(tuple(features.shape), (2, 3, 11))
        self.assertTrue(torch.equal(
            model.corrected_candidate_logits(top.values, features), top.values
        ))

    def test_extract_triplet_examples_builds_three_way_targets(self):
        logits = torch.tensor([
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ])
        package = extract_triplet_examples(
            logits,
            torch.randn(2, 768),
            torch.tensor([0, 2]),
            torch.arange(4),
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, 4, dtype=torch.bool).fill_diagonal_(False),
            torch.randn(4, 768), torch.randn(4, 768),
            torch.randn(4, 768), torch.randn(4, 8, 768),
            threshold=1.0,
        )
        self.assertEqual(tuple(package[0].shape), (2, 3))
        self.assertEqual(tuple(package[1].shape), (2, 3, 11))
        self.assertTrue(torch.equal(package[2], torch.tensor([0, 2])))

    def test_lscr_config_binds_three_way_training(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-085_lscr/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.lscr.v1")
        self.assertEqual(config["candidate_count"], 3)
        self.assertEqual(config["training_scope"], "related_top3_true_contained")

    def test_matched_hard_pairs_keep_all_errors_and_equal_correct(self):
        logits = torch.tensor([
            [1.1, 1.0], [2.0, 1.0], [1.2, 1.0],
            [3.0, 1.0], [1.3, 1.0], [4.0, 1.0],
        ])
        targets = torch.tensor([0, 1, 0, 0, 1, 0])
        selected, stats = matched_hard_pair_indices(logits, targets)
        self.assertTrue(torch.equal(selected, torch.tensor([0, 1, 2, 4])))
        self.assertEqual(stats["error_count"], 2)
        self.assertEqual(stats["matched_correct_count"], 2)

    def test_mhps_config_binds_matched_sampling(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-087_mhps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.mhps.v1")
        self.assertEqual(
            config["pair_sampling"],
            "all_errors_plus_equal_lowest_margin_correct",
        )

    def test_focal_pair_loss_downweights_easy_correct_pair(self):
        losses = focal_pair_losses(
            torch.tensor([[5.0, 0.0], [0.1, 0.0]]),
            torch.tensor([0, 0]),
            gamma=2.0,
        )
        self.assertLess(float(losses[0]), float(losses[1]))
        self.assertTrue(bool(torch.isfinite(losses).all()))

    def test_fbps_config_binds_focal_objective(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-088_fbps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.fbps.v1")
        self.assertEqual(config["training_objective"], "focal_pair_ce")
        self.assertEqual(config["focal_gamma"], 2.0)

    def test_bias_free_selector_has_no_trainable_bias(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = BiasFreeSemanticNeighborSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(12), torch.ones(12), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        self.assertEqual(float(model.selector_bias), 0.0)
        self.assertEqual(
            [name for name, _ in model.named_parameters()], ["selector_weight"]
        )

    def test_bfps_config_binds_zero_bias(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-089_bfps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.bfps.v1")
        self.assertEqual(config["selector_bias_mode"], "fixed_zero")
        self.assertEqual(config["semantic_neighbor_k"], 3)

    def test_antisymmetric_augmentation_is_exact_mirror(self):
        logits = torch.tensor([[2.0, 1.0], [3.0, 0.5]])
        features = torch.randn(2, 12)
        targets = torch.tensor([0, 1])
        weights = torch.tensor([0.2, 0.8])
        augmented = antisymmetric_pair_augmentation(
            logits, features, targets, weights
        )
        self.assertTrue(torch.equal(augmented[0][2:], logits.flip(1)))
        self.assertTrue(torch.equal(augmented[1][2:], -features))
        self.assertTrue(torch.equal(augmented[2], torch.tensor([0, 1, 1, 0])))
        self.assertTrue(torch.equal(augmented[3], torch.tensor([0.2, 0.8, 0.2, 0.8])))

    def test_antisymmetric_selector_is_swap_equivariant(self):
        groups = torch.arange(200) // 2
        adjacency = semantic_neighbor_adjacency(torch.randn(200, 32), 3)
        model = AntisymmetricPairSelector(
            torch.randn(200, 768), 13.0,
            torch.randn(200, 768), torch.randn(200, 768), groups,
            0.25, 0.1, torch.zeros(12), torch.ones(12), 0.5,
            class_name_prototypes=torch.randn(200, 768),
            role_sentence_prototypes=torch.randn(200, 8, 768),
            semantic_adjacency=adjacency,
        )
        with torch.no_grad():
            model.selector_weight.copy_(torch.randn(12))
        logits = torch.tensor([[2.0, 1.0]])
        features = torch.randn(1, 12)
        original = model.corrected_pair_logits(logits, features)
        mirrored = model.corrected_pair_logits(logits.flip(1), -features).flip(1)
        self.assertTrue(torch.allclose(original, mirrored))

    def test_aps_config_binds_mirror_training(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-090_aps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.aps.v1")
        self.assertEqual(config["pair_augmentation"], "swap_and_negate")
        self.assertEqual(config["gate_margin_mode"], "absolute")

    def test_true_class_balancing_weights_have_mean_one(self):
        weights, stats = true_class_balancing_weights(
            torch.tensor([0, 0, 1, 2, 2, 2])
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertEqual(stats["present_class_count"], 3.0)
        self.assertGreater(stats["max_weight"], stats["min_weight"])

    def test_cups_config_binds_true_class_balance(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-091_cups/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.cups.v1")
        self.assertEqual(
            config["true_class_balance"], "inverse_pair_frequency_mean1"
        )

    def test_teacher_forced_pairs_use_true_class_for_errors(self):
        logits = torch.tensor([
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ])
        package = extract_teacher_forced_pairs(
            logits,
            torch.randn(2, 768),
            torch.tensor([0, 2]),
            torch.arange(4),
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, 4, dtype=torch.bool).fill_diagonal_(False),
            torch.randn(4, 768), torch.randn(4, 768),
            torch.randn(4, 768), torch.randn(4, 8, 768),
            threshold=1.0,
            error_weight_floor=0.25,
        )
        self.assertTrue(torch.equal(package[2], torch.tensor([0, 1])))
        self.assertEqual(float(package[0][1, 1]), 1.0)
        self.assertGreaterEqual(float(package[4][1]), 0.25)
        self.assertEqual(tuple(package[1].shape), (2, 12))

    def test_tfps_config_binds_teacher_forced_errors(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-092_tfps/configs/RUN-001.yaml"
        )
        self.assertEqual(config["schema_version"], "gzsl-paper.tfps.v1")
        self.assertEqual(
            config["training_scope"],
            "wrong_top1_vs_true_correct_top1_vs_top2",
        )
        self.assertEqual(config["error_weight_floor"], 0.25)


if __name__ == "__main__":
    unittest.main()
