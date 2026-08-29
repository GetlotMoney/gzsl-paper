from __future__ import annotations

import numpy as np
import torch

from tools.diagnose_tristate_predicates import (
    PredicateReader,
    correction_metrics,
    nearest_same_role,
    predicate_contribution,
    shuffled_failure_gates,
    shuffled_query_map,
    split_classes,
)


def test_reader_returns_shared_query_scores_and_attention():
    reader = PredicateReader(rank=8)
    patches = torch.randn(2, 5, 768)
    predicates = torch.randn(3, 768)
    logits, attention = reader.evidence(patches, predicates, return_attention=True)
    assert logits.shape == (2, 3)
    assert attention.shape == (2, 5, 3)
    assert torch.allclose(attention.sum(dim=1), torch.ones(2, 3), atol=1e-6)


def test_split_and_shuffled_map_are_deterministic_and_disjoint():
    labels = np.repeat(np.arange(150), 2)
    train_a, evaluation_a = split_classes(labels, seed=7, train_count=100)
    train_b, evaluation_b = split_classes(labels, seed=7, train_count=100)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(evaluation_a, evaluation_b)
    assert not set(train_a) & set(evaluation_a)
    mapping = shuffled_query_map(train_a, seed=7, enabled=True)
    assert set(mapping) == set(train_a.tolist())
    assert set(mapping.values()) == set(train_a.tolist())
    assert all(source != target for source, target in mapping.items())


def test_nearest_same_role_never_returns_the_query_class():
    predicates = torch.randn(8, 6, 768)
    classes = np.arange(8)
    neighbors = nearest_same_role(predicates, classes, count=3)
    assert neighbors.shape == (8, 6, 3)
    for local, class_id in enumerate(classes):
        assert class_id not in neighbors[local]


def test_fixed_one_vs_k_tristate_does_not_depend_on_total_candidate_count():
    assert predicate_contribution(torch.tensor([0.8, 0.3]), 0.5) == (1, 0.5)
    assert predicate_contribution(torch.tensor([0.8, 0.3, 0.2, 0.1]), 0.5) == (1, 0.5)
    state, contribution = predicate_contribution(torch.tensor([0.2, 0.9, 0.1]), 0.5)
    assert state == -1 and contribution < 0
    assert predicate_contribution(torch.tensor([0.2, 0.3, 0.1]), 0.5) == (0, 0.0)


def test_all_unknown_evidence_never_damages_a_correct_parent_by_argmax_tie():
    evidence = np.zeros((2, 2), dtype=np.float32)
    labels = np.asarray([0, 1])
    parent = labels.copy()
    true_local = np.asarray([0, 1])
    metrics = correction_metrics(
        evidence,
        labels,
        np.asarray([0, 1]),
        parent,
        true_local,
        np.asarray([True, True]),
    )
    assert metrics["eligible_parent_error_count"] == 0
    assert metrics["correct_sample_evidence_reversal_fraction"] == 0.0


def test_shuffled_control_must_fail_every_main_non_deletion_gate():
    config = {
        "pairwise_accuracy_gate": 0.65,
        "error_correction_gate": 0.60,
        "correct_damage_gate": 0.10,
    }
    shuffled = {
        "pairwise_hard_negative_accuracy": 0.70,
        "mean8_and_evidence": {
            "error_pair_true_preferred_fraction": 0.50,
            "correct_sample_evidence_reversal_fraction": 0.20,
        },
    }
    gates = shuffled_failure_gates(shuffled, config)
    assert gates["shuffled_pairwise_failed"] is False
    assert gates["shuffled_error_correction_failed"] is True
    assert gates["shuffled_correct_damage_failed"] is True
    assert not all(gates.values())
