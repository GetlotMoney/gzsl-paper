from __future__ import annotations

import numpy as np
import torch

from tools.diagnose_tristate_predicates import (
    PredicateReader,
    correction_metrics,
    evidence_scores,
    nearest_same_role,
    shuffled_query_map,
    split_classes,
    visibility_thresholds,
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


def test_nearest_same_role_never_returns_the_query_class():
    predicates = torch.randn(8, 6, 768)
    classes = np.arange(8)
    neighbors = nearest_same_role(predicates, classes, count=3)
    assert neighbors.shape == (8, 6, 3)
    for local, class_id in enumerate(classes):
        assert class_id not in neighbors[local]


def test_visibility_and_correction_contracts():
    train_scores = np.asarray(
        [
            [[0.9, 0.2], [0.1, 0.3]],
            [[0.8, 0.1], [0.2, 0.4]],
        ],
        dtype=np.float32,
    )
    thresholds = visibility_thresholds(train_scores, 0.10)
    evidence, visible = evidence_scores(train_scores, thresholds)
    assert evidence.shape == (2, 2)
    assert visible.shape == (2, 2)
    labels = np.asarray([0, 1])
    parent = np.asarray([1, 1])
    true_local = np.asarray([0, 1])
    metrics = correction_metrics(
        evidence,
        labels,
        np.asarray([0, 1]),
        parent,
        true_local,
        np.asarray([True, True]),
    )
    assert metrics["eligible_parent_error_count"] == 1
    assert 0.0 <= metrics["error_pair_true_preferred_fraction"] <= 1.0
