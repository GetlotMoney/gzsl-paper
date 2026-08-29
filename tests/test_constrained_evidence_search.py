from __future__ import annotations

import numpy as np

from tools.diagnose_constrained_evidence_search import (
    duplicate_assignment,
    exact_assignment,
    fast_assignment,
    independent_assignment,
    pool_regions,
)


def test_capacity_one_prevents_duplicate_patch_use():
    edges = np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    independent_score, independent_path = independent_assignment(edges)
    dp_score, dp_path, _ = exact_assignment(edges, capacity=1, top_r=False)
    assert independent_score == np.float32(1.7)
    assert duplicate_assignment(independent_path)
    assert not duplicate_assignment(dp_path)
    assert np.isclose(dp_score, 1.1)


def test_capacity_two_recovers_legitimate_overlap():
    edges = np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    dp_score, dp_path, _ = exact_assignment(edges, capacity=2, top_r=False)
    assert np.isclose(dp_score, 1.7)
    assert duplicate_assignment(dp_path)


def test_unknown_skips_negative_edges():
    edges = -np.ones((3, 5), dtype=np.float32)
    score, path, _ = exact_assignment(edges, capacity=1, top_r=False)
    assert score == 0.0
    assert path == {}


def test_top_r_is_exact_for_six_role_capacity_one_and_two():
    rng = np.random.default_rng(7)
    for capacity in (1, 2):
        for _ in range(20):
            edges = rng.normal(size=(6, 30)).astype(np.float32)
            top_score, _, _ = exact_assignment(edges, capacity=capacity, top_r=True)
            full_score, _, _ = exact_assignment(edges, capacity=capacity, top_r=False)
            assert np.isclose(top_score, full_score)
            fast_score, _, _ = fast_assignment(edges, capacity=capacity)
            assert np.isclose(fast_score, full_score)


def test_region_pooling_is_two_by_two_mean():
    edges = np.arange(576, dtype=np.float32).reshape(1, 576)
    pooled = pool_regions(edges)
    assert pooled.shape == (1, 144)
    assert np.isclose(pooled[0, 0], np.mean([0, 1, 24, 25]))
