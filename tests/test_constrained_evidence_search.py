from __future__ import annotations

import numpy as np

from tools.diagnose_constrained_evidence_search import (
    duplicate_assignment,
    exact_assignment,
    fast_assignment,
    independent_assignment,
    pool_regions,
    top_r_equivalence,
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


def test_real_oracle_binds_each_production_solver_mode():
    rng = np.random.default_rng(3)
    edges = rng.normal(size=(1, 3, 576)).astype(np.float32)
    labels = np.asarray([7])
    mapping = {7: [(0, 0), (1, 1), (2, 2)]}
    modes = ("patch_capacity1", "patch_capacity2", "region_capacity1")
    result = top_r_equivalence(edges, labels, mapping, modes, count=1)
    assert set(result) == set(modes)
    for mode in modes:
        assert result[mode]["checked"] == 1
        assert result[mode]["production_vs_full_maximum_abs"] <= 1e-6
        assert result[mode]["top_r_dp_vs_full_maximum_abs"] <= 1e-6
