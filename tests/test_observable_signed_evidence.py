from __future__ import annotations

import numpy as np
import torch

from tools.diagnose_observable_signed_evidence import (
    SharedEvidenceReader,
    derangement,
    fixed_reference_d,
    region_bounds,
    regions_overlap,
    sampling_sha256,
    shuffled_query_bank,
    state_loss,
    state_probabilities,
    intervention_trace_sha256,
    minimum_patch_identity,
)


def test_one_reader_returns_scores_and_attention():
    reader = SharedEvidenceReader(rank=8)
    scores, attention = reader.evidence(
        torch.randn(2, 7, 768), torch.randn(5, 768), attention=True
    )
    assert scores.shape == (2, 5)
    assert attention.shape == (2, 7, 5)
    assert torch.allclose(attention.sum(dim=1), torch.ones(2, 5), atol=1e-6)


def test_three_probabilities_sum_to_one_and_contribution_is_expectation():
    observability = torch.tensor([0.2, 0.8])
    signed = torch.tensor([-1.5, 2.0])
    unknown, support, refute, contribution = state_probabilities(observability, signed)
    assert torch.allclose(unknown + support + refute, torch.ones(2), atol=1e-6)
    assert torch.allclose(support - refute, contribution, atol=1e-6)


def test_fixed_reference_is_invariant_to_class_axis_permutation():
    scores = torch.randn(3, 200, 6)
    original = fixed_reference_d(scores)
    permutation = torch.randperm(200)
    restored = fixed_reference_d(scores[:, permutation])[:, torch.argsort(permutation)]
    assert torch.allclose(original, restored, atol=1e-6)


def test_fixed_reference_is_stable_and_keeps_candidate_gradient_at_large_gap():
    scores = torch.zeros(1, 200, 6, requires_grad=True)
    with torch.no_grad():
        scores[0, 0, 0] = 100.0
    signed = fixed_reference_d(scores)
    assert torch.allclose(signed[0, 0, 0], torch.tensor(100.0), atol=1e-5)
    signed[0, 0, 0].backward()
    assert torch.allclose(scores.grad[0, 0, 0], torch.tensor(1.0), atol=1e-6)


def test_derangement_has_no_fixed_class():
    values = np.arange(100)
    output = derangement(values, seed=7)
    assert set(output.tolist()) == set(values.tolist())
    assert not np.any(output == values)


def test_classification_loss_cannot_change_observability():
    reader = SharedEvidenceReader(rank=8)
    patches = torch.randn(2, 4, 768)
    class_queries = torch.randn(200, 6, 768)
    role_queries = torch.nn.functional.normalize(class_queries.mean(dim=0), dim=-1)
    before = reader.observability(patches, role_queries).detach().clone()
    loss = state_loss(
        reader,
        patches,
        0,
        class_queries,
        role_queries,
        np.arange(100),
        0.20,
    )
    optimizer = torch.optim.SGD(reader.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = reader.observability(patches, role_queries).detach()
    assert torch.equal(before, after)


def test_shuffled_query_control_changes_every_text_slot():
    queries = torch.randn(200, 6, 768)
    shuffled, _ = shuffled_query_bank(queries, seed=7)
    equal = torch.isclose(queries, shuffled).all(dim=-1)
    assert not bool(equal.any())


def test_region_is_fixed_area_and_inside_336_pixels():
    for index in (0, 23, 24 * 12 + 12, 575):
        top, bottom, left, right = region_bounds(index, patch_side=4)
        assert 0 <= top < bottom <= 336
        assert 0 <= left < right <= 336
        assert bottom - top == 56
        assert right - left == 56


def test_region_overlap_detection_is_explicit_for_audit_output():
    assert regions_overlap(0, 1, patch_side=4)
    assert not regions_overlap(0, 24 * 12 + 12, patch_side=4)


def test_sampling_identity_binds_random_but_allows_text_dependent_selected_regions():
    cache = {
        "rows": np.asarray([1, 2]),
        "classes": np.asarray([3, 4]),
        "roles": np.asarray([0, 1]),
        "random_indices": np.asarray([10, 20]),
        "selected_o_indices": np.asarray([30, 40]),
        "selected_d_indices": np.asarray([50, 60]),
    }
    changed_selected = {key: value.copy() for key, value in cache.items()}
    changed_selected["selected_o_indices"][0] = 31
    assert sampling_sha256(cache) == sampling_sha256(changed_selected)
    assert intervention_trace_sha256(cache) != intervention_trace_sha256(changed_selected)
    changed_random = {key: value.copy() for key, value in cache.items()}
    changed_random["random_indices"][0] = 11
    assert sampling_sha256(cache) != sampling_sha256(changed_random)


def test_patch_identity_gate_uses_real_and_shuffled_train_and_eval_paths():
    def result(train_mean, train_minimum, eval_mean, eval_minimum):
        return {
            "causal_train_identity": {
                "mean_patch_cosine": train_mean,
                "minimum_image_mean_patch_cosine": train_minimum,
            },
            "causal_eval_identity": {
                "mean_patch_cosine": eval_mean,
                "minimum_image_mean_patch_cosine": eval_minimum,
            },
        }

    real = result(1.0, 0.999, 0.998, 0.997)
    shuffled = result(0.996, 0.995, 0.994, 0.80)
    assert minimum_patch_identity(real, shuffled) == 0.80
