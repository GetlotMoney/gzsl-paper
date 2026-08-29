from __future__ import annotations

import numpy as np
import torch

from tools.diagnose_observable_signed_evidence import (
    SharedEvidenceReader,
    detached_observability_loss,
    derangement,
    fixed_reference_d,
    region_bounds,
    state_probabilities,
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


def test_derangement_has_no_fixed_class():
    values = np.arange(100)
    output = derangement(values, seed=7)
    assert set(output.tolist()) == set(values.tolist())
    assert not np.any(output == values)


def test_classification_loss_cannot_change_observability():
    role_losses = torch.tensor([1.0, 2.0], requires_grad=True)
    observability = torch.tensor([0.2, 0.8], requires_grad=True)
    loss = detached_observability_loss(role_losses, observability)
    loss.backward()
    assert observability.grad is None
    assert role_losses.grad is not None


def test_region_is_fixed_area_and_inside_336_pixels():
    for index in (0, 23, 24 * 12 + 12, 575):
        top, bottom, left, right = region_bounds(index, patch_side=4)
        assert 0 <= top < bottom <= 336
        assert 0 <= left < right <= 336
        assert bottom - top == 56
        assert right - left == 56
