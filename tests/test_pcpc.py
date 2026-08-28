from __future__ import annotations

import pytest
import torch

from model.innovations.pcpc import (
    PairContrastPatchComparator,
    pairwise_hard_negative_loss,
)


def _inputs():
    generator = torch.Generator().manual_seed(20260828)
    logits = torch.randn(4, 10, generator=generator)
    patches = torch.randn(4, 6, 768, generator=generator)
    role_text = torch.randn(10, 8, 768, generator=generator)
    return logits, patches, role_text


def test_pcpc_shape_finite_and_exact_off_path():
    logits, patches, role_text = _inputs()
    model = PairContrastPatchComparator(rank=16)
    off = model(logits, patches, role_text, enabled=False)
    assert off is logits
    corrected, diagnostics = model(
        logits, patches, role_text, enabled=True, return_diagnostics=True
    )
    assert corrected.shape == logits.shape
    assert diagnostics["signed_patch_role"].shape == (4, 6, 8)
    assert diagnostics["patch_weights"].shape == (4, 6, 8)
    assert diagnostics["delta"].shape == (4,)
    assert torch.isfinite(corrected).all()
    assert torch.isfinite(diagnostics["delta"]).all()
    # Zero-initialized bounded strength is also neutral before learning.
    assert torch.equal(corrected, logits)


def test_pcpc_changes_only_top2_and_is_zero_sum():
    logits, patches, role_text = _inputs()
    model = PairContrastPatchComparator(rank=16)
    with torch.no_grad():
        model.raw_strength.fill_(0.5)
    corrected, diagnostics = model(
        logits, patches, role_text, return_diagnostics=True
    )
    correction = corrected - logits
    candidates = diagnostics["candidate_ids"]
    candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
    candidate_mask.scatter_(1, candidates, True)
    assert torch.count_nonzero(correction.masked_select(~candidate_mask)) == 0
    assert torch.allclose(correction.sum(dim=1), torch.zeros(4), atol=1e-7, rtol=0.0)
    gathered = correction.gather(1, candidates)
    assert torch.allclose(gathered[:, 0], -gathered[:, 1], atol=1e-7, rtol=0.0)


def test_swapping_candidates_negates_same_patch_evidence():
    _, patches, role_text = _inputs()
    model = PairContrastPatchComparator(rank=16)
    candidates = torch.tensor([[1, 3], [2, 7], [4, 0], [8, 5]])
    forward = model.pair_evidence(patches, role_text, candidates)
    swapped = model.pair_evidence(patches, role_text, candidates.flip(1))
    assert torch.allclose(swapped["delta"], -forward["delta"], atol=1e-7, rtol=0.0)
    assert torch.allclose(
        swapped["signed_patch_role"], -forward["signed_patch_role"],
        atol=1e-7, rtol=0.0,
    )
    assert torch.allclose(swapped["patch_weights"], forward["patch_weights"])


def test_pcpc_shared_parameters_and_seen_only_pairwise_loss_have_gradients():
    logits, patches, role_text = _inputs()
    model = PairContrastPatchComparator(rank=16)
    with torch.no_grad():
        model.raw_strength.fill_(0.25)
    corrected = model(logits, patches, role_text)
    targets = torch.tensor([0, 1, 2, 3])
    loss = pairwise_hard_negative_loss(
        corrected, targets, torch.arange(6), margin=0.02
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.raw_strength.grad is not None
    assert float(model.raw_strength.grad.abs()) > 0.0
    assert model.visual_projection.weight.grad is not None
    assert model.text_projection.weight.grad is not None
    assert float(model.visual_projection.weight.grad.norm()) > 0.0
    assert float(model.text_projection.weight.grad.norm()) > 0.0
    parameter_names = set(dict(model.named_parameters()))
    assert parameter_names == {
        "raw_strength", "visual_projection.weight", "text_projection.weight"
    }

    with pytest.raises(ValueError, match="seen"):
        pairwise_hard_negative_loss(
            corrected.detach(), torch.tensor([0, 1, 2, 9]), torch.arange(6)
        )


def test_pcpc_rejects_invalid_shapes_and_nonfinite_inputs():
    logits, patches, role_text = _inputs()
    model = PairContrastPatchComparator(rank=16)
    with pytest.raises(ValueError, match="patches"):
        model(logits, patches[..., :32], role_text)
    broken = logits.clone()
    broken[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        model(broken, patches, role_text)
