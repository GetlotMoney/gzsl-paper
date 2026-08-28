from __future__ import annotations

import torch
import torch.nn.functional as F

from model.innovations.gave import GeodesicAlignedVisualEvidence, geodesic_tangent


def _basis(index: int) -> torch.Tensor:
    value = torch.zeros(768)
    value[index] = 1.0
    return value


def _inputs():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    roles = torch.zeros(4, 8, 768)
    roles[:, :, 2] = 1.0
    mean8 = torch.stack((_basis(0), _basis(0), _basis(3), _basis(4)))
    value = torch.stack(
        (
            F.normalize(_basis(0) + 0.4 * _basis(1), dim=0),
            F.normalize(_basis(0) - 0.4 * _basis(1), dim=0),
            _basis(3),
            _basis(4),
        )
    )
    patches = torch.stack(
        (
            F.normalize(_basis(2) + 0.5 * _basis(1), dim=0),
            F.normalize(_basis(2) + 0.4 * _basis(1), dim=0),
            F.normalize(_basis(5) - 0.2 * _basis(1), dim=0),
        )
    ).unsqueeze(0)
    return logits, patches, roles, mean8, value


def test_geodesic_tangent_and_degenerate_direction():
    base = torch.stack((_basis(0), _basis(2)))
    value = torch.stack((F.normalize(_basis(0) + _basis(1), dim=0), _basis(2)))
    direction, valid = geodesic_tangent(base, value)
    assert valid.tolist() == [True, False]
    assert torch.allclose(direction[0], _basis(1), atol=1e-6)
    assert torch.equal(direction[1], torch.zeros(768))


def test_supportive_patches_favor_matching_geodesic_direction():
    logits, patches, roles, mean8, value = _inputs()
    model = GeodesicAlignedVisualEvidence(top_candidates=2, visible_roles=3)
    parts = model.components(logits, patches, roles, mean8, value)
    assert parts["candidate_ids"].tolist() == [[0, 1]]
    assert float(parts["coverage"][0, 0]) > 0.0
    assert float(parts["coverage"][0, 1]) < 0.0
    assert float(parts["relative_evidence"].sum()) == 0.0


def test_overall_role_is_not_a_patch_query():
    logits, patches, roles, mean8, value = _inputs()
    model = GeodesicAlignedVisualEvidence(top_candidates=2, visible_roles=3)
    before = model.components(logits, patches, roles, mean8, value)
    changed = roles.clone()
    changed[:, 6] = torch.randn(4, 768, generator=torch.Generator().manual_seed(9))
    after = model.components(logits, patches, changed, mean8, value)
    assert torch.equal(before["role_evidence"], after["role_evidence"])
    assert torch.equal(before["relative_evidence"], after["relative_evidence"])


def test_zero_strength_and_disabled_paths_exactly_reproduce_parent():
    logits, patches, roles, mean8, value = _inputs()
    model = GeodesicAlignedVisualEvidence(top_candidates=2, visible_roles=3)
    assert torch.equal(model(logits, patches, roles, mean8, value), logits)
    assert torch.equal(
        model(logits, patches, roles, mean8, value, enabled=False), logits
    )
    with torch.no_grad():
        model.raw_strength.fill_(0.5)
    corrected, diagnostics = model(
        logits, patches, roles, mean8, value, return_diagnostics=True
    )
    assert not torch.equal(corrected[:, :2], logits[:, :2])
    assert torch.equal(corrected[:, 2:], logits[:, 2:])
    assert torch.allclose(diagnostics["candidate_delta"].sum(dim=1), torch.zeros(1))


def test_degenerate_candidate_has_zero_visual_evidence():
    logits, patches, roles, mean8, value = _inputs()
    model = GeodesicAlignedVisualEvidence(top_candidates=2, visible_roles=3)
    parts = model.components(
        logits,
        patches,
        roles,
        mean8,
        value,
        candidate_ids=torch.tensor([[2, 3]]),
    )
    assert not bool(parts["valid_direction"].any())
    assert torch.equal(parts["coverage"], torch.zeros_like(parts["coverage"]))
