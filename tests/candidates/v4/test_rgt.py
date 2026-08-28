from __future__ import annotations

import torch
import torch.nn.functional as F

from model.candidates.v4.idea_159_rgt.module import RefutationGatedTransport


def _basis(index: int) -> torch.Tensor:
    value = torch.zeros(768)
    value[index] = 1.0
    return value


def _inputs(refuting: bool):
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    roles = torch.zeros(3, 8, 768)
    roles[:, :, 2] = 1.0
    mean8 = torch.stack((_basis(0), _basis(0), _basis(3)))
    direction = torch.stack((_basis(1), -_basis(1), torch.zeros(768)))
    theta = torch.tensor([0.4, 0.4, 0.0])
    sign = -1.0 if refuting else 1.0
    patches = torch.stack(
        (
            F.normalize(_basis(2) + sign * 0.5 * _basis(1), dim=0),
            F.normalize(_basis(2) + sign * 0.4 * _basis(1), dim=0),
            _basis(5),
        )
    ).unsqueeze(0)
    prototypes = mean8.clone()
    prototypes[:2] = F.normalize(
        torch.cos(theta[:2]).unsqueeze(-1) * mean8[:2]
        + torch.sin(theta[:2]).unsqueeze(-1) * direction[:2],
        dim=-1,
    )
    images = _basis(0).unsqueeze(0)
    return logits, images, patches, roles, mean8, direction, theta, prototypes


def test_refuting_patches_produce_transport_attenuation():
    logits, _, patches, roles, mean8, direction, theta, _ = _inputs(True)
    model = RefutationGatedTransport(top_candidates=2, visible_roles=3)
    parts = model.refutation_components(logits, patches, roles, mean8, direction, theta)
    assert float(parts["refutation_ratio"][0, 0]) > 0.0
    assert float(parts["support"][0, 0]) == 0.0


def test_supportive_patches_do_not_attenuate_candidate():
    logits, _, patches, roles, mean8, direction, theta, _ = _inputs(False)
    model = RefutationGatedTransport(top_candidates=2, visible_roles=3)
    parts = model.refutation_components(logits, patches, roles, mean8, direction, theta)
    assert float(parts["refutation_ratio"][0, 0]) == 0.0
    assert float(parts["support"][0, 0]) > 0.0


def test_zero_strength_exactly_reproduces_parent_logits():
    logits, images, patches, roles, mean8, direction, theta, prototypes = _inputs(True)
    model = RefutationGatedTransport(top_candidates=2, visible_roles=3)
    parts = model.refutation_components(logits, patches, roles, mean8, direction, theta)
    corrected, adjusted_theta = model.attenuated_logits(
        logits, images, prototypes, mean8, direction, theta, torch.tensor(1.0),
        parts, strength=0.0,
    )
    assert torch.equal(corrected, logits)
    assert torch.equal(adjusted_theta, parts["candidate_theta"])


def test_full_refutation_moves_candidate_back_toward_mean8():
    logits, images, patches, roles, mean8, direction, theta, prototypes = _inputs(True)
    model = RefutationGatedTransport(top_candidates=2, visible_roles=3)
    parts = model.refutation_components(logits, patches, roles, mean8, direction, theta)
    _, adjusted_theta = model.attenuated_logits(
        logits, images, prototypes, mean8, direction, theta, torch.tensor(1.0),
        parts, strength=1.0,
    )
    assert float(adjusted_theta[0, 0]) < float(parts["candidate_theta"][0, 0])


def test_overall_role_and_seen_theta_cannot_change_refutation():
    logits, _, patches, roles, mean8, direction, theta, _ = _inputs(True)
    model = RefutationGatedTransport(top_candidates=3, visible_roles=3)
    before = model.refutation_components(logits, patches, roles, mean8, direction, theta)
    changed = roles.clone()
    changed[:, 6] = torch.randn(3, 768, generator=torch.Generator().manual_seed(4))
    after = model.refutation_components(logits, patches, changed, mean8, direction, theta)
    assert torch.equal(before["refutation_ratio"], after["refutation_ratio"])
    seen_position = before["candidate_ids"].eq(2)
    assert torch.equal(
        before["refutation_ratio"].masked_select(seen_position), torch.zeros(1)
    )
