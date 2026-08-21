from __future__ import annotations

import torch
import unittest

from model.tg_vpr_h1 import TGVPRH1FixedEqual
from model.tries.v2_try_001 import TGVPRH1UnseenValueTransfer


def _models():
    generator = torch.Generator().manual_seed(812)
    sentences = torch.randn(200, 8, 768, generator=generator)
    seenclasses = torch.tensor([i for i in range(200) if i % 4 != 0])
    centroids = torch.randn(150, 768, generator=generator)
    baseline = TGVPRH1FixedEqual(sentences, seenclasses, centroids, dropout=0.0)
    candidate = TGVPRH1UnseenValueTransfer(
        sentences, seenclasses, centroids, dropout=0.0
    )
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    return baseline.eval(), candidate.eval(), seenclasses


def test_try_has_identical_state_identity_and_seen_prototypes():
    baseline, candidate, seenclasses = _models()
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    baseline_proto = baseline.prototypes()
    candidate_proto = candidate.prototypes()
    assert torch.equal(
        baseline_proto.index_select(0, seenclasses),
        candidate_proto.index_select(0, seenclasses),
    )


def test_try_changes_only_unseen_prototypes_and_stays_finite():
    baseline, candidate, seenclasses = _models()
    allclasses = torch.arange(200)
    unseen = allclasses[~torch.isin(allclasses, seenclasses)]
    baseline_proto = baseline.prototypes()
    candidate_proto = candidate.prototypes()
    assert not torch.equal(
        baseline_proto.index_select(0, unseen),
        candidate_proto.index_select(0, unseen),
    )
    assert torch.isfinite(candidate_proto).all()
    assert torch.allclose(candidate_proto.norm(dim=-1), torch.ones(200), atol=1e-6)


def test_constrained_transfer_is_closer_to_baseline_than_full_transfer():
    baseline, full, seenclasses = _models()
    constrained = TGVPRH1UnseenValueTransfer(
        baseline.sentence_embeds,
        seenclasses,
        baseline.visual_centroids,
        dropout=0.0,
        transfer_strength=0.1,
    ).eval()
    constrained.load_state_dict(baseline.state_dict(), strict=True)
    allclasses = torch.arange(200)
    unseen = allclasses[~torch.isin(allclasses, seenclasses)]
    base_u = baseline.prototypes().index_select(0, unseen)
    full_u = full.prototypes().index_select(0, unseen)
    constrained_u = constrained.prototypes().index_select(0, unseen)
    assert torch.equal(
        constrained.prototypes().index_select(0, seenclasses),
        baseline.prototypes().index_select(0, seenclasses),
    )
    assert (constrained_u - base_u).norm() < (full_u - base_u).norm()


class V2Try001Test(unittest.TestCase):
    def test_seen_identity(self):
        test_try_has_identical_state_identity_and_seen_prototypes()

    def test_unseen_change(self):
        test_try_changes_only_unseen_prototypes_and_stays_finite()

    def test_constrained_transfer(self):
        test_constrained_transfer_is_closer_to_baseline_than_full_transfer()


if __name__ == "__main__":
    unittest.main()
