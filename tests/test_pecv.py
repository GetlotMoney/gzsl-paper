import torch

from model.innovations.pecv import (
    PairwiseErrorCorrectingVerifier,
    corrected_topk_scores,
)
from model.innovations.train_pecv_gate import (
    _stable_topk_local,
    _truth_injected_train_candidates,
)


def _inputs(batch=4, classes=6, roles=8, dim=12):
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(batch, dim, generator=generator)
    prototypes = torch.randn(classes, dim, generator=generator)
    role_text = torch.randn(classes, roles, dim, generator=generator)
    return images, prototypes, role_text


def test_pair_correction_is_antisymmetric():
    images, prototypes, role_text = _inputs()
    model = PairwiseErrorCorrectingVerifier(role_count=8, hidden_dim=16)
    a = torch.tensor([0, 1, 2, 3])
    b = torch.tensor([1, 2, 3, 4])
    ab = model.correction(
        images,
        prototypes[a],
        prototypes[b],
        role_text[a],
        role_text[b],
    )
    ba = model.correction(
        images,
        prototypes[b],
        prototypes[a],
        role_text[b],
        role_text[a],
    )
    torch.testing.assert_close(ab, -ba, rtol=0, atol=1e-6)


def test_module_off_returns_parent_tensor_exactly():
    images, prototypes, role_text = _inputs()
    candidates = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    parent = torch.randn(4, 3, generator=torch.Generator().manual_seed(9))
    output = corrected_topk_scores(
        parent, images, candidates, prototypes, role_text, verifier=None
    )
    assert output is parent
    assert torch.equal(output, parent)


def test_pair_residual_is_zero_sum():
    images, prototypes, role_text = _inputs()
    candidates = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    parent = torch.randn(4, 3, generator=torch.Generator().manual_seed(9))
    model = PairwiseErrorCorrectingVerifier(role_count=8, hidden_dim=16)
    output = corrected_topk_scores(
        parent, images, candidates, prototypes, role_text, verifier=model
    )
    torch.testing.assert_close(
        (output - parent).sum(dim=1), torch.zeros(4), rtol=0, atol=1e-6
    )


def test_stable_topk_breaks_ties_by_global_id():
    logits = torch.tensor([[1.0, 1.0, 0.0]])
    local_to_global = torch.tensor([20, 10, 30])
    ranked = _stable_topk_local(
        logits, torch.arange(3), local_to_global, top_k=3
    )
    assert ranked.tolist() == [[1, 0, 2]]


def test_truth_injected_candidates_use_four_strongest_wrong_classes():
    logits = torch.tensor([[0.4, 0.9, 0.8, 0.7, 0.6, 0.5]])
    truth = torch.tensor([0])
    candidates = _truth_injected_train_candidates(
        logits,
        truth,
        torch.arange(6),
        torch.arange(100, 106),
    )
    assert candidates.tolist() == [[0, 1, 2, 3, 4]]
