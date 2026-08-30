import torch

from model.innovations.pecv import (
    PairwiseErrorCorrectingVerifier,
    corrected_topk_scores,
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
