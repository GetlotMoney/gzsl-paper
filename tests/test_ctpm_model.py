import torch

from model.frameworks.v6.ctpm import CTPMModel, attention_diversity_loss, pair_ce_loss


def _unit_rows(tensor):
    return torch.nn.functional.normalize(tensor.float(), dim=-1)


def _model():
    generator = torch.Generator().manual_seed(123)
    class_names = _unit_rows(torch.randn(5, 768, generator=generator))
    roles = _unit_rows(torch.randn(5, 8, 768, generator=generator))
    return CTPMModel(class_names, roles, hidden_dim=12, patch_projection_dim=16)


def test_ctpm_forward_shapes_and_same_pair_for_off_paths():
    torch.manual_seed(1)
    model = _model()
    images = _unit_rows(torch.randn(4, 768))
    patches = _unit_rows(torch.randn(4, 36, 768))

    full = model(images, patches)
    s_off = model(images, patches, enable_s=False)
    v_off = model(images, patches, enable_v=False)
    i_off = model(images, patches, enable_i=False)

    assert tuple(full.logits.shape) == (4, 5)
    assert tuple(full.attention.shape) == (4, 8, 36)
    assert torch.equal(full.top2_global, s_off.top2_global)
    assert torch.equal(full.top2_global, v_off.top2_global)
    assert torch.equal(full.top2_global, i_off.top2_global)
    assert torch.allclose(full.attention.sum(dim=-1), torch.ones(4, 8), atol=1e-6)


def test_ctpm_pair_correction_is_antisymmetric():
    torch.manual_seed(2)
    model = _model()
    images = _unit_rows(torch.randn(3, 768))
    patches = _unit_rows(torch.randn(3, 36, 768))
    out = model(images, patches)
    without_role = out.logits - model._role_logits(images, torch.arange(5), True)
    delta = without_role - out.base_logits
    assert torch.allclose(delta.sum(dim=1), torch.zeros(3), atol=1e-6)
    inactive = torch.ones_like(delta, dtype=torch.bool)
    inactive.scatter_(1, out.top2_local, False)
    assert torch.allclose(delta[inactive], torch.zeros_like(delta[inactive]), atol=1e-6)


def test_ctpm_losses_and_gradients_reach_all_three_modules():
    torch.manual_seed(3)
    model = _model()
    images = _unit_rows(torch.randn(6, 768))
    patches = _unit_rows(torch.randn(6, 36, 768))
    labels = torch.tensor([0, 1, 2, 3, 4, 0])

    out = model(images, patches, labels=labels)
    loss = torch.nn.functional.cross_entropy(out.logits, labels)
    loss = loss + 0.1 * pair_ce_loss(out, labels) + 0.01 * attention_diversity_loss(out.attention)
    loss.backward()

    groups = model.parameter_groups()
    for name, params in groups.items():
        total = sum(float(p.grad.detach().abs().sum()) for p in params if p.grad is not None)
        assert total > 0.0, name


def test_ctpm_zsl_axis_outputs_global_predictions():
    torch.manual_seed(4)
    model = _model()
    images = _unit_rows(torch.randn(2, 768))
    patches = _unit_rows(torch.randn(2, 36, 768))
    axis = torch.tensor([2, 4])
    out = model(images, patches, class_ids=axis)
    assert tuple(out.logits.shape) == (2, 2)
    assert set(out.top2_global.reshape(-1).tolist()) <= {2, 4}
