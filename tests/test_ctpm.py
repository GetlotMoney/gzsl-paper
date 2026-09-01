import torch

from model.frameworks.v6.ctpm import (
    CTPMModel,
    attention_diversity_loss,
    ctpm_loss,
    stable_top2,
)


def _unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def _fixture(batch=5, classes=9):
    torch.manual_seed(208)
    names = _unit(torch.randn(classes, 768))
    roles = _unit(torch.randn(classes, 8, 768))
    model = CTPMModel(names, roles, hidden_dim=12, patch_projection_dim=16)
    image = _unit(torch.randn(batch, 768))
    patches = _unit(torch.randn(batch, 36, 768))
    return model, image, patches


def test_stable_top2_uses_class_id_tie_break():
    logits = torch.tensor(
        [
            [1.0, 3.0, 3.0, 2.0],
            [5.0, 5.0, 4.0, 5.0],
        ]
    )
    top2 = stable_top2(logits)

    assert top2.tolist() == [[1, 2], [0, 1]]


def test_forward_keeps_pair_identity_and_antisymmetric_scatter_across_offs():
    model, image, patches = _fixture()

    full = model(image, patches)
    s_off = model(image, patches, enable_s=False)
    v_off = model(image, patches, enable_v=False)
    i_off = model(image, patches, enable_i=False)

    for out in (s_off, v_off, i_off):
        assert torch.equal(out.top2_local, full.top2_local)
        assert torch.equal(out.top2_global, full.top2_global)

    assert torch.allclose(full.correction.sum(dim=1), torch.zeros(image.shape[0]))
    assert torch.allclose(s_off.d_s, torch.zeros_like(s_off.d_s))
    assert torch.allclose(s_off.role_logits, torch.zeros_like(s_off.role_logits))
    assert torch.allclose(s_off.interaction_input[:, 1], torch.zeros_like(s_off.d_s))
    assert torch.allclose(v_off.d_v, torch.zeros_like(v_off.d_v))
    assert torch.allclose(v_off.visual_evidence, torch.zeros_like(v_off.visual_evidence))
    assert torch.allclose(v_off.interaction_input[:, 2], torch.zeros_like(v_off.d_v))
    assert torch.allclose(v_off.interaction_input[:, 3], torch.zeros_like(v_off.d_v))
    assert torch.allclose(i_off.d_i, torch.zeros_like(i_off.d_i))


def test_role_query_control_is_separate_from_s_off():
    model, image, patches = _fixture()

    full = model(image, patches)
    s_off = model(image, patches, enable_s=False)
    query_off = model(image, patches, query_mode="class_name_difference")

    assert torch.allclose(full.semantic_evidence, s_off.semantic_evidence)
    assert not torch.allclose(full.semantic_evidence, query_off.semantic_evidence)


def test_attention_is_role_patch_distribution_and_diversity_is_finite():
    model, image, patches = _fixture()

    out = model(image, patches)
    attention = out.attention

    assert attention.shape == (image.shape[0], 8, 36)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(image.shape[0], 8))
    assert torch.isfinite(attention_diversity_loss(attention))
    assert attention.std() > 0


def test_full_loss_backpropagates_to_all_three_modules_and_attention():
    model, image, patches = _fixture(batch=7, classes=11)

    out = model(image, patches)
    labels = out.top2_global[:, 1].detach().clone()
    loss, parts = ctpm_loss(out, labels)
    loss.backward()

    groups = model.parameter_groups()
    assert parts["pair_count"].item() == image.shape[0]
    assert sum(p.grad.abs().sum() for p in groups["semantic"] if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in groups["visual"] if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in groups["interaction"] if p.grad is not None) > 0
    assert model.patch_query.weight.grad.abs().sum() > 0
    assert model.patch_key.weight.grad.abs().sum() > 0


def test_full_ce_alone_reaches_every_margin_component_at_update0():
    model, image, patches = _fixture(batch=7, classes=11)
    output = model(image, patches)
    labels = output.top2_global[:, 1].detach().clone()
    torch.nn.functional.cross_entropy(output.logits, labels).backward()
    assert model.raw_role_weights.grad.abs().sum() > 0
    assert model.semantic_margin.net[0].weight.grad.abs().sum() > 0
    assert model.semantic_margin.net[-1].weight.grad.abs().sum() > 0
    assert model.patch_query.weight.grad.abs().sum() > 0
    assert model.patch_key.weight.grad.abs().sum() > 0
    assert model.visual_margin.net[0].weight.grad.abs().sum() > 0
    assert model.visual_margin.net[-1].weight.grad.abs().sum() > 0
    assert model.interaction_margin.net[0].weight.grad.abs().sum() > 0
    assert model.interaction_margin.net[-1].weight.grad.abs().sum() > 0
