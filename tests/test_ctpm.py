import torch

from model.frameworks.v6.ctpm import (
    CTPMModel,
    attention_diversity_loss,
    balanced_interaction_direction_loss,
    balanced_pair_ce,
    ctpm_loss,
    interaction_direction_targets,
    isolated_interaction_margin,
    pair_scatter,
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


def test_forward_uses_semantic_pair_and_keeps_it_for_visual_interaction_offs():
    model, image, patches = _fixture()

    full = model(image, patches)
    s_off = model(image, patches, enable_s=False)
    parent = model(image, patches, enable_s=False, enable_v=False, enable_i=False)
    v_off = model(image, patches, enable_v=False)
    i_off = model(image, patches, enable_i=False)

    for out in (v_off, i_off):
        assert torch.equal(out.top2_local, full.top2_local)
        assert torch.equal(out.top2_global, full.top2_global)
    assert torch.equal(s_off.top2_global, parent.top2_global)

    assert torch.allclose(full.correction.sum(dim=1), torch.zeros(image.shape[0]))
    assert torch.allclose(s_off.d_s, torch.zeros_like(s_off.d_s))
    assert torch.allclose(s_off.role_logits, torch.zeros_like(s_off.role_logits))
    assert torch.allclose(s_off.interaction_input[:, 1], torch.zeros_like(s_off.d_s))
    assert torch.allclose(v_off.d_v, torch.zeros_like(v_off.d_v))
    assert torch.allclose(v_off.candidate_visual_evidence, torch.zeros_like(v_off.candidate_visual_evidence))
    assert torch.allclose(v_off.interaction_input[:, 2], torch.zeros_like(v_off.d_v))
    assert torch.allclose(v_off.interaction_input[:, 3], torch.zeros_like(v_off.d_v))
    assert v_off.interaction_input[:, 5:].abs().sum() > 0  # role-patch I remains active.
    assert torch.allclose(i_off.d_i, torch.zeros_like(i_off.d_i))


def test_role_query_control_is_separate_from_s_off():
    model, image, patches = _fixture()

    full = model(image, patches)
    v_off = model(image, patches, enable_v=False)
    query_off = model(image, patches, query_mode="class_name_difference")

    assert torch.allclose(full.semantic_evidence, v_off.semantic_evidence)
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


def test_pfde_losses_train_semantic_visual_and_interaction_branches_independently():
    model, image, patches = _fixture(batch=8, classes=11)
    out = model(image, patches)
    labels = out.top2_global[:, 0].detach().clone()
    labels[::2] = out.top2_global[::2, 1]

    detached_base = out.base_logits.detach()
    s_logits = (detached_base + pair_scatter(
        out.top2_local, out.d_s, out.logits.size(1)
    )).gather(1, out.top2_local)
    s_loss, s_skipped = balanced_pair_ce(s_logits, out.top2_global, labels)
    assert not s_skipped
    s_loss.backward(retain_graph=True)
    groups = model.parameter_groups()
    assert sum(p.grad.abs().sum() for p in groups["semantic"] if p.grad is not None) > 0
    assert all(p.grad is None for p in groups["visual"] + groups["interaction"])

    model.zero_grad(set_to_none=True)
    v_logits = (detached_base + pair_scatter(
        out.top2_local, out.d_v, out.logits.size(1)
    )).gather(1, out.top2_local)
    v_loss, v_skipped = balanced_pair_ce(v_logits, out.top2_global, labels)
    assert not v_skipped
    v_loss.backward(retain_graph=True)
    assert sum(p.grad.abs().sum() for p in groups["visual"] if p.grad is not None) > 0
    assert all(p.grad is None for p in groups["semantic"] + groups["interaction"])

    model.zero_grad(set_to_none=True)
    d_i = isolated_interaction_margin(model, out)
    i_logits = (detached_base + pair_scatter(
        out.top2_local, d_i, out.logits.size(1)
    )).gather(1, out.top2_local)
    i_loss, i_skipped = balanced_pair_ce(
        i_logits, out.top2_global, labels
    )
    assert not i_skipped
    i_loss.backward()
    assert sum(p.grad.abs().sum() for p in groups["interaction"] if p.grad is not None) > 0
    assert all(p.grad is None for p in groups["semantic"] + groups["visual"])


def test_dial_exact_residual_target_penalizes_overcorrection_and_exposes_three_states():
    pairs = torch.tensor([[2, 3], [4, 5], [6, 7], [8, 9]])
    labels = torch.tensor([3, 4, 7, 8])
    prefix = torch.tensor([[0.0, 1.5], [1.5, 0.0], [0.0, 0.0], [0.0, 0.0]])
    target, c1, c2 = interaction_direction_targets(
        prefix, pairs, labels, target_margin=0.25
    )
    assert target.tolist() == [0.0, 0.0, 0.25, -0.25]
    assert c1.tolist() == [False, True, False, True]
    assert c2.tolist() == [True, False, True, False]
    loss, skipped = balanced_interaction_direction_loss(
        torch.tensor([2.0, -2.0, 0.25, -0.25]), prefix, pairs, labels,
        target_margin=0.25,
    )
    assert not skipped
    assert loss > 0  # already-satisfied rows must be pulled back to zero.
