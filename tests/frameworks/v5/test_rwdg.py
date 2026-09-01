import torch
import torch.nn.functional as F

from model.frameworks.v5.rwdg import (
    ACTION_COUNT,
    ACTION_HEAD_INPUT_DIM,
    FEATURE_DIM,
    HIDDEN_DIM,
    RoleWindowDenseGlimpse,
    dense_utility_loss,
    make_action_positions,
    pool_action_windows,
    stable_top2_by_logit_then_class_id,
)


def _assets(class_count=4):
    generator = torch.Generator().manual_seed(17)
    names = F.normalize(torch.randn(class_count, FEATURE_DIM, generator=generator), dim=-1)
    roles = F.normalize(
        torch.randn(class_count, 8, FEATURE_DIM, generator=generator), dim=-1
    )
    class_ids = torch.arange(class_count) * 3
    return roles, names, class_ids


def test_action_geometry_and_window_pooling_are_fixed():
    positions = make_action_positions()
    assert positions.shape == (25, 8)
    assert torch.allclose(positions[0], torch.tensor([0, 0, .25, .25, .125, .125, .25, .25]))
    assert torch.allclose(positions[-1], torch.tensor([.75, .75, 1, 1, .875, .875, .25, .25]))

    patches = torch.zeros(1, 24, 24, FEATURE_DIM)
    patches[0, :, :, 0] = torch.arange(24 * 24).reshape(24, 24)
    pooled = pool_action_windows(patches.reshape(1, 576, FEATURE_DIM))
    assert pooled.shape == (1, ACTION_COUNT, FEATURE_DIM)
    assert pooled[0, 0, 0] == patches[0, :6, :6, 0].mean()
    assert pooled[0, -1, 0] == patches[0, 18:24, 18:24, 0].mean()


def test_stable_top2_breaks_equal_logits_by_smaller_global_class_id():
    logits = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 2.0]])
    class_ids = torch.tensor([9, 3, 7])
    top2 = stable_top2_by_logit_then_class_id(logits, class_ids)
    assert top2.tolist() == [[1, 2], [1, 2]]


def test_full_attention_shapes_threshold_and_off_paths():
    roles, names, class_ids = _assets()
    model = RoleWindowDenseGlimpse(roles, names, class_ids, seed=7)
    full = F.normalize(torch.randn(3, FEATURE_DIM), dim=-1)
    patches = F.normalize(torch.randn(3, 576, FEATURE_DIM), dim=-1)

    state = model.utility_state(full, patches)
    assert state.utility_logits.shape == (3, 25)
    assert state.attention.shape == (3, 8, 25)
    assert state.action_head_input.shape == (3, 25, ACTION_HEAD_INPUT_DIM)
    assert torch.allclose(state.attention.sum(dim=2), torch.ones(3, 8), atol=1e-6)
    assert torch.equal(state.selected_action, torch.zeros(3, dtype=torch.long))
    assert torch.equal(state.trigger, torch.zeros(3, dtype=torch.bool))
    assert torch.allclose(state.utility, torch.full((3, 25), 0.5))

    semantic_off = model.utility_state(full, patches, semantic_off=True)
    visual_off = model.utility_state(full, None, visual_off=True)
    assert semantic_off.utility_logits.shape == state.utility_logits.shape
    assert visual_off.window_features.shape == (3, 25, FEATURE_DIM)
    assert torch.allclose(
        visual_off.window_features,
        F.normalize(full, dim=-1)[:, None, :].expand(-1, 25, -1),
    )


def test_semantic_off_does_not_read_role_embedding_buffer():
    roles, names, class_ids = _assets()
    model = RoleWindowDenseGlimpse(roles, names, class_ids)
    model.semantic.role_embeddings.fill_(float("nan"))
    full = F.normalize(torch.randn(2, FEATURE_DIM), dim=-1)
    off = model.parent_state(full, semantic_off=True)
    assert torch.isfinite(off.questions).all()
    assert not torch.isfinite(model.parent_state(full, semantic_off=False).questions).all()


def test_train100_state_loads_into_different_eval_axis():
    train_roles, train_names, train_ids = _assets(class_count=3)
    eval_roles, eval_names, eval_ids = _assets(class_count=5)
    train_model = RoleWindowDenseGlimpse(train_roles, train_names, train_ids)
    eval_model = RoleWindowDenseGlimpse(eval_roles, eval_names, eval_ids)
    state = train_model.state_dict()
    assert not any("role_embeddings" in key for key in state)
    assert not any("name_embeddings" in key for key in state)
    assert not any("class_ids" in key for key in state)
    eval_model.load_state_dict(state, strict=True)


def test_dense_targets_encode_leader_challenger_and_outside():
    names = torch.zeros(3, FEATURE_DIM)
    names[0, 0] = 1
    names[1, 1] = 1
    names[2, 2] = 1
    roles = names[:, None, :].expand(-1, 8, -1).clone()
    model = RoleWindowDenseGlimpse(roles, names, torch.tensor([0, 1, 2]))

    full = names[0].expand(3, -1).clone()
    crops = torch.zeros(3, ACTION_COUNT, FEATURE_DIM)
    crops[:, :10, 0] = 1
    crops[:, 10:, 1] = 1
    targets, group, _ = model.dense_utility_targets(
        full, crops, torch.tensor([0, 1, 2])
    )
    assert group.tolist() == [0, 1, 2]
    assert torch.equal(targets[0, :10], torch.ones(10))
    assert torch.equal(targets[0, 10:], torch.zeros(15))
    assert torch.equal(targets[1, :10], torch.zeros(10))
    assert torch.equal(targets[1, 10:], torch.ones(15))
    assert torch.equal(targets[2], torch.zeros(25))


def test_interaction_off_bypasses_margin_and_full_swaps_only_top2():
    names = torch.zeros(3, FEATURE_DIM)
    names[0, 0] = 1
    names[1, 1] = 1
    names[2, 2] = 1
    roles = names[:, None, :].expand(-1, 8, -1).clone()
    model = RoleWindowDenseGlimpse(roles, names, torch.tensor([0, 1, 2]))
    full = names[0].unsqueeze(0)
    patches = names[0].reshape(1, 1, FEATURE_DIM).expand(1, 576, -1)

    off = model(full, patches, None, interaction_off=True)
    assert off.crop_margin is None
    assert torch.equal(off.logits, off.parent_logits)

    model.visual.utility_output.weight.data.fill_(1.0)
    crop = names[1].unsqueeze(0)
    output = model(full, patches, crop)
    assert bool(output.utility_state.trigger.item())
    assert bool(output.swapped.item())
    assert output.logits.argmax(dim=1).item() == 1
    assert torch.equal(output.logits[:, 2], output.parent_logits[:, 2])


def test_zero_head_then_second_step_reaches_every_registered_projection():
    roles, names, class_ids = _assets(class_count=5)
    model = RoleWindowDenseGlimpse(roles, names, class_ids, seed=7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    full = F.normalize(torch.randn(4, FEATURE_DIM), dim=-1)
    patches = F.normalize(torch.randn(4, 576, FEATURE_DIM), dim=-1)
    target = torch.ones(4, ACTION_COUNT)

    first = model.utility_state(full, patches)
    loss = dense_utility_loss(first.utility_logits, target)
    loss.backward()
    assert model.visual.utility_output.weight.grad.abs().max() > 0
    assert model.visual.utility_hidden.weight.grad.abs().max() == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = model.utility_state(full, patches)
    dense_utility_loss(second.utility_logits, target).backward()
    required = (
        model.semantic.role_projection.weight,
        model.semantic.name_projection.weight,
        model.visual.window_key.weight,
        model.visual.window_value.weight,
        model.visual.role_value.weight,
        model.visual.utility_hidden.weight,
        model.visual.utility_output.weight,
    )
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().max() > 0
        for parameter in required
    )
    assert HIDDEN_DIM == 64
