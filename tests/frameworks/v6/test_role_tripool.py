import torch
import torch.nn.functional as F

from model.frameworks.v6.role_tripool import (
    ACTION_COUNT,
    ACTION_HEAD_INPUT_DIM,
    CORRECTION_CLASS,
    DAMAGE_CLASS,
    FEATURE_DIM,
    NEUTRAL_CLASS,
    RoleTriPoolGlimpse,
    make_action_positions,
    pool_role_window_statistics,
    tri_state_action_loss,
)


def _assets(class_count=4):
    generator = torch.Generator().manual_seed(17)
    names = F.normalize(torch.randn(class_count, FEATURE_DIM, generator=generator), dim=-1)
    roles = F.normalize(torch.randn(class_count, 8, FEATURE_DIM, generator=generator), dim=-1)
    return roles, names, torch.arange(class_count) * 3


def test_role_window_statistics_preserve_single_patch_peak():
    role_deltas = torch.zeros(1, 8, FEATURE_DIM)
    role_deltas[0, 0, 0] = 1.0
    patches = torch.zeros(1, 576, FEATURE_DIM)
    patches[:, :, 1] = 1.0
    patches[0, 0] = 0.0
    patches[0, 0, 0] = 1.0
    scores, stats = pool_role_window_statistics(patches, role_deltas)
    assert scores.shape == (1, 25, 8, 36)
    assert stats.shape == (1, 25, 8, 3)
    assert torch.isclose(stats[0, 0, 0, 0], torch.tensor(1.0 / 36.0))
    assert stats[0, 0, 0, 1] == 1.0
    assert stats[0, 0, 0, 2] == 0.0
    assert stats[0, -1, 0].tolist() == [0.0, 0.0, 0.0]


def test_action_geometry_and_input_contract_are_fixed():
    positions = make_action_positions()
    assert positions.shape == (25, 8)
    assert ACTION_HEAD_INPUT_DIM == 34
    assert torch.allclose(positions[0], torch.tensor([0, 0, .25, .25, .125, .125, .25, .25]))
    assert torch.allclose(positions[-1], torch.tensor([.75, .75, 1, 1, .875, .875, .25, .25]))


def test_zero_head_abstains_and_off_paths_keep_shapes():
    roles, names, class_ids = _assets()
    model = RoleTriPoolGlimpse(roles, names, class_ids, seed=7)
    full = F.normalize(torch.randn(3, FEATURE_DIM), dim=-1)
    patches = F.normalize(torch.randn(3, 576, FEATURE_DIM), dim=-1)
    state = model.utility_state(full, patches)
    assert state.utility_logits.shape == (3, 25, 3)
    assert state.tri_state_probabilities.shape == (3, 25, 3)
    assert state.role_statistics.shape == (3, 25, 8, 3)
    assert state.action_head_input.shape == (3, 25, ACTION_HEAD_INPUT_DIM)
    assert torch.equal(state.selected_action, torch.zeros(3, dtype=torch.long))
    assert torch.equal(state.trigger, torch.zeros(3, dtype=torch.bool))
    assert torch.allclose(state.utility, torch.zeros(3, ACTION_COUNT))
    assert model.utility_state(full, patches, semantic_off=True).utility_logits.shape == state.utility_logits.shape
    visual_off = model.utility_state(full, None, visual_off=True)
    assert visual_off.role_statistics.shape == state.role_statistics.shape
    assert torch.allclose(visual_off.role_statistics[..., 0], visual_off.role_statistics[..., 1])
    assert torch.allclose(visual_off.role_statistics[..., 0], visual_off.role_statistics[..., 2])


def test_semantic_off_does_not_read_role_embedding_buffer():
    roles, names, class_ids = _assets()
    model = RoleTriPoolGlimpse(roles, names, class_ids)
    model.semantic.role_embeddings.fill_(float("nan"))
    full = F.normalize(torch.randn(2, FEATURE_DIM), dim=-1)
    assert torch.isfinite(model.parent_state(full, semantic_off=True).role_deltas).all()
    assert not torch.isfinite(model.parent_state(full).role_deltas).all()


def test_train_axis_state_loads_into_different_eval_axis():
    train = RoleTriPoolGlimpse(*_assets(3))
    evaluate = RoleTriPoolGlimpse(*_assets(5))
    state = train.state_dict()
    assert not any(key.endswith(("role_embeddings", "name_embeddings", "class_ids")) for key in state)
    evaluate.load_state_dict(state, strict=True)


def test_signed_targets_encode_damage_neutral_and_correction():
    names = torch.zeros(3, FEATURE_DIM)
    names[0, 0], names[1, 1], names[2, 2] = 1, 1, 1
    roles = names[:, None, :].expand(-1, 8, -1).clone()
    model = RoleTriPoolGlimpse(roles, names, torch.tensor([0, 1, 2]))
    full = names[0].expand(3, -1).clone()
    crops = torch.zeros(3, ACTION_COUNT, FEATURE_DIM)
    crops[:, :10, 0] = 1
    crops[:, 10:, 1] = 1
    targets, groups, _ = model.signed_action_targets(full, crops, torch.tensor([0, 1, 2]))
    assert groups.tolist() == [0, 1, 2]
    assert torch.equal(targets[0, :10], torch.full((10,), NEUTRAL_CLASS))
    assert torch.equal(targets[0, 10:], torch.full((15,), DAMAGE_CLASS))
    assert torch.equal(targets[1, :10], torch.full((10,), NEUTRAL_CLASS))
    assert torch.equal(targets[1, 10:], torch.full((15,), CORRECTION_CLASS))
    assert torch.equal(targets[2], torch.full((25,), NEUTRAL_CLASS))


def test_tri_state_loss_uses_all_window_labels():
    logits = torch.zeros(2, ACTION_COUNT, 3)
    targets = torch.full((2, ACTION_COUNT), NEUTRAL_CLASS)
    targets[0, 0] = DAMAGE_CLASS
    targets[1, 4] = CORRECTION_CLASS
    expected = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
    assert torch.allclose(tri_state_action_loss(logits, targets), expected)
