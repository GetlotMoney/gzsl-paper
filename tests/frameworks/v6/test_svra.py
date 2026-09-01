import torch
import torch.nn.functional as F

from model.frameworks.v6.svra import (
    ACTION_COUNT,
    ACTION_HEAD_INPUT_DIM,
    CEILING_RISK_INPUT_DIM,
    FEATURE_DIM,
    HIDDEN_DIM,
    ROLE_ORDER,
    SemanticVisualRiskArbiter,
    build_ceiling_risk_inputs,
    make_action_positions,
    pool_action_windows,
    risk_arbiter_targets,
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


def test_semantic_input_contract_is_frozen_role_tensor_not_runtime_prompts():
    assert ROLE_ORDER == (
        "beak",
        "head_features",
        "body_plumage",
        "wings",
        "tail",
        "legs",
        "overall_appearance",
        "unique_discriminative_features",
    )
    roles, names, class_ids = _assets()
    model = SemanticVisualRiskArbiter(roles, names, class_ids, seed=7)

    assert model.semantic.role_embeddings.shape == (4, 8, FEATURE_DIM)
    assert model.semantic.name_embeddings.shape == (4, FEATURE_DIM)


def test_action_geometry_window_pooling_and_stable_top2_are_fixed():
    positions = make_action_positions()
    assert positions.shape == (25, 8)
    assert torch.allclose(positions[0], torch.tensor([0, 0, 0.25, 0.25, 0.125, 0.125, 0.25, 0.25]))
    assert torch.allclose(positions[-1], torch.tensor([0.75, 0.75, 1, 1, 0.875, 0.875, 0.25, 0.25]))

    patches = torch.zeros(1, 24, 24, FEATURE_DIM)
    patches[0, :, :, 0] = torch.arange(24 * 24).reshape(24, 24)
    pooled = pool_action_windows(patches.reshape(1, 576, FEATURE_DIM))
    assert pooled.shape == (1, ACTION_COUNT, FEATURE_DIM)
    assert pooled[0, 0, 0] == patches[0, :6, :6, 0].mean()
    assert pooled[0, -1, 0] == patches[0, 18:24, 18:24, 0].mean()

    logits = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 2.0]])
    class_ids = torch.tensor([9, 3, 7])
    assert stable_top2_by_logit_then_class_id(logits, class_ids).tolist() == [[1, 2], [1, 2]]


def test_full_shapes_semantic_zero_off_and_visual_cls_broadcast():
    roles, names, class_ids = _assets()
    model = SemanticVisualRiskArbiter(roles, names, class_ids, seed=7)
    full = F.normalize(torch.randn(3, FEATURE_DIM), dim=-1)
    patches = F.normalize(torch.randn(3, 576, FEATURE_DIM), dim=-1)

    state = model.action_state(full, patches)
    assert model.policy_state(full, patches).utility_logits.shape == (3, ACTION_COUNT)
    assert set(model.risk_probabilities(state)) == {
        "triggered4d",
        "all_row4d",
        "ceiling13d",
    }
    assert state.utility_logits.shape == (3, 25)
    assert state.policy_logits.shape == (3, 26)
    assert state.attention.shape == (3, 8, 25)
    assert state.action_head_input.shape == (3, 25, ACTION_HEAD_INPUT_DIM)
    assert torch.allclose(state.attention.sum(dim=2), torch.ones(3, 8), atol=1e-6)
    assert torch.equal(state.selected_action, torch.zeros(3, dtype=torch.long))
    assert torch.equal(state.trigger, torch.zeros(3, dtype=torch.bool))
    assert torch.allclose(state.utility, torch.full((3, 25), 1.0 / 26.0))

    semantic_off = model.action_state(full, patches, semantic_off=True)
    assert semantic_off.pair.questions.shape == (3, 8, HIDDEN_DIM)
    assert torch.equal(semantic_off.pair.questions, torch.zeros_like(semantic_off.pair.questions))

    visual_off = model.action_state(full, None, visual_off=True)
    assert visual_off.window_features.shape == (3, 25, FEATURE_DIM)
    assert torch.allclose(
        visual_off.window_features,
        F.normalize(full, dim=-1)[:, None, :].expand(-1, 25, -1),
    )


def test_zero_head_initial_full_keeps_parent_and_forced_risk_swaps_top2():
    names = torch.zeros(3, FEATURE_DIM)
    names[0, 0] = 1
    names[1, 1] = 1
    names[2, 2] = 1
    roles = names[:, None, :].expand(-1, 8, -1).clone()
    model = SemanticVisualRiskArbiter(roles, names, torch.tensor([0, 1, 2]))
    full = names[0].unsqueeze(0)
    patches = names[0].reshape(1, 1, FEATURE_DIM).expand(1, 576, -1)

    output = model(full, patches)
    assert torch.equal(output.logits, output.parent_logits)
    assert torch.allclose(output.swap_probability, torch.full((1,), 0.5))
    assert not bool(output.swapped.item())

    model.visual.utility_output.weight.data.fill_(1.0)
    model.interaction.output.bias.data.fill_(1.0)
    swapped = model(full, patches)
    assert bool(swapped.action_state.trigger.item())
    assert bool(swapped.swapped.item())
    assert swapped.logits.argmax(dim=1).item() == 1
    assert torch.equal(swapped.logits[:, 2], swapped.parent_logits[:, 2])


def test_13d_ceiling_input_builder_and_risk_targets_shape():
    roles, names, class_ids = _assets()
    model = SemanticVisualRiskArbiter(roles, names, class_ids)
    full = F.normalize(torch.randn(2, FEATURE_DIM), dim=-1)
    patches = F.normalize(torch.randn(2, 576, FEATURE_DIM), dim=-1)
    state = model.action_state(full, patches)

    ceiling_inputs = build_ceiling_risk_inputs(
        state.pair.parent_stats,
        state.selected_policy_confidence,
        state.selected_action,
    )
    assert ceiling_inputs.shape == (2, CEILING_RISK_INPUT_DIM)

    targets, mask, group = risk_arbiter_targets(
        state.pair,
        torch.tensor([0, 3]),
        class_ids,
        trigger=torch.ones(2, dtype=torch.bool),
    )
    assert targets.shape == (2,)
    assert mask.shape == (2,)
    assert group.shape == (2,)
