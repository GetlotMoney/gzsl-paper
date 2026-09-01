import hashlib
import json
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.frameworks.v5.cuav import CUAVModel, cuav_policy_loss, standardize
from tools.prepare_cuav_assets import WINDOWS, WINDOW_SHA


def _model(classes=5):
    generator=torch.Generator().manual_seed(7)
    names=F.normalize(torch.randn(classes,768,generator=generator),dim=-1)
    return CUAVModel(names,torch.arange(classes))


def _inputs(batch=2):
    generator=torch.Generator().manual_seed(17)
    full=F.normalize(torch.randn(batch,768,generator=generator),dim=-1)
    crops=F.normalize(torch.randn(batch,25,768,generator=generator),dim=-1)
    return full,crops


def test_geometry_hash_is_frozen():
    digest=hashlib.sha256(json.dumps(WINDOWS,separators=(",",":")).encode()).hexdigest()
    assert digest==WINDOW_SHA
    assert len(WINDOWS)==25 and WINDOWS[12]==(9,9)


def test_standardize_uses_active_axis_and_fixed_eps():
    values=torch.tensor([[1.0,2.0,3.0]])
    output=standardize(values)
    assert abs(float(output.mean()))<1e-6
    assert torch.allclose(output.var(-1,unbiased=False),torch.ones(1),atol=2e-6)


def test_policy_starts_uniform_and_two_step_gradient_contract():
    model=_model(); full,crops=_inputs(); optimizer=torch.optim.AdamW(model.visual_module.parameters(),lr=1e-3)
    first=model.training_forward(full,crops)
    assert torch.equal(first["policy"],torch.full_like(first["policy"],1/25))
    loss=cuav_policy_loss(first,torch.tensor([0,1]))["total"]; loss.backward()
    assert float(model.visual_module.action_projection.weight.grad.abs().max())>0
    assert float(model.visual_module.image_projection.weight.grad.abs().max())==0
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    second=model.training_forward(full,crops); cuav_policy_loss(second,torch.tensor([0,1]))["total"].backward()
    assert float(model.visual_module.image_projection.weight.grad.abs().max())>0
    assert float(model.visual_module.query_projection.weight.grad.abs().max())>0
    assert float(model.visual_module.stats_projection.weight.grad.abs().max())>0


def test_s_off_zeros_ambiguity_state():
    model=_model(); full,_=_inputs()
    with patch.object(model.semantic_module,"forward",wraps=model.semantic_module.forward) as call:
        output=model.policy(full,semantic_off=True)
    assert call.call_args.kwargs["semantic_off"] is True
    assert output["policy_logits"].shape==(2,25)


def test_interaction_off_bypasses_relative_solver():
    model=_model(); full,crops=_inputs(); selected=crops[:,0]
    with patch.object(model.interaction_module,"full_update",side_effect=AssertionError("relative solver opened")):
        output=model.selected_update(full,selected,interaction_off=True)
    assert output["logits"].shape==(2,5)


def test_policy_loss_is_finite():
    model=_model(); output=model.training_forward(*_inputs())
    losses=cuav_policy_loss(output,torch.tensor([0,1]))
    assert all(torch.isfinite(value) for value in losses.values())
