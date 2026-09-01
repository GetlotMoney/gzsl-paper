from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.frameworks.v5.oref import OREFModel, oref_loss, stable_rivals


def _model(classes=5):
    generator = torch.Generator().manual_seed(7)
    names = F.normalize(torch.randn(classes, 768, generator=generator), dim=-1)
    roles = F.normalize(torch.randn(classes, 8, 768, generator=generator), dim=-1)
    return OREFModel(names, roles, torch.arange(classes), candidate_chunk_size=2)


def _inputs(batch=2, tokens=12):
    generator = torch.Generator().manual_seed(17)
    cls = F.normalize(torch.randn(batch, 768, generator=generator), dim=-1)
    patches = F.normalize(torch.randn(batch, tokens, 768, generator=generator), dim=-1)
    return cls, patches


def test_stable_rivals_use_global_id_tie_break():
    logits = torch.tensor([[1.0, 2.0, 2.0, 0.0]])
    assert stable_rivals(logits, torch.tensor([9, 7, 3, 1])).tolist() == [[2, 2, 1, 2]]


def test_zero_output_adapter_is_exact_identity():
    model = _model()
    _, patches = _inputs()
    assert torch.allclose(model.visual_module.adapt(patches), patches, atol=1e-6, rtol=1e-6)


def test_full_forward_and_two_step_gradient_contract():
    model = _model()
    cls, patches = _inputs()
    optimizer = torch.optim.AdamW(model.visual_module.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    first = model(cls, patches, mode="full")
    loss = oref_loss(first, torch.tensor([0, 1]))["total"]
    loss.backward()
    wi = model.visual_module.input_projection.weight
    wo = model.visual_module.output_projection.weight
    assert float(wo.grad.abs().max()) > 0
    assert float(wi.grad.abs().max()) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = model(cls, patches, mode="full")
    oref_loss(second, torch.tensor([0, 1]))["total"].backward()
    assert float(wo.grad.abs().max()) > 0
    assert float(wi.grad.abs().max()) > 0


def test_s_off_physically_avoids_role_queries():
    model = _model()
    with patch.object(
        model.semantic_module, "role_chunk", side_effect=AssertionError("role path opened")
    ):
        output = model(*_inputs(), mode="s_off")
    assert output["logits"].shape == (2, 5)


def test_v_off_physically_avoids_patch_adapter():
    model = _model()
    cls, _ = _inputs()
    with patch.object(
        model.visual_module, "adapt", side_effect=AssertionError("patch path opened")
    ):
        output = model(cls, None, mode="v_off")
    assert output["logits"].shape == (2, 5)


def test_i_off_physically_avoids_falsification_solver():
    model = _model()
    with patch.object(
        model.interaction_module,
        "falsification_score",
        side_effect=AssertionError("falsification solver opened"),
    ):
        output = model(*_inputs(), mode="i_off")
    assert output["logits"].shape == (2, 5)


def test_preliminary_control_modes_are_finite():
    model = _model()
    for mode in ("full", "filip", "signed_ledger", "ledger_mlp"):
        output = model(*_inputs(), mode=mode)
        assert output["logits"].shape == (2, 5)
        assert torch.isfinite(output["logits"]).all()
