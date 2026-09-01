import torch
import torch.nn.functional as F

from model.frameworks.v5.rceg import (
    MASK_COUNT,
    ROLE_COUNT,
    RCEGModel,
    stable_rival_positions,
    rceg_loss,
)
from model.frameworks.v5.train_rceg import same_class_cycle
from tools.prepare_rceg_assets import interleaved_masks


def _model(class_count=5):
    generator = torch.Generator().manual_seed(7)
    names = F.normalize(torch.randn(class_count, 768, generator=generator), dim=-1)
    roles = F.normalize(
        torch.randn(class_count, ROLE_COUNT, 768, generator=generator), dim=-1
    )
    return RCEGModel(names, roles, torch.arange(class_count), candidate_chunk_size=2)


def _inputs(batch=2):
    generator = torch.Generator().manual_seed(17)
    image = F.normalize(torch.randn(batch, 768, generator=generator), dim=-1)
    masked_cls = F.normalize(
        torch.randn(batch, MASK_COUNT, 768, generator=generator), dim=-1
    )
    visible = F.normalize(
        torch.randn(batch, MASK_COUNT, 6, 768, generator=generator), dim=-1
    )
    target = F.normalize(
        torch.randn(batch, MASK_COUNT, 1024, generator=generator), dim=-1
    )
    return image, masked_cls, visible, target


def test_stable_rival_positions_uses_global_id_tie_break():
    logits = torch.tensor([[1.0, 2.0, 2.0, 0.0]])
    class_ids = torch.tensor([9, 7, 3, 1])
    rivals = stable_rival_positions(logits, class_ids)
    # Global class 3 wins the tie. Its own rival is global class 7.
    assert rivals.tolist() == [[2, 2, 1, 2]]


def test_full_forward_is_finite_and_three_modules_receive_contract_inputs():
    model = _model()
    output = model(*_inputs(), mode="full")
    assert output["logits"].shape == (2, 5)
    assert output["name_error"].shape == (2, 5, MASK_COUNT)
    assert output["role_error"].shape == (2, 5, MASK_COUNT)
    assert torch.isfinite(output["logits"]).all()
    losses = rceg_loss(output, torch.tensor([0, 1]), mode="full")
    losses["total"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.interaction_module.parameters()
    )


def test_s_off_exactly_returns_name_only_parent_logits():
    model = _model()
    values = _inputs()
    parent = model(*values, mode="parent")["logits"]
    s_off = model(*values, mode="s_off")
    assert torch.equal(s_off["score"], torch.zeros_like(s_off["score"]))
    assert torch.equal(s_off["logits"], parent)


def test_s_off_never_calls_role_evidence(monkeypatch):
    model = _model()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("s_off must not read role-conditioned evidence")

    monkeypatch.setattr(model.visual_module, "role_evidence", fail_if_called)
    output = model(*_inputs(), mode="s_off")
    assert torch.equal(output["score"], torch.zeros_like(output["score"]))


def test_v_and_i_off_keep_same_output_interface():
    model = _model()
    values = _inputs()
    full = model(*values, mode="full")
    for mode in ("v_off", "i_off", "reference_difficulty"):
        output = model(*values, mode=mode)
        assert output["logits"].shape == full["logits"].shape
        assert output["score"].shape == full["score"].shape
        assert torch.isfinite(output["logits"]).all()


def test_target_free_never_requires_target_tensor():
    model = _model()
    image, masked_cls, visible, _ = _inputs()
    output = model(image, masked_cls, visible, None, mode="target_free")
    assert output["logits"].shape == (2, 5)
    assert "name_error" not in output
    losses = rceg_loss(output, torch.tensor([0, 1]), mode="target_free")
    assert float(losses["sign"]) == 0.0


def test_interleaved_masks_cover_every_patch_once():
    masks = interleaved_masks()
    assert masks.shape == (4, 576)
    assert masks.sum(dim=1).tolist() == [144, 144, 144, 144]
    assert torch.equal(masks.sum(dim=0), torch.ones(576, dtype=torch.long))


def test_same_class_target_cycle_has_no_fixed_points():
    labels = torch.tensor([3, 3, 3, 8, 8])
    mapping = same_class_cycle(labels)
    assert not bool(mapping.eq(torch.arange(labels.numel())).any())
    assert torch.equal(labels[mapping], labels)
