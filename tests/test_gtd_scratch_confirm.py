from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from model.frameworks.v4.train import (
    build_model,
    load_assets,
    load_config,
    tensor_mapping_sha256,
)
from tools.reproducibility import configure_reproducibility


ROOT = Path(__file__).resolve().parents[1]
TG_CONFIG = ROOT / "config/tries/v3_try_040_tg_scratch_fixed150.yaml"
GTD_CONFIG = ROOT / "config/tries/v3_try_041_gtd_scratch_fixed150.yaml"


def test_scratch_configs_are_matched_except_registered_condition():
    tg, _ = load_config(TG_CONFIG)
    gtd, _ = load_config(GTD_CONFIG)
    assert tg["experiment_id"] == "V3-TRY-040"
    assert gtd["experiment_id"] == "V3-TRY-041"
    assert tg["tg_checkpoint"] is None and gtd["tg_checkpoint"] is None
    assert tg["parent_metrics_percent"] is None
    assert tg["gate_loss_weight"] == 0.0
    assert gtd["gate_loss_weight"] == 1.0
    assert tg["tg_learning_rate"] == tg["tg_min_learning_rate"] == 1e-4
    allowed = {"experiment_id", "condition_id", "gate_loss_weight"}
    assert {key for key in tg if tg[key] != gtd[key]} == allowed


def test_local_asset_identity_and_required_tensors_are_exact():
    config, _ = load_config(TG_CONFIG)
    tensors = load_assets(config)
    assert tuple(tensors["train_features"].shape) == (7057, 768)
    assert tuple(tensors["role_sentence_embeds"].shape) == (200, 8, 768)
    assert torch.unique(tensors["train_labels"]).numel() == 150


def test_scratch_conditions_have_identical_initial_tg_and_parent_gradients():
    tg_config, _ = load_config(TG_CONFIG)
    gtd_config, _ = load_config(GTD_CONFIG)
    tensors = load_assets(tg_config)

    configure_reproducibility(7, strict_determinism=False)
    control = build_model(tg_config, tensors, torch.device("cpu"))
    configure_reproducibility(7, strict_determinism=False)
    candidate = build_model(gtd_config, tensors, torch.device("cpu"))
    assert tensor_mapping_sha256(dict(control.parent.state_dict())) == tensor_mapping_sha256(
        dict(candidate.parent.state_dict())
    )

    control.train()
    candidate.train()
    images = tensors["train_features"][:4].float()
    seen = control.seen_classes
    targets = torch.arange(4)
    shared_rng = torch.get_rng_state()

    control.zero_grad(set_to_none=True)
    torch.set_rng_state(shared_rng)
    control_logits = control.parent.logits(images, seen)
    control_loss = F.cross_entropy(control_logits, targets) + 0.1 * control.parent.topology_loss()
    control_loss.backward()
    control_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in control.parent.named_parameters()
        if parameter.grad is not None
    }

    candidate.zero_grad(set_to_none=True)
    torch.set_rng_state(shared_rng)
    candidate_logits = candidate.parent.logits(images, seen)
    parent_loss = F.cross_entropy(candidate_logits, targets) + 0.1 * candidate.parent.topology_loss()
    features = torch.randn(5, 6, generator=torch.Generator().manual_seed(41))
    targets_ratio = torch.rand(5, generator=torch.Generator().manual_seed(42))
    gate_loss = F.smooth_l1_loss(candidate.gate.raw_ratio(features), targets_ratio)
    (parent_loss + gate_loss).backward()
    candidate_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in candidate.parent.named_parameters()
        if parameter.grad is not None
    }
    assert torch.equal(control_logits, candidate_logits)
    assert control_grads.keys() == candidate_grads.keys()
    assert all(torch.equal(control_grads[name], candidate_grads[name]) for name in control_grads)
    assert any(parameter.grad is not None for parameter in candidate.gate.parameters())
    assert all(parameter.grad is None for parameter in control.gate.parameters())


def test_real_reproducibility_metadata_is_weights_only_safe(tmp_path: Path):
    metadata = configure_reproducibility(7, strict_determinism=False)
    assert type(metadata["torch_version"]) is str
    assert type(metadata["cuda_version"]) is str
    path = tmp_path / "metadata.pth"
    torch.save({"reproducibility": copy.deepcopy(metadata)}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["reproducibility"] == metadata
