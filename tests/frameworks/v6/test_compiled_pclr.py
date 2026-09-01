from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.frameworks.v6.compiled_pclr import (
    CLASS_COUNT,
    EDGE_COUNT,
    EMBED_DIM,
    READER_HIDDEN_DIM,
    ROLE_COUNT,
    CompiledPCLRHead,
)
from model.frameworks.v6.train_compiled_pclr import (
    _finite_source_gradients,
    evaluate_head,
    gate_b_contract_passed,
    load_compiled_config,
)


def _edges() -> torch.Tensor:
    values: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for index in range(CLASS_COUNT):
        pair = tuple(sorted((index, (index + 1) % CLASS_COUNT)))
        if pair not in used:
            used.add(pair)
            values.append(pair)
    for left in range(CLASS_COUNT):
        for right in range(left + 1, CLASS_COUNT):
            pair = (left, right)
            if pair in used:
                continue
            used.add(pair)
            values.append(pair)
            if len(values) == EDGE_COUNT:
                return torch.tensor(values, dtype=torch.long)
    raise AssertionError("没有生成438条唯一边")


def _head() -> CompiledPCLRHead:
    generator = torch.Generator().manual_seed(201)
    base = torch.randn(CLASS_COUNT, EMBED_DIM, generator=generator)
    roles = torch.randn(CLASS_COUNT, ROLE_COUNT, EMBED_DIM, generator=generator)
    relations = F.normalize(
        torch.randn(EDGE_COUNT, 2, EMBED_DIM, generator=generator), dim=-1
    )
    w1 = torch.randn(READER_HIDDEN_DIM, EMBED_DIM, generator=generator) * 0.01
    b1 = torch.zeros(READER_HIDDEN_DIM)
    w2 = torch.zeros(EMBED_DIM, READER_HIDDEN_DIM)
    b2 = torch.zeros(EMBED_DIM)
    return CompiledPCLRHead(
        base_prototypes=base,
        role_prototypes=roles,
        relation_embeddings=relations,
        edge_index=_edges(),
        seen_classes=torch.arange(150),
        scale=20.0,
        reader_in_state=(w1, b1),
        reader_out_state=(w2, b2),
    )


def test_compiled_relation_matches_edge_space() -> None:
    head = _head()
    images = torch.randn(5, EMBED_DIM, generator=torch.Generator().manual_seed(7))
    readout = head.read_images(images)
    edges = head.edge_index
    incidence = torch.zeros(EDGE_COUNT, CLASS_COUNT, dtype=torch.float64)
    rows = torch.arange(EDGE_COUNT)
    incidence[rows, edges[:, 0]] = 1.0
    incidence[rows, edges[:, 1]] = -1.0
    mapping = torch.linalg.solve(
        incidence.T @ incidence + 0.3 * torch.eye(CLASS_COUNT, dtype=torch.float64),
        incidence.T,
    )
    direction = (head.relation_embeddings[:, 0] - head.relation_embeddings[:, 1]).double()
    edge_space = (readout.double() @ direction.T / 0.2) @ mapping.T
    prototype_space = readout @ head.compiled_g.T
    assert torch.allclose(edge_space.float(), prototype_space, atol=2e-5, rtol=0.0)


def test_export_and_module_off_paths() -> None:
    head = _head()
    images = torch.randn(4, EMBED_DIM, generator=torch.Generator().manual_seed(8))
    full = head(images)
    export = head.export()
    image = F.normalize(images, dim=-1)
    readout = F.normalize(
        images
        + F.linear(
            F.gelu(F.linear(images, export.reader_in_weight, export.reader_in_bias)),
            export.reader_out_weight,
            export.reader_out_bias,
        ),
        dim=-1,
    )
    deployed = torch.cat((image, readout), dim=1) @ export.q.T + export.bias
    assert tuple(export.q.shape) == (CLASS_COUNT, 2 * EMBED_DIM)
    assert tuple(export.bias.shape) == (CLASS_COUNT,)
    assert torch.allclose(full, deployed, atol=1e-5, rtol=0.0)

    s_off = head(images, semantic_enabled=False)
    v_off = head(images, visual_enabled=False)
    i_off = head(images, interaction_enabled=False)
    assert not torch.equal(full, s_off)
    assert torch.allclose(full, v_off, atol=1e-6, rtol=0.0)  # zero-init Reader residual
    expected_i_off = image @ head.image_q().T + head.seen_bias
    assert torch.allclose(i_off, expected_i_off, atol=1e-5, rtol=0.0)


def test_final_ce_and_relation_ce_reach_all_trainable_parameters() -> None:
    head = _head()
    images = torch.randn(50, EMBED_DIM, generator=torch.Generator().manual_seed(9))
    targets = torch.arange(50)
    losses = head.training_losses(images, targets, relation_loss_weight=1.0)
    losses["total"].backward()
    assert set(losses) == {"total", "classification", "relation"}
    assert all(torch.isfinite(value) for value in losses.values())
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert head.raw_alpha.grad is not None and float(head.raw_alpha.grad.abs()) > 0.0
    assert head.raw_role_weights.grad is not None
    assert float(head.raw_role_weights.grad.norm()) > 0.0


def test_parameter_contract_and_config() -> None:
    head = _head()
    contract = head.parameter_contract()
    assert {row["name"] for row in contract} == {
        "reader_in",
        "reader_out",
        "raw_alpha",
        "raw_role_weights",
        "base_q",
        "compiled_g",
        "seen_bias",
    }
    config_path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "v6"
        / "innovation"
        / "V6-INNOVATION-001_COMPILED_PCLR"
        / "configs"
        / "RUN-001.yaml"
    )
    config, config_sha = load_compiled_config(config_path)
    assert config["candidate_top_k"] is None
    assert config["base_commit"] == "52b511d77b4ad048f35b40dc3cbd9afd092167e9"
    assert len(config_sha) == 64


def test_evaluate_head_reports_full_and_all_off_conditions() -> None:
    head = _head()
    generator = torch.Generator().manual_seed(10)
    tensors = {
        "test_seen_features": torch.randn(150, EMBED_DIM, generator=generator),
        "test_seen_labels": torch.arange(150),
        "test_unseen_features": torch.randn(50, EMBED_DIM, generator=generator),
        "test_unseen_labels": torch.arange(150, 200),
    }
    result = evaluate_head(head, tensors, torch.device("cpu"))
    assert set(result["metrics"]) == {"full", "s_off", "v_off", "i_off"}
    for metrics in result["metrics"].values():
        assert set(metrics) == {"U", "S", "H", "ZS"}
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert set(result["transitions"]) == {"s_off", "v_off", "i_off"}


def test_update_zero_can_never_pass_gate_b() -> None:
    metrics = {
        "full": {"U": 82.0, "S": 82.0, "H": 82.0, "ZS": 90.0},
        "s_off": {"U": 80.0, "S": 80.0, "H": 80.0, "ZS": 88.0},
        "v_off": {"U": 80.0, "S": 80.0, "H": 80.0, "ZS": 88.0},
        "i_off": {"U": 80.0, "S": 80.0, "H": 80.0, "ZS": 88.0},
    }
    assert not gate_b_contract_passed(
        metrics,
        best_update=0,
        required_module_delta_h=1.0,
        max_us_gap=8.0,
    )
    assert gate_b_contract_passed(
        metrics,
        best_update=141,
        required_module_delta_h=1.0,
        max_us_gap=8.0,
    )


def test_source_sync_is_eval_mode_and_rng_neutral() -> None:
    head = _head()

    class FakeSource(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = nn.Parameter(torch.randn(CLASS_COUNT, EMBED_DIM))
            self.parent = SimpleNamespace(
                tg_vpr=SimpleNamespace(
                    sentence_embeds=torch.randn(CLASS_COUNT, ROLE_COUNT, EMBED_DIM)
                )
            )

        def scale(self):
            return torch.tensor(20.0)

        def prototypes(self):
            return F.dropout(self.base, p=0.5, training=self.training)

    source = FakeSource().train()
    rng_before = torch.random.get_rng_state().clone()
    head.sync_source_prototypes(source)
    assert source.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    expected = F.normalize(source.base.detach(), dim=-1) * 20.0
    assert torch.allclose(head.base_q, expected, atol=1e-6, rtol=0.0)


def test_source_gradient_receipt_ignores_inactive_groups() -> None:
    class FakeParent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.active = nn.Linear(3, 2)
            self.inactive = nn.Parameter(torch.ones(4))

        def parameter_groups(self):
            return {"active": list(self.active.parameters()), "inactive": []}

    source = SimpleNamespace(parent=FakeParent(), gate=nn.Linear(2, 1))
    for parameter in source.parent.active.parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in source.gate.parameters():
        parameter.grad = torch.ones_like(parameter)
    assert source.parent.inactive.grad is None
    receipt = _finite_source_gradients(source)
    assert receipt
    assert all("inactive" not in name for name in receipt)
