from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from model.innovations.bd_tst import BalancedDecoupledTST
from model.innovations.elpt import fixed_class_folds
from model.innovations.mmt_tst import (
    MarginTargetTable,
    MinimumMarginGate,
    geodesic_basis,
    geodesic_transport,
    mmt_losses,
)
from model.innovations.train_fresh_effective import (
    FreshSchedule,
    build_model,
    canonical_sha256,
    evaluation_updates,
    gradient_report,
    load_config,
    primary_batch_prefix_sha256,
)
from model.innovations.train_gtd_tst import tensor_mapping_sha256
from model.paper_v2 import PaperV2ThreeModuleModel
from tools.summarize_fresh_effective import summarize


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / f"config/tries/v3_try_{attempt}_fresh_effective.yaml"
    for attempt in ("042", "043", "044", "045")
]


def synthetic_assets() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(941)
    labels = torch.arange(7057).remainder(150).long()
    return {
        "train_features": torch.randn(7057, 768, generator=generator),
        "train_labels": labels,
        "test_seen_features": torch.randn(150, 768, generator=generator),
        "test_seen_labels": torch.arange(150),
        "test_unseen_features": torch.randn(50, 768, generator=generator),
        "test_unseen_labels": torch.arange(150, 200),
        "role_sentence_embeds": torch.randn(200, 8, 768, generator=generator),
    }


def test_configs_define_only_fresh_one_stage_matched_conditions():
    loaded = [load_config(path)[0] for path in CONFIGS]
    assert [row["experiment_id"] for row in loaded] == [
        "V3-TRY-042", "V3-TRY-043", "V3-TRY-044", "V3-TRY-045"
    ]
    assert [row["module"] for row in loaded] == ["tg", "gtd", "mmt", "bd"]
    for row in loaded:
        assert row["initialization_strategy"] == "fresh_seeded_tg"
        assert row["training_strategy"] == "one_stage_simultaneous"
        assert row["stagewise_training"] is False
        assert row["checkpoint_handoff"] is False
        assert row["module_pretraining"] is False
        assert row["tg_checkpoint"] is None
        assert row["tg_checkpoint_sha256"] is None
        assert row["pretrained_module_checkpoint"] is None
        assert row["parent_metrics_percent"] is None
        assert row["tg_learning_rate"] == row["tg_min_learning_rate"] == 1e-4
        assert row["random_seed"] == 7
        assert row["total_updates"] == 21171


def test_schema_rejects_checkpoint_and_unknown_keys(tmp_path: Path):
    raw = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
    raw["tg_checkpoint"] = "/tmp/trained.pth"
    path = tmp_path / "bad-checkpoint.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="初始化"):
        load_config(path)
    raw["tg_checkpoint"] = None
    raw["surprise_checkpoint"] = None
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="字段错误"):
        load_config(path)


def test_four_conditions_have_identical_fresh_parent_rng_and_primary_batches():
    tensors = synthetic_assets()
    identities = []
    for path in CONFIGS:
        config, _ = load_config(path)
        torch.manual_seed(7)
        bundle = build_model(config, tensors, torch.device("cpu"))
        primary = torch.Generator(device="cpu").manual_seed(config["random_seed"])
        identities.append({
            "tg": tensor_mapping_sha256(dict(bundle.parent.tg_vpr.state_dict())),
            "parent": tensor_mapping_sha256(dict(bundle.parent.state_dict())),
            "global_rng": canonical_sha256(torch.get_rng_state()),
            "batch_initial": canonical_sha256(primary.get_state()),
            "batch_1_142": primary_batch_prefix_sha256(primary.get_state()),
        })
        del bundle
    assert all(row == identities[0] for row in identities[1:])


def test_all_conditions_share_parent_forward_and_candidate_gate_gets_first_step_gradient():
    tensors = synthetic_assets()
    shared_rng = None
    reference_logits = None
    for path in CONFIGS:
        config, _ = load_config(path)
        torch.manual_seed(7)
        bundle = build_model(config, tensors, torch.device("cpu"))
        bundle.model.train()
        if shared_rng is None:
            shared_rng = torch.get_rng_state()
        torch.set_rng_state(shared_rng)
        images = tensors["train_features"][:4]
        seen = bundle.parent.seen_classes
        logits = bundle.parent.logits(images, seen)
        topology = bundle.parent.topology_loss()
        main = F.cross_entropy(logits, torch.arange(4)) + 0.1 * topology
        if reference_logits is None:
            reference_logits = logits.detach().clone()
        else:
            assert torch.equal(logits, reference_logits)
        module_parameters = bundle.module_parameters()
        if bundle.module_name == "gtd":
            auxiliary = F.smooth_l1_loss(
                bundle.model.gate.raw_ratio(torch.randn(6, 6)), torch.rand(6)
            )
        elif bundle.module_name == "mmt":
            output = bundle.model.gate(torch.randn(6, 8), torch.ones(6))
            auxiliary = output["move_logit"].square().mean() + output["theta_amount"].mean()
        elif bundle.module_name == "bd":
            auxiliary = bundle.model.gate(torch.randn(6, 4)).mean()
        else:
            auxiliary = main.new_zeros(())
        (main + auxiliary).backward()
        tg_report = gradient_report(list(bundle.parent.parameter_groups()["tg_vpr"]))
        assert tg_report["any_nonzero_gradient"]
        if module_parameters:
            report = gradient_report(module_parameters)
            assert report["all_gradients_present"] and report["any_nonzero_gradient"]


def test_schedule_and_evaluation_contract_are_exact():
    tg = torch.nn.Parameter(torch.zeros(()))
    gate = torch.nn.Parameter(torch.zeros(()))
    tg_optimizer = torch.optim.Adam([tg], lr=1e-4)
    gate_optimizer = torch.optim.Adam([gate], lr=1e-4)
    schedule = FreshSchedule(tg_optimizer, gate_optimizer)
    assert schedule.learning_rates(1) == (1e-4, 1e-5)
    assert schedule.learning_rates(705) == (1e-4, 1e-4)
    assert schedule.learning_rates(21171) == (1e-4, 1e-5)
    assert len(evaluation_updates()) == 151
    assert evaluation_updates()[-2:] == (21150, 21171)


def _fake_result(experiment_id: str, module: str, h: float, module_delta: float) -> dict:
    return {
        "experiment_id": experiment_id,
        "module": module,
        "initialization_strategy": "fresh_seeded_tg",
        "loaded_training_checkpoints": [],
        "initial_tg_state_sha256": "a" * 64,
        "initial_parent_state_sha256": "b" * 64,
        "primary_batch_generator_initial_sha256": "c" * 64,
        "primary_batches_updates_1_142_sha256": "d" * 64,
        "best_metrics": {"U": 79.0, "S": 77.0, "H": h, "ZS": 86.0},
        "best_full_minus_off_delta": {"U": 0.0, "S": 0.0, "H": module_delta, "ZS": 0.0},
    }


def test_summary_enforces_cross_run_and_same_checkpoint_double_gate():
    payloads = [
        _fake_result("V3-TRY-042", "tg", 76.5, 0.0),
        _fake_result("V3-TRY-043", "gtd", 77.6, 1.05),
        _fake_result("V3-TRY-044", "mmt", 77.35, 0.85),
        _fake_result("V3-TRY-045", "bd", 77.6, 0.7),
    ]
    result = summarize(payloads)
    assert [row["decision"] for row in result["candidates"]] == [
        "strong_keep", "weak_keep", "drop"
    ]
    broken = copy.deepcopy(payloads)
    broken[-1]["primary_batches_updates_1_142_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="不匹配"):
        summarize(broken)


def test_canonical_checkpoint_payload_is_weights_only_safe(tmp_path: Path):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    parameter.sum().backward()
    optimizer.step()
    payload = {
        "model_state_dict": {"weight": parameter.detach().clone()},
        "optimizer_state_dict": optimizer.state_dict(),
        "rng": {"cpu": torch.get_rng_state(), "cuda": []},
        "canonical": canonical_sha256(optimizer.state_dict()),
    }
    path = tmp_path / "safe.pth"
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored["canonical"] == canonical_sha256(restored["optimizer_state_dict"])


def test_mmt_geodesic_and_teacher_loss_keep_audited_formula():
    base = F.normalize(torch.randn(5, 768, generator=torch.Generator().manual_seed(81)), dim=-1)
    value = F.normalize(base + 0.2 * torch.randn(5, 768, generator=torch.Generator().manual_seed(82)), dim=-1)
    direction, cap, valid = geodesic_basis(
        base, value, global_theta_max=0.5, tangent_epsilon=1e-6
    )
    assert valid.all()
    assert torch.allclose(
        geodesic_transport(base, direction, torch.zeros(5)), base, atol=1e-7, rtol=0.0
    )
    gate = MinimumMarginGate()
    features = torch.randn(5, 8, generator=torch.Generator().manual_seed(83))
    output = gate(features, cap)
    zeros = torch.zeros(5)
    table = MarginTargetTable(
        class_ids=torch.arange(5), features=features, base=base, direction=direction,
        theta_cap=cap, theta_target=zeros, move_target=zeros, credible=torch.ones(5, dtype=torch.bool),
        status=torch.zeros(5, dtype=torch.long), target_positive_base=torch.ones(5),
        target_direction_score=zeros, soft_negative=zeros, fold_margin=zeros,
        leak_base_scores=torch.zeros(5, 5), leak_direction_scores=torch.zeros(5, 5),
    )
    losses = mmt_losses(output, table, margin_scale=0.02, leak_tolerance=0.005)
    total = losses["move"] + losses["theta"] + losses["margin"] + losses["zero"] + losses["leak"]
    total.backward()
    assert all(parameter.grad is not None for parameter in gate.parameters())
    assert torch.isfinite(total)


def test_bd_balanced_auxiliary_is_gradient_isolated_from_fresh_tg():
    generator = torch.Generator().manual_seed(92)
    parent = PaperV2ThreeModuleModel(
        torch.randn(200, 8, 768, generator=generator), torch.arange(150),
        torch.randn(150, 768, generator=generator), tg_vpr_mode="full",
        transport_mode="off", ccgr_mode="off", dropout=0.5,
        inner_ratio=0.35, outer_ratio=0.65, temperature=0.07,
    )
    model = BalancedDecoupledTST(parent, torch.arange(150), gate_initialization_seed=1557)
    fold_seen, fold_unseen = fixed_class_folds(torch.arange(150))[0]
    labels = torch.cat((fold_seen[:25], fold_unseen[:25]))
    images = torch.randn(50, 768, generator=generator)
    auxiliary = model.auxiliary_objective(images, labels, fold_seen, fold_unseen, 0.1)["loss"]
    tg_parameters = model.tg_parameters()
    gate_parameters = model.gate_parameters()
    tg_grads = torch.autograd.grad(auxiliary, tg_parameters, retain_graph=True, allow_unused=True)
    assert all(value is None or float(value.norm()) == 0.0 for value in tg_grads)
    auxiliary.backward()
    assert all(parameter.grad is not None for parameter in gate_parameters)
    assert any(float(parameter.grad.norm()) > 0.0 for parameter in gate_parameters)
