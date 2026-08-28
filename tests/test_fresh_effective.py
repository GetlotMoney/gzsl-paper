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
    make_mmt_teacher,
    primary_batch_prefix_sha256,
    restore_checkpoint_objects,
    seal_checkpoint,
    validate_checkpoint,
)
from model.innovations.train_gtd_tst import tensor_mapping_sha256
from model.innovations.train_gtd_tst import refresh_oracle_targets
from model.paper_v2 import PaperV2ThreeModuleModel
from model.tg_vpr_h1 import train as h1
from tools.summarize_fresh_effective import summarize
from tools.runtime import sha256_file


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
        torch.random.default_generator.manual_seed(7)
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
        torch.random.default_generator.manual_seed(7)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="server CUDA RNG专项")
def test_cuda_four_conditions_preserve_rng_teacher_and_first_parent_step():
    tensors = synthetic_assets()
    device = torch.device("cuda:0")
    evidence = []
    for path in CONFIGS:
        config, _ = load_config(path)
        torch.random.default_generator.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        bundle = build_model(config, tensors, device)
        post_build = canonical_sha256(torch.cuda.get_rng_state(device))
        labels = tensors["train_labels"].long()
        seen_cpu = bundle.parent.seen_classes.cpu()
        centroids = h1.visual_centroids(tensors["train_features"], labels, seen_cpu).to(device)
        folds = fixed_class_folds(seen_cpu)
        if bundle.module_name == "gtd":
            refresh_oracle_targets(bundle.model, centroids, folds, 0.1)
        elif bundle.module_name == "mmt":
            make_mmt_teacher(bundle.model, centroids, config, device)
        pre_main = canonical_sha256(torch.cuda.get_rng_state(device))
        primary = torch.Generator(device="cpu").manual_seed(7)
        indices = torch.randperm(7057, generator=primary)[:50].to(device)
        images = tensors["train_features"].to(device).index_select(0, indices)
        labels_device = labels.to(device)
        seen = seen_cpu.to(device)
        global_to_seen = torch.full((200,), -1, dtype=torch.long, device=device)
        global_to_seen[seen] = torch.arange(150, device=device)
        targets = global_to_seen.index_select(0, labels_device.index_select(0, indices))
        bundle.model.train()
        bundle.model.zero_grad(set_to_none=True)
        logits = bundle.parent.logits(images, seen)
        topology = bundle.parent.topology_loss()
        ce = F.cross_entropy(logits, targets)
        (ce + 0.1 * topology).backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in bundle.parent.tg_vpr.named_parameters()
        }
        evidence.append({
            "post_build": post_build,
            "pre_main": pre_main,
            "logits": logits.detach().cpu(),
            "topology": topology.detach().cpu(),
            "ce": ce.detach().cpu(),
            "gradients": gradients,
        })
        del bundle, images, logits, topology, ce
        torch.cuda.empty_cache()
    reference = evidence[0]
    for row in evidence[1:]:
        assert row["post_build"] == reference["post_build"]
        assert row["pre_main"] == reference["pre_main"]
        assert torch.equal(row["logits"], reference["logits"])
        assert torch.equal(row["topology"], reference["topology"])
        assert torch.equal(row["ce"], reference["ce"])
        assert row["gradients"].keys() == reference["gradients"].keys()
        for name, gradient in row["gradients"].items():
            expected = reference["gradients"][name]
            assert (gradient is None) == (expected is None)
            if gradient is not None:
                assert torch.equal(gradient, expected)


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


EXPECTED_COMMIT = "f" * 40


def _fake_history(
    *, final_h: float, module_off_history: list[dict] | None = None,
    gap: float = 2.0,
) -> list[dict]:
    updates = [0] + [141 * index for index in range(1, 151)] + [21171]
    rows = []
    for index, update in enumerate(updates):
        h = final_h - 0.001 * (151 - index)
        u = h + gap / 2.0
        s = h - gap / 2.0
        full = {"U": u, "S": s, "H": h, "ZS": 85.0 + index * 0.001}
        off = full if module_off_history is None else {
            metric: float(module_off_history[index][metric]) for metric in ("U", "S", "H", "ZS")
        }
        row = {
            **full,
            "update": update,
            "evaluation_index": index,
            "model_state_sha256": f"{index:064x}",
            "module_off_metrics": off,
            "full_minus_off_delta": {
                metric: float(full[metric]) - float(off[metric])
                for metric in ("U", "S", "H", "ZS")
            },
        }
        rows.append(row)
    return rows


def _fake_result(
    experiment_id: str, module: str, history: list[dict],
) -> dict:
    best = history[-1]
    config_path = ROOT / f"config/tries/v3_try_{experiment_id[-3:]}_fresh_effective.yaml"
    return {
        "experiment_id": experiment_id,
        "module": module,
        "code_commit": EXPECTED_COMMIT,
        "config_sha256": sha256_file(config_path),
        "initialization_strategy": "fresh_seeded_tg",
        "loaded_training_checkpoints": [],
        "asset_id": "CUB_openai_vitl14_336_dynamic_v3_v1",
        "asset_manifest_sha256": "3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6",
        "initial_tg_state_sha256": "a" * 64,
        "initial_parent_state_sha256": "b" * 64,
        "primary_batch_generator_initial_sha256": "c" * 64,
        "primary_batches_updates_1_142_sha256": "d" * 64,
        "post_build_cuda_rng_sha256": "e" * 64,
        "update1_pre_main_cuda_rng_sha256": "f" * 64,
        "random_seed": 7,
        "batch_size": 50,
        "total_updates": 21171,
        "eval_interval_steps": 141,
        "tg_learning_rate": 1e-4,
        "history_length": 152,
        "best_metrics": best,
        "best_update": 21171,
        "best_zs_observation": {"ZS": best["ZS"], "update": 21171, "metrics": best},
        "best_full_minus_off_delta": best["full_minus_off_delta"],
    }


def test_summary_enforces_cross_run_and_same_checkpoint_double_gate():
    control_history = _fake_history(final_h=76.5)
    gtd_history = _fake_history(final_h=77.6, module_off_history=control_history)
    mmt_history = _fake_history(final_h=77.35, module_off_history=control_history)
    bd_history = _fake_history(final_h=77.6, module_off_history=control_history, gap=9.0)
    payloads = [
        _fake_result("V3-TRY-042", "tg", control_history),
        _fake_result("V3-TRY-043", "gtd", gtd_history),
        _fake_result("V3-TRY-044", "mmt", mmt_history),
        _fake_result("V3-TRY-045", "bd", bd_history),
    ]
    histories = {
        row["experiment_id"]: history
        for row, history in zip(payloads, (control_history, gtd_history, mmt_history, bd_history))
    }
    result = summarize(
        payloads, histories, expected_code_commit=EXPECTED_COMMIT, repo_root=ROOT
    )
    assert [row["decision"] for row in result["candidates"]] == [
        "strong_keep", "weak_keep", "drop"
    ]
    broken = copy.deepcopy(payloads)
    broken[-1]["primary_batches_updates_1_142_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="不匹配"):
        summarize(broken, histories, expected_code_commit=EXPECTED_COMMIT, repo_root=ROOT)
    trajectory_broken = copy.deepcopy(histories)
    trajectory_broken["V3-TRY-044"][17]["module_off_metrics"]["H"] += 0.01
    trajectory_broken["V3-TRY-044"][17]["full_minus_off_delta"]["H"] = (
        trajectory_broken["V3-TRY-044"][17]["H"]
        - trajectory_broken["V3-TRY-044"][17]["module_off_metrics"]["H"]
    )
    invalid = summarize(
        payloads, trajectory_broken,
        expected_code_commit=EXPECTED_COMMIT, repo_root=ROOT,
    )
    assert invalid["candidates"][1]["decision"] == "implementation_invalid"


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


def _checkpoint_row(
    *, update: int, index: int, h: float, zs: float, state: dict[str, torch.Tensor],
) -> dict:
    full = {"U": h + 1.0, "S": h - 1.0, "H": h, "ZS": zs}
    return {
        **full,
        "update": update,
        "evaluation_index": index,
        "module_off_metrics": copy.deepcopy(full),
        "full_minus_off_delta": {name: 0.0 for name in ("U", "S", "H", "ZS")},
        "model_state_sha256": canonical_sha256(state),
    }


def _valid_tg_checkpoint():
    torch.random.default_generator.manual_seed(311)
    model = torch.nn.Linear(3, 2)
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    current_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    schedule = FreshSchedule(optimizer)
    schedule.set_for_update(141)
    primary = torch.Generator().manual_seed(7)
    auxiliary = torch.Generator().manual_seed(7151)
    history = [
        _checkpoint_row(update=0, index=0, h=70.0, zs=80.0, state=initial_state),
        _checkpoint_row(update=141, index=1, h=71.0, zs=81.0, state=current_state),
    ]
    initial_identity = {
        "initialization_strategy": "fresh_seeded_tg",
        "initial_tg_state_sha256": "a" * 64,
        "post_build_cuda_rng_sha256": "b" * 64,
    }
    checkpoint = {
        "experiment_id": "V3-TRY-042",
        "code_commit": EXPECTED_COMMIT,
        "config_sha256": "c" * 64,
        "initial_identity": initial_identity,
        "update": 141,
        "model_state_dict": current_state,
        "tg_optimizer_state_dict": optimizer.state_dict(),
        "gate_optimizer_state_dict": None,
        "scheduler_state_dict": schedule.state_dict(),
        "best_update": 141,
        "best_metrics": copy.deepcopy(history[1]),
        "best_model_state_dict": copy.deepcopy(current_state),
        "best_zs_observation": {
            "ZS": 81.0, "update": 141, "metrics": copy.deepcopy(history[1])
        },
        "best_zs_model_state_dict": copy.deepcopy(current_state),
        "teacher_state": None,
        "teacher_history": [],
        "first_update_gradients": {
            "update1_pre_main_cuda_rng_sha256": "d" * 64,
            "tg": {"any_nonzero_gradient": True},
            "module": None,
        },
        "rng_state": {
            "primary": primary.get_state(),
            "auxiliary": auxiliary.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "history": history,
        "reproducibility": {},
    }
    seal_checkpoint(checkpoint)
    return checkpoint, initial_identity


def test_checkpoint_semantic_validation_rejects_tampering_and_positive_resume():
    checkpoint, initial_identity = _valid_tg_checkpoint()
    validate_checkpoint(
        checkpoint, module_name="tg", experiment_id="V3-TRY-042",
        code_commit=EXPECTED_COMMIT, config_sha="c" * 64,
        initial_identity=initial_identity,
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    schedule = FreshSchedule(optimizer)
    primary = torch.Generator().manual_seed(1)
    auxiliary = torch.Generator().manual_seed(2)
    saved_cpu_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all()
    try:
        restore_checkpoint_objects(
            checkpoint, model=model, tg_optimizer=optimizer, gate_optimizer=None,
            scheduler=schedule, primary_generator=primary,
            auxiliary_generator=auxiliary,
        )
    finally:
        torch.set_rng_state(saved_cpu_rng)
        torch.cuda.set_rng_state_all(saved_cuda_rng)
    assert canonical_sha256(model.state_dict()) == checkpoint["canonical_digests"]["model"]

    tampered_best = copy.deepcopy(checkpoint)
    tampered_best["best_metrics"]["H"] -= 1.0
    seal_checkpoint(tampered_best)
    with pytest.raises(ValueError, match="best-H"):
        validate_checkpoint(
            tampered_best, module_name="tg", experiment_id="V3-TRY-042",
            code_commit=EXPECTED_COMMIT, config_sha="c" * 64,
            initial_identity=initial_identity,
        )
    tampered_history = copy.deepcopy(checkpoint)
    tampered_history["history"][1]["update"] = 140
    seal_checkpoint(tampered_history)
    with pytest.raises(ValueError, match="history update"):
        validate_checkpoint(
            tampered_history, module_name="tg", experiment_id="V3-TRY-042",
            code_commit=EXPECTED_COMMIT, config_sha="c" * 64,
            initial_identity=initial_identity,
        )
    tampered_teacher = copy.deepcopy(checkpoint)
    tampered_teacher["teacher_history"] = [{"update": 1, "sha256": "e" * 64}]
    tampered_teacher["teacher_state"] = {"fake": torch.zeros(1)}
    seal_checkpoint(tampered_teacher)
    with pytest.raises(ValueError, match="非teacher模块"):
        validate_checkpoint(
            tampered_teacher, module_name="tg", experiment_id="V3-TRY-042",
            code_commit=EXPECTED_COMMIT, config_sha="c" * 64,
            initial_identity=initial_identity,
        )
    tampered_scheduler = copy.deepcopy(checkpoint)
    tampered_scheduler["scheduler_state_dict"]["last_update"] = 140
    seal_checkpoint(tampered_scheduler)
    with pytest.raises(ValueError, match="scheduler"):
        validate_checkpoint(
            tampered_scheduler, module_name="tg", experiment_id="V3-TRY-042",
            code_commit=EXPECTED_COMMIT, config_sha="c" * 64,
            initial_identity=initial_identity,
        )
    tampered_digest = copy.deepcopy(checkpoint)
    tampered_digest["semantic_evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="semantic evidence"):
        validate_checkpoint(
            tampered_digest, module_name="tg", experiment_id="V3-TRY-042",
            code_commit=EXPECTED_COMMIT, config_sha="c" * 64,
            initial_identity=initial_identity,
        )


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
