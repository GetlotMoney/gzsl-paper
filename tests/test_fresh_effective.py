from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from model.innovations.elpt import fixed_class_folds
from model.innovations.train_fresh_effective import (
    FreshSchedule,
    build_model,
    canonical_sha256,
    evaluation_updates,
    full_and_off_prototypes,
    gradient_report,
    load_config,
    primary_batch_prefix_sha256,
    restore_checkpoint_objects,
    seal_checkpoint,
    teacher_refresh_updates,
    validate_checkpoint,
)
from model.innovations.train_gtd_tst import tensor_mapping_sha256
from model.innovations.train_gtd_tst import refresh_oracle_targets
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / f"config/tries/v3_try_{attempt}_fresh_effective.yaml"
    for attempt in ("042", "043")
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


def test_teacher_refresh_schedule_includes_final_partial_interval():
    updates = teacher_refresh_updates(21171)
    assert len(updates) == 151
    assert updates[:2] == (1, 142)
    assert updates[-2:] == (21010, 21151)


def test_real_reproducibility_payload_is_weights_only_safe(tmp_path: Path):
    payload = configure_reproducibility(
        7, strict_determinism=True, deterministic_warn_only=False
    )
    assert type(payload["torch_version"]) is str
    assert type(payload["cuda_version"]) is str
    path = tmp_path / "reproducibility.pth"
    torch.save({"reproducibility": payload}, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored == {"reproducibility": payload}


def test_completed_teacher_checkpoint_validates_and_restores_for_finalization():
    checkpoint, initial_identity = _valid_tg_checkpoint()
    state = checkpoint["model_state_dict"]
    updates = (0, *evaluation_updates())
    history = [
        _checkpoint_row(update=update, index=index, h=70.0, zs=80.0, state=state)
        for index, update in enumerate(updates)
    ]
    teacher_state = {"payload": torch.zeros(1)}
    teacher_sha = canonical_sha256(teacher_state)
    optimizer_model = torch.nn.Linear(3, 2)
    optimizer_model.load_state_dict(state)
    completed_tg_optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=1e-4)
    completed_gate_optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=1e-4)
    completed_schedule = FreshSchedule(completed_tg_optimizer, completed_gate_optimizer)
    completed_schedule.set_for_update(21171)
    checkpoint.update({
        "experiment_id": "V3-TRY-043",
        "update": 21171,
        "scheduler_state_dict": completed_schedule.state_dict(),
        "tg_optimizer_state_dict": completed_tg_optimizer.state_dict(),
        "gate_optimizer_state_dict": completed_gate_optimizer.state_dict(),
        "history": history,
        "best_update": 0,
        "best_metrics": copy.deepcopy(history[0]),
        "best_model_state_dict": copy.deepcopy(state),
        "best_zs_observation": {
            "ZS": 80.0, "update": 0, "metrics": copy.deepcopy(history[0])
        },
        "best_zs_model_state_dict": copy.deepcopy(state),
        "teacher_state": teacher_state,
        "teacher_history": [
            {"update": update, "sha256": teacher_sha}
            for update in teacher_refresh_updates(21171)
        ],
        "first_update_gradients": {
            **checkpoint["first_update_gradients"],
            "module": {"any_nonzero_gradient": True},
        },
    })
    seal_checkpoint(checkpoint)
    validate_checkpoint(
        checkpoint, module_name="gtd", experiment_id="V3-TRY-043",
        code_commit=EXPECTED_COMMIT, config_sha="c" * 64,
        initial_identity=initial_identity,
    )

    model = torch.nn.Linear(3, 2)
    tg_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gate_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    schedule = FreshSchedule(tg_optimizer, gate_optimizer)
    saved_cpu_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all()
    try:
        restore_checkpoint_objects(
            checkpoint, model=model, tg_optimizer=tg_optimizer,
            gate_optimizer=gate_optimizer, scheduler=schedule,
            primary_generator=torch.Generator(),
        )
    finally:
        torch.set_rng_state(saved_cpu_rng)
        torch.cuda.set_rng_state_all(saved_cuda_rng)
    assert schedule.last_update == 21171

    missing_refresh = copy.deepcopy(checkpoint)
    missing_refresh["teacher_history"].pop()
    seal_checkpoint(missing_refresh)
    with pytest.raises(ValueError, match="teacher update/schema"):
        validate_checkpoint(
            missing_refresh, module_name="gtd", experiment_id="V3-TRY-043",
            code_commit=EXPECTED_COMMIT, config_sha="c" * 64,
            initial_identity=initial_identity,
        )


def test_configs_define_only_fresh_one_stage_matched_conditions():
    loaded = [load_config(path)[0] for path in CONFIGS]
    assert [row["experiment_id"] for row in loaded] == ["V3-TRY-042", "V3-TRY-043"]
    assert [row["module"] for row in loaded] == ["tg", "gtd"]
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


def test_tg_and_gtd_have_identical_fresh_parent_rng_and_primary_batches():
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


def test_zero_initialized_gtd_is_exact_tg_module_off_path():
    tensors = synthetic_assets()
    config, _ = load_config(CONFIGS[1])
    torch.random.default_generator.manual_seed(7)
    bundle = build_model(config, tensors, torch.device("cpu"))
    bundle.model.eval()
    full, off = full_and_off_prototypes(bundle)
    prototype_bundle = bundle.model.prototype_bundle()
    assert torch.count_nonzero(prototype_bundle["theta"]) == 0
    assert torch.equal(full, off)
    images = tensors["test_unseen_features"][:4]
    assert torch.equal(
        F.normalize(images.float(), dim=-1) @ full.T * bundle.parent.scale(),
        F.normalize(images.float(), dim=-1) @ off.T * bundle.parent.scale(),
    )


def test_tg_and_gtd_share_parent_forward_and_gtd_gate_gets_first_step_gradient():
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
        else:
            auxiliary = main.new_zeros(())
        (main + auxiliary).backward()
        tg_report = gradient_report(list(bundle.parent.parameter_groups()["tg_vpr"]))
        assert tg_report["any_nonzero_gradient"]
        if module_parameters:
            report = gradient_report(module_parameters)
            assert report["all_gradients_present"] and report["any_nonzero_gradient"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="server CUDA RNG专项")
def test_cuda_tg_and_gtd_preserve_rng_teacher_and_first_parent_step():
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
    saved_cpu_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all()
    try:
        restore_checkpoint_objects(
            checkpoint, model=model, tg_optimizer=optimizer, gate_optimizer=None,
            scheduler=schedule, primary_generator=primary,
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
