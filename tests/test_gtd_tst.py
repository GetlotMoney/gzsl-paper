from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from model.innovations.elpt import fixed_class_folds
from model.innovations.gtd_tst import (
    GTDTSTModel,
    closed_form_alignment_angle,
    geodesic_geometry,
    geodesic_points,
    select_oracle_targets,
)
from model.innovations.train_gtd_tst import (
    GroupwiseSchedule,
    TEACHER_REFRESH_UPDATES,
    _predict,
    evaluation_updates,
    gtd_screen_decision,
    load_config,
    refresh_oracle_targets,
    teacher_packages_sha256,
    teacher_refresh_record,
)
from model.paper_v2 import PaperV2ThreeModuleModel


def _basis(index: int) -> torch.Tensor:
    value = torch.zeros(768)
    value[index] = 1.0
    return value


def _parent(seen: torch.Tensor | None = None) -> PaperV2ThreeModuleModel:
    generator = torch.Generator().manual_seed(22022)
    roles = F.normalize(torch.randn(200, 8, 768, generator=generator), dim=-1)
    seen = torch.arange(150) if seen is None else torch.as_tensor(seen).long()
    centroids = F.normalize(torch.randn(150, 768, generator=generator), dim=-1)
    return PaperV2ThreeModuleModel(
        roles,
        seen,
        centroids,
        tg_vpr_mode="full",
        transport_mode="off",
        ccgr_mode="off",
        dropout=0.5,
        inner_ratio=0.35,
        outer_ratio=0.65,
        temperature=0.07,
    )


def test_exact_geodesic_and_closed_form_boundaries():
    mean8 = torch.stack((_basis(0), _basis(0), _basis(0)))
    angle = torch.tensor([0.5, 0.5, math.pi])
    value = torch.stack(
        (
            math.cos(0.5) * _basis(0) + math.sin(0.5) * _basis(1),
            math.cos(0.5) * _basis(0) + math.sin(0.5) * _basis(1),
            -_basis(0),
        )
    )
    support = F.normalize(torch.randn(8, 768, generator=torch.Generator().manual_seed(2)), dim=-1)
    geometry = geodesic_geometry(mean8, value, support)
    assert geometry.valid.tolist() == [True, True, False]
    theta = torch.tensor([0.0, 0.25, 0.0])
    points = geodesic_points(geometry.mean8, geometry.direction, theta)
    assert torch.equal(points[0], geometry.mean8[0])
    assert torch.allclose(points.norm(dim=-1), torch.ones(3), atol=1e-6)
    measured = torch.acos((points[1] * geometry.mean8[1]).sum().clamp(-1.0, 1.0))
    assert torch.allclose(measured, torch.tensor(0.25), atol=1e-6)
    visual = torch.stack((_basis(0), _basis(1), _basis(0)))
    closed = closed_form_alignment_angle(
        visual,
        geometry.mean8,
        geometry.direction,
        geometry.angle_limit,
    )
    assert float(closed[0]) == 0.0
    assert torch.allclose(closed[1], geometry.angle_limit[1])
    assert float(closed[2]) == 0.0


def test_oracle_all_ties_and_same_direction_degeneracy_choose_zero():
    theta_grid = torch.linspace(0.0, 0.5, 33).repeat(2, 1)
    objective = torch.ones_like(theta_grid)
    selected, gain, useful = select_oracle_targets(
        theta_grid, objective, torch.tensor([True, False])
    )
    assert torch.equal(selected, torch.zeros(2))
    assert torch.equal(gain, torch.zeros(2))
    assert useful.tolist() == [False, False]
    base = torch.stack((_basis(0), _basis(1)))
    support = F.normalize(torch.randn(6, 768, generator=torch.Generator().manual_seed(9)), dim=-1)
    geometry = geodesic_geometry(base, base.clone(), support)
    assert geometry.valid.tolist() == [False, False]
    assert torch.equal(geometry.angle_limit, torch.zeros(2))


def test_zero_gate_strictly_reproduces_tg_and_moves_only_unseen():
    parent = _parent().eval()
    model = GTDTSTModel(parent, torch.arange(150)).eval()
    bundle = model.prototype_bundle()
    assert torch.equal(bundle["final"], bundle["parent"])
    images = F.normalize(torch.randn(4, 768, generator=torch.Generator().manual_seed(7)), dim=-1)
    assert torch.equal(model.logits(images), parent.logits(images))
    unseen = torch.arange(150, 200)
    assert torch.equal(model.logits(images, unseen), parent.logits(images, unseen))
    with torch.no_grad():
        model.gate.network[-1].bias.fill_(0.25)
    moved = model.prototype_bundle()
    assert torch.equal(
        moved["final"].index_select(0, torch.arange(150)),
        moved["parent"].index_select(0, torch.arange(150)),
    )
    assert not torch.equal(
        moved["final"].index_select(0, unseen),
        moved["parent"].index_select(0, unseen),
    )


def test_seen_teacher_is_detached_and_smooth_l1_updates_only_gate():
    model = GTDTSTModel(_parent(), torch.arange(150)).train()
    with torch.no_grad():
        centroids = model.parent.tg_vpr.value_candidate(torch.arange(150))
    packages = refresh_oracle_targets(model, centroids, fixed_class_folds(torch.arange(150)), 0.1)
    assert len(packages) == 3
    coverage = torch.cat([item["class_ids"].cpu() for item in packages]).sort().values
    assert torch.equal(coverage, torch.arange(150))
    for package in packages:
        assert not package["features"].requires_grad
        assert not package["target_ratio"].requires_grad
        assert bool(((0.0 <= package["target_ratio"]) & (package["target_ratio"] <= 1.0)).all())
        assert bool((package["oracle_gain"] >= -1e-6).all())
    for parameter in model.parameters():
        parameter.grad = None
    positives = [package for package in packages if bool(package["move_mask"].any())]
    assert positives, "受控Value中心必须生成至少一个真实正theta target"
    package = positives[0]
    mask = package["move_mask"]
    raw = model.gate.raw_ratio(package["features"])
    F.smooth_l1_loss(raw[mask], package["target_ratio"][mask]).backward()
    gate_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.gate.parameters()
        if parameter.grad is not None
    )
    parent_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parent.parameters()
        if parameter.grad is not None
    )
    assert gate_gradient > 0.0
    assert parent_gradient == 0.0
    for parameter in model.parameters():
        parameter.grad = None
    images = F.normalize(torch.randn(4, 768, generator=torch.Generator().manual_seed(31)), dim=-1)
    logits = model.parent.logits(images, torch.arange(150))
    F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert any(parameter.grad is not None for parameter in model.parent.parameters())
    assert all(parameter.grad is None for parameter in model.gate.parameters())


def test_noncontiguous_global_seen_and_zs_ids_are_preserved():
    seen = torch.tensor([value for value in range(200) if value % 4 != 0])
    unseen = torch.tensor([value for value in range(200) if value % 4 == 0])
    assert seen.numel() == 150 and unseen.numel() == 50
    parent = _parent(seen).eval()
    model = GTDTSTModel(parent, seen).eval()
    bundle = model.prototype_bundle()
    assert torch.equal(bundle["final"], bundle["parent"])
    chosen = unseen[:8]
    features = bundle["final"].index_select(0, chosen).detach()
    predictions = _predict(
        features,
        bundle["final"].detach(),
        model.scale().detach(),
        torch.device("cpu"),
        unseen,
        batch_size=3,
    )
    assert torch.equal(predictions, chosen)


def test_fixed150_schedule_evaluations_and_config_contract():
    tg = torch.nn.Parameter(torch.tensor(1.0))
    gate = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(
        [{"params": [tg], "lr": 1e-5}, {"params": [gate], "lr": 1e-4}]
    )
    schedule = GroupwiseSchedule(
        optimizer,
        total_updates=21171,
        warmup_updates=705,
        tg_min_multiplier=0.1,
        gate_min_multiplier=0.1,
    )
    schedule.set_for_update(1)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-5]
    schedule.set_for_update(705)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    state = schedule.state_dict()
    schedule.set_for_update(706)
    schedule.load_state_dict(state)
    assert schedule.last_update == 705
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    generator = torch.Generator(device="cpu").manual_seed(7)
    _ = torch.randperm(7057, generator=generator)[:50]
    generator_state = generator.get_state()
    expected_next = torch.randperm(7057, generator=generator)[:50]
    restored_generator = torch.Generator(device="cpu")
    restored_generator.set_state(generator_state)
    assert torch.equal(
        expected_next,
        torch.randperm(7057, generator=restored_generator)[:50],
    )
    torch.manual_seed(77)
    cpu_state = torch.get_rng_state()
    expected_random = torch.rand(4)
    torch.set_rng_state(cpu_state)
    assert torch.equal(expected_random, torch.rand(4))
    schedule.set_for_update(21171)
    assert [round(group["lr"], 12) for group in optimizer.param_groups] == [1e-6, 1e-5]
    points = evaluation_updates()
    assert len(points) == 151
    assert points[-2:] == (21150, 21171)
    assert len(TEACHER_REFRESH_UPDATES) == 150
    assert TEACHER_REFRESH_UPDATES[:2] == (1, 142)
    assert TEACHER_REFRESH_UPDATES[-1] == 21010
    config, _ = load_config(Path("config/tries/v3_try_022_gtd_tst_fixed150.yaml"))
    assert config["experiment_id"] == "V3-TRY-022"
    assert config["unseen_images_used_for_gradient"] is False
    assert config["early_stopping_enabled"] is False
    assert gtd_screen_decision(0.799999, 1.0) == "drop_fixed_150"
    assert gtd_screen_decision(0.8, 1.0) == "pending_matched_try020_comparison"
    assert gtd_screen_decision(1.0, 1.0) == "pending_matched_try020_comparison"


def test_teacher_refresh_record_binds_model_package_folds_and_targets():
    model = GTDTSTModel(_parent(), torch.arange(150)).eval()
    folds = fixed_class_folds(torch.arange(150))
    with torch.no_grad():
        centers = model.parent.tg_vpr.value_candidate(torch.arange(150))
    packages = refresh_oracle_targets(model, centers, folds, 0.1)
    record = teacher_refresh_record(update=1, model=model, packages=packages, folds=folds)
    assert record["update"] == 1
    assert len(record["model_state_sha256"]) == 64
    assert record["package_sha256"] == teacher_packages_sha256(packages)
    assert len(record["fold_pseudo_unseen_class_ids"]) == 3
    assert len(record["class_ids"]) == 150
    assert len(record["target_ratio"]) == 150
    assert len(record["move_mask"]) == 150
    assert len(record["oracle_gain"]) == 150
