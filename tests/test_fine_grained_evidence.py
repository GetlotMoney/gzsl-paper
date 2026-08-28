from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from model.innovations.elpt import fixed_class_folds
from model.innovations.pcpc import pairwise_hard_negative_loss
from model.innovations.train_fresh_effective import (
    BATCH_SIZE,
    CLASS_COUNT,
    SEEN_COUNT,
    SCHEMA,
    FreshSchedule,
    _visual_batch,
    build_model,
    candidate_logits,
    canonical_sha256,
    evaluate,
    gradient_report,
    load_config,
    load_visual_assets,
    restore_checkpoint_objects,
)
from model.innovations.train_gtd_tst import load_assets, refresh_oracle_targets
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility


ROOT = Path(__file__).resolve().parents[1]


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


def screen_config(module: str) -> dict:
    config = yaml.safe_load(
        (ROOT / "config/tries/v3_try_043_fresh_effective.yaml").read_text(encoding="utf-8")
    )
    identities = {
        "gtd": ("V3-TRY-048", "TG_PLUS_GTD_FRESH_CONTROL", "IDEA-155"),
        "lver": ("V3-TRY-049", "TG_PLUS_GTD_PLUS_LVER_FRESH", "IDEA-156"),
        "pcpc": ("V3-TRY-050", "TG_PLUS_GTD_PLUS_PCPC_FRESH", "IDEA-157"),
    }
    experiment_id, condition_id, idea_id = identities[module]
    config.update(
        {
            "schema_version": SCHEMA,
            "experiment_id": experiment_id,
            "condition_id": condition_id,
            "idea_id": idea_id,
            "module": module,
            "initialization_strategy": (
                "fresh_seeded_tg_gtd" if module == "gtd" else "fresh_seeded_tg_gtd_visual"
            ),
            "lver_asset_manifest": "/asset/lver/asset_manifest.json" if module == "lver" else None,
            "lver_asset_manifest_sha256": "1" * 64 if module == "lver" else None,
            "lver_asset_id": "lver-fixed" if module == "lver" else None,
            "pcpc_asset_manifest": "/asset/pcpc/asset_manifest.json" if module == "pcpc" else None,
            "pcpc_asset_manifest_sha256": "2" * 64 if module == "pcpc" else None,
            "pcpc_asset_id": "pcpc-fixed" if module == "pcpc" else None,
            "visual_loss_weight": 1.0,
            "lver_hidden_dim": 16,
            "lver_margin_threshold": 0.25,
            "lver_margin_temperature": 0.1,
            "lver_local_temperature": 0.07,
            "lver_max_strength": 5.0,
            "pcpc_rank": 32,
            "pcpc_patch_temperature": 0.07,
            "pcpc_max_logit_correction": 1.0,
            "pcpc_pair_margin": 0.02,
        }
    )
    return config


def test_three_screen_configs_are_exact_and_reject_cross_asset(tmp_path: Path):
    for module in ("gtd", "lver", "pcpc"):
        path = tmp_path / f"{module}.yaml"
        path.write_text(yaml.safe_dump(screen_config(module), sort_keys=False), encoding="utf-8")
        loaded, _ = load_config(path)
        assert loaded["module"] == module
    broken = screen_config("lver")
    broken["pcpc_asset_manifest"] = "/wrong"
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(broken, sort_keys=False), encoding="utf-8")
    try:
        load_config(path)
    except ValueError:
        pass
    else:
        raise AssertionError("LVER配置不得同时绑定PCPC资产。")


def test_legacy_tg_control_stays_on_nonvisual_asset_and_prototype_evaluation_path():
    config, _ = load_config(ROOT / "config/tries/v3_try_042_fresh_effective.yaml")
    tensors = synthetic_assets()
    assert load_visual_assets(config, tensors) is tensors
    torch.manual_seed(7)
    bundle = build_model(config, tensors, torch.device("cpu"))
    metrics = evaluate(bundle, tensors, torch.device("cpu"))
    assert metrics["full_minus_off_delta"] == {"U": 0.0, "S": 0.0, "H": 0.0, "ZS": 0.0}


def test_candidates_share_exact_fresh_tg_and_gtd_initialization():
    tensors = synthetic_assets()
    identities = {}
    for module in ("gtd", "lver", "pcpc"):
        torch.manual_seed(7)
        bundle = build_model(screen_config(module), tensors, torch.device("cpu"))
        identities[module] = {
            "parent": canonical_sha256(bundle.parent.state_dict()),
            "gate": canonical_sha256(bundle.model.gate.state_dict()),
        }
    assert identities["gtd"] == identities["lver"] == identities["pcpc"]


def test_candidate_off_is_exact_gtd_and_metrics_have_full_off_contract():
    tensors = synthetic_assets()
    generator = torch.Generator().manual_seed(77)
    for module in ("lver", "pcpc"):
        torch.manual_seed(7)
        bundle = build_model(screen_config(module), tensors, torch.device("cpu"))
        bundle.model.eval()
        images = tensors["test_seen_features"][:4]
        visual = (
            torch.randn(4, 4, 768, generator=generator)
            if module == "lver"
            else torch.randn(4, 6, 768, generator=generator)
        )
        off = candidate_logits(bundle, images, visual, enabled=False)
        base = candidate_logits(bundle, images, visual, enabled=True)
        # Both candidates start with an exactly neutral scalar strength.
        assert torch.equal(off, base)

        payload = copy.copy(tensors)
        if module == "lver":
            payload.update(
                {
                    "test_seen_local_views": torch.randn(150, 4, 768, generator=generator),
                    "test_unseen_local_views": torch.randn(50, 4, 768, generator=generator),
                }
            )
        else:
            payload.update(
                {
                    "test_seen_patches": torch.randn(150, 6, 768, generator=generator),
                    "test_unseen_patches": torch.randn(50, 6, 768, generator=generator),
                }
            )
        metrics = evaluate(bundle, payload, torch.device("cpu"))
        assert set(metrics) == {"U", "S", "H", "ZS", "module_off_metrics", "full_minus_off_delta"}
        assert metrics["full_minus_off_delta"] == {"U": 0.0, "S": 0.0, "H": 0.0, "ZS": 0.0}


@pytest.mark.parametrize("module", ["lver", "pcpc"])
def test_visual_candidate_checkpoint_roundtrip_restores_next_batch_and_lr(
    module: str, tmp_path: Path
):
    torch.manual_seed(7)
    bundle = build_model(screen_config(module), synthetic_assets(), torch.device("cpu"))
    tg_parameters = list(bundle.parent.parameter_groups()["tg_vpr"])
    module_parameters = bundle.module_parameters()
    tg_optimizer = torch.optim.Adam(tg_parameters, lr=1e-4, weight_decay=1e-4)
    gate_optimizer = torch.optim.Adam(module_parameters, lr=1e-4, weight_decay=1e-4)
    scheduler = FreshSchedule(tg_optimizer, gate_optimizer)
    scheduler.set_for_update(17)
    loss = sum(parameter.square().mean() for parameter in tg_parameters)
    loss = loss + sum(parameter.square().mean() for parameter in module_parameters)
    loss.backward()
    tg_optimizer.step()
    gate_optimizer.step()
    primary = torch.Generator().manual_seed(7)
    rng_state = {
        "primary": primary.get_state().clone(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    optimizer_payload = {
        "tg": tg_optimizer.state_dict(),
        "gate": gate_optimizer.state_dict(),
    }
    checkpoint = {
        "model_state_dict": copy.deepcopy(bundle.model.state_dict()),
        "tg_optimizer_state_dict": optimizer_payload["tg"],
        "gate_optimizer_state_dict": optimizer_payload["gate"],
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": rng_state,
        "canonical_digests": {
            "model": canonical_sha256(bundle.model.state_dict()),
            "optimizer": canonical_sha256(optimizer_payload),
            "rng": canonical_sha256(rng_state),
        },
    }
    path = tmp_path / f"{module}.pth"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    expected_generator = torch.Generator()
    expected_generator.set_state(rng_state["primary"])
    expected_batch = torch.randperm(7057, generator=expected_generator)[:BATCH_SIZE]
    with torch.no_grad():
        next(bundle.model.parameters()).add_(1.0)
    scheduler.set_for_update(18)
    restore_checkpoint_objects(
        loaded,
        model=bundle.model,
        tg_optimizer=tg_optimizer,
        gate_optimizer=gate_optimizer,
        scheduler=scheduler,
        primary_generator=primary,
    )
    actual_batch = torch.randperm(7057, generator=primary)[:BATCH_SIZE]
    assert torch.equal(actual_batch, expected_batch)
    scheduler.set_for_update(18)
    assert [group["lr"] for group in tg_optimizer.param_groups] == [1e-4]
    assert gate_optimizer.param_groups[0]["lr"] == scheduler.learning_rates(18)[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one physical CUDA GPU")
@pytest.mark.parametrize("attempt,module", [("049", "lver"), ("050", "pcpc")])
def test_real_gpu_candidate_microbatch(attempt: str, module: str):
    config_path = ROOT / f"config/tries/v3_try_{attempt}_fine_grained_evidence.yaml"
    config, _ = load_config(config_path)
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    device = torch.device("cuda:0")
    tensors = load_visual_assets(config, load_assets(config))
    bundle = build_model(config, tensors, device)
    labels_cpu = tensors["train_labels"].long()
    seen = bundle.parent.seen_classes.to(device)
    global_to_seen = torch.full((CLASS_COUNT,), -1, dtype=torch.long, device=device)
    global_to_seen[seen] = torch.arange(SEEN_COUNT, device=device)
    centroids = h1.visual_centroids(
        tensors["train_features"], labels_cpu, bundle.parent.seen_classes.cpu()
    ).to(device)
    teacher = refresh_oracle_targets(
        bundle.model,
        centroids,
        fixed_class_folds(bundle.parent.seen_classes.cpu()),
        float(config["gtd_theta_penalty"]),
    )
    indices_cpu = torch.randperm(7057, generator=torch.Generator().manual_seed(7))[:BATCH_SIZE]
    indices = indices_cpu.to(device)
    images = tensors["train_features"].index_select(0, indices_cpu).to(device).float()
    labels = labels_cpu.index_select(0, indices_cpu).to(device)
    targets = global_to_seen.index_select(0, labels)
    parent_seen = bundle.parent.prototypes().index_select(0, seen)
    parent_logits = (
        F.normalize(images, dim=-1)
        @ F.normalize(parent_seen, dim=-1).T
        * bundle.parent.scale()
    )
    main = F.cross_entropy(parent_logits, targets) + 0.1 * bundle.parent.topology_loss()
    package = teacher[0]
    gtd = F.smooth_l1_loss(
        bundle.model.gate.raw_ratio(package["features"]), package["target_ratio"]
    )
    visual_key = "train_local_views" if module == "lver" else "train_patches"
    visual = _visual_batch(tensors[visual_key], indices_cpu, device)
    if module == "lver":
        corrected = bundle.visual(
            parent_logits.detach(), visual, parent_seen.detach(), images.detach()
        )
        visual_loss = F.cross_entropy(corrected, targets)
    else:
        role_text = bundle.parent.tg_vpr.sentence_embeds.index_select(0, seen).detach()
        corrected = bundle.visual(parent_logits.detach(), visual, role_text)
        visual_loss = pairwise_hard_negative_loss(
            corrected, targets, torch.arange(SEEN_COUNT, device=device), margin=0.02
        )
    total = main + gtd + visual_loss
    total.backward()
    assert torch.isfinite(total)
    tg_report = gradient_report(list(bundle.parent.parameter_groups()["tg_vpr"]))
    module_report = gradient_report(bundle.module_parameters())
    assert tg_report["any_nonzero_gradient"]
    assert module_report["all_gradients_present"]
    assert module_report["any_nonzero_gradient"]
