from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

from model.innovations.train_fresh_effective import (
    SCHEMA,
    build_model,
    candidate_logits,
    canonical_sha256,
    evaluate,
    load_config,
)


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
        "gtd": ("V3-TRY-046", "TG_PLUS_GTD_FRESH_CONTROL", "IDEA-155"),
        "lver": ("V3-TRY-047", "TG_PLUS_GTD_PLUS_LVER_FRESH", "IDEA-156"),
        "pcpc": ("V3-TRY-048", "TG_PLUS_GTD_PLUS_PCPC_FRESH", "IDEA-157"),
    }
    experiment_id, condition_id, idea_id = identities[module]
    config.update(
        {
            "schema_version": SCHEMA,
            "experiment_id": experiment_id,
            "condition_id": condition_id,
            "idea_id": idea_id,
            "module": module,
            "initialization_strategy": "fresh_seeded_tg_gtd_visual",
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
