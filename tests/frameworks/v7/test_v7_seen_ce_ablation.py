from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.frameworks.v6.compiled_pclr import EMBED_DIM, ROLE_COUNT, CompiledPCLRHead, initialized_reader_states
from model.frameworks.v7 import train_one_text_seen_ce as tune013
from model.frameworks.v7 import train_seen_ce_ablation as ablation


def _base_config(tmp_path: Path, condition: str) -> dict:
    identity = tune013.IDENTITIES["CUB"]
    source = tmp_path / "CUB_source.yaml"
    formal = tmp_path / "CUB_formal.pth"
    full = tmp_path / "RUN-CUB.yaml"
    metrics = tmp_path / "metrics.json"
    model = tmp_path / "model_best.pth"
    source.write_text("dataset: CUB\n", encoding="utf-8")
    formal.write_bytes(b"formal")
    full.write_text("full: reference\n", encoding="utf-8")
    metrics.write_text('{"H": 79.94579718163422}\n', encoding="utf-8")
    model.write_bytes(b"model")
    contract = ablation.CONDITIONS[condition]
    return {
        "schema_version": ablation.SCHEMA,
        "ablation_experiment_id": ablation.EXPERIMENT_ID,
        "experiment_id": f"{ablation.EXPERIMENT_ID}-{contract['run_id']}",
        "condition": condition,
        "run_id": contract["run_id"],
        "dataset": "CUB",
        "base_commit": tune013.BASE_COMMIT,
        "source_config": str(source.resolve()),
        "source_config_sha256": identity["source_config_sha256"],
        "formal_checkpoint": str(formal.resolve()),
        "formal_checkpoint_sha256": "f" * 64,
        "formal_checkpoint_usage": "baseline_identity_only_not_training_initialization",
        "formal_full_metrics_percent": {"U": 1.0, "S": 2.0, "H": 1.3333333333333333, "ZS": 3.0},
        "full_reference_tune_experiment_id": "V7-TUNE-013-CUB-ONE-TEXT-SEEN-CE",
        "full_reference_code_commit": ablation.FULL_REFERENCE_CODE_COMMIT,
        "full_reference_config": str(full.resolve()),
        "full_reference_config_sha256": ablation.FULL_REFERENCE_CONFIG_SHA,
        "full_reference_metrics_percent": {"H": ablation.FULL_REFERENCE_H},
        "full_reference_metrics_uri": str(metrics.resolve()),
        "full_reference_metrics_sha256": ablation.FULL_REFERENCE_METRICS_SHA,
        "full_reference_model_uri": str(model.resolve()),
        "full_reference_model_sha256": ablation.FULL_REFERENCE_MODEL_SHA,
        "device": "cpu",
        "random_seed": 7,
        "batch_size": 50,
        "nominal_epochs": 200,
        "total_updates": identity["total_updates"],
        "eval_interval_steps": identity["eval_interval_steps"],
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-5,
        "weight_decay": 0.0,
        "relation_loss_weight": 1.0,
        "ridge_lambda": 0.3,
        "relation_temperature": 0.2,
        "direction_temperature": 0.07,
        "top_k": 3,
        "seen_logit_gamma": identity["seen_logit_gamma"],
        "alpha_max": 2.0,
        "initial_alpha": tune013.INITIAL_ALPHA,
        "role_weight_max": 1.0,
        "initial_role_weights": tune013.INITIAL_ROLE_WEIGHTS,
        "relation_embedding_mode": "one_text_uniform_role_difference",
        "relation_endpoint_scale": 0.5,
        "classification_ce_scope": "seen_only_train_classes",
        "expected_direction_skip_seen_class_ids": identity["direction_skip_seen_class_ids"],
        "best_selection_metric": "official_condition_H_post_update",
        "official_test_evaluations": math.ceil(identity["total_updates"] / identity["eval_interval_steps"]),
        "required_i_off_delta_h": 0.0,
        "required_v_off_delta_h": 0.0,
        "require_full_not_below_formal": False,
        "fresh_source_initialization": True,
        "test_used_for_selection": True,
        "test_used_for_hyperparameter_selection": True,
        "nested_official_test_selection": False,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
        "human_annotations_used": False,
        "expert_attributes_used": False,
        "llm_world_knowledge_used": True,
        "freeze_role_weights": contract["freeze_role_weights"],
        "freeze_reader": contract["freeze_reader"],
        "freeze_alpha": contract["freeze_alpha"],
        "semantic_enabled": contract["semantic_enabled"],
        "visual_enabled": contract["visual_enabled"],
        "interaction_enabled": contract["interaction_enabled"],
        "direction_loss_enabled": contract["direction_loss_enabled"],
    }


def _write_config(tmp_path: Path, condition: str) -> Path:
    path = tmp_path / f"{condition}.yaml"
    path.write_text(yaml.safe_dump(_base_config(tmp_path, condition), sort_keys=False), encoding="utf-8")
    return path


def _sha_for(path: Path) -> str:
    name = path.name
    if name == "CUB_source.yaml":
        return tune013.IDENTITIES["CUB"]["source_config_sha256"]
    if name == "CUB_formal.pth":
        return "f" * 64
    if name == "RUN-CUB.yaml":
        return ablation.FULL_REFERENCE_CONFIG_SHA
    if name == "metrics.json":
        return ablation.FULL_REFERENCE_METRICS_SHA
    if name == "model_best.pth":
        return ablation.FULL_REFERENCE_MODEL_SHA
    return "c" * 64


def _head(class_count: int = 6) -> CompiledPCLRHead:
    generator = torch.Generator().manual_seed(22)
    base = torch.randn(class_count, EMBED_DIM, generator=generator)
    roles = torch.randn(class_count, ROLE_COUNT, EMBED_DIM, generator=generator)
    relations, edges, _graph = tune013.one_text_edges_and_relations(roles, top_k=3)
    reader_in, reader_out = initialized_reader_states()
    return CompiledPCLRHead(
        base_prototypes=base,
        role_prototypes=roles,
        relation_embeddings=relations,
        edge_index=edges,
        seen_classes=torch.arange(class_count - 2),
        scale=10.0,
        reader_in_state=reader_in,
        reader_out_state=reader_out,
        ridge_lambda=0.3,
        relation_temperature=0.2,
        direction_temperature=0.07,
        seen_logit_gamma=0.05,
        alpha_max=2.0,
        initial_alpha=tune013.INITIAL_ALPHA,
        role_weight_max=1.0,
        initial_role_weights=torch.tensor(tune013.INITIAL_ROLE_WEIGHTS),
    )


def _tiny_train_inputs():
    images = torch.randn(4, EMBED_DIM, generator=torch.Generator().manual_seed(8))
    targets = torch.tensor([0, 1, 2, 3])
    seen = torch.arange(4)
    global_to_seen = torch.full((6,), -1, dtype=torch.long)
    global_to_seen[seen] = torch.arange(4)
    return images, targets, seen, global_to_seen


def test_config_identity_accepts_all_four_registered_conditions(monkeypatch, tmp_path):
    monkeypatch.setattr(ablation, "sha256_file", _sha_for)
    monkeypatch.setattr(tune013, "sha256_file", _sha_for)
    for condition, contract in ablation.CONDITIONS.items():
        config, config_sha = ablation.load_ablation_config(_write_config(tmp_path, condition))
        assert config["condition"] == condition
        assert config["run_id"] == contract["run_id"]
        assert config["full_reference_metrics_percent"] == {"H": ablation.FULL_REFERENCE_H}
        assert config_sha == "c" * 64
    bad = _write_config(tmp_path, "V+I-off")
    data = yaml.safe_load(bad.read_text(encoding="utf-8"))
    data["visual_enabled"] = True
    bad.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="配置身份|visual_enabled"):
        ablation.load_ablation_config(bad)


def test_v_i_off_has_no_reader_or_alpha_grad_and_output_ignores_reader(tmp_path):
    head = _head()
    config = _base_config(tmp_path, "V+I-off")
    ablation.apply_ablation_trainability(head, config)
    assert head.raw_role_weights.requires_grad is True
    assert head.raw_alpha.requires_grad is False
    assert all(not parameter.requires_grad for parameter in head.reader_in.parameters())
    assert all(not parameter.requires_grad for parameter in head.reader_out.parameters())
    images, targets, seen, global_to_seen = _tiny_train_inputs()
    before = ablation.condition_logits(head, images, config).detach()
    with torch.no_grad():
        head.reader_in.weight.add_(10.0)
        head.reader_out.bias.add_(10.0)
    after = ablation.condition_logits(head, images, config).detach()
    assert torch.allclose(before, after)
    losses = ablation.training_losses(
        head,
        images,
        targets,
        config,
        seen_device=seen,
        global_to_seen=global_to_seen,
    )
    losses["total"].backward()
    assert head.raw_role_weights.grad is not None
    assert head.raw_alpha.grad is None
    assert head.reader_in.weight.grad is None
    assert head.reader_out.weight.grad is None
    receipt = ablation.gradient_receipt(head)
    assert receipt["raw_alpha"] is None
    assert receipt["reader_in.weight"] is None
    assert receipt["raw_role_weights"] is not None
    export = ablation.condition_export(head, config)
    assert torch.count_nonzero(export.q[:, EMBED_DIM:]).item() == 0
    assert torch.count_nonzero(export.reader_in_weight).item() == 0
    assert torch.count_nonzero(export.reader_out_weight).item() == 0


def test_i_off_disables_relation_classification_but_keeps_reader_direction_ce(tmp_path):
    head = _head()
    config = _base_config(tmp_path, "I-off")
    ablation.apply_ablation_trainability(head, config)
    images, targets, seen, global_to_seen = _tiny_train_inputs()
    losses = ablation.training_losses(
        head,
        images,
        targets,
        config,
        seen_device=seen,
        global_to_seen=global_to_seen,
    )
    losses["total"].backward()
    assert head.raw_alpha.grad is None
    assert head.raw_role_weights.grad is not None
    assert head.reader_in.weight.grad is not None
    assert head.reader_out.weight.grad is not None
    assert torch.count_nonzero(head.reader_out.weight.grad).item() > 0
    receipt = ablation.gradient_receipt(head)
    assert receipt["raw_alpha"] is None
    assert receipt["reader_out.weight"] is not None
    export = ablation.condition_export(head, config)
    assert torch.count_nonzero(export.q[:, EMBED_DIM:]).item() == 0
    assert torch.allclose(export.reader_in_weight, head.reader_in.weight.detach().cpu())
    assert torch.allclose(export.reader_out_weight, head.reader_out.weight.detach().cpu())


def test_v_off_freezes_reader_and_drops_direction_ce_while_alpha_trains(tmp_path):
    head = _head()
    config = _base_config(tmp_path, "V-off")
    ablation.apply_ablation_trainability(head, config)
    images, targets, seen, global_to_seen = _tiny_train_inputs()
    losses = ablation.training_losses(
        head,
        images,
        targets,
        config,
        seen_device=seen,
        global_to_seen=global_to_seen,
    )
    assert float(losses["relation"].detach()) == pytest.approx(0.0)
    losses["total"].backward()
    assert head.raw_alpha.grad is not None
    assert head.raw_role_weights.grad is not None
    assert head.reader_in.weight.grad is None
    assert head.reader_out.weight.grad is None
    export = ablation.condition_export(head, config)
    assert torch.count_nonzero(export.q[:, EMBED_DIM:]).item() > 0
    assert torch.count_nonzero(export.reader_in_weight).item() == 0


def test_s_off_only_freezes_role_weights(tmp_path):
    head = _head()
    config = _base_config(tmp_path, "S-off")
    ablation.apply_ablation_trainability(head, config)
    images, targets, seen, global_to_seen = _tiny_train_inputs()
    assert torch.allclose(
        ablation.condition_logits(head, images, config),
        head(images, semantic_enabled=False),
    )
    losses = ablation.training_losses(
        head,
        images,
        targets,
        config,
        seen_device=seen,
        global_to_seen=global_to_seen,
    )
    losses["total"].backward()
    assert head.raw_role_weights.grad is None
    assert head.raw_alpha.grad is not None
    assert head.reader_in.weight.grad is not None
    assert head.reader_out.weight.grad is not None
