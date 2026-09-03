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
from model.frameworks.v7 import train_class_held_out_vi as tune014
from model.frameworks.v7 import train_one_text_seen_ce as tune013


def _write_config(tmp_path: Path) -> Path:
    identity = tune014.IDENTITIES["CUB"]
    source = tmp_path / "CUB_source.yaml"
    formal = tmp_path / "CUB_formal.pth"
    source.write_text("dataset: CUB\n", encoding="utf-8")
    formal.write_bytes(b"formal")
    config = {
        "schema_version": tune014.SCHEMA,
        "experiment_id": identity["experiment_id"],
        "dataset": "CUB",
        "base_commit": tune013.BASE_COMMIT,
        "code_parent_commit": tune014.CODE_PARENT_COMMIT,
        "source_config": str(source.resolve()),
        "source_config_sha256": identity["source_config_sha256"],
        "formal_checkpoint": str(formal.resolve()),
        "formal_checkpoint_sha256": "f" * 64,
        "formal_checkpoint_usage": "baseline_identity_only_not_training_initialization",
        "formal_full_metrics_percent": {"U": 1.0, "S": 2.0, "H": 1.3333333333333333, "ZS": 3.0},
        "device": "cpu",
        "random_seed": 7,
        "batch_size": 50,
        "nominal_epochs": 200,
        "total_updates": identity["total_updates"],
        "eval_interval_steps": identity["eval_interval_steps"],
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-5,
        "weight_decay": 0.0,
        "relation_loss_weight": 0.0,
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
        "classification_ce_scope": "class_held_out_outer_for_vi_seen_only_for_s",
        "expected_direction_skip_seen_class_ids": identity["direction_skip_seen_class_ids"],
        "best_selection_metric": "official_full_H_post_update",
        "official_test_evaluations": math.ceil(identity["total_updates"] / identity["eval_interval_steps"]),
        "required_i_off_delta_h": 0.0,
        "required_v_off_delta_h": 0.0,
        "require_full_not_below_formal": True,
        "fresh_source_initialization": True,
        "test_used_for_selection": True,
        "test_used_for_hyperparameter_selection": True,
        "nested_official_test_selection": False,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
        "human_annotations_used": False,
        "expert_attributes_used": False,
        "llm_world_knowledge_used": True,
        "meta_algorithm": "first_order_class_held_out_maml",
        "meta_second_order": False,
        "meta_fold_schedule": "rank_modulo_update_mod_3",
        "meta_inner_steps": 1,
        "meta_inner_learning_rate": 0.01,
        "meta_inner_batch_size": 50,
        "meta_outer_batch_size": 50,
        "meta_outer_candidate_scope": "all_train_seen_classes",
        "meta_outer_loss_weight": 1.0,
        "s_classification_gradient_source": "ordinary_seen_batch_ce",
        "vi_classification_gradient_source": "class_disjoint_pseudo_unseen_outer_ce",
        "temporary_inner_optimizer_steps_only": True,
    }
    path = tmp_path / "RUN-CUB.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _sha_for(path: Path) -> str:
    name = path.name
    if name == "CUB_source.yaml":
        return tune014.IDENTITIES["CUB"]["source_config_sha256"]
    if name == "CUB_formal.pth":
        return "f" * 64
    return "c" * 64


def _head(class_count: int = 12, seen_count: int = 9) -> CompiledPCLRHead:
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
        seen_classes=torch.arange(seen_count),
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


def _toy_train_set(class_count: int = 9, samples_per_class: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(44)
    labels = torch.arange(class_count).repeat_interleave(samples_per_class)
    features = torch.randn(labels.numel(), EMBED_DIM, generator=generator)
    return features, labels


def test_config_identity_for_cub(monkeypatch, tmp_path):
    monkeypatch.setattr(tune014, "sha256_file", _sha_for)
    config, config_sha = tune014.load_class_held_out_vi_config(_write_config(tmp_path))
    assert config["experiment_id"] == tune014.IDENTITIES["CUB"]["experiment_id"]
    assert config["classification_ce_scope"] == "class_held_out_outer_for_vi_seen_only_for_s"
    assert config["code_parent_commit"] == tune014.CODE_PARENT_COMMIT
    assert config["meta_second_order"] is False
    assert config["meta_outer_candidate_scope"] == "all_train_seen_classes"
    assert config["temporary_inner_optimizer_steps_only"] is True
    assert config_sha == "c" * 64
    bad = _write_config(tmp_path)
    data = yaml.safe_load(bad.read_text(encoding="utf-8"))
    data["meta_second_order"] = True
    bad.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="配置身份"):
        tune014.load_class_held_out_vi_config(bad)


def test_inner_step_is_temporary_and_does_not_mutate_formal_head_parameters():
    head = _head()
    features, labels = _toy_train_set()
    folds = [(torch.tensor([0, 1, 3, 4, 6, 7]), torch.tensor([2, 5, 8]))]
    before = {name: value.detach().clone() for name, value in head.named_parameters()}
    tune014.first_order_class_held_out_vi_episode(
        head,
        features,
        labels,
        pseudo_seen=folds[0][0],
        pseudo_unseen=folds[0][1],
        outer_candidate_classes=torch.arange(9),
        generator=torch.Generator().manual_seed(1),
        inner_batch_size=8,
        outer_batch_size=8,
        inner_learning_rate=0.01,
        outer_loss_weight=1.0,
    )
    for name, value in head.named_parameters():
        assert torch.allclose(value.detach(), before[name])


def test_outer_loss_gives_reader_and_alpha_gradients_but_not_s_gradients():
    head = _head()
    features, labels = _toy_train_set()
    receipt = tune014.first_order_class_held_out_vi_episode(
        head,
        features,
        labels,
        pseudo_seen=torch.tensor([0, 1, 3, 4, 6, 7]),
        pseudo_unseen=torch.tensor([2, 5, 8]),
        outer_candidate_classes=torch.arange(9),
        generator=torch.Generator().manual_seed(2),
        inner_batch_size=8,
        outer_batch_size=8,
        inner_learning_rate=0.01,
        outer_loss_weight=1.0,
    )
    assert set(receipt["outer_gradient_norms"]) == set(tune014.META_PARAM_NAMES)
    assert head.raw_alpha.grad is not None
    assert head.raw_alpha.grad.norm().item() > 0.0
    assert head.reader_in.weight.grad is not None
    assert head.reader_out.weight.grad is not None
    assert head.raw_role_weights.grad is None


def test_pseudo_unseen_classes_are_not_in_inner_episode():
    head = _head()
    features, labels = _toy_train_set()
    receipt = tune014.first_order_class_held_out_vi_episode(
        head,
        features,
        labels,
        pseudo_seen=torch.tensor([0, 1, 3, 4, 6, 7]),
        pseudo_unseen=torch.tensor([2, 5, 8]),
        outer_candidate_classes=torch.arange(9),
        generator=torch.Generator().manual_seed(3),
        inner_batch_size=8,
        outer_batch_size=8,
        inner_learning_rate=0.01,
        outer_loss_weight=1.0,
    )
    assert set(receipt["inner_class_ids"]).isdisjoint(receipt["pseudo_unseen_class_ids"])
    assert set(receipt["inner_class_ids"]).issubset(receipt["outer_candidate_class_ids"])
    assert set(receipt["pseudo_unseen_class_ids"]).issubset(receipt["outer_candidate_class_ids"])
    with pytest.raises(ValueError, match="不相交"):
        tune014.first_order_class_held_out_vi_episode(
            head,
            features,
            labels,
            pseudo_seen=torch.tensor([0, 1, 2]),
            pseudo_unseen=torch.tensor([2, 5, 8]),
            outer_candidate_classes=torch.arange(9),
            generator=torch.Generator().manual_seed(3),
            inner_batch_size=8,
            outer_batch_size=8,
            inner_learning_rate=0.01,
            outer_loss_weight=1.0,
        )


def test_outer_candidate_axis_includes_meta_seen_columns_with_direct_gradients():
    logits = torch.randn(4, 12, generator=torch.Generator().manual_seed(6), requires_grad=True)
    pseudo_seen = torch.tensor([0, 1, 3, 4, 6, 7])
    pseudo_unseen = torch.tensor([2, 5, 8])
    outer_targets = torch.tensor([2, 5, 8, 2])
    loss = tune014.candidate_classification_loss(
        logits,
        outer_targets,
        candidate_classes=torch.arange(9),
        class_count=12,
    )
    loss.backward()
    assert torch.isin(outer_targets, pseudo_unseen).all()
    assert torch.count_nonzero(logits.grad[:, pseudo_seen]).item() > 0
    assert torch.count_nonzero(logits.grad[:, pseudo_unseen]).item() > 0
    assert torch.count_nonzero(logits.grad[:, [9, 10, 11]]).item() == 0


def test_gradient_helpers_only_accept_train_tensors_not_test_tensors():
    features, labels = _toy_train_set()
    pseudo_seen = torch.tensor([0, 1, 3, 4, 6, 7])
    pseudo_unseen = torch.tensor([2, 5, 8])
    inner_ids = tune014.sample_class_batch_indices(
        labels,
        pseudo_seen,
        batch_size=12,
        generator=torch.Generator().manual_seed(4),
        device=torch.device("cpu"),
    )
    outer_ids = tune014.sample_class_batch_indices(
        labels,
        pseudo_unseen,
        batch_size=12,
        generator=torch.Generator().manual_seed(5),
        device=torch.device("cpu"),
    )
    assert torch.isin(labels.index_select(0, inner_ids), pseudo_seen).all()
    assert torch.isin(labels.index_select(0, outer_ids), pseudo_unseen).all()
    assert features.index_select(0, inner_ids).shape == (12, EMBED_DIM)


def test_best_selection_rejects_update_zero():
    metrics = {
        "full": {"H": 10.0},
        "s_off": {"H": 8.0},
        "v_off": {"H": 9.0},
        "i_off": {"H": 9.0},
        "signflip": {"H": 8.0},
        "role_shuffle": {"H": 8.0},
    }
    config = {"required_i_off_delta_h": 0.0, "required_v_off_delta_h": 0.0}
    assert tune014.class_held_out_contract(metrics, {"H": 9.0}, config, best_update=0)[0] is False
    assert tune014.class_held_out_contract(metrics, {"H": 9.0}, config, best_update=141)[0] is True
