from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.frameworks.v6.compiled_pclr import EMBED_DIM, ROLE_COUNT
from model.frameworks.v7 import train_text_relation_operator as tune015
from model.frameworks.v7.relation_operator import (
    TextRelationOperatorDeployment,
    TextRelationOperatorHead,
    text_relation_graph,
    visual_edge_targets,
)


def _write_config(tmp_path: Path, dataset: str = "CUB") -> Path:
    identity = tune015.IDENTITIES[dataset]
    source = tmp_path / f"{dataset}_source.yaml"
    formal = tmp_path / f"{dataset}_formal.pth"
    source.write_text("dataset: test\n", encoding="utf-8")
    formal.write_bytes(b"formal")
    config = {
        "schema_version": tune015.SCHEMA,
        "experiment_id": identity["experiment_id"],
        "dataset": dataset,
        "base_commit": tune015.BASE_COMMIT,
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
        "relation_loss_weight": 1.0,
        "ridge_lambda": 0.3,
        "relation_temperature": 0.2,
        "top_k": 3,
        "seen_logit_gamma": identity["seen_logit_gamma"],
        "alpha_max": 2.0,
        "initial_alpha": tune015.INITIAL_ALPHA,
        "role_weight_max": 1.0,
        "initial_role_weights": tune015.INITIAL_ROLE_WEIGHTS,
        "operator_rank": 32,
        "operator_init_std": 0.01,
        "operator_residual_mode": "identity_residual_unit_normalized",
        "relation_embedding_mode": "one_text_direction_to_visual_centroid_difference",
        "classification_ce_scope": "seen_only_train_classes",
        "direction_alignment_source": "trainval_seen_visual_centroids_only",
        "best_selection_metric": "official_full_H_post_update",
        "official_test_evaluations": math.ceil(identity["total_updates"] / identity["eval_interval_steps"]),
        "required_i_off_delta_h": 0.0,
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
    }
    path = tmp_path / f"{dataset}_tune015.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _sha_for(path: Path) -> str:
    name = path.name
    for dataset, identity in tune015.IDENTITIES.items():
        if name == f"{dataset}_source.yaml":
            return identity["source_config_sha256"]
    if name.endswith("_formal.pth"):
        return "f" * 64
    return "c" * 64


def _head(class_count: int = 8, seen_count: int = 5) -> TextRelationOperatorHead:
    generator = torch.Generator().manual_seed(42)
    base = torch.randn(class_count, EMBED_DIM, generator=generator)
    roles = torch.randn(class_count, ROLE_COUNT, EMBED_DIM, generator=generator)
    text_directions, edges, _graph = text_relation_graph(roles, top_k=3)
    seen = torch.arange(seen_count)
    centroids = torch.randn(seen_count, EMBED_DIM, generator=generator)
    seen_mask, targets = visual_edge_targets(centroids, seen, edges, class_count)
    return TextRelationOperatorHead(
        base_prototypes=base,
        role_prototypes=roles,
        text_directions=text_directions,
        edge_index=edges,
        seen_classes=seen,
        visual_seen_edge_mask=seen_mask,
        visual_seen_targets=targets,
        scale=10.0,
        ridge_lambda=0.3,
        relation_temperature=0.2,
        seen_logit_gamma=0.05,
        alpha_max=2.0,
        initial_alpha=tune015.INITIAL_ALPHA,
        role_weight_max=1.0,
        initial_role_weights=torch.tensor(tune015.INITIAL_ROLE_WEIGHTS),
        operator_rank=8,
        operator_init_std=0.01,
    )


def test_config_identity_for_cub(monkeypatch, tmp_path):
    monkeypatch.setattr(tune015, "sha256_file", _sha_for)
    config, config_sha = tune015.load_text_relation_operator_config(_write_config(tmp_path, "CUB"))
    assert config["experiment_id"] == tune015.IDENTITIES["CUB"]["experiment_id"]
    assert config["base_commit"] == tune015.BASE_COMMIT
    assert config["classification_ce_scope"] == "seen_only_train_classes"
    assert config["direction_alignment_source"] == "trainval_seen_visual_centroids_only"
    assert config["operator_rank"] == 32
    assert config["operator_residual_mode"] == "identity_residual_unit_normalized"
    assert config["fresh_source_initialization"] is True
    assert config["unseen_images_used_for_gradient"] is False
    assert config_sha == "c" * 64
    bad = _write_config(tmp_path, "CUB")
    data = yaml.safe_load(bad.read_text(encoding="utf-8"))
    data["direction_alignment_source"] = "test_centroids"
    bad.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="配置身份"):
        tune015.load_text_relation_operator_config(bad)


def test_visual_targets_use_train_seen_centroids_only():
    generator = torch.Generator().manual_seed(3)
    roles = torch.randn(6, ROLE_COUNT, EMBED_DIM, generator=generator)
    text_directions, edges, graph = text_relation_graph(roles, top_k=3)
    seen = torch.arange(4)
    centroids = torch.randn(4, EMBED_DIM, generator=generator)
    seen_mask, targets = visual_edge_targets(centroids, seen, edges, class_count=6)
    assert text_directions.shape == (graph["edge_count"], EMBED_DIM)
    assert tuple(seen_mask.shape) == (edges.size(0),)
    assert targets.shape == (int(seen_mask.sum()), EMBED_DIM)
    assert torch.allclose(targets.norm(dim=1), torch.ones(targets.size(0)), atol=1e-6)
    with pytest.raises(ValueError, match="centroids"):
        visual_edge_targets(torch.randn(6, EMBED_DIM), seen, edges, class_count=6)


def test_operator_gradients_and_no_reader_contract():
    head = _head()
    images = torch.randn(6, EMBED_DIM, generator=torch.Generator().manual_seed(9))
    targets = torch.tensor([0, 1, 2, 3, 4, 0])
    seen = torch.arange(5)
    global_to_seen = torch.full((8,), -1, dtype=torch.long)
    global_to_seen[seen] = torch.arange(5)
    losses = head.training_losses(
        images,
        targets,
        seen_device=seen,
        global_to_seen=global_to_seen,
        relation_loss_weight=1.0,
    )
    losses["total"].backward()
    assert losses["classification"].detach().isfinite()
    assert losses["relation"].detach().isfinite()
    assert head.operator_down.grad is not None
    assert head.operator_up.grad is not None
    assert torch.count_nonzero(head.operator_up.grad).item() > 0
    assert not any("reader" in name for name, _parameter in head.named_parameters())
    assert all("reader" not in item["name"] for item in head.parameter_contract())


def test_initial_operator_is_identity_unit_direction_and_gradients_are_bounded():
    head = _head()
    initial = head.operator_edge_directions()
    assert torch.allclose(initial, head.text_directions, atol=1e-7, rtol=0.0)
    assert torch.allclose(initial.norm(dim=1), torch.ones(head.edge_count), atol=1e-7, rtol=0.0)
    loss = head.visual_direction_alignment_loss()
    loss.backward()
    assert torch.isfinite(head.operator_up.grad).all()
    assert torch.isfinite(head.operator_down.grad).all()
    assert float(head.operator_up.grad.detach().abs().max()) < 1.0
    assert float(head.operator_down.grad.detach().abs().max()) == pytest.approx(0.0)


def test_unseen_edges_are_generated_by_shared_operator():
    head = _head(class_count=10, seen_count=6)
    edge_has_unseen = ~head.visual_seen_edge_mask
    assert bool(edge_has_unseen.any())
    all_edges = head.operator_edge_directions()
    assert all_edges.shape == (head.edge_count, EMBED_DIM)
    assert all_edges[edge_has_unseen].shape[0] > 0
    assert all_edges[edge_has_unseen].requires_grad


def test_export_logits_equivalent_and_export_has_no_reader_or_graph():
    head = _head()
    with torch.no_grad():
        head.operator_up.normal_(mean=0.0, std=0.01)
    images = torch.randn(4, EMBED_DIM, generator=torch.Generator().manual_seed(11))
    expected = head(images)
    export = head.export().__dict__
    assert set(export) == {"q", "bias"}
    deployment = TextRelationOperatorDeployment.from_export(export)
    actual = deployment(images)
    assert actual.shape == (4, head.class_count)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-6)


def test_evaluate_uses_fresh_current_head_best_metric_and_200_class_axis():
    class Source(torch.nn.Module):
        def __init__(self, head: TextRelationOperatorHead):
            super().__init__()
            self._head = head

        def prototypes(self):
            return F.normalize(self._head.base_q / 10.0, dim=-1)

        def scale(self):
            return torch.tensor(10.0)

    head = _head(class_count=200, seen_count=150)
    images_seen = torch.randn(150, EMBED_DIM, generator=torch.Generator().manual_seed(13))
    images_unseen = torch.randn(50, EMBED_DIM, generator=torch.Generator().manual_seed(14))
    tensors = {
        "test_seen_features": images_seen,
        "test_seen_labels": torch.arange(150),
        "test_unseen_features": images_unseen,
        "test_unseen_labels": 150 + torch.arange(50),
    }
    metrics = tune015.evaluate(head, Source(head), tensors, torch.device("cpu"))
    assert set(metrics["metrics"]) == {"full", "s_off", "i_off"}
    assert set(metrics["metrics"]["full"]) == {"U", "S", "H", "ZS"}
    assert set(metrics["parent_metrics"]) == {"U", "S", "H", "ZS"}
    assert head.export().q.shape == (200, EMBED_DIM)
    passed, deltas = tune015.contract(
        {
            "full": {"H": 10.0},
            "s_off": {"H": 9.0},
            "i_off": {"H": 9.5},
        },
        {"H": 9.0},
        {"required_i_off_delta_h": 0.0},
    )
    assert passed is True
    assert deltas["i_off"] == pytest.approx(0.5)


def test_build_head_computes_centroids_from_train_features(monkeypatch):
    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(16)
            self.parent = SimpleNamespace(
                tg_vpr=SimpleNamespace(sentence_embeds=torch.randn(200, ROLE_COUNT, EMBED_DIM, generator=generator))
            )
            self.seen_classes = torch.arange(150)
            self._prototypes = torch.randn(200, EMBED_DIM, generator=generator)

        def prototypes(self):
            return self._prototypes

        def scale(self):
            return torch.tensor(10.0)

    calls = {}

    def fake_centroids(features, labels, seen):
        calls["features_ptr"] = features.data_ptr()
        calls["labels_ptr"] = labels.data_ptr()
        calls["seen"] = seen.clone()
        return torch.randn(150, EMBED_DIM, generator=torch.Generator().manual_seed(17))

    monkeypatch.setattr(tune015.h1, "visual_centroids", fake_centroids)
    tensors = {
        "train_features": torch.randn(150, EMBED_DIM, generator=torch.Generator().manual_seed(18)),
        "train_labels": torch.arange(150),
    }
    config = {
        "dataset": "CUB",
        "top_k": 3,
        "ridge_lambda": 0.3,
        "relation_temperature": 0.2,
        "seen_logit_gamma": 0.91,
        "alpha_max": 2.0,
        "initial_alpha": tune015.INITIAL_ALPHA,
        "role_weight_max": 1.0,
        "initial_role_weights": tune015.INITIAL_ROLE_WEIGHTS,
        "operator_rank": 32,
        "operator_init_std": 0.01,
        "operator_residual_mode": "identity_residual_unit_normalized",
    }
    head, graph = tune015.build_head(Source(), tensors, config, torch.device("cpu"))
    assert calls["features_ptr"] == tensors["train_features"].data_ptr()
    assert calls["labels_ptr"] == tensors["train_labels"].data_ptr()
    assert torch.equal(calls["seen"], torch.arange(150))
    assert graph["reader_removed"] is True
    assert int(head.visual_seen_edge_mask.sum()) > 0
