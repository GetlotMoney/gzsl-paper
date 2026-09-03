from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.frameworks.v6.compiled_pclr import EMBED_DIM, ROLE_COUNT, CompiledPCLRHead, initialized_reader_states
from model.frameworks.v7 import train_one_text_seen_ce as tune013


def _write_config(tmp_path: Path, dataset: str = "CUB") -> Path:
    identity = tune013.IDENTITIES[dataset]
    source = tmp_path / f"{dataset}_source.yaml"
    formal = tmp_path / f"{dataset}_formal.pth"
    source.write_text("dataset: test\n", encoding="utf-8")
    formal.write_bytes(b"formal")
    config = {
        "schema_version": tune013.SCHEMA,
        "experiment_id": identity["experiment_id"],
        "dataset": dataset,
        "base_commit": tune013.BASE_COMMIT,
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
    }
    path = tmp_path / f"{dataset}_tune013.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _sha_for(path: Path) -> str:
    name = path.name
    for dataset, identity in tune013.IDENTITIES.items():
        if name == f"{dataset}_source.yaml":
            return identity["source_config_sha256"]
    if name.endswith("_formal.pth"):
        return "f" * 64
    return "c" * 64


def test_config_identity_for_cub(monkeypatch, tmp_path):
    monkeypatch.setattr(tune013, "sha256_file", _sha_for)
    config, config_sha = tune013.load_one_text_seen_ce_config(_write_config(tmp_path, "CUB"))
    assert config["experiment_id"] == tune013.IDENTITIES["CUB"]["experiment_id"]
    assert config["classification_ce_scope"] == "seen_only_train_classes"
    assert config["formal_checkpoint_usage"] == "baseline_identity_only_not_training_initialization"
    assert config["fresh_source_initialization"] is True
    assert config["nested_official_test_selection"] is False
    assert config["llm_world_knowledge_used"] is True
    assert config_sha == "c" * 64
    bad = _write_config(tmp_path, "CUB")
    data = yaml.safe_load(bad.read_text(encoding="utf-8"))
    data["classification_ce_scope"] = "all_classes"
    bad.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="配置身份"):
        tune013.load_one_text_seen_ce_config(bad)


def test_one_text_relations_have_exact_unit_difference_and_topk_union():
    generator = torch.Generator().manual_seed(12)
    roles = torch.randn(6, ROLE_COUNT, EMBED_DIM, generator=generator)
    relations, edges, graph = tune013.one_text_edges_and_relations(roles, top_k=3)
    assert relations.shape == (graph["edge_count"], 2, EMBED_DIM)
    assert edges.shape == (graph["edge_count"], 2)
    assert torch.unique(edges, dim=0).size(0) == edges.size(0)
    assert bool(edges[:, 0].lt(edges[:, 1]).all())
    diff = relations[:, 0] - relations[:, 1]
    assert torch.allclose(diff.norm(dim=1), torch.ones(edges.size(0)), atol=1e-6)
    assert torch.allclose(relations[:, 0], 0.5 * diff)
    assert torch.allclose(relations[:, 1], -0.5 * diff)


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


def test_controls_replace_compiled_g_not_relation_embeddings_and_restore_full():
    head = _head()
    controls = tune013.relation_controls(head, seed=7)
    images = torch.randn(4, EMBED_DIM, generator=torch.Generator().manual_seed(7))
    before_g = head.compiled_g.detach().clone()
    before_rel = head.relation_embeddings.detach().clone()
    full_before = head(images)
    outputs = tune013._condition_logits(head, images, controls)
    full_after = head(images)
    assert torch.allclose(head.compiled_g, before_g)
    assert torch.allclose(head.relation_embeddings, before_rel)
    assert torch.allclose(outputs["full"], full_before)
    assert torch.allclose(full_after, full_before)
    assert not torch.allclose(outputs["signflip"], outputs["full"])
    assert not torch.allclose(outputs["role_shuffle"], outputs["full"])


def test_build_head_uses_initialized_reader_and_formal_constants():
    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(3)
            self.parent = SimpleNamespace(
                tg_vpr=SimpleNamespace(sentence_embeds=torch.randn(6, ROLE_COUNT, EMBED_DIM, generator=generator))
            )
            self.seen_classes = torch.arange(4)
            self._prototypes = torch.randn(6, EMBED_DIM, generator=generator)

        def prototypes(self):
            return self._prototypes

        def scale(self):
            return torch.tensor(10.0)

    config = {
        "top_k": 3,
        "ridge_lambda": 0.3,
        "relation_temperature": 0.2,
        "direction_temperature": 0.07,
        "seen_logit_gamma": 0.05,
        "alpha_max": 2.0,
        "initial_alpha": tune013.INITIAL_ALPHA,
        "role_weight_max": 1.0,
        "initial_role_weights": tune013.INITIAL_ROLE_WEIGHTS,
    }
    head, graph = tune013.build_head(Source(), config, torch.device("cpu"))
    reader_in, reader_out = initialized_reader_states()
    assert torch.allclose(head.reader_in.weight, reader_in[0])
    assert torch.allclose(head.reader_out.weight, reader_out[0])
    assert float(head.alpha().detach()) == pytest.approx(tune013.INITIAL_ALPHA)
    assert torch.allclose(head.role_weights().detach(), torch.tensor(tune013.INITIAL_ROLE_WEIGHTS), atol=1e-6)
    assert graph["edge_count"] == head.edge_count


def test_contract_requires_formal_floor_and_negative_controls():
    formal = {"U": 10.0, "S": 10.0, "H": 10.0, "ZS": 10.0}
    config = {"required_i_off_delta_h": 0.0, "required_v_off_delta_h": 0.0}
    metrics = {
        "full": {"H": 10.0},
        "s_off": {"H": 8.0},
        "v_off": {"H": 9.9},
        "i_off": {"H": 9.8},
        "signflip": {"H": 9.0},
        "role_shuffle": {"H": 9.5},
    }
    passed, deltas = tune013.contract(metrics, formal, config)
    assert passed is True
    assert deltas["i_off"] == pytest.approx(0.2)
    metrics["role_shuffle"]["H"] = 10.0
    assert tune013.contract(metrics, formal, config)[0] is False
    metrics["role_shuffle"]["H"] = 9.0
    metrics["full"]["H"] = 9.999
    assert tune013.contract(metrics, formal, config)[0] is False


def test_seen_only_classification_ce_excludes_unseen_logit_gradients():
    logits = torch.randn(3, 6, generator=torch.Generator().manual_seed(5), requires_grad=True)
    seen = torch.tensor([0, 2, 4, 5])
    global_to_seen = torch.full((6,), -1, dtype=torch.long)
    global_to_seen[seen] = torch.arange(seen.numel())
    targets = torch.tensor([0, 4, 2])
    loss = tune013.seen_only_classification_loss(
        logits,
        targets,
        seen_device=seen,
        global_to_seen=global_to_seen,
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, [1, 3]]).item() == 0
    assert torch.count_nonzero(logits.grad[:, seen]).item() > 0
    with pytest.raises(ValueError, match="seen-only"):
        tune013.seen_only_classification_loss(
            logits.detach(),
            torch.tensor([1, 4, 2]),
            seen_device=seen,
            global_to_seen=global_to_seen,
        )


def test_training_losses_seen_only_keep_relation_reader_output_gradient():
    head = _head(class_count=6)
    images = torch.randn(4, EMBED_DIM, generator=torch.Generator().manual_seed(8))
    targets = torch.tensor([0, 1, 2, 3])
    seen = torch.arange(4)
    global_to_seen = torch.full((6,), -1, dtype=torch.long)
    global_to_seen[seen] = torch.arange(4)
    losses = tune013.training_losses_seen_only(
        head,
        images,
        targets,
        seen_device=seen,
        global_to_seen=global_to_seen,
        relation_loss_weight=1.0,
    )
    losses["total"].backward()
    assert losses["classification"].detach().isfinite()
    assert losses["relation"].detach().isfinite()
    assert head.reader_in.weight.grad is not None
    assert head.reader_out.weight.grad is not None
    assert torch.count_nonzero(head.reader_out.weight.grad).item() > 0


def test_load_training_source_does_not_load_formal_checkpoint(monkeypatch, tmp_path):
    source_path = tmp_path / "CUB_source.yaml"
    formal_path = tmp_path / "CUB_formal.pth"
    source_path.write_text("dataset: CUB\n", encoding="utf-8")
    formal_path.write_bytes(b"formal")
    config = {
        "dataset": "CUB",
        "source_config": str(source_path.resolve()),
        "source_config_sha256": tune013.IDENTITIES["CUB"]["source_config_sha256"],
        "formal_checkpoint": str(formal_path.resolve()),
        "formal_checkpoint_sha256": "f" * 64,
    }
    monkeypatch.setattr(tune013, "sha256_file", _sha_for)
    monkeypatch.setattr(tune013, "load_config", lambda path: ({"dataset": "CUB"}, tune013.IDENTITIES["CUB"]["source_config_sha256"]))
    monkeypatch.setattr(tune013, "load_assets", lambda cfg: {"marker": torch.tensor(1)})

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = torch.nn.Linear(2, 2)
            self.gate = torch.nn.Linear(2, 1)

    monkeypatch.setattr(tune013, "build_model", lambda cfg, tensors, device: Source())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("formal checkpoint must not be loaded")))
    source, tensors, source_config = tune013.load_training_source(config, torch.device("cpu"))
    assert source_config["dataset"] == "CUB"
    assert tensors["marker"].item() == 1
    assert all(parameter.requires_grad for parameter in source.parent.parameters())
    assert all(parameter.requires_grad for parameter in source.gate.parameters())
