from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model.frameworks.v6.compiled_pclr import CompiledPCLRHead, initialized_reader_states
from model.frameworks.v7.train_multidataset import build_head
from tools.build_v7_relation_asset import load_relation_rows
from tools.runtime import sha256_file


def _dynamic_head() -> CompiledPCLRHead:
    generator = torch.Generator().manual_seed(17)
    reader_in, reader_out = initialized_reader_states()
    return CompiledPCLRHead(
        base_prototypes=torch.randn(5, 768, generator=generator),
        role_prototypes=torch.randn(5, 8, 768, generator=generator),
        relation_embeddings=F.normalize(torch.randn(3, 2, 768, generator=generator), dim=-1),
        edge_index=torch.tensor([[0, 1], [0, 3], [3, 4]], dtype=torch.long),
        seen_classes=torch.tensor([0, 1, 2]),
        scale=20.0,
        reader_in_state=reader_in,
        reader_out_state=reader_out,
        seen_logit_gamma=0.1,
    )


def test_build_head_does_not_require_reader_on_gtd_source() -> None:
    class _TG:
        sentence_embeds = torch.randn(5, 8, 768, generator=torch.Generator().manual_seed(21))

    class _Parent:
        tg_vpr = _TG()

    class _Source:
        parent = _Parent()
        seen_classes = torch.tensor([0, 1, 2])
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def prototypes(self):
            return torch.randn(5, 768, generator=torch.Generator().manual_seed(22))

        def scale(self):
            return torch.tensor(20.0)

    config = {
        "ridge_lambda": 0.3, "relation_temperature": 0.2,
        "direction_temperature": 0.07, "seen_logit_gamma": 0.1,
        "alpha_max": 2.0, "initial_alpha": 0.7258594751358033,
        "role_weight_max": 1.0,
        "initial_role_weights": [0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.36, 0.0],
    }
    relations = F.normalize(torch.randn(3, 2, 768, generator=torch.Generator().manual_seed(23)), dim=-1)
    edges = torch.tensor([[0, 1], [0, 3], [3, 4]])
    head = build_head(_Source(), relations, edges, config, torch.device("cpu"))
    assert tuple(head.reader_in.weight.shape) == (64, 768)


def test_dynamic_class_and_edge_shapes_export() -> None:
    head = _dynamic_head()
    images = torch.randn(4, 768, generator=torch.Generator().manual_seed(18))
    assert head.class_count == 5
    assert head.edge_count == 3
    assert tuple(head(images).shape) == (4, 5)
    assert tuple(head.export().q.shape) == (5, 1536)
    assert tuple(head.export().bias.shape) == (5,)


def test_direction_loss_skips_seen_class_without_seen_seen_edge() -> None:
    head = _dynamic_head()
    images = torch.randn(2, 768, generator=torch.Generator().manual_seed(19))
    # Class 2 has no seen-seen incident edge, but class 0 does. The batch remains valid.
    loss = head.relation_direction_loss(images, torch.tensor([0, 2]))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_direction_loss_allows_batch_with_only_uncovered_seen_class() -> None:
    head = _dynamic_head()
    images = torch.randn(2, 768, generator=torch.Generator().manual_seed(20))
    loss = head.relation_direction_loss(images, torch.tensor([2, 2]))
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert head.reader_in.weight.grad is not None


def test_v7_relation_shard_validation(tmp_path: Path) -> None:
    request = {
        "schema_version": "gzsl-paper.pclr-graph-request.v2",
        "dataset": "AWA2",
        "class_count": 3,
        "seen_count": 2,
        "edge_count": 2,
        "direction_count": 4,
        "graph_source": "OpenAI_CLIP_class_name_template_union_top3",
        "top_k": 3,
        "clip_checkpoint_sha256": "a" * 64,
        "clip_python_source_sha256": "b" * 64,
        "edges": [
            {"edge_id": 0, "a_id": 0, "b_id": 1, "a_name": "antelope", "b_name": "deer"},
            {"edge_id": 1, "a_id": 1, "b_id": 2, "a_name": "deer", "b_name": "cow"},
        ],
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    shard = {
        "schema_version": "gzsl-paper.pclr-relations-shard.v2",
        "dataset": "AWA2",
        "generator": {"provider": "test"},
        "range": [0, 1],
        "rows": [
            {"edge_id": 0, "a_id": 0, "b_id": 1, "a_over_b": "antelope rather than deer: shows a slimmer body and longer swept horns.", "b_over_a": "deer rather than antelope: shows branching antlers and a stockier neck."},
            {"edge_id": 1, "a_id": 1, "b_id": 2, "a_over_b": "deer rather than cow: shows a lighter frame and narrow legs.", "b_over_a": "cow rather than deer: shows a broad torso and heavy muzzle."},
        ],
    }
    shard_path = tmp_path / "shard.json"
    shard_path.write_text(json.dumps(shard), encoding="utf-8")
    config = {
        "dataset": "AWA2",
        "request": str(request_path),
        "request_sha256": sha256_file(request_path),
        "clip_checkpoint_sha256": "a" * 64,
        "clip_python_source_sha256": "b" * 64,
        "shards": [{"path": str(shard_path), "sha256": sha256_file(shard_path)}],
    }
    metadata, rows = load_relation_rows(config)
    assert metadata["edge_count"] == 2
    assert len(rows) == 2


def test_v7_relation_shard_rejects_same_direction_body(tmp_path: Path) -> None:
    request = {
        "schema_version": "gzsl-paper.pclr-graph-request.v2",
        "dataset": "SUN",
        "class_count": 2,
        "seen_count": 1,
        "edge_count": 1,
        "direction_count": 2,
        "graph_source": "OpenAI_CLIP_class_name_template_union_top3",
        "top_k": 3,
        "clip_checkpoint_sha256": "a" * 64,
        "clip_python_source_sha256": "b" * 64,
        "edges": [{"edge_id": 0, "a_id": 0, "b_id": 1, "a_name": "abbey", "b_name": "church"}],
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    shard = {
        "schema_version": "gzsl-paper.pclr-relations-shard.v2",
        "dataset": "SUN",
        "generator": {"provider": "test"},
        "range": [0, 0],
        "rows": [{
            "edge_id": 0, "a_id": 0, "b_id": 1,
            "a_over_b": "abbey rather than church: shows a tall stone nave with arched cloisters.",
            "b_over_a": "church rather than abbey: shows a tall stone nave with arched cloisters.",
        }],
    }
    shard_path = tmp_path / "shard.json"
    shard_path.write_text(json.dumps(shard), encoding="utf-8")
    config = {
        "dataset": "SUN", "request": str(request_path),
        "request_sha256": sha256_file(request_path),
        "clip_checkpoint_sha256": "a" * 64, "clip_python_source_sha256": "b" * 64,
        "shards": [{"path": str(shard_path), "sha256": sha256_file(shard_path)}],
    }
    try:
        load_relation_rows(config)
    except ValueError as error:
        assert "双方向正文相同" in str(error)
    else:
        raise AssertionError("同正文双方向关系必须被拒绝")
