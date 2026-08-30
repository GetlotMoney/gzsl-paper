import hashlib
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

from model.innovations.evaluate_pclr_multidataset import (
    load_multidataset_config,
    load_relation_asset,
)


def test_awa2_and_sun_generic_pclr_configs_are_fixed():
    expected = {
        "AWA2": ("config/tries/v4_confirm_003_pclr_awa2.yaml", 5, 0.05),
        "SUN": ("config/tries/v4_confirm_003_pclr_sun.yaml", 60, 0.15),
    }
    for dataset, (path, candidate_top_k, gamma) in expected.items():
        config, digest = load_multidataset_config(Path(path))
        assert config["dataset"] == dataset
        assert config["candidate_top_k"] == candidate_top_k
        assert config["seen_logit_gamma"] == gamma
        assert config["nested_official_test_selection"] is True
        assert config["generic_class_name_directions"] is True
        assert config["llm_world_knowledge_used"] is False
        assert len(digest) == 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_asset(root: Path, edges: torch.Tensor, *, class_count: int = 4):
    relations = F.normalize(torch.randn(len(edges), 2, 768), dim=-1)
    torch.save(relations.float(), root / "relation_sentence_embeds.pt")
    torch.save(edges.long(), root / "edge_index.pt")
    (root / "relation_texts.json").write_text("{}\n", encoding="utf-8")
    outputs = {
        name: _sha(root / name)
        for name in (
            "relation_sentence_embeds.pt",
            "edge_index.pt",
            "relation_texts.json",
        )
    }
    manifest = {
        "schema_version": "gzsl-paper.pclr-generic-relation-asset.v1",
        "dataset": "AWA2",
        "class_count": class_count,
        "seen_count": 2,
        "edge_count": len(edges),
        "direction_count": 2 * len(edges),
        "embedding_dimension": 768,
        "graph_source": "OpenAI_CLIP_class_name_template_union_top3",
        "human_annotations_used": False,
        "llm_world_knowledge_used": False,
        "generic_class_name_directions": True,
        "outputs_sha256": outputs,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_relation_asset_dynamic_axis_rejects_bad_edges_and_manifest():
    cases = {
        "negative": torch.tensor([[-1, 1], [0, 2], [2, 3]]),
        "self_loop": torch.tensor([[0, 1], [2, 2], [2, 3]]),
        "duplicate": torch.tensor([[0, 1], [0, 1], [2, 3]]),
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        valid_root = root / "valid"
        valid_root.mkdir()
        valid_manifest = _write_asset(
            valid_root, torch.tensor([[0, 1], [1, 2], [2, 3]])
        )
        config = {
            "dataset": "AWA2",
            "relation_manifest": str(valid_manifest.resolve()),
            "relation_manifest_sha256": _sha(valid_manifest),
        }
        relations, edges, _ = load_relation_asset(
            config, class_count=4, seen_count=2
        )
        assert tuple(relations.shape) == (3, 2, 768)
        assert tuple(edges.shape) == (3, 2)

        mismatch_root = root / "mismatch"
        mismatch_root.mkdir()
        mismatch_manifest = _write_asset(
            mismatch_root,
            torch.tensor([[0, 1], [1, 2], [2, 3]]),
            class_count=5,
        )
        mismatch_config = {
            **config,
            "relation_manifest": str(mismatch_manifest.resolve()),
            "relation_manifest_sha256": _sha(mismatch_manifest),
        }
        with pytest.raises(ValueError):
            load_relation_asset(mismatch_config, class_count=4, seen_count=2)

        for name, bad_edges in cases.items():
            case_root = root / name
            case_root.mkdir()
            manifest = _write_asset(case_root, bad_edges)
            bad_config = {
                **config,
                "relation_manifest": str(manifest.resolve()),
                "relation_manifest_sha256": _sha(manifest),
            }
            with pytest.raises(ValueError):
                load_relation_asset(bad_config, class_count=4, seen_count=2)
