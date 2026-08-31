from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import yaml

from tools.build_pclr_relation_asset import (
    ASSET_SCHEMA,
    CONFIG_SCHEMA,
    ENCODER_IDENTITY_SCHEMA,
    EXPECTED_EDGE_COUNT,
    EXPECTED_RANGES,
    load_relation_texts,
    run,
)
from tools.runtime import sha256_file


class BuildPCLRRelationAssetTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        checkpoint = root / "clip.pt"
        checkpoint.write_bytes(b"official checkpoint fixture")
        checkpoint_sha = sha256_file(checkpoint)
        class_names_path = root / "class_names.json"
        display_names = [f"Class {index}" for index in range(200)]
        class_names_path.write_text(
            json.dumps({"xlsa": display_names, "display": display_names}),
            encoding="utf-8",
        )
        class_names_sha = sha256_file(class_names_path)

        pairs = []
        for a_id in range(200):
            for b_id in range(a_id + 1, 200):
                pairs.append((a_id, b_id))
                if len(pairs) == EXPECTED_EDGE_COUNT:
                    break
            if len(pairs) == EXPECTED_EDGE_COUNT:
                break
        edges = []
        for edge_id, (a_id, b_id) in enumerate(pairs):
            a_name, b_name = f"Class {a_id}", f"Class {b_id}"
            edges.append(
                {
                    "edge_id": edge_id,
                    "a_id": a_id,
                    "a_name": a_name,
                    "b_id": b_id,
                    "b_name": b_name,
                    "a_over_b_prompt": f"Start exactly with: {a_name} rather than {b_name}:",
                    "b_over_a_prompt": f"Start exactly with: {b_name} rather than {a_name}:",
                }
            )
        request = {
            "schema_version": "gzsl-paper.pclr-relation-request.v1",
            "class_count": 200,
            "seen_count": 150,
            "edge_count": EXPECTED_EDGE_COUNT,
            "class_names_sha256": class_names_sha,
            "clip_checkpoint_sha256": checkpoint_sha,
            "clip_python_source_sha256": "2" * 64,
            "graph_source": "OpenAI_CLIP_class_name_template_union_top3",
            "template": "a photo of a {class}",
            "seen_induced_min_degree": 1,
            "edges": edges,
        }
        request_path = root / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")

        shard_specs = []
        for index, (lower, upper) in enumerate(EXPECTED_RANGES):
            rows = []
            for edge in edges[lower : upper + 1]:
                a_name, b_name = edge["a_name"], edge["b_name"]
                rows.append(
                    {
                        "edge_id": edge["edge_id"],
                        "a_id": edge["a_id"],
                        "b_id": edge["b_id"],
                        "a_over_b": (
                            f"{a_name} rather than {b_name}: shows visible plumage pattern "
                            f"number {edge['edge_id']}."
                        ),
                        "b_over_a": (
                            f"{b_name} rather than {a_name}: shows visible bill shape "
                            f"number {edge['edge_id']}."
                        ),
                    }
                )
            shard = {
                "schema_version": "gzsl-paper.pclr-relations-shard.v1",
                "generator": {
                    "provider": "Codex sub-agent",
                    "task": f"/root/pclr_relations_{index}",
                    "generated_at": "2026-08-31",
                },
                "range": [lower, upper],
                "rows": rows,
            }
            shard_path = root / f"shard{index}.json"
            shard_path.write_text(json.dumps(shard), encoding="utf-8")
            shard_specs.append(
                {"path": str(shard_path.resolve()), "sha256": sha256_file(shard_path)}
            )

        parent = {
            "schema_version": "gzsl-paper.clip-assets.v1",
            "asset_id": "parent",
            "dataset": "CUB",
            "clip_model": "OpenAI ViT-L/14@336px",
            "counts": {"train": 7057, "test_seen": 1764, "test_unseen": 2967},
            "class_order_sha256": class_names_sha,
            "clip_checkpoint_sha256": checkpoint_sha,
            "clip_python_source_sha256": "3" * 64,
            "v3_dynamic_extensions": {"human_annotations_used": False},
            "outputs_sha256": {"class_names.json": class_names_sha},
        }
        parent_path = root / "parent_manifest.json"
        parent_path.write_text(json.dumps(parent), encoding="utf-8")
        config = {
            "schema_version": CONFIG_SCHEMA,
            "dataset": "CUB",
            "request": str(request_path.resolve()),
            "request_sha256": sha256_file(request_path),
            "shards": shard_specs,
            "parent_manifest": str(parent_path.resolve()),
            "parent_manifest_sha256": sha256_file(parent_path),
            "clip_checkpoint": str(checkpoint.resolve()),
            "clip_checkpoint_sha256": checkpoint_sha,
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return config_path

    @staticmethod
    def _fake_encoder(texts, checkpoint, device_name, batch_size):
        values = torch.arange(len(texts) * 768).reshape(len(texts), 768).float() + 1
        return F.normalize(values, dim=-1)

    @staticmethod
    def _encoder_identity():
        return {
            "schema_version": ENCODER_IDENTITY_SCHEMA,
            "implementation": "deterministic_test_encoder_v1",
        }

    def test_builds_content_addressed_asset_with_fixed_shapes_and_disclosures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            checkpoint_sha = yaml.safe_load(config.read_text(encoding="utf-8"))[
                "clip_checkpoint_sha256"
            ]
            with mock.patch(
                "tools.build_pclr_relation_asset.OFFICIAL_CHECKPOINT_SHA256",
                checkpoint_sha,
            ):
                result = run(
                    config,
                    root / "assets",
                    device_name="cpu",
                    _text_encoder=self._fake_encoder,
                    _encoder_identity=self._encoder_identity(),
                )
            output = Path(result["asset_directory"])
            self.assertEqual(result["schema_version"], ASSET_SCHEMA)
            self.assertFalse(result["human_annotations_used"])
            self.assertTrue(result["llm_world_knowledge_used"])
            self.assertFalse(result["relation_encoder_matches_parent"])
            self.assertEqual(
                tuple(torch.load(output / "edge_index.pt", weights_only=True).shape),
                (EXPECTED_EDGE_COUNT, 2),
            )
            self.assertEqual(
                tuple(
                    torch.load(
                        output / "relation_sentence_embeds.pt", weights_only=True
                    ).shape
                ),
                (EXPECTED_EDGE_COUNT, 2, 768),
            )
            self.assertEqual(
                sha256_file(output / "asset_manifest.json"),
                result["asset_manifest_sha256"],
            )

    def test_rejects_non_visual_content_before_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._fixture(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            shard_path = Path(config["shards"][0]["path"])
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            shard["rows"][0]["a_over_b"] = (
                "Class 0 rather than Class 1: lives in a coastal habitat."
            )
            shard_path.write_text(json.dumps(shard), encoding="utf-8")
            config["shards"][0]["sha256"] = sha256_file(shard_path)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "非可见形态"):
                load_relation_texts(loaded)


if __name__ == "__main__":
    unittest.main()
