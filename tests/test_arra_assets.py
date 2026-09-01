from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.frameworks.v6.arra_assets import (
    DEFAULT_V5_R2_CODE_COMMIT,
    DEFAULT_V5_R2_CONFIG_SHA256,
    EMBED_DIM,
    ROLE_NAMES,
    ARRADatasetSpec,
    load_arra_eval_assets,
    load_arra_train_assets,
    load_v5_r2_initialization,
)
from tools.runtime import sha256_file


class ARRAAssetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = ARRADatasetSpec(
            dataset="CUB",
            class_count=4,
            seen_count=2,
            train_count=3,
            test_seen_count=2,
            test_unseen_count=2,
            edge_count=3,
        )
        self.visual_dir = self.root / "visual"
        self.relation_dir = self.root / "relation"
        self.visual_dir.mkdir()
        self.relation_dir.mkdir()
        self._write_visual_asset()
        self._write_relation_asset()
        self._write_checkpoint()
        self.config = {
            "asset_manifest": str(self.visual_manifest),
            "asset_manifest_sha256": sha256_file(self.visual_manifest),
            "asset_id": "synthetic_dynamic_asset",
            "relation_asset_manifest": str(self.relation_manifest),
            "relation_asset_manifest_sha256": sha256_file(self.relation_manifest),
            "relation_asset_id": "synthetic_relation_asset",
            "v5_r2_checkpoint": str(self.checkpoint),
            "v5_r2_checkpoint_sha256": sha256_file(self.checkpoint),
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _save_tensor(self, name: str, value: torch.Tensor) -> str:
        path = self.visual_dir / name
        torch.save(value, path)
        return sha256_file(path)

    def _save_patch(self, name: str, shape: tuple[int, ...]) -> str:
        path = self.visual_dir / name
        values = np.arange(np.prod(shape), dtype=np.float16).reshape(shape)
        np.save(path, values)
        return sha256_file(path)

    def _write_visual_asset(self) -> None:
        generator = torch.Generator().manual_seed(206)
        self.train_features = torch.randn(self.spec.train_count, EMBED_DIM, generator=generator)
        self.train_labels = torch.tensor([0, 1, 0], dtype=torch.long)
        self.role_sentence_embeds = F.normalize(
            torch.randn(self.spec.class_count, 8, EMBED_DIM, generator=generator),
            dim=-1,
        )
        self.test_seen_labels = torch.tensor([0, 1], dtype=torch.long)
        self.test_unseen_labels = torch.tensor([2, 3], dtype=torch.long)
        outputs = {
            "train_features.pt": self._save_tensor("train_features.pt", self.train_features),
            "train_labels.pt": self._save_tensor("train_labels.pt", self.train_labels),
            "test_seen_features.pt": self._save_tensor(
                "test_seen_features.pt",
                torch.randn(self.spec.test_seen_count, EMBED_DIM, generator=generator),
            ),
            "test_seen_labels.pt": self._save_tensor("test_seen_labels.pt", self.test_seen_labels),
            "test_unseen_features.pt": self._save_tensor(
                "test_unseen_features.pt",
                torch.randn(self.spec.test_unseen_count, EMBED_DIM, generator=generator),
            ),
            "test_unseen_labels.pt": self._save_tensor(
                "test_unseen_labels.pt",
                self.test_unseen_labels,
            ),
            "role_sentence_embeds.pt": self._save_tensor(
                "role_sentence_embeds.pt",
                self.role_sentence_embeds,
            ),
            "train_coarse_patch_features.npy": self._save_patch(
                "train_coarse_patch_features.npy",
                (self.spec.train_count, 36, EMBED_DIM),
            ),
            "test_seen_coarse_patch_features.npy": self._save_patch(
                "test_seen_coarse_patch_features.npy",
                (self.spec.test_seen_count, 36, EMBED_DIM),
            ),
            "test_unseen_coarse_patch_features.npy": self._save_patch(
                "test_unseen_coarse_patch_features.npy",
                (self.spec.test_unseen_count, 36, EMBED_DIM),
            ),
        }
        manifest = {
            "schema_version": "gzsl-paper.clip-assets.v1",
            "dataset": "CUB",
            "asset_id": "synthetic_dynamic_asset",
            "class_count": self.spec.class_count,
            "seen_class_count": self.spec.seen_count,
            "unseen_class_count": self.spec.class_count - self.spec.seen_count,
            "train_count": self.spec.train_count,
            "test_seen_count": self.spec.test_seen_count,
            "test_unseen_count": self.spec.test_unseen_count,
            "seen_classes": [0, 1],
            "unseen_classes": [2, 3],
            "role_names": list(ROLE_NAMES),
            "outputs_sha256": outputs,
        }
        self.visual_manifest = self.visual_dir / "asset_manifest.json"
        self.visual_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def _write_relation_output(self, name: str, value) -> str:
        path = self.relation_dir / name
        if isinstance(value, torch.Tensor):
            torch.save(value, path)
        else:
            path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return sha256_file(path)

    def _write_relation_asset(self) -> None:
        generator = torch.Generator().manual_seed(207)
        self.relation_embeds = F.normalize(
            torch.randn(self.spec.edge_count, 2, EMBED_DIM, generator=generator),
            dim=-1,
        )
        self.edge_index = torch.tensor([[0, 1], [0, 2], [1, 3]], dtype=torch.long)
        relation_texts = {
            "schema_version": "gzsl-paper.pclr-relation-texts.v1",
            "human_annotations_used": False,
            "llm_world_knowledge_used": True,
            "rows": [
                {
                    "edge_id": index,
                    "a_id": int(edge[0]),
                    "b_id": int(edge[1]),
                    "a_over_b": f"class{edge[0]} rather than class{edge[1]}: cue",
                    "b_over_a": f"class{edge[1]} rather than class{edge[0]}: cue",
                }
                for index, edge in enumerate(self.edge_index.tolist())
            ],
        }
        outputs = {
            "relation_texts.json": self._write_relation_output("relation_texts.json", relation_texts),
            "relation_sentence_embeds.pt": self._write_relation_output(
                "relation_sentence_embeds.pt",
                self.relation_embeds,
            ),
            "edge_index.pt": self._write_relation_output("edge_index.pt", self.edge_index),
        }
        manifest = {
            "schema_version": "gzsl-paper.pclr-relation-asset.v1",
            "asset_id": "synthetic_relation_asset",
            "dataset": "CUB",
            "class_count": self.spec.class_count,
            "seen_count": self.spec.seen_count,
            "edge_count": self.spec.edge_count,
            "direction_count": 2 * self.spec.edge_count,
            "embedding_dimension": EMBED_DIM,
            "graph_source": "OpenAI_CLIP_class_name_template_union_top3",
            "template": "a photo of a {class}",
            "seen_induced_min_degree": 1,
            "parent_manifest_sha256": sha256_file(self.visual_manifest),
            "human_annotations_used": False,
            "llm_world_knowledge_used": True,
            "relation_encoder_matches_parent": True,
            "outputs_sha256": outputs,
        }
        self.relation_manifest = self.relation_dir / "asset_manifest.json"
        self.relation_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def _write_checkpoint(self) -> None:
        generator = torch.Generator().manual_seed(208)
        p_v5 = F.normalize(torch.randn(self.spec.class_count, EMBED_DIM, generator=generator), dim=-1)
        reader = {
            "reader_in.weight": torch.randn(64, EMBED_DIM, generator=generator),
            "reader_in.bias": torch.randn(64, generator=generator),
            "reader_out.weight": torch.randn(EMBED_DIM, 64, generator=generator),
            "reader_out.bias": torch.randn(EMBED_DIM, generator=generator),
        }
        payload = {
            "code_commit": DEFAULT_V5_R2_CODE_COMMIT,
            "config_sha256": DEFAULT_V5_R2_CONFIG_SHA256,
            "arra_initialization": {
                "p_v5": p_v5,
                "scale": torch.tensor(14.25),
                "reader_state_dict": reader,
                "source_eval_anchor_replay_max_abs": 0.0,
                "source_eval_scale_replay_abs": 0.0,
            },
        }
        self.checkpoint = self.root / "model_best.pth"
        torch.save(payload, self.checkpoint)

    def test_train_loader_keeps_official_test_assets_unopened(self):
        (self.visual_dir / "test_seen_features.pt").unlink()
        assets = load_arra_train_assets(self.config, spec=self.spec)

        self.assertEqual(tuple(assets.train_features.shape), (3, EMBED_DIM))
        self.assertEqual(tuple(assets.train_coarse_patches.shape), (3, 36, EMBED_DIM))
        self.assertIsInstance(assets.train_coarse_patches, np.memmap)
        self.assertEqual(assets.seen_classes.tolist(), [0, 1])
        self.assertEqual(assets.unseen_classes.tolist(), [2, 3])
        self.assertEqual(tuple(assets.p_v5.shape), (4, EMBED_DIM))
        self.assertAlmostEqual(float(assets.scale), 14.25)
        self.assertEqual(assets.identity["source_eval_anchor_replay_max_abs"], 0.0)
        expected_d = F.normalize(self.relation_embeds[:, 0] - self.relation_embeds[:, 1], dim=-1)
        self.assertTrue(torch.allclose(assets.relation_directions, expected_d))

    def test_eval_loader_opens_test_patches_and_not_checkpoint_or_train_split(self):
        self.checkpoint.unlink()
        (self.visual_dir / "train_features.pt").unlink()
        assets = load_arra_eval_assets(self.config, spec=self.spec)

        self.assertEqual(tuple(assets.test_seen_features.shape), (2, EMBED_DIM))
        self.assertEqual(tuple(assets.test_unseen_features.shape), (2, EMBED_DIM))
        self.assertIsInstance(assets.test_seen_coarse_patches, np.memmap)
        self.assertIsInstance(assets.test_unseen_coarse_patches, np.memmap)
        self.assertEqual(assets.seen_classes.tolist(), [0, 1])
        self.assertEqual(assets.unseen_classes.tolist(), [2, 3])

    def test_graph_free_eval_loader_does_not_open_relation_assets(self):
        self.relation_manifest.unlink()
        assets = load_arra_eval_assets(
            self.config,
            spec=self.spec,
            include_relation_assets=False,
        )
        self.assertIsNone(assets.relation_sentence_embeds)
        self.assertIsNone(assets.relation_directions)
        self.assertIsNone(assets.edge_index)
        self.assertTrue(assets.identity["graph_free_eval_assets"])
        self.assertIsNone(assets.identity["relation_asset"])

    def test_v5_initialization_binds_checkpoint_identity(self):
        init = load_v5_r2_initialization(
            self.config,
            role_sentence_embeds=self.role_sentence_embeds,
            train_features=self.train_features,
            train_labels=self.train_labels,
            relation_sentence_embeds=self.relation_embeds,
            edge_index=self.edge_index,
            spec=self.spec,
        )
        self.assertEqual(init.checkpoint_code_commit, DEFAULT_V5_R2_CODE_COMMIT)
        self.assertEqual(init.checkpoint_config_sha256, DEFAULT_V5_R2_CONFIG_SHA256)
        self.assertAlmostEqual(float(init.scale), 14.25)
        self.assertEqual(init.source_eval_anchor_replay_max_abs, 0.0)

        bad_config = dict(self.config)
        bad_config["v5_r2_code_commit"] = "wrong"
        with self.assertRaisesRegex(ValueError, "code commit"):
            load_v5_r2_initialization(
                bad_config,
                role_sentence_embeds=self.role_sentence_embeds,
                train_features=self.train_features,
                train_labels=self.train_labels,
                relation_sentence_embeds=self.relation_embeds,
                edge_index=self.edge_index,
                spec=self.spec,
            )

    def test_relation_manifest_must_bind_visual_manifest_sha(self):
        relation_manifest = json.loads(self.relation_manifest.read_text(encoding="utf-8"))
        relation_manifest["parent_manifest_sha256"] = "0" * 64
        self.relation_manifest.write_text(json.dumps(relation_manifest, indent=2) + "\n", encoding="utf-8")
        self.config["relation_asset_manifest_sha256"] = sha256_file(self.relation_manifest)

        with self.assertRaisesRegex(ValueError, "relation asset identity"):
            load_arra_train_assets(self.config, spec=self.spec)


if __name__ == "__main__":
    unittest.main()
