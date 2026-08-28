from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.trainers.paper_v2 import load_assets
from tools.derive_paper_clip_text_asset import (
    CLIP_SOURCE_FILES,
    ENCODER_IDENTITY_SCHEMA,
    REUSED_OUTPUTS,
    _production_encoder_identity,
    derived_asset_id,
    natural_class_name,
    run,
    validate_clip_friendly_v2,
)
from tools.gzsl_data import class_order_sha256
from tools.runtime import sha256_file


class DerivePaperClipTextAssetTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, str]:
        parent = root / "parent"
        parent.mkdir()
        class_names = ("class_one", "grizzly+bear", "class_three")
        tensors = {
            "train_features.pt": F.normalize(torch.randn(4, 768), dim=-1),
            "train_labels.pt": torch.tensor([0, 0, 1, 1]),
            "test_seen_features.pt": F.normalize(torch.randn(2, 768), dim=-1),
            "test_seen_labels.pt": torch.tensor([0, 1]),
            "test_unseen_features.pt": F.normalize(torch.randn(2, 768), dim=-1),
            "test_unseen_labels.pt": torch.tensor([2, 2]),
            "class_name_embeds.pt": F.normalize(torch.randn(3, 768), dim=-1),
        }
        for filename, value in tensors.items():
            torch.save(value, parent / filename)
        (parent / "class_names.json").write_text(
            json.dumps({"xlsa": list(class_names), "display": list(class_names), "prompts": []}),
            encoding="utf-8",
        )
        outputs = {filename: sha256_file(parent / filename) for filename in REUSED_OUTPUTS}
        checkpoint = root / "clip.pt"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_sha = sha256_file(checkpoint)
        role_names = [
            "role-0",
            "role-1",
            "role-2",
            "role-3",
            "role-4",
            "role-5",
            "overall_appearance",
            "unique_discriminative_features",
        ]
        manifest = {
            "schema_version": "gzsl-paper.clip-assets.v1",
            "dataset": "AWA2",
            "asset_id": "parent-asset",
            "source_config_sha256": "a" * 64,
            "model": "ViT-L/14@336px",
            "clip_checkpoint_sha256": checkpoint_sha,
            "class_count": 3,
            "seen_class_count": 2,
            "unseen_class_count": 1,
            "train_count": 4,
            "test_seen_count": 2,
            "test_unseen_count": 2,
            "seen_classes": [0, 1],
            "unseen_classes": [2],
            "class_order_sha256": class_order_sha256(class_names),
            "role_names": role_names,
            "role_text_generator": {"generation_method": "v1"},
            "source_uris": {},
            "inputs_sha256": {"clip_checkpoint": checkpoint_sha},
            "outputs_sha256": outputs,
        }
        manifest_path = parent / "asset_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        role_path = root / "roles-v2.json"
        descriptions = []
        for name in ("class one", "grizzly bear", "class three"):
            descriptions.append(
                [
                    f"a photo of a {name}, showing visible detail number {index}."
                    for index in range(8)
                ]
            )
        role_path.write_text(
            json.dumps(
                {
                    "schema_version": "gzsl-paper.role-texts.v1",
                    "dataset": "AWA2",
                    "class_order_sha256": class_order_sha256(class_names),
                    "role_names": role_names,
                    "generator": {
                        "generation_method": "clip_anchored_class_specific_eight_role_descriptions_v2"
                    },
                    "descriptions": descriptions,
                }
            ),
            encoding="utf-8",
        )
        config = root / "derive.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "gzsl-paper.clip-text-asset-derivation.v1",
                    "dataset": "AWA2",
                    "parent_manifest": str(manifest_path.resolve()),
                    "parent_manifest_sha256": sha256_file(manifest_path),
                    "role_texts": str(role_path.resolve()),
                    "role_texts_sha256": sha256_file(role_path),
                    "clip_checkpoint": str(checkpoint.resolve()),
                    "clip_checkpoint_sha256": checkpoint_sha,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return config, checkpoint, checkpoint_sha

    @staticmethod
    def _encoder_identity(batch_size: int = 256) -> dict:
        return {
            "schema_version": ENCODER_IDENTITY_SCHEMA,
            "implementation": "deterministic_test_encoder_v1",
            "batch_size": batch_size,
        }

    @staticmethod
    def _fake_encoder(texts, checkpoint, device_name, batch_size):
        return F.normalize(
            torch.arange(len(texts) * 768).reshape(len(texts), 768).float() + 1,
            dim=-1,
        )

    def _run_fixture(
        self,
        config: Path,
        output_root: Path,
        checkpoint_sha: str,
        *,
        encoder=None,
    ) -> dict:
        with mock.patch(
            "tools.derive_paper_clip_text_asset.OFFICIAL_CHECKPOINT_SHA256",
            checkpoint_sha,
        ):
            return run(
                config,
                output_root,
                device_name="cpu",
                _text_encoder=encoder or self._fake_encoder,
                _encoder_identity=self._encoder_identity(),
            )

    def test_derives_content_addressed_asset_hardlinks_cache_and_loads_for_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, checkpoint_sha = self._fixture(root)
            output_root = root / "assets"
            result = self._run_fixture(config, output_root, checkpoint_sha)
            output = Path(result["asset_directory"])

            self.assertEqual(output.parent, output_root.resolve())
            self.assertEqual(output.name, result["asset_id"])
            self.assertEqual(result["text_asset_version"], "text-v2")
            self.assertEqual(result["derived_from_asset_id"], "parent-asset")
            self.assertTrue(result["reused_visual_and_label_cache"])
            role_path = output / "role_sentence_embeds.pt"
            self.assertEqual(
                tuple(torch.load(role_path, weights_only=True).shape), (3, 8, 768)
            )
            raw_config = yaml.safe_load(config.read_text(encoding="utf-8"))
            self.assertEqual(
                result["asset_id"],
                derived_asset_id(
                    raw_config,
                    sha256_file(role_path),
                    self._encoder_identity(),
                ),
            )
            for filename in REUSED_OUTPUTS:
                source = root / "parent" / filename
                destination = output / filename
                self.assertTrue(os.path.samefile(source, destination))
                self.assertEqual(source.stat().st_ino, destination.stat().st_ino)
                self.assertEqual(
                    sha256_file(destination),
                    sha256_file(source),
                )

            manifest_path = output / "asset_manifest.json"
            tensors, loaded_manifest, loaded_path = load_assets(
                {
                    "schema_version": "gzsl-paper.paper-v2-run.v1",
                    "dataset": "AWA2",
                    "asset_manifest": str(manifest_path),
                    "asset_manifest_sha256": sha256_file(manifest_path),
                }
            )
            self.assertEqual(loaded_path, manifest_path)
            self.assertEqual(loaded_manifest["asset_id"], result["asset_id"])
            self.assertEqual(
                tuple(tensors["role_sentence_embeds"].shape), (3, 8, 768)
            )
            self.assertEqual(
                loaded_manifest["text_encoder_identity_sha256"],
                result["text_encoder_identity_sha256"],
            )

    def test_asset_id_changes_with_role_embedding_or_encoder_identity(self):
        config = {
            "parent_manifest_sha256": "1" * 64,
            "role_texts_sha256": "2" * 64,
            "clip_checkpoint_sha256": "3" * 64,
        }
        first = derived_asset_id(config, "4" * 64, self._encoder_identity())
        changed_embedding = derived_asset_id(
            config, "5" * 64, self._encoder_identity()
        )
        changed_encoder = derived_asset_id(
            config, "4" * 64, self._encoder_identity(batch_size=32)
        )
        self.assertNotEqual(first, changed_embedding)
        self.assertNotEqual(first, changed_encoder)

    def test_rejects_swapped_overall_and_unique_role_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, checkpoint_sha = self._fixture(root)
            values = yaml.safe_load(config.read_text(encoding="utf-8"))
            role_path = Path(values["role_texts"])
            payload = json.loads(role_path.read_text(encoding="utf-8"))
            payload["role_names"][6], payload["role_names"][7] = (
                payload["role_names"][7],
                payload["role_names"][6],
            )
            role_path.write_text(json.dumps(payload), encoding="utf-8")
            values["role_texts_sha256"] = sha256_file(role_path)
            config.write_text(
                yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "role_names"):
                self._run_fixture(config, root / "assets", checkpoint_sha)
            self.assertFalse((root / "assets").exists())

    def test_rejects_hardlink_sha_drift_and_cleans_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, checkpoint_sha = self._fixture(root)
            output_root = root / "assets"
            real_link = os.link
            mutated = False

            def drifting_link(source, destination):
                nonlocal mutated
                real_link(source, destination)
                if not mutated:
                    mutated = True
                    with Path(source).open("ab") as stream:
                        stream.write(b"drift")

            with mock.patch(
                "tools.derive_paper_clip_text_asset.os.link",
                side_effect=drifting_link,
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA发生变化"):
                    self._run_fixture(config, output_root, checkpoint_sha)
            self.assertTrue(output_root.is_dir())
            self.assertEqual(list(output_root.iterdir()), [])

    def test_encoder_failure_cleans_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, checkpoint_sha = self._fixture(root)
            output_root = root / "assets"

            def failing_encoder(texts, checkpoint, device_name, batch_size):
                raise RuntimeError("synthetic encoder failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic encoder failure"):
                self._run_fixture(
                    config,
                    output_root,
                    checkpoint_sha,
                    encoder=failing_encoder,
                )
            self.assertTrue(output_root.is_dir())
            self.assertEqual(list(output_root.iterdir()), [])

    def test_rejects_role_text_sha_mismatch_before_creating_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, checkpoint_sha = self._fixture(root)
            values = yaml.safe_load(config.read_text(encoding="utf-8"))
            Path(values["role_texts"]).write_text("changed", encoding="utf-8")
            output_root = root / "assets"
            with self.assertRaisesRegex(ValueError, "原文SHA"):
                self._run_fixture(config, output_root, checkpoint_sha)
            self.assertFalse(output_root.exists())

    def test_production_encoder_identity_hashes_five_files_and_checks_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "clip"
            package.mkdir()
            for index, filename in enumerate(CLIP_SOURCE_FILES):
                (package / filename).write_bytes(
                    f"source-{index}".encode("utf-8")
                )
            direct_url = {
                "url": "https://github.com/openai/CLIP.git",
                "vcs_info": {"commit_id": "d05afc4", "vcs": "git"},
            }
            fake_clip = types.ModuleType("clip")
            fake_clip.__file__ = str(package / "__init__.py")

            class FakeDistribution:
                version = "1.0"

                @staticmethod
                def read_text(name):
                    return (
                        json.dumps(direct_url)
                        if name == "direct_url.json"
                        else None
                    )

            parent = {
                "clip_python_source_sha256": sha256_file(package / "clip.py"),
                "clip_distribution_version": "1.0",
                "clip_distribution_direct_url": direct_url,
                "clip_checkpoint_sha256": "f" * 64,
            }
            with mock.patch.dict(sys.modules, {"clip": fake_clip}), mock.patch(
                "tools.derive_paper_clip_text_asset.importlib.metadata.distribution",
                return_value=FakeDistribution(),
            ):
                identity = _production_encoder_identity(parent, 32)
                self.assertEqual(
                    identity["schema_version"], ENCODER_IDENTITY_SCHEMA
                )
                self.assertEqual(identity["batch_size"], 32)
                self.assertEqual(
                    set(identity["clip_source_files_sha256"]),
                    set(CLIP_SOURCE_FILES),
                )
                for filename in CLIP_SOURCE_FILES:
                    self.assertEqual(
                        identity["clip_source_files_sha256"][filename],
                        sha256_file(package / filename),
                    )
                with self.assertRaisesRegex(ValueError, "direct_url"):
                    _production_encoder_identity(
                        {
                            **parent,
                            "clip_distribution_direct_url": {"url": "wrong"},
                        },
                        32,
                    )

    def test_natural_class_name_removes_separators_and_adjacent_duplicates(self):
        self.assertEqual(natural_class_name("grizzly+bear"), "grizzly bear")
        self.assertEqual(natural_class_name("airport_airport"), "airport")

    def test_frozen_display_name_can_correct_xlsa_spelling(self):
        generator = {
            "generation_method": "clip_anchored_class_specific_eight_role_descriptions_v2",
            "display_names": ["Arctic Tern"],
        }
        rows = [
            [
                f"a photo of an Arctic Tern, showing concrete visible detail {index}."
                for index in range(8)
            ]
        ]
        validate_clip_friendly_v2(("Artic_Tern",), rows, generator)

    def test_rejects_plus_sign_and_duplicate_generic_text(self):
        names = ("grizzly+bear",)
        generator = {
            "generation_method": "clip_anchored_class_specific_eight_role_descriptions_v2"
        }
        bad = [["a photo of a grizzly+bear, showing body."] * 8]
        with self.assertRaises(ValueError):
            validate_clip_friendly_v2(names, bad, generator)


if __name__ == "__main__":
    unittest.main()
