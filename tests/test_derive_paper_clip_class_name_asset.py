from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import yaml

from model.train_paper_v2 import load_assets
from tools.derive_paper_clip_class_name_asset import (
    CLASS_NAME_VERSION,
    REUSED_OUTPUTS,
    canonical_class_prompts,
    run,
)
from tools.derive_paper_clip_text_asset import ENCODER_IDENTITY_SCHEMA
from tools.gzsl_data import class_order_sha256
from tools.runtime import sha256_file


class CanonicalClassNameAssetTest(unittest.TestCase):
    @staticmethod
    def _identity() -> dict:
        return {
            "schema_version": ENCODER_IDENTITY_SCHEMA,
            "implementation": "deterministic_test_encoder_v1",
            "batch_size": 16,
        }

    @staticmethod
    def _encoder(texts, checkpoint, device_name, batch_size):
        del checkpoint, device_name, batch_size
        values = torch.arange(len(texts) * 768).reshape(len(texts), 768).float() + 1
        return F.normalize(values, dim=-1)

    def _fixture(self, root: Path) -> tuple[Path, Path, str]:
        parent = root / "parent"
        parent.mkdir()
        class_names = ("class_one", "grizzly+bear", "airport_airport")
        role_names = [f"role-{index}" for index in range(6)] + [
            "overall_appearance",
            "unique_discriminative_features",
        ]
        tensors = {
            "train_features.pt": F.normalize(torch.randn(4, 768), dim=-1),
            "train_labels.pt": torch.tensor([0, 0, 1, 1]),
            "test_seen_features.pt": F.normalize(torch.randn(2, 768), dim=-1),
            "test_seen_labels.pt": torch.tensor([0, 1]),
            "test_unseen_features.pt": F.normalize(torch.randn(2, 768), dim=-1),
            "test_unseen_labels.pt": torch.tensor([2, 2]),
            "class_name_embeds.pt": F.normalize(torch.randn(3, 768), dim=-1),
            "role_sentence_embeds.pt": F.normalize(torch.randn(3, 8, 768), dim=-1),
        }
        for filename, value in tensors.items():
            torch.save(value, parent / filename)
        (parent / "class_names.json").write_text(
            json.dumps(
                {
                    "xlsa": list(class_names),
                    "display": list(class_names),
                    "prompts": [f"a photo of a {value}." for value in class_names],
                }
            ),
            encoding="utf-8",
        )
        outputs = {
            filename: sha256_file(parent / filename)
            for filename in (*tensors, "class_names.json")
        }
        checkpoint = root / "clip.pt"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_sha = sha256_file(checkpoint)
        role_path = root / "roles-v2.json"
        displays = ["class one", "grizzly bear", "airport"]
        articles = ["a", "a", "the"]
        descriptions = [
            [
                f"a photo of {article} {name}, showing visible detail number {index}."
                for index in range(8)
            ]
            for name, article in zip(displays, articles, strict=True)
        ]
        role_payload = {
            "schema_version": "gzsl-paper.role-texts.v1",
            "dataset": "AWA2",
            "class_order_sha256": class_order_sha256(class_names),
            "role_names": role_names,
            "generator": {
                "generation_method": "clip_anchored_class_specific_eight_role_descriptions_v2",
                "display_names": displays,
            },
            "descriptions": descriptions,
        }
        role_path.write_text(json.dumps(role_payload), encoding="utf-8")
        manifest = {
            "schema_version": "gzsl-paper.clip-assets.v1",
            "dataset": "AWA2",
            "asset_id": "text-v2-parent",
            "model": "ViT-L/14@336px",
            "clip_checkpoint_sha256": checkpoint_sha,
            "text_asset_version": "text-v2",
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
            "source_uris": {"role_texts": str(role_path.resolve())},
            "inputs_sha256": {
                "role_texts": sha256_file(role_path),
                "clip_checkpoint": checkpoint_sha,
            },
            "outputs_sha256": outputs,
        }
        manifest_path = parent / "asset_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        config = root / "derive.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "gzsl-paper.clip-class-name-asset-derivation.v1",
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

    def test_builds_exact_shared_role_prefix_prompts(self):
        names = ("class_one", "airport_airport")
        displays = ["class one", "airport"]
        descriptions = [
            [f"a photo of a class one, showing detail {index}." for index in range(8)],
            [f"a photo of the airport, showing detail {index}." for index in range(8)],
        ]
        generator = {
            "generation_method": "clip_anchored_class_specific_eight_role_descriptions_v2",
            "display_names": displays,
        }
        actual_displays, prompts = canonical_class_prompts(names, descriptions, generator)
        self.assertEqual(actual_displays, displays)
        self.assertEqual(prompts, ["a photo of a class one.", "a photo of the airport."])

    def test_rejects_inconsistent_role_prefixes(self):
        descriptions = [
            [f"a photo of a bird, showing detail {index}." for index in range(7)]
            + ["a photo of the bird, showing unique detail."]
        ]
        generator = {
            "generation_method": "clip_anchored_class_specific_eight_role_descriptions_v2",
            "display_names": ["bird"],
        }
        with self.assertRaisesRegex(ValueError, "共享同一照片前缀"):
            canonical_class_prompts(("bird",), descriptions, generator)

    def test_derives_new_class_names_and_hardlinks_every_other_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config, _, checkpoint_sha = self._fixture(root)
            output_root = root / "assets"
            with mock.patch(
                "tools.derive_paper_clip_class_name_asset.OFFICIAL_CHECKPOINT_SHA256",
                checkpoint_sha,
            ):
                result = run(
                    config,
                    output_root,
                    device_name="cpu",
                    batch_size=16,
                    _text_encoder=self._encoder,
                    _encoder_identity=self._identity(),
                )
            output = Path(result["asset_directory"])
            self.assertEqual(result["class_name_text_version"], CLASS_NAME_VERSION)
            names = json.loads((output / "class_names.json").read_text(encoding="utf-8"))
            self.assertEqual(
                names["prompts"],
                [
                    "a photo of a class one.",
                    "a photo of a grizzly bear.",
                    "a photo of the airport.",
                ],
            )
            for filename in REUSED_OUTPUTS:
                self.assertTrue(os.path.samefile(root / "parent" / filename, output / filename))
            self.assertNotEqual(
                sha256_file(root / "parent" / "class_name_embeds.pt"),
                sha256_file(output / "class_name_embeds.pt"),
            )
            manifest_path = output / "asset_manifest.json"
            tensors, manifest, _ = load_assets(
                {
                    "schema_version": "gzsl-paper.paper-v2-run.v1",
                    "dataset": "AWA2",
                    "asset_manifest": str(manifest_path),
                    "asset_manifest_sha256": sha256_file(manifest_path),
                }
            )
            self.assertEqual(manifest["asset_id"], result["asset_id"])
            self.assertEqual(tuple(tensors["class_name_embeds"].shape), (3, 768))


if __name__ == "__main__":
    unittest.main()
