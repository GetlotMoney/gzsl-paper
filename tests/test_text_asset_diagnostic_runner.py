import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from tools import run_text_asset_diagnostic as runner
from tools.runtime import sha256_file


COMMIT = "a" * 40
CLASS_ORDER_SHA = "1" * 64


class TextAssetDiagnosticRunnerTest(unittest.TestCase):
    @staticmethod
    def _write_asset(root: Path) -> tuple[Path, Path]:
        asset_root = root / "asset"
        asset_root.mkdir()
        train_features = torch.zeros(4, 768)
        train_features[0:2, 0] = 1.0
        train_features[2:4, 1] = 1.0
        train_labels = torch.tensor([0, 0, 1, 1])
        class_names = torch.zeros(3, 768)
        class_names[0, 0] = 1.0
        class_names[1, 1] = 1.0
        class_names[2, 2] = 1.0
        roles = class_names[:, None, :].expand(-1, 8, -1).clone()
        tensors = {
            "train_features.pt": train_features,
            "train_labels.pt": train_labels,
            "class_name_embeds.pt": class_names,
            "role_sentence_embeds.pt": roles,
        }
        for filename, value in tensors.items():
            torch.save(value, asset_root / filename)
        manifest = {
            "schema_version": "gzsl-paper.clip-assets.v1",
            "dataset": "CUB",
            "asset_id": "synthetic-text-v1",
            "class_count": 3,
            "seen_class_count": 2,
            "train_count": 4,
            "seen_classes": [0, 1],
            "class_order_sha256": CLASS_ORDER_SHA,
            "outputs_sha256": {
                filename: sha256_file(asset_root / filename) for filename in tensors
            },
        }
        manifest_path = asset_root / "asset_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        variant = roles.clone()
        variant[:, 1::2] = torch.roll(variant[:, 1::2], shifts=3, dims=-1)
        variant_path = root / "text-v0-role-embeddings.pt"
        torch.save(variant, variant_path)
        return manifest_path, variant_path

    @staticmethod
    def _config(manifest_path: Path, variant_path: Path, run_id: str) -> dict:
        return {
            "schema_version": runner.CONFIG_SCHEMA,
            "experiment_id": runner.EXPERIMENT_ID,
            "run_id": run_id,
            "dataset": "CUB",
            "asset_manifest": str(manifest_path.resolve()),
            "asset_manifest_sha256": sha256_file(manifest_path),
            "base_role_name": "text-v1",
            "role_variants": [
                {
                    "name": "text-v0",
                    "path": str(variant_path.resolve()),
                    "sha256": sha256_file(variant_path),
                    "class_order_sha256": CLASS_ORDER_SHA,
                    "class_order_evidence": "same frozen Xian class-order export",
                }
            ],
            "official_test_loaded": False,
            "seen_images_only": True,
            "unseen_images_used": False,
            "diagnostic_no_model": True,
        }

    @staticmethod
    def _write_config(root: Path, config: dict) -> Path:
        path = root / "diagnostic-run.yaml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def test_end_to_end_materializes_four_diagnostic_artifacts_and_no_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_path, variant_path = self._write_asset(root)
            run_id = "RUN-CUB-TEXT-DIAGNOSTIC-001"
            config = self._config(manifest_path, variant_path, run_id)
            config_path = self._write_config(root, config)
            output_dir = root / run_id

            with (
                patch.object(runner, "require_clean_code_tree") as require_clean,
                patch.object(runner, "current_code_commit", return_value=COMMIT),
            ):
                returned = runner.run(
                    config_path,
                    output_dir,
                    expected_commit=COMMIT,
                )

            require_clean.assert_called_once_with()
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                runner.EXPECTED_ARTIFACTS,
            )
            self.assertEqual(
                (output_dir / "config.snapshot.yaml").read_bytes(),
                config_path.read_bytes(),
            )
            self.assertFalse(any(output_dir.glob("*.pth")))
            self.assertFalse(
                any(
                    token in path.name.lower()
                    for path in output_dir.iterdir()
                    for token in ("model", "checkpoint")
                )
            )

            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics, returned)
            self.assertEqual(metrics["schema_version"], "gzsl-paper.text-asset-diagnostics.v1")
            self.assertEqual(metrics["run_config_schema_version"], runner.CONFIG_SCHEMA)
            self.assertEqual(metrics["experiment_id"], runner.EXPERIMENT_ID)
            self.assertEqual(metrics["run_id"], run_id)
            self.assertEqual(metrics["dataset"], "CUB")
            self.assertEqual(metrics["code_commit"], COMMIT)
            self.assertEqual(metrics["config_sha256"], sha256_file(config_path))
            self.assertTrue(metrics["diagnostic_no_model"])
            self.assertIs(metrics["official_test_loaded"], False)
            self.assertIs(metrics["seen_images_only"], True)
            self.assertIs(metrics["unseen_images_used"], False)
            self.assertEqual(
                set(metrics["text_versions"]),
                {"class-name", "text-v1", "text-v0"},
            )
            for version in metrics["text_versions"].values():
                self.assertIn("alignment", version)
                self.assertIn("corresponding_class_cosine", version["alignment"])
                self.assertIn("hardest_negative_margin", version["alignment"])
                self.assertIn("visual_to_text_top1", version["alignment"])
            self.assertEqual(
                metrics["role_variant_sources"]["text-v0"]["class_order_evidence"],
                config["role_variants"][0]["class_order_evidence"],
            )

            fingerprints = json.loads(
                (output_dir / "data_fingerprints.json").read_text(encoding="utf-8")
            )
            self.assertEqual(fingerprints["code_commit"], COMMIT)
            self.assertEqual(fingerprints["config"]["sha256"], sha256_file(config_path))
            self.assertEqual(
                fingerprints["asset_manifest_sha256"], sha256_file(manifest_path)
            )
            self.assertEqual(fingerprints["class_order_sha256"], CLASS_ORDER_SHA)
            self.assertEqual(
                set(fingerprints["asset_inputs"]),
                {
                    "asset_manifest.json",
                    "train_features.pt",
                    "train_labels.pt",
                    "class_name_embeds.pt",
                    "role_sentence_embeds.pt",
                },
            )
            self.assertEqual(
                fingerprints["role_variants"][0]["sha256"],
                sha256_file(variant_path),
            )
            self.assertTrue(fingerprints["diagnostic_no_model"])
            log = (output_dir / "training.log").read_text(encoding="utf-8")
            self.assertIn("run_type=seen-only text asset diagnostic", log)
            self.assertIn("official_test_loaded=false", log)
            self.assertIn("diagnostic_no_model=true", log)
            self.assertNotIn("epoch=", log.lower())
            self.assertNotIn("optimizer=", log.lower())

    def test_rejects_every_official_or_unseen_test_boundary_violation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_path, variant_path = self._write_asset(root)
            valid = self._config(manifest_path, variant_path, "RUN-BOUNDARY")
            violations = {
                "official_test_loaded": True,
                "seen_images_only": False,
                "unseen_images_used": True,
            }
            for field, value in violations.items():
                with self.subTest(field=field):
                    invalid = {**valid, field: value}
                    with self.assertRaisesRegex(ValueError, field):
                        runner.validate_config(invalid)

    def test_variant_requires_matching_class_order_sha_and_nonempty_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_path, variant_path = self._write_asset(root)
            config = self._config(manifest_path, variant_path, "RUN-CLASS-ORDER")

            missing_evidence = json.loads(json.dumps(config))
            missing_evidence["role_variants"][0]["class_order_evidence"] = "  "
            with self.assertRaisesRegex(ValueError, "class_order_evidence"):
                runner.validate_config(missing_evidence)

            wrong_order = json.loads(json.dumps(config))
            wrong_order["role_variants"][0]["class_order_sha256"] = "2" * 64
            with self.assertRaisesRegex(ValueError, "类别顺序SHA"):
                runner._load_manifest_identity(wrong_order)

    def test_commit_clean_tree_and_output_identity_are_hard_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_path, variant_path = self._write_asset(root)
            run_id = "RUN-IDENTITY"
            config_path = self._write_config(
                root, self._config(manifest_path, variant_path, run_id)
            )
            output_dir = root / run_id

            with (
                patch.object(runner, "require_clean_code_tree") as require_clean,
                patch.object(runner, "current_code_commit", return_value="b" * 40),
            ):
                with self.assertRaisesRegex(RuntimeError, "expected-commit"):
                    runner.run(config_path, output_dir, expected_commit=COMMIT)
            require_clean.assert_called_once_with()
            self.assertFalse(output_dir.exists())

            with (
                patch.object(
                    runner,
                    "require_clean_code_tree",
                    side_effect=RuntimeError("dirty tree"),
                ),
                patch.object(runner, "current_code_commit") as current_commit,
            ):
                with self.assertRaisesRegex(RuntimeError, "dirty tree"):
                    runner.run(config_path, output_dir, expected_commit=COMMIT)
            current_commit.assert_not_called()
            self.assertFalse(output_dir.exists())

            with self.assertRaisesRegex(ValueError, "run_id"):
                runner.validate_output_path(root / "WRONG-NAME", run_id)
            existing = root / run_id
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                runner.validate_output_path(existing, run_id)

    def test_config_requires_absolute_manifest_and_variant_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest_path, variant_path = self._write_asset(root)
            valid = self._config(manifest_path, variant_path, "RUN-ABSOLUTE")

            relative_manifest = {**valid, "asset_manifest": "asset/asset_manifest.json"}
            with self.assertRaisesRegex(ValueError, "asset_manifest必须是绝对路径"):
                runner.validate_config(relative_manifest)

            relative_variant = json.loads(json.dumps(valid))
            relative_variant["role_variants"][0]["path"] = "roles.pt"
            with self.assertRaisesRegex(ValueError, "path必须是绝对路径"):
                runner.validate_config(relative_variant)


if __name__ == "__main__":
    unittest.main()
