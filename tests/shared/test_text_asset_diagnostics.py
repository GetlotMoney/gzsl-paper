import json
from pathlib import Path
import tempfile
import unittest

import torch

from tools.diagnose_paper_text_assets import (
    diagnose_tensors,
    load_role_variant,
    parse_role_variant,
    role_difference_metrics,
    run,
    text_alignment_metrics,
)
from tools.runtime import sha256_file


class TextAssetDiagnosticPureFunctionTest(unittest.TestCase):
    def test_margin_and_bidirectional_top1_detect_swapped_classes(self):
        visual_centers = torch.eye(3)
        swapped_texts = torch.stack(
            (visual_centers[1], visual_centers[0], visual_centers[2])
        )
        metrics = text_alignment_metrics(
            visual_centers,
            swapped_texts,
            torch.tensor([0, 1, 2]),
        )

        cosine = metrics["corresponding_class_cosine"]
        self.assertAlmostEqual(cosine["mean"], 1.0 / 3.0, places=6)
        self.assertEqual(cosine["min"], 0.0)
        self.assertEqual(cosine["max"], 1.0)
        margin = metrics["hardest_negative_margin"]
        self.assertAlmostEqual(margin["mean"], -1.0 / 3.0, places=6)
        self.assertEqual(margin["min"], -1.0)
        self.assertEqual(margin["max"], 1.0)
        self.assertAlmostEqual(margin["positive_margin_rate"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(
            metrics["visual_to_text_top1"]["per_class_rate"], 1.0 / 3.0, places=6
        )
        self.assertAlmostEqual(
            metrics["text_to_visual_top1"]["per_class_rate"], 1.0 / 3.0, places=6
        )

    def test_role_difference_separates_collapsed_from_orthogonal_roles(self):
        collapsed = torch.zeros(1, 8, 8)
        collapsed[:, :, 0] = 1.0
        diverse = torch.eye(8).unsqueeze(0)

        collapsed_stats = role_difference_metrics(collapsed, torch.tensor([0]))
        diverse_stats = role_difference_metrics(diverse, torch.tensor([0]))

        collapsed_pairs = collapsed_stats["within_class_role_pairwise_cosine"]
        diverse_pairs = diverse_stats["within_class_role_pairwise_cosine"]
        self.assertEqual(collapsed_pairs["pairs_per_class"], 28)
        self.assertAlmostEqual(collapsed_pairs["mean"], 1.0, places=6)
        self.assertAlmostEqual(collapsed_pairs["std"], 0.0, places=6)
        self.assertAlmostEqual(diverse_pairs["mean"], 0.0, places=6)
        self.assertAlmostEqual(
            collapsed_stats["role_to_Mean8_cosine_distance"]["mean"], 0.0, places=6
        )
        self.assertAlmostEqual(
            diverse_stats["role_to_Mean8_cosine_distance"]["mean"],
            1.0 - 1.0 / (8.0**0.5),
            places=6,
        )
        self.assertAlmostEqual(
            collapsed_stats["normalized_role_variance_around_class_mean"]["mean"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            diverse_stats["normalized_role_variance_around_class_mean"]["mean"],
            7.0 / 8.0,
            places=6,
        )

    def test_tensor_entry_point_uses_training_centers(self):
        features = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        labels = torch.tensor([0, 0, 1, 1])
        class_names = torch.eye(2)
        roles = class_names[:, None, :].expand(-1, 8, -1).clone()
        result = diagnose_tensors(
            features,
            labels,
            class_names,
            {"text-v1": roles},
            torch.tensor([0, 1]),
        )
        self.assertEqual(result["seen_visual_center_count"], 2)
        self.assertEqual(
            result["text_versions"]["class-name"]["alignment"]["visual_to_text_top1"]["percent"],
            100.0,
        )
        self.assertEqual(
            result["text_versions"]["text-v1"]["alignment"]["text_to_visual_top1"]["percent"],
            100.0,
        )


class TextAssetDiagnosticBoundaryTest(unittest.TestCase):
    @staticmethod
    def _write_asset(root: Path) -> Path:
        train_features = torch.zeros(4, 768)
        train_features[0:2, 0] = 1.0
        train_features[2:4, 1] = 1.0
        train_labels = torch.tensor([0, 0, 1, 1])
        class_names = torch.zeros(3, 768)
        class_names[0, 0] = 1.0
        class_names[1, 1] = 1.0
        class_names[2, 2] = 1.0
        roles = class_names[:, None, :].expand(-1, 8, -1).clone()
        values = {
            "train_features.pt": train_features,
            "train_labels.pt": train_labels,
            "class_name_embeds.pt": class_names,
            "role_sentence_embeds.pt": roles,
        }
        for filename, value in values.items():
            torch.save(value, root / filename)
        manifest = {
            "schema_version": "gzsl-paper.clip-assets.v1",
            "dataset": "CUB",
            "asset_id": "synthetic-seen-only",
            "class_count": 3,
            "seen_class_count": 2,
            "train_count": 4,
            "seen_classes": [0, 1],
            "class_order_sha256": "1" * 64,
            "outputs_sha256": {
                filename: sha256_file(root / filename) for filename in values
            },
        }
        manifest_path = root / "asset_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_end_to_end_output_discloses_seen_only_boundary_and_input_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._write_asset(root)
            output = root / "diagnostic" / "result.json"
            payload = run(
                manifest_path,
                output,
                expected_manifest_sha256=sha256_file(manifest_path),
            )
            self.assertTrue(output.is_file())
            self.assertIs(payload["official_test_loaded"], False)
            self.assertIs(payload["seen_images_only"], True)
            self.assertIs(payload["unseen_images_used"], False)
            self.assertEqual(payload["input_sha256"]["asset_manifest.json"], sha256_file(manifest_path))
            self.assertEqual(
                set(payload["input_sha256"]),
                {
                    "asset_manifest.json",
                    "train_features.pt",
                    "train_labels.pt",
                    "class_name_embeds.pt",
                    "role_sentence_embeds.pt",
                },
            )

    def test_role_variant_checks_sha_shape_class_axis_and_finite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_path = root / "old_roles.pt"
            valid = torch.ones(3, 8, 768)
            torch.save(valid, valid_path)
            loaded, actual = load_role_variant(valid_path, sha256_file(valid_path), 3)
            self.assertTrue(torch.equal(loaded, valid))
            self.assertEqual(actual, sha256_file(valid_path))

            wrong_axis_path = root / "wrong_axis.pt"
            torch.save(torch.ones(2, 8, 768), wrong_axis_path)
            with self.assertRaisesRegex(ValueError, "形状错误"):
                load_role_variant(wrong_axis_path, sha256_file(wrong_axis_path), 3)

            nonfinite_path = root / "nonfinite.pt"
            nonfinite = valid.clone()
            nonfinite[0, 0, 0] = float("nan")
            torch.save(nonfinite, nonfinite_path)
            with self.assertRaisesRegex(ValueError, "NaN或Inf"):
                load_role_variant(nonfinite_path, sha256_file(nonfinite_path), 3)

    def test_variant_parser_supports_windows_paths(self):
        expected_sha = "a" * 64
        name, path, parsed_sha = parse_role_variant(
            f"old-text=C:\\assets\\old_roles.pt={expected_sha}"
        )
        self.assertEqual(name, "old-text")
        self.assertEqual(str(path), "C:\\assets\\old_roles.pt")
        self.assertEqual(parsed_sha, expected_sha)

    def test_diagnostic_source_has_no_official_split_cache_keys(self):
        source_path = Path(__file__).resolve().parents[2] / "tools" / "diagnose_paper_text_assets.py"
        source = source_path.read_text(encoding="utf-8").lower()
        for forbidden in ("test_seen", "test_unseen"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
