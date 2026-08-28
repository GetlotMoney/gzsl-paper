from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from tools.prepare_lver_view_assets import (
    CROP_NAMES,
    NORMALIZED_CROP_BOXES,
    SCHEMA_VERSION,
    FourCropDataset,
    build_manifest,
    crop_pil_views,
    encode_four_crops,
    pixel_crop_boxes,
    run,
)


def _tensor_preprocess(image: Image.Image) -> torch.Tensor:
    width, height = image.size
    values = torch.tensor(list(image.get_flattened_data()), dtype=torch.float32)
    return values.reshape(height, width, 3).permute(2, 0, 1)


class _FakeEncoder:
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        means = images.float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        basis = torch.arange(1, 769, dtype=torch.float32, device=images.device).unsqueeze(0)
        return basis + means


class LVERViewAssetTests(unittest.TestCase):
    def test_fixed_boxes_and_view_order(self):
        image = Image.new("RGB", (8, 8))
        for y in range(8):
            for x in range(8):
                image.putpixel((x, y), (x, y, 0))

        self.assertEqual(
            pixel_crop_boxes(8, 8),
            ((0, 0, 6, 6), (2, 0, 8, 6), (0, 2, 6, 8), (2, 2, 8, 8)),
        )
        views = crop_pil_views(image)
        self.assertEqual(CROP_NAMES, ("top_left", "top_right", "bottom_left", "bottom_right"))
        self.assertEqual([view.size for view in views], [(6, 6)] * 4)
        self.assertEqual([view.getpixel((0, 0))[:2] for view in views], [(0, 0), (2, 0), (0, 2), (2, 2)])

    def test_dataset_and_fake_encoder_return_n_by_four_by_768(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, color in enumerate((32, 192)):
                path = Path(directory) / f"{index}.png"
                Image.new("RGB", (8, 8), (color, color, color)).save(path)
                paths.append(path)
            dataset = FourCropDataset(paths, _tensor_preprocess)
            self.assertEqual(tuple(dataset[0].shape), (4, 3, 6, 6))
            encoded = encode_four_crops(
                _FakeEncoder(), _tensor_preprocess, paths, torch.device("cpu"), 2, 0
            )
            self.assertEqual(tuple(encoded.shape), (2, 4, 768))
            self.assertTrue(torch.isfinite(encoded).all())
            self.assertTrue(torch.allclose(encoded.norm(dim=-1), torch.ones(2, 4)))

    def test_manifest_records_parent_crop_runtime_and_output_contract(self):
        manifest = build_manifest(
            dataset="CUB",
            code_commit="a" * 40,
            script_sha256="b" * 64,
            source_config=Path("source.yaml"),
            source_config_sha256="c" * 64,
            parent_manifest=Path("parent.json"),
            parent_manifest_sha256="d" * 64,
            parent_asset_id="parent-v1",
            clip_checkpoint=Path("ViT-L-14-336px.pt"),
            clip_checkpoint_sha256="e" * 64,
            clip_runtime={"python_source_sha256": "f" * 64, "distribution_version": "1.0"},
            counts={"train": 2, "test_seen": 1, "test_unseen": 1},
            class_order_sha="1" * 64,
            raw_image_order_sha="2" * 64,
            parent_raw_image_order_sha="9" * 64,
            parent_raw_image_order_matches=False,
            full_row_alignment={"all_splits_verified": True},
            inputs_sha256={"res101": "3" * 64},
            outputs_sha256={"train_local_view_features.pt": "4" * 64},
            output_stats={
                "train_local_view_features.pt": {
                    "shape": [2, 4, 768],
                    "dtype": "torch.float32",
                    "l2_norm_min": 1.0,
                    "l2_norm_max": 1.0,
                    "finite": True,
                }
            },
            determinism_check={"repeat_bitwise_equal": True},
            parent_parity={"minimum_cosine": 0.9999},
        )
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["parent"]["asset_id"], "parent-v1")
        self.assertEqual(manifest["crop_semantics"]["order"], list(CROP_NAMES))
        self.assertEqual(
            manifest["crop_semantics"]["normalized_boxes_xyxy"],
            [list(box) for box in NORMALIZED_CROP_BOXES],
        )
        self.assertTrue(manifest["crop_semantics"]["crop_before_clip_preprocess"])
        self.assertFalse(manifest["crop_semantics"]["human_annotations_used"])
        self.assertEqual(
            manifest["source_alignment"],
            {
                "parent_raw_image_order_and_size_sha256": "9" * 64,
                "raw_image_order_fingerprint_matches_parent": False,
                "alignment_contract": "same_xlsa_res101_att_splits_class_order_and_all_split_labels_plus_full_view_parity",
                "aligned_through_linux_manifest": True,
            },
        )
        self.assertEqual(manifest["full_row_alignment"], {"all_splits_verified": True})
        self.assertFalse(manifest["unseen_images_used_for_gradient"])
        self.assertIn("python_source_sha256", manifest["clip"])
        self.assertEqual(
            manifest["output_tensors"]["train_local_view_features.pt"]["shape"],
            [2, 4, 768],
        )

    def test_run_rejects_existing_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory)
            with self.assertRaisesRegex(FileExistsError, "必须不存在"):
                run(
                    Path("missing-source.yaml"),
                    Path("missing-parent.json"),
                    Path("missing-alignment.json"),
                    "0" * 64,
                    existing,
                    device_name="cpu",
                    batch_size=1,
                    workers=0,
                )


if __name__ == "__main__":
    unittest.main()
