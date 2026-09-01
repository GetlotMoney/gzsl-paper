import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from model.frameworks.v6.rwdg_data import (
    ManifestContract,
    RWDGAssetConfig,
    RWDGDataError,
    TensorContract,
    load_rwdg_gate_data,
    sha256_file,
)


WINDOW_SHA = "4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb"
RAW_ORDER = np.asarray([11788, 42, 9000, 314, 7000, 105], dtype=np.int64)


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _write_pt(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return sha256_file(path)


def _write_patch_memmap(path: Path, rows: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(rows, 576, 768))
    array[:] = 0
    for index in range(rows):
        array[index, 0, 0] = np.float16(100 + index)
    array.flush()
    del array
    return sha256_file(path)


def _write_subset(
    root: Path,
    name: str,
    raw_indices: np.ndarray,
    *,
    crop_features_present: bool = False,
    crop_feature_output: bool = False,
) -> tuple[str, str]:
    directory = root / name
    raw_sha = _write_pt(directory / "raw_indices.pt", torch.as_tensor(raw_indices, dtype=torch.long))
    outputs = {"raw_indices.pt": raw_sha}
    if crop_feature_output:
        crop_path = directory / "crop_features.npy"
        crop = np.lib.format.open_memmap(crop_path, mode="w+", dtype=np.float16, shape=(len(raw_indices), 25, 768))
        crop[:] = 0
        crop.flush()
        del crop
        outputs["crop_features.npy"] = sha256_file(crop_path)
    manifest = {
        "schema_version": "gzsl-paper.cuav-crop-subset.v1",
        "subset": name,
        "count": int(len(raw_indices)),
        "crop_action_sha256": WINDOW_SHA,
        "crop_features_present": crop_features_present,
        "outputs_sha256": outputs,
    }
    manifest_path = directory / "asset_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    return str(manifest_path.relative_to(root)), manifest_sha


def _make_config(
    tmp_path: Path,
    *,
    train_raw: np.ndarray | None = None,
    eval_raw: np.ndarray | None = None,
    eval_crop_features_present: bool = False,
    eval_crop_feature_output: bool = False,
) -> RWDGAssetConfig:
    train_raw = np.asarray([11788, 9000, 7000], dtype=np.int64) if train_raw is None else train_raw
    eval_raw = np.asarray([42, 314, 105], dtype=np.int64) if eval_raw is None else eval_raw

    text_dir = tmp_path / "text"
    role_path = text_dir / "role_sentence_embeds.pt"
    name_path = text_dir / "class_name_embeds.pt"
    role_sha = _write_pt(role_path, torch.zeros(200, 8, 768, dtype=torch.float32))
    name_sha = _write_pt(name_path, torch.zeros(200, 768, dtype=torch.float32))
    text_manifest_path = text_dir / "asset_manifest.json"
    text_sha = _write_json(
        text_manifest_path,
        {
            "schema_version": "mock.text.v1",
            "outputs_sha256": {
                "role_sentence_embeds.pt": role_sha,
                "class_name_embeds.pt": name_sha,
            },
        },
    )

    patch_dir = tmp_path / "patch"
    cls = torch.zeros(len(RAW_ORDER), 768, dtype=torch.float32)
    for index in range(len(RAW_ORDER)):
        cls[index, 0] = float(10 + index)
    cls_path = patch_dir / "train_features.pt"
    patch_path = patch_dir / "train_patch_features.npy"
    cls_sha = _write_pt(cls_path, cls)
    patch_sha = _write_patch_memmap(patch_path, len(RAW_ORDER))
    patch_manifest_path = patch_dir / "asset_manifest.json"
    patch_manifest_sha = _write_json(
        patch_manifest_path,
        {
            "schema_version": "mock.projected_patch.v1",
            "description": "ln_post visual.proj l2 576 24",
            "outputs_sha256": {
                "train_features.pt": cls_sha,
                "train_patch_features.npy": patch_sha,
            },
        },
    )

    att_path = tmp_path / "att_splits.mat"
    savemat(att_path, {"trainval_loc": (RAW_ORDER + 1).reshape(-1, 1)})

    bundle_dir = tmp_path / "cuav_bundle"
    train_rel, train_sha = _write_subset(bundle_dir, "dev_train", train_raw)
    eval_rel, eval_sha = _write_subset(
        bundle_dir,
        "dev_eval",
        eval_raw,
        crop_features_present=eval_crop_features_present,
        crop_feature_output=eval_crop_feature_output,
    )
    oracle_rel, oracle_sha = _write_subset(bundle_dir, "dev_eval_oracle", eval_raw)
    bundle_path = bundle_dir / "asset_manifest.json"
    bundle_sha = _write_json(
        bundle_path,
        {
            "schema_version": "gzsl-paper.cuav-crop-bundle.v1",
            "subsets": {
                "dev_train": {"path": train_rel, "sha256": train_sha},
                "dev_eval": {"path": eval_rel, "sha256": eval_sha},
                "dev_eval_oracle": {"path": oracle_rel, "sha256": oracle_sha},
            },
        },
    )

    return RWDGAssetConfig(
        text_manifest=ManifestContract(str(text_manifest_path), text_sha),
        role_tensor=TensorContract(str(role_path), role_sha, (200, 8, 768), "float32"),
        name_tensor=TensorContract(str(name_path), name_sha, (200, 768), "float32"),
        patch_manifest=ManifestContract(str(patch_manifest_path), patch_manifest_sha),
        cls_tensor=TensorContract(str(cls_path), cls_sha, (len(RAW_ORDER), 768), "float32"),
        patch_tensor=TensorContract(str(patch_path), patch_sha, (len(RAW_ORDER), 576, 768), "float16"),
        cuav_bundle_manifest=ManifestContract(str(bundle_path), bundle_sha),
        dev_train_manifest_sha256=train_sha,
        dev_eval_manifest_sha256=eval_sha,
        dev_eval_oracle_manifest_sha256=oracle_sha,
        att_splits_mat_path=str(att_path),
        trainval_count=len(RAW_ORDER),
    )


def test_subset_views_map_raw_global_indices_to_trainval_positions(tmp_path):
    config = _make_config(tmp_path)

    _, views = load_rwdg_gate_data(config, strict_sha=True, validate_tensor_values=False)
    train = views["dev_train"].batch([0, 1, 2], include_patches=True, as_torch=False)
    eval_batch = views["dev_eval"].batch([0, 1, 2], include_patches=False, as_torch=False)

    assert train["raw_global_indices"].tolist() == [11788, 9000, 7000]
    assert train["trainval_positions"].tolist() == [0, 2, 4]
    assert train["cls"][:, 0].tolist() == [10.0, 12.0, 14.0]
    assert train["patches"][:, 0, 0].tolist() == [100.0, 102.0, 104.0]
    assert eval_batch["raw_global_indices"].tolist() == [42, 314, 105]
    assert eval_batch["trainval_positions"].tolist() == [1, 3, 5]


@pytest.mark.parametrize(
    ("flag", "output_name"),
    [
        (True, False),
        (False, True),
    ],
)
def test_dev_eval_crop_features_flag_or_output_is_rejected(tmp_path, flag, output_name):
    config = _make_config(
        tmp_path,
        eval_crop_features_present=flag,
        eval_crop_feature_output=output_name,
    )

    with pytest.raises(RWDGDataError, match="crop_features_present|forbidden crop handles"):
        load_rwdg_gate_data(config, strict_sha=True, validate_tensor_values=False)


def test_train_eval_overlap_is_rejected(tmp_path):
    config = _make_config(
        tmp_path,
        train_raw=np.asarray([11788, 9000, 7000], dtype=np.int64),
        eval_raw=np.asarray([42, 9000, 105], dtype=np.int64),
    )

    with pytest.raises(RWDGDataError, match="overlap"):
        load_rwdg_gate_data(config, strict_sha=True, validate_tensor_values=False)


def test_train_eval_union_must_cover_trainval_axis(tmp_path):
    config = _make_config(
        tmp_path,
        train_raw=np.asarray([11788, 9000], dtype=np.int64),
        eval_raw=np.asarray([42, 314, 105], dtype=np.int64),
    )

    with pytest.raises(RWDGDataError, match="union size"):
        load_rwdg_gate_data(config, strict_sha=True, validate_tensor_values=False)
