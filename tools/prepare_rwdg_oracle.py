from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F


ROOT = Path("/data/lby/projects/cv_project/GZSL_Warehouse")
PATCH_ROOT = ROOT / "assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1"
BUNDLE_ROOT = ROOT / "tries/v5/cuav/assets/V5-TRY-005-PRELIM"
TEXT_ROOT = ROOT / "assets/clip_vitl14_336/CUB/69c9c6d82a755fe8"
ATT_SPLITS = ROOT / "datasets/splits/xlsa17/data/CUB/att_splits.mat"

EXPECTED = {
    "patch_manifest": "d096087c9bd37d90157688e21e79b8ba6a61f0ea9b1fa91f4f544f8bc1dd1ad0",
    "patch_cls": "5c6e69fbfca4d41d73e133c6085e058e2c6f25237a34a7d00902c80b00b9db9a",
    "text_manifest": "52c50c2f55250399bce360a218c30e70b66945953bec7e825e8dd8f20dddf91f",
    "name_embeddings": "c3a2f177f728621a56d1e972b91614346eee47a749e2902db0af33fac0543232",
    "bundle_manifest": "0b956bb4445033e14bb692dd725fbf894db1f8e2fc337d78cf4a1b3b63cd3450",
    "eval_manifest": "2342392b5fb6f839c07a78922e2c3c59de63f68016e8d7cdbe9f6f11770d8af2",
    "oracle_manifest": "2e4db3b918b7ea915e272d4266fd6241e0aa7624e9d5483a7ffdcbf9348b9fea",
    "att_splits": "d7f5b4c2cb7853acdce43a9e87607ceed30bdf18c60344be3a266de29b6751e3",
    "action_geometry": "4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA mismatch for {path}: {actual} != {expected}")


def load_tensor(path: Path, manifest: dict, filename: str) -> torch.Tensor:
    require_sha(path, manifest["outputs_sha256"][filename])
    return torch.load(path, map_location="cpu", weights_only=True)


def stable_top2(logits: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    by_id = torch.argsort(class_ids, stable=True)
    ranked = torch.argsort(
        logits.index_select(1, by_id), dim=1, descending=True, stable=True
    )
    return by_id[ranked[:, :2]]


def class_vector(
    labels: torch.Tensor, predictions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(labels, sorted=True)
    vector = torch.stack(
        [predictions[labels.eq(class_id)].eq(class_id).double().mean() for class_id in classes]
    )
    return classes, vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    generator_sha = sha256_file(script_path)
    paths = {
        "patch_manifest": PATCH_ROOT / "asset_manifest.json",
        "patch_cls": PATCH_ROOT / "train_features.pt",
        "text_manifest": TEXT_ROOT / "asset_manifest.json",
        "name_embeddings": TEXT_ROOT / "class_name_embeds.pt",
        "bundle_manifest": BUNDLE_ROOT / "asset_manifest.json",
        "eval_manifest": BUNDLE_ROOT / "dev_eval/asset_manifest.json",
        "oracle_manifest": BUNDLE_ROOT / "dev_eval_oracle/asset_manifest.json",
        "att_splits": ATT_SPLITS,
    }
    for key, path in paths.items():
        require_sha(path, EXPECTED[key])

    patch_manifest = json.loads(paths["patch_manifest"].read_text(encoding="utf-8"))
    text_manifest = json.loads(paths["text_manifest"].read_text(encoding="utf-8"))
    bundle_manifest = json.loads(paths["bundle_manifest"].read_text(encoding="utf-8"))
    eval_manifest = json.loads(paths["eval_manifest"].read_text(encoding="utf-8"))
    oracle_manifest = json.loads(paths["oracle_manifest"].read_text(encoding="utf-8"))

    if bundle_manifest["common_identity"]["crop_action_sha256"] != EXPECTED["action_geometry"]:
        raise ValueError("action geometry mismatch")
    if oracle_manifest["crop_action_sha256"] != EXPECTED["action_geometry"]:
        raise ValueError("oracle action geometry mismatch")

    raw_indices = load_tensor(
        BUNDLE_ROOT / "dev_eval_oracle/raw_indices.pt", oracle_manifest, "raw_indices.pt"
    ).long()
    labels = load_tensor(
        BUNDLE_ROOT / "dev_eval_oracle/labels.pt", oracle_manifest, "labels.pt"
    ).long()
    class_ids = load_tensor(
        BUNDLE_ROOT / "dev_eval_oracle/class_ids.pt", oracle_manifest, "class_ids.pt"
    ).long()
    subset_names = load_tensor(
        BUNDLE_ROOT / "dev_eval_oracle/name_embeddings.pt",
        oracle_manifest,
        "name_embeddings.pt",
    ).float()
    full_names = load_tensor(
        paths["name_embeddings"], text_manifest, "class_name_embeds.pt"
    ).float()
    if not torch.equal(subset_names, full_names.index_select(0, class_ids)):
        raise ValueError("oracle name axis differs from authoritative text asset")

    crop_path = BUNDLE_ROOT / "dev_eval_oracle/crop_features.npy"
    require_sha(crop_path, oracle_manifest["outputs_sha256"]["crop_features.npy"])
    crop_memmap = np.load(crop_path, mmap_mode="r")
    if crop_memmap.shape != (2355, 25, 768) or crop_memmap.dtype != np.float16:
        raise ValueError(f"crop table contract mismatch: {crop_memmap.shape}/{crop_memmap.dtype}")

    trainval_global = (
        sio.loadmat(ATT_SPLITS)["trainval_loc"].reshape(-1).astype(np.int64) - 1
    )
    raw_to_trainval = {int(raw): pos for pos, raw in enumerate(trainval_global.tolist())}
    positions = np.asarray([raw_to_trainval[int(raw)] for raw in raw_indices.tolist()])
    full_cls_all = load_tensor(paths["patch_cls"], patch_manifest, "train_features.pt").float()
    full_cls = full_cls_all.index_select(0, torch.from_numpy(positions).long())

    names = F.normalize(subset_names, dim=-1)
    parent_logits = F.normalize(full_cls, dim=-1) @ names.T / 0.07
    top2 = stable_top2(parent_logits, class_ids)
    parent_prediction = class_ids[top2[:, 0]]

    crops = torch.from_numpy(np.asarray(crop_memmap).copy()).float()
    crop_logits = torch.einsum("bad,cd->bac", F.normalize(crops, dim=-1), names) / 0.07
    rows = torch.arange(labels.numel())
    leader = top2[:, 0]
    challenger = top2[:, 1]
    choices = torch.where(
        crop_logits[rows, :, leader] >= crop_logits[rows, :, challenger],
        class_ids[leader][:, None],
        class_ids[challenger][:, None],
    )
    correctable = choices.eq(labels[:, None]).any(dim=1)
    parent_correct = parent_prediction.eq(labels)
    truth_is_challenger = labels.eq(class_ids[challenger])
    oracle_prediction = parent_prediction.clone()
    corrected = (~parent_correct) & correctable
    oracle_prediction[corrected] = labels[corrected]

    classes, parent_vector = class_vector(labels, parent_prediction)
    oracle_classes, oracle_vector = class_vector(labels, oracle_prediction)
    if not torch.equal(classes, oracle_classes) or classes.numel() != 50:
        raise ValueError("class vector identity mismatch")
    parent_macro = 100.0 * float(parent_vector.mean())
    oracle_macro = 100.0 * float(oracle_vector.mean())
    damages = int((parent_correct & oracle_prediction.ne(labels)).sum())

    row_sha = hashlib.sha256(raw_indices.numpy().astype("<i8").tobytes()).hexdigest()
    axis_sha = hashlib.sha256(class_ids.numpy().astype("<i8").tobytes()).hexdigest()
    result = {
        "schema_version": "gzsl-paper.v5-rwdg-projected-pair-oracle.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_path": str(script_path),
        "generator_sha256": generator_sha,
        "idea_id": "IDEA-193",
        "experiment_id": "V5-TRY-007-RWDG-GATE0-ORACLE",
        "asset_identity": {
            **{key: {"path": str(paths[key]), "sha256": EXPECTED[key]} for key in paths},
            "crop_features_path": str(crop_path),
            "crop_features_sha256": oracle_manifest["outputs_sha256"]["crop_features.npy"],
            "action_geometry_sha256": EXPECTED["action_geometry"],
            "row_order_sha256": row_sha,
            "class_axis_sha256": axis_sha,
            "bundle_id": bundle_manifest["common_identity"]["bundle_id"],
        },
        "rows": int(labels.numel()),
        "active_classes": int(class_ids.numel()),
        "metric_classes": int(classes.numel()),
        "parent_macro_top1_percent": parent_macro,
        "pair_crop_oracle25_macro_top1_percent": oracle_macro,
        "oracle_gain_pp": oracle_macro - parent_macro,
        "parent_micro_correct": int(parent_correct.sum()),
        "parent_wrong": int((~parent_correct).sum()),
        "truth_is_challenger_count": int(truth_is_challenger.sum()),
        "correctable_any_row_count": int(correctable.sum()),
        "correctable_reachable_count": int((truth_is_challenger & correctable).sum()),
        "corrections": int(corrected.sum()),
        "damages": damages,
        "net_corrections": int(corrected.sum()) - damages,
        "pair_reachable_fraction": float((parent_correct | truth_is_challenger).double().mean()),
        "oracle_all25_opened": True,
        "used_for_training": False,
        "official_test_loaded": False,
        "unseen_images_used_for_gradient": False,
        "pclr_online_inference": False,
        "gates": {
            "oracle_gain_at_least_1pp": oracle_macro - parent_macro >= 1.0,
            "damage_zero": damages == 0,
            "rows_2355": int(labels.numel()) == 2355,
            "axis_150": int(class_ids.numel()) == 150,
        },
    }
    if not all(result["gates"].values()):
        raise RuntimeError(f"oracle preflight failed: {result['gates']}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **result["gates"]}))


if __name__ == "__main__":
    main()
