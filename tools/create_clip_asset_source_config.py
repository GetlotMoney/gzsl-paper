"""Create one server-bound CLIP asset source config after role texts are frozen."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tools.runtime import sha256_file


WAREHOUSE = Path("/data/lby/projects/cv_project/GZSL_Warehouse")
DATASET_SETTINGS = {
    "CUB": {
        "archive": "CUB_200_2011.tgz",
        "archive_sha": "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081",
        "res101_sha": "9a97c71951f9ac9f5c3708e6e55386e53f3d433289ed5e87a18b4af2a1a0fca1",
        "att_sha": "d7f5b4c2cb7853acdce43a9e87607ceed30bdf18c60344be3a266de29b6751e3",
        "anchors": ["CUB_200_2011/images", "images"],
    },
    "AWA2": {
        "archive": "AwA2-data.zip",
        "archive_sha": "cc5a849879165acaa2b52f1de3f146ffcd1c475f6ef85bab0152c763e573744f",
        "res101_sha": "3592d2002a4e2d7ea4ef19069f9c35b8ad86c5e1a43b2d30257eba7018fd6dfa",
        "att_sha": "8bb1527856c797dcd3bac981b25832bca8e7fe04f3fc2ab2322850a74e848df1",
        "anchors": ["Animals_with_Attributes2/JPEGImages", "JPEGImages"],
    },
    "SUN": {
        "archive": "SUNAttributeDB_Images.tar.gz",
        "archive_sha": "ba27e21563227a915ad8b2f38877235984c7a200631cb39fd8e4a558f843fa86",
        "res101_sha": "f511e7280b28316312b0ddfa1f43a0289c3931b184a6a608b2e264d85e53cf65",
        "att_sha": "274b7f0761d1e76a63fca11dea1db6f5780fb1be919dfed0a24b3fda12f9616b",
        "anchors": ["SUN/images", "images"],
    },
}


def build_config(dataset: str, role_texts: Path) -> dict:
    dataset = dataset.upper()
    settings = DATASET_SETTINGS[dataset]
    if not role_texts.is_file():
        raise FileNotFoundError(f"八角色原文不存在：{role_texts}")
    split_root = WAREHOUSE / "datasets/splits/xlsa17/data" / dataset
    return {
        "schema_version": "gzsl-paper.clip-asset-source.v1",
        "dataset": dataset,
        "raw_root": str(WAREHOUSE / "datasets/raw" / dataset),
        "raw_archive": str(WAREHOUSE / "datasets/downloads" / settings["archive"]),
        "image_path_anchors": settings["anchors"],
        "res101": str(split_root / "res101.mat"),
        "att_splits": str(split_root / "att_splits.mat"),
        "role_texts": str(role_texts.resolve()),
        "clip_checkpoint": str(WAREHOUSE / "assets/clip_checkpoints/ViT-L-14-336px.pt"),
        "expected_sha256": {
            "raw_archive": settings["archive_sha"],
            "res101": settings["res101_sha"],
            "att_splits": settings["att_sha"],
            "role_texts": sha256_file(role_texts),
            "clip_checkpoint": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SETTINGS), required=True)
    parser.add_argument("--role-texts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"资产配置已存在：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(build_config(args.dataset, args.role_texts), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
