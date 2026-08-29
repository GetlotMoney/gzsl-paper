"""Create a frozen, dataset-specific eight-role text request from xlsa17 metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.gzsl_data import clean_class_name, class_order_sha256, load_xlsa_split


ROLE_SCHEMAS = {
    "CUB": [
        "beak",
        "head_features",
        "body_plumage",
        "wings",
        "tail",
        "legs",
        "overall_appearance",
        "unique_discriminative_features",
    ],
    "AWA2": [
        "head_and_face",
        "body_shape_and_size",
        "coat_color_and_texture",
        "limbs_and_tail",
        "posture_and_movement",
        "habitat_and_visual_context",
        "overall_appearance",
        "unique_discriminative_features",
    ],
    "SUN": [
        "spatial_layout",
        "primary_objects",
        "secondary_objects",
        "materials_and_surfaces",
        "color_and_lighting",
        "spatial_geometry_and_depth",
        "overall_appearance",
        "unique_discriminative_features",
    ],
}


def build_request(dataset: str, res101: Path, att_splits: Path) -> dict:
    dataset = dataset.upper()
    if dataset not in ROLE_SCHEMAS:
        raise ValueError("dataset只允许CUB/AWA2/SUN。")
    split = load_xlsa_split(res101, att_splits)
    roles = ROLE_SCHEMAS[dataset]
    classes = []
    for class_id, raw_name in enumerate(split.class_names):
        clean_name = clean_class_name(raw_name)
        classes.append(
            {
                "class_id": class_id,
                "xlsa_name": raw_name,
                "display_name": clean_name,
                "requested_roles": roles,
            }
        )
    return {
        "schema_version": "gzsl-paper.role-text-request.v1",
        "dataset": dataset,
        "class_order_sha256": class_order_sha256(split.class_names),
        "role_names": roles,
        "generation_contract": {
            "language": "English",
            "sentences_per_class": 8,
            "max_words_per_sentence": 35,
            "visual_evidence_only": True,
            "no_human_attributes": True,
            "no_accuracy_based_selection": True,
            "instruction": (
                "For every class, write exactly one visually grounded sentence for each "
                "requested role. Mention observable appearance or scene content only. Do not "
                "refer to dataset splits, seen/unseen status, human attribute vectors, or model predictions."
            ),
        },
        "classes": classes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(ROLE_SCHEMAS))
    parser.add_argument("--res101", type=Path, required=True)
    parser.add_argument("--att-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出已存在：{args.output}")
    payload = build_request(args.dataset, args.res101, args.att_splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
