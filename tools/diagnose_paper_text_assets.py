"""Diagnose FRAMEWORK-V2 text assets with training images only."""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.run_contract import atomic_write_json
from tools.runtime import sha256_file


ASSET_SCHEMA = "gzsl-paper.clip-assets.v1"
OUTPUT_SCHEMA = "gzsl-paper.text-asset-diagnostics.v1"
EMBEDDING_DIMENSION = 768
ROLE_COUNT = 8
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ASSET_FILES = (
    "train_features.pt",
    "train_labels.pt",
    "class_name_embeds.pt",
    "role_sentence_embeds.pt",
)


def _validate_sha256(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in string.hexdigits for character in normalized):
        raise ValueError(f"{name}不是64位SHA256。")
    return normalized


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}必须是torch.Tensor。")
    return value


def _require_finite_nonzero_rows(value: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name}必须是浮点tensor。")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}包含NaN或Inf。")
    flattened = value.reshape(-1, value.shape[-1]).float()
    if torch.any(torch.linalg.vector_norm(flattened, dim=-1) <= 0):
        raise ValueError(f"{name}包含零向量。")


def _normalize_rows(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.detach().cpu().float()
    if value.ndim != 2:
        raise ValueError(f"{name}必须是二维tensor。")
    _require_finite_nonzero_rows(value, name)
    return F.normalize(value, dim=-1)


def _summary(values: torch.Tensor, *, include_std: bool = False) -> dict[str, float]:
    values = values.detach().cpu().float().reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("统计输入必须是非空有限tensor。")
    result = {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    if include_std:
        result["std"] = float(values.std(unbiased=False))
    return result


def normalized_visual_centers(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    seen_classes: torch.Tensor,
) -> torch.Tensor:
    """Return one L2-normalized mean training-image feature per seen class."""

    features = _require_tensor(train_features, "train_features").detach().cpu().float()
    labels = _require_tensor(train_labels, "train_labels").detach().cpu().long()
    class_ids = _require_tensor(seen_classes, "seen_classes").detach().cpu().long()
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.numel():
        raise ValueError("训练特征与标签形状不一致。")
    if class_ids.ndim != 1 or class_ids.numel() == 0 or class_ids.unique().numel() != class_ids.numel():
        raise ValueError("seen_classes必须是一维非空且无重复。")
    if not torch.isfinite(features).all():
        raise ValueError("训练特征包含NaN或Inf。")
    centers = []
    for class_id in class_ids.tolist():
        members = features[labels.eq(class_id)]
        if members.numel() == 0:
            raise ValueError(f"seen类别{class_id}没有训练图像。")
        centers.append(members.mean(dim=0))
    return _normalize_rows(torch.stack(centers), "seen_visual_centers")


def mean8_prototypes(role_embeddings: torch.Tensor) -> torch.Tensor:
    roles = _require_tensor(role_embeddings, "role_embeddings").detach().cpu().float()
    if roles.ndim != 3 or roles.shape[1] != ROLE_COUNT:
        raise ValueError(f"role_embeddings必须是[class_count,{ROLE_COUNT},dimension]。")
    _require_finite_nonzero_rows(roles, "role_embeddings")
    return _normalize_rows(roles.mean(dim=1), "Mean8")


def text_alignment_metrics(
    visual_centers: torch.Tensor,
    text_embeddings: torch.Tensor,
    seen_classes: torch.Tensor,
) -> dict[str, object]:
    """Compute class-level alignment, hardest-negative margin, and bidirectional retrieval."""

    centers = _normalize_rows(visual_centers, "visual_centers")
    texts = _normalize_rows(text_embeddings, "text_embeddings")
    class_ids = _require_tensor(seen_classes, "seen_classes").detach().cpu().long()
    if class_ids.ndim != 1 or class_ids.numel() < 2:
        raise ValueError("hardest-negative诊断至少需要两个seen类别。")
    if centers.shape[0] != class_ids.numel() or centers.shape[1] != texts.shape[1]:
        raise ValueError("视觉中心、seen类别和文本embedding维度不一致。")
    if int(class_ids.min()) < 0 or int(class_ids.max()) >= texts.shape[0]:
        raise ValueError("seen类别超出文本类别轴。")

    seen_texts = texts.index_select(0, class_ids)
    similarities = centers @ seen_texts.T
    corresponding = similarities.diagonal()
    hardest_negative_scores = similarities.clone()
    hardest_negative_scores.fill_diagonal_(float("-inf"))
    hardest_negative = hardest_negative_scores.max(dim=1).values
    margins = corresponding - hardest_negative
    positive = margins > 0
    expected = torch.arange(class_ids.numel())
    visual_to_text = similarities.argmax(dim=1).eq(expected)
    text_to_visual = similarities.T.argmax(dim=1).eq(expected)

    def retrieval(values: torch.Tensor) -> dict[str, int | float | str]:
        correct = int(values.sum())
        count = int(values.numel())
        rate = correct / count
        return {
            "definition": "macro over one normalized visual center per seen class",
            "correct_count": correct,
            "class_count": count,
            "per_class_rate": rate,
            "percent": 100.0 * rate,
        }

    return {
        "corresponding_class_cosine": _summary(corresponding),
        "hardest_negative_margin": {
            **_summary(margins),
            "positive_count": int(positive.sum()),
            "class_count": int(positive.numel()),
            "positive_margin_rate": float(positive.float().mean()),
            "positive_margin_percent": 100.0 * float(positive.float().mean()),
        },
        "visual_to_text_top1": retrieval(visual_to_text),
        "text_to_visual_top1": retrieval(text_to_visual),
    }


def role_difference_metrics(
    role_embeddings: torch.Tensor,
    seen_classes: torch.Tensor,
) -> dict[str, object]:
    """Measure whether the eight roles carry non-collapsed, class-local differences."""

    roles = _require_tensor(role_embeddings, "role_embeddings").detach().cpu().float()
    class_ids = _require_tensor(seen_classes, "seen_classes").detach().cpu().long()
    if roles.ndim != 3 or roles.shape[1] != ROLE_COUNT:
        raise ValueError(f"role_embeddings必须是[class_count,{ROLE_COUNT},dimension]。")
    if class_ids.ndim != 1 or class_ids.numel() == 0:
        raise ValueError("seen_classes必须是一维非空tensor。")
    if int(class_ids.min()) < 0 or int(class_ids.max()) >= roles.shape[0]:
        raise ValueError("seen类别超出角色embedding类别轴。")
    _require_finite_nonzero_rows(roles, "role_embeddings")

    selected = roles.index_select(0, class_ids)
    normalized_roles = F.normalize(selected, dim=-1)
    pair_indices = torch.triu_indices(ROLE_COUNT, ROLE_COUNT, offset=1)
    cosine_matrices = normalized_roles @ normalized_roles.transpose(1, 2)
    pairwise = cosine_matrices[:, pair_indices[0], pair_indices[1]]
    raw_means = normalized_roles.mean(dim=1)
    normalized_means = _normalize_rows(raw_means, "seen_Mean8")
    role_to_mean_cosine = (normalized_roles * normalized_means[:, None, :]).sum(dim=-1)
    cosine_distance = 1.0 - role_to_mean_cosine
    squared_deviation = (normalized_roles - raw_means[:, None, :]).square().sum(dim=-1)
    per_class_variance = squared_deviation.mean(dim=1)

    return {
        "within_class_role_pairwise_cosine": {
            "pairs_per_class": int(pairwise.shape[1]),
            "value_count": int(pairwise.numel()),
            **_summary(pairwise, include_std=True),
        },
        "per_class_pairwise_cosine_mean": _summary(pairwise.mean(dim=1), include_std=True),
        "role_to_Mean8_cosine_distance": {
            "value_count": int(cosine_distance.numel()),
            **_summary(cosine_distance, include_std=True),
            "per_role_mean": [float(value) for value in cosine_distance.mean(dim=0)],
        },
        "normalized_role_variance_around_class_mean": {
            "definition": "mean squared Euclidean deviation of normalized roles from their class role mean",
            **_summary(per_class_variance, include_std=True),
        },
    }


def parse_role_variant(value: str) -> tuple[str, Path, str]:
    name, first_separator, remainder = value.partition("=")
    path_text, last_separator, expected_sha = remainder.rpartition("=")
    if not first_separator or not last_separator or not name.strip() or not path_text.strip():
        raise ValueError("--role-variant格式必须为NAME=PATH=SHA256。")
    return name.strip(), Path(path_text), _validate_sha256(expected_sha.strip(), "role variant SHA")


def _validate_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ASSET_SCHEMA:
        raise ValueError("资产manifest schema错误。")
    if manifest.get("dataset") not in ("CUB", "AWA2", "SUN"):
        raise ValueError("资产manifest dataset错误。")
    for key in ("class_count", "seen_class_count", "train_count", "seen_classes", "outputs_sha256"):
        if key not in manifest:
            raise ValueError(f"资产manifest缺少{key}。")
    class_count = int(manifest["class_count"])
    seen_classes = [int(value) for value in manifest["seen_classes"]]
    if class_count <= 0 or len(seen_classes) != int(manifest["seen_class_count"]):
        raise ValueError("资产manifest类别计数错误。")
    if len(set(seen_classes)) != len(seen_classes) or any(
        value < 0 or value >= class_count for value in seen_classes
    ):
        raise ValueError("资产manifest seen类别轴错误。")
    outputs = manifest["outputs_sha256"]
    if not isinstance(outputs, dict):
        raise ValueError("资产manifest outputs_sha256错误。")
    for filename in TRAINING_ASSET_FILES:
        if filename not in outputs:
            raise ValueError(f"资产manifest缺少训练诊断输入：{filename}")
        _validate_sha256(outputs[filename], f"{filename} SHA")
    return manifest


def load_seen_only_assets(
    manifest_path: Path,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict, dict[str, str]]:
    """Load only the four whitelisted training/text tensors and verify every SHA."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"资产manifest不存在：{manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        expected = _validate_sha256(expected_manifest_sha256, "asset manifest SHA")
        if manifest_sha != expected:
            raise ValueError(f"资产manifest SHA不匹配：{manifest_sha}")
    manifest = _validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    tensors: dict[str, torch.Tensor] = {}
    input_sha = {"asset_manifest.json": manifest_sha}
    for filename in TRAINING_ASSET_FILES:
        path = manifest_path.parent / filename
        expected = str(manifest["outputs_sha256"][filename]).lower()
        if not path.is_file():
            raise FileNotFoundError(f"诊断输入不存在：{path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"诊断输入SHA不匹配：{filename}={actual}")
        tensors[filename.removesuffix(".pt")] = _require_tensor(
            torch.load(path, map_location="cpu", weights_only=True), filename
        )
        input_sha[filename] = actual

    class_count = int(manifest["class_count"])
    train_count = int(manifest["train_count"])
    expected_shapes = {
        "train_features": (train_count, EMBEDDING_DIMENSION),
        "train_labels": (train_count,),
        "class_name_embeds": (class_count, EMBEDDING_DIMENSION),
        "role_sentence_embeds": (class_count, ROLE_COUNT, EMBEDDING_DIMENSION),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name}形状错误：{tuple(tensors[name].shape)} != {shape}")
    labels = tensors["train_labels"]
    if torch.is_floating_point(labels) or torch.is_complex(labels):
        raise TypeError("train_labels必须是整数tensor。")
    if labels.numel() == 0:
        raise ValueError("train_labels不能为空。")
    labels = labels.long()
    if int(labels.min()) < 0 or int(labels.max()) >= class_count:
        raise ValueError("train_labels超出类别轴。")
    manifest_seen = sorted(int(value) for value in manifest["seen_classes"])
    if torch.unique(labels, sorted=True).tolist() != manifest_seen:
        raise ValueError("训练标签类别与manifest seen类别不一致。")
    tensors["train_labels"] = labels
    _require_finite_nonzero_rows(tensors["train_features"], "train_features")
    _require_finite_nonzero_rows(tensors["class_name_embeds"], "class_name_embeds")
    _require_finite_nonzero_rows(tensors["role_sentence_embeds"], "role_sentence_embeds")
    return tensors, manifest, input_sha


def load_role_variant(
    path: Path,
    expected_sha256: str,
    class_count: int,
) -> tuple[torch.Tensor, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"role variant不存在：{path}")
    actual = sha256_file(path)
    if actual != _validate_sha256(expected_sha256, "role variant SHA"):
        raise ValueError(f"role variant SHA不匹配：{actual}")
    value = _require_tensor(
        torch.load(path, map_location="cpu", weights_only=True), "role variant"
    ).detach().cpu()
    expected_shape = (int(class_count), ROLE_COUNT, EMBEDDING_DIMENSION)
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"role variant形状错误：{tuple(value.shape)} != {expected_shape}")
    _require_finite_nonzero_rows(value, "role variant")
    return value, actual


def diagnose_tensors(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    class_name_embeddings: torch.Tensor,
    role_versions: dict[str, torch.Tensor],
    seen_classes: torch.Tensor,
) -> dict[str, object]:
    """Pure tensor entry point used by the CLI and unit tests."""

    centers = normalized_visual_centers(train_features, train_labels, seen_classes)
    versions: dict[str, object] = {
        "class-name": {
            "text_construction": "a photo of a {class name}",
            "alignment": text_alignment_metrics(centers, class_name_embeddings, seen_classes),
        }
    }
    for name, roles in role_versions.items():
        if name in versions:
            raise ValueError(f"文本版本名重复：{name}")
        versions[name] = {
            "text_construction": "L2-normalized mean of eight role sentence embeddings",
            "alignment": text_alignment_metrics(centers, mean8_prototypes(roles), seen_classes),
            "role_difference": role_difference_metrics(roles, seen_classes),
        }
    return {
        "seen_visual_center_count": int(centers.shape[0]),
        "text_versions": versions,
    }


@torch.no_grad()
def run(
    manifest_path: Path,
    output_json: Path,
    *,
    base_role_name: str = "text-v1",
    expected_manifest_sha256: str | None = None,
    role_variants: list[tuple[str, Path, str]] | None = None,
) -> dict[str, object]:
    if not base_role_name.strip() or base_role_name == "class-name":
        raise ValueError("base role name不能为空或class-name。")
    output_json = output_json.resolve()
    if output_json == REPOSITORY_ROOT or output_json.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("诊断JSON必须写到仓库外。")
    if output_json.exists():
        raise FileExistsError(f"诊断JSON已存在，禁止覆盖：{output_json}")

    tensors, manifest, input_sha = load_seen_only_assets(
        manifest_path, expected_manifest_sha256
    )
    class_count = int(manifest["class_count"])
    variants: dict[str, torch.Tensor] = {
        base_role_name: tensors["role_sentence_embeds"]
    }
    variant_sources: dict[str, dict[str, str | int]] = {}
    for name, path, expected_sha in role_variants or []:
        if name in variants or name == "class-name":
            raise ValueError(f"role variant名称重复或保留：{name}")
        value, actual_sha = load_role_variant(path, expected_sha, class_count)
        variants[name] = value
        variant_sources[name] = {
            "path": str(path.resolve()),
            "sha256": actual_sha,
            "class_axis_count": class_count,
            "class_order_sha256": str(manifest["class_order_sha256"]),
        }

    seen_classes = torch.tensor(manifest["seen_classes"], dtype=torch.long)
    diagnostics = diagnose_tensors(
        tensors["train_features"],
        tensors["train_labels"],
        tensors["class_name_embeds"],
        variants,
        seen_classes,
    )
    payload: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "dataset": manifest["dataset"],
        "asset_id": manifest["asset_id"],
        "asset_manifest": str(manifest_path.resolve()),
        "official_test_loaded": False,
        "seen_images_only": True,
        "unseen_images_used": False,
        "class_count": class_count,
        "seen_class_count": int(manifest["seen_class_count"]),
        "train_image_count": int(manifest["train_count"]),
        "seen_classes": [int(value) for value in manifest["seen_classes"]],
        "class_order_sha256": str(manifest["class_order_sha256"]),
        "input_sha256": input_sha,
        "role_variant_sources": variant_sources,
        **diagnostics,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_json, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest-sha256")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--base-role-name", default="text-v1")
    parser.add_argument(
        "--role-variant",
        action="append",
        default=[],
        metavar="NAME=PATH=SHA256",
    )
    args = parser.parse_args()
    result = run(
        args.asset_manifest,
        args.output_json,
        base_role_name=args.base_role_name,
        expected_manifest_sha256=args.asset_manifest_sha256,
        role_variants=[parse_role_variant(value) for value in args.role_variant],
    )
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "output_json": str(args.output_json.resolve()),
                "text_versions": list(result["text_versions"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
