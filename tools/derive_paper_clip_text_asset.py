"""Derive a new paper asset by re-encoding role texts and reusing visual caches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import string
import tempfile
from pathlib import Path
from typing import Callable

import torch
import yaml

from tools.gzsl_data import class_order_sha256, clean_class_name
from tools.prepare_paper_clip_assets import (
    MODEL_NAME,
    OFFICIAL_CHECKPOINT_SHA256,
    _atomic_torch_save,
    _encode_texts,
    load_role_texts,
)
from tools.runtime import sha256_file


CONFIG_SCHEMA = "gzsl-paper.clip-text-asset-derivation.v1"
ASSET_SCHEMA = "gzsl-paper.clip-assets.v1"
TEXT_VERSION = "text-v2"
ENCODER_IDENTITY_SCHEMA = "gzsl-paper.clip-text-encoder.v1"
ASSET_IDENTITY_SCHEMA = "gzsl-paper.clip-text-derived-asset.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLIP_SOURCE_FILES = (
    "__init__.py",
    "clip.py",
    "model.py",
    "simple_tokenizer.py",
    "bpe_simple_vocab_16e6.txt.gz",
)
CONFIG_KEYS = {
    "schema_version",
    "dataset",
    "parent_manifest",
    "parent_manifest_sha256",
    "role_texts",
    "role_texts_sha256",
    "clip_checkpoint",
    "clip_checkpoint_sha256",
}
REUSED_OUTPUTS = (
    "train_features.pt",
    "train_labels.pt",
    "test_seen_features.pt",
    "test_seen_labels.pt",
    "test_unseen_features.pt",
    "test_unseen_labels.pt",
    "class_name_embeds.pt",
    "class_names.json",
)


def _validate_sha256(value: object, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in string.hexdigits for character in normalized):
        raise ValueError(f"{name}不是64位SHA256。")
    return normalized


def _absolute_file(value: object, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"{name}必须是绝对路径。")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}不存在：{path}")
    return path


def load_config(path: Path) -> tuple[dict, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != CONFIG_KEYS:
        actual = set(payload) if isinstance(payload, dict) else set()
        raise ValueError(
            f"text-v2派生配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("text-v2派生配置schema错误。")
    if payload["dataset"] not in ("CUB", "AWA2", "SUN"):
        raise ValueError("dataset只允许CUB/AWA2/SUN。")
    for key in (
        "parent_manifest_sha256",
        "role_texts_sha256",
        "clip_checkpoint_sha256",
    ):
        payload[key] = _validate_sha256(payload[key], key)
    for key in ("parent_manifest", "role_texts", "clip_checkpoint"):
        payload[key] = str(_absolute_file(payload[key], key))
    return payload, sha256_file(path)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_encoder_identity(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != ENCODER_IDENTITY_SCHEMA
        or not isinstance(value.get("implementation"), str)
        or not value["implementation"].strip()
    ):
        raise ValueError("text encoder identity不完整。")
    try:
        _canonical_sha256(value)
    except (TypeError, ValueError) as error:
        raise ValueError("text encoder identity必须可稳定JSON序列化。") from error
    return copy.deepcopy(value)


def derived_asset_id(
    config: dict, role_embedding_sha256: str, encoder_identity: object
) -> str:
    encoder_identity = _validated_encoder_identity(encoder_identity)
    identity = {
        key: config[key]
        for key in (
            "parent_manifest_sha256",
            "role_texts_sha256",
            "clip_checkpoint_sha256",
        )
    }
    identity["schema_version"] = ASSET_IDENTITY_SCHEMA
    identity["role_sentence_embeds_sha256"] = _validate_sha256(
        role_embedding_sha256, "role_sentence_embeds_sha256"
    )
    identity["encoder_identity"] = encoder_identity
    return _canonical_sha256(identity)[:16]


def natural_class_name(value: str) -> str:
    tokens = clean_class_name(value).replace("+", " ").split()
    collapsed: list[str] = []
    for token in tokens:
        if not collapsed or token.casefold() != collapsed[-1].casefold():
            collapsed.append(token)
    return " ".join(collapsed)


def validate_clip_friendly_v2(
    class_names: tuple[str, ...],
    descriptions: list[list[str]],
    generator_identity: object,
) -> None:
    if not isinstance(generator_identity, dict) or generator_identity.get(
        "generation_method"
    ) != "clip_anchored_class_specific_eight_role_descriptions_v2":
        raise ValueError("text-v2 generator身份错误。")
    if len(descriptions) != len(class_names):
        raise ValueError("text-v2类别数量错误。")
    display_names = generator_identity.get("display_names")
    if display_names is None:
        display_names = [natural_class_name(value) for value in class_names]
    elif (
        not isinstance(display_names, list)
        or len(display_names) != len(class_names)
        or any(not isinstance(value, str) or not value.strip() for value in display_names)
    ):
        raise ValueError("generator.display_names必须逐类提供非空冻结名称。")
    generic_only = {
        "body",
        "head",
        "wing",
        "wings",
        "tail",
        "legs",
        "appearance",
        "feature",
        "scene",
        "environment",
    }
    for class_index, (raw_name, rows) in enumerate(zip(class_names, descriptions, strict=True)):
        natural = " ".join(display_names[class_index].split()).casefold()
        if "+" in natural or "_" in natural:
            raise ValueError(f"类别{class_index}的display name含非自然分隔符。")
        anchors = tuple(
            f"a photo of {article} {natural}, showing " for article in ("a", "an", "the")
        )
        normalized_rows: set[str] = set()
        for role_index, sentence in enumerate(rows):
            stripped = " ".join(sentence.split())
            lowered = stripped.casefold()
            if len(stripped.split()) > 22:
                raise ValueError(f"类别{class_index}角色{role_index}超过22词。")
            if "+" in stripped or "_" in stripped:
                raise ValueError(f"类别{class_index}角色{role_index}含非自然类名分隔符。")
            prefix = next((value for value in anchors if lowered.startswith(value)), None)
            if prefix is None:
                raise ValueError(f"类别{class_index}角色{role_index}缺少自然类名照片锚点。")
            detail = lowered[len(prefix) :].strip(" .,:;!?")
            detail_words = detail.split()
            if not detail_words or detail in generic_only:
                raise ValueError(f"类别{class_index}角色{role_index}缺少具体可见特征。")
            normalized = " ".join(
                "".join(character for character in word if character.isalnum())
                for word in lowered.split()
            )
            if normalized in normalized_rows:
                raise ValueError(f"类别{class_index}的8个角色包含重复句。")
            normalized_rows.add(normalized)


def _load_parent(config: dict) -> tuple[dict, Path, tuple[str, ...]]:
    manifest_path = Path(config["parent_manifest"])
    if sha256_file(manifest_path) != config["parent_manifest_sha256"]:
        raise ValueError("parent manifest SHA不匹配。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ASSET_SCHEMA or manifest.get("dataset") != config["dataset"]:
        raise ValueError("parent manifest身份错误。")
    if manifest.get("model") != MODEL_NAME:
        raise ValueError("parent asset不是冻结的OpenAI ViT-L/14@336px模型。")
    if manifest.get("clip_checkpoint_sha256") != config["clip_checkpoint_sha256"]:
        raise ValueError("parent asset与派生配置的CLIP checkpoint不一致。")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or not set(REUSED_OUTPUTS).issubset(outputs):
        raise ValueError("parent manifest缺少可复用缓存。")
    for filename in REUSED_OUTPUTS:
        source = manifest_path.parent / filename
        if not source.is_file() or sha256_file(source) != outputs[filename]:
            raise ValueError(f"parent缓存缺失或SHA错误：{filename}")
    names_payload = json.loads((manifest_path.parent / "class_names.json").read_text(encoding="utf-8"))
    class_names = tuple(str(value) for value in names_payload.get("xlsa", ()))
    if len(class_names) != int(manifest["class_count"]):
        raise ValueError("parent class_names类别数量错误。")
    if class_order_sha256(class_names) != manifest["class_order_sha256"]:
        raise ValueError("parent class_names类别顺序SHA错误。")
    return manifest, manifest_path, class_names


def _production_encoder_identity(parent: dict, batch_size: int) -> dict:
    if int(batch_size) <= 0:
        raise ValueError("batch-size必须是正整数。")

    import clip

    package_root = Path(clip.__file__).resolve().parent
    source_hashes: dict[str, str] = {}
    for filename in CLIP_SOURCE_FILES:
        source = package_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"OpenAI CLIP源码文件不存在：{source}")
        source_hashes[filename] = sha256_file(source)

    expected_clip_source_sha = parent.get("clip_python_source_sha256")
    if (
        not isinstance(expected_clip_source_sha, str)
        or source_hashes["clip.py"] != expected_clip_source_sha
    ):
        raise ValueError("运行时OpenAI CLIP clip.py SHA与parent asset不一致。")

    distribution = importlib.metadata.distribution("clip")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    if (
        "clip_distribution_version" not in parent
        or distribution.version != parent["clip_distribution_version"]
    ):
        raise ValueError("运行时OpenAI CLIP distribution version与parent asset不一致。")
    if (
        "clip_distribution_direct_url" not in parent
        or direct_url != parent["clip_distribution_direct_url"]
    ):
        raise ValueError("运行时OpenAI CLIP direct_url与parent asset不一致。")

    return {
        "schema_version": ENCODER_IDENTITY_SCHEMA,
        "implementation": "openai_clip_encode_text_l2_normalized_v1",
        "model": MODEL_NAME,
        "clip_checkpoint_sha256": parent["clip_checkpoint_sha256"],
        "embedding_dimension": 768,
        "tokenize_truncate": False,
        "l2_normalized": True,
        "batch_size": int(batch_size),
        "clip_distribution_version": distribution.version,
        "clip_distribution_direct_url": direct_url,
        "clip_source_files_sha256": source_hashes,
    }


def _production_text_encoder(
    texts: list[str], checkpoint: Path, device_name: str, batch_size: int
) -> torch.Tensor:
    import clip

    model, _ = clip.load(str(checkpoint), device=torch.device(device_name), jit=False)
    model.eval()
    return _encode_texts(model, clip, texts, torch.device(device_name), batch_size)


def run(
    config_path: Path,
    output_root: Path,
    *,
    device_name: str,
    batch_size: int = 256,
    _text_encoder: Callable[[list[str], Path, str, int], torch.Tensor] | None = None,
    _encoder_identity: dict | None = None,
) -> dict:
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise ValueError("output-root必须是绝对路径。")
    output_root = output_root.resolve()
    if output_root == REPOSITORY_ROOT or output_root.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("text-v2资产必须写到Git仓库外。")
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"output-root不是目录：{output_root}")
    if int(batch_size) <= 0:
        raise ValueError("batch-size必须是正整数。")

    config, config_sha = load_config(config_path)
    checkpoint = Path(config["clip_checkpoint"])
    role_text_path = Path(config["role_texts"])
    if sha256_file(checkpoint) != config["clip_checkpoint_sha256"]:
        raise ValueError("CLIP checkpoint SHA不匹配。")
    if config["clip_checkpoint_sha256"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("CLIP checkpoint不是OpenAI ViT-L/14@336px官方权重。")
    if sha256_file(role_text_path) != config["role_texts_sha256"]:
        raise ValueError("text-v2原文SHA不匹配。")
    parent, parent_manifest_path, class_names = _load_parent(config)
    if _text_encoder is None:
        if _encoder_identity is not None:
            raise ValueError("生产encoder identity必须由运行时自动采集。")
        encoder_identity = _production_encoder_identity(parent, int(batch_size))
        encoder = _production_text_encoder
    else:
        if _encoder_identity is None:
            raise ValueError("注入text encoder时必须提供版本化identity。")
        encoder_identity = _validated_encoder_identity(_encoder_identity)
        encoder = _text_encoder
    roles, descriptions, generator_identity = load_role_texts(
        role_text_path, config["dataset"], class_names
    )
    parent_roles = parent.get("role_names")
    if not isinstance(parent_roles, list) or roles != parent_roles:
        raise ValueError("text-v2 role_names与parent asset冻结顺序不一致。")
    validate_clip_friendly_v2(class_names, descriptions, generator_identity)
    flattened = [sentence for rows in descriptions for sentence in rows]

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".clip-text-asset.", dir=output_root))
    try:
        role_embeddings = encoder(flattened, checkpoint, device_name, int(batch_size))
        expected_shape = (int(parent["class_count"]) * 8, 768)
        if (
            tuple(role_embeddings.shape) != expected_shape
            or not torch.isfinite(role_embeddings).all()
        ):
            raise RuntimeError(
                f"text-v2 embedding形状或有限性错误：{tuple(role_embeddings.shape)}"
            )
        norms = torch.linalg.vector_norm(role_embeddings.float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0):
            raise RuntimeError("text-v2 embedding没有逐句L2归一化。")
        role_embeddings = role_embeddings.reshape(
            int(parent["class_count"]), 8, 768
        ).cpu()
        role_embedding_path = temporary / "role_sentence_embeds.pt"
        _atomic_torch_save(role_embedding_path, role_embeddings)
        role_embedding_sha = sha256_file(role_embedding_path)
        asset_id = derived_asset_id(config, role_embedding_sha, encoder_identity)
        output_dir = output_root / asset_id
        if output_dir.exists():
            raise FileExistsError(f"派生资产目录已存在：{output_dir}")

        hardlink_checks: dict[str, bool] = {}
        for filename in REUSED_OUTPUTS:
            source = parent_manifest_path.parent / filename
            destination = temporary / filename
            os.link(source, destination)
            hardlink_checks[filename] = os.path.samefile(source, destination)
            if not hardlink_checks[filename]:
                raise RuntimeError(f"视觉缓存未按hardlink复用：{filename}")
        output_sha = {
            filename: sha256_file(temporary / filename)
            for filename in (*REUSED_OUTPUTS, "role_sentence_embeds.pt")
        }
        for filename in REUSED_OUTPUTS:
            parent_sha = parent["outputs_sha256"][filename]
            if output_sha[filename] != parent_sha:
                raise RuntimeError(f"hardlink后parent缓存SHA发生变化：{filename}")

        if sha256_file(config_path) != config_sha:
            raise RuntimeError("派生配置在运行期间发生变化。")
        if sha256_file(parent_manifest_path) != config["parent_manifest_sha256"]:
            raise RuntimeError("parent manifest在运行期间发生变化。")
        if sha256_file(role_text_path) != config["role_texts_sha256"]:
            raise RuntimeError("text-v2原文在运行期间发生变化。")
        if sha256_file(checkpoint) != config["clip_checkpoint_sha256"]:
            raise RuntimeError("CLIP checkpoint在运行期间发生变化。")
        if _text_encoder is None:
            final_encoder_identity = _production_encoder_identity(
                parent, int(batch_size)
            )
            if final_encoder_identity != encoder_identity:
                raise RuntimeError("OpenAI CLIP encoder identity在运行期间发生变化。")

        manifest = copy.deepcopy(parent)
        manifest.update(
            {
                "asset_id": asset_id,
                "source_config_sha256": config_sha,
                "model": MODEL_NAME,
                "role_names": roles,
                "role_text_generator": generator_identity,
                "outputs_sha256": output_sha,
                "text_encoder_identity": encoder_identity,
                "text_encoder_identity_sha256": _canonical_sha256(
                    encoder_identity
                ),
                "derived_from_asset_id": parent["asset_id"],
                "derived_from_manifest_sha256": config["parent_manifest_sha256"],
                "derivation_kind": "role_text_only_reencode_visual_cache_hardlinked",
                "text_asset_version": TEXT_VERSION,
                "reused_visual_and_label_cache": True,
                "hardlink_verified": hardlink_checks,
            }
        )
        if _text_encoder is None:
            manifest["clip_source_files_sha256"] = encoder_identity[
                "clip_source_files_sha256"
            ]
        manifest["source_uris"] = dict(parent.get("source_uris", {}))
        manifest["source_uris"]["role_texts"] = str(role_text_path)
        manifest["source_uris"]["clip_checkpoint"] = str(checkpoint)
        manifest["inputs_sha256"] = dict(parent.get("inputs_sha256", {}))
        manifest["inputs_sha256"]["role_texts"] = config["role_texts_sha256"]
        manifest["inputs_sha256"]["clip_checkpoint"] = config["clip_checkpoint_sha256"]
        (temporary / "asset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"派生资产目录在发布前已出现：{output_dir}")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    manifest["asset_manifest_sha256"] = sha256_file(output_dir / "asset_manifest.json")
    manifest["asset_directory"] = str(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.output_root,
                device_name=args.device,
                batch_size=args.batch_size,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
