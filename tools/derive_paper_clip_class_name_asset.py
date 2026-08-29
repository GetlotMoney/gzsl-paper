"""Derive canonical Pure-CLIP class-name embeddings from frozen text-v2 prefixes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import string
import tempfile
from pathlib import Path
from typing import Callable

import torch
import yaml

from tools.derive_paper_clip_text_asset import (
    ASSET_SCHEMA,
    ENCODER_IDENTITY_SCHEMA,
    MODEL_NAME,
    OFFICIAL_CHECKPOINT_SHA256,
    REPOSITORY_ROOT,
    _canonical_sha256,
    _production_encoder_identity,
    _production_text_encoder,
    _validated_encoder_identity,
    natural_class_name,
    validate_clip_friendly_v2,
)
from tools.gzsl_data import class_order_sha256
from tools.prepare_paper_clip_assets import _atomic_torch_save, load_role_texts
from tools.runtime import sha256_file


CONFIG_SCHEMA = "gzsl-paper.clip-class-name-asset-derivation.v1"
ASSET_IDENTITY_SCHEMA = "gzsl-paper.clip-canonical-class-name-asset.v1"
CLASS_NAME_VERSION = "canonical-class-name-v2"
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
    "role_sentence_embeds.pt",
)
ALL_OUTPUTS = (*REUSED_OUTPUTS, "class_name_embeds.pt", "class_names.json")
PREFIX_PATTERN = re.compile(r"^(a photo of (?:a|an|the) .+?), showing ", re.IGNORECASE)


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
            f"canonical class-name配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("canonical class-name配置schema错误。")
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


def canonical_class_prompts(
    class_names: tuple[str, ...],
    descriptions: list[list[str]],
    generator_identity: object,
) -> tuple[list[str], list[str]]:
    """Use the exact shared `a photo of ...` prefix from each class's eight v2 sentences."""

    validate_clip_friendly_v2(class_names, descriptions, generator_identity)
    if not isinstance(generator_identity, dict):
        raise ValueError("text-v2 generator身份错误。")
    display_names = generator_identity.get("display_names")
    if display_names is None:
        display_names = [natural_class_name(value) for value in class_names]
    if not isinstance(display_names, list) or len(display_names) != len(class_names):
        raise ValueError("冻结display_names数量错误。")

    prompts: list[str] = []
    for class_index, (display_name, rows) in enumerate(
        zip(display_names, descriptions, strict=True)
    ):
        prefixes = []
        for role_index, sentence in enumerate(rows):
            match = PREFIX_PATTERN.match(" ".join(sentence.split()))
            if match is None:
                raise ValueError(f"类别{class_index}角色{role_index}缺少可提取的照片前缀。")
            prefixes.append(match.group(1))
        if len({value.casefold() for value in prefixes}) != 1:
            raise ValueError(f"类别{class_index}的8句没有共享同一照片前缀。")
        expected_tail = " ".join(str(display_name).split()).casefold()
        if not prefixes[0].casefold().endswith(" " + expected_tail):
            raise ValueError(f"类别{class_index}照片前缀与冻结display name不一致。")
        prompts.append(prefixes[0] + ".")
    return [" ".join(str(value).split()) for value in display_names], prompts


def derived_asset_id(
    config: dict,
    class_name_embedding_sha256: str,
    class_names_json_sha256: str,
    encoder_identity: object,
) -> str:
    identity = {
        "schema_version": ASSET_IDENTITY_SCHEMA,
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "role_texts_sha256": config["role_texts_sha256"],
        "clip_checkpoint_sha256": config["clip_checkpoint_sha256"],
        "class_name_embeds_sha256": _validate_sha256(
            class_name_embedding_sha256, "class_name_embeds_sha256"
        ),
        "class_names_json_sha256": _validate_sha256(
            class_names_json_sha256, "class_names_json_sha256"
        ),
        "encoder_identity": _validated_encoder_identity(encoder_identity),
    }
    return _canonical_sha256(identity)[:16]


def _load_parent(config: dict) -> tuple[dict, Path, tuple[str, ...]]:
    manifest_path = Path(config["parent_manifest"])
    if sha256_file(manifest_path) != config["parent_manifest_sha256"]:
        raise ValueError("parent manifest SHA不匹配。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ASSET_SCHEMA or manifest.get("dataset") != config["dataset"]:
        raise ValueError("parent manifest身份错误。")
    if manifest.get("model") != MODEL_NAME:
        raise ValueError("parent asset模型身份错误。")
    if manifest.get("clip_checkpoint_sha256") != config["clip_checkpoint_sha256"]:
        raise ValueError("parent asset与配置的CLIP checkpoint不一致。")
    if manifest.get("text_asset_version") != "text-v2":
        raise ValueError("parent asset不是冻结text-v2资产。")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or not set(ALL_OUTPUTS).issubset(outputs):
        raise ValueError("parent manifest缺少完整缓存。")
    for filename in ALL_OUTPUTS:
        source = manifest_path.parent / filename
        if not source.is_file() or sha256_file(source) != outputs[filename]:
            raise ValueError(f"parent缓存缺失或SHA错误：{filename}")
    class_names_path = manifest_path.parent / "class_names.json"
    names_payload = json.loads(class_names_path.read_text(encoding="utf-8"))
    class_names = tuple(str(value) for value in names_payload.get("xlsa", ()))
    if len(class_names) != int(manifest["class_count"]):
        raise ValueError("parent class_names类别数量错误。")
    if class_order_sha256(class_names) != manifest["class_order_sha256"]:
        raise ValueError("parent class_names类别顺序SHA错误。")
    role_path = Path(config["role_texts"])
    if str(Path(manifest["source_uris"]["role_texts"]).resolve()) != str(role_path.resolve()):
        raise ValueError("配置role_texts与parent manifest来源不一致。")
    if manifest.get("inputs_sha256", {}).get("role_texts") != config["role_texts_sha256"]:
        raise ValueError("配置role_texts SHA与parent manifest不一致。")
    return manifest, manifest_path, class_names


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
        raise ValueError("canonical class-name资产必须写到Git仓库外。")
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"output-root不是目录：{output_root}")
    if int(batch_size) <= 0:
        raise ValueError("batch-size必须是正整数。")

    config, config_sha = load_config(Path(config_path))
    checkpoint = Path(config["clip_checkpoint"])
    role_text_path = Path(config["role_texts"])
    if sha256_file(checkpoint) != config["clip_checkpoint_sha256"]:
        raise ValueError("CLIP checkpoint SHA不匹配。")
    if config["clip_checkpoint_sha256"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("CLIP checkpoint不是官方ViT-L/14@336px权重。")
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
    if roles != parent.get("role_names"):
        raise ValueError("text-v2 role_names与parent asset不一致。")
    display_names, prompts = canonical_class_prompts(
        class_names, descriptions, generator_identity
    )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".clip-class-name-asset.", dir=output_root))
    try:
        embeddings = encoder(prompts, checkpoint, device_name, int(batch_size))
        expected_shape = (int(parent["class_count"]), 768)
        if tuple(embeddings.shape) != expected_shape or not torch.isfinite(embeddings).all():
            raise RuntimeError(f"class-name embedding形状或有限性错误：{tuple(embeddings.shape)}")
        norms = torch.linalg.vector_norm(embeddings.float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0):
            raise RuntimeError("class-name embedding没有逐类L2归一化。")
        embeddings = embeddings.detach().cpu()
        class_embedding_path = temporary / "class_name_embeds.pt"
        _atomic_torch_save(class_embedding_path, embeddings)
        (temporary / "class_names.json").write_text(
            json.dumps(
                {"xlsa": list(class_names), "display": display_names, "prompts": prompts},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        hardlink_checks: dict[str, bool] = {}
        for filename in REUSED_OUTPUTS:
            source = parent_manifest_path.parent / filename
            destination = temporary / filename
            os.link(source, destination)
            hardlink_checks[filename] = os.path.samefile(source, destination)
            if not hardlink_checks[filename]:
                raise RuntimeError(f"父缓存未按hardlink复用：{filename}")

        output_sha = {filename: sha256_file(temporary / filename) for filename in ALL_OUTPUTS}
        for filename in REUSED_OUTPUTS:
            if output_sha[filename] != parent["outputs_sha256"][filename]:
                raise RuntimeError(f"hardlink后parent缓存SHA发生变化：{filename}")
        asset_id = derived_asset_id(
            config,
            output_sha["class_name_embeds.pt"],
            output_sha["class_names.json"],
            encoder_identity,
        )
        output_dir = output_root / asset_id
        if output_dir.exists():
            raise FileExistsError(f"canonical class-name资产目录已存在：{output_dir}")

        if sha256_file(Path(config_path)) != config_sha:
            raise RuntimeError("派生配置在运行期间发生变化。")
        if sha256_file(parent_manifest_path) != config["parent_manifest_sha256"]:
            raise RuntimeError("parent manifest在运行期间发生变化。")
        if sha256_file(role_text_path) != config["role_texts_sha256"]:
            raise RuntimeError("text-v2原文在运行期间发生变化。")
        if sha256_file(checkpoint) != config["clip_checkpoint_sha256"]:
            raise RuntimeError("CLIP checkpoint在运行期间发生变化。")
        if _text_encoder is None:
            final_encoder_identity = _production_encoder_identity(parent, int(batch_size))
            if final_encoder_identity != encoder_identity:
                raise RuntimeError("CLIP text encoder identity在运行期间发生变化。")

        manifest = copy.deepcopy(parent)
        manifest.update(
            {
                "asset_id": asset_id,
                "source_config_sha256": config_sha,
                "outputs_sha256": output_sha,
                "class_name_text_version": CLASS_NAME_VERSION,
                "class_name_prompt_source": "exact shared prefix of frozen text-v2 role sentences",
                "class_name_prompt_count": len(prompts),
                "text_encoder_identity": encoder_identity,
                "text_encoder_identity_sha256": _canonical_sha256(encoder_identity),
                "derived_from_asset_id": parent["asset_id"],
                "derived_from_manifest_sha256": config["parent_manifest_sha256"],
                "derivation_kind": "canonical_class_name_reencode_role_and_visual_cache_hardlinked",
                "hardlink_verified": hardlink_checks,
            }
        )
        (temporary / "asset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
