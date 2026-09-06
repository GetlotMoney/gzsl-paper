"""Build the frozen PCLR directional-relation text asset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
import shutil
import string
import tempfile
from pathlib import Path
from typing import Callable

import torch
import yaml

from tools.derive_paper_clip_text_asset import CLIP_SOURCE_FILES, _production_text_encoder
from tools.prepare_paper_clip_assets import (
    MODEL_NAME,
    OFFICIAL_CHECKPOINT_SHA256,
    _atomic_torch_save,
)
from tools.runtime import sha256_file


CONFIG_SCHEMA = "gzsl-paper.pclr-relation-asset-config.v1"
REQUEST_SCHEMA = "gzsl-paper.pclr-relation-request.v1"
SHARD_SCHEMA = "gzsl-paper.pclr-relations-shard.v1"
TEXT_SCHEMA = "gzsl-paper.pclr-relation-texts.v1"
ASSET_SCHEMA = "gzsl-paper.pclr-relation-asset.v1"
ENCODER_IDENTITY_SCHEMA = "gzsl-paper.clip-text-encoder.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RANGES = ((0, 145), (146, 291), (292, 437))
EXPECTED_EDGE_COUNT = 438
EXPECTED_CLASS_COUNT = 200
EXPECTED_SEEN_COUNT = 150
EXPECTED_TEMPLATE = "a photo of a {class}"
CONFIG_KEYS = {
    "schema_version",
    "dataset",
    "request",
    "request_sha256",
    "shards",
    "parent_manifest",
    "parent_manifest_sha256",
    "clip_checkpoint",
    "clip_checkpoint_sha256",
}
PROHIBITED_CONTENT = re.compile(
    r"\b(?:habitat|inhabits?|lives?\s+in|found\s+in|native\s+to|endemic|"
    r"migrat(?:e|es|ing|ion|ory)|behavio(?:u)?r|feeds?\s+on|diet|"
    r"geograph(?:y|ic|ical))\b",
    flags=re.IGNORECASE,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(c not in string.hexdigits for c in normalized):
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
    actual = set(payload) if isinstance(payload, dict) else set()
    if not isinstance(payload, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"PCLR资产配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if payload["schema_version"] != CONFIG_SCHEMA or payload["dataset"] != "CUB":
        raise ValueError("PCLR首轮资产只允许固定CUB schema。")
    for key in (
        "request_sha256",
        "parent_manifest_sha256",
        "clip_checkpoint_sha256",
    ):
        payload[key] = _validate_sha256(payload[key], key)
    for key in ("request", "parent_manifest", "clip_checkpoint"):
        payload[key] = str(_absolute_file(payload[key], key))
    shards = payload["shards"]
    if not isinstance(shards, list) or len(shards) != len(EXPECTED_RANGES):
        raise ValueError("PCLR资产固定要求三个关系文本分片。")
    normalized_shards = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != {"path", "sha256"}:
            raise ValueError(f"PCLR shard{index}配置字段错误。")
        normalized_shards.append(
            {
                "path": str(_absolute_file(shard["path"], f"shard{index}.path")),
                "sha256": _validate_sha256(shard["sha256"], f"shard{index}.sha256"),
            }
        )
    payload["shards"] = normalized_shards
    return payload, sha256_file(path)


def _validate_one_sentence(text: str, name: str) -> str:
    normalized = " ".join(text.split())
    if not normalized or normalized[-1] not in ".!?":
        raise ValueError(f"{name}必须是以句号结束的单句。")
    if any(mark in normalized[:-1] for mark in ".!?"):
        raise ValueError(f"{name}包含多个英文句。")
    if PROHIBITED_CONTENT.search(normalized):
        raise ValueError(f"{name}包含非可见形态内容。")
    return normalized


def load_relation_texts(config: dict) -> tuple[dict, list[dict]]:
    request_path = Path(config["request"])
    if sha256_file(request_path) != config["request_sha256"]:
        raise ValueError("PCLR request SHA不匹配。")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    edges = request.get("edges")
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or request.get("class_count") != EXPECTED_CLASS_COUNT
        or request.get("seen_count") != EXPECTED_SEEN_COUNT
        or request.get("edge_count") != EXPECTED_EDGE_COUNT
        or request.get("graph_source") != "OpenAI_CLIP_class_name_template_union_top3"
        or request.get("template") != EXPECTED_TEMPLATE
        or request.get("seen_induced_min_degree", 0) < 1
        or request.get("clip_checkpoint_sha256") != config["clip_checkpoint_sha256"]
        or _validate_sha256(
            request.get("clip_python_source_sha256"),
            "request.clip_python_source_sha256",
        )
        != request.get("clip_python_source_sha256")
        or not isinstance(edges, list)
        or len(edges) != EXPECTED_EDGE_COUNT
    ):
        raise ValueError("PCLR request图、split或CLIP身份错误。")
    edge_by_id = {}
    pairs = set()
    for expected_id, edge in enumerate(edges):
        if not isinstance(edge, dict) or edge.get("edge_id") != expected_id:
            raise ValueError("PCLR request edge_id必须从0连续排列。")
        a_id, b_id = edge.get("a_id"), edge.get("b_id")
        if (
            not isinstance(a_id, int)
            or not isinstance(b_id, int)
            or not 0 <= a_id < b_id < EXPECTED_CLASS_COUNT
            or (a_id, b_id) in pairs
        ):
            raise ValueError(f"PCLR request edge {expected_id}端点或唯一性错误。")
        if not isinstance(edge.get("a_name"), str) or not isinstance(edge.get("b_name"), str):
            raise ValueError(f"PCLR request edge {expected_id}类名错误。")
        pairs.add((a_id, b_id))
        edge_by_id[expected_id] = edge

    rows = []
    generators = []
    for index, (shard_spec, expected_range) in enumerate(
        zip(config["shards"], EXPECTED_RANGES, strict=True)
    ):
        shard_path = Path(shard_spec["path"])
        if sha256_file(shard_path) != shard_spec["sha256"]:
            raise ValueError(f"PCLR shard{index} SHA不匹配。")
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        generator = shard.get("generator")
        shard_rows = shard.get("rows")
        if (
            shard.get("schema_version") != SHARD_SCHEMA
            or shard.get("range") != list(expected_range)
            or not isinstance(generator, dict)
            or generator
            != {
                "provider": "Codex sub-agent",
                "task": f"/root/pclr_relations_{index}",
                "generated_at": "2026-08-31",
            }
            or not isinstance(shard_rows, list)
            or len(shard_rows) != expected_range[1] - expected_range[0] + 1
        ):
            raise ValueError(f"PCLR shard{index} schema、生成身份或范围错误。")
        generators.append({**generator, "sha256": shard_spec["sha256"]})
        for expected_id, row in zip(
            range(expected_range[0], expected_range[1] + 1), shard_rows, strict=True
        ):
            if (
                not isinstance(row, dict)
                or set(row) != {"edge_id", "a_id", "b_id", "a_over_b", "b_over_a"}
                or row.get("edge_id") != expected_id
            ):
                raise ValueError(f"PCLR shard{index} row {expected_id}字段或顺序错误。")
            edge = edge_by_id[expected_id]
            if row.get("a_id") != edge["a_id"] or row.get("b_id") != edge["b_id"]:
                raise ValueError(f"PCLR row {expected_id}端点与request不一致。")
            a_prefix = f"{edge['a_name']} rather than {edge['b_name']}:"
            b_prefix = f"{edge['b_name']} rather than {edge['a_name']}:"
            a_over_b = _validate_one_sentence(str(row.get("a_over_b", "")), f"row{expected_id}.a_over_b")
            b_over_a = _validate_one_sentence(str(row.get("b_over_a", "")), f"row{expected_id}.b_over_a")
            if not a_over_b.startswith(a_prefix) or not b_over_a.startswith(b_prefix):
                raise ValueError(f"PCLR row {expected_id}方向前缀错误。")
            rows.append(
                {
                    "edge_id": expected_id,
                    "a_id": edge["a_id"],
                    "b_id": edge["b_id"],
                    "a_over_b": a_over_b,
                    "b_over_a": b_over_a,
                }
            )
    if [row["edge_id"] for row in rows] != list(range(EXPECTED_EDGE_COUNT)):
        raise RuntimeError("PCLR三个shard没有完整覆盖固定关系图。")
    texts = {
        "schema_version": TEXT_SCHEMA,
        "dataset": "CUB",
        "request_sha256": config["request_sha256"],
        "class_names_sha256": request["class_names_sha256"],
        "graph_source": request["graph_source"],
        "template": request["template"],
        "generator_shards": generators,
        "human_annotations_used": False,
        "llm_world_knowledge_used": True,
        "rows": rows,
    }
    return texts, rows


def _load_parent(config: dict, request: dict) -> dict:
    manifest_path = Path(config["parent_manifest"])
    if sha256_file(manifest_path) != config["parent_manifest_sha256"]:
        raise ValueError("PCLR parent manifest SHA不匹配。")
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = parent.get("counts")
    outputs = parent.get("outputs_sha256")
    extensions = parent.get("v3_dynamic_extensions")
    if (
        parent.get("schema_version") != "gzsl-paper.clip-assets.v1"
        or parent.get("dataset") != "CUB"
        or parent.get("clip_model") != f"OpenAI {MODEL_NAME}"
        or counts != {"train": 7057, "test_seen": 1764, "test_unseen": 2967}
        or not isinstance(outputs, dict)
        or outputs.get("class_names.json") != request["class_names_sha256"]
        or parent.get("clip_checkpoint_sha256") != config["clip_checkpoint_sha256"]
        or not isinstance(extensions, dict)
        or extensions.get("human_annotations_used") is not False
    ):
        raise ValueError("PCLR parent数据、类别轴或CLIP身份错误。")
    class_names_path = manifest_path.parent / "class_names.json"
    if (
        not class_names_path.is_file()
        or sha256_file(class_names_path) != request["class_names_sha256"]
    ):
        raise ValueError("PCLR parent class_names.json缺失或SHA错误。")
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    display = class_names.get("display")
    if not isinstance(display, list) or len(display) != EXPECTED_CLASS_COUNT:
        raise ValueError("PCLR parent display类别轴错误。")
    request_names = {}
    for edge in request["edges"]:
        request_names[int(edge["a_id"])] = edge["a_name"]
        request_names[int(edge["b_id"])] = edge["b_name"]
    if request_names != {index: name for index, name in enumerate(display)}:
        raise ValueError("PCLR request类名映射与parent display类别轴不一致。")
    return parent


def _runtime_encoder_identity(
    parent: dict, batch_size: int, expected_clip_source_sha256: str
) -> dict:
    import clip

    package_root = Path(clip.__file__).resolve().parent
    source_hashes = {}
    for filename in CLIP_SOURCE_FILES:
        source = package_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"PCLR OpenAI CLIP源码缺失：{source}")
        source_hashes[filename] = sha256_file(source)
    if source_hashes["clip.py"] != expected_clip_source_sha256:
        raise ValueError("PCLR运行时OpenAI CLIP源码与关系图预注册身份不一致。")
    distribution = importlib.metadata.distribution("clip")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
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
        raise ValueError("PCLR output-root必须是绝对路径。")
    output_root = output_root.resolve()
    if output_root == REPOSITORY_ROOT or output_root.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("PCLR正式资产必须写到Git仓库外。")
    if int(batch_size) <= 0:
        raise ValueError("PCLR batch-size必须为正数。")
    config, config_sha = load_config(config_path)
    checkpoint = Path(config["clip_checkpoint"])
    if sha256_file(checkpoint) != config["clip_checkpoint_sha256"]:
        raise ValueError("PCLR CLIP checkpoint SHA不匹配。")
    if config["clip_checkpoint_sha256"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("PCLR只允许官方OpenAI ViT-L/14@336px checkpoint。")
    request = json.loads(Path(config["request"]).read_text(encoding="utf-8"))
    parent = _load_parent(config, request)
    text_payload, rows = load_relation_texts(config)
    flattened = [
        sentence
        for row in rows
        for sentence in (row["a_over_b"], row["b_over_a"])
    ]
    if _text_encoder is None:
        if _encoder_identity is not None:
            raise ValueError("生产encoder identity必须由运行时自动采集。")
        encoder_identity = _runtime_encoder_identity(
            parent,
            int(batch_size),
            request["clip_python_source_sha256"],
        )
        encoder = _production_text_encoder
    else:
        if (
            not isinstance(_encoder_identity, dict)
            or _encoder_identity.get("schema_version") != ENCODER_IDENTITY_SCHEMA
        ):
            raise ValueError("测试encoder必须提供版本化identity。")
        encoder_identity = copy.deepcopy(_encoder_identity)
        encoder = _text_encoder

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pclr-relation-asset.", dir=output_root))
    try:
        relation_text_path = temporary / "relation_texts.json"
        relation_text_path.write_text(
            json.dumps(text_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        embeddings = encoder(flattened, checkpoint, device_name, int(batch_size))
        expected = (EXPECTED_EDGE_COUNT * 2, 768)
        if tuple(embeddings.shape) != expected or not torch.isfinite(embeddings).all():
            raise RuntimeError(f"PCLR关系embedding shape或有限性错误：{tuple(embeddings.shape)}")
        norms = torch.linalg.vector_norm(embeddings.float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0):
            raise RuntimeError("PCLR关系embedding没有逐句L2归一化。")
        embedding_path = temporary / "relation_sentence_embeds.pt"
        edge_path = temporary / "edge_index.pt"
        _atomic_torch_save(
            embedding_path,
            embeddings.reshape(EXPECTED_EDGE_COUNT, 2, 768).detach().cpu(),
        )
        _atomic_torch_save(
            edge_path,
            torch.tensor([[row["a_id"], row["b_id"]] for row in rows], dtype=torch.long),
        )
        outputs_sha = {
            name: sha256_file(temporary / name)
            for name in (
                "relation_texts.json",
                "relation_sentence_embeds.pt",
                "edge_index.pt",
            )
        }
        identity = {
            "schema_version": ASSET_SCHEMA,
            "request_sha256": config["request_sha256"],
            "shard_sha256": [item["sha256"] for item in config["shards"]],
            "clip_checkpoint_sha256": config["clip_checkpoint_sha256"],
            "relation_clip_python_source_sha256": request[
                "clip_python_source_sha256"
            ],
            "parent_clip_python_source_sha256": parent.get(
                "clip_python_source_sha256"
            ),
            "relation_encoder_matches_parent": (
                request["clip_python_source_sha256"]
                == parent.get("clip_python_source_sha256")
            ),
            "outputs_sha256": outputs_sha,
            "encoder_identity_sha256": _canonical_sha256(encoder_identity),
        }
        asset_id = f"CUB_pclr_relations_{_canonical_sha256(identity)[:16]}"
        output_dir = output_root / asset_id
        if output_dir.exists():
            raise FileExistsError(f"PCLR关系资产目录已存在：{output_dir}")
        manifest = {
            "schema_version": ASSET_SCHEMA,
            "asset_id": asset_id,
            "dataset": "CUB",
            "class_count": EXPECTED_CLASS_COUNT,
            "seen_count": EXPECTED_SEEN_COUNT,
            "edge_count": EXPECTED_EDGE_COUNT,
            "direction_count": EXPECTED_EDGE_COUNT * 2,
            "embedding_dimension": 768,
            "model": MODEL_NAME,
            "source_config_sha256": config_sha,
            "request_sha256": config["request_sha256"],
            "shard_sha256": [item["sha256"] for item in config["shards"]],
            "parent_manifest_sha256": config["parent_manifest_sha256"],
            "clip_checkpoint_sha256": config["clip_checkpoint_sha256"],
            "relation_clip_python_source_sha256": request[
                "clip_python_source_sha256"
            ],
            "parent_clip_python_source_sha256": parent.get(
                "clip_python_source_sha256"
            ),
            "relation_encoder_matches_parent": (
                request["clip_python_source_sha256"]
                == parent.get("clip_python_source_sha256")
            ),
            "class_names_sha256": request["class_names_sha256"],
            "graph_source": request["graph_source"],
            "template": request["template"],
            "seen_induced_min_degree": request["seen_induced_min_degree"],
            "human_annotations_used": False,
            "llm_world_knowledge_used": True,
            "text_encoder_identity": encoder_identity,
            "text_encoder_identity_sha256": _canonical_sha256(encoder_identity),
            "outputs_sha256": outputs_sha,
        }
        (temporary / "asset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        **manifest,
        "asset_directory": str(output_dir),
        "asset_manifest_sha256": sha256_file(output_dir / "asset_manifest.json"),
    }


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
