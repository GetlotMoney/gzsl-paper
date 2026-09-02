"""Build frozen AWA2/SUN V7 directional-relation text assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import yaml

from tools.derive_paper_clip_text_asset import _production_text_encoder
from tools.prepare_paper_clip_assets import (
    MODEL_NAME,
    OFFICIAL_CHECKPOINT_SHA256,
    _atomic_torch_save,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v7-relation-asset-build.v1"
REQUEST_SCHEMA = "gzsl-paper.pclr-graph-request.v2"
SHARD_SCHEMA = "gzsl-paper.pclr-relations-shard.v2"
TEXT_SCHEMA = "gzsl-paper.v7-relation-texts.v1"
ASSET_SCHEMA = "gzsl-paper.v7-relation-asset.v1"
CONFIG_KEYS = {
    "schema_version",
    "dataset",
    "request",
    "request_sha256",
    "shards",
    "clip_checkpoint",
    "clip_checkpoint_sha256",
    "clip_python_source_sha256",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("V7关系资产配置字段错误。")
    if config["schema_version"] != SCHEMA or config["dataset"] not in {"AWA2", "SUN"}:
        raise ValueError("V7关系资产schema或dataset错误。")
    for key in ("request", "clip_checkpoint"):
        value = Path(config[key])
        if not value.is_absolute() or not value.is_file():
            raise ValueError(f"V7关系资产{key}必须是存在的绝对文件。")
    if not isinstance(config["shards"], list) or not config["shards"]:
        raise ValueError("V7关系资产shards必须是非空列表。")
    for item in config["shards"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("V7关系资产shard字段错误。")
        shard = Path(item["path"])
        if not shard.is_absolute() or not shard.is_file():
            raise ValueError("V7关系资产shard必须是存在的绝对文件。")
    return config, sha256_file(path)


def _visible_policy(dataset: str, text: str) -> None:
    lowered = text.casefold()
    banned = (
        ("habitat", "lives in", "native to", "nocturnal", "diet", "prey", "hunts", "can fly")
        if dataset == "AWA2"
        else ("located in", "typically used", "historically", "culturally", "is a place for")
    )
    if any(value in lowered for value in banned):
        raise ValueError(f"{dataset}关系句包含非可见知识：{text}")


def _comparison_body(text: str) -> str:
    return text.split(":", 1)[1].strip().casefold()


def _validate_clip_context(clip_module, texts: list[str]) -> None:
    for direction_id, text in enumerate(texts):
        try:
            clip_module.tokenize([text], truncate=False)
        except RuntimeError as error:
            raise ValueError(f"V7关系句超过CLIP上下文：direction_id={direction_id}") from error


def _asset_manifest(
    metadata: dict,
    *,
    asset_id: str,
    config_sha: str,
    output_sha: dict[str, str],
    checkpoint_sha: str,
    clip_source_sha: str,
) -> dict:
    return {
        **metadata,
        "schema_version": ASSET_SCHEMA,
        "asset_id": asset_id,
        "embedding_dimension": 768,
        "clip_model": f"OpenAI {MODEL_NAME}",
        "clip_checkpoint_sha256": checkpoint_sha,
        "clip_python_source_sha256": clip_source_sha,
        "config_sha256": config_sha,
        "outputs_sha256": output_sha,
    }


def load_relation_rows(config: dict) -> tuple[dict, list[dict]]:
    request_path = Path(config["request"])
    if sha256_file(request_path) != config["request_sha256"]:
        raise ValueError("V7关系request SHA不匹配。")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    edges = request.get("edges")
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or request.get("dataset") != config["dataset"]
        or request.get("graph_source") != "OpenAI_CLIP_class_name_template_union_top3"
        or request.get("top_k") != 3
        or request.get("clip_checkpoint_sha256") != config["clip_checkpoint_sha256"]
        or request.get("clip_python_source_sha256") != config["clip_python_source_sha256"]
        or not isinstance(edges, list)
        or len(edges) != int(request.get("edge_count", -1))
        or int(request.get("direction_count", -1)) != 2 * len(edges)
        or int(request.get("class_count", -1)) < 2
        or not 0 < int(request.get("seen_count", -1)) < int(request.get("class_count", -1))
    ):
        raise ValueError("V7关系request身份错误。")
    edge_by_id = {int(row["edge_id"]): row for row in edges}
    if set(edge_by_id) != set(range(len(edges))):
        raise ValueError("V7关系request edge_id不连续。")
    endpoints = [(int(row["a_id"]), int(row["b_id"])) for row in edges]
    if (
        len(set(endpoints)) != len(endpoints)
        or any(a < 0 or a >= b or b >= int(request["class_count"]) for a, b in endpoints)
    ):
        raise ValueError("V7关系request边必须唯一、升序且位于类别轴内。")

    output = []
    expected_next = 0
    generator_records = []
    for item in config["shards"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"V7关系shard SHA不匹配：{path.name}")
        shard = json.loads(path.read_text(encoding="utf-8"))
        start, end = shard.get("range", [None, None])
        rows = shard.get("rows")
        if (
            shard.get("schema_version") != SHARD_SCHEMA
            or shard.get("dataset") != config["dataset"]
            or start != expected_next
            or not isinstance(end, int)
            or not isinstance(rows, list)
            or len(rows) != end - start + 1
        ):
            raise ValueError(f"V7关系shard范围或schema错误：{path.name}")
        generator_records.append({**shard.get("generator", {}), "sha256": item["sha256"]})
        for expected_id, row in zip(range(start, end + 1), rows, strict=True):
            if set(row) != {"edge_id", "a_id", "b_id", "a_over_b", "b_over_a"}:
                raise ValueError(f"V7关系row字段错误：{expected_id}")
            edge = edge_by_id[expected_id]
            if (
                row["edge_id"] != expected_id
                or row["a_id"] != edge["a_id"]
                or row["b_id"] != edge["b_id"]
            ):
                raise ValueError(f"V7关系row端点错误：{expected_id}")
            prefixes = (
                f"{edge['a_name']} rather than {edge['b_name']}:",
                f"{edge['b_name']} rather than {edge['a_name']}:",
            )
            texts = (str(row["a_over_b"]).strip(), str(row["b_over_a"]).strip())
            for text, prefix in zip(texts, prefixes, strict=True):
                if "\n" in text or len(text) <= len(prefix) + 8 or len(text) > 700:
                    raise ValueError(f"V7关系句长度或换行错误：{expected_id}")
                if not text.startswith(prefix):
                    raise ValueError(f"V7关系句前缀错误：{expected_id}")
                _visible_policy(config["dataset"], text)
            output.append(
                {
                    "edge_id": expected_id,
                    "a_id": edge["a_id"],
                    "b_id": edge["b_id"],
                    "a_over_b": texts[0],
                    "b_over_a": texts[1],
                }
            )
        expected_next = end + 1
    if expected_next != len(edges):
        raise ValueError("V7关系shards没有完整覆盖全部边。")
    generic_fragments = (
        "category-specific fixtures",
        "category-specific objects",
        "visible features associated with",
        "distinct scene",
        "structures, surfaces, objects, and spatial layout",
    )
    body_targets: dict[str, set[int]] = {}
    for row in output:
        bodies = (_comparison_body(row["a_over_b"]), _comparison_body(row["b_over_a"]))
        if bodies[0] == bodies[1]:
            raise ValueError(f"V7关系双方向正文相同：{row['edge_id']}")
        for body, target in zip(bodies, (row["a_id"], row["b_id"]), strict=True):
            if any(fragment in body for fragment in generic_fragments):
                raise ValueError(f"V7关系句使用泛化模板：{row['edge_id']}")
            body_targets.setdefault(body, set()).add(int(target))
    reused = [body for body, targets in body_targets.items() if len(targets) > 1]
    if reused:
        raise ValueError(f"V7关系正文被不同目标类别复用：{reused[0]}")
    metadata = {
        "schema_version": TEXT_SCHEMA,
        "dataset": config["dataset"],
        "class_count": request["class_count"],
        "seen_count": request["seen_count"],
        "edge_count": request["edge_count"],
        "direction_count": request["direction_count"],
        "request_sha256": config["request_sha256"],
        "graph_source": request["graph_source"],
        "top_k": request["top_k"],
        "generator_shards": generator_records,
        "human_annotations_used": False,
        "llm_world_knowledge_used": True,
        "visible_only_annotation": True,
    }
    return metadata, output


def run(config_path: Path, output_root: Path, device: str, batch_size: int) -> dict:
    config, config_sha = load_config(config_path)
    checkpoint = Path(config["clip_checkpoint"])
    if (
        sha256_file(checkpoint) != config["clip_checkpoint_sha256"]
        or config["clip_checkpoint_sha256"] != OFFICIAL_CHECKPOINT_SHA256
    ):
        raise ValueError("V7关系CLIP checkpoint身份错误。")
    import clip

    clip_source = Path(clip.__file__).resolve().parent / "clip.py"
    if sha256_file(clip_source) != config["clip_python_source_sha256"]:
        raise ValueError("V7关系OpenAI CLIP源码身份错误。")
    metadata, rows = load_relation_rows(config)
    flattened = [row[key] for row in rows for key in ("a_over_b", "b_over_a")]
    _validate_clip_context(clip, flattened)
    embeddings = _production_text_encoder(flattened, checkpoint, device, int(batch_size))
    expected = (int(metadata["direction_count"]), 768)
    if tuple(embeddings.shape) != expected or not torch.isfinite(embeddings).all():
        raise RuntimeError("V7关系embedding shape或有限性错误。")
    norms = embeddings.float().norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0):
        raise RuntimeError("V7关系embedding没有逐句L2归一化。")
    request = json.loads(Path(config["request"]).read_text(encoding="utf-8"))
    edges = torch.tensor(
        [[row["a_id"], row["b_id"]] for row in request["edges"]], dtype=torch.long
    )
    output_root = Path(output_root).resolve()
    if not output_root.is_absolute():
        raise ValueError("V7关系output_root必须是绝对路径。")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v7-relations.", dir=output_root))
    try:
        text_path = temporary / "relation_texts.json"
        text_payload = {**metadata, "rows": rows}
        text_path.write_text(
            json.dumps(text_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _atomic_torch_save(
            temporary / "relation_sentence_embeds.pt",
            embeddings.reshape(metadata["edge_count"], 2, 768).cpu(),
        )
        _atomic_torch_save(temporary / "edge_index.pt", edges)
        output_sha = {
            name: sha256_file(temporary / name)
            for name in ("relation_texts.json", "relation_sentence_embeds.pt", "edge_index.pt")
        }
        asset_id = _canonical_sha256(
            {
                "asset_schema": ASSET_SCHEMA,
                "config": config_sha,
                "metadata": metadata,
                "outputs": output_sha,
                "model": MODEL_NAME,
            }
        )[:16]
        output_dir = output_root / f"{config['dataset']}_v7_relations_{asset_id}"
        if output_dir.exists():
            raise FileExistsError(f"V7关系资产已存在：{output_dir}")
        manifest = _asset_manifest(
            metadata,
            asset_id=asset_id,
            config_sha=config_sha,
            output_sha=output_sha,
            checkpoint_sha=config["clip_checkpoint_sha256"],
            clip_source_sha=config["clip_python_source_sha256"],
        )
        (temporary / "asset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except Exception:
        for child in temporary.glob("*"):
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise
    result = {**manifest, "output_dir": str(output_dir), "asset_manifest_sha256": sha256_file(output_dir / "asset_manifest.json")}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run(args.config, args.output_root, args.device, args.batch_size)


if __name__ == "__main__":
    main()
