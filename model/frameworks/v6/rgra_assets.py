"""Asset and warm-start loading for IDEA-205 RGRA.

This module is intentionally only about identities and tensors.  Training code
should call ``load_rgra_train_assets`` so official test images stay unopened
until evaluation code calls ``load_rgra_eval_assets``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from tools.runtime import sha256_file


EMBED_DIM = 768
ROLE_COUNT = 8
RGRA_GROUP_COUNT = 3

DEFAULT_ASSET_ID = "CUB_openai_vitl14_336_dynamic_v3_v1"
DEFAULT_ASSET_MANIFEST_SHA256 = (
    "3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6"
)
DEFAULT_RELATION_ASSET_ID = "CUB_pclr_relations_453d684b5080f477"
DEFAULT_RELATION_MANIFEST_SHA256 = (
    "0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4"
)
DEFAULT_V5_R2_CHECKPOINT_SHA256 = (
    "16b5071f21a3217e58a72315029c28b8cfd97b68f812641bd0145d3f5e0702ab"
)
DEFAULT_V5_R2_CONFIG_SHA256 = (
    "0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49"
)
DEFAULT_V5_R2_CODE_COMMIT = "b0a756dd624e883eb50d19a2455ba06bdc73f118"

ROLE_NAMES = (
    "beak",
    "head_features",
    "body_plumage",
    "wings",
    "tail",
    "legs",
    "overall_appearance",
    "unique_discriminative_features",
)

PATCH_OUTPUTS = {
    "train": "train_coarse_patch_features.npy",
    "test_seen": "test_seen_coarse_patch_features.npy",
    "test_unseen": "test_unseen_coarse_patch_features.npy",
}


@dataclass(frozen=True)
class RGRADatasetSpec:
    dataset: str
    class_count: int
    seen_count: int
    train_count: int
    test_seen_count: int
    test_unseen_count: int
    edge_count: int = 438


CUB_SPEC = RGRADatasetSpec(
    dataset="CUB",
    class_count=200,
    seen_count=150,
    train_count=7057,
    test_seen_count=1764,
    test_unseen_count=2967,
)


@dataclass(frozen=True)
class RGRAV5Initialization:
    p_v5: torch.Tensor
    scale: torch.Tensor
    reader_state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    checkpoint_code_commit: str
    checkpoint_config_sha256: str


@dataclass(frozen=True)
class RGRARelationAssets:
    relation_sentence_embeds: torch.Tensor
    relation_directions: torch.Tensor
    edge_index: torch.Tensor
    identity: dict[str, Any]


@dataclass(frozen=True)
class RGRATrainAssets:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    train_coarse_patches: np.memmap
    role_sentence_embeds: torch.Tensor
    p_v5: torch.Tensor
    scale: torch.Tensor
    reader_state_dict: dict[str, torch.Tensor]
    relation_directions: torch.Tensor
    edge_index: torch.Tensor
    seen_classes: torch.Tensor
    unseen_classes: torch.Tensor
    identity: dict[str, Any]


@dataclass(frozen=True)
class RGRAEvalAssets:
    test_seen_features: torch.Tensor
    test_seen_labels: torch.Tensor
    test_seen_coarse_patches: np.memmap
    test_unseen_features: torch.Tensor
    test_unseen_labels: torch.Tensor
    test_unseen_coarse_patches: np.memmap
    role_sentence_embeds: torch.Tensor
    relation_directions: torch.Tensor | None
    edge_index: torch.Tensor | None
    seen_classes: torch.Tensor
    unseen_classes: torch.Tensor
    identity: dict[str, Any]


def _config_value(config: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return default


def _expected_sha(config: Mapping[str, Any], *names: str, default: str) -> str:
    value = _config_value(config, *names, default=default)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{names[0]} must be a SHA-256 hex string.")
    return value.lower()


def _path_from_config(config: Mapping[str, Any], *names: str) -> Path:
    value = _config_value(config, *names)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing path config value: {names[0]}.")
    return Path(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _verified_manifest(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{name} manifest is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{name} manifest SHA mismatch: {actual}")
    return _read_json(path)


def _output_path(manifest_path: Path, manifest: Mapping[str, Any], filename: str) -> Path:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or filename not in outputs:
        raise ValueError(f"Manifest does not bind required output: {filename}")
    return manifest_path.parent / filename


def _verify_output(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    filename: str,
) -> Path:
    path = _output_path(manifest_path, manifest, filename)
    expected = str(manifest["outputs_sha256"][filename]).lower()
    if not path.is_file():
        raise ValueError(f"Bound asset file is missing: {filename}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Bound asset file SHA mismatch: {filename}: {actual}")
    return path


def _load_torch_output(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    filename: str,
) -> torch.Tensor:
    path = _verify_output(manifest_path, manifest, filename)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Asset output is not a tensor: {filename}")
    return value


def _load_memmap_output(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    filename: str,
    expected_shape: tuple[int, ...],
) -> np.memmap:
    path = _verify_output(manifest_path, manifest, filename)
    value = np.load(path, mmap_mode="r")
    if not isinstance(value, np.memmap):
        raise ValueError(f"Patch output must be opened as a memmap: {filename}")
    if tuple(value.shape) != expected_shape or value.dtype != np.float16:
        raise ValueError(
            f"Patch output shape/dtype mismatch: {filename}: "
            f"{tuple(value.shape)} {value.dtype}"
        )
    return value


def _validate_visual_manifest(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    spec: RGRADatasetSpec,
) -> None:
    expected_asset_id = _config_value(config, "asset_id", default=DEFAULT_ASSET_ID)
    counts = {
        "train": spec.train_count,
        "test_seen": spec.test_seen_count,
        "test_unseen": spec.test_unseen_count,
    }
    scalar_counts_match = (
        int(manifest.get("class_count", -1)) == spec.class_count
        and int(manifest.get("seen_class_count", -1)) == spec.seen_count
        and int(manifest.get("train_count", -1)) == spec.train_count
        and int(manifest.get("test_seen_count", -1)) == spec.test_seen_count
        and int(manifest.get("test_unseen_count", -1)) == spec.test_unseen_count
    )
    manifest_counts = manifest.get("counts")
    nested_counts_match = manifest_counts == counts
    invalid = (
        manifest.get("schema_version") != "gzsl-paper.clip-assets.v1"
        or manifest.get("asset_id") != expected_asset_id
        or manifest.get("dataset") != spec.dataset
        or not (scalar_counts_match or nested_counts_match)
    )
    if invalid:
        raise ValueError("RGRA visual asset identity, dataset, or counts mismatch.")
    forbidden = {
        "attributes",
        "class_attributes",
        "part_labels",
        "parts",
        "boxes",
        "bounding_boxes",
        "expert_residuals",
    }
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or forbidden.intersection(outputs):
        raise ValueError("RGRA visual asset contains forbidden supervised fields.")
    roles = manifest.get("role_names")
    if isinstance(roles, list) and tuple(roles) != ROLE_NAMES:
        raise ValueError("RGRA role order is not the frozen 6+1+1 order.")


def _classes_from_manifest(
    manifest: Mapping[str, Any],
    train_labels: torch.Tensor | None,
    spec: RGRADatasetSpec,
    *,
    test_seen_labels: torch.Tensor | None = None,
    test_unseen_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "seen_classes" in manifest and "unseen_classes" in manifest:
        seen = torch.tensor(manifest["seen_classes"], dtype=torch.long).sort().values
        unseen = torch.tensor(manifest["unseen_classes"], dtype=torch.long).sort().values
    elif train_labels is not None:
        seen = torch.unique(train_labels.long(), sorted=True)
        all_classes = torch.arange(spec.class_count, dtype=torch.long)
        unseen = all_classes[~torch.isin(all_classes, seen)]
    elif test_seen_labels is not None and test_unseen_labels is not None:
        seen = torch.unique(test_seen_labels.long(), sorted=True)
        unseen = torch.unique(test_unseen_labels.long(), sorted=True)
    else:
        raise ValueError("RGRA class axes cannot be derived from manifest or labels.")
    all_classes = torch.cat((seen, unseen)).sort().values
    if (
        seen.ndim != 1
        or unseen.ndim != 1
        or seen.numel() != spec.seen_count
        or unseen.numel() != spec.class_count - spec.seen_count
        or seen.unique().numel() != seen.numel()
        or unseen.unique().numel() != unseen.numel()
        or bool(torch.isin(seen, unseen).any())
        or not torch.equal(all_classes, torch.arange(spec.class_count))
    ):
        raise ValueError("RGRA seen/unseen class axis mismatch.")
    if train_labels is not None and not torch.equal(
        torch.unique(train_labels.long(), sorted=True), seen
    ):
        raise ValueError("RGRA train labels do not match manifest seen classes.")
    return seen, unseen


def _validate_tensor(
    value: torch.Tensor,
    name: str,
    expected_shape: tuple[int, ...],
    *,
    integer: bool = False,
) -> torch.Tensor:
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} shape mismatch: {tuple(value.shape)}")
    if integer:
        if value.dtype not in (torch.int64, torch.long):
            raise ValueError(f"{name} must be int64 labels.")
        return value.long()
    value = value.detach().cpu().float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return value


def _validate_role_tensor(role_sentence_embeds: torch.Tensor, spec: RGRADatasetSpec) -> torch.Tensor:
    roles = _validate_tensor(
        role_sentence_embeds,
        "role_sentence_embeds",
        (spec.class_count, ROLE_COUNT, EMBED_DIM),
    )
    norms = torch.linalg.vector_norm(roles, dim=-1)
    if bool(norms.le(0.0).any()):
        raise ValueError("role_sentence_embeds contains zero rows.")
    return roles


def _validate_relation_assets(
    relation_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    visual_manifest_sha256: str,
    spec: RGRADatasetSpec,
) -> None:
    expected_asset_id = _config_value(
        config, "relation_asset_id", default=DEFAULT_RELATION_ASSET_ID
    )
    outputs = relation_manifest.get("outputs_sha256")
    required_outputs = {
        "relation_texts.json",
        "relation_sentence_embeds.pt",
        "edge_index.pt",
    }
    invalid = (
        relation_manifest.get("schema_version")
        != "gzsl-paper.pclr-relation-asset.v1"
        or relation_manifest.get("asset_id") != expected_asset_id
        or relation_manifest.get("dataset") != spec.dataset
        or int(relation_manifest.get("class_count", -1)) != spec.class_count
        or int(relation_manifest.get("seen_count", -1)) != spec.seen_count
        or int(relation_manifest.get("edge_count", -1)) != spec.edge_count
        or int(relation_manifest.get("direction_count", -1)) != 2 * spec.edge_count
        or int(relation_manifest.get("embedding_dimension", -1)) != EMBED_DIM
        or relation_manifest.get("graph_source")
        != "OpenAI_CLIP_class_name_template_union_top3"
        or relation_manifest.get("template") != "a photo of a {class}"
        or int(relation_manifest.get("seen_induced_min_degree", 0)) < 1
        or relation_manifest.get("parent_manifest_sha256") != visual_manifest_sha256
        or relation_manifest.get("human_annotations_used") is not False
        or relation_manifest.get("llm_world_knowledge_used") is not True
        or not isinstance(relation_manifest.get("relation_encoder_matches_parent"), bool)
        or not isinstance(outputs, dict)
        or set(outputs) != required_outputs
    )
    if invalid:
        raise ValueError("RGRA relation asset identity, graph, or disclosure mismatch.")


def _load_relation_texts(
    relation_manifest_path: Path,
    relation_manifest: Mapping[str, Any],
    edge_index: torch.Tensor,
) -> None:
    relation_texts_path = _verify_output(
        relation_manifest_path,
        relation_manifest,
        "relation_texts.json",
    )
    relation_texts = _read_json(relation_texts_path)
    rows = relation_texts.get("rows")
    if (
        relation_texts.get("schema_version")
        != "gzsl-paper.pclr-relation-texts.v1"
        or relation_texts.get("human_annotations_used") is not False
        or relation_texts.get("llm_world_knowledge_used") is not True
        or not isinstance(rows, list)
        or len(rows) != edge_index.size(0)
    ):
        raise ValueError("RGRA relation texts schema or disclosure mismatch.")
    for edge_id, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != {"edge_id", "a_id", "b_id", "a_over_b", "b_over_a"}
            or row["edge_id"] != edge_id
            or [row["a_id"], row["b_id"]] != edge_index[edge_id].tolist()
        ):
            raise ValueError(f"RGRA relation text row does not match edge {edge_id}.")
        a_prefix = str(row["a_over_b"]).split(":", 1)[0]
        b_prefix = str(row["b_over_a"]).split(":", 1)[0]
        if " rather than " not in a_prefix:
            raise ValueError(f"RGRA relation text row lacks direction prefix: {edge_id}.")
        a_name, b_name = a_prefix.split(" rather than ", 1)
        if b_prefix != f"{b_name} rather than {a_name}":
            raise ValueError(f"RGRA relation text row is not reciprocal: {edge_id}.")


def load_rgra_relation_assets(
    config: Mapping[str, Any],
    *,
    visual_manifest_sha256: str,
    spec: RGRADatasetSpec = CUB_SPEC,
) -> RGRARelationAssets:
    relation_manifest_path = _path_from_config(config, "relation_asset_manifest")
    expected_sha = _expected_sha(
        config,
        "relation_asset_manifest_sha256",
        default=DEFAULT_RELATION_MANIFEST_SHA256,
    )
    relation_manifest = _verified_manifest(
        relation_manifest_path,
        expected_sha,
        "RGRA relation asset",
    )
    _validate_relation_assets(relation_manifest, config, visual_manifest_sha256, spec)

    relation_embeds = _load_torch_output(
        relation_manifest_path, relation_manifest, "relation_sentence_embeds.pt"
    )
    edge_index = _load_torch_output(
        relation_manifest_path, relation_manifest, "edge_index.pt"
    )
    relation_embeds = _validate_tensor(
        relation_embeds,
        "relation_sentence_embeds",
        (spec.edge_count, 2, EMBED_DIM),
    )
    edge_index = _validate_tensor(
        edge_index,
        "edge_index",
        (spec.edge_count, 2),
        integer=True,
    )
    if (
        not bool((edge_index[:, 0] < edge_index[:, 1]).all())
        or int(edge_index.min()) < 0
        or int(edge_index.max()) >= spec.class_count
        or torch.unique(edge_index, dim=0).size(0) != spec.edge_count
    ):
        raise ValueError("RGRA edge_index endpoints or uniqueness mismatch.")
    norms = torch.linalg.vector_norm(relation_embeds, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0):
        raise ValueError("RGRA relation embeddings must be per-direction L2-normalized.")
    _load_relation_texts(relation_manifest_path, relation_manifest, edge_index)

    directions = F.normalize(
        relation_embeds[:, 0, :] - relation_embeds[:, 1, :],
        dim=-1,
    )
    if not torch.isfinite(directions).all():
        raise ValueError("RGRA relation directions contain NaN/Inf.")
    return RGRARelationAssets(
        relation_sentence_embeds=relation_embeds,
        relation_directions=directions,
        edge_index=edge_index,
        identity={
            "asset_id": relation_manifest["asset_id"],
            "manifest_sha256": expected_sha,
            "outputs_sha256": dict(relation_manifest["outputs_sha256"]),
            "relation_encoder_matches_parent": relation_manifest[
                "relation_encoder_matches_parent"
            ],
        },
    )


def _reader_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    required = {
        "reader_in.weight": (64, EMBED_DIM),
        "reader_in.bias": (64,),
        "reader_out.weight": (EMBED_DIM, 64),
        "reader_out.bias": (EMBED_DIM,),
    }
    reader = {}
    for name, shape in required.items():
        value = state_dict.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"V5 R2 checkpoint missing reader tensor: {name}")
        tensor = value.detach().cpu().float()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"V5 R2 reader tensor contains NaN/Inf: {name}")
        reader[name] = tensor
    return reader


def _validate_initial_values(
    p_v5: torch.Tensor,
    scale: torch.Tensor | float,
    spec: RGRADatasetSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = _validate_tensor(p_v5, "p_v5", (spec.class_count, EMBED_DIM))
    prototypes = F.normalize(prototypes, dim=-1)
    scale_tensor = torch.as_tensor(scale).detach().cpu().float()
    if scale_tensor.numel() != 1 or not torch.isfinite(scale_tensor).all():
        raise ValueError("V5 scale must be a finite scalar.")
    if float(scale_tensor.reshape(())) <= 0.0:
        raise ValueError("V5 scale must be positive.")
    return prototypes, scale_tensor.reshape(())


def _checkpoint_payload(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = _path_from_config(
        config,
        "v5_r2_checkpoint",
        "v5_checkpoint",
        "source_checkpoint",
    )
    expected_sha = _expected_sha(
        config,
        "v5_r2_checkpoint_sha256",
        "v5_checkpoint_sha256",
        "source_checkpoint_sha256",
        default=DEFAULT_V5_R2_CHECKPOINT_SHA256,
    )
    if not path.is_file():
        raise ValueError(f"V5 R2 checkpoint is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f"V5 R2 checkpoint SHA mismatch: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("V5 R2 checkpoint payload must be a dict.")
    return payload, actual


def _materialize_v5_r2_initialization(
    state_dict: Mapping[str, torch.Tensor],
    role_sentence_embeds: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    relation_sentence_embeds: torch.Tensor,
    edge_index: torch.Tensor,
    spec: RGRADatasetSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    from model.frameworks.v2 import train as h1
    from model.frameworks.v4.model import PaperV2ThreeModuleModel
    from model.frameworks.v4.pclr import PCLRModel

    labels = train_labels.long()
    seen = torch.unique(labels, sorted=True)
    centroids = h1.visual_centroids(train_features, labels, seen)
    parent = PaperV2ThreeModuleModel(
        role_sentence_embeds,
        seen,
        centroids,
        tg_vpr_mode="full",
        transport_mode="off",
        ccgr_mode="off",
        dropout=0.5,
        inner_ratio=0.35,
        outer_ratio=0.65,
        temperature=0.07,
    )
    model = PCLRModel(
        parent,
        seen,
        relation_sentence_embeds,
        edge_index,
        class_count=spec.class_count,
        hidden_dim=16,
        max_transport_step=1.5,
        grid_points=33,
        reader_hidden_dim=64,
        reader_seed=18601,
        temperature=0.07,
        ridge_lambda=0.03,
        potential_cap=0.5,
        max_beta=0.25,
        initial_beta=0.05,
        candidate_top_k=15,
        correction_scale=2.38,
        seen_logit_gamma=0.525,
    )
    missing, unexpected = model.load_state_dict(dict(state_dict), strict=True)
    if missing or unexpected:
        raise ValueError(
            f"V5 R2 checkpoint state mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    with torch.no_grad():
        return model.prototypes().detach().cpu(), model.scale().detach().cpu()


def load_v5_r2_initialization(
    config: Mapping[str, Any],
    *,
    role_sentence_embeds: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    relation_sentence_embeds: torch.Tensor,
    edge_index: torch.Tensor,
    spec: RGRADatasetSpec = CUB_SPEC,
) -> RGRAV5Initialization:
    payload, checkpoint_sha = _checkpoint_payload(config)
    expected_code = _config_value(
        config,
        "v5_r2_code_commit",
        "v5_source_code_commit",
        "source_code_commit",
        default=DEFAULT_V5_R2_CODE_COMMIT,
    )
    expected_config_sha = _expected_sha(
        config,
        "v5_r2_config_sha256",
        "v5_source_config_sha256",
        "source_config_sha256",
        default=DEFAULT_V5_R2_CONFIG_SHA256,
    )
    if payload.get("code_commit") != expected_code:
        raise ValueError("V5 R2 checkpoint code commit mismatch.")
    if payload.get("config_sha256") != expected_config_sha:
        raise ValueError("V5 R2 checkpoint config SHA mismatch.")

    explicit = payload.get("rgra_initialization")
    if isinstance(explicit, dict):
        p_v5, scale = _validate_initial_values(explicit["p_v5"], explicit["scale"], spec)
        reader_source = explicit.get("reader_state_dict", payload.get("reader_state_dict", {}))
        reader = _reader_from_state_dict(reader_source)
    else:
        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("V5 R2 checkpoint lacks model_state_dict.")
        reader = _reader_from_state_dict(state_dict)
        p_v5_raw, scale_raw = _materialize_v5_r2_initialization(
            state_dict,
            role_sentence_embeds,
            train_features,
            train_labels,
            relation_sentence_embeds,
            edge_index,
            spec,
        )
        p_v5, scale = _validate_initial_values(p_v5_raw, scale_raw, spec)
    return RGRAV5Initialization(
        p_v5=p_v5,
        scale=scale,
        reader_state_dict=reader,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_code_commit=str(expected_code),
        checkpoint_config_sha256=expected_config_sha,
    )


def _load_visual_manifest(
    config: Mapping[str, Any],
    spec: RGRADatasetSpec,
) -> tuple[Path, dict[str, Any], str]:
    manifest_path = _path_from_config(config, "asset_manifest", "visual_asset_manifest")
    expected_sha = _expected_sha(
        config,
        "asset_manifest_sha256",
        "visual_asset_manifest_sha256",
        default=DEFAULT_ASSET_MANIFEST_SHA256,
    )
    manifest = _verified_manifest(manifest_path, expected_sha, "RGRA visual asset")
    _validate_visual_manifest(manifest, config, spec)
    return manifest_path, manifest, expected_sha


def load_rgra_train_assets(
    config: Mapping[str, Any],
    *,
    spec: RGRADatasetSpec = CUB_SPEC,
) -> RGRATrainAssets:
    manifest_path, manifest, manifest_sha = _load_visual_manifest(config, spec)
    train_features = _validate_tensor(
        _load_torch_output(manifest_path, manifest, "train_features.pt"),
        "train_features",
        (spec.train_count, EMBED_DIM),
    )
    train_labels = _validate_tensor(
        _load_torch_output(manifest_path, manifest, "train_labels.pt"),
        "train_labels",
        (spec.train_count,),
        integer=True,
    )
    role_sentence_embeds = _validate_role_tensor(
        _load_torch_output(manifest_path, manifest, "role_sentence_embeds.pt"),
        spec,
    )
    train_patches = _load_memmap_output(
        manifest_path,
        manifest,
        PATCH_OUTPUTS["train"],
        (spec.train_count, 36, EMBED_DIM),
    )
    seen, unseen = _classes_from_manifest(manifest, train_labels, spec)
    relations = load_rgra_relation_assets(
        config,
        visual_manifest_sha256=manifest_sha,
        spec=spec,
    )
    init = load_v5_r2_initialization(
        config,
        role_sentence_embeds=role_sentence_embeds,
        train_features=train_features,
        train_labels=train_labels,
        relation_sentence_embeds=relations.relation_sentence_embeds,
        edge_index=relations.edge_index,
        spec=spec,
    )
    return RGRATrainAssets(
        train_features=train_features,
        train_labels=train_labels,
        train_coarse_patches=train_patches,
        role_sentence_embeds=role_sentence_embeds,
        p_v5=init.p_v5,
        scale=init.scale,
        reader_state_dict=init.reader_state_dict,
        relation_directions=relations.relation_directions,
        edge_index=relations.edge_index,
        seen_classes=seen,
        unseen_classes=unseen,
        identity={
            "asset_id": manifest["asset_id"],
            "asset_manifest_sha256": manifest_sha,
            "relation_asset": relations.identity,
            "v5_r2_checkpoint_sha256": init.checkpoint_sha256,
            "v5_r2_code_commit": init.checkpoint_code_commit,
            "v5_r2_config_sha256": init.checkpoint_config_sha256,
            "patch_outputs": {PATCH_OUTPUTS["train"]: manifest["outputs_sha256"][PATCH_OUTPUTS["train"]]},
        },
    )


def load_rgra_eval_assets(
    config: Mapping[str, Any],
    *,
    spec: RGRADatasetSpec = CUB_SPEC,
    include_relation_assets: bool = True,
) -> RGRAEvalAssets:
    manifest_path, manifest, manifest_sha = _load_visual_manifest(config, spec)
    role_sentence_embeds = _validate_role_tensor(
        _load_torch_output(manifest_path, manifest, "role_sentence_embeds.pt"),
        spec,
    )
    test_seen_features = _validate_tensor(
        _load_torch_output(manifest_path, manifest, "test_seen_features.pt"),
        "test_seen_features",
        (spec.test_seen_count, EMBED_DIM),
    )
    test_seen_labels = _validate_tensor(
        _load_torch_output(manifest_path, manifest, "test_seen_labels.pt"),
        "test_seen_labels",
        (spec.test_seen_count,),
        integer=True,
    )
    test_unseen_features = _validate_tensor(
        _load_torch_output(manifest_path, manifest, "test_unseen_features.pt"),
        "test_unseen_features",
        (spec.test_unseen_count, EMBED_DIM),
    )
    test_unseen_labels = _validate_tensor(
        _load_torch_output(manifest_path, manifest, "test_unseen_labels.pt"),
        "test_unseen_labels",
        (spec.test_unseen_count,),
        integer=True,
    )
    seen, unseen = _classes_from_manifest(
        manifest,
        None,
        spec,
        test_seen_labels=test_seen_labels,
        test_unseen_labels=test_unseen_labels,
    )
    if not torch.equal(torch.unique(test_seen_labels, sorted=True), seen):
        raise ValueError("RGRA test_seen labels do not match seen classes.")
    if not torch.equal(torch.unique(test_unseen_labels, sorted=True), unseen):
        raise ValueError("RGRA test_unseen labels do not match unseen classes.")
    test_seen_patches = _load_memmap_output(
        manifest_path,
        manifest,
        PATCH_OUTPUTS["test_seen"],
        (spec.test_seen_count, 36, EMBED_DIM),
    )
    test_unseen_patches = _load_memmap_output(
        manifest_path,
        manifest,
        PATCH_OUTPUTS["test_unseen"],
        (spec.test_unseen_count, 36, EMBED_DIM),
    )
    relations = (
        load_rgra_relation_assets(
            config,
            visual_manifest_sha256=manifest_sha,
            spec=spec,
        )
        if include_relation_assets
        else None
    )
    return RGRAEvalAssets(
        test_seen_features=test_seen_features,
        test_seen_labels=test_seen_labels,
        test_seen_coarse_patches=test_seen_patches,
        test_unseen_features=test_unseen_features,
        test_unseen_labels=test_unseen_labels,
        test_unseen_coarse_patches=test_unseen_patches,
        role_sentence_embeds=role_sentence_embeds,
        relation_directions=(None if relations is None else relations.relation_directions),
        edge_index=(None if relations is None else relations.edge_index),
        seen_classes=seen,
        unseen_classes=unseen,
        identity={
            "asset_id": manifest["asset_id"],
            "asset_manifest_sha256": manifest_sha,
            "relation_asset": None if relations is None else relations.identity,
            "graph_free_eval_assets": not include_relation_assets,
            "patch_outputs": {
                PATCH_OUTPUTS["test_seen"]: manifest["outputs_sha256"][
                    PATCH_OUTPUTS["test_seen"]
                ],
                PATCH_OUTPUTS["test_unseen"]: manifest["outputs_sha256"][
                    PATCH_OUTPUTS["test_unseen"]
                ],
            },
        },
    )


def validate_finite_scalar(value: torch.Tensor, name: str) -> float:
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.numel() != 1 or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite scalar.")
    scalar = float(tensor.reshape(()))
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar
