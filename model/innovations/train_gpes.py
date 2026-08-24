from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.ebc import EpisodicBiasCalibration
from model.innovations.gpes import (
    AntisymmetricPairSelector,
    BiasFreeSemanticNeighborSelector,
    CenteredRoleGatedPairSelector,
    CrossSourceDisagreementSelector,
    GatedPairEvidenceSelector,
    LocalSemanticCompetitionResolver,
    NonlinearGatedPairSelector,
    NeighborhoodDegreePairSelector,
    NonlinearResidualPairSelector,
    PairDiscriminativeRoleSelector,
    ReciprocalSemanticNeighborPairSelector,
    RoleDisagreementScaleSelector,
    RoleVotePairSelector,
    RoleUncertaintyGatedSelector,
    RoleAwareGatedPairSelector,
    SemanticNeighborPairSelector,
    SemanticGatedPairSelector,
    StagedRoleDisagreementScaleSelector,
    StagedDiscriminativeRoleSelector,
    TextOnlyGatedPairSelector,
    TriadicCompetitionPairSelector,
    TrustRegionRoleDisagreementScaleSelector,
    semantic_neighbor_adjacency,
    reciprocal_neighbor_confidence,
    pair_role_distance_weights,
    top_discriminative_role_difference,
)
from model.innovations.lpsr import orthogonal_local_text_residuals
from model.innovations.sdcr import SentenceDropoutConservativeRouting
from model.innovations.tigr import taxonomic_suffix_group_ids
from model.innovations.train_agct import (
    _base_logits,
    derive_train_threshold,
    select_margin_threshold,
)
from model.innovations.train_ccpe import _precompute_scores
from model.innovations.train_chen_style import (
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.innovations.train_sebc import _load_main
from model.tg_vpr_h1 import train as h1
from tools.cub_data import load_cub_split
from tools.diagnose_sdcr_errors import load_class_names
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
    require_finite_gradients,
)
from tools.runtime import sha256_file

EVALUATION_PROTOCOL = "chen_shiming_code_aligned_test_selected_gzsl"
CONFIG_KEYS = {
    "schema_version", "experiment_id", "idea_id", "framework_id", "dataset",
    "evaluation_protocol", "test_used_for_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "feature_provenance_complete", "text_cache_provenance_complete",
    "base_model", "base_model_sha256", "sdrs_model", "sdrs_model_sha256",
    "sebc_model", "sebc_model_sha256", "casr_model", "casr_model_sha256",
    "sdcr_model", "sdcr_model_sha256", "parent_metrics_percent",
    "class_name_embeddings", "class_name_embeddings_sha256",
    "eight_sentence_embeddings", "eight_sentence_embeddings_sha256",
    "claude_embeddings", "claude_embeddings_sha256", "merge_embeddings",
    "merge_embeddings_sha256", "patch_inputs", "patch_sha256", "patch_top_k",
    "patch_chunk_size", "group_rule", "threshold_source", "threshold_quantile",
    "margin_temperature", "max_delta", "device", "random_seed", "batch_size",
    "epochs", "niters", "report_interval", "optimizer", "learning_rate",
    "weight_decay", "inputs", "expected_sha256", "class_order_sha256",
}

SOFT_PAIR_SCHEMAS = frozenset({
    "gzsl-paper.gwps.v1", "gzsl-paper.bgwps.v1", "gzsl-paper.mbgwps.v1",
    "gzsl-paper.nps.v1", "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1",
    "gzsl-paper.rgwps.v1", "gzsl-paper.crgwps.v1", "gzsl-paper.snps.v1",
    "gzsl-paper.msnps.v1", "gzsl-paper.rsnps.v1", "gzsl-paper.tcps.v1",
    "gzsl-paper.pdrs.v1", "gzsl-paper.etpc.v1", "gzsl-paper.rdss.v1",
    "gzsl-paper.srdss.v1", "gzsl-paper.trdss.v1", "gzsl-paper.rvps.v1",
    "gzsl-paper.csds.v1", "gzsl-paper.rugs.v1", "gzsl-paper.ndps.v1",
    "gzsl-paper.lscr.v1", "gzsl-paper.mhps.v1", "gzsl-paper.fbps.v1",
    "gzsl-paper.bfps.v1", "gzsl-paper.aps.v1", "gzsl-paper.cups.v1",
    "gzsl-paper.tfps.v1", "gzsl-paper.edps.v1", "gzsl-paper.edps2.v1",
    "gzsl-paper.sedps.v1", "gzsl-paper.ceps.v1", "gzsl-paper.jeds.v1",
    "gzsl-paper.nrps.v1", "gzsl-paper.tdrs.v1",
})
TEXT_ONLY_SCHEMAS = SOFT_PAIR_SCHEMAS - frozenset({
    "gzsl-paper.gwps.v1", "gzsl-paper.bgwps.v1", "gzsl-paper.mbgwps.v1",
    "gzsl-paper.nps.v1",
})
SEMANTIC_NEIGHBOR_SCHEMAS = frozenset({
    schema for schema in TEXT_ONLY_SCHEMAS
    if schema not in {
        "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1",
        "gzsl-paper.rgwps.v1", "gzsl-paper.crgwps.v1",
    }
})
TWELVE_FEATURE_SCHEMAS = SEMANTIC_NEIGHBOR_SCHEMAS - frozenset({
    "gzsl-paper.lscr.v1",
}) | frozenset({"gzsl-paper.crgwps.v1"})
NAME_FEATURE_SCHEMAS = TWELVE_FEATURE_SCHEMAS | frozenset({
    "gzsl-paper.sgwps.v1",
})
ROLE_FEATURE_SCHEMAS = TWELVE_FEATURE_SCHEMAS | frozenset({
    "gzsl-paper.rgwps.v1",
})
MODEL_CLASS_NAME_SCHEMAS = NAME_FEATURE_SCHEMAS | frozenset({
    "gzsl-paper.lscr.v1",
})
MODEL_ROLE_SCHEMAS = ROLE_FEATURE_SCHEMAS | frozenset({
    "gzsl-paper.lscr.v1",
})
ADJACENCY_MODEL_SCHEMAS = SEMANTIC_NEIGHBOR_SCHEMAS - frozenset({
    "gzsl-paper.rsnps.v1",
})
EVIDENCE_DROPOUT_SCHEMAS = frozenset({
    "gzsl-paper.edps.v1", "gzsl-paper.edps2.v1", "gzsl-paper.sedps.v1",
    "gzsl-paper.ceps.v1", "gzsl-paper.jeds.v1", "gzsl-paper.nrps.v1",
})
STAGED_SNPS_SCHEMAS = frozenset({
    "gzsl-paper.srdss.v1", "gzsl-paper.trdss.v1", "gzsl-paper.rugs.v1",
    "gzsl-paper.sedps.v1", "gzsl-paper.ceps.v1", "gzsl-paper.jeds.v1",
})
STAGED_SEDPS_SCHEMAS = frozenset({
    "gzsl-paper.nrps.v1", "gzsl-paper.tdrs.v1",
})


def class_balanced_pair_weights(
    pair_targets: torch.Tensor,
    soft_weights: torch.Tensor,
    exponent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(pair_targets.long(), minlength=2).float()
    if bool((counts == 0).any()):
        raise ValueError("B-GWPS pair标签必须同时包含top1/top2真类。")
    class_weights = (pair_targets.numel() / (2.0 * counts)).pow(float(exponent))
    combined = soft_weights.float() * class_weights.index_select(
        0, pair_targets.long()
    )
    combined = combined / combined.mean().clamp_min(1e-8)
    return combined, class_weights


def minimal_flip_delta_targets(
    pair_logits: torch.Tensor,
    pair_targets: torch.Tensor,
    max_delta: float,
) -> torch.Tensor:
    """正确top1目标为0；错误top2目标为刚好消除当前pair margin。"""
    if pair_logits.ndim != 2 or pair_logits.shape[1] != 2:
        raise ValueError("minimal flip pair_logits必须是[N,2]。")
    if tuple(pair_targets.shape) != (pair_logits.shape[0],):
        raise ValueError("minimal flip pair_targets必须是[N]。")
    margin = pair_logits[:, 0] - pair_logits[:, 1]
    targets = torch.where(
        pair_targets.long().eq(1), -0.5 * margin, torch.zeros_like(margin)
    )
    return targets.clamp(min=-float(max_delta), max=float(max_delta))


def matched_hard_pair_indices(
    pair_logits: torch.Tensor, pair_targets: torch.Tensor
) -> tuple[torch.Tensor, dict[str, int]]:
    """保留全部top2错误pair，并匹配等量最小margin正确pair。"""
    if pair_logits.ndim != 2 or pair_logits.shape[1] != 2:
        raise ValueError("MHPS pair_logits必须是[N,2]。")
    error_indices = torch.nonzero(pair_targets.long().eq(1), as_tuple=False).squeeze(1)
    correct_indices = torch.nonzero(pair_targets.long().eq(0), as_tuple=False).squeeze(1)
    if error_indices.numel() == 0 or correct_indices.numel() < error_indices.numel():
        raise ValueError("MHPS错误pair为空或正确pair不足。")
    correct_margins = (
        pair_logits.index_select(0, correct_indices)[:, 0]
        - pair_logits.index_select(0, correct_indices)[:, 1]
    )
    hard_correct = correct_indices.index_select(
        0, correct_margins.argsort()[: error_indices.numel()]
    )
    selected = torch.cat((error_indices, hard_correct)).sort().values
    return selected, {
        "original_count": int(pair_targets.numel()),
        "error_count": int(error_indices.numel()),
        "matched_correct_count": int(hard_correct.numel()),
        "selected_count": int(selected.numel()),
    }


def focal_pair_losses(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float
) -> torch.Tensor:
    """对容易正确pair降权，同时保留全部训练样本。"""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("FBPS logits必须是[N,2]。")
    if float(gamma) <= 0:
        raise ValueError("FBPS gamma必须为正。")
    ce = F.cross_entropy(logits, targets.long(), reduction="none")
    probability = torch.softmax(logits, dim=1).gather(
        1, targets.long().unsqueeze(1)
    ).squeeze(1)
    return (1.0 - probability).pow(float(gamma)) * ce


def antisymmetric_pair_augmentation(
    pair_logits: torch.Tensor,
    pair_features: torch.Tensor,
    pair_targets: torch.Tensor,
    pair_weights: torch.Tensor,
):
    """交换pair顺序、取反差值并翻转标签，生成严格镜像训练集。"""
    mirrored_logits = pair_logits.flip(dims=(1,))
    mirrored_features = -pair_features
    mirrored_targets = 1 - pair_targets.long()
    return (
        torch.cat((pair_logits, mirrored_logits), dim=0),
        torch.cat((pair_features, mirrored_features), dim=0),
        torch.cat((pair_targets.long(), mirrored_targets), dim=0),
        torch.cat((pair_weights, pair_weights), dim=0),
    )


def true_class_balancing_weights(
    true_class_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """按pair真实类别频次生成均值为1的温和类别均衡权重。"""
    counts = torch.bincount(true_class_ids.long(), minlength=200).float()
    present = counts.gt(0)
    if int(present.sum()) < 2:
        raise ValueError("CUPS真实类别数量不足。")
    weights_per_class = torch.zeros_like(counts)
    weights_per_class[present] = (
        true_class_ids.numel() / (float(present.sum()) * counts[present])
    )
    weights = weights_per_class.index_select(0, true_class_ids.long())
    return weights, {
        "present_class_count": float(present.sum()),
        "min_class_count": float(counts[present].min()),
        "max_class_count": float(counts[present].max()),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "weight_std": float(weights.std(unbiased=False)),
    }


def mask_pair_evidence_feature(
    features: torch.Tensor, feature_mean: torch.Tensor, feature_index: int
) -> torch.Tensor:
    """用训练均值替换一个非margin维度，使标准化后的该维严格为0。"""
    if features.ndim != 2 or not 1 <= int(feature_index) < features.shape[1]:
        raise ValueError("EDPS只能屏蔽非margin证据维度。")
    masked = features.clone()
    masked[:, int(feature_index)] = feature_mean[int(feature_index)]
    return masked


def pair_correction_consistency_loss(
    masked_correction: torch.Tensor, full_correction: torch.Tensor
) -> torch.Tensor:
    """约束缺失一个证据时的pair修正接近完整证据修正。"""
    if (
        masked_correction.ndim != 2
        or masked_correction.shape[1] != 2
        or full_correction.shape != masked_correction.shape
    ):
        raise ValueError("CEPS一致性输入必须是相同形状的[N,2]。")
    return F.mse_loss(masked_correction, full_correction)


def all_single_evidence_omissions(
    features: torch.Tensor, feature_mean: torch.Tensor
) -> torch.Tensor:
    """返回11个leave-one-evidence-out视图，margin维始终保留。"""
    if features.ndim != 2 or features.shape[1] != 12:
        raise ValueError("JEDS输入必须是[B,12]。")
    if tuple(feature_mean.shape) != (12,):
        raise ValueError("JEDS特征均值必须是[12]。")
    return torch.stack(
        [
            mask_pair_evidence_feature(features, feature_mean, feature_index)
            for feature_index in range(1, 12)
        ],
        dim=0,
    )


def hard_margin_only_for_schema(schema: str) -> bool:
    return schema not in SOFT_PAIR_SCHEMAS


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    schema = config.get("schema_version") if isinstance(config, dict) else None
    if schema in ("gzsl-paper.msnps.v1", "gzsl-paper.rsnps.v1"):
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "semantic_neighbor_rule"}
    elif schema == "gzsl-paper.tcps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "context_feature"}
    elif schema == "gzsl-paper.pdrs.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "pair_role_weighting"}
    elif schema == "gzsl-paper.etpc.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "training_objective"}
    elif schema == "gzsl-paper.rdss.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "context_feature"}
    elif schema == "gzsl-paper.srdss.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "context_feature",
            "snps_model", "snps_model_sha256", "training_scope",
        }
    elif schema == "gzsl-paper.trdss.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "context_feature",
            "snps_model", "snps_model_sha256", "training_scope",
            "trust_region_weight",
        }
    elif schema == "gzsl-paper.rvps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "context_feature"}
    elif schema == "gzsl-paper.csds.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "context_feature"}
    elif schema == "gzsl-paper.rugs.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "context_feature",
            "snps_model", "snps_model_sha256", "training_scope", "max_gamma",
        }
    elif schema == "gzsl-paper.ndps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "context_feature"}
    elif schema == "gzsl-paper.lscr.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "candidate_count",
            "training_scope",
        }
    elif schema == "gzsl-paper.mhps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "pair_sampling"}
    elif schema == "gzsl-paper.fbps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "training_objective",
            "focal_gamma",
        }
    elif schema == "gzsl-paper.bfps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k", "selector_bias_mode"}
    elif schema == "gzsl-paper.aps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "selector_bias_mode",
            "pair_augmentation", "gate_margin_mode",
        }
    elif schema == "gzsl-paper.cups.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k",
            "true_class_balance",
        }
    elif schema == "gzsl-paper.tfps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "training_scope",
            "error_weight_floor",
        }
    elif schema == "gzsl-paper.tdrs.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "context_feature",
            "sedps_model", "sedps_model_sha256", "training_scope",
        }
    elif schema == "gzsl-paper.nrps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "evidence_drop_count",
            "evidence_drop_scope", "evidence_drop_schedule", "sedps_model",
            "sedps_model_sha256", "training_scope", "selector_hidden_dim",
            "max_raw_residual",
        }
    elif schema == "gzsl-paper.jeds.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "evidence_drop_count",
            "evidence_drop_scope", "evidence_drop_schedule", "snps_model",
            "snps_model_sha256", "training_scope", "training_objective",
        }
    elif schema == "gzsl-paper.ceps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "evidence_drop_count",
            "evidence_drop_scope", "evidence_drop_schedule", "snps_model",
            "snps_model_sha256", "training_scope", "training_objective",
            "consistency_weight",
        }
    elif schema == "gzsl-paper.sedps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "evidence_drop_count",
            "evidence_drop_scope", "evidence_drop_schedule", "snps_model",
            "snps_model_sha256", "training_scope",
        }
    elif schema in EVIDENCE_DROPOUT_SCHEMAS:
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {
            "pair_training_scope", "semantic_neighbor_k", "evidence_drop_count",
            "evidence_drop_scope", "evidence_drop_schedule",
        }
    elif schema == "gzsl-paper.snps.v1":
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope", "semantic_neighbor_k"}
    elif schema in (
        "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1",
        "gzsl-paper.rgwps.v1", "gzsl-paper.crgwps.v1",
    ):
        expected_keys = (
            CONFIG_KEYS
            - {
                "feature_provenance_complete", "patch_inputs", "patch_sha256",
                "patch_top_k", "patch_chunk_size",
            }
        ) | {"pair_training_scope"}
    elif schema == "gzsl-paper.nps.v1":
        expected_keys = CONFIG_KEYS | {"pair_training_scope", "selector_hidden_dim"}
    elif schema == "gzsl-paper.egpes.v1":
        expected_keys = CONFIG_KEYS | {"pair_training_quantile"}
    elif schema in ("gzsl-paper.bgwps.v1", "gzsl-paper.mbgwps.v1"):
        expected_keys = CONFIG_KEYS | {"pair_training_scope", "pair_class_balance"}
    elif schema == "gzsl-paper.gwps.v1":
        expected_keys = CONFIG_KEYS | {"pair_training_scope"}
    else:
        expected_keys = CONFIG_KEYS
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"GPES配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    identity = {
        "gzsl-paper.gpes.v1": ("V2-INNOVATION-062", "IDEA-096"),
        "gzsl-paper.gwps.v1": ("V2-INNOVATION-063", "IDEA-097"),
        "gzsl-paper.bgwps.v1": ("V2-INNOVATION-064", "IDEA-098"),
        "gzsl-paper.mbgwps.v1": ("V2-INNOVATION-065", "IDEA-099"),
        "gzsl-paper.egpes.v1": ("V2-INNOVATION-066", "IDEA-100"),
        "gzsl-paper.nps.v1": ("V2-INNOVATION-067", "IDEA-101"),
        "gzsl-paper.tgwps.v1": ("V2-INNOVATION-068", "IDEA-102"),
        "gzsl-paper.sgwps.v1": ("V2-INNOVATION-069", "IDEA-103"),
        "gzsl-paper.rgwps.v1": ("V2-INNOVATION-070", "IDEA-104"),
        "gzsl-paper.crgwps.v1": ("V2-INNOVATION-071", "IDEA-105"),
        "gzsl-paper.snps.v1": ("V2-INNOVATION-072", "IDEA-106"),
        "gzsl-paper.msnps.v1": ("V2-INNOVATION-073", "IDEA-107"),
        "gzsl-paper.rsnps.v1": ("V2-INNOVATION-074", "IDEA-108"),
        "gzsl-paper.tcps.v1": ("V2-INNOVATION-075", "IDEA-109"),
        "gzsl-paper.pdrs.v1": ("V2-INNOVATION-076", "IDEA-110"),
        "gzsl-paper.etpc.v1": ("V2-INNOVATION-077", "IDEA-111"),
        "gzsl-paper.rdss.v1": ("V2-INNOVATION-078", "IDEA-112"),
        "gzsl-paper.srdss.v1": ("V2-INNOVATION-079", "IDEA-113"),
        "gzsl-paper.trdss.v1": ("V2-INNOVATION-080", "IDEA-114"),
        "gzsl-paper.rvps.v1": ("V2-INNOVATION-081", "IDEA-115"),
        "gzsl-paper.csds.v1": ("V2-INNOVATION-082", "IDEA-116"),
        "gzsl-paper.rugs.v1": ("V2-INNOVATION-083", "IDEA-117"),
        "gzsl-paper.ndps.v1": ("V2-INNOVATION-084", "IDEA-118"),
        "gzsl-paper.lscr.v1": ("V2-INNOVATION-085", "IDEA-119"),
        "gzsl-paper.mhps.v1": ("V2-INNOVATION-087", "IDEA-121"),
        "gzsl-paper.fbps.v1": ("V2-INNOVATION-088", "IDEA-122"),
        "gzsl-paper.bfps.v1": ("V2-INNOVATION-089", "IDEA-123"),
        "gzsl-paper.aps.v1": ("V2-INNOVATION-090", "IDEA-124"),
        "gzsl-paper.cups.v1": ("V2-INNOVATION-091", "IDEA-125"),
        "gzsl-paper.tfps.v1": ("V2-INNOVATION-092", "IDEA-126"),
        "gzsl-paper.edps.v1": ("V2-INNOVATION-093", "IDEA-127"),
        "gzsl-paper.edps2.v1": ("V2-INNOVATION-094", "IDEA-127"),
        "gzsl-paper.sedps.v1": ("V2-INNOVATION-095", "IDEA-128"),
        "gzsl-paper.ceps.v1": ("V2-INNOVATION-096", "IDEA-129"),
        "gzsl-paper.jeds.v1": ("V2-INNOVATION-097", "IDEA-130"),
        "gzsl-paper.nrps.v1": ("V2-INNOVATION-098", "IDEA-131"),
        "gzsl-paper.tdrs.v1": ("V2-INNOVATION-099", "IDEA-132"),
    }.get(schema)
    if identity is None or (
        config["experiment_id"], config["idea_id"]
    ) != identity:
        raise ValueError("GPES身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("GPES协议边界错误。")
    if (
        schema not in TEXT_ONLY_SCHEMAS
        and config["feature_provenance_complete"] is not False
    ) or config["text_cache_provenance_complete"] is not False:
        raise ValueError("GPES cache provenance边界错误。")
    if (
        (
            schema not in TEXT_ONLY_SCHEMAS
            and (
                int(config["patch_top_k"]) != 2
                or int(config["patch_chunk_size"]) != 16
            )
        )
        or config["group_rule"] != "class_name_last_token_min2"
        or config["threshold_source"] != (
            "train_wrong_suffix_or_semantic_neighbor_margin"
            if schema in SEMANTIC_NEIGHBOR_SCHEMAS
            else "train_wrong_same_group_margin"
        )
        or float(config["threshold_quantile"]) != 0.25
        or float(config["margin_temperature"]) != 0.1
        or float(config["max_delta"]) != 0.5
        or int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
        or config["optimizer"] != "Adam"
        or float(config["learning_rate"]) not in (
            (0.001, 0.0001)
            if schema in {
                "gzsl-paper.sedps.v1", "gzsl-paper.ceps.v1",
                "gzsl-paper.jeds.v1",
                "gzsl-paper.nrps.v1",
            }
            else (0.001,)
        )
        or float(config["weight_decay"]) != 0.0001
    ):
        raise ValueError("GPES训练参数错误。")
    if schema in (
        "gzsl-paper.gwps.v1", "gzsl-paper.bgwps.v1", "gzsl-paper.mbgwps.v1",
        "gzsl-paper.nps.v1", "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1",
        "gzsl-paper.rgwps.v1",
        "gzsl-paper.crgwps.v1",
    ) and config[
        "pair_training_scope"
    ] != "all_same_group_top2_soft_gate":
        raise ValueError("GWPS必须使用全同族top2与soft gate加权。")
    if schema == "gzsl-paper.snps.v1" and config[
        "pair_training_scope"
    ] != f"suffix_or_semantic_top{int(config['semantic_neighbor_k'])}_soft_gate":
        raise ValueError("SNPS pair scope必须与semantic_neighbor_k一致。")
    if schema == "gzsl-paper.msnps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_mutual_semantic_top5_soft_gate":
        raise ValueError("M-SNPS必须使用类名族群并互为语义top5 soft gate。")
    if schema == "gzsl-paper.rsnps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_reciprocal_semantic_top5_soft_gate":
        raise ValueError("R-SNPS必须使用互惠加权语义top5 soft gate。")
    if schema == "gzsl-paper.tcps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("TCPS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.pdrs.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("PDRS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.etpc.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("ETPC必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.rdss.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("RDSS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.srdss.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("S-RDSS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.trdss.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("TR-RDSS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.rvps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("RVPS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.csds.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("CSDS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.rugs.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("RUGS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.ndps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("NDPS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.lscr.v1" and config[
        "pair_training_scope"
    ] != "related_top3_true_contained_soft_gate":
        raise ValueError("LSCR必须使用相关top3真类包含样本。")
    if schema == "gzsl-paper.mhps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("MHPS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.fbps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("FBPS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.bfps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("BFPS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.aps.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("APS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.cups.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("CUPS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.tfps.v1" and config[
        "pair_training_scope"
    ] != "teacher_forced_related_top1_true":
        raise ValueError("TFPS必须使用教师强制相关pair。")
    if schema in EVIDENCE_DROPOUT_SCHEMAS and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("EDPS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.tdrs.v1" and config[
        "pair_training_scope"
    ] != "suffix_or_semantic_top3_soft_gate":
        raise ValueError("TDRS必须使用稳定语义top3 soft gate。")
    if schema == "gzsl-paper.bgwps.v1" and config[
        "pair_class_balance"
    ] != "inverse_frequency":
        raise ValueError("B-GWPS必须使用pair标签逆频率平衡。")
    if schema == "gzsl-paper.mbgwps.v1" and config[
        "pair_class_balance"
    ] != "sqrt_inverse_frequency":
        raise ValueError("M-BGWPS必须使用平方根逆频率平衡。")
    if schema == "gzsl-paper.egpes.v1" and float(config[
        "pair_training_quantile"
    ]) != 0.5:
        raise ValueError("E-GPES训练pair门槛必须为50分位。")
    if schema == "gzsl-paper.nps.v1" and int(config[
        "selector_hidden_dim"
    ]) != 8:
        raise ValueError("NPS hidden_dim必须为8。")
    if schema == "gzsl-paper.snps.v1" and int(config[
        "semantic_neighbor_k"
    ]) not in (3, 5):
        raise ValueError("SNPS semantic_neighbor_k只允许3或5。")
    if schema == "gzsl-paper.msnps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 5:
        raise ValueError("M-SNPS semantic_neighbor_k必须为5。")
    if schema == "gzsl-paper.rsnps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 5:
        raise ValueError("R-SNPS semantic_neighbor_k必须为5。")
    if schema == "gzsl-paper.tcps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("TCPS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.pdrs.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("PDRS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.etpc.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("ETPC semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.rdss.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("RDSS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.srdss.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("S-RDSS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.trdss.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("TR-RDSS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.rvps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("RVPS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.csds.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("CSDS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.rugs.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("RUGS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.ndps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("NDPS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.lscr.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("LSCR semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.mhps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("MHPS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.fbps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("FBPS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.bfps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("BFPS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.aps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("APS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.cups.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("CUPS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.tfps.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("TFPS semantic_neighbor_k必须为3。")
    if schema in EVIDENCE_DROPOUT_SCHEMAS and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("EDPS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.tdrs.v1" and int(config[
        "semantic_neighbor_k"
    ]) != 3:
        raise ValueError("TDRS semantic_neighbor_k必须为3。")
    if schema == "gzsl-paper.msnps.v1" and config[
        "semantic_neighbor_rule"
    ] != "mutual_top5":
        raise ValueError("M-SNPS必须使用mutual_top5。")
    if schema == "gzsl-paper.rsnps.v1" and config[
        "semantic_neighbor_rule"
    ] != "reciprocity_weighted_top5":
        raise ValueError("R-SNPS必须使用reciprocity_weighted_top5。")
    if schema == "gzsl-paper.tcps.v1" and config[
        "context_feature"
    ] != "top2_minus_top3_margin":
        raise ValueError("TCPS必须使用top2_minus_top3_margin。")
    if schema == "gzsl-paper.pdrs.v1" and config[
        "pair_role_weighting"
    ] != "cosine_distance_mean1":
        raise ValueError("PDRS必须使用cosine_distance_mean1。")
    if schema == "gzsl-paper.etpc.v1" and config[
        "training_objective"
    ] != "minimal_flip_regression":
        raise ValueError("ETPC必须使用minimal_flip_regression。")
    if schema == "gzsl-paper.rdss.v1" and config[
        "context_feature"
    ] != "raw_role_difference_std":
        raise ValueError("RDSS必须使用raw_role_difference_std。")
    if schema == "gzsl-paper.srdss.v1" and (
        config["context_feature"] != "raw_role_difference_std"
        or config["training_scope"] != "freeze_snps_train_scale_only"
    ):
        raise ValueError("S-RDSS必须冻结SNPS且只训练尺度系数。")
    if schema == "gzsl-paper.trdss.v1" and (
        config["context_feature"] != "raw_role_difference_std"
        or config["training_scope"] != "snps_initialized_joint_trust_region"
        or float(config["trust_region_weight"]) != 0.1
    ):
        raise ValueError("TR-RDSS信赖域配置错误。")
    if schema == "gzsl-paper.rvps.v1" and config[
        "context_feature"
    ] != "signed_role_vote_mean":
        raise ValueError("RVPS必须使用signed_role_vote_mean。")
    if schema == "gzsl-paper.csds.v1" and config[
        "context_feature"
    ] != "absolute_claude_merge_pair_gap":
        raise ValueError("CSDS必须使用absolute_claude_merge_pair_gap。")
    if schema == "gzsl-paper.rugs.v1" and (
        config["context_feature"] != "raw_role_difference_std"
        or config["training_scope"] != "freeze_snps_train_gamma_only"
        or float(config["max_gamma"]) != 1.0
    ):
        raise ValueError("RUGS乘法不确定性门控配置错误。")
    if schema == "gzsl-paper.ndps.v1" and config[
        "context_feature"
    ] != "semantic_log_degree_difference":
        raise ValueError("NDPS必须使用semantic_log_degree_difference。")
    if schema == "gzsl-paper.lscr.v1" and (
        int(config["candidate_count"]) != 3
        or config["training_scope"] != "related_top3_true_contained"
    ):
        raise ValueError("LSCR三类训练配置错误。")
    if schema == "gzsl-paper.mhps.v1" and config[
        "pair_sampling"
    ] != "all_errors_plus_equal_lowest_margin_correct":
        raise ValueError("MHPS pair_sampling错误。")
    if schema == "gzsl-paper.fbps.v1" and (
        config["training_objective"] != "focal_pair_ce"
        or float(config["focal_gamma"]) != 2.0
    ):
        raise ValueError("FBPS focal配置错误。")
    if schema == "gzsl-paper.bfps.v1" and config[
        "selector_bias_mode"
    ] != "fixed_zero":
        raise ValueError("BFPS selector bias必须固定为0。")
    if schema == "gzsl-paper.aps.v1" and (
        config["selector_bias_mode"] != "fixed_zero"
        or config["pair_augmentation"] != "swap_and_negate"
        or config["gate_margin_mode"] != "absolute"
    ):
        raise ValueError("APS反对称训练配置错误。")
    if schema == "gzsl-paper.cups.v1" and config[
        "true_class_balance"
    ] != "inverse_pair_frequency_mean1":
        raise ValueError("CUPS真实类别均衡配置错误。")
    if schema == "gzsl-paper.tfps.v1" and (
        config["training_scope"] != "wrong_top1_vs_true_correct_top1_vs_top2"
        or float(config["error_weight_floor"]) != 0.25
    ):
        raise ValueError("TFPS教师强制配置错误。")
    if schema in EVIDENCE_DROPOUT_SCHEMAS - {"gzsl-paper.jeds.v1"} and (
        int(config["evidence_drop_count"]) != 1
        or config["evidence_drop_scope"] != "non_margin_11_features"
        or config["evidence_drop_schedule"] != "cyclic_seed_offset"
    ):
        raise ValueError("EDPS证据dropout配置错误。")
    if schema == "gzsl-paper.jeds.v1" and (
        int(config["evidence_drop_count"]) != 11
        or config["evidence_drop_scope"] != "all_non_margin_11_features"
        or config["evidence_drop_schedule"] != "all_omissions_each_batch"
        or config["training_scope"] != "initialize_snps_then_jackknife_finetune"
        or config["training_objective"] != "mean_pair_ce_over_11_omissions"
        or float(config["learning_rate"]) != 0.0001
    ):
        raise ValueError("JEDS jackknife训练配置错误。")
    if schema == "gzsl-paper.nrps.v1" and (
        config["training_scope"]
        != "freeze_sedps_train_nonlinear_residual_only"
        or int(config["selector_hidden_dim"]) != 8
        or float(config["max_raw_residual"]) != 0.25
        or float(config["learning_rate"]) != 0.0001
    ):
        raise ValueError("NRPS非线性残差配置错误。")
    if schema == "gzsl-paper.tdrs.v1" and (
        config["context_feature"] != "top_text_distance_role_image_difference"
        or config["training_scope"]
        != "freeze_sedps_train_discriminative_role_only"
    ):
        raise ValueError("TDRS判别角色配置错误。")
    if schema == "gzsl-paper.sedps.v1" and config[
        "training_scope"
    ] != "initialize_snps_then_evidence_dropout_finetune":
        raise ValueError("S-EDPS必须从SNPS权重开始证据dropout微调。")
    if schema == "gzsl-paper.ceps.v1" and (
        config["training_scope"]
        != "initialize_snps_then_evidence_consistency_finetune"
        or config["training_objective"]
        != "masked_pair_ce_plus_full_correction_consistency"
        or float(config["consistency_weight"]) not in (0.1, 100.0)
        or float(config["learning_rate"]) != 0.0001
    ):
        raise ValueError("CEPS分阶段一致性配置错误。")
    return config, sha256_file(path)


def extract_pair_examples(
    logits,
    images,
    patch_scores,
    targets,
    ids,
    group_ids,
    claude_prototypes,
    merge_prototypes,
    threshold,
    hard_margin_only: bool = True,
    margin_temperature: float = 0.1,
    extra_prototypes: torch.Tensor | None = None,
    role_prototypes: torch.Tensor | None = None,
    center_role_features: bool = False,
    pair_adjacency: torch.Tensor | None = None,
    pair_confidence: torch.Tensor | None = None,
    third_class_context: bool = False,
    pair_role_weighting: bool = False,
    role_scale_context: bool = False,
    role_vote_context: bool = False,
    source_disagreement_context: bool = False,
    neighbor_degree_context: bool = False,
    discriminative_role_context: bool = False,
):
    top = logits.topk(2, dim=1)
    global_ids = ids.index_select(0, top.indices.reshape(-1)).reshape_as(top.indices)
    groups = group_ids.index_select(0, global_ids.reshape(-1).cpu()).reshape_as(
        global_ids.cpu()
    ).to(logits.device)
    same_group = groups[:, 0].eq(groups[:, 1]) & groups[:, 0].ge(0)
    suffix_group = same_group
    relation_weights = torch.ones_like(top.values[:, 0])
    if pair_adjacency is not None:
        adjacency = pair_adjacency.to(logits.device)
        same_group = same_group | adjacency[
            global_ids[:, 0], global_ids[:, 1]
        ]
    if pair_confidence is not None:
        confidence = pair_confidence.to(logits.device)[
            global_ids[:, 0], global_ids[:, 1]
        ]
        relation_weights = torch.where(
            suffix_group, torch.ones_like(confidence), confidence
        )
        same_group = same_group | confidence.gt(0)
    contains_true = top.indices.eq(targets.unsqueeze(1)).any(dim=1)
    margin = top.values[:, 0] - top.values[:, 1]
    selected = same_group & contains_true
    if hard_margin_only:
        selected = selected & margin.le(float(threshold))
    soft_weights = torch.sigmoid(
        (float(threshold) - margin) / float(margin_temperature)
    )
    normalized = F.normalize(images.float(), dim=-1)
    claude_logits = normalized @ claude_prototypes.index_select(0, ids).T
    merge_logits = normalized @ merge_prototypes.index_select(0, ids).T
    values = [
        margin,
        claude_logits.gather(1, top.indices)[:, 0]
        - claude_logits.gather(1, top.indices)[:, 1],
        merge_logits.gather(1, top.indices)[:, 0]
        - merge_logits.gather(1, top.indices)[:, 1],
    ]
    if extra_prototypes is not None:
        extra_logits = normalized @ extra_prototypes.index_select(0, ids).T
        values.append(
            extra_logits.gather(1, top.indices)[:, 0]
            - extra_logits.gather(1, top.indices)[:, 1]
        )
    if role_prototypes is not None:
        if tuple(role_prototypes.shape[1:]) != (8, 768):
            raise ValueError("R-GWPS role_prototypes必须是[C,8,768]。")
        role_logits = torch.einsum(
            "bd,crd->bcr", normalized, role_prototypes.index_select(0, ids)
        )
        role_top2 = role_logits.gather(
            1, top.indices.unsqueeze(-1).expand(-1, -1, 8)
        )
        role_diffs = role_top2[:, 0] - role_top2[:, 1]
        role_scale = role_diffs.std(dim=1, unbiased=False)
        role_vote = torch.sign(role_diffs).mean(dim=1)
        if center_role_features:
            role_diffs = role_diffs - role_diffs.mean(dim=1, keepdim=True)
            role_diffs = role_diffs / role_diffs.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
        if pair_role_weighting:
            role_diffs = role_diffs * pair_role_distance_weights(
                role_prototypes, global_ids
            )
        values.extend(role_diffs.unbind(dim=1))
        if role_scale_context:
            values.append(role_scale)
        if role_vote_context:
            values.append(role_vote)
        if discriminative_role_context:
            values.append(
                top_discriminative_role_difference(
                    role_prototypes,
                    global_ids,
                    role_top2[:, 0] - role_top2[:, 1],
                )
            )
    if patch_scores is not None:
        local_patch = patch_scores.to(logits.device).float()
        if local_patch.shape[1] == 200 and ids.numel() != 200:
            local_patch = local_patch.index_select(1, ids)
        values.append(
            local_patch.gather(1, top.indices)[:, 0]
            - local_patch.gather(1, top.indices)[:, 1]
        )
    if third_class_context:
        top3 = logits.topk(3, dim=1)
        values.append(top3.values[:, 1] - top3.values[:, 2])
    if source_disagreement_context:
        values.append((values[1] - values[2]).abs())
    if neighbor_degree_context:
        if pair_adjacency is None:
            raise ValueError("NDPS需要pair_adjacency。")
        log_degree = torch.log1p(
            pair_adjacency.to(logits.device).float().sum(dim=1)
        )
        values.append(
            log_degree.index_select(0, global_ids[:, 0])
            - log_degree.index_select(0, global_ids[:, 1])
        )
    features = torch.stack(values, dim=1)
    pair_targets = top.indices[:, 1].eq(targets).long()
    true_global_ids = ids.index_select(0, targets.long())
    return (
        top.values[selected].detach().cpu(),
        features[selected].detach().cpu(),
        pair_targets[selected].detach().cpu(),
        int(selected.sum()),
        (soft_weights[selected] * relation_weights[selected]).detach().cpu(),
        true_global_ids[selected].detach().cpu(),
    )


def extract_triplet_examples(
    logits,
    images,
    targets,
    ids,
    group_ids,
    semantic_adjacency,
    claude_prototypes,
    merge_prototypes,
    class_name_prototypes,
    role_prototypes,
    threshold,
    margin_temperature=0.1,
):
    top = logits.topk(3, dim=1)
    global_ids = ids.index_select(0, top.indices.reshape(-1)).reshape_as(top.indices)
    groups = group_ids.to(logits.device).index_select(
        0, global_ids.reshape(-1)
    ).reshape_as(global_ids)
    related = (
        ((groups[:, 0:1] == groups[:, 1:]) & groups[:, 0:1].ge(0)).any(dim=1)
        | semantic_adjacency.to(logits.device)[
            global_ids[:, 0:1].expand(-1, 2), global_ids[:, 1:]
        ].any(dim=1)
    )
    target_matches = top.indices.eq(targets.unsqueeze(1))
    selected = related & target_matches.any(dim=1)
    margin = top.values[:, 0] - top.values[:, 1]
    soft_weights = torch.sigmoid(
        (float(threshold) - margin) / float(margin_temperature)
    )
    normalized_images = F.normalize(images.float(), dim=-1)
    sources = []
    for prototypes in (
        claude_prototypes, merge_prototypes, class_name_prototypes
    ):
        scores = normalized_images @ prototypes.index_select(0, ids).T
        sources.append(scores.gather(1, top.indices).unsqueeze(-1))
    role_logits = torch.einsum(
        "bd,crd->bcr", normalized_images, role_prototypes.index_select(0, ids)
    )
    role_top3 = role_logits.gather(
        1, top.indices.unsqueeze(-1).expand(-1, -1, 8)
    )
    features = torch.cat((*sources, role_top3), dim=2)
    features = features - features.mean(dim=1, keepdim=True)
    local_targets = target_matches.long().argmax(dim=1)
    true_global_ids = ids.index_select(0, targets.long())
    return (
        top.values[selected].detach().cpu(),
        features[selected].detach().cpu(),
        local_targets[selected].detach().cpu(),
        int(selected.sum()),
        soft_weights[selected].detach().cpu(),
        true_global_ids[selected].detach().cpu(),
    )


def extract_teacher_forced_pairs(
    logits,
    images,
    targets,
    ids,
    group_ids,
    semantic_adjacency,
    claude_prototypes,
    merge_prototypes,
    class_name_prototypes,
    role_prototypes,
    threshold,
    margin_temperature=0.1,
    error_weight_floor=0.25,
):
    top2 = logits.topk(2, dim=1)
    predicted = top2.indices[:, 0]
    wrong = predicted.ne(targets)
    competitor = torch.where(wrong, targets, top2.indices[:, 1])
    pair_indices = torch.stack((predicted, competitor), dim=1)
    pair_values = logits.gather(1, pair_indices)
    global_ids = ids.index_select(0, pair_indices.reshape(-1)).reshape_as(pair_indices)
    groups = group_ids.to(logits.device).index_select(
        0, global_ids.reshape(-1)
    ).reshape_as(global_ids)
    related = (
        (groups[:, 0] == groups[:, 1]) & groups[:, 0].ge(0)
    ) | semantic_adjacency.to(logits.device)[global_ids[:, 0], global_ids[:, 1]]
    margin = pair_values[:, 0] - pair_values[:, 1]
    soft_weights = torch.sigmoid(
        (float(threshold) - margin) / float(margin_temperature)
    )
    soft_weights = torch.where(
        wrong,
        soft_weights.clamp_min(float(error_weight_floor)),
        soft_weights,
    )
    normalized = F.normalize(images.float(), dim=-1)
    values = [margin]
    for prototypes in (
        claude_prototypes, merge_prototypes, class_name_prototypes
    ):
        scores = normalized @ prototypes.index_select(0, ids).T
        gathered = scores.gather(1, pair_indices)
        values.append(gathered[:, 0] - gathered[:, 1])
    role_logits = torch.einsum(
        "bd,crd->bcr", normalized, role_prototypes.index_select(0, ids)
    )
    role_pair = role_logits.gather(
        1, pair_indices.unsqueeze(-1).expand(-1, -1, 8)
    )
    role_diffs = role_pair[:, 0] - role_pair[:, 1]
    role_diffs = role_diffs - role_diffs.mean(dim=1, keepdim=True)
    role_diffs = role_diffs / role_diffs.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    values.extend(role_diffs.unbind(dim=1))
    features = torch.stack(values, dim=1)
    true_global_ids = ids.index_select(0, targets.long())
    return (
        pair_values[related].detach().cpu(),
        features[related].detach().cpu(),
        wrong[related].long().detach().cpu(),
        int(related.sum()),
        soft_weights[related].detach().cpu(),
        true_global_ids[related].detach().cpu(),
    )


@torch.no_grad()
def derive_relation_threshold(
    parent,
    sdrs,
    calibrator,
    sdcr,
    train_features,
    labels,
    seen_classes,
    group_ids,
    pair_adjacency,
    device,
    quantile,
    relation_name="suffix_group_or_semantic_top5",
):
    """从seen训练错误中为类名族群并语义邻居关系固定margin门槛。"""
    mapping = torch.full((200,), -1, dtype=torch.long)
    mapping[seen_classes] = torch.arange(seen_classes.numel())
    ids = seen_classes.to(device)
    local_groups = group_ids.index_select(0, seen_classes).to(device)
    adjacency = pair_adjacency.to(device)
    margins, related_values, wrongs = [], [], []
    for start in range(0, train_features.shape[0], 512):
        images = train_features[start : start + 512].to(device).float()
        logits = _base_logits(
            parent, sdrs, calibrator, sdcr, images, ids, seen_classes
        )
        top = logits.topk(2, dim=1)
        global_ids = ids.index_select(0, top.indices.reshape(-1)).reshape_as(
            top.indices
        )
        groups = local_groups.index_select(
            0, top.indices.reshape(-1)
        ).reshape_as(top.indices)
        same_group = groups[:, 0].eq(groups[:, 1]) & groups[:, 0].ge(0)
        related = same_group | adjacency[global_ids[:, 0], global_ids[:, 1]]
        targets = mapping[labels[start : start + 512]].to(device)
        margins.append((top.values[:, 0] - top.values[:, 1]).cpu())
        related_values.append(related.cpu())
        wrongs.append(top.indices[:, 0].ne(targets).cpu())
    stats = select_margin_threshold(
        torch.cat(margins), torch.cat(related_values), torch.cat(wrongs), quantile
    )
    threshold, details = stats
    details = {
        **details,
        "source": "train_wrong_suffix_or_semantic_neighbor",
        "relation": relation_name,
    }
    return threshold, details


@torch.no_grad()
def evaluate(
    parent, sdrs, calibrator, model, tensors, scores,
    seen_classes, unseen_classes, device,
):
    def predict(features, patch_scores, class_ids=None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = features.to(device).float()
        parent_logits = F.normalize(images, dim=-1) @ parent.prototypes().index_select(0, ids).T * parent.scale()
        parent_logits = sdrs(parent_logits, images, ids)
        parent_logits = calibrator(
            parent_logits, torch.isin(ids.cpu(), seen_classes).to(device)
        )
        local_patch = None if patch_scores is None else patch_scores.to(device)
        predictions = model(
            parent_logits, images, local_patch, ids
        ).argmax(1).cpu()
        return predictions if class_ids is None else class_ids[predictions]

    seen_predictions = predict(
        tensors["seen_features"], None if scores is None else scores["seen"]
    )
    unseen_predictions = predict(
        tensors["unseen_features"], None if scores is None else scores["unseen"]
    )
    zsl_predictions = predict(
        tensors["unseen_features"],
        None if scores is None else scores["unseen"],
        unseen_classes,
    )
    seen = h1.per_class_accuracy(tensors["seen_labels"], seen_predictions, seen_classes)
    unseen = h1.per_class_accuracy(
        tensors["unseen_labels"], unseen_predictions, unseen_classes
    )
    zsl = h1.per_class_accuracy(
        tensors["unseen_labels"], zsl_predictions, unseen_classes
    )
    return {
        "U": unseen * 100,
        "S": seen * 100,
        "H": 2 * seen * unseen / (seen + unseen) * 100,
        "ZS": zsl * 100,
    }


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    commit = current_code_commit()
    if commit != expected_commit:
        raise ValueError("expected-commit不一致。")
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config)
    input_sha = verify_inputs(config, paths)
    for key in (
        "base_model", "sdrs_model", "sebc_model", "casr_model", "sdcr_model",
        "class_name_embeddings", "eight_sentence_embeddings", "claude_embeddings",
        "merge_embeddings",
    ):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"GPES {key} SHA错误。")
    if config["schema_version"] in STAGED_SNPS_SCHEMAS and sha256_file(
        Path(config["snps_model"])
    ) != config["snps_model_sha256"]:
        raise ValueError("分阶段SNPS父模型SHA错误。")
    if config["schema_version"] in STAGED_SEDPS_SCHEMAS and sha256_file(
        Path(config["sedps_model"])
    ) != config["sedps_model_sha256"]:
        raise ValueError("分阶段S-EDPS父模型SHA错误。")
    text_only = config["schema_version"] in TEXT_ONLY_SCHEMAS
    if not text_only:
        for split, path_text in config["patch_inputs"].items():
            if sha256_file(h1.repo_path(path_text)) != config["patch_sha256"][split]:
                raise ValueError(f"GPES {split} patch SHA错误。")
    device = torch.device(config["device"])
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    log = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    old_stdout = sys.stdout
    sys.stdout = h1.TeeStream(sys.stdout, log)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(
            seed, strict_determinism=True, deterministic_warn_only=False
        )
        sentence = torch.load(paths["sentence_embeds"], map_location="cpu", weights_only=True)
        features = torch.load(paths["train_features"], map_location="cpu", weights_only=True)
        labels = torch.load(paths["train_labels"], map_location="cpu", weights_only=True).long()
        official = {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in OFFICIAL_KEYS
        }
        class_names_tensor = torch.load(
            Path(config["class_name_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        sentence8 = torch.load(
            Path(config["eight_sentence_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        claude = torch.load(
            Path(config["claude_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        merge = torch.load(
            Path(config["merge_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        seen_classes = torch.unique(labels, sorted=True)
        all_classes = torch.arange(200)
        unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seen_classes) or not torch.equal(
            checked_unseen, unseen_classes
        ):
            raise ValueError("GPES CUB类别边界错误。")
        parent, sdrs = _load_main(
            config, sentence, labels, features,
            class_names_tensor, seen_classes, device
        )
        calibrator_payload = torch.load(
            Path(config["sebc_model"]), map_location="cpu", weights_only=False
        )
        calibrator = EpisodicBiasCalibration(
            float(calibrator_payload["config"]["max_gamma"])
        ).to(device)
        calibrator.load_state_dict(
            calibrator_payload["calibrator_state_dict"], strict=True
        )
        calibrator.eval()
        for parameter in calibrator.parameters():
            parameter.requires_grad_(False)
        casr_payload = torch.load(
            Path(config["casr_model"]), map_location="cpu", weights_only=False
        )
        sdcr_payload = torch.load(
            Path(config["sdcr_model"]), map_location="cpu", weights_only=False
        )
        sdcr = SentenceDropoutConservativeRouting(
            sentence8,
            class_names_tensor,
            torch.softmax(
                casr_payload["aosr_state_dict"]["raw_sentence_weights"].float(), dim=0
            ).to(device),
            float(sdcr_payload["fixed_beta"]),
            float(sdcr_payload["config"]["max_logit_residual"]),
            int(sdcr_payload["config"].get("drop_count", 1)),
        ).to(device)
        sdcr.load_state_dict(sdcr_payload["sdcr_state_dict"], strict=True)
        sdcr.eval()
        group_ids = taxonomic_suffix_group_ids(load_class_names(paths["att_splits"]))
        semantic_adjacency = None
        semantic_confidence = None
        if config["schema_version"] in SEMANTIC_NEIGHBOR_SCHEMAS:
            if config["schema_version"] == "gzsl-paper.rsnps.v1":
                semantic_confidence = reciprocal_neighbor_confidence(
                    sdcr.prototypes(use_dropout=False),
                    int(config["semantic_neighbor_k"]),
                )
                semantic_adjacency = semantic_confidence.gt(0)
            else:
                semantic_adjacency = semantic_neighbor_adjacency(
                    sdcr.prototypes(use_dropout=False),
                    int(config["semantic_neighbor_k"]),
                    mutual_only=(
                        config["schema_version"] == "gzsl-paper.msnps.v1"
                    ),
                )
            threshold, threshold_stats = derive_relation_threshold(
                parent, sdrs, calibrator, sdcr, features, labels,
                seen_classes, group_ids, semantic_adjacency, device,
                float(config["threshold_quantile"]),
                relation_name=(
                    "suffix_group_or_mutual_semantic_top5"
                    if config["schema_version"] == "gzsl-paper.msnps.v1"
                    else (
                        "suffix_group_or_reciprocity_weighted_semantic_top5"
                        if config["schema_version"] == "gzsl-paper.rsnps.v1"
                        else f"suffix_group_or_semantic_top{int(config['semantic_neighbor_k'])}"
                    )
                ),
            )
        else:
            threshold, threshold_stats = derive_train_threshold(
                parent, sdrs, calibrator, sdcr, features, labels,
                seen_classes, group_ids, device,
                float(config["threshold_quantile"]),
            )
        pair_training_threshold = threshold
        pair_training_threshold_stats = threshold_stats
        if config["schema_version"] == "gzsl-paper.egpes.v1":
            pair_training_threshold, pair_training_threshold_stats = derive_train_threshold(
                parent,
                sdrs,
                calibrator,
                sdcr,
                features,
                labels,
                seen_classes,
                group_ids,
                device,
                float(config["pair_training_quantile"]),
            )
        names_n = F.normalize(class_names_tensor.float(), dim=-1)
        claude_n = F.normalize(claude.float(), dim=-1)
        merge_n = F.normalize(merge.float(), dim=-1)
        claude_orth = F.normalize(
            claude_n - (claude_n * names_n).sum(dim=-1, keepdim=True) * names_n,
            dim=-1,
        )
        merge_orth = F.normalize(
            merge_n - (merge_n * names_n).sum(dim=-1, keepdim=True) * names_n,
            dim=-1,
        )
        if text_only:
            print("using patch-free text-only pair features")
            scores = None
        else:
            print("precomputing GPES patch scores")
            scores = _precompute_scores(
                config,
                orthogonal_local_text_residuals(sentence, class_names_tensor),
                device,
            )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        ids = seen_classes.to(device)
        pair_logits_list, feature_list, target_list, pair_weight_list = [], [], [], []
        true_class_list = []
        triplet_mode = config["schema_version"] == "gzsl-paper.lscr.v1"
        teacher_forced_mode = config["schema_version"] == "gzsl-paper.tfps.v1"
        hard_margin_only = hard_margin_only_for_schema(config["schema_version"])
        for start in range(0, features.shape[0], 512):
            images = features[start : start + 512].to(device).float()
            parent_logits = F.normalize(images, dim=-1) @ parent.prototypes().index_select(0, ids).T * parent.scale()
            parent_logits = sdrs(parent_logits, images, ids)
            parent_logits = calibrator(
                parent_logits, torch.ones(150, dtype=torch.bool, device=device)
            )
            logits = sdcr(parent_logits, images, ids)
            if triplet_mode:
                package = extract_triplet_examples(
                    logits,
                    images,
                    mapping[labels[start : start + 512]].to(device),
                    ids,
                    group_ids,
                    semantic_adjacency,
                    claude_orth,
                    merge_orth,
                    names_n,
                    F.normalize(sentence8.float(), dim=-1),
                    pair_training_threshold,
                    margin_temperature=float(config["margin_temperature"]),
                )
            elif teacher_forced_mode:
                package = extract_teacher_forced_pairs(
                    logits,
                    images,
                    mapping[labels[start : start + 512]].to(device),
                    ids,
                    group_ids,
                    semantic_adjacency,
                    claude_orth,
                    merge_orth,
                    names_n,
                    F.normalize(sentence8.float(), dim=-1),
                    pair_training_threshold,
                    margin_temperature=float(config["margin_temperature"]),
                    error_weight_floor=float(config["error_weight_floor"]),
                )
            else:
                package = extract_pair_examples(
                logits,
                images,
                None if scores is None else scores["train"][start : start + 512],
                mapping[labels[start : start + 512]].to(device),
                ids,
                group_ids,
                claude_orth,
                merge_orth,
                pair_training_threshold,
                hard_margin_only=hard_margin_only,
                margin_temperature=float(config["margin_temperature"]),
                extra_prototypes=(
                    names_n
                    if config["schema_version"] in NAME_FEATURE_SCHEMAS
                    else None
                ),
                role_prototypes=(
                    F.normalize(sentence8.float(), dim=-1)
                    if config["schema_version"] in ROLE_FEATURE_SCHEMAS
                    else None
                ),
                center_role_features=(
                    config["schema_version"] in TWELVE_FEATURE_SCHEMAS
                ),
                pair_adjacency=semantic_adjacency,
                pair_confidence=semantic_confidence,
                third_class_context=(
                    config["schema_version"] == "gzsl-paper.tcps.v1"
                ),
                pair_role_weighting=(
                    config["schema_version"] == "gzsl-paper.pdrs.v1"
                ),
                role_scale_context=(
                    config["schema_version"] in (
                        "gzsl-paper.rdss.v1", "gzsl-paper.srdss.v1",
                        "gzsl-paper.trdss.v1",
                        "gzsl-paper.rugs.v1",
                    )
                ),
                role_vote_context=(
                    config["schema_version"] == "gzsl-paper.rvps.v1"
                ),
                source_disagreement_context=(
                    config["schema_version"] == "gzsl-paper.csds.v1"
                ),
                neighbor_degree_context=(
                    config["schema_version"] == "gzsl-paper.ndps.v1"
                ),
                discriminative_role_context=(
                    config["schema_version"] == "gzsl-paper.tdrs.v1"
                ),
            )
            pair_logits_list.append(package[0])
            feature_list.append(package[1])
            target_list.append(package[2])
            pair_weight_list.append(package[4])
            true_class_list.append(package[5])
        pair_logits = torch.cat(pair_logits_list)
        pair_features = torch.cat(feature_list)
        pair_targets = torch.cat(target_list)
        pair_weights = torch.cat(pair_weight_list)
        pair_true_classes = torch.cat(true_class_list)
        pair_sampling_stats = None
        if config["schema_version"] == "gzsl-paper.cups.v1":
            class_weights, pair_sampling_stats = true_class_balancing_weights(
                pair_true_classes
            )
            pair_weights = pair_weights * class_weights
        if config["schema_version"] == "gzsl-paper.aps.v1":
            original_count = int(pair_targets.numel())
            pair_logits, pair_features, pair_targets, pair_weights = (
                antisymmetric_pair_augmentation(
                    pair_logits, pair_features, pair_targets, pair_weights
                )
            )
            pair_sampling_stats = {
                "original_count": original_count,
                "augmented_count": int(pair_targets.numel()),
                "augmentation": "swap_and_negate",
            }
        if config["schema_version"] == "gzsl-paper.mhps.v1":
            selected_indices, pair_sampling_stats = matched_hard_pair_indices(
                pair_logits, pair_targets
            )
            pair_logits = pair_logits.index_select(0, selected_indices)
            pair_features = pair_features.index_select(0, selected_indices)
            pair_targets = pair_targets.index_select(0, selected_indices)
            pair_weights = pair_weights.index_select(0, selected_indices)
        if config["schema_version"] in (
            "gzsl-paper.gpes.v1", "gzsl-paper.egpes.v1"
        ):
            pair_weights = torch.ones_like(pair_weights)
        pair_class_weights = torch.ones(3 if triplet_mode else 2)
        if config["schema_version"] in (
            "gzsl-paper.bgwps.v1", "gzsl-paper.mbgwps.v1"
        ):
            pair_weights, pair_class_weights = class_balanced_pair_weights(
                pair_targets,
                pair_weights,
                exponent=(
                    0.5
                    if config["schema_version"] == "gzsl-paper.mbgwps.v1"
                    else 1.0
                ),
            )
        expected_target_count = 3 if triplet_mode else 2
        if (
            pair_targets.numel() < 50
            or pair_targets.unique().numel() != expected_target_count
        ):
            raise ValueError("GPES成对训练样本不足或标签退化。")
        feature_rows = (
            pair_features.reshape(-1, 11) if triplet_mode else pair_features
        )
        feature_mean = feature_rows.mean(dim=0)
        feature_std = feature_rows.std(dim=0, unbiased=False).clamp_min(1e-6)
        pair_dataset_stats = {
            "count": int(pair_targets.numel()),
            "top1_target_rate": float(pair_targets.eq(0).float().mean()),
            "feature_mean": [float(value) for value in feature_mean],
            "feature_std": [float(value) for value in feature_std],
            "pair_weight_mean": float(pair_weights.mean()),
            "pair_weight_std": float(pair_weights.std(unbiased=False)),
            "pair_class_weights": [
                float(value) for value in pair_class_weights
            ],
            "inference_threshold": float(threshold),
            "training_threshold": float(pair_training_threshold),
            "training_threshold_stats": pair_training_threshold_stats,
            "semantic_edge_count": (
                0
                if semantic_adjacency is None
                else int(semantic_adjacency.sum().item() // 2)
            ),
            "semantic_mutual_edge_count": (
                0
                if semantic_confidence is None
                else int(semantic_confidence.eq(1.0).sum().item() // 2)
            ),
            "semantic_one_way_edge_count": (
                0
                if semantic_confidence is None
                else int(semantic_confidence.eq(0.5).sum().item() // 2)
            ),
            "pair_sampling_stats": pair_sampling_stats,
            "evidence_dropout_counts": [0] * 11,
        }
        if config["schema_version"] == "gzsl-paper.nps.v1":
            model_class = NonlinearGatedPairSelector
        elif config["schema_version"] == "gzsl-paper.tgwps.v1":
            model_class = TextOnlyGatedPairSelector
        elif config["schema_version"] == "gzsl-paper.sgwps.v1":
            model_class = SemanticGatedPairSelector
        elif config["schema_version"] == "gzsl-paper.rgwps.v1":
            model_class = RoleAwareGatedPairSelector
        elif config["schema_version"] == "gzsl-paper.crgwps.v1":
            model_class = CenteredRoleGatedPairSelector
        elif config["schema_version"] in (
            "gzsl-paper.snps.v1", "gzsl-paper.msnps.v1"
        ):
            model_class = SemanticNeighborPairSelector
        elif config["schema_version"] == "gzsl-paper.rsnps.v1":
            model_class = ReciprocalSemanticNeighborPairSelector
        elif config["schema_version"] == "gzsl-paper.tcps.v1":
            model_class = TriadicCompetitionPairSelector
        elif config["schema_version"] == "gzsl-paper.pdrs.v1":
            model_class = PairDiscriminativeRoleSelector
        elif config["schema_version"] == "gzsl-paper.etpc.v1":
            model_class = SemanticNeighborPairSelector
        elif config["schema_version"] == "gzsl-paper.rdss.v1":
            model_class = RoleDisagreementScaleSelector
        elif config["schema_version"] == "gzsl-paper.srdss.v1":
            model_class = StagedRoleDisagreementScaleSelector
        elif config["schema_version"] == "gzsl-paper.trdss.v1":
            model_class = TrustRegionRoleDisagreementScaleSelector
        elif config["schema_version"] == "gzsl-paper.rvps.v1":
            model_class = RoleVotePairSelector
        elif config["schema_version"] == "gzsl-paper.csds.v1":
            model_class = CrossSourceDisagreementSelector
        elif config["schema_version"] == "gzsl-paper.rugs.v1":
            model_class = RoleUncertaintyGatedSelector
        elif config["schema_version"] == "gzsl-paper.ndps.v1":
            model_class = NeighborhoodDegreePairSelector
        elif config["schema_version"] == "gzsl-paper.nrps.v1":
            model_class = NonlinearResidualPairSelector
        elif config["schema_version"] == "gzsl-paper.tdrs.v1":
            model_class = StagedDiscriminativeRoleSelector
        elif config["schema_version"] == "gzsl-paper.lscr.v1":
            model_class = LocalSemanticCompetitionResolver
        elif config["schema_version"] == "gzsl-paper.mhps.v1":
            model_class = SemanticNeighborPairSelector
        elif config["schema_version"] == "gzsl-paper.fbps.v1":
            model_class = SemanticNeighborPairSelector
        elif config["schema_version"] == "gzsl-paper.bfps.v1":
            model_class = BiasFreeSemanticNeighborSelector
        elif config["schema_version"] == "gzsl-paper.aps.v1":
            model_class = AntisymmetricPairSelector
        elif config["schema_version"] == "gzsl-paper.cups.v1":
            model_class = SemanticNeighborPairSelector
        elif config["schema_version"] == "gzsl-paper.tfps.v1":
            model_class = SemanticNeighborPairSelector
        elif config["schema_version"] in EVIDENCE_DROPOUT_SCHEMAS:
            model_class = SemanticNeighborPairSelector
        else:
            model_class = GatedPairEvidenceSelector
        model_kwargs = {
            "sdcr_prototypes": sdcr.prototypes(use_dropout=False).detach(),
            "sdcr_beta": float(sdcr_payload["fixed_beta"]),
            "claude_prototypes": claude_orth,
            "merge_prototypes": merge_orth,
            "group_ids": group_ids,
            "margin_threshold": threshold,
            "margin_temperature": float(config["margin_temperature"]),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "max_delta": float(config["max_delta"]),
        }
        if config["schema_version"] == "gzsl-paper.nps.v1":
            model_kwargs["hidden_dim"] = int(config["selector_hidden_dim"])
        if config["schema_version"] in MODEL_CLASS_NAME_SCHEMAS:
            model_kwargs["class_name_prototypes"] = names_n
        if config["schema_version"] in MODEL_ROLE_SCHEMAS:
            model_kwargs["role_sentence_prototypes"] = sentence8
        if config["schema_version"] in ADJACENCY_MODEL_SCHEMAS:
            model_kwargs["semantic_adjacency"] = semantic_adjacency
        if config["schema_version"] == "gzsl-paper.rsnps.v1":
            model_kwargs["semantic_confidence"] = semantic_confidence
        if config["schema_version"] in (
            "gzsl-paper.srdss.v1", "gzsl-paper.trdss.v1", "gzsl-paper.rugs.v1"
        ):
            snps_payload = torch.load(
                Path(config["snps_model"]), map_location="cpu", weights_only=False
            )
            snps_state = snps_payload["gpes_state_dict"]
            model_kwargs.update(
                {
                    "base_selector_weight": snps_state["selector_weight"],
                    "base_selector_bias": snps_state["selector_bias"],
                    "base_feature_mean": snps_state["feature_mean"],
                    "base_feature_std": snps_state["feature_std"],
                }
            )
        if config["schema_version"] == "gzsl-paper.rugs.v1":
            model_kwargs["max_gamma"] = float(config["max_gamma"])
        if config["schema_version"] in {
            "gzsl-paper.nrps.v1", "gzsl-paper.tdrs.v1"
        }:
            sedps_payload = torch.load(
                Path(config["sedps_model"]), map_location="cpu", weights_only=False
            )
            sedps_state = sedps_payload["gpes_state_dict"]
            model_kwargs.update({
                "base_selector_weight": sedps_state["selector_weight"],
                "base_selector_bias": sedps_state["selector_bias"],
                "base_feature_mean": sedps_state["feature_mean"],
                "base_feature_std": sedps_state["feature_std"],
            })
            if config["schema_version"] == "gzsl-paper.nrps.v1":
                model_kwargs.update({
                    "hidden_dim": int(config["selector_hidden_dim"]),
                    "max_raw_residual": float(config["max_raw_residual"]),
                })
        model = model_class(**model_kwargs).to(device)
        if config["schema_version"] in {
            "gzsl-paper.sedps.v1", "gzsl-paper.ceps.v1",
            "gzsl-paper.jeds.v1",
        }:
            snps_payload = torch.load(
                Path(config["snps_model"]), map_location="cpu", weights_only=False
            )
            model.load_state_dict(snps_payload["gpes_state_dict"], strict=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        best_metrics = evaluate(
            parent, sdrs, calibrator, model, official, scores,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"GPES初始态未复现SDCR：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = -1
        history = []
        generator = torch.Generator().manual_seed(seed)
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "gpes_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "pair_dataset_stats": pair_dataset_stats,
                "config": config,
                "code_commit": commit,
                "reproducibility": reproducibility,
            },
        )
        for iteration in range(int(config["niters"])):
            batch = random_batch_indices(
                pair_targets.numel(), int(config["batch_size"]), generator
            )
            batch_pair_logits = pair_logits.index_select(0, batch).to(device)
            batch_pair_targets = pair_targets.index_select(0, batch).to(device)
            batch_features = pair_features.index_select(0, batch).to(device)
            full_batch_features = batch_features
            if config["schema_version"] == "gzsl-paper.jeds.v1":
                batch_features = all_single_evidence_omissions(
                    batch_features, feature_mean.to(device)
                )
                pair_dataset_stats["evidence_dropout_counts"] = [
                    count + 1
                    for count in pair_dataset_stats["evidence_dropout_counts"]
                ]
            elif config["schema_version"] in EVIDENCE_DROPOUT_SCHEMAS:
                masked_feature = 1 + ((iteration + seed) % 11)
                batch_features = mask_pair_evidence_feature(
                    batch_features, feature_mean.to(device), masked_feature
                )
                pair_dataset_stats["evidence_dropout_counts"][
                    masked_feature - 1
                ] += 1
            if config["schema_version"] == "gzsl-paper.jeds.v1":
                view_count, view_batch, feature_dim = batch_features.shape
                corrected = model.corrected_pair_logits(
                    batch_pair_logits.repeat(view_count, 1),
                    batch_features.reshape(view_count * view_batch, feature_dim),
                )
            else:
                corrected = (
                    model.corrected_candidate_logits(batch_pair_logits, batch_features)
                    if triplet_mode
                    else model.corrected_pair_logits(batch_pair_logits, batch_features)
                )
            if config["schema_version"] == "gzsl-paper.etpc.v1":
                target_delta = minimal_flip_delta_targets(
                    batch_pair_logits,
                    batch_pair_targets,
                    float(config["max_delta"]),
                )
                applied_delta = corrected[:, 0] - batch_pair_logits[:, 0]
                per_pair_loss = (applied_delta - target_delta).square()
            elif config["schema_version"] == "gzsl-paper.jeds.v1":
                per_pair_loss = F.cross_entropy(
                    corrected,
                    batch_pair_targets.repeat(view_count),
                    reduction="none",
                ).reshape(view_count, view_batch).mean(dim=0)
            elif config["schema_version"] == "gzsl-paper.fbps.v1":
                per_pair_loss = focal_pair_losses(
                    corrected, batch_pair_targets, float(config["focal_gamma"])
                )
            else:
                per_pair_loss = F.cross_entropy(
                    corrected,
                    batch_pair_targets,
                    reduction="none",
                )
            batch_weights = pair_weights.index_select(0, batch).to(device)
            loss = (per_pair_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
            if config["schema_version"] == "gzsl-paper.ceps.v1":
                full_corrected = model.corrected_pair_logits(
                    batch_pair_logits, full_batch_features
                )
                loss = loss + float(config["consistency_weight"]) * (
                    pair_correction_consistency_loss(corrected, full_corrected)
                )
            if config["schema_version"] == "gzsl-paper.trdss.v1":
                loss = loss + float(config["trust_region_weight"]) * model.trust_region_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(model)
            optimizer.step()
            if hasattr(model, "project_parameters"):
                model.project_parameters()
            if iteration % int(config["report_interval"]) == 0:
                metrics = evaluate(
                    parent, sdrs, calibrator, model, official, scores,
                    seen_classes, unseen_classes, device
                )
                stats = model.stats()
                history.append(
                    {
                        "iteration": iteration,
                        "pair_loss": float(loss.detach()),
                        "official_metrics_percent": metrics,
                        "selector_stats": stats,
                    }
                )
                if metrics["H"] > best_h:
                    best_h = metrics["H"]
                    best_metrics = metrics
                    best_state = copy.deepcopy(model.state_dict())
                    best_iteration = iteration
                    atomic_torch_save(
                        output_dir / "model_best.pth",
                        {
                            "gpes_state_dict": best_state,
                            "best_metrics_percent": best_metrics,
                            "selected_iteration": best_iteration,
                            "pair_dataset_stats": pair_dataset_stats,
                            "config": config,
                            "code_commit": commit,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f} loss={float(loss):.6f}"
                )
        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "gpes_state_dict": copy.deepcopy(model.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "history": history,
                "pair_dataset_stats": pair_dataset_stats,
                "config": config,
                "code_commit": commit,
            },
        )
        model.load_state_dict(best_state, strict=True)
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                **(
                    {}
                    if text_only
                    else {"patch_files": config["patch_sha256"]}
                ),
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "casr_model": config["casr_model_sha256"],
                "sdcr_model": config["sdcr_model_sha256"],
                "claude_embeddings": config["claude_embeddings_sha256"],
                "merge_embeddings": config["merge_embeddings_sha256"],
                **(
                    {"snps_model": config["snps_model_sha256"]}
                    if config["schema_version"] in STAGED_SNPS_SCHEMAS
                    else {}
                ),
                **(
                    {"sedps_model": config["sedps_model_sha256"]}
                    if config["schema_version"] in STAGED_SEDPS_SCHEMAS
                    else {}
                ),
            },
        )
        metrics = {
            "experiment_id": config["experiment_id"],
            "idea_id": config["idea_id"],
            "run_id": run_id,
            "code_commit": commit,
            "config_sha256": config_sha,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "parent_metrics_percent": expected_parent,
            "best_metrics_percent": best_metrics,
            "selected_iteration": best_iteration,
            "pair_dataset_stats": pair_dataset_stats,
            "selector_stats": model.stats(),
            "official_test_evaluation_count": len(history) + 1,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", metrics)
        print(metrics)
        return metrics
    finally:
        sys.stdout.flush()
        sys.stdout = old_stdout
        log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.run_id)


if __name__ == "__main__":
    main()
