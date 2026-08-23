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
    GatedPairEvidenceSelector,
    NonlinearGatedPairSelector,
    RoleAwareGatedPairSelector,
    SemanticGatedPairSelector,
    TextOnlyGatedPairSelector,
)
from model.innovations.lpsr import orthogonal_local_text_residuals
from model.innovations.sdcr import SentenceDropoutConservativeRouting
from model.innovations.tigr import taxonomic_suffix_group_ids
from model.innovations.train_agct import derive_train_threshold
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


def hard_margin_only_for_schema(schema: str) -> bool:
    return schema not in (
        "gzsl-paper.gwps.v1",
        "gzsl-paper.bgwps.v1",
        "gzsl-paper.mbgwps.v1",
        "gzsl-paper.nps.v1",
        "gzsl-paper.tgwps.v1",
        "gzsl-paper.sgwps.v1",
        "gzsl-paper.rgwps.v1",
    )


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    schema = config.get("schema_version") if isinstance(config, dict) else None
    if schema in (
        "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1", "gzsl-paper.rgwps.v1"
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
        schema not in (
            "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1", "gzsl-paper.rgwps.v1"
        )
        and config["feature_provenance_complete"] is not False
    ) or config["text_cache_provenance_complete"] is not False:
        raise ValueError("GPES cache provenance边界错误。")
    if (
        (
            schema not in (
                "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1",
                "gzsl-paper.rgwps.v1",
            )
            and (
                int(config["patch_top_k"]) != 2
                or int(config["patch_chunk_size"]) != 16
            )
        )
        or config["group_rule"] != "class_name_last_token_min2"
        or config["threshold_source"] != "train_wrong_same_group_margin"
        or float(config["threshold_quantile"]) != 0.25
        or float(config["margin_temperature"]) != 0.1
        or float(config["max_delta"]) != 0.5
        or int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
        or config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.001
        or float(config["weight_decay"]) != 0.0001
    ):
        raise ValueError("GPES训练参数错误。")
    if schema in (
        "gzsl-paper.gwps.v1", "gzsl-paper.bgwps.v1", "gzsl-paper.mbgwps.v1",
        "gzsl-paper.nps.v1", "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1",
        "gzsl-paper.rgwps.v1",
    ) and config[
        "pair_training_scope"
    ] != "all_same_group_top2_soft_gate":
        raise ValueError("GWPS必须使用全同族top2与soft gate加权。")
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
):
    top = logits.topk(2, dim=1)
    global_ids = ids.index_select(0, top.indices.reshape(-1)).reshape_as(top.indices)
    groups = group_ids.index_select(0, global_ids.reshape(-1).cpu()).reshape_as(
        global_ids.cpu()
    ).to(logits.device)
    same_group = groups[:, 0].eq(groups[:, 1]) & groups[:, 0].ge(0)
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
        values.extend((role_top2[:, 0] - role_top2[:, 1]).unbind(dim=1))
    if patch_scores is not None:
        local_patch = patch_scores.to(logits.device).float()
        if local_patch.shape[1] == 200 and ids.numel() != 200:
            local_patch = local_patch.index_select(1, ids)
        values.append(
            local_patch.gather(1, top.indices)[:, 0]
            - local_patch.gather(1, top.indices)[:, 1]
        )
    features = torch.stack(values, dim=1)
    pair_targets = top.indices[:, 1].eq(targets).long()
    return (
        top.values[selected].detach().cpu(),
        features[selected].detach().cpu(),
        pair_targets[selected].detach().cpu(),
        int(selected.sum()),
        soft_weights[selected].detach().cpu(),
    )


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
    text_only = config["schema_version"] in (
        "gzsl-paper.tgwps.v1", "gzsl-paper.sgwps.v1", "gzsl-paper.rgwps.v1"
    )
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
        threshold, threshold_stats = derive_train_threshold(
            parent, sdrs, calibrator, sdcr, features, labels,
            seen_classes, group_ids, device, float(config["threshold_quantile"])
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
        hard_margin_only = hard_margin_only_for_schema(config["schema_version"])
        for start in range(0, features.shape[0], 512):
            images = features[start : start + 512].to(device).float()
            parent_logits = F.normalize(images, dim=-1) @ parent.prototypes().index_select(0, ids).T * parent.scale()
            parent_logits = sdrs(parent_logits, images, ids)
            parent_logits = calibrator(
                parent_logits, torch.ones(150, dtype=torch.bool, device=device)
            )
            logits = sdcr(parent_logits, images, ids)
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
                    if config["schema_version"] in (
                        "gzsl-paper.sgwps.v1", "gzsl-paper.rgwps.v1"
                    )
                    else None
                ),
                role_prototypes=(
                    F.normalize(sentence8.float(), dim=-1)
                    if config["schema_version"] == "gzsl-paper.rgwps.v1"
                    else None
                ),
            )
            pair_logits_list.append(package[0])
            feature_list.append(package[1])
            target_list.append(package[2])
            pair_weight_list.append(package[4])
        pair_logits = torch.cat(pair_logits_list)
        pair_features = torch.cat(feature_list)
        pair_targets = torch.cat(target_list)
        pair_weights = torch.cat(pair_weight_list)
        if config["schema_version"] in (
            "gzsl-paper.gpes.v1", "gzsl-paper.egpes.v1"
        ):
            pair_weights = torch.ones_like(pair_weights)
        pair_class_weights = torch.ones(2)
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
        if pair_targets.numel() < 50 or pair_targets.unique().numel() != 2:
            raise ValueError("GPES成对训练样本不足或标签退化。")
        feature_mean = pair_features.mean(dim=0)
        feature_std = pair_features.std(dim=0, unbiased=False).clamp_min(1e-6)
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
        }
        if config["schema_version"] == "gzsl-paper.nps.v1":
            model_class = NonlinearGatedPairSelector
        elif config["schema_version"] == "gzsl-paper.tgwps.v1":
            model_class = TextOnlyGatedPairSelector
        elif config["schema_version"] == "gzsl-paper.sgwps.v1":
            model_class = SemanticGatedPairSelector
        elif config["schema_version"] == "gzsl-paper.rgwps.v1":
            model_class = RoleAwareGatedPairSelector
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
        if config["schema_version"] in (
            "gzsl-paper.sgwps.v1", "gzsl-paper.rgwps.v1"
        ):
            model_kwargs["class_name_prototypes"] = names_n
        if config["schema_version"] == "gzsl-paper.rgwps.v1":
            model_kwargs["role_sentence_prototypes"] = sentence8
        model = model_class(**model_kwargs).to(device)
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
            corrected = model.corrected_pair_logits(
                pair_logits.index_select(0, batch).to(device),
                pair_features.index_select(0, batch).to(device),
            )
            per_pair_loss = F.cross_entropy(
                corrected,
                pair_targets.index_select(0, batch).to(device),
                reduction="none",
            )
            batch_weights = pair_weights.index_select(0, batch).to(device)
            loss = (per_pair_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(model)
            optimizer.step()
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
