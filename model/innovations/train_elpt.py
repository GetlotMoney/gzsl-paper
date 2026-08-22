from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.func import functional_call

from model.innovations.elpt import (
    ELPTGate,
    VariableClassTGVPR,
    blend_prototypes,
    fixed_class_folds,
    gate_features,
    semantic_pca_folds,
    topology_loss,
)
from model.innovations.tst import (
    TangentStepGate,
    NeighborhoodResidualGate,
    SummaryResidualGate,
    bidirectional_centroid_contrastive_loss,
    centroid_alignment_loss,
    centroid_contrastive_loss,
    tangent_transport,
)
from model.tg_vpr_h1 import TGVPRH1FixedEqual
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "attempt_id",
    "idea_id",
    "framework_id",
    "base_config",
    "base_checkpoint",
    "base_checkpoint_sha256",
    "seed",
    "fold_count",
    "fold_epochs",
    "gate_epochs",
    "gate_batch_half",
    "gate_lr",
    "gate_weight_decay",
    "topology_weight",
}
OPTIONAL_CONFIG_KEYS = {
    "gate_max_alpha",
    "alpha_penalty",
    "fold_checkpoint_dir",
    "gate_feature_mode",
    "gate_ensemble",
    "transport_mode",
    "gate_max_step",
    "centroid_alignment_weight",
    "parent_metrics_percent",
    "centroid_alignment_mode",
    "gate_initialization_ensemble",
    "pseudo_unseen_ce_weight",
    "pseudo_unseen_loss_mode",
    "gate_architecture",
    "parent_gate_model",
    "parent_gate_model_sha256",
    "max_residual_step",
    "gate_training_mode",
    "inner_lr",
    "meta_gradient_mode",
    "seen_gradient_weight",
    "fold_strategy",
    "gradient_normalization",
}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class FrozenPrototypeClassifier(nn.Module):
    def __init__(self, prototypes: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("_prototypes", prototypes.detach())
        self.register_buffer("_scale", scale.detach())

    def prototypes(self):
        return self._prototypes

    def scale(self):
        return self._scale


def load_config(path: Path):
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ELPT配置必须是字典。")
    missing = REQUIRED_CONFIG_KEYS - set(raw)
    extra = set(raw) - REQUIRED_CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
    if missing or extra:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(
            f"ELPT配置字段错误；缺少={sorted(missing)}，多出={sorted(extra)}。"
        )
    raw.setdefault("gate_max_alpha", 1.0)
    raw.setdefault("alpha_penalty", 0.0)
    raw.setdefault("fold_checkpoint_dir", None)
    raw.setdefault("gate_feature_mode", "summary")
    raw.setdefault("gate_ensemble", False)
    raw.setdefault("transport_mode", "convex_blend")
    raw.setdefault("gate_max_step", 1.5)
    raw.setdefault("centroid_alignment_weight", 0.0)
    raw.setdefault("parent_metrics_percent", None)
    raw.setdefault("centroid_alignment_mode", "pairwise")
    raw.setdefault("gate_initialization_ensemble", 1)
    raw.setdefault("pseudo_unseen_ce_weight", 0.0)
    raw.setdefault("pseudo_unseen_loss_mode", "cross_entropy")
    raw.setdefault("gate_architecture", "direct")
    raw.setdefault("parent_gate_model", None)
    raw.setdefault("parent_gate_model_sha256", None)
    raw.setdefault("max_residual_step", 0.1)
    raw.setdefault("gate_training_mode", "joint_balanced_ce")
    raw.setdefault("inner_lr", 0.01)
    raw.setdefault("meta_gradient_mode", "outer_only")
    raw.setdefault("seen_gradient_weight", 0.0)
    raw.setdefault("fold_strategy", "fixed_mod3")
    raw.setdefault("gradient_normalization", "none")
    if raw["schema_version"] not in (
        "gzsl-paper.elpt.v1",
        "gzsl-paper.tst.v1",
        "gzsl-paper.cata.v1",
        "gzsl-paper.cata.v2",
        "gzsl-paper.cata.v3",
        "gzsl-paper.cata.v4",
        "gzsl-paper.purl.v1",
        "gzsl-paper.purl.v2",
        "gzsl-paper.ntr.v1",
        "gzsl-paper.ntr-residual.v1",
        "gzsl-paper.ntr-dispersion.v1",
        "gzsl-paper.bmr.v1",
        "gzsl-paper.pgo.v1",
        "gzsl-paper.pgo.v2",
    ):
        raise ValueError("ELPT schema错误。")
    valid_elpt = raw["attempt_id"] in {"V2-TRY-006", "V2-TRY-007", "V2-TRY-008", "V2-TRY-009"} and raw["idea_id"] == "IDEA-002"
    valid_tst = raw["attempt_id"] in {
        "V2-TRY-015",
        "V2-TRY-016",
        "V2-TRY-017",
        "V2-TRY-018",
    } and raw["idea_id"] == "IDEA-005"
    valid_cata = raw["attempt_id"] in {
        "V2-TRY-021",
        "V2-TRY-022",
        "V2-TRY-023",
        "V2-TRY-024",
    } and raw["idea_id"] == "IDEA-007"
    valid_purl = raw["attempt_id"] in {"V2-TRY-026", "V2-TRY-027"} and raw["idea_id"] == "IDEA-009"
    valid_ntr = raw["attempt_id"] in {
        "V2-TRY-028",
        "V2-TRY-029",
        "V2-TRY-030",
        "V2-TRY-031",
        "V2-TRY-032",
        "V2-TRY-033",
        "V2-TRY-034",
        "V2-TRY-035",
        "V2-TRY-036",
    } and raw["idea_id"] == "IDEA-010"
    valid_bmr = raw["attempt_id"] in {
        "V2-TRY-037",
        "V2-TRY-038",
        "V2-TRY-039",
        "V2-TRY-040",
    } and raw["idea_id"] == "IDEA-011"
    valid_pgo = raw["attempt_id"] in {"V2-TRY-048", "V2-TRY-049"} and raw["idea_id"] == "IDEA-015"
    if not (valid_elpt or valid_tst or valid_cata or valid_purl or valid_ntr or valid_bmr or valid_pgo):
        raise ValueError("ELPT首次TRY身份不匹配。")
    if raw["framework_id"] != "FRAMEWORK-V2":
        raise ValueError("ELPT只接受FRAMEWORK-V2。")
    if int(raw["fold_count"]) != 3 or int(raw["fold_epochs"]) != 50:
        raise ValueError("ELPT固定3折和50轮fold训练。")
    if int(raw["gate_epochs"]) != 20 or int(raw["gate_batch_half"]) != 32:
        raise ValueError("ELPT固定20轮gate训练和32/32平衡batch。")
    if float(raw["gate_lr"]) != 1e-3 or float(raw["gate_weight_decay"]) != 1e-4:
        raise ValueError("ELPT gate优化器参数不匹配。")
    if float(raw["topology_weight"]) != 0.1:
        raise ValueError("ELPT固定topology_weight=0.1。")
    if raw["attempt_id"] == "V2-TRY-006" and (
        float(raw["gate_max_alpha"]) != 1.0 or float(raw["alpha_penalty"]) != 0.0
    ):
        raise ValueError("TRY-006必须使用无上限补救的初始ELPT。")
    if raw["attempt_id"] == "V2-TRY-007" and (
        float(raw["gate_max_alpha"]) != 0.25 or float(raw["alpha_penalty"]) != 0.01
    ):
        raise ValueError("TRY-007必须使用0.25上限和0.01 alpha约束。")
    if raw["attempt_id"] == "V2-TRY-007" and not raw["fold_checkpoint_dir"]:
        raise ValueError("TRY-007必须复用TRY-006 fold checkpoint。")
    if raw["attempt_id"] == "V2-TRY-008" and (
        float(raw["gate_max_alpha"]) != 0.25
        or float(raw["alpha_penalty"]) != 0.01
        or raw["gate_feature_mode"] != "top5_vector"
        or not raw["fold_checkpoint_dir"]
    ):
        raise ValueError("TRY-008必须使用受约束top5向量gate并复用fold checkpoint。")
    if raw["attempt_id"] == "V2-TRY-009" and (
        float(raw["gate_max_alpha"]) != 0.25
        or float(raw["alpha_penalty"]) != 0.01
        or raw["gate_feature_mode"] != "top5_vector"
        or raw["gate_ensemble"] is not True
        or not raw["fold_checkpoint_dir"]
    ):
        raise ValueError("TRY-009必须使用三折独立gate ensemble。")
    if valid_tst and (
        raw["transport_mode"] != "tangent"
        or float(raw["gate_max_step"]) != 1.5
        or raw["gate_feature_mode"] != "summary"
        or raw["gate_ensemble"] is not False
    ):
        raise ValueError("TST必须使用冻结的切空间步长gate结构。")
    if raw["attempt_id"] == "V2-TRY-015" and not raw["fold_checkpoint_dir"]:
        raise ValueError("TRY-015必须复用seed7 ELPT fold checkpoint。")
    if raw["attempt_id"] in {"V2-TRY-016", "V2-TRY-017", "V2-TRY-018"} and raw["fold_checkpoint_dir"] is not None:
        raise ValueError("TST多seed RUN必须从头训练各自fold权重。")
    if valid_cata:
        parent = raw["parent_metrics_percent"]
        if (
            raw["transport_mode"] != "tangent"
            or float(raw["gate_max_step"]) != 1.5
            or float(raw["centroid_alignment_weight"]) != 0.1
            or raw["gate_feature_mode"] != "summary"
            or raw["gate_ensemble"] is not False
            or not isinstance(parent, dict)
            or set(parent) != {"U", "S", "H", "ZS"}
        ):
            raise ValueError("CATA首次TRY身份或父指标不匹配。")
        expected_mode = {
            "V2-TRY-021": "pairwise",
            "V2-TRY-022": "contrastive",
            "V2-TRY-023": "bidirectional_contrastive",
            "V2-TRY-024": "contrastive",
        }[raw["attempt_id"]]
        if raw["centroid_alignment_mode"] != expected_mode:
            raise ValueError("CATA对齐模式与TRY身份不匹配。")
        expected_ensemble = 3 if raw["attempt_id"] == "V2-TRY-024" else 1
        if int(raw["gate_initialization_ensemble"]) != expected_ensemble:
            raise ValueError("CATA初始化ensemble与TRY身份不匹配。")
    if valid_purl:
        parent = raw["parent_metrics_percent"]
        if (
            raw["transport_mode"] != "tangent"
            or float(raw["gate_max_step"]) != 1.5
            or float(raw["pseudo_unseen_ce_weight"]) != 1.0
            or float(raw["centroid_alignment_weight"]) != 0.0
            or raw["gate_feature_mode"] != "summary"
            or raw["gate_ensemble"] is not False
            or int(raw["gate_initialization_ensemble"]) != 1
            or not raw["fold_checkpoint_dir"]
            or not isinstance(parent, dict)
            or set(parent) != {"U", "S", "H", "ZS"}
        ):
            raise ValueError("PURL首次TRY身份不匹配。")
        expected_mode = "cross_entropy" if raw["attempt_id"] == "V2-TRY-026" else "focal_gamma2"
        if raw["pseudo_unseen_loss_mode"] != expected_mode:
            raise ValueError("PURL风险模式与TRY身份不匹配。")
    if valid_ntr:
        parent = raw["parent_metrics_percent"]
        if (
            raw["transport_mode"] != "tangent"
            or float(raw["gate_max_step"]) != 1.5
            or raw["gate_feature_mode"]
            != ("summary_std" if raw["attempt_id"] == "V2-TRY-036" else "top5_vector")
            or raw["gate_ensemble"] is not False
            or int(raw["gate_initialization_ensemble"]) != 1
            or float(raw["centroid_alignment_weight"]) != 0.0
            or float(raw["pseudo_unseen_ce_weight"]) != 0.0
            or not isinstance(parent, dict)
            or set(parent) != {"U", "S", "H", "ZS"}
        ):
            raise ValueError("NTR首次TRY身份不匹配。")
        if raw["attempt_id"] == "V2-TRY-028" and not raw["fold_checkpoint_dir"]:
            raise ValueError("NTR seed7必须复用对应fold权重。")
        if raw["attempt_id"] in {"V2-TRY-029", "V2-TRY-030", "V2-TRY-031"} and raw["fold_checkpoint_dir"] is not None:
            raise ValueError("NTR多seed必须从头训练各自fold权重。")
        if raw["attempt_id"] in {"V2-TRY-032", "V2-TRY-033", "V2-TRY-034", "V2-TRY-035"} and (
            raw["gate_architecture"] != "neighborhood_residual"
            or not raw["parent_gate_model"]
            or not raw["parent_gate_model_sha256"]
            or float(raw["max_residual_step"]) != 0.1
            or not raw["fold_checkpoint_dir"]
        ):
            raise ValueError("NTR补救1必须使用冻结TST gate的邻域残差。")
        if raw["attempt_id"] == "V2-TRY-036" and (
            raw["gate_architecture"] != "direct" or not raw["fold_checkpoint_dir"]
        ):
            raise ValueError("NTR补救2必须使用5维离散度gate并复用fold权重。")
    if valid_bmr:
        parent = raw["parent_metrics_percent"]
        if (
            raw["transport_mode"] != "tangent"
            or raw["gate_feature_mode"] != "summary"
            or raw["gate_architecture"]
            != (
                "bilevel_residual"
                if raw["attempt_id"] in {"V2-TRY-039", "V2-TRY-040"}
                else "direct"
            )
            or raw["gate_training_mode"] != "bilevel_first_order"
            or float(raw["inner_lr"]) != 0.01
            or float(raw["centroid_alignment_weight"]) != 0.0
            or float(raw["pseudo_unseen_ce_weight"]) != 0.0
            or (
                raw["attempt_id"] != "V2-TRY-040"
                and not raw["fold_checkpoint_dir"]
            )
            or not isinstance(parent, dict)
            or set(parent) != {"U", "S", "H", "ZS"}
        ):
            raise ValueError("BMR首次TRY身份不匹配。")
        expected_mode = (
            "pcgrad_seen_outer" if raw["attempt_id"] == "V2-TRY-038" else "outer_only"
        )
        expected_seen_weight = 1.0 if raw["attempt_id"] == "V2-TRY-038" else 0.0
        if (
            raw["meta_gradient_mode"] != expected_mode
            or float(raw["seen_gradient_weight"]) != expected_seen_weight
        ):
            raise ValueError("BMR梯度合并模式与TRY身份不匹配。")
        if raw["attempt_id"] in {"V2-TRY-039", "V2-TRY-040"} and (
            raw["gate_architecture"] != "bilevel_residual"
            or not raw["parent_gate_model"]
            or not raw["parent_gate_model_sha256"]
            or float(raw["max_residual_step"]) != 0.1
        ):
            raise ValueError("BMR补救2必须使用冻结TST Gate的有界残差。")
        expected_fold_strategy = (
            "semantic_pca_blocks"
            if raw["attempt_id"] == "V2-TRY-040"
            else "fixed_mod3"
        )
        if raw["fold_strategy"] != expected_fold_strategy:
            raise ValueError("BMR fold策略与TRY身份不匹配。")
        if raw["attempt_id"] == "V2-TRY-040" and raw["fold_checkpoint_dir"] is not None:
            raise ValueError("BMR困难折必须从头训练fold权重。")
    if valid_pgo:
        parent = raw["parent_metrics_percent"]
        if (
            raw["transport_mode"] != "tangent"
            or raw["gate_feature_mode"] != "summary"
            or raw["gate_architecture"] != "bilevel_residual"
            or raw["gate_training_mode"] != "pcgrad_joint_residual"
            or not raw["parent_gate_model"]
            or not raw["parent_gate_model_sha256"]
            or float(raw["max_residual_step"]) != 0.1
            or not raw["fold_checkpoint_dir"]
            or not isinstance(parent, dict)
            or set(parent) != {"U", "S", "H", "ZS"}
        ):
            raise ValueError("PGO首次TRY身份不匹配。")
        expected_normalization = "none" if raw["attempt_id"] == "V2-TRY-048" else "unit_global_norm"
        if raw["gradient_normalization"] != expected_normalization:
            raise ValueError("PGO梯度归一化模式与TRY身份不匹配。")
    return raw, sha256_file(path)


def _make_gate(config, input_dim, device):
    if config["gate_architecture"] == "bilevel_residual":
        path = Path(config["parent_gate_model"])
        if sha256_file(path) != config["parent_gate_model_sha256"]:
            raise ValueError("BMR父TST gate SHA不匹配。")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        base_gate = TangentStepGate(input_dim=4, max_step=1.5)
        base_gate.load_state_dict(payload["gate_state_dict"], strict=True)
        return SummaryResidualGate(
            base_gate, max_delta=float(config["max_residual_step"])
        ).to(device)
    if config["gate_architecture"] == "neighborhood_residual":
        path = Path(config["parent_gate_model"])
        if sha256_file(path) != config["parent_gate_model_sha256"]:
            raise ValueError("NTR父TST gate SHA不匹配。")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        base_gate = TangentStepGate(input_dim=4, max_step=1.5)
        base_gate.load_state_dict(payload["gate_state_dict"], strict=True)
        return NeighborhoodResidualGate(
            base_gate, max_delta=float(config["max_residual_step"])
        ).to(device)
    if config["transport_mode"] == "tangent":
        return TangentStepGate(
            input_dim=input_dim,
            max_step=float(config["gate_max_step"]),
        ).to(device)
    return ELPTGate(
        input_dim=input_dim,
        max_alpha=float(config["gate_max_alpha"]),
    ).to(device)


def _gate_input_dim(mode):
    return {"summary": 4, "summary_std": 5, "top5_vector": 8}[mode]


def _transport(base, value, coefficient, mode):
    if mode == "tangent":
        return tangent_transport(base, value, coefficient)
    return blend_prototypes(base, value, coefficient)


def _centroid_loss(prototypes, centroids, mode):
    if mode == "bidirectional_contrastive":
        return bidirectional_centroid_contrastive_loss(prototypes, centroids)
    if mode == "contrastive":
        return centroid_contrastive_loss(prototypes, centroids)
    return centroid_alignment_loss(prototypes, centroids)


def _pseudo_unseen_risk(logits, targets, mode):
    if mode == "focal_gamma2":
        log_probability = F.log_softmax(logits, dim=-1)
        log_correct = log_probability.gather(1, targets.unsqueeze(1)).squeeze(1)
        correct_probability = log_correct.exp()
        return (-((1.0 - correct_probability).square()) * log_correct).mean()
    return F.cross_entropy(logits, targets)


def _stage_boundaries(stages):
    total = 0
    boundaries = []
    for stage in stages:
        total += int(stage["epochs"])
        boundaries.append(total)
    return boundaries


def _train_fold(
    fold_id,
    pseudo_seen,
    sentence_embeds,
    train_features,
    train_labels,
    base_config,
    device,
    output_dir,
    seed,
    print_log,
):
    mask = torch.isin(train_labels.long(), pseudo_seen)
    features = train_features[mask]
    labels = train_labels.long()[mask]
    centroids = h1.visual_centroids(train_features, train_labels.long(), pseudo_seen)
    model = VariableClassTGVPR(
        sentence_embeds,
        pseudo_seen,
        centroids,
        dropout=base_config["dropout"],
        inner_ratio=base_config["inner_ratio"],
        outer_ratio=base_config["outer_ratio"],
        temperature=base_config["temperature"],
    ).to(device)
    stages = base_config["lr_stages"]
    boundaries = _stage_boundaries(stages)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(stages[0]["lr"]),
        weight_decay=float(base_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(stages[0]["epochs"]),
        eta_min=float(stages[0]["eta_min"]),
    )
    active_stage = 0
    label_map = torch.full((200,), -1, dtype=torch.long)
    label_map[pseudo_seen] = torch.arange(pseudo_seen.numel())
    generator = torch.Generator(device="cpu").manual_seed(seed * 100 + fold_id)
    iters = labels.numel() // int(base_config["batch_size"])
    for epoch in range(1, 51):
        target_stage = next(i for i, boundary in enumerate(boundaries) if epoch <= boundary)
        if target_stage != active_stage:
            active_stage = target_stage
            stage = stages[active_stage]
            for group in optimizer.param_groups:
                group["lr"] = float(stage["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(stage["epochs"]),
                eta_min=float(stage["eta_min"]),
            )
        model.train()
        loss_sum = 0.0
        for _ in range(iters):
            indices = torch.randperm(labels.numel(), generator=generator)[: int(base_config["batch_size"])]
            batch = features[indices].to(device).float()
            targets = label_map[labels[indices]].to(device)
            optimizer.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model.logits(batch, pseudo_seen), targets)
            topo = model.topology_loss()
            loss = ce + float(base_config["topology_weight"]) * topo
            if not torch.isfinite(loss):
                raise FloatingPointError(f"fold {fold_id} loss非有限。")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        scheduler.step()
        if epoch in (1, 10, 20, 30, 40, 50):
            print_log(f"fold={fold_id} epoch={epoch} loss={loss_sum/iters:.6f}")
    model.eval()
    checkpoint = {
        "fold_id": fold_id,
        "seed": seed,
        "pseudo_seen": pseudo_seen.tolist(),
        "model_state_dict": copy.deepcopy(model.state_dict()),
    }
    torch.save(checkpoint, output_dir / f"fold_{fold_id}.pth")
    return model


def _load_fold_checkpoint(
    fold_id,
    pseudo_seen,
    sentence_embeds,
    train_features,
    train_labels,
    base_config,
    device,
    checkpoint_dir,
):
    path = Path(checkpoint_dir) / f"fold_{fold_id}.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("fold_id") != fold_id or checkpoint.get("pseudo_seen") != pseudo_seen.tolist():
        raise ValueError(f"fold {fold_id} checkpoint身份不匹配。")
    centroids = h1.visual_centroids(train_features, train_labels.long(), pseudo_seen)
    model = VariableClassTGVPR(
        sentence_embeds,
        pseudo_seen,
        centroids,
        dropout=base_config["dropout"],
        inner_ratio=base_config["inner_ratio"],
        outer_ratio=base_config["outer_ratio"],
        temperature=base_config["temperature"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _fold_package(
    model,
    pseudo_seen,
    pseudo_unseen,
    tensors,
    seenclasses,
    device,
    feature_mode,
):
    model.eval()
    with torch.no_grad():
        base_all = model.base_prototypes()
        fold_full = model.prototypes()
        value = model.value_candidate(pseudo_unseen.to(device))
        features = gate_features(
            base_all.index_select(0, pseudo_unseen.to(device)),
            value,
            base_all.index_select(0, pseudo_seen.to(device)),
            mode=feature_mode,
        )
    seen_sample_mask = torch.isin(tensors["train_labels"].long(), pseudo_seen)
    unseen_sample_mask = torch.isin(tensors["train_labels"].long(), pseudo_unseen)
    return {
        "pseudo_seen": pseudo_seen,
        "pseudo_unseen": pseudo_unseen,
        "base_all": base_all.detach().cpu(),
        "fold_full": fold_full.detach().cpu(),
        "value": value.detach().cpu(),
        "gate_features": features.detach().cpu(),
        "seen_indices": seen_sample_mask.nonzero(as_tuple=False).flatten(),
        "unseen_indices": unseen_sample_mask.nonzero(as_tuple=False).flatten(),
        "scale": model.scale().detach().cpu(),
        "seenclasses": seenclasses,
        "pseudo_unseen_centroids": h1.visual_centroids(
            tensors["train_features"],
            tensors["train_labels"].long(),
            pseudo_unseen,
        ),
    }


def _train_gate(packages, tensors, config, device, seed, print_log):
    if config["gate_training_mode"] == "pcgrad_joint_residual":
        return _train_gate_pcgrad_joint(packages, tensors, config, device, seed, print_log)
    if config["gate_training_mode"] == "bilevel_first_order":
        return _train_gate_bilevel(packages, tensors, config, device, seed, print_log)
    if int(config["gate_initialization_ensemble"]) > 1:
        return _train_gate_initialization_ensemble(
            packages, tensors, config, device, seed, print_log
        )
    if config["gate_ensemble"]:
        return _train_gate_ensemble(packages, tensors, config, device, seed, print_log)
    input_dim = _gate_input_dim(config["gate_feature_mode"])
    gate = _make_gate(config, input_dim, device)
    optimizer = torch.optim.Adam(
        gate.parameters(),
        lr=float(config["gate_lr"]),
        weight_decay=float(config["gate_weight_decay"]),
    )
    label_map = torch.full((200,), -1, dtype=torch.long)
    seenclasses = packages[0]["seenclasses"]
    label_map[seenclasses] = torch.arange(150)
    generators = [
        torch.Generator(device="cpu").manual_seed(seed * 1000 + fold_id)
        for fold_id in range(3)
    ]
    half = int(config["gate_batch_half"])
    for epoch in range(1, int(config["gate_epochs"]) + 1):
        gate.train()
        total_loss = 0.0
        total_steps = 0
        for fold_id, package in enumerate(packages):
            seen_indices = package["seen_indices"]
            unseen_indices = package["unseen_indices"]
            steps = min(seen_indices.numel() // half, unseen_indices.numel() // half)
            for _ in range(steps):
                generator = generators[fold_id]
                si = seen_indices[torch.randperm(seen_indices.numel(), generator=generator)[:half]]
                ui = unseen_indices[torch.randperm(unseen_indices.numel(), generator=generator)[:half]]
                indices = torch.cat((si, ui))
                images = tensors["train_features"][indices].to(device).float()
                targets = label_map[tensors["train_labels"].long()[indices]].to(device)
                base_all = package["base_all"].to(device)
                final_all = package["fold_full"].to(device).clone()
                features = package["gate_features"].to(device)
                alpha = gate(features)
                pseudo_unseen = package["pseudo_unseen"].to(device)
                final_all[pseudo_unseen] = _transport(
                    base_all.index_select(0, pseudo_unseen),
                    package["value"].to(device),
                    alpha,
                    config["transport_mode"],
                )
                competition = final_all.index_select(0, seenclasses.to(device))
                logits = F.normalize(images, dim=-1) @ competition.T * package["scale"].to(device)
                ce = F.cross_entropy(logits, targets)
                pseudo_unseen_mask = torch.isin(
                    tensors["train_labels"].long()[indices],
                    package["pseudo_unseen"],
                ).to(device)
                pseudo_unseen_ce = _pseudo_unseen_risk(
                    logits[pseudo_unseen_mask],
                    targets[pseudo_unseen_mask],
                    config["pseudo_unseen_loss_mode"],
                )
                topo = topology_loss(
                    base_all.index_select(0, seenclasses.to(device)), competition
                )
                alpha_regularization = alpha.square().mean()
                alignment = _centroid_loss(
                    final_all.index_select(0, pseudo_unseen),
                    package["pseudo_unseen_centroids"].to(device),
                    config["centroid_alignment_mode"],
                )
                loss = (
                    ce
                    + float(config["topology_weight"]) * topo
                    + float(config["alpha_penalty"]) * alpha_regularization
                    + float(config["centroid_alignment_weight"]) * alignment
                    + float(config["pseudo_unseen_ce_weight"])
                    * pseudo_unseen_ce
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("ELPT gate loss非有限。")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in gate.parameters()
                ):
                    raise FloatingPointError("ELPT gate梯度非有限。")
                optimizer.step()
                total_loss += float(loss.detach())
                total_steps += 1
        print_log(f"gate_epoch={epoch} loss={total_loss/total_steps:.6f}")
    return gate.eval()


def _first_order_adapted_parameters(gate, inner_loss, inner_lr):
    parameters = {
        name: parameter
        for name, parameter in gate.named_parameters()
        if parameter.requires_grad
    }
    gradients = torch.autograd.grad(
        inner_loss, tuple(parameters.values()), create_graph=False
    )
    return {
        name: parameter - float(inner_lr) * gradient.detach()
        for (name, parameter), gradient in zip(parameters.items(), gradients)
    }


def _pcgrad_merge(primary_gradients, anchor_gradients, anchor_weight=1.0):
    dot = sum(
        (primary * anchor).sum()
        for primary, anchor in zip(primary_gradients, anchor_gradients)
    )
    anchor_norm = sum(anchor.square().sum() for anchor in anchor_gradients).clamp_min(1e-12)
    conflict = bool(float(dot.detach()) < 0.0)
    if conflict:
        primary_gradients = tuple(
            primary - dot / anchor_norm * anchor
            for primary, anchor in zip(primary_gradients, anchor_gradients)
        )
    merged = tuple(
        primary + float(anchor_weight) * anchor
        for primary, anchor in zip(primary_gradients, anchor_gradients)
    )
    return merged, conflict


def _symmetric_pcgrad_merge(first_gradients, second_gradients, normalize=False):
    if normalize:
        first_norm = torch.sqrt(
            sum(first.square().sum() for first in first_gradients).clamp_min(1e-12)
        )
        second_norm = torch.sqrt(
            sum(second.square().sum() for second in second_gradients).clamp_min(1e-12)
        )
        first_gradients = tuple(first / first_norm for first in first_gradients)
        second_gradients = tuple(second / second_norm for second in second_gradients)
    dot = sum(
        (first * second).sum()
        for first, second in zip(first_gradients, second_gradients)
    )
    conflict = bool(float(dot.detach()) < 0.0)
    if conflict:
        first_norm = sum(first.square().sum() for first in first_gradients).clamp_min(1e-12)
        second_norm = sum(second.square().sum() for second in second_gradients).clamp_min(1e-12)
        projected_first = tuple(
            first - dot / second_norm * second
            for first, second in zip(first_gradients, second_gradients)
        )
        projected_second = tuple(
            second - dot / first_norm * first
            for first, second in zip(first_gradients, second_gradients)
        )
        first_gradients, second_gradients = projected_first, projected_second
    return tuple(
        0.5 * (first + second)
        for first, second in zip(first_gradients, second_gradients)
    ), conflict


def _train_gate_pcgrad_joint(packages, tensors, config, device, seed, print_log):
    gate = _make_gate(config, _gate_input_dim(config["gate_feature_mode"]), device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in gate.parameters() if parameter.requires_grad),
        lr=float(config["gate_lr"]),
        weight_decay=float(config["gate_weight_decay"]),
    )
    seenclasses = packages[0]["seenclasses"]
    label_map = torch.full((200,), -1, dtype=torch.long)
    label_map[seenclasses] = torch.arange(150)
    half = int(config["gate_batch_half"])
    generators = [
        torch.Generator(device="cpu").manual_seed(seed * 15000 + fold_id)
        for fold_id in range(3)
    ]
    for epoch in range(1, int(config["gate_epochs"]) + 1):
        seen_sum = unseen_sum = 0.0
        step_count = conflict_count = 0
        for fold_id, package in enumerate(packages):
            seen_indices = package["seen_indices"]
            unseen_indices = package["unseen_indices"]
            steps = min(seen_indices.numel() // half, unseen_indices.numel() // half)
            for _ in range(steps):
                generator = generators[fold_id]
                si = seen_indices[torch.randperm(seen_indices.numel(), generator=generator)[:half]]
                ui = unseen_indices[torch.randperm(unseen_indices.numel(), generator=generator)[:half]]
                seen_images = tensors["train_features"][si].to(device).float()
                unseen_images = tensors["train_features"][ui].to(device).float()
                seen_targets = label_map[tensors["train_labels"].long()[si]].to(device)
                unseen_targets = label_map[tensors["train_labels"].long()[ui]].to(device)
                named_parameters = tuple(
                    (name, parameter)
                    for name, parameter in gate.named_parameters()
                    if parameter.requires_grad
                )
                parameter_dict = dict(named_parameters)
                seen_logits, base_seen, competition_seen = _bilevel_logits(
                    package, gate, parameter_dict, seen_images, seenclasses, device, config
                )
                unseen_logits, base_unseen, competition_unseen = _bilevel_logits(
                    package, gate, parameter_dict, unseen_images, seenclasses, device, config
                )
                seen_loss = F.cross_entropy(seen_logits, seen_targets) + float(
                    config["topology_weight"]
                ) * topology_loss(
                    base_seen.index_select(0, seenclasses.to(device)), competition_seen
                )
                unseen_loss = F.cross_entropy(unseen_logits, unseen_targets) + float(
                    config["topology_weight"]
                ) * topology_loss(
                    base_unseen.index_select(0, seenclasses.to(device)), competition_unseen
                )
                parameters = tuple(parameter for _, parameter in named_parameters)
                seen_gradients = torch.autograd.grad(seen_loss, parameters)
                unseen_gradients = torch.autograd.grad(unseen_loss, parameters)
                merged, conflict = _symmetric_pcgrad_merge(
                    seen_gradients,
                    unseen_gradients,
                    normalize=(config["gradient_normalization"] == "unit_global_norm"),
                )
                optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(parameters, merged):
                    parameter.grad = gradient.detach()
                optimizer.step()
                seen_sum += float(seen_loss.detach())
                unseen_sum += float(unseen_loss.detach())
                step_count += 1
                conflict_count += int(conflict)
        print_log(
            f"pgo_epoch={epoch} seen_loss={seen_sum/step_count:.6f} "
            f"unseen_loss={unseen_sum/step_count:.6f} "
            f"conflict_rate={conflict_count/step_count:.6f}"
        )
    return gate.eval()


def _bilevel_logits(package, gate, gate_parameters, images, seenclasses, device, config):
    base_all = package["base_all"].to(device)
    final_all = package["fold_full"].to(device).clone()
    coefficient = functional_call(
        gate, gate_parameters, (package["gate_features"].to(device),)
    )
    pseudo_unseen = package["pseudo_unseen"].to(device)
    final_all[pseudo_unseen] = _transport(
        base_all.index_select(0, pseudo_unseen),
        package["value"].to(device),
        coefficient,
        config["transport_mode"],
    )
    competition = final_all.index_select(0, seenclasses.to(device))
    logits = F.normalize(images, dim=-1) @ competition.T * package["scale"].to(device)
    return logits, base_all, competition


def _train_gate_bilevel(packages, tensors, config, device, seed, print_log):
    gate = _make_gate(config, _gate_input_dim(config["gate_feature_mode"]), device)
    optimizer = torch.optim.Adam(
        gate.parameters(),
        lr=float(config["gate_lr"]),
        weight_decay=float(config["gate_weight_decay"]),
    )
    seenclasses = packages[0]["seenclasses"]
    label_map = torch.full((200,), -1, dtype=torch.long)
    label_map[seenclasses] = torch.arange(150)
    half = int(config["gate_batch_half"])
    generators = [
        torch.Generator(device="cpu").manual_seed(seed * 11000 + fold_id)
        for fold_id in range(3)
    ]
    for epoch in range(1, int(config["gate_epochs"]) + 1):
        gate.train()
        inner_sum = outer_sum = 0.0
        step_count = conflict_count = 0
        for fold_id, package in enumerate(packages):
            seen_indices = package["seen_indices"]
            unseen_indices = package["unseen_indices"]
            steps = min(seen_indices.numel() // half, unseen_indices.numel() // half)
            for _ in range(steps):
                generator = generators[fold_id]
                inner_indices = seen_indices[
                    torch.randperm(seen_indices.numel(), generator=generator)[:half]
                ]
                outer_indices = unseen_indices[
                    torch.randperm(unseen_indices.numel(), generator=generator)[:half]
                ]
                inner_images = tensors["train_features"][inner_indices].to(device).float()
                outer_images = tensors["train_features"][outer_indices].to(device).float()
                inner_targets = label_map[
                    tensors["train_labels"].long()[inner_indices]
                ].to(device)
                outer_targets = label_map[
                    tensors["train_labels"].long()[outer_indices]
                ].to(device)
                parameters = {
                    name: parameter
                    for name, parameter in gate.named_parameters()
                    if parameter.requires_grad
                }
                inner_logits, _, _ = _bilevel_logits(
                    package,
                    gate,
                    parameters,
                    inner_images,
                    seenclasses,
                    device,
                    config,
                )
                inner_loss = F.cross_entropy(inner_logits, inner_targets)
                adapted = _first_order_adapted_parameters(
                    gate, inner_loss, config["inner_lr"]
                )
                outer_logits, base_all, competition = _bilevel_logits(
                    package,
                    gate,
                    adapted,
                    outer_images,
                    seenclasses,
                    device,
                    config,
                )
                outer_ce = F.cross_entropy(outer_logits, outer_targets)
                topology = topology_loss(
                    base_all.index_select(0, seenclasses.to(device)), competition
                )
                outer_loss = outer_ce + float(config["topology_weight"]) * topology
                if not torch.isfinite(inner_loss) or not torch.isfinite(outer_loss):
                    raise FloatingPointError("BMR内外层loss非有限。")
                optimizer.zero_grad(set_to_none=True)
                if config["meta_gradient_mode"] == "pcgrad_seen_outer":
                    named_parameters = tuple(
                        (name, parameter)
                        for name, parameter in gate.named_parameters()
                        if parameter.requires_grad
                    )
                    outer_gradients = torch.autograd.grad(
                        outer_loss,
                        tuple(parameter for _, parameter in named_parameters),
                    )
                    retention_logits, _, _ = _bilevel_logits(
                        package,
                        gate,
                        dict(named_parameters),
                        inner_images,
                        seenclasses,
                        device,
                        config,
                    )
                    retention_loss = F.cross_entropy(retention_logits, inner_targets)
                    seen_gradients = torch.autograd.grad(
                        retention_loss,
                        tuple(parameter for _, parameter in named_parameters),
                    )
                    merged, conflict = _pcgrad_merge(
                        outer_gradients,
                        seen_gradients,
                        config["seen_gradient_weight"],
                    )
                    for (_, parameter), gradient in zip(named_parameters, merged):
                        parameter.grad = gradient.detach()
                    conflict_count += int(conflict)
                else:
                    outer_loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in gate.parameters()
                ):
                    raise FloatingPointError("BMR gate梯度非有限。")
                optimizer.step()
                inner_sum += float(inner_loss.detach())
                outer_sum += float(outer_loss.detach())
                step_count += 1
        print_log(
            f"bmr_epoch={epoch} inner_ce={inner_sum/step_count:.6f} "
            f"outer_loss={outer_sum/step_count:.6f} "
            f"conflict_rate={conflict_count/step_count:.6f}"
        )
    return gate.eval()


def _train_gate_initialization_ensemble(
    packages, tensors, config, device, seed, print_log
):
    gates = []
    label_map = torch.full((200,), -1, dtype=torch.long)
    seenclasses = packages[0]["seenclasses"]
    label_map[seenclasses] = torch.arange(150)
    half = int(config["gate_batch_half"])
    input_dim = _gate_input_dim(config["gate_feature_mode"])
    for member in range(int(config["gate_initialization_ensemble"])):
        torch.manual_seed(seed * 100 + member)
        gate = _make_gate(config, input_dim, device)
        optimizer = torch.optim.Adam(
            gate.parameters(),
            lr=float(config["gate_lr"]),
            weight_decay=float(config["gate_weight_decay"]),
        )
        generators = [
            torch.Generator(device="cpu").manual_seed(
                seed * 10000 + member * 1000 + fold_id
            )
            for fold_id in range(3)
        ]
        for epoch in range(1, int(config["gate_epochs"]) + 1):
            gate.train()
            loss_sum = 0.0
            step_count = 0
            for fold_id, package in enumerate(packages):
                seen_indices = package["seen_indices"]
                unseen_indices = package["unseen_indices"]
                steps = min(seen_indices.numel() // half, unseen_indices.numel() // half)
                for _ in range(steps):
                    generator = generators[fold_id]
                    si = seen_indices[
                        torch.randperm(seen_indices.numel(), generator=generator)[:half]
                    ]
                    ui = unseen_indices[
                        torch.randperm(unseen_indices.numel(), generator=generator)[:half]
                    ]
                    indices = torch.cat((si, ui))
                    images = tensors["train_features"][indices].to(device).float()
                    targets = label_map[tensors["train_labels"].long()[indices]].to(device)
                    base_all = package["base_all"].to(device)
                    final_all = package["fold_full"].to(device).clone()
                    coefficient = gate(package["gate_features"].to(device))
                    pseudo_unseen = package["pseudo_unseen"].to(device)
                    final_all[pseudo_unseen] = _transport(
                        base_all.index_select(0, pseudo_unseen),
                        package["value"].to(device),
                        coefficient,
                        config["transport_mode"],
                    )
                    competition = final_all.index_select(0, seenclasses.to(device))
                    logits = F.normalize(images, dim=-1) @ competition.T
                    logits = logits * package["scale"].to(device)
                    ce = F.cross_entropy(logits, targets)
                    pseudo_unseen_mask = torch.isin(
                        tensors["train_labels"].long()[indices],
                        package["pseudo_unseen"],
                    ).to(device)
                    pseudo_unseen_ce = _pseudo_unseen_risk(
                        logits[pseudo_unseen_mask],
                        targets[pseudo_unseen_mask],
                        config["pseudo_unseen_loss_mode"],
                    )
                    topo = topology_loss(
                        base_all.index_select(0, seenclasses.to(device)), competition
                    )
                    alignment = _centroid_loss(
                        final_all.index_select(0, pseudo_unseen),
                        package["pseudo_unseen_centroids"].to(device),
                        config["centroid_alignment_mode"],
                    )
                    loss = (
                        ce
                        + float(config["topology_weight"]) * topo
                        + float(config["alpha_penalty"]) * coefficient.square().mean()
                        + float(config["centroid_alignment_weight"]) * alignment
                        + float(config["pseudo_unseen_ce_weight"])
                        * pseudo_unseen_ce
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach())
                    step_count += 1
            if epoch in (1, 5, 10, 15, 20):
                print_log(
                    f"gate_member={member} epoch={epoch} "
                    f"loss={loss_sum/step_count:.6f}"
                )
        gates.append(gate.eval())
    return nn.ModuleList(gates)


def _train_gate_ensemble(packages, tensors, config, device, seed, print_log):
    gates = []
    label_map = torch.full((200,), -1, dtype=torch.long)
    seenclasses = packages[0]["seenclasses"]
    label_map[seenclasses] = torch.arange(150)
    half = int(config["gate_batch_half"])
    input_dim = _gate_input_dim(config["gate_feature_mode"])
    for fold_id, package in enumerate(packages):
        gate = _make_gate(config, input_dim, device)
        optimizer = torch.optim.Adam(
            gate.parameters(),
            lr=float(config["gate_lr"]),
            weight_decay=float(config["gate_weight_decay"]),
        )
        generator = torch.Generator(device="cpu").manual_seed(seed * 2000 + fold_id)
        seen_indices = package["seen_indices"]
        unseen_indices = package["unseen_indices"]
        steps = min(seen_indices.numel() // half, unseen_indices.numel() // half)
        for epoch in range(1, int(config["gate_epochs"]) + 1):
            gate.train()
            loss_sum = 0.0
            for _ in range(steps):
                si = seen_indices[torch.randperm(seen_indices.numel(), generator=generator)[:half]]
                ui = unseen_indices[torch.randperm(unseen_indices.numel(), generator=generator)[:half]]
                indices = torch.cat((si, ui))
                images = tensors["train_features"][indices].to(device).float()
                targets = label_map[tensors["train_labels"].long()[indices]].to(device)
                base_all = package["base_all"].to(device)
                final_all = package["fold_full"].to(device).clone()
                alpha = gate(package["gate_features"].to(device))
                pseudo_unseen = package["pseudo_unseen"].to(device)
                final_all[pseudo_unseen] = _transport(
                    base_all.index_select(0, pseudo_unseen),
                    package["value"].to(device),
                    alpha,
                    config["transport_mode"],
                )
                competition = final_all.index_select(0, seenclasses.to(device))
                logits = F.normalize(images, dim=-1) @ competition.T * package["scale"].to(device)
                ce = F.cross_entropy(logits, targets)
                pseudo_unseen_mask = torch.isin(
                    tensors["train_labels"].long()[indices],
                    package["pseudo_unseen"],
                ).to(device)
                pseudo_unseen_ce = _pseudo_unseen_risk(
                    logits[pseudo_unseen_mask],
                    targets[pseudo_unseen_mask],
                    config["pseudo_unseen_loss_mode"],
                )
                topo = topology_loss(
                    base_all.index_select(0, seenclasses.to(device)), competition
                )
                loss = (
                    ce
                    + float(config["topology_weight"]) * topo
                    + float(config["alpha_penalty"]) * alpha.square().mean()
                    + float(config["centroid_alignment_weight"])
                    * _centroid_loss(
                        final_all.index_select(0, pseudo_unseen),
                        package["pseudo_unseen_centroids"].to(device),
                        config["centroid_alignment_mode"],
                    )
                    + float(config["pseudo_unseen_ce_weight"])
                    * pseudo_unseen_ce
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
            if epoch in (1, 5, 10, 15, 20):
                print_log(
                    f"gate_fold={fold_id} epoch={epoch} loss={loss_sum/steps:.6f}"
                )
        gates.append(gate.eval())
    return nn.ModuleList(gates)


def _candidate_prototypes(
    model,
    gate,
    seenclasses,
    unseenclasses,
    device,
    feature_mode,
    folds,
    transport_mode="convex_blend",
    initialization_ensemble=False,
):
    model.eval()
    with torch.no_grad():
        base_all = model.base_prototypes()
        final_all = model.prototypes().clone()
        value = model.value_candidate(unseenclasses.to(device))
        if isinstance(gate, nn.ModuleList) and initialization_ensemble:
            features = gate_features(
                base_all.index_select(0, unseenclasses.to(device)),
                value,
                base_all.index_select(0, seenclasses.to(device)),
                mode=feature_mode,
            )
            alpha = torch.stack([member(features) for member in gate]).mean(dim=0)
        elif isinstance(gate, nn.ModuleList):
            alpha_values = []
            for fold_gate, (pseudo_seen, _) in zip(gate, folds):
                features = gate_features(
                    base_all.index_select(0, unseenclasses.to(device)),
                    value,
                    base_all.index_select(0, pseudo_seen.to(device)),
                    mode=feature_mode,
                )
                alpha_values.append(fold_gate(features))
            alpha = torch.stack(alpha_values).mean(dim=0)
        else:
            features = gate_features(
                base_all.index_select(0, unseenclasses.to(device)),
                value,
                base_all.index_select(0, seenclasses.to(device)),
                mode=feature_mode,
            )
            alpha = gate(features)
        final_all[unseenclasses.to(device)] = _transport(
            base_all.index_select(0, unseenclasses.to(device)),
            value,
            alpha,
            transport_mode,
        )
    return final_all, alpha


def run(config_path: Path, output_dir: Path, expected_commit: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前HEAD不一致。")
    config, config_sha = load_config(config_path)
    base_config_path = Path(config["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = Path.cwd() / base_config_path
    base_config, base_config_sha = h1.load_config(base_config_path)
    paths = h1.resolve_paths(base_config)
    input_sha = h1.verify_inputs(base_config, paths, h1.TRAINING_KEYS)
    checkpoint_path = Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path) != config["base_checkpoint_sha256"]:
        raise ValueError("V2基线checkpoint SHA不匹配。")
    device = torch.device(base_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ELPT正式TRY要求CUDA。")

    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)

    def print_log(value):
        print(value)

    try:
        seed = int(config["seed"])
        configure_reproducibility(seed, strict_determinism=True, deterministic_warn_only=False)
        tensors = {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in ("sentence_embeds", "train_features", "train_labels")
        }
        if set(tensors) != {"sentence_embeds", "train_features", "train_labels"}:
            raise RuntimeError("ELPT训练阶段只能加载seen训练输入。")
        seenclasses = torch.unique(tensors["train_labels"].long(), sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        if config["fold_strategy"] == "semantic_pca_blocks":
            folds = semantic_pca_folds(seenclasses, tensors["sentence_embeds"])
        else:
            folds = fixed_class_folds(seenclasses)
        packages = []
        for fold_id, (pseudo_seen, pseudo_unseen) in enumerate(folds):
            if config["fold_checkpoint_dir"]:
                fold_model = _load_fold_checkpoint(
                    fold_id,
                    pseudo_seen,
                    tensors["sentence_embeds"],
                    tensors["train_features"],
                    tensors["train_labels"],
                    base_config,
                    device,
                    config["fold_checkpoint_dir"],
                )
                print_log(f"fold={fold_id} reused_from={config['fold_checkpoint_dir']}")
            else:
                fold_model = _train_fold(
                    fold_id,
                    pseudo_seen,
                    tensors["sentence_embeds"],
                    tensors["train_features"],
                    tensors["train_labels"],
                    base_config,
                    device,
                    output_dir,
                    seed,
                    print_log,
                )
            packages.append(
                _fold_package(
                    fold_model,
                    pseudo_seen,
                    pseudo_unseen,
                    tensors,
                    seenclasses,
                    device,
                    config["gate_feature_mode"],
                )
            )
            del fold_model
            torch.cuda.empty_cache()

        gate = _train_gate(packages, tensors, config, device, seed, print_log)
        torch.save(
            {
                "attempt_id": config["attempt_id"],
                "code_commit": code_commit,
                "config": config,
                "gate_state_dict": copy.deepcopy(gate.state_dict()),
            },
            output_dir / "gate_model.pth",
        )

        # official test只在所有训练完成后加载。
        input_sha.update(h1.verify_inputs(base_config, paths, h1.OFFICIAL_KEYS))
        tensors.update(
            {
                name: torch.load(paths[name], map_location="cpu", weights_only=True)
                for name in h1.OFFICIAL_KEYS
            }
        )
        baseline_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        baseline = TGVPRH1FixedEqual(
            tensors["sentence_embeds"],
            seenclasses,
            h1.visual_centroids(
                tensors["train_features"], tensors["train_labels"].long(), seenclasses
            ),
            dropout=base_config["dropout"],
            inner_ratio=base_config["inner_ratio"],
            outer_ratio=base_config["outer_ratio"],
            temperature=base_config["temperature"],
        )
        baseline.load_state_dict(baseline_checkpoint["model_state_dict"], strict=True)
        variable = VariableClassTGVPR(
            tensors["sentence_embeds"],
            seenclasses,
            h1.visual_centroids(
                tensors["train_features"], tensors["train_labels"].long(), seenclasses
            ),
            dropout=base_config["dropout"],
            inner_ratio=base_config["inner_ratio"],
            outer_ratio=base_config["outer_ratio"],
            temperature=base_config["temperature"],
        )
        variable.load_state_dict(baseline_checkpoint["model_state_dict"], strict=True)
        baseline = baseline.to(device).eval()
        variable = variable.to(device).eval()
        baseline_metrics = h1.evaluate(
            baseline, tensors, seenclasses, unseenclasses, device
        )
        candidate_prototypes, alpha = _candidate_prototypes(
            variable,
            gate,
            seenclasses,
            unseenclasses,
            device,
            config["gate_feature_mode"],
            folds,
            config["transport_mode"],
            int(config["gate_initialization_ensemble"]) > 1,
        )
        candidate = FrozenPrototypeClassifier(
            candidate_prototypes, variable.scale()
        ).to(device)
        candidate_metrics = h1.evaluate(
            candidate, tensors, seenclasses, unseenclasses, device
        )
        delta = {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in ("U", "S", "H", "ZS")
        }
        alpha_stats = {
            "mean": float(alpha.mean()),
            "std": float(alpha.std(unbiased=False)),
            "min": float(alpha.min()),
            "max": float(alpha.max()),
        }
        if config["transport_mode"] == "tangent":
            base_unseen = variable.base_prototypes().index_select(
                0, unseenclasses.to(device)
            )
            moved_unseen = candidate_prototypes.index_select(
                0, unseenclasses.to(device)
            )
            angles = torch.rad2deg(
                torch.acos((base_unseen * moved_unseen).sum(dim=-1).clamp(-1.0, 1.0))
            )
            angle_stats = {
                "mean_degrees": float(angles.mean()),
                "max_degrees": float(angles.max()),
            }
            if config["parent_metrics_percent"] is not None:
                parent_delta = {
                    key: candidate_metrics[key] - float(config["parent_metrics_percent"][key])
                    for key in ("U", "S", "H", "ZS")
                }
                success = (
                    parent_delta["H"] >= 0.05
                    and parent_delta["U"] >= -2.0
                    and parent_delta["S"] >= -2.0
                    and 0.02 < alpha_stats["mean"] < 1.45
                    and alpha_stats["std"] > 0.01
                    and angle_stats["max_degrees"] < 45.0
                )
            else:
                parent_delta = None
                success = (
                    delta["H"] >= 0.20
                    and candidate_metrics["U"] > baseline_metrics["U"]
                    and delta["S"] >= -2.0
                    and 0.02 < alpha_stats["mean"] < 1.45
                    and alpha_stats["std"] > 0.01
                    and angle_stats["max_degrees"] < 45.0
                )
        else:
            angle_stats = None
            parent_delta = None
            success = (
                candidate_metrics["H"] > 74.023182
                and candidate_metrics["U"] > baseline_metrics["U"]
                and delta["S"] >= -2.0
                and 0.02 < alpha_stats["mean"] < 0.50
                and alpha_stats["std"] > 0.01
            )
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        metrics = {
            "attempt_id": config["attempt_id"],
            "idea_id": config["idea_id"],
            "framework_id": config["framework_id"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "base_config_sha256": base_config_sha,
            "base_checkpoint_sha256": config["base_checkpoint_sha256"],
            "evaluation_protocol": h1.EVALUATION_PROTOCOL,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "folds": [
                {
                    "pseudo_seen": pseudo_seen.tolist(),
                    "pseudo_unseen": pseudo_unseen.tolist(),
                }
                for pseudo_seen, pseudo_unseen in folds
            ],
            "baseline_metrics_percent": baseline_metrics,
            "candidate_metrics_percent": candidate_metrics,
            "delta_percent_points": delta,
            "alpha_stats": alpha_stats,
            "gate_max_alpha": float(config["gate_max_alpha"]),
            "alpha_penalty": float(config["alpha_penalty"]),
            "fold_checkpoint_dir": config["fold_checkpoint_dir"],
            "gate_feature_mode": config["gate_feature_mode"],
            "gate_ensemble": bool(config["gate_ensemble"]),
            "gate_initialization_ensemble": int(
                config["gate_initialization_ensemble"]
            ),
            "transport_mode": config["transport_mode"],
            "angle_stats": angle_stats,
            "centroid_alignment_weight": float(
                config["centroid_alignment_weight"]
            ),
            "centroid_alignment_mode": config["centroid_alignment_mode"],
            "pseudo_unseen_ce_weight": float(
                config["pseudo_unseen_ce_weight"]
            ),
            "pseudo_unseen_loss_mode": config["pseudo_unseen_loss_mode"],
            "gate_architecture": config["gate_architecture"],
            "parent_gate_model_sha256": config["parent_gate_model_sha256"],
            "max_residual_step": float(config["max_residual_step"]),
            "gate_training_mode": config["gate_training_mode"],
            "inner_lr": float(config["inner_lr"]),
            "meta_gradient_mode": config["meta_gradient_mode"],
            "seen_gradient_weight": float(config["seen_gradient_weight"]),
            "fold_strategy": config["fold_strategy"],
            "gradient_normalization": config["gradient_normalization"],
            "parent_metrics_percent": config["parent_metrics_percent"],
            "delta_vs_parent_percent_points": parent_delta,
            "success": success,
            "gate_model_sha256": sha256_file(output_dir / "gate_model.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", metrics)
        print_log(metrics)
        return metrics
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit)


if __name__ == "__main__":
    main()
