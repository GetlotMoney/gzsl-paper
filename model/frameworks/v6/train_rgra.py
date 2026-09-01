"""One-stage RGRA training and same-checkpoint evaluation contracts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from model.frameworks.v6.rgra import RGRA_CONDITIONS, RGRAModel
from model.frameworks.v6.rgra_assets import (
    RGRAEvalAssets,
    RGRATrainAssets,
    load_rgra_eval_assets,
    load_rgra_train_assets,
)
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v6-rgra-train.v1"
EXPORT_SCHEMA = "gzsl-paper.v6-rgra-graph-free-export.v1"
RECEIPT_SCHEMA = "gzsl-paper.v6-rgra-result.v1"
CONDITIONS = ("full", "s_off", "v_off", "i_off", "additive", "shuffled")
CONFIG_KEYS = {
    "schema_version", "experiment_id", "framework_id", "dataset", "condition_id",
    "base_commit", "asset_manifest", "asset_manifest_sha256", "asset_id",
    "coarse_patch_files_sha256", "relation_asset_manifest",
    "relation_asset_manifest_sha256", "relation_asset_id", "source_config",
    "source_checkpoint", "source_checkpoint_sha256", "source_code_commit",
    "source_config_sha256", "parent_metrics_percent", "target_h",
    "required_module_delta_h", "output_dir", "device", "random_seed", "batch_size",
    "eval_batch_size", "nominal_epochs", "total_updates", "eval_interval_steps",
    "learning_rate", "min_learning_rate", "weight_decay", "hidden_dim",
    "topology_loss_weight", "direction_loss_weight", "relation_ridge",
    "visual_temperature", "relation_temperature", "seen_logit_gamma",
    "max_rho_s", "initial_rho_s", "max_beta_v", "initial_beta_v",
    "max_alpha", "initial_alpha", "require_clean_tree",
    "feature_provenance_complete", "pclr_online_inference",
    "test_used_for_selection", "test_used_for_hyperparameter_selection",
    "unseen_images_used_for_gradient", "strict_blind_claim",
    "human_annotations_used", "expert_attributes_used", "llm_world_knowledge_used",
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


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    parent = config.get("parent_metrics_percent", {}) if isinstance(config, dict) else {}
    invalid = (
        not isinstance(config, dict)
        or actual != CONFIG_KEYS
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != "V6-TRY-008"
        or config.get("framework_id") != "FRAMEWORK-V6-DEVELOPMENT"
        or config.get("condition_id") != "RGRA_ONE_STAGE_E2E"
        or config.get("base_commit") != "52b511d77b4ad048f35b40dc3cbd9afd092167e9"
        or config.get("dataset") != "CUB"
        or int(config.get("batch_size", 0)) != 50
        or int(config.get("eval_batch_size", 0)) <= 0
        or int(config.get("nominal_epochs", 0)) != 200
        or int(config.get("total_updates", 0)) != 28228
        or int(config.get("eval_interval_steps", 0)) != 141
        or float(config.get("learning_rate", 0.0)) <= 0.0
        or not 0.0 <= float(config.get("min_learning_rate", -1.0))
        <= float(config.get("learning_rate", 0.0))
        or float(config.get("required_module_delta_h", -1.0)) != 1.0
        or abs(float(parent.get("H", -1.0)) - 81.068777) > 1e-6
        or config.get("require_clean_tree") is not True
        or config.get("feature_provenance_complete") is not False
        or config.get("pclr_online_inference") is not False
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not False
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
        or config.get("expert_attributes_used") is not False
        or config.get("llm_world_knowledge_used") is not True
    )
    if invalid:
        raise ValueError("RGRA config identity, budget, parent, or disclosure changed.")
    source_config = Path(config["source_config"])
    if (
        not source_config.is_file()
        or sha256_file(source_config) != config["source_config_sha256"]
    ):
        raise ValueError("RGRA V5 source config path or SHA mismatch.")
    return config, sha256_file(path)


def build_model(
    config: Mapping[str, Any], assets: RGRATrainAssets, device: torch.device
) -> RGRAModel:
    return RGRAModel(
        assets.role_sentence_embeds,
        assets.p_v5,
        assets.relation_directions,
        assets.edge_index,
        assets.seen_classes,
        class_count=200,
        hidden_dim=int(config["hidden_dim"]),
        relation_ridge=float(config["relation_ridge"]),
        visual_temperature=float(config["visual_temperature"]),
        relation_temperature=float(config["relation_temperature"]),
        seen_logit_gamma=float(config["seen_logit_gamma"]),
        max_rho_s=float(config["max_rho_s"]),
        initial_rho_s=float(config["initial_rho_s"]),
        max_beta_v=float(config["max_beta_v"]),
        initial_beta_v=float(config["initial_beta_v"]),
        max_alpha=float(config["max_alpha"]),
        initial_alpha=float(config["initial_alpha"]),
        scale=assets.scale,
        reader_state_dict=assets.reader_state_dict,
    ).to(device)


def _patch_batch(
    patches: np.memmap, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    rows = np.asarray(indices.detach().cpu(), dtype=np.int64)
    values = np.asarray(patches[rows], dtype=np.float32).copy()
    return torch.from_numpy(values).to(device=device)


def _sequential_patch_batch(
    patches: np.memmap, start: int, stop: int, device: torch.device
) -> torch.Tensor:
    values = np.asarray(patches[start:stop], dtype=np.float32).copy()
    return torch.from_numpy(values).to(device=device)


def _gradient_norms(model: RGRAModel) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, parameters in model.training_parameter_groups().items():
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"RGRA non-finite gradient: {name}")
                total += float(parameter.grad.detach().float().norm().cpu())
        result[name] = total
    return result


def micro_contract(
    model: RGRAModel,
    assets: RGRATrainAssets,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    count = int(config["batch_size"])
    rows = torch.arange(count, dtype=torch.long)
    cls = assets.train_features.index_select(0, rows).to(device).float()
    patches = _patch_batch(assets.train_coarse_patches, rows, device)
    targets = assets.train_labels.index_select(0, rows).to(device).long()

    model.train()
    cls_only = model.classification_gradient_norms(cls, patches, targets)
    cls_group_norms = {name: float(cls_only[name]) for name in ("rsc", "rva", "rfm")}
    if any(not np.isfinite(value) or value <= 0.0 for value in cls_group_norms.values()):
        raise RuntimeError(f"L_cls did not reach all RGRA modules: {cls_group_norms}")

    model.zero_grad(set_to_none=True)
    total, parts = model.total_loss(
        cls,
        patches,
        targets,
        topology_weight=float(config["topology_loss_weight"]),
        direction_weight=float(config["direction_loss_weight"]),
    )
    total.backward()
    total_group_norms = _gradient_norms(model)
    if any(value <= 0.0 for value in total_group_norms.values()):
        raise RuntimeError(f"total loss did not reach all RGRA modules: {total_group_norms}")

    model.eval()
    with torch.no_grad():
        components = model.score_components(cls, patches, condition="full")
        attention = components["attention"]
        if not torch.isfinite(attention).all() or float(attention.std()) <= 0.0:
            raise RuntimeError("RGRA attention is invalid or uniform.")
        attention_sum_error = float((attention.sum(dim=-1) - 1.0).abs().max().cpu())
        if attention_sum_error > 1e-5:
            raise RuntimeError("RGRA attention does not sum to one.")
        shapes = {}
        for condition in CONDITIONS:
            logits = model.logits(cls, patches, condition=condition)
            if tuple(logits.shape) != (count, 200) or not torch.isfinite(logits).all():
                raise RuntimeError(f"RGRA {condition} micro logits invalid.")
            shapes[condition] = list(logits.shape)
        alpha_zero = model.logits(cls, patches, condition="full", alpha_override=0.0)
        i_off = model.logits(cls, patches, condition="i_off")
        alpha_zero_i_off_max_abs = float((alpha_zero - i_off).abs().max().cpu())
        if alpha_zero_i_off_max_abs > 1e-6:
            raise RuntimeError("RGRA alpha=0 does not close exactly to I-off.")
    model.zero_grad(set_to_none=True)
    return {
        "official_test_loaded": False,
        "batch_size": count,
        "cls_only": {"loss": float(cls_only["classification_loss"]), **cls_group_norms},
        "total_loss": {
            "value": float(total.detach().cpu()),
            "parts": {key: float(value.detach().cpu()) for key, value in parts.items()},
            "gradient_norms": total_group_norms,
        },
        "attention_std": float(attention.std().cpu()),
        "attention_sum_max_abs_error": attention_sum_error,
        "condition_shapes": shapes,
        "alpha_zero_i_off_max_abs": alpha_zero_i_off_max_abs,
    }


@torch.no_grad()
def _condition_predictions(
    model: RGRAModel,
    assets: RGRAEvalAssets,
    device: torch.device,
    condition: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model.eval()

    def logits_for(features: torch.Tensor, patches: np.memmap) -> torch.Tensor:
        rows = []
        for start in range(0, features.size(0), batch_size):
            stop = min(start + batch_size, features.size(0))
            cls = features[start:stop].to(device).float()
            patch = _sequential_patch_batch(patches, start, stop, device)
            logits = model.logits(cls, patch, condition=condition)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"RGRA {condition} official logits non-finite.")
            rows.append(logits.cpu())
        return torch.cat(rows)

    seen_logits = logits_for(assets.test_seen_features, assets.test_seen_coarse_patches)
    unseen_logits = logits_for(
        assets.test_unseen_features, assets.test_unseen_coarse_patches
    )
    unseen_axis = assets.unseen_classes.long()
    return {
        "seen": seen_logits.argmax(dim=1),
        "unseen": unseen_logits.argmax(dim=1),
        "zs": unseen_axis.index_select(
            0, unseen_logits.index_select(1, unseen_axis).argmax(dim=1)
        ),
    }


def _metrics(
    predictions: Mapping[str, torch.Tensor], assets: RGRAEvalAssets
) -> dict[str, float]:
    seen_accuracy = 100.0 * per_class_accuracy(
        assets.test_seen_labels, predictions["seen"], assets.seen_classes
    )
    unseen_accuracy = 100.0 * per_class_accuracy(
        assets.test_unseen_labels, predictions["unseen"], assets.unseen_classes
    )
    zs_accuracy = 100.0 * per_class_accuracy(
        assets.test_unseen_labels, predictions["zs"], assets.unseen_classes
    )
    harmonic = (
        2.0 * seen_accuracy * unseen_accuracy / (seen_accuracy + unseen_accuracy)
        if seen_accuracy + unseen_accuracy
        else 0.0
    )
    return {"U": unseen_accuracy, "S": seen_accuracy, "H": harmonic, "ZS": zs_accuracy}


def evaluate_condition(
    model: RGRAModel,
    assets: RGRAEvalAssets,
    device: torch.device,
    condition: str,
    batch_size: int,
    *,
    include_predictions: bool = False,
) -> dict[str, Any]:
    if condition not in RGRA_CONDITIONS:
        raise ValueError(f"unknown RGRA condition: {condition}")
    predictions = _condition_predictions(model, assets, device, condition, batch_size)
    result: dict[str, Any] = {"metrics": _metrics(predictions, assets)}
    if include_predictions:
        result["predictions"] = predictions
    return result


def evaluate_all_conditions(
    model: RGRAModel,
    assets: RGRAEvalAssets,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    results = {
        condition: evaluate_condition(
            model, assets, device, condition, batch_size, include_predictions=True
        )
        for condition in CONDITIONS
    }
    full_h = float(results["full"]["metrics"]["H"])
    gaps = {
        condition: full_h - float(results[condition]["metrics"]["H"])
        for condition in ("s_off", "v_off", "i_off", "additive", "shuffled")
    }
    labels = torch.cat((assets.test_seen_labels, assets.test_unseen_labels))
    full_predictions = torch.cat(
        (results["full"]["predictions"]["seen"], results["full"]["predictions"]["unseen"])
    )
    transitions = {}
    for condition in ("additive", "shuffled"):
        control_predictions = torch.cat(
            (results[condition]["predictions"]["seen"], results[condition]["predictions"]["unseen"])
        )
        corrected = int((full_predictions.eq(labels) & control_predictions.ne(labels)).sum())
        damaged = int((full_predictions.ne(labels) & control_predictions.eq(labels)).sum())
        transitions[condition] = {"corrected": corrected, "damaged": damaged, "net": corrected - damaged}
    for value in results.values():
        value.pop("predictions", None)
    return {"conditions": results, "gaps_H": gaps, "transitions": transitions}


def _new_or_resume_output(output_dir: Path, resume_from: Path | None) -> Path:
    if resume_from is None:
        return prepare_output_dir(output_dir)
    resolved_output = output_dir.resolve()
    resolved_resume = resume_from.resolve()
    if not resolved_output.is_dir() or resolved_resume != resolved_output / "checkpoint_last.pth":
        raise ValueError("RGRA resume must use output_dir/checkpoint_last.pth.")
    return resolved_output


def run(
    config_path: Path,
    expected_commit: str,
    expected_config_sha: str,
    *,
    output_dir: Path | None = None,
    micro_batch_only: bool = False,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("RGRA expected commit mismatch.")
    config, config_sha = load_config(config_path)
    if config_sha != expected_config_sha:
        raise ValueError("RGRA expected config SHA mismatch.")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RGRA official run requires CUDA.")
    destination = _new_or_resume_output(Path(output_dir or config["output_dir"]), resume_from)
    log_handle = (destination / "training.log").open(
        "a" if resume_from is not None else "x", encoding="utf-8", buffering=1
    )
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        reproducibility = configure_reproducibility(
            int(config["random_seed"]), strict_determinism=True, deterministic_warn_only=False
        )
        train_assets = load_rgra_train_assets(config)
        model = build_model(config, train_assets, device)
        micro = micro_contract(model, train_assets, config, device)
        print(f"RGRA commit={code_commit} config_sha={config_sha}")
        print(json.dumps({"micro": micro}, sort_keys=True))
        if micro_batch_only:
            result = {
                "schema_version": "gzsl-paper.v6-rgra-micro.v1",
                "experiment_id": config["experiment_id"],
                "code_commit": code_commit,
                "config_sha256": config_sha,
                "train_asset_identity": train_assets.identity,
                "micro": micro,
                "unseen_images_used_for_gradient": False,
            }
            atomic_write_json(destination / "micro_batch_receipt.json", result)
            return result

        eval_assets = load_rgra_eval_assets(config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config["total_updates"]), eta_min=float(config["min_learning_rate"])
        )
        generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
        start_update = 0
        history: list[dict[str, Any]] = []
        best_metrics: dict[str, float] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_update = 0
        best_zs: dict[str, Any] | None = None
        if resume_from is not None:
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("code_commit") != code_commit
                or checkpoint.get("config_sha256") != config_sha
                or not 0 <= int(checkpoint.get("update", -1)) < int(config["total_updates"])
            ):
                raise ValueError("RGRA resume checkpoint identity mismatch.")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            generator.set_state(checkpoint["batch_generator_state"])
            torch.set_rng_state(checkpoint["cpu_rng_state"])
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
            start_update = int(checkpoint["update"])
            history = checkpoint["history"]
            best_metrics = checkpoint["best_metrics"]
            best_state = checkpoint["best_model_state_dict"]
            best_update = int(checkpoint["best_update"])
            best_zs = checkpoint["best_zs_observation"]

        total_updates = int(config["total_updates"])
        eval_interval = int(config["eval_interval_steps"])
        eval_batch_size = int(config["eval_batch_size"])
        for update in range(start_update, total_updates + 1):
            resumed_checkpoint_row = resume_from is not None and update == start_update
            if (
                not resumed_checkpoint_row
                and (update == 0 or update % eval_interval == 0 or update == total_updates)
            ):
                full = evaluate_condition(model, eval_assets, device, "full", eval_batch_size)["metrics"]
                row = {"evaluation_index": len(history), "update": update, **full}
                history.append(row)
                if best_metrics is None or float(full["H"]) > float(best_metrics["H"]):
                    best_metrics = dict(full)
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    best_update = update
                if best_zs is None or float(full["ZS"]) > float(best_zs["ZS"]):
                    best_zs = {"update": update, **full}
                print(
                    f"eval={row['evaluation_index']} update={update} U={full['U']:.6f} "
                    f"S={full['S']:.6f} H={full['H']:.6f} ZS={full['ZS']:.6f}"
                )
                atomic_torch_save(
                    destination / "checkpoint_last.pth",
                    {
                        "schema_version": SCHEMA,
                        "experiment_id": config["experiment_id"],
                        "code_commit": code_commit,
                        "config_sha256": config_sha,
                        "update": update,
                        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "batch_generator_state": generator.get_state(),
                        "cpu_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                        "history": history,
                        "best_update": best_update,
                        "best_metrics": best_metrics,
                        "best_model_state_dict": best_state,
                        "best_zs_observation": best_zs,
                        "train_asset_identity": train_assets.identity,
                        "eval_asset_identity": eval_assets.identity,
                    },
                )
            if update == total_updates:
                break
            model.train()
            indices = torch.randperm(train_assets.train_labels.numel(), generator=generator)[
                : int(config["batch_size"])
            ]
            cls = train_assets.train_features.index_select(0, indices).to(device).float()
            patches = _patch_batch(train_assets.train_coarse_patches, indices, device)
            targets = train_assets.train_labels.index_select(0, indices).to(device).long()
            optimizer.zero_grad(set_to_none=True)
            total, _parts = model.total_loss(
                cls, patches, targets,
                topology_weight=float(config["topology_loss_weight"]),
                direction_weight=float(config["direction_loss_weight"]),
            )
            if not torch.isfinite(total):
                raise FloatingPointError("RGRA total loss non-finite.")
            total.backward()
            _gradient_norms(model)
            optimizer.step()
            scheduler.step()

        if best_state is None or best_metrics is None or best_zs is None:
            raise RuntimeError("RGRA did not select a Full checkpoint.")
        model.load_state_dict(best_state, strict=True)
        final_controls = evaluate_all_conditions(model, eval_assets, device, eval_batch_size)
        final_full = final_controls["conditions"]["full"]["metrics"]
        if any(
            abs(float(final_full[name]) - float(best_metrics[name])) > 1e-6
            for name in ("U", "S", "H", "ZS")
        ):
            raise RuntimeError("RGRA best checkpoint metrics did not reproduce.")
        parent_h = float(config["parent_metrics_percent"]["H"])
        module_gaps = {name: float(final_controls["gaps_H"][name]) for name in ("s_off", "v_off", "i_off")}
        framework_passed = float(best_metrics["H"]) > parent_h and all(
            value >= float(config["required_module_delta_h"]) for value in module_gaps.values()
        )
        non_equivalence_passed = all(
            float(final_controls["gaps_H"][name]) >= 0.5
            and int(final_controls["transitions"][name]["net"]) >= 20
            for name in ("additive", "shuffled")
        )

        export_package = model.export_graph_free_state()
        forbidden = ("relation_embeddings", "edge_index", "incidence", "laplacian")
        export_keys = tuple(export_package["state_dict"])
        if any(token in key for token in forbidden for key in export_keys):
            raise RuntimeError("RGRA export contains forbidden online-graph tensors.")
        deployed = RGRAModel.from_graph_free_state(export_package).to(device).eval()
        probe_rows = torch.arange(min(8, train_assets.train_labels.numel()))
        probe_cls = train_assets.train_features.index_select(0, probe_rows).to(device).float()
        probe_patches = _patch_batch(train_assets.train_coarse_patches, probe_rows, device)
        with torch.no_grad():
            export_max_abs = float(
                (model(probe_cls, probe_patches) - deployed(probe_cls, probe_patches)).abs().max().cpu()
            )
        if export_max_abs > 1e-5:
            raise RuntimeError("RGRA graph-free export parity failed.")

        model_path = destination / "model_best.pth"
        atomic_torch_save(
            model_path,
            {
                "schema_version": SCHEMA,
                "experiment_id": config["experiment_id"],
                "code_commit": code_commit,
                "config_sha256": config_sha,
                "best_update": best_update,
                "best_metrics": best_metrics,
                "model_state_dict": best_state,
                "train_asset_identity": train_assets.identity,
                "eval_asset_identity": eval_assets.identity,
            },
        )
        export_path = destination / "rgra_graph_free_export.pt"
        atomic_torch_save(
            export_path,
            {
                "schema_version": EXPORT_SCHEMA,
                "experiment_id": config["experiment_id"],
                "code_commit": code_commit,
                "config_sha256": config_sha,
                "package": export_package,
                "export_max_abs": export_max_abs,
            },
        )
        atomic_write_json(destination / "evaluation_history.json", {"rows": history})
        result = {
            "schema_version": RECEIPT_SCHEMA,
            "experiment_id": config["experiment_id"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "best_update": best_update,
            "best_metrics": best_metrics,
            "best_zs_observation": best_zs,
            "final_same_checkpoint": final_controls,
            "parent_H": parent_h,
            "parent_delta_H": float(best_metrics["H"]) - parent_h,
            "module_gaps_H": module_gaps,
            "framework_performance_contract_passed": framework_passed,
            "conditional_non_equivalence_passed": non_equivalence_passed,
            "h80_is_target_not_gate": True,
            "target_h": float(config["target_h"]),
            "micro": micro,
            "train_asset_identity": train_assets.identity,
            "eval_asset_identity": eval_assets.identity,
            "model_best_sha256": sha256_file(model_path),
            "graph_free_export_sha256": sha256_file(export_path),
            "graph_free_export_max_abs": export_max_abs,
            "pclr_online_inference": False,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "feature_provenance_complete": False,
            "reproducibility": reproducibility,
        }
        atomic_write_json(destination / "metrics.json", result)
        print(json.dumps(result, sort_keys=True))
        return result
    finally:
        sys.stdout = original_stdout
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--micro-batch-only", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    result = run(
        args.config,
        args.expected_commit,
        args.expected_config_sha,
        output_dir=args.output_dir,
        micro_batch_only=args.micro_batch_only,
        resume_from=args.resume_from,
    )
    print(json.dumps({"schema_version": result["schema_version"]}, sort_keys=True))


if __name__ == "__main__":
    main()
