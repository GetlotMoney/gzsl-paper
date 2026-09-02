"""PFDE R3 training: prototype-first S with separated V and role-patch I."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v6.ctpm import (
    CTPMModel, attention_diversity_loss, balanced_pair_ce,
    isolated_interaction_margin, pair_scatter,
)
from model.frameworks.v6.ctpm_assets import (
    CTPMTrainAssets, load_ctpm_eval_assets, load_ctpm_train_assets,
)
from model.frameworks.v6.evaluate_ctpm import evaluate
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_torch_save, atomic_write_json, current_code_commit,
    prepare_output_dir, require_clean_code_tree,
)
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v6-pfde-train.v1"
RESULT_SCHEMA = "gzsl-paper.v6-pfde-result.v1"
CONFIG_KEYS = {
    "schema_version", "experiment_id", "framework_id", "dataset", "condition_id",
    "code_parent_commit", "asset_manifest", "asset_manifest_sha256", "asset_id",
    "coarse_patch_files_sha256", "top2_gate_result", "top2_gate_result_sha256",
    "top2_gate_script_sha256", "parent_metrics_percent",
    "semantic_parent_metrics_percent", "required_module_delta_h",
    "output_dir", "device", "random_seed", "batch_size", "eval_batch_size",
    "nominal_epochs", "total_updates", "eval_interval_steps", "weight_decay",
    "hidden_dim", "patch_projection_dim",
    "max_margin", "max_role_weight", "semantic_learning_rate",
    "visual_interaction_learning_rate", "semantic_min_learning_rate",
    "visual_interaction_min_learning_rate", "attention_diversity_weight",
    "logit_scale", "require_clean_tree", "rescue_of_experiment_id",
    "rescue_index", "ctpm_failure_result", "ctpm_failure_result_sha256",
    "brpl_failure_result", "brpl_failure_result_sha256",
    "dial_failure_result", "dial_failure_result_sha256",
    "test_used_for_selection", "test_used_for_hyperparameter_selection",
    "nested_official_test_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "human_annotations_used", "expert_attributes_used",
    "teacher_or_distillation_used", "online_pclr_used",
}


class TeeStream:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    parent = config.get("parent_metrics_percent", {}) if isinstance(config, dict) else {}
    invalid = (
        not isinstance(config, dict) or actual != CONFIG_KEYS
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != "V6-TRY-010-R3"
        or config.get("framework_id") != "FRAMEWORK-V6-DEVELOPMENT"
        or config.get("condition_id") != "PFDE_FIXED200"
        or config.get("code_parent_commit") != "52b511d77b4ad048f35b40dc3cbd9afd092167e9"
        or config.get("dataset") != "CUB"
        or int(config.get("random_seed", -1)) != 7
        or int(config.get("batch_size", 0)) != 50
        or int(config.get("nominal_epochs", 0)) != 200
        or int(config.get("total_updates", 0)) != 28228
        or int(config.get("eval_interval_steps", 0)) != 141
        or float(config.get("required_module_delta_h", -1)) != 1.0
        or float(config.get("max_margin", -1)) != 2.0
        or float(config.get("attention_diversity_weight", -1)) != 0.01
        or float(config.get("semantic_learning_rate", -1)) != 3e-5
        or float(config.get("visual_interaction_learning_rate", -1)) != 3e-4
        or float(config.get("semantic_min_learning_rate", -1)) != 3e-6
        or float(config.get("visual_interaction_min_learning_rate", -1)) != 3e-5
        or config.get("rescue_of_experiment_id") != "V6-TRY-010"
        or int(config.get("rescue_index", -1)) != 3
        or abs(float(config.get("logit_scale", 0)) - 1.0 / 0.07) > 1e-12
        or abs(float(parent.get("H", -1)) - 63.192339) > 1e-6
        or config.get("require_clean_tree") is not True
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not True
        or config.get("nested_official_test_selection") is not True
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
        or config.get("expert_attributes_used") is not False
        or config.get("teacher_or_distillation_used") is not False
        or config.get("online_pclr_used") is not False
        or abs(float(config.get("semantic_parent_metrics_percent", {}).get("H", -1)) - 68.750566) > 1e-6
    )
    if invalid:
        raise ValueError("PFDE config identity, budget, parent, or disclosure mismatch.")
    return config, sha256_file(path)


def evaluation_updates(total: int, interval: int) -> set[int]:
    return set(range(0, total + 1, interval)) | {total}


def _patch_batch(patches: np.memmap, indices: torch.Tensor, device):
    rows = np.asarray(indices.cpu(), dtype=np.int64)
    return torch.from_numpy(np.asarray(patches[rows], dtype=np.float32).copy()).to(device)


def build_model(config: Mapping[str, Any], assets: CTPMTrainAssets, device):
    return CTPMModel(
        assets.class_name_embeds, assets.role_sentence_embeds,
        scale=float(config["logit_scale"]), hidden_dim=int(config["hidden_dim"]),
        patch_projection_dim=int(config["patch_projection_dim"]),
        max_margin=float(config["max_margin"]),
        max_role_weight=float(config["max_role_weight"]),
    ).to(device)


def _component_gradient_norms(model: CTPMModel) -> dict[str, float]:
    groups = {
        "role_weights": (model.raw_role_weights,),
        "semantic_hidden": tuple(model.semantic_margin.net[0].parameters()),
        "semantic_output": tuple(model.semantic_margin.net[-1].parameters()),
        "patch_query": tuple(model.patch_query.parameters()),
        "patch_key": tuple(model.patch_key.parameters()),
        "visual_hidden": tuple(model.visual_margin.net[0].parameters()),
        "visual_output": tuple(model.visual_margin.net[-1].parameters()),
        "interaction_hidden": tuple(model.interaction_margin.net[0].parameters()),
        "interaction_output": tuple(model.interaction_margin.net[-1].parameters()),
    }
    result = {}
    for name, params in groups.items():
        value = 0.0
        for parameter in params:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"DIAL non-finite gradient: {name}")
                value += float(parameter.grad.detach().float().norm().cpu())
        result[name] = value
    return result


def _group_gradient_norms(model: CTPMModel) -> dict[str, float]:
    return {
        name: sum(
            float(parameter.grad.detach().float().norm().cpu())
            for parameter in parameters
            if parameter.grad is not None
        )
        for name, parameters in model.parameter_groups().items()
    }


def pfde_losses(
    model: CTPMModel,
    output,
    labels: torch.Tensor,
    seen: torch.Tensor,
    global_to_seen: torch.Tensor,
    attention_diversity_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
    """Full CE plus independent S, V, and role-patch I pair supervision."""
    full_ce = F.cross_entropy(
        output.logits.index_select(1, seen), global_to_seen[labels]
    )
    detached_base = output.base_logits.detach()

    semantic_pair_logits = (
        detached_base + pair_scatter(output.top2_local, output.d_s, output.logits.size(1))
    ).gather(1, output.top2_local)
    semantic_pair, semantic_skipped = balanced_pair_ce(
        semantic_pair_logits, output.top2_global, labels
    )

    visual_pair_logits = (
        detached_base + pair_scatter(output.top2_local, output.d_v, output.logits.size(1))
    ).gather(1, output.top2_local)
    visual_pair, visual_skipped = balanced_pair_ce(
        visual_pair_logits, output.top2_global, labels
    )

    d_i_isolated = isolated_interaction_margin(model, output)
    interaction_pair_logits = (
        detached_base + pair_scatter(output.top2_local, d_i_isolated, output.logits.size(1))
    ).gather(1, output.top2_local)
    interaction_pair, interaction_skipped = balanced_pair_ce(
        interaction_pair_logits, output.top2_global, labels
    )
    diversity = attention_diversity_loss(output.attention)
    total = (
        full_ce + semantic_pair + visual_pair + interaction_pair
        + float(attention_diversity_weight) * diversity
    )
    return total, {
        "full_ce": full_ce,
        "semantic_pair": semantic_pair,
        "visual_pair": visual_pair,
        "interaction_pair": interaction_pair,
        "diversity": diversity,
        "semantic_skipped": semantic_skipped,
        "visual_skipped": visual_skipped,
        "interaction_skipped": interaction_skipped,
    }


def first_balanced_pair_rows(model: CTPMModel, assets: CTPMTrainAssets, config) -> tuple[int, torch.Tensor]:
    """Replay the frozen stream until both live-pair targets are present."""
    generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
    count = int(assets.train_labels.numel())
    device = next(model.parameters()).device
    for update in range(int(config["total_updates"])):
        rows = torch.randperm(count, generator=generator)[: int(config["batch_size"])]
        images = assets.train_features.index_select(0, rows).to(device).float()
        labels = assets.train_labels.index_select(0, rows).to(device).long()
        patches = _patch_batch(assets.train_patches, rows, device)
        with torch.no_grad():
            output = model(images, patches, labels=labels)
            target = labels.eq(output.top2_global[:, 1])
            in_pair = labels[:, None].eq(output.top2_global).any(dim=1)
            if bool((in_pair & ~target).any()) and bool((in_pair & target).any()):
                return update, rows
    raise RuntimeError("PFDE frozen stream never contains both live-pair targets.")


def micro_contract(model: CTPMModel, assets: CTPMTrainAssets, config, device):
    update_index, rows = first_balanced_pair_rows(model, assets, config)
    images = assets.train_features.index_select(0, rows).to(device).float()
    labels = assets.train_labels.index_select(0, rows).to(device).long()
    patches = _patch_batch(assets.train_patches, rows, device)
    seen = assets.seen_classes.to(device)
    global_to_seen = torch.full((model.class_count,), -1, dtype=torch.long, device=device)
    global_to_seen[seen] = torch.arange(seen.numel(), device=device)
    model.zero_grad(set_to_none=True)
    output = model(images, patches, labels=labels)
    full_ce = F.cross_entropy(output.logits.index_select(1, seen), global_to_seen[labels])
    full_ce.backward(retain_graph=True)
    gradients = _component_gradient_norms(model)
    if any(not np.isfinite(value) or value <= 0 for value in gradients.values()):
        raise RuntimeError(f"PFDE Full CE did not reach every component: {gradients}")
    model.zero_grad(set_to_none=True)
    total, losses = pfde_losses(
        model, output, labels, seen, global_to_seen,
        float(config["attention_diversity_weight"]),
    )
    if any(bool(losses[name]) for name in (
        "semantic_skipped", "visual_skipped", "interaction_skipped"
    )):
        raise RuntimeError("PFDE micro batch unexpectedly lacks one pair target.")
    losses["semantic_pair"].backward(retain_graph=True)
    semantic_grads = _group_gradient_norms(model)
    if semantic_grads["semantic"] <= 0 or semantic_grads["visual"] != 0 or semantic_grads["interaction"] != 0:
        raise RuntimeError(f"PFDE semantic auxiliary isolation failed: {semantic_grads}")
    model.zero_grad(set_to_none=True)
    losses["visual_pair"].backward(retain_graph=True)
    visual_grads = _group_gradient_norms(model)
    if visual_grads["visual"] <= 0 or visual_grads["semantic"] != 0 or visual_grads["interaction"] != 0:
        raise RuntimeError(f"PFDE visual auxiliary isolation failed: {visual_grads}")
    model.zero_grad(set_to_none=True)
    losses["interaction_pair"].backward(retain_graph=True)
    interaction_grads = _group_gradient_norms(model)
    if interaction_grads["interaction"] <= 0 or interaction_grads["semantic"] != 0 or interaction_grads["visual"] != 0:
        raise RuntimeError(f"PFDE interaction auxiliary isolation failed: {interaction_grads}")
    model.zero_grad(set_to_none=True)
    total.backward()
    with torch.no_grad():
        variants = {
            "full": {}, "S_off": {"enable_s": False},
            "V_off": {"enable_v": False}, "I_off": {"enable_i": False},
        }
        pairs = {}
        for name, kwargs in variants.items():
            current = model(images, patches, **kwargs)
            pairs[name] = current.top2_global
            if not torch.isfinite(current.logits).all():
                raise RuntimeError(f"PFDE {name} micro logits non-finite.")
        if not torch.equal(pairs["full"], pairs["V_off"]) or not torch.equal(pairs["full"], pairs["I_off"]):
            raise RuntimeError("PFDE V/I off changed the semantic candidate pair.")
        correction_sum = float(output.correction.sum(dim=1).abs().max().cpu())
        attention_error = float((output.attention.sum(dim=-1) - 1).abs().max().cpu())
        if correction_sum > 1e-6 or attention_error > 1e-6 or float(output.attention.std()) <= 0:
            raise RuntimeError("PFDE antisymmetry or attention contract failed.")
    model.zero_grad(set_to_none=True)
    return {
        "official_test_loaded": False,
        "full_ce": float(full_ce.detach().cpu()),
        "semantic_pair": float(losses["semantic_pair"].detach().cpu()),
        "visual_pair": float(losses["visual_pair"].detach().cpu()),
        "interaction_pair": float(losses["interaction_pair"].detach().cpu()),
        "attention_diversity": float(losses["diversity"].detach().cpu()),
        "component_gradient_norms": gradients,
        "semantic_auxiliary_group_gradient_norms": semantic_grads,
        "visual_auxiliary_group_gradient_norms": visual_grads,
        "interaction_auxiliary_group_gradient_norms": interaction_grads,
        "sampling_update_index": update_index,
        "sampling_rows_sha256": hashlib.sha256(rows.numpy().astype(np.int64).tobytes()).hexdigest(),
        "pair_mask_count": int(output.pair_mask.sum()),
        "correction_sum_max_abs": correction_sum,
        "attention_sum_max_abs": attention_error,
        "attention_std": float(output.attention.std().cpu()),
    }


def module_success(metrics: Mapping[str, Any], required: float) -> bool:
    return float(metrics["H"]) > float(metrics["semantic_parent_metrics"]["H"]) and all(
        float(metrics["full_minus_off_delta"][name]["H"]) >= required
        for name in ("S_off", "V_off", "I_off")
    )


def run(
    config_path: Path,
    expected_commit: str,
    expected_config_sha: str,
    *,
    output_dir=None,
    micro_batch_only: bool = False,
):
    require_clean_code_tree()
    if current_code_commit() != expected_commit:
        raise ValueError("PFDE expected commit mismatch.")
    config, config_sha = load_config(config_path)
    if config_sha != expected_config_sha:
        raise ValueError("PFDE expected config SHA mismatch.")
    destination = prepare_output_dir(Path(output_dir or config["output_dir"]))
    log_handle = (destination / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        reproducibility = configure_reproducibility(
            int(config["random_seed"]), strict_determinism=True,
            deterministic_warn_only=False,
        )
        device = torch.device(config["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PFDE formal training requires CUDA.")
        train_assets = load_ctpm_train_assets(config)
        model = build_model(config, train_assets, device)
        micro = micro_contract(model, train_assets, config, device)
        print(f"PFDE commit={expected_commit} config_sha={config_sha}")
        print(json.dumps({"micro": micro}, sort_keys=True))
        if micro_batch_only:
            result = {
                "schema_version": "gzsl-paper.v6-pfde-micro.v1",
                "experiment_id": config["experiment_id"],
                "code_commit": expected_commit,
                "config_sha256": config_sha,
                "micro": micro,
                "train_asset_identity": train_assets.identity,
                "official_test_loaded": False,
                "unseen_images_used_for_gradient": False,
            }
            atomic_write_json(destination / "micro_batch_receipt.json", result)
            return result
        eval_assets = load_ctpm_eval_assets(config)
        groups = model.parameter_groups()
        optimizer = torch.optim.AdamW(
            [
                {"params": groups["semantic"], "lr": float(config["semantic_learning_rate"])},
                {"params": groups["visual"], "lr": float(config["visual_interaction_learning_rate"])},
                {"params": groups["interaction"], "lr": float(config["visual_interaction_learning_rate"])},
            ],
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 0.1 + 0.9 * (1.0 + np.cos(np.pi * step / int(config["total_updates"]))) / 2.0,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
        eval_points = evaluation_updates(int(config["total_updates"]), int(config["eval_interval_steps"]))
        history, best_metrics, best_state, best_update, best_zs = [], None, None, 0, None
        semantic_skips, visual_skips, interaction_skips, train_updates = 0, 0, 0, 0
        seen = train_assets.seen_classes.to(device)
        global_to_seen = torch.full((model.class_count,), -1, dtype=torch.long, device=device)
        global_to_seen[seen] = torch.arange(seen.numel(), device=device)

        for update in range(int(config["total_updates"]) + 1):
            if update in eval_points:
                metrics = evaluate(model, eval_assets, device, batch_size=int(config["eval_batch_size"]))
                row = {"evaluation_index": len(history), "update": update, **metrics}
                history.append(row)
                if update == 0 and any(
                    abs(
                        float(row["parent_metrics"][metric])
                        - float(config["parent_metrics_percent"][metric])
                    )
                    > 1e-6
                    for metric in ("U", "S", "H", "ZS")
                ):
                    raise RuntimeError("PFDE class-name parent did not reproduce gate metrics.")
                gaps = row["full_minus_off_delta"]
                print(
                    f"eval={row['evaluation_index']} update={update} H={row['H']:.6f} "
                    f"Sgap={gaps['S_off']['H']:.6f} Vgap={gaps['V_off']['H']:.6f} "
                    f"Igap={gaps['I_off']['H']:.6f}"
                )
                if best_metrics is None or float(row["H"]) > float(best_metrics["H"]):
                    best_metrics = copy.deepcopy(row)
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    best_update = update
                if best_zs is None or float(row["ZS"]) > float(best_zs["ZS"]):
                    best_zs = {
                        "update": update,
                        "U": row["U"], "S": row["S"],
                        "H": row["H"], "ZS": row["ZS"],
                    }
                atomic_torch_save(
                    destination / "checkpoint_last.pth",
                    {
                        "schema_version": SCHEMA, "experiment_id": config["experiment_id"],
                        "code_commit": expected_commit, "config": config,
                        "config_sha256": config_sha, "update": update,
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_update": best_update, "best_metrics": best_metrics,
                        "best_model_state_dict": best_state, "history": history,
                        "best_zs_observation": best_zs,
                        "semantic_auxiliary_skips": semantic_skips,
                        "visual_auxiliary_skips": visual_skips,
                        "interaction_auxiliary_skips": interaction_skips,
                        "asset_identity": train_assets.identity,
                        "reproducibility": reproducibility,
                    },
                )
            if update == int(config["total_updates"]):
                break
            model.train()
            rows = torch.randperm(train_assets.train_labels.numel(), generator=generator)[: int(config["batch_size"])]
            images = train_assets.train_features.index_select(0, rows).to(device).float()
            labels = train_assets.train_labels.index_select(0, rows).to(device).long()
            patches = _patch_batch(train_assets.train_patches, rows, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images, patches, labels=labels)
            total, losses = pfde_losses(
                model, output, labels, seen, global_to_seen,
                float(config["attention_diversity_weight"]),
            )
            semantic_skips += int(bool(losses["semantic_skipped"]))
            visual_skips += int(bool(losses["visual_skipped"]))
            interaction_skips += int(bool(losses["interaction_skipped"]))
            train_updates += 1
            if not torch.isfinite(total):
                raise FloatingPointError("PFDE loss non-finite.")
            total.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"PFDE gradient non-finite: {name}")
            optimizer.step()
            scheduler.step()

        if best_metrics is None or best_state is None or best_zs is None:
            raise RuntimeError("PFDE did not select a checkpoint.")
        model.load_state_dict(best_state, strict=True)
        final_metrics = evaluate(model, eval_assets, device, batch_size=int(config["eval_batch_size"]))
        if any(abs(float(final_metrics[name]) - float(best_metrics[name])) > 1e-6 for name in ("U", "S", "H", "ZS")):
            raise RuntimeError("PFDE best checkpoint metrics did not reproduce.")
        semantic_skip_rate = semantic_skips / max(train_updates, 1)
        visual_skip_rate = visual_skips / max(train_updates, 1)
        interaction_skip_rate = interaction_skips / max(train_updates, 1)
        if semantic_skip_rate > 0.05 or visual_skip_rate > 0.05 or interaction_skip_rate > 0.05:
            raise RuntimeError(
                "PFDE auxiliary skip rate exceeded 5%: "
                f"S={semantic_skip_rate} V={visual_skip_rate} I={interaction_skip_rate}"
            )
        model_path = destination / "model_best.pth"
        atomic_torch_save(
            model_path,
            {
                "schema_version": SCHEMA, "experiment_id": config["experiment_id"],
                "code_commit": expected_commit, "config": config,
                "config_sha256": config_sha, "best_update": best_update,
                "best_metrics": final_metrics, "model_state_dict": best_state,
                "asset_identity": train_assets.identity,
            },
        )
        atomic_write_json(destination / "evaluation_history.json", {"rows": history})
        result = {
            "schema_version": RESULT_SCHEMA, "experiment_id": config["experiment_id"],
            "code_commit": expected_commit, "config_sha256": config_sha,
            "best_update": best_update, "best_metrics": final_metrics,
            "best_zs_observation": best_zs,
            "module_success": module_success(
                final_metrics, float(config["required_module_delta_h"]),
            ),
            "h80_required": False, "micro": micro,
            "auxiliary_skip_rates": {
                "semantic": semantic_skip_rate,
                "visual": visual_skip_rate,
                "interaction": interaction_skip_rate,
            },
            "train_asset_identity": train_assets.identity,
            "eval_asset_identity": eval_assets.identity,
            "model_sha256": sha256_file(model_path),
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
        }
        atomic_write_json(destination / "metrics.json", result)
        print(json.dumps(result, sort_keys=True))
        return result
    finally:
        sys.stdout = original_stdout
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--micro-batch-only", action="store_true")
    args = parser.parse_args()
    run(
        args.config, args.expected_commit, args.expected_config_sha,
        output_dir=args.output_dir,
        micro_batch_only=args.micro_batch_only,
    )


if __name__ == "__main__": main()
