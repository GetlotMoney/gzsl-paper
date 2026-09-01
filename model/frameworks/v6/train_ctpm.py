"""Train CTPM on the Chen-style CUB split."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v6.ctpm import CTPMModel, attention_diversity_loss, pair_ce_loss
from model.frameworks.v6.ctpm_assets import DATASET_SPECS, load_ctpm_assets
from model.frameworks.v6.evaluate_ctpm import evaluate
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_torch_save, atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v6-ctpm-train.v1"
REQUIRED_KEYS = {
    "schema_version",
    "experiment_id",
    "framework_id",
    "dataset",
    "condition_id",
    "code_parent_commit",
    "base_asset_manifest",
    "base_asset_manifest_sha256",
    "base_asset_id",
    "visual_asset_manifest",
    "visual_asset_manifest_sha256",
    "visual_asset_id",
    "class_name_asset_manifest",
    "class_name_asset_manifest_sha256",
    "class_name_asset_id",
    "top2_gate_result",
    "top2_gate_result_sha256",
    "parent_metrics_percent",
    "required_module_delta_h",
    "device",
    "random_seed",
    "batch_size",
    "eval_batch_size",
    "nominal_epochs",
    "total_updates",
    "eval_interval_steps",
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "hidden_dim",
    "patch_projection_dim",
    "max_margin",
    "max_role_weight",
    "pair_loss_weight",
    "attention_diversity_weight",
    "logit_scale",
    "test_used_for_selection",
    "test_used_for_hyperparameter_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "human_annotations_used",
    "expert_attributes_used",
    "teacher_or_distillation_used",
    "online_pclr_used",
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


def load_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != REQUIRED_KEYS:
        raise ValueError(f"CTPM config fields mismatch; missing={sorted(REQUIRED_KEYS-actual)} extra={sorted(actual-REQUIRED_KEYS)}")
    spec = DATASET_SPECS.get(config["dataset"])
    if (
        config["schema_version"] != SCHEMA
        or config["experiment_id"] != "V6-TRY-010"
        or config["framework_id"] != "FRAMEWORK-V6-DEVELOPMENT"
        or config["condition_id"] != "CTPM_FULL_FIXED200"
        or config["code_parent_commit"] != "52b511d77b4ad048f35b40dc3cbd9afd092167e9"
        or spec is None
        or int(config["random_seed"]) != 7
        or int(config["batch_size"]) != 50
        or int(config["nominal_epochs"]) != 200
        or int(config["total_updates"]) != spec["train_count"] * 200 // 50
        or int(config["eval_interval_steps"]) != spec["train_count"] // 50
        or float(config["required_module_delta_h"]) != 1.0
        or float(config["max_margin"]) != 2.0
        or float(config["pair_loss_weight"]) != 0.1
        or float(config["attention_diversity_weight"]) != 0.01
        or float(config["logit_scale"]) != 1.0 / 0.07
        or config["test_used_for_selection"] is not True
        or config["test_used_for_hyperparameter_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
        or config["human_annotations_used"] is not False
        or config["expert_attributes_used"] is not False
        or config["teacher_or_distillation_used"] is not False
        or config["online_pclr_used"] is not False
    ):
        raise ValueError("CTPM config identity, budget or protocol boundary is invalid.")
    for key in ("top2_gate_result",):
        path_value = Path(config[key])
        if not path_value.is_absolute() or not path_value.is_file() or sha256_file(path_value) != config[f"{key}_sha256"]:
            raise ValueError(f"{key} path/SHA invalid.")
    parent = config["parent_metrics_percent"]
    if set(parent) != {"U", "S", "H", "ZS"}:
        raise ValueError("parent_metrics_percent must contain U/S/H/ZS.")
    return config, sha256_file(path)


def evaluation_updates(train_count: int, nominal_epochs: int, batch_size: int) -> list[int]:
    interval = train_count // batch_size
    updates = [0]
    updates.extend(1 + interval * index for index in range(int(nominal_epochs)))
    total = train_count * nominal_epochs // batch_size
    if updates[-1] != total:
        updates.append(total)
    return sorted(set(updates))


def _success(metrics: dict, parent: dict, required: float) -> bool:
    return (
        float(metrics["H"]) > float(parent["H"])
        and all(float(metrics["full_minus_off_delta"][name]["H"]) >= float(required) for name in ("S_off", "V_off", "I_off"))
    )


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str | None = None) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("CTPM expected-commit does not match clean HEAD.")
    config, config_sha = load_config(config_path)
    if expected_config_sha is not None and expected_config_sha != config_sha:
        raise ValueError("CTPM expected-config-sha mismatch.")
    output_dir = prepare_output_dir(output_dir)
    (output_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        reproducibility = configure_reproducibility(int(config["random_seed"]), strict_determinism=True, deterministic_warn_only=False)
        print(f"CTPM RUN={config['experiment_id']} commit={code_commit} config_sha={config_sha}")
        device = torch.device(config["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CTPM formal training requires CUDA.")
        assets = load_ctpm_assets(config)
        model = CTPMModel(
            assets.class_name_embeds,
            assets.role_sentence_embeds,
            scale=float(config["logit_scale"]),
            hidden_dim=int(config["hidden_dim"]),
            patch_projection_dim=int(config["patch_projection_dim"]),
            max_margin=float(config["max_margin"]),
            max_role_weight=float(config["max_role_weight"]),
        ).to(device)
        groups = model.parameter_groups()
        if any(not params for params in groups.values()):
            raise RuntimeError("CTPM S/V/I parameter groups must be non-empty.")
        optimizer = torch.optim.AdamW(
            [{"params": params, "lr": float(config["learning_rate"])} for params in groups.values()],
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["total_updates"]),
            eta_min=float(config["min_learning_rate"]),
        )
        train_features = assets.train_features.to(device).float()
        train_labels = assets.train_labels.to(device).long()
        train_patches = assets.train_patches
        generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
        eval_set = set(evaluation_updates(train_features.size(0), int(config["nominal_epochs"]), int(config["batch_size"])))
        history = []
        best_metrics = None
        best_state = copy.deepcopy(model.state_dict())
        best_update = 0
        interval_sums: dict[str, float] = {}
        interval_steps = 0

        for update in range(0, int(config["total_updates"]) + 1):
            if update in eval_set:
                metrics = evaluate(model, assets, device, batch_size=int(config["eval_batch_size"]))
                metrics.update(
                    {
                        "evaluation_index": len(history),
                        "update": update,
                        "delta_H_vs_config_parent": float(metrics["H"]) - float(config["parent_metrics_percent"]["H"]),
                        "train": {name: value / max(1, interval_steps) for name, value in interval_sums.items()},
                    }
                )
                history.append(metrics)
                gaps = metrics["full_minus_off_delta"]
                print(
                    f"eval={metrics['evaluation_index']} update={update} "
                    f"U={metrics['U']:.6f} S={metrics['S']:.6f} H={metrics['H']:.6f} ZS={metrics['ZS']:.6f} "
                    f"dH={metrics['delta_H_vs_config_parent']:.6f} "
                    f"Sgap={gaps['S_off']['H']:.6f} Vgap={gaps['V_off']['H']:.6f} Igap={gaps['I_off']['H']:.6f}"
                )
                interval_sums = {}
                interval_steps = 0
                if best_metrics is None or float(metrics["H"]) > float(best_metrics["H"]):
                    best_metrics = copy.deepcopy(metrics)
                    best_state = copy.deepcopy(model.state_dict())
                    best_update = update
                atomic_torch_save(
                    output_dir / "checkpoint_last.pth",
                    {
                        "experiment_id": config["experiment_id"],
                        "code_commit": code_commit,
                        "config": config,
                        "config_sha256": config_sha,
                        "update": update,
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_update": best_update,
                        "best_metrics": best_metrics,
                        "best_model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
                        "history": history,
                        "asset_identity": assets.identity,
                        "reproducibility": reproducibility,
                    },
                )
            if update == int(config["total_updates"]):
                break
            model.train()
            indices_cpu = torch.randperm(train_features.size(0), generator=generator)[: int(config["batch_size"])]
            indices = indices_cpu.to(device)
            images = train_features.index_select(0, indices)
            labels = train_labels.index_select(0, indices)
            patches = train_patches.index_select(0, indices_cpu).to(device).float()
            optimizer.zero_grad(set_to_none=True)
            out = model(images, patches, labels=labels)
            ce = F.cross_entropy(out.logits, labels)
            pair = pair_ce_loss(out, labels)
            diversity = attention_diversity_loss(out.attention)
            total = ce + float(config["pair_loss_weight"]) * pair + float(config["attention_diversity_weight"]) * diversity
            if not torch.isfinite(total):
                raise FloatingPointError("CTPM loss contains NaN/Inf.")
            total.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"CTPM gradient contains NaN/Inf: {name}")
            optimizer.step()
            scheduler.step()
            interval_steps += 1
            for name, value in {
                "total": total,
                "ce": ce,
                "pair_ce": pair,
                "attention_diversity": diversity,
                "lr": torch.tensor(optimizer.param_groups[0]["lr"], device=device),
                "pair_mask_rate": out.pair_mask.float().mean() if out.pair_mask is not None else torch.zeros((), device=device),
            }.items():
                interval_sums[name] = interval_sums.get(name, 0.0) + float(value.detach())

        assert best_metrics is not None
        model_best = {
            "experiment_id": config["experiment_id"],
            "code_commit": code_commit,
            "config": config,
            "config_sha256": config_sha,
            "best_update": best_update,
            "best_metrics": best_metrics,
            "model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
            "asset_identity": assets.identity,
        }
        atomic_torch_save(output_dir / "model_best.pth", model_best)
        atomic_write_json(output_dir / "evaluation_history.json", {"rows": history})
        result = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "best_update": best_update,
            "best_metrics": best_metrics,
            "parent_metrics_percent": config["parent_metrics_percent"],
            "module_success": _success(best_metrics, config["parent_metrics_percent"], float(config["required_module_delta_h"])),
            "required_module_delta_h": float(config["required_module_delta_h"]),
            "h80_required": False,
            "test_used_for_selection": True,
            "test_used_for_hyperparameter_selection": True,
            "unseen_images_used_for_gradient": False,
            "strict_blind_claim": False,
            "stop_reason": "completed_fixed_200",
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "evaluation_history_sha256": sha256_file(output_dir / "evaluation_history.json"),
        }
        atomic_write_json(output_dir / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return result
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha")
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.expected_config_sha)


if __name__ == "__main__":
    main()
