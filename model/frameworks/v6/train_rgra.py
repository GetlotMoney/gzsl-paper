"""Train RGRA with one optimizer and one Full-checkpoint selection rule."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v6.rgra import RGRAModel
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_torch_save, atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v6-rgra-train.v1"
CONFIG_KEYS = {
    "schema_version", "experiment_id", "framework_id", "dataset", "condition_id",
    "base_commit", "asset_manifest", "asset_manifest_sha256", "asset_id",
    "coarse_patch_files_sha256", "relation_asset_manifest",
    "relation_asset_manifest_sha256", "relation_asset_id", "source_checkpoint",
    "source_checkpoint_sha256", "source_code_commit", "source_config_sha256",
    "parent_metrics_percent", "target_h", "required_module_delta_h",
    "device", "random_seed", "batch_size", "nominal_epochs", "total_updates",
    "eval_interval_steps", "learning_rate", "min_learning_rate", "weight_decay",
    "hidden_dim", "topology_loss_weight", "direction_loss_weight",
    "relation_ridge", "visual_temperature", "relation_temperature",
    "seen_logit_gamma", "max_rho_s", "initial_rho_s", "max_beta_v",
    "initial_beta_v", "max_alpha", "initial_alpha",
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


def load_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if (
        not isinstance(config, dict)
        or actual != CONFIG_KEYS
        or config.get("schema_version") != SCHEMA
        or config.get("experiment_id") != "V6-TRY-008"
        or config.get("framework_id") != "FRAMEWORK-V6-DEVELOPMENT"
        or config.get("condition_id") != "RGRA_ONE_STAGE_E2E"
        or config.get("base_commit") != "52b511d77b4ad048f35b40dc3cbd9afd092167e9"
        or config.get("dataset") != "CUB"
        or int(config.get("batch_size", 0)) != 50
        or int(config.get("nominal_epochs", 0)) != 200
        or int(config.get("total_updates", 0)) != 28228
        or int(config.get("eval_interval_steps", 0)) != 141
        or config.get("test_used_for_selection") is not True
        or config.get("test_used_for_hyperparameter_selection") is not False
        or config.get("unseen_images_used_for_gradient") is not False
        or config.get("strict_blind_claim") is not False
        or config.get("human_annotations_used") is not False
        or config.get("expert_attributes_used") is not False
    ):
        raise ValueError("RGRA config identity/disclosure changed.")
    return config, sha256_file(path)


def _torch_asset(root: Path, manifest: dict, name: str):
    path = root / name
    expected = manifest["outputs_sha256"].get(name)
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"asset file mismatch: {name}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _npy_asset(root: Path, config: dict, name: str) -> torch.Tensor:
    path = root / name
    expected = config["coarse_patch_files_sha256"][name]
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"coarse patch file mismatch: {name}")
    value = torch.from_numpy(np.load(path))
    if value.ndim != 3 or tuple(value.shape[1:]) != (36, 768):
        raise ValueError(f"coarse patch shape mismatch: {name}")
    return value


def load_assets(config: dict) -> dict[str, torch.Tensor]:
    manifest_path = Path(config["asset_manifest"])
    if not manifest_path.is_file() or sha256_file(manifest_path) != config["asset_manifest_sha256"]:
        raise ValueError("asset_manifest SHA mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    if manifest.get("schema_version") != "gzsl-paper.clip-assets.v1" or manifest.get("asset_id") != config["asset_id"]:
        raise ValueError("asset manifest identity mismatch.")
    tensors = {
        "train_features": _torch_asset(root, manifest, "train_features.pt"),
        "train_labels": _torch_asset(root, manifest, "train_labels.pt"),
        "test_seen_features": _torch_asset(root, manifest, "test_seen_features.pt"),
        "test_seen_labels": _torch_asset(root, manifest, "test_seen_labels.pt"),
        "test_unseen_features": _torch_asset(root, manifest, "test_unseen_features.pt"),
        "test_unseen_labels": _torch_asset(root, manifest, "test_unseen_labels.pt"),
        "role_sentence_embeds": _torch_asset(root, manifest, "role_sentence_embeds.pt"),
        "train_coarse_patch_features": _npy_asset(root, config, "train_coarse_patch_features.npy"),
        "test_seen_coarse_patch_features": _npy_asset(root, config, "test_seen_coarse_patch_features.npy"),
        "test_unseen_coarse_patch_features": _npy_asset(root, config, "test_unseen_coarse_patch_features.npy"),
    }
    relation_manifest_path = Path(config["relation_asset_manifest"])
    if not relation_manifest_path.is_file() or sha256_file(relation_manifest_path) != config["relation_asset_manifest_sha256"]:
        raise ValueError("relation asset manifest SHA mismatch.")
    relation_manifest = json.loads(relation_manifest_path.read_text(encoding="utf-8"))
    if relation_manifest.get("asset_id") != config["relation_asset_id"] or relation_manifest.get("edge_count") != 438:
        raise ValueError("relation asset identity mismatch.")
    rel_root = relation_manifest_path.parent
    outputs = relation_manifest["outputs_sha256"]
    for name in ("relation_sentence_embeds.pt", "edge_index.pt"):
        if not (rel_root / name).is_file() or sha256_file(rel_root / name) != outputs[name]:
            raise ValueError(f"relation asset mismatch: {name}")
    tensors["relation_sentence_embeds"] = torch.load(rel_root / "relation_sentence_embeds.pt", map_location="cpu", weights_only=True)
    tensors["edge_index"] = torch.load(rel_root / "edge_index.pt", map_location="cpu", weights_only=True)
    return tensors


def _source_prototypes(config: dict, tensors: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    checkpoint_path = Path(config["source_checkpoint"])
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != config["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint SHA mismatch.")
    try:
        from model.frameworks.v4.train import build_model as build_v4_model
        source_config = {
            "schema_version": "gzsl-paper.v4-tuned-local-pclr-train.v1",
            "dataset": "CUB",
            "asset_manifest": config["asset_manifest"],
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "asset_id": config["asset_id"],
            "relation_asset_manifest": config["relation_asset_manifest"],
            "relation_asset_manifest_sha256": config["relation_asset_manifest_sha256"],
            "relation_asset_id": config["relation_asset_id"],
            "tg_checkpoint": None,
            "tg_checkpoint_sha256": None,
            "hidden_dim": 16,
            "max_transport_step": 1.5,
            "grid_points": 33,
            "reader_hidden_dim": 64,
            "reader_seed": 18601,
            "relation_temperature": 0.07,
            "ridge_lambda": 0.03,
            "potential_cap": 0.5,
            "max_beta": 0.25,
            "initial_beta": 0.05,
            "candidate_top_k": 15,
            "correction_scale": 2.38,
            "seen_logit_gamma": 0.525,
        }
        v4_tensors = {k: tensors[k] for k in ("train_features", "train_labels", "role_sentence_embeds", "relation_sentence_embeds", "edge_index")}
        model = build_v4_model(source_config, v4_tensors, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get("code_commit") != config["source_code_commit"] or checkpoint.get("config_sha256") != config["source_config_sha256"]:
            raise ValueError("source checkpoint identity mismatch.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.prototypes().detach().cpu()
    except Exception as exc:
        raise ValueError(f"could not materialize source V5 prototypes: {exc}") from exc


def build_model(config: dict, tensors: dict[str, torch.Tensor], device: torch.device) -> RGRAModel:
    labels = tensors["train_labels"].long()
    seen = torch.unique(labels, sorted=True)
    p_v5 = _source_prototypes(config, tensors, device)
    return RGRAModel(
        tensors["role_sentence_embeds"], seen, tensors["relation_sentence_embeds"],
        tensors["edge_index"], p_v5=p_v5, class_count=200,
        hidden_dim=int(config["hidden_dim"]),
        relation_ridge=float(config["relation_ridge"]),
        visual_temperature=float(config["visual_temperature"]),
        relation_temperature=float(config["relation_temperature"]),
        seen_logit_gamma=float(config["seen_logit_gamma"]),
        max_rho_s=float(config["max_rho_s"]), initial_rho_s=float(config["initial_rho_s"]),
        max_beta_v=float(config["max_beta_v"]), initial_beta_v=float(config["initial_beta_v"]),
        max_alpha=float(config["max_alpha"]), initial_alpha=float(config["initial_alpha"]),
    ).to(device)


def _metrics(pred: dict[str, torch.Tensor], tensors: dict[str, torch.Tensor], model: RGRAModel) -> dict[str, float]:
    seen = model.seen_classes.cpu()
    unseen = model.unseen_classes.cpu()
    s = 100.0 * per_class_accuracy(tensors["test_seen_labels"].long(), pred["seen"], seen)
    u = 100.0 * per_class_accuracy(tensors["test_unseen_labels"].long(), pred["unseen"], unseen)
    z = 100.0 * per_class_accuracy(tensors["test_unseen_labels"].long(), pred["zs"], unseen)
    return {"U": float(u), "S": float(s), "H": float(2.0 * s * u / (s + u) if s + u else 0.0), "ZS": float(z)}


@torch.no_grad()
def _predict(model: RGRAModel, cls_features: torch.Tensor, patch_features: torch.Tensor, device: torch.device, mode: str, class_ids: torch.Tensor | None = None) -> torch.Tensor:
    axis = torch.arange(model.class_count, device=device) if class_ids is None else class_ids.to(device).long()
    out = []
    for start in range(0, cls_features.size(0), 96):
        cls = cls_features[start:start + 96].to(device).float()
        patches = patch_features[start:start + 96].to(device).float()
        logits = model.logits(cls, patches, mode=mode, class_ids=None if class_ids is None else axis)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"RGRA {mode} logits contain NaN/Inf.")
        out.append(axis[logits.argmax(dim=1)].cpu())
    return torch.cat(out)


@torch.no_grad()
def evaluate(model: RGRAModel, tensors: dict[str, torch.Tensor], device: torch.device) -> dict:
    model.eval()
    unseen = model.unseen_classes.cpu()
    results = {}
    predictions = {}
    for mode in ("full", "s_off", "v_off", "i_off", "additive", "shuffled"):
        predictions[mode] = {
            "seen": _predict(model, tensors["test_seen_features"], tensors["test_seen_coarse_patch_features"], device, mode),
            "unseen": _predict(model, tensors["test_unseen_features"], tensors["test_unseen_coarse_patch_features"], device, mode),
            "zs": _predict(model, tensors["test_unseen_features"], tensors["test_unseen_coarse_patch_features"], device, mode, unseen),
        }
        results[mode] = _metrics(predictions[mode], tensors, model)
    full = results["full"]
    results["module_deltas"] = {name: {m: full[m] - results[name][m] for m in ("U", "S", "H", "ZS")} for name in ("s_off", "v_off", "i_off", "additive", "shuffled")}
    return results


def cls_gradient_smoke(model: RGRAModel, cls: torch.Tensor, patches: torch.Tensor, targets: torch.Tensor, seen: torch.Tensor) -> dict[str, float]:
    model.train()
    global_to_seen = torch.full((model.class_count,), -1, dtype=torch.long, device=targets.device)
    global_to_seen[seen] = torch.arange(seen.numel(), device=targets.device)
    local_targets = global_to_seen[targets]
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(cls, patches).index_select(1, seen), local_targets)
    loss.backward()
    out = {"loss": float(loss.detach())}
    for name, params in model.training_parameter_groups().items():
        total = cls.new_zeros(())
        for param in params:
            if param.grad is not None:
                total = total + param.grad.detach().float().norm()
        out[f"{name}_grad_norm"] = float(total)
        if out[f"{name}_grad_norm"] <= 0.0:
            raise RuntimeError(f"L_cls did not reach RGRA {name} parameters.")
    return out


def run(config_path: Path, output_dir: Path, expected_commit: str, expected_config_sha: str | None = None, micro_batch_only: bool = False) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("RGRA expected-commit mismatch.")
    config, config_sha = load_config(config_path)
    if expected_config_sha is not None and config_sha != expected_config_sha:
        raise ValueError("RGRA expected-config-sha mismatch.")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RGRA formal run requires CUDA.")
    tensors = load_assets(config)
    out_dir = prepare_output_dir(output_dir)
    (out_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log_handle = (out_dir / "training.log").open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = TeeStream(sys.stdout, log_handle)
    try:
        reproducibility = configure_reproducibility(int(config["random_seed"]), strict_determinism=True, deterministic_warn_only=False)
        print(f"RGRA RUN={config['experiment_id']} commit={code_commit} config_sha={config_sha}")
        print("disclosure test_used_for_selection=true unseen_images_used_for_gradient=false strict_blind_claim=false")
        model = build_model(config, tensors, device)
        train_features = tensors["train_features"].to(device).float()
        train_patches = tensors["train_coarse_patch_features"].to(device).float()
        train_labels = tensors["train_labels"].to(device).long()
        seen = torch.unique(train_labels, sorted=True)
        global_to_seen = torch.full((model.class_count,), -1, dtype=torch.long, device=device)
        global_to_seen[seen] = torch.arange(seen.numel(), device=device)
        first = torch.arange(min(int(config["batch_size"]), train_features.size(0)), device=device)
        smoke = cls_gradient_smoke(model, train_features[first], train_patches[first], train_labels[first], seen)
        with torch.no_grad():
            old_alpha = model.raw_alpha.detach().clone()
            model.raw_alpha.fill_(-80.0)
            alpha_zero = model.logits(train_features[first], train_patches[first], mode="full")
            i_off = model.logits(train_features[first], train_patches[first], mode="i_off")
            model.raw_alpha.copy_(old_alpha)
            alpha_zero_i_off_max_abs = float((alpha_zero - i_off).abs().max().detach())
        if alpha_zero_i_off_max_abs > 1e-5:
            raise RuntimeError("alpha=0 parity with I-off failed.")
        if micro_batch_only:
            result = {"schema_version": SCHEMA, "experiment_id": config["experiment_id"], "code_commit": code_commit, "config_sha256": config_sha, "micro_batch": smoke, "alpha_zero_i_off_max_abs": alpha_zero_i_off_max_abs}
            atomic_write_json(out_dir / "micro_batch_receipt.json", result)
            print(json.dumps(result, sort_keys=True))
            return result
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        batch_generator = torch.Generator(device="cpu").manual_seed(int(config["random_seed"]))
        eval_updates = set(range(0, int(config["total_updates"]) + 1, int(config["eval_interval_steps"])))
        eval_updates.add(int(config["total_updates"]))
        history = []
        best_metrics = None
        best_state = None
        best_update = 0
        best_zs = None
        for update in range(0, int(config["total_updates"]) + 1):
            if update in eval_updates:
                metrics = evaluate(model, tensors, device)
                metrics.update({"update": update, "evaluation_index": len(history)})
                history.append(metrics)
                print(f"eval={len(history)-1} update={update} U={metrics['full']['U']:.6f} S={metrics['full']['S']:.6f} H={metrics['full']['H']:.6f} ZS={metrics['full']['ZS']:.6f}")
                if best_metrics is None or metrics["full"]["H"] > best_metrics["full"]["H"]:
                    best_metrics = copy.deepcopy(metrics)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_update = update
                if best_zs is None or metrics["full"]["ZS"] > best_zs["ZS"]:
                    best_zs = {"ZS": metrics["full"]["ZS"], "update": update, "metrics": copy.deepcopy(metrics)}
                checkpoint = {"experiment_id": config["experiment_id"], "code_commit": code_commit, "config_sha256": config_sha, "update": update, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "best_update": best_update, "best_metrics": best_metrics, "best_model_state_dict": best_state, "best_zs_observation": best_zs, "history": history, "reproducibility": reproducibility, "batch_generator_state": batch_generator.get_state(), "cpu_rng_state": torch.get_rng_state(), "cuda_rng_state_all": torch.cuda.get_rng_state_all()}
                atomic_torch_save(out_dir / "checkpoint_last.pth", checkpoint)
            if update == int(config["total_updates"]):
                break
            model.train()
            indices = torch.randperm(train_features.size(0), generator=batch_generator)[: int(config["batch_size"])].to(device)
            cls = train_features.index_select(0, indices)
            patches = train_patches.index_select(0, indices)
            targets = train_labels.index_select(0, indices)
            local_targets = global_to_seen.index_select(0, targets)
            optimizer.zero_grad(set_to_none=True)
            logits = model(cls, patches).index_select(1, seen)
            loss_cls = F.cross_entropy(logits, local_targets)
            loss_topology = model.topology_loss()
            loss_direction = model.direction_loss(cls, targets)
            total = loss_cls + float(config["topology_loss_weight"]) * loss_topology + float(config["direction_loss_weight"]) * loss_direction
            if not torch.isfinite(total):
                raise FloatingPointError("RGRA loss contains NaN/Inf.")
            total.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"RGRA gradient contains NaN/Inf: {name}")
            optimizer.step()
        atomic_torch_save(out_dir / "model_best.pth", {"experiment_id": config["experiment_id"], "code_commit": code_commit, "config_sha256": config_sha, "best_update": best_update, "best_metrics": best_metrics, "model_state_dict": best_state})
        result = {"schema_version": SCHEMA, "experiment_id": config["experiment_id"], "code_commit": code_commit, "config_sha256": config_sha, "best_update": best_update, "best_metrics": best_metrics, "best_zs_observation": best_zs, "history_length": len(history), "target_h": float(config["target_h"]), "required_module_delta_h": float(config["required_module_delta_h"]), "test_used_for_selection": True, "unseen_images_used_for_gradient": False, "strict_blind_claim": False}
        atomic_write_json(out_dir / "evaluation_history.json", {"rows": history})
        atomic_write_json(out_dir / "metrics.json", result)
        print(json.dumps(result, sort_keys=True))
        return result
    finally:
        sys.stdout = original_stdout
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha")
    parser.add_argument("--micro-batch-only", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.expected_config_sha, args.micro_batch_only)


if __name__ == "__main__":
    main()

