from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v4.tg import fixed_class_folds
from model.candidates.v2.modules.sccc import SampleConditionedCompetitionCalibration
from model.candidates.v2.trainers.train_chen_class_exclusive import balanced_fold_batch
from model.candidates.v2.trainers.train_chen_style import OFFICIAL_KEYS, resolve_paths, verify_inputs
from model.candidates.v2.modules.unified_seen import UnifiedSeenPrototypeModel
from model.frameworks.v2 import train as h1
from tools.cub_data import load_cub_split
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
    "schema_version", "experiment_id", "idea_id", "condition_id", "framework_id",
    "dataset", "evaluation_protocol", "test_used_for_selection",
    "unseen_images_used_for_gradient", "strict_blind_claim", "parent_model",
    "parent_model_sha256", "parent_metrics_percent", "device", "random_seed",
    "batch_size", "batch_half", "epochs", "niters", "report_interval",
    "optimizer", "learning_rate", "weight_decay", "hidden_dim", "max_gamma",
    "fold_count", "inputs", "expected_sha256", "class_order_sha256",
}
CONFIG_KEYS_V2 = CONFIG_KEYS | {"gamma_mode"}


def load_config(path: Path) -> tuple[dict, str]:
    path = h1.repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SCCC配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    expected_keys = (
        CONFIG_KEYS_V2
        if isinstance(config, dict) and config.get("schema_version") == "gzsl-paper.sccc.v2"
        else CONFIG_KEYS
    )
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"SCCC配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    if config["schema_version"] not in ("gzsl-paper.sccc.v1", "gzsl-paper.sccc.v2"):
        raise ValueError("SCCC配置schema错误。")
    if config["experiment_id"] != "V2-INNOVATION-010" or config["idea_id"] != "IDEA-044":
        raise ValueError("SCCC实验/idea身份错误。")
    if config["condition_id"] != "SCCC" or config["framework_id"] != "FRAMEWORK-V2":
        raise ValueError("SCCC condition/framework身份错误。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("SCCC协议错误。")
    if config["test_used_for_selection"] is not True or config["unseen_images_used_for_gradient"] is not False:
        raise ValueError("SCCC固定test-selected且真实unseen图像不进梯度。")
    if config["strict_blind_claim"] is not False:
        raise ValueError("SCCC不得标记strict blind。")
    if int(config["batch_size"]) != 50 or int(config["batch_half"]) != 25:
        raise ValueError("SCCC固定batch 50/half 25。")
    if int(config["epochs"]) != 200 or int(config["niters"]) != 28228 or int(config["report_interval"]) != 141:
        raise ValueError("SCCC固定200名义epoch/28228步/141评估间隔。")
    if config["optimizer"] != "Adam" or float(config["learning_rate"]) != 1e-3:
        raise ValueError("SCCC固定Adam lr=1e-3。")
    config.setdefault("gamma_mode", "signed")
    expected_gamma = (
        ("nonnegative", 0.5)
        if config["schema_version"] == "gzsl-paper.sccc.v2"
        else ("signed", 2.0)
    )
    if int(config["hidden_dim"]) != 16 or (config["gamma_mode"], float(config["max_gamma"])) != expected_gamma or int(config["fold_count"]) != 3:
        raise ValueError("SCCC gate参数错误。")
    if set(config["parent_metrics_percent"]) != {"U", "S", "H", "ZS"}:
        raise ValueError("SCCC父指标不完整。")
    return config, sha256_file(path)


def evaluate_sccc(model, parent, tensors, seenclasses, unseenclasses, device):
    model.eval(); parent.eval()
    prototypes = parent.prototypes()
    all_mask = torch.zeros(200, dtype=torch.bool, device=device)
    all_mask[seenclasses.to(device)] = True

    def predict(features):
        logits = F.normalize(features.to(device).float(), dim=-1) @ prototypes.T * parent.scale()
        adjusted = model(logits, all_mask)
        return adjusted.argmax(dim=1).cpu(), logits

    with torch.no_grad():
        seen_pred, _ = predict(tensors["seen_features"])
        unseen_pred, unseen_logits = predict(tensors["unseen_features"])
        unseen_only = unseenclasses[
            unseen_logits[:, unseenclasses.to(device)].argmax(dim=1).cpu()
        ]
    seen = h1.per_class_accuracy(tensors["seen_labels"], seen_pred, seenclasses)
    unseen = h1.per_class_accuracy(tensors["unseen_labels"], unseen_pred, unseenclasses)
    zsl = h1.per_class_accuracy(tensors["unseen_labels"], unseen_only, unseenclasses)
    harmonic = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    return {"U": unseen * 100, "S": seen * 100, "H": harmonic * 100, "ZS": zsl * 100}


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree(); code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config); input_sha = verify_inputs(config, paths)
    parent_path = Path(config["parent_model"])
    if sha256_file(parent_path) != config["parent_model_sha256"]:
        raise ValueError("SCCC父模型SHA不匹配。")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SCCC要求CUDA。")
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout; sys.stdout = h1.TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(seed, strict_determinism=True, deterministic_warn_only=False)
        sentence = torch.load(paths["sentence_embeds"], map_location="cpu", weights_only=True)
        train_features = torch.load(paths["train_features"], map_location="cpu", weights_only=True)
        train_labels = torch.load(paths["train_labels"], map_location="cpu", weights_only=True).long()
        official = {name: torch.load(paths[name], map_location="cpu", weights_only=True) for name in OFFICIAL_KEYS}
        seenclasses = torch.unique(train_labels, sorted=True); allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], train_labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seenclasses) or not torch.equal(checked_unseen, unseenclasses):
            raise RuntimeError("SCCC split不一致。")
        centroids = h1.visual_centroids(train_features, train_labels, seenclasses)
        parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        parent_config = parent_payload["config"]
        parent = UnifiedSeenPrototypeModel(
            sentence, seenclasses, centroids, active_classes=allclasses,
            dropout=float(parent_config["dropout"]), inner_ratio=float(parent_config["inner_ratio"]),
            outer_ratio=float(parent_config["outer_ratio"]), temperature=float(parent_config["temperature"]),
            transport_hidden_dim=int(parent_config["transport_hidden_dim"]),
            generator_hidden_dim=int(parent_config["generator_hidden_dim"]),
            max_transport_step=float(parent_config["max_transport_step"]),
            max_generator_magnitude=float(parent_config["max_generator_magnitude"]),
        ).to(device)
        parent.load_state_dict(parent_payload["model_state_dict"], strict=True); parent.eval()
        for parameter in parent.parameters(): parameter.requires_grad_(False)
        parent_prototypes = parent.prototypes().detach(); parent_scale = parent.scale().detach()
        model = SampleConditionedCompetitionCalibration(
            hidden_dim=int(config["hidden_dim"]), max_gamma=float(config["max_gamma"]),
            gamma_mode=config["gamma_mode"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        mapping = torch.full((200,), -1, dtype=torch.long); mapping[seenclasses] = torch.arange(150)
        folds = fixed_class_folds(seenclasses); generator = torch.Generator(device="cpu").manual_seed(seed)
        history = []; best_h = float("-inf"); best_metrics = best_state = best_iteration = best_epoch = None
        best_zs = float("-inf"); first_gradient_norm = None
        print(f"SCCC parent_H={config['parent_metrics_percent']['H']} niters={config['niters']}")
        for iteration in range(int(config["niters"])):
            fold_id = iteration % int(config["fold_count"]); pseudo_seen, pseudo_unseen = folds[fold_id]
            indices = balanced_fold_batch(train_labels, pseudo_seen, pseudo_unseen, int(config["batch_half"]), generator)
            images = train_features.index_select(0, indices).to(device).float(); targets = mapping[train_labels[indices]].to(device)
            parent_logits = F.normalize(images, dim=-1) @ parent_prototypes.index_select(0, seenclasses.to(device)).T * parent_scale
            fold_mask = torch.isin(seenclasses, pseudo_seen).to(device)
            adjusted = model(parent_logits, fold_mask)
            loss = F.cross_entropy(adjusted, targets)
            optimizer.zero_grad(set_to_none=True); loss.backward(); require_finite_gradients(model)
            if iteration == 0:
                first_gradient_norm = float(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm())
                if first_gradient_norm <= 0: raise RuntimeError("SCCC首批梯度必须非零。")
            optimizer.step()
            if iteration % int(config["report_interval"]) == 0:
                metrics = evaluate_sccc(model, parent, official, seenclasses, unseenclasses, device)
                with torch.no_grad():
                    probe_logits = F.normalize(train_features[:512].to(device).float(), dim=-1) @ parent_prototypes.T * parent_scale
                    actual_mask = torch.zeros(200, dtype=torch.bool, device=device); actual_mask[seenclasses.to(device)] = True
                    stats = model.stats(probe_logits, actual_mask)
                row = {"iteration": iteration, "nominal_epoch": iteration // int(config["report_interval"]), "loss": float(loss.detach()), "official_metrics_percent": metrics, "gamma_stats": stats}
                history.append(row); best_zs = max(best_zs, metrics["ZS"])
                if metrics["H"] > best_h:
                    best_h = metrics["H"]; best_metrics = metrics; best_iteration = iteration; best_epoch = row["nominal_epoch"]; best_state = copy.deepcopy(model.state_dict())
                    atomic_torch_save(output_dir / "model_best.pth", {"config": config, "code_commit": code_commit, "selected_iteration": best_iteration, "selected_nominal_epoch": best_epoch, "best_metrics_percent": best_metrics, "sccc_state_dict": best_state, "parent_model_sha256": config["parent_model_sha256"], "reproducibility": reproducibility})
                print(f"iter={iteration} epoch={row['nominal_epoch']} H={metrics['H']:.6f} best_H={best_h:.6f} gamma={stats['mean']:.6f}")
        atomic_torch_save(output_dir / "checkpoint_last.pth", {"config": config, "code_commit": code_commit, "sccc_state_dict": copy.deepcopy(model.state_dict()), "optimizer_state_dict": optimizer.state_dict(), "best_state_dict": best_state, "best_metrics_percent": best_metrics, "selected_iteration": best_iteration, "history": history})
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha, "parent_model": config["parent_model_sha256"]})
        payload = {"experiment_id": config["experiment_id"], "idea_id": config["idea_id"], "run_id": run_id, "framework_id": config["framework_id"], "evaluation_protocol": EVALUATION_PROTOCOL, "test_used_for_selection": True, "unseen_images_used_for_gradient": False, "strict_blind_claim": False, "gamma_mode": config["gamma_mode"], "max_gamma": float(config["max_gamma"]), "code_commit": code_commit, "config_sha256": config_sha, "parent_model_sha256": config["parent_model_sha256"], "parent_metrics_percent": config["parent_metrics_percent"], "best_metrics_percent": best_metrics, "delta_vs_parent_percent_points": {key: best_metrics[key] - float(config["parent_metrics_percent"][key]) for key in ("U", "S", "H", "ZS")}, "selected_iteration": best_iteration, "selected_nominal_epoch": best_epoch, "official_test_evaluation_count": len(history), "best_zs_observation_percent": best_zs, "first_gradient_norm": first_gradient_norm, "model_sha256": sha256_file(output_dir / "model_best.pth"), "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth")}
        atomic_write_json(output_dir / "metrics.json", payload); print(payload); return best_metrics
    finally:
        sys.stdout.flush(); sys.stdout = original_stdout; log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--expected-commit", required=True); parser.add_argument("--run-id", required=True); args = parser.parse_args(); run(args.config, args.output_dir, args.expected_commit, args.run_id)


if __name__ == "__main__": main()
