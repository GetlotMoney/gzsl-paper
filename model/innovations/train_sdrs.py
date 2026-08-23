from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.sdrs import SemanticDisagreementResidualScaling
from model.innovations.train_chen_style import (
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.innovations.unified_seen import UnifiedSeenPrototypeModel
from model.tg_vpr_h1 import train as h1
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
    "schema_version", "experiment_id", "idea_id", "framework_id", "dataset",
    "evaluation_protocol", "test_used_for_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "base_model", "base_model_sha256", "ncra_model",
    "ncra_model_sha256", "parent_metrics_percent", "class_name_embeddings",
    "class_name_embeddings_sha256", "device", "random_seed", "batch_size", "epochs",
    "niters", "report_interval", "optimizer", "learning_rate", "weight_decay",
    "max_delta", "inputs", "expected_sha256", "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"SDRS配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    delta_by_schema = {
        "gzsl-paper.sdrs.v1": 5.0,
        "gzsl-paper.sdrs.v2": 0.5,
    }
    if (
        config["schema_version"] not in delta_by_schema
        or config["experiment_id"] != "V2-INNOVATION-012"
        or config["idea_id"] != "IDEA-046"
    ):
        raise ValueError("SDRS身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("SDRS协议边界错误。")
    if (
        int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
    ):
        raise ValueError("SDRS Chen训练量错误。")
    if (
        config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
        or float(config["max_delta"]) != delta_by_schema[config["schema_version"]]
    ):
        raise ValueError("SDRS优化参数错误。")
    return config, sha256_file(path)


def evaluate(model, parent, tensors, seen_classes, unseen_classes, device):
    parent.eval()
    model.eval()
    prototypes = parent.prototypes()

    def predict(features, class_ids=None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = features.to(device).float()
        base = F.normalize(images, dim=-1) @ prototypes.index_select(0, ids).T * parent.scale()
        predictions = model(base, images, ids).argmax(1).cpu()
        return predictions if class_ids is None else class_ids[predictions]

    with torch.no_grad():
        seen_predictions = predict(tensors["seen_features"])
        unseen_predictions = predict(tensors["unseen_features"])
        zsl_predictions = predict(tensors["unseen_features"], unseen_classes)
    seen = h1.per_class_accuracy(tensors["seen_labels"], seen_predictions, seen_classes)
    unseen = h1.per_class_accuracy(tensors["unseen_labels"], unseen_predictions, unseen_classes)
    zsl = h1.per_class_accuracy(tensors["unseen_labels"], zsl_predictions, unseen_classes)
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
    base_path = Path(config["base_model"])
    ncra_path = Path(config["ncra_model"])
    names_path = Path(config["class_name_embeddings"])
    if sha256_file(base_path) != config["base_model_sha256"]:
        raise ValueError("SDRS基础模型SHA错误。")
    if sha256_file(ncra_path) != config["ncra_model_sha256"]:
        raise ValueError("SDRS NCRA父模型SHA错误。")
    if sha256_file(names_path) != config["class_name_embeddings_sha256"]:
        raise ValueError("SDRS类名cache SHA错误。")

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
        seen_classes = torch.unique(labels, sorted=True)
        all_classes = torch.arange(200)
        unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
        cub_seen, cub_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(cub_seen, seen_classes) or not torch.equal(cub_unseen, unseen_classes):
            raise ValueError("CUB类别边界与cache不一致。")

        centroids = h1.visual_centroids(features, labels, seen_classes)
        base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
        parent_config = base_payload["config"]
        parent = UnifiedSeenPrototypeModel(
            sentence, seen_classes, centroids, active_classes=all_classes,
            dropout=float(parent_config["dropout"]),
            inner_ratio=float(parent_config["inner_ratio"]),
            outer_ratio=float(parent_config["outer_ratio"]),
            temperature=float(parent_config["temperature"]),
            transport_hidden_dim=int(parent_config["transport_hidden_dim"]),
            generator_hidden_dim=int(parent_config["generator_hidden_dim"]),
            max_transport_step=float(parent_config["max_transport_step"]),
            max_generator_magnitude=float(parent_config["max_generator_magnitude"]),
        ).to(device)
        parent.load_state_dict(base_payload["model_state_dict"], strict=True)
        parent.eval()
        for parameter in parent.parameters():
            parameter.requires_grad_(False)

        names = torch.load(names_path, map_location="cpu", weights_only=True).to(device)
        ncra_payload = torch.load(ncra_path, map_location="cpu", weights_only=False)
        ncra_max_beta = float(ncra_payload["config"]["max_beta"])
        raw_beta = ncra_payload["ncra_state_dict"]["raw_beta"].float()
        base_beta = float(ncra_max_beta * torch.tanh(raw_beta))
        prototypes = parent.prototypes().detach()
        scale = parent.scale().detach()
        model = SemanticDisagreementResidualScaling(
            prototypes, names, seen_classes.to(device), base_beta,
            max_delta=float(config["max_delta"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
        )
        label_mapping = torch.full((200,), -1, dtype=torch.long)
        label_mapping[seen_classes] = torch.arange(150)
        generator = torch.Generator().manual_seed(seed)

        history = []
        best_metrics = evaluate(
            model, parent, official, seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"SDRS关闭态未复现NCRA父指标：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = -1
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "sdrs_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "base_beta": base_beta,
                "config": config,
                "code_commit": commit,
                "reproducibility": reproducibility,
            },
        )

        for iteration in range(int(config["niters"])):
            indices = random_batch_indices(
                labels.numel(), int(config["batch_size"]), generator
            )
            images = features[indices].to(device).float()
            targets = label_mapping[labels[indices]].to(device)
            base_logits = (
                F.normalize(images, dim=-1)
                @ prototypes.index_select(0, seen_classes.to(device)).T
                * scale
            )
            logits = model(base_logits, images, seen_classes)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(model)
            optimizer.step()

            if iteration % int(config["report_interval"]) == 0:
                metrics = evaluate(
                    model, parent, official, seen_classes, unseen_classes, device
                )
                delta = float(model.delta().detach())
                history.append(
                    {
                        "iteration": iteration,
                        "official_metrics_percent": metrics,
                        "delta": delta,
                        "loss": float(loss.detach()),
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
                            "sdrs_state_dict": best_state,
                            "best_metrics_percent": best_metrics,
                            "selected_iteration": best_iteration,
                            "base_beta": base_beta,
                            "config": config,
                            "code_commit": commit,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f} delta={delta:.6f}"
                )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "sdrs_state_dict": copy.deepcopy(model.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "history": history,
                "base_beta": base_beta,
                "config": config,
                "code_commit": commit,
            },
        )
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                "base_model": config["base_model_sha256"],
                "ncra_model": config["ncra_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
            },
        )
        best_raw = best_state["raw_slope"]
        best_delta = float(torch.tanh(best_raw) * float(config["max_delta"]))
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
            "delta_vs_parent_percent_points": {
                key: best_metrics[key] - float(expected_parent[key])
                for key in ("U", "S", "H", "ZS")
            },
            "selected_iteration": best_iteration,
            "base_beta": base_beta,
            "learned_delta": best_delta,
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
