from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.cnra import ClassNameResidualAlignment
from model.innovations.ebc import EpisodicBiasCalibration
from model.innovations.train_chen_style import (
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.innovations.train_sebc import _load_main
from model.innovations.semantic_orthogonal import classwise_bi_orthogonal_residual
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
COMMON_CONFIG_KEYS = {
    "schema_version", "experiment_id", "idea_id", "framework_id", "dataset",
    "evaluation_protocol", "test_used_for_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "text_cache_provenance_complete", "base_model",
    "base_model_sha256", "sdrs_model", "sdrs_model_sha256", "sebc_model",
    "sebc_model_sha256", "parent_metrics_percent", "comparison_H",
    "class_name_embeddings", "class_name_embeddings_sha256", "device", "random_seed",
    "batch_size", "epochs", "niters", "report_interval", "optimizer",
    "learning_rate", "weight_decay", "max_beta", "inputs", "expected_sha256",
    "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    identity_by_schema = {
        "gzsl-paper.clre.v1": (
            "V2-INNOVATION-024", "IDEA-058", "claude_embeddings",
            77.6665326315915,
        ),
        "gzsl-paper.mlre.v1": (
            "V2-INNOVATION-026", "IDEA-060", "merge_embeddings",
            77.80809298394227,
        ),
        "gzsl-paper.oclr.v1": (
            "V2-INNOVATION-029", "IDEA-063", "claude_embeddings",
            77.82913952565472,
        ),
        "gzsl-paper.omlr.v1": (
            "V2-INNOVATION-031", "IDEA-065", "merge_embeddings",
            78.0721851209539,
        ),
        "gzsl-paper.bocr.v1": (
            "V2-INNOVATION-032", "IDEA-066", "claude_embeddings",
            78.0721851209539,
        ),
    }
    identity = identity_by_schema.get(config.get("schema_version")) if isinstance(config, dict) else None
    cache_key = identity[2] if identity is not None else "unknown_embeddings"
    expected_keys = COMMON_CONFIG_KEYS | {cache_key, f"{cache_key}_sha256"}
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"CLRE配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    if (
        identity is None
        or config["experiment_id"] != identity[0]
        or config["idea_id"] != identity[1]
    ):
        raise ValueError("CLRE身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("CLRE协议边界错误。")
    if config["text_cache_provenance_complete"] is not False:
        raise ValueError("Claude cache来源未完整，不得标成完整。")
    if (
        int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
    ):
        raise ValueError("CLRE Chen训练量错误。")
    if (
        config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
        or float(config["max_beta"]) != 20.0
        or abs(float(config["comparison_H"]) - identity[3]) > 1e-9
    ):
        raise ValueError("CLRE优化参数或比较门槛错误。")
    return config, sha256_file(path)


@torch.no_grad()
def evaluate(
    parent, sdrs, calibrator, model, tensors, seen_classes, unseen_classes, device
):
    prototypes = parent.prototypes()

    def predict(features, class_ids=None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = features.to(device).float()
        base = F.normalize(images, dim=-1) @ prototypes.index_select(0, ids).T * parent.scale()
        logits = sdrs(base, images, ids)
        seen_mask = torch.isin(ids.cpu(), seen_classes).to(device)
        logits = calibrator(logits, seen_mask)
        predictions = model(logits, images, ids).argmax(1).cpu()
        return predictions if class_ids is None else class_ids[predictions]

    seen_predictions = predict(tensors["seen_features"])
    unseen_predictions = predict(tensors["unseen_features"])
    zsl_predictions = predict(tensors["unseen_features"], unseen_classes)
    seen = h1.per_class_accuracy(tensors["seen_labels"], seen_predictions, seen_classes)
    unseen = h1.per_class_accuracy(
        tensors["unseen_labels"], unseen_predictions, unseen_classes
    )
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
    cache_key = (
        "claude_embeddings"
        if config["schema_version"] in (
            "gzsl-paper.clre.v1", "gzsl-paper.oclr.v1", "gzsl-paper.bocr.v1"
        )
        else "merge_embeddings"
    )
    for key in (
        "base_model", "sdrs_model", "sebc_model",
        "class_name_embeddings", cache_key,
    ):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"CLRE {key} SHA错误。")
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
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seen_classes) or not torch.equal(checked_unseen, unseen_classes):
            raise ValueError("CLRE CUB类别边界错误。")

        names = torch.load(
            Path(config[cache_key]), map_location="cpu", weights_only=True
        ).to(device)
        if tuple(names.shape) != (200, 768):
            raise ValueError("Claude原型必须是[200,768]。")
        class_names = torch.load(
            Path(config["class_name_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        if config["schema_version"] in ("gzsl-paper.oclr.v1", "gzsl-paper.omlr.v1"):
            normalized_names = F.normalize(class_names.float(), dim=-1)
            normalized_residual = F.normalize(names.float(), dim=-1)
            names = F.normalize(
                normalized_residual
                - (normalized_residual * normalized_names).sum(
                    dim=-1, keepdim=True
                ) * normalized_names,
                dim=-1,
            )
        parent, sdrs = _load_main(
            config, sentence, labels, features, class_names, seen_classes, device
        )
        if config["schema_version"] == "gzsl-paper.bocr.v1":
            names = classwise_bi_orthogonal_residual(
                names, class_names, parent.prototypes().detach()
            )
        sebc_payload = torch.load(
            Path(config["sebc_model"]), map_location="cpu", weights_only=False
        )
        calibrator = EpisodicBiasCalibration(
            float(sebc_payload["config"]["max_gamma"])
        ).to(device)
        calibrator.load_state_dict(
            sebc_payload["calibrator_state_dict"], strict=True
        )
        calibrator.eval()
        for parameter in calibrator.parameters():
            parameter.requires_grad_(False)

        model = ClassNameResidualAlignment(names, float(config["max_beta"])).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
        )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generator = torch.Generator().manual_seed(seed)
        prototypes = parent.prototypes().detach()
        scale = parent.scale().detach()
        class_ids = seen_classes.to(device)
        seen_mask = torch.ones(150, dtype=torch.bool, device=device)

        best_metrics = evaluate(
            parent, sdrs, calibrator, model, official,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"CLRE关闭态未复现SEBC父指标：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = -1
        history = []
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "clre_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "config": config,
                "code_commit": commit,
                "reproducibility": reproducibility,
            },
        )

        for iteration in range(int(config["niters"])):
            indices = random_batch_indices(
                labels.numel(), int(config["batch_size"]), generator
            )
            images = features.index_select(0, indices).to(device).float()
            targets = mapping[labels.index_select(0, indices)].to(device)
            base = F.normalize(images, dim=-1) @ prototypes.index_select(0, class_ids).T * scale
            logits = sdrs(base, images, class_ids)
            logits = calibrator(logits, seen_mask)
            logits = model(logits, images, class_ids)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(model)
            optimizer.step()

            if iteration % int(config["report_interval"]) == 0:
                metrics = evaluate(
                    parent, sdrs, calibrator, model, official,
                    seen_classes, unseen_classes, device
                )
                beta = float(model.beta().detach())
                history.append(
                    {
                        "iteration": iteration,
                        "loss": float(loss.detach()),
                        "official_metrics_percent": metrics,
                        "beta": beta,
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
                            "clre_state_dict": best_state,
                            "best_metrics_percent": best_metrics,
                            "selected_iteration": best_iteration,
                            "config": config,
                            "code_commit": commit,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f} beta={beta:.6f}"
                )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "clre_state_dict": copy.deepcopy(model.state_dict()),
                "best_state_dict": best_state,
                "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration,
                "history": history,
                "config": config,
                "code_commit": commit,
            },
        )
        atomic_write_json(
            output_dir / "data_fingerprints.json",
            {
                "files": input_sha,
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
                cache_key: config[f"{cache_key}_sha256"],
            },
        )
        best_beta = float(
            torch.tanh(best_state["raw_beta"]) * float(config["max_beta"])
        )
        metrics = {
            "experiment_id": config["experiment_id"],
            "idea_id": config["idea_id"],
            "run_id": run_id,
            "code_commit": commit,
            "config_sha256": config_sha,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "text_cache_provenance_complete": False,
            "parent_metrics_percent": expected_parent,
            "comparison_H": float(config["comparison_H"]),
            "best_metrics_percent": best_metrics,
            "delta_vs_parent_percent_points": {
                key: best_metrics[key] - float(expected_parent[key])
                for key in ("U", "S", "H", "ZS")
            },
            "selected_iteration": best_iteration,
            "learned_beta": best_beta,
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
