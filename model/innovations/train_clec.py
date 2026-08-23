from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.ccpe import class_conditioned_patch_scores
from model.innovations.clec import CrossLLMLocalEvidenceComposition
from model.innovations.ebc import EpisodicBiasCalibration
from model.innovations.lpsr import orthogonal_local_text_residuals
from model.innovations.train_chen_style import (
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.innovations.train_sebc import _load_main
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
    "strict_blind_claim", "feature_provenance_complete", "base_model",
    "base_model_sha256", "sdrs_model", "sdrs_model_sha256", "sebc_model",
    "sebc_model_sha256", "ccpe_model", "ccpe_model_sha256", "clre_model",
    "clre_model_sha256", "sebc_metrics_percent", "comparison_H",
    "class_name_embeddings", "class_name_embeddings_sha256", "claude_embeddings",
    "claude_embeddings_sha256", "patch_inputs", "patch_sha256", "patch_top_k",
    "patch_chunk_size", "device", "random_seed", "batch_size", "epochs", "niters",
    "report_interval", "optimizer", "learning_rate", "weight_decay",
    "max_patch_scale_residual", "inputs", "expected_sha256", "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"CLEC配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if (
        config["schema_version"] != "gzsl-paper.clec.v1"
        or config["experiment_id"] != "V2-INNOVATION-025"
        or config["idea_id"] != "IDEA-059"
    ):
        raise ValueError("CLEC身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("CLEC协议边界错误。")
    if config["feature_provenance_complete"] is not False:
        raise ValueError("CLEC输入provenance未完整，不得标成完整。")
    if set(config["patch_inputs"]) != {"train", "seen", "unseen"}:
        raise ValueError("CLEC patch输入必须包含train/seen/unseen。")
    if set(config["patch_sha256"]) != {"train", "seen", "unseen"}:
        raise ValueError("CLEC patch SHA必须包含train/seen/unseen。")
    if (
        int(config["patch_top_k"]) != 2
        or int(config["patch_chunk_size"]) != 16
        or int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
    ):
        raise ValueError("CLEC patch或Chen训练量错误。")
    if (
        config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
        or float(config["max_patch_scale_residual"]) != 0.25
        or abs(float(config["comparison_H"]) - 77.80809298394227) > 1e-9
    ):
        raise ValueError("CLEC优化参数或比较门槛错误。")
    return config, sha256_file(path)


def _precompute_scores(config, text_prototypes, device):
    scores = {}
    for split, path_text in config["patch_inputs"].items():
        path = h1.repo_path(path_text)
        if sha256_file(path) != config["patch_sha256"][split]:
            raise ValueError(f"CLEC {split} patch SHA错误。")
        patches = torch.load(path, map_location="cpu", weights_only=True)
        scores[split] = class_conditioned_patch_scores(
            patches, text_prototypes, int(config["patch_top_k"]), device,
            chunk_size=int(config["patch_chunk_size"]),
        )
        del patches
    return scores


@torch.no_grad()
def evaluate(
    parent, sdrs, calibrator, model, tensors, scores,
    seen_classes, unseen_classes, device
):
    prototypes = parent.prototypes()

    def predict(features, patch_scores, class_ids=None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = features.to(device).float()
        local = patch_scores.to(device).float()
        base = F.normalize(images, dim=-1) @ prototypes.index_select(0, ids).T * parent.scale()
        logits = sdrs(base, images, ids)
        seen_mask = torch.isin(ids.cpu(), seen_classes).to(device)
        logits = calibrator(logits, seen_mask)
        predictions = model(logits, images, local, ids).argmax(1).cpu()
        return predictions if class_ids is None else class_ids[predictions]

    seen_predictions = predict(tensors["seen_features"], scores["seen"])
    unseen_predictions = predict(tensors["unseen_features"], scores["unseen"])
    zsl_predictions = predict(
        tensors["unseen_features"], scores["unseen"], unseen_classes
    )
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
    for key in (
        "base_model", "sdrs_model", "sebc_model", "ccpe_model", "clre_model",
        "class_name_embeddings", "claude_embeddings",
    ):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"CLEC {key} SHA错误。")
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
        class_names = torch.load(
            Path(config["class_name_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        claude = torch.load(
            Path(config["claude_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        seen_classes = torch.unique(labels, sorted=True)
        all_classes = torch.arange(200)
        unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seen_classes) or not torch.equal(checked_unseen, unseen_classes):
            raise ValueError("CLEC CUB类别边界错误。")

        parent, sdrs = _load_main(
            config, sentence, labels, features, class_names, seen_classes, device
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

        ccpe_payload = torch.load(
            Path(config["ccpe_model"]), map_location="cpu", weights_only=False
        )
        ccpe_beta = float(
            float(ccpe_payload["config"]["max_beta"])
            * torch.tanh(ccpe_payload["ccpe_state_dict"]["raw_beta"].float())
        )
        clre_payload = torch.load(
            Path(config["clre_model"]), map_location="cpu", weights_only=False
        )
        clre_beta = float(
            float(clre_payload["config"]["max_beta"])
            * torch.tanh(clre_payload["clre_state_dict"]["raw_beta"].float())
        )
        model = CrossLLMLocalEvidenceComposition(
            claude, clre_beta, ccpe_beta,
            float(config["max_patch_scale_residual"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
        )

        local_text = orthogonal_local_text_residuals(sentence, class_names)
        print("precomputing CLEC top2 patch scores")
        scores = _precompute_scores(config, local_text, device)
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generator = torch.Generator().manual_seed(seed)
        prototypes = parent.prototypes().detach()
        scale = parent.scale().detach()
        class_ids = seen_classes.to(device)
        seen_mask = torch.ones(150, dtype=torch.bool, device=device)

        best_metrics = evaluate(
            parent, sdrs, calibrator, model, official, scores,
            seen_classes, unseen_classes, device
        )
        initial_metrics = dict(best_metrics)
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = -1
        history = []
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "clec_state_dict": best_state,
                "initial_composition_metrics_percent": initial_metrics,
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
            local = scores["train"].index_select(0, indices).to(device).float()
            targets = mapping[labels.index_select(0, indices)].to(device)
            base = F.normalize(images, dim=-1) @ prototypes.index_select(0, class_ids).T * scale
            logits = sdrs(base, images, class_ids)
            logits = calibrator(logits, seen_mask)
            logits = model(logits, images, local, class_ids)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_gradients(model)
            optimizer.step()

            if iteration % int(config["report_interval"]) == 0:
                metrics = evaluate(
                    parent, sdrs, calibrator, model, official, scores,
                    seen_classes, unseen_classes, device
                )
                patch_scale = float(model.patch_scale().detach())
                history.append(
                    {
                        "iteration": iteration,
                        "loss": float(loss.detach()),
                        "official_metrics_percent": metrics,
                        "patch_scale": patch_scale,
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
                            "clec_state_dict": best_state,
                            "initial_composition_metrics_percent": initial_metrics,
                            "best_metrics_percent": best_metrics,
                            "selected_iteration": best_iteration,
                            "config": config,
                            "code_commit": commit,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f} patch_scale={patch_scale:.6f}"
                )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "clec_state_dict": copy.deepcopy(model.state_dict()),
                "best_state_dict": best_state,
                "initial_composition_metrics_percent": initial_metrics,
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
                "patch_files": config["patch_sha256"],
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "ccpe_model": config["ccpe_model_sha256"],
                "clre_model": config["clre_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
                "claude_embeddings": config["claude_embeddings_sha256"],
            },
        )
        best_patch_scale = float(
            1.0 + float(config["max_patch_scale_residual"])
            * torch.tanh(best_state["raw_patch_scale"])
        )
        metrics = {
            "experiment_id": config["experiment_id"],
            "idea_id": config["idea_id"],
            "run_id": run_id,
            "code_commit": commit,
            "config_sha256": config_sha,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "feature_provenance_complete": False,
            "sebc_metrics_percent": config["sebc_metrics_percent"],
            "comparison_H": float(config["comparison_H"]),
            "initial_composition_metrics_percent": initial_metrics,
            "best_metrics_percent": best_metrics,
            "selected_iteration": best_iteration,
            "fixed_clre_beta": clre_beta,
            "fixed_ccpe_beta": ccpe_beta,
            "learned_patch_scale": best_patch_scale,
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
