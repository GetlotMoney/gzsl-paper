from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.ccpe import (
    ClassConditionedPatchEvidence,
    ClassReliabilityPatchEvidence,
    DualScalePatchEvidence,
    PatchConsensusMarginEvidence,
    class_conditioned_patch_mean_gap_scores,
    class_conditioned_patch_scores,
    multi_part_patch_scores,
    normalize_patch_scores_by_seen_reference,
    spatially_coherent_patch_scores,
)
from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
from model.candidates.v2.modules.lpsr import (
    local_text_orthogonal_reliability,
    orthogonal_local_text_residuals,
    orthogonal_part_text_residuals,
)
from model.candidates.v2.modules.lvpg import fit_local_visual_prototypes
from model.candidates.v2.trainers.train_chen_style import (
    OFFICIAL_KEYS,
    random_batch_indices,
    resolve_paths,
    verify_inputs,
)
from model.candidates.v2.trainers.train_sebc import _load_main
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
    "schema_version", "experiment_id", "idea_id", "framework_id", "dataset",
    "evaluation_protocol", "test_used_for_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim", "feature_provenance_complete", "base_model",
    "base_model_sha256", "sdrs_model", "sdrs_model_sha256", "sebc_model",
    "sebc_model_sha256", "parent_metrics_percent", "class_name_embeddings",
    "class_name_embeddings_sha256", "patch_inputs", "patch_sha256", "patch_top_k",
    "patch_chunk_size", "device", "random_seed", "batch_size", "epochs", "niters",
    "report_interval", "optimizer", "learning_rate", "weight_decay", "max_beta",
    "inputs", "expected_sha256", "class_order_sha256",
}


def load_config(path: Path):
    path = h1.repo_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    schema = config.get("schema_version") if isinstance(config, dict) else None
    expected_keys = CONFIG_KEYS
    if schema in ("gzsl-paper.dspe.v1", "gzsl-paper.dspe.v2"):
        expected_keys = expected_keys | {"normalized_max_beta"}
    if schema == "gzsl-paper.dspe.v2":
        expected_keys = expected_keys | {"ccpe_model", "ccpe_model_sha256"}
    if schema == "gzsl-paper.pcme.v1":
        expected_keys = expected_keys | {
            "ccpe_model", "ccpe_model_sha256", "gap_max_beta"
        }
    if schema == "gzsl-paper.crpe.v1":
        expected_keys = expected_keys | {
            "ccpe_model", "ccpe_model_sha256", "delta_max_beta"
        }
    if schema == "gzsl-paper.lvpg.v1":
        expected_keys = expected_keys | {"ridge"}
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"CCPE配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    identity_by_schema = {
        "gzsl-paper.ccpe.v1": ("V2-INNOVATION-015", "IDEA-049", 8, 16, 10.0),
        "gzsl-paper.ccpe.v2": ("V2-INNOVATION-015", "IDEA-049", 4, 16, 10.0),
        "gzsl-paper.ccpe.v3": ("V2-INNOVATION-015", "IDEA-049", 2, 16, 10.0),
        "gzsl-paper.scpe.v1": ("V2-INNOVATION-016", "IDEA-050", 2, 16, 10.0),
        "gzsl-paper.mppe.v1": ("V2-INNOVATION-017", "IDEA-051", 1, 4, 10.0),
        "gzsl-paper.cnpe.v1": ("V2-INNOVATION-018", "IDEA-052", 2, 16, 2.0),
        "gzsl-paper.dspe.v1": ("V2-INNOVATION-019", "IDEA-053", 2, 16, 10.0),
        "gzsl-paper.dspe.v2": ("V2-INNOVATION-019", "IDEA-053", 2, 16, 10.0),
        "gzsl-paper.pcme.v1": ("V2-INNOVATION-020", "IDEA-054", 2, 16, 10.0),
        "gzsl-paper.crpe.v1": ("V2-INNOVATION-022", "IDEA-056", 2, 16, 10.0),
        "gzsl-paper.lvpg.v1": ("V2-INNOVATION-023", "IDEA-057", 2, 16, 10.0),
    }
    identity = identity_by_schema.get(config.get("schema_version"))
    if (
        identity is None
        or config["experiment_id"] != identity[0]
        or config["idea_id"] != identity[1]
    ):
        raise ValueError("CCPE身份错误。")
    if (
        config["evaluation_protocol"] != EVALUATION_PROTOCOL
        or config["test_used_for_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    ):
        raise ValueError("CCPE协议边界错误。")
    if config["feature_provenance_complete"] is not False:
        raise ValueError("遗留CLIP patch来源未补齐，不得标成完整。")
    if set(config["patch_inputs"]) != {"train", "seen", "unseen"}:
        raise ValueError("CCPE patch输入必须包含train/seen/unseen。")
    if set(config["patch_sha256"]) != {"train", "seen", "unseen"}:
        raise ValueError("CCPE patch SHA必须包含train/seen/unseen。")
    if (
        int(config["patch_top_k"]) != identity[2]
        or int(config["patch_chunk_size"]) != identity[3]
        or int(config["batch_size"]) != 50
        or int(config["epochs"]) != 200
        or int(config["niters"]) != 28228
        or int(config["report_interval"]) != 141
    ):
        raise ValueError("CCPE patch或Chen训练量错误。")
    if (
        config["optimizer"] != "Adam"
        or float(config["learning_rate"]) != 0.01
        or float(config["weight_decay"]) != 0.0
        or float(config["max_beta"]) != identity[4]
    ):
        raise ValueError("CCPE优化参数错误。")
    if (
        config["schema_version"] in ("gzsl-paper.dspe.v1", "gzsl-paper.dspe.v2")
        and float(config["normalized_max_beta"]) != 2.0
    ):
        raise ValueError("DSPE normalized_max_beta必须为2.0。")
    if (
        config["schema_version"] == "gzsl-paper.pcme.v1"
        and float(config["gap_max_beta"]) != 5.0
    ):
        raise ValueError("PCME gap_max_beta必须为5.0。")
    if (
        config["schema_version"] == "gzsl-paper.crpe.v1"
        and float(config["delta_max_beta"]) != 2.0
    ):
        raise ValueError("CRPE delta_max_beta必须为2.0。")
    if (
        config["schema_version"] == "gzsl-paper.lvpg.v1"
        and float(config["ridge"]) != 0.1
    ):
        raise ValueError("LVPG ridge必须为0.1。")
    return config, sha256_file(path)


def _precompute_scores(config, text_prototypes, device):
    scores = {}
    for split, path_text in config["patch_inputs"].items():
        path = h1.repo_path(path_text)
        if sha256_file(path) != config["patch_sha256"][split]:
            raise ValueError(f"CCPE {split} patch SHA错误。")
        patches = torch.load(path, map_location="cpu", weights_only=True)
        if config["schema_version"] == "gzsl-paper.pcme.v1":
            scores[split] = class_conditioned_patch_mean_gap_scores(
                patches, text_prototypes, device,
                chunk_size=int(config["patch_chunk_size"]),
            )
        elif config["schema_version"] == "gzsl-paper.mppe.v1":
            scores[split] = multi_part_patch_scores(
                patches, text_prototypes, device,
                chunk_size=int(config["patch_chunk_size"]),
            )
        elif config["schema_version"] == "gzsl-paper.scpe.v1":
            scores[split] = spatially_coherent_patch_scores(
                patches, text_prototypes, device,
                chunk_size=int(config["patch_chunk_size"]),
            )
        else:
            scores[split] = class_conditioned_patch_scores(
                patches, text_prototypes, int(config["patch_top_k"]), device,
                chunk_size=int(config["patch_chunk_size"]),
            )
        del patches
    if config["schema_version"] in ("gzsl-paper.dspe.v1", "gzsl-paper.dspe.v2"):
        normalized, _, _ = normalize_patch_scores_by_seen_reference(scores)
        scores = {
            split: torch.cat((values, normalized[split]), dim=1)
            for split, values in scores.items()
        }
    elif config["schema_version"] == "gzsl-paper.cnpe.v1":
        scores, _, _ = normalize_patch_scores_by_seen_reference(scores)
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
        local_scores = patch_scores.to(device).float()
        base = F.normalize(images, dim=-1) @ prototypes.index_select(0, ids).T * parent.scale()
        logits = sdrs(base, images, ids)
        seen_mask = torch.isin(ids.cpu(), seen_classes).to(device)
        logits = calibrator(logits, seen_mask)
        predictions = model(logits, local_scores, ids).argmax(1).cpu()
        return predictions if class_ids is None else class_ids[predictions]

    seen_predictions = predict(tensors["seen_features"], scores["seen"])
    unseen_predictions = predict(tensors["unseen_features"], scores["unseen"])
    zsl_predictions = predict(
        tensors["unseen_features"], scores["unseen"], unseen_classes
    )
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
    for key in ("base_model", "sdrs_model", "sebc_model", "class_name_embeddings"):
        if sha256_file(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"CCPE {key} SHA错误。")
    if (
        config["schema_version"] in (
            "gzsl-paper.dspe.v2", "gzsl-paper.pcme.v1", "gzsl-paper.crpe.v1"
        )
        and sha256_file(Path(config["ccpe_model"])) != config["ccpe_model_sha256"]
    ):
        raise ValueError("DSPE CCPE父模型SHA错误。")
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
        names = torch.load(
            Path(config["class_name_embeddings"]), map_location="cpu", weights_only=True
        ).to(device)
        seen_classes = torch.unique(labels, sorted=True)
        all_classes = torch.arange(200)
        unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seen_classes) or not torch.equal(checked_unseen, unseen_classes):
            raise ValueError("CCPE CUB类别边界错误。")

        parent, sdrs = _load_main(
            config, sentence, labels, features, names, seen_classes, device
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

        if config["schema_version"] == "gzsl-paper.mppe.v1":
            text_residual = orthogonal_part_text_residuals(sentence, names)
        else:
            text_residual = orthogonal_local_text_residuals(sentence, names)
        if config["schema_version"] == "gzsl-paper.lvpg.v1":
            train_patch_path = h1.repo_path(config["patch_inputs"]["train"])
            if sha256_file(train_patch_path) != config["patch_sha256"]["train"]:
                raise ValueError("LVPG train patch SHA错误。")
            train_patches = torch.load(
                train_patch_path, map_location="cpu", weights_only=True
            )
            text_residual, _ = fit_local_visual_prototypes(
                train_patches, labels, text_residual, seen_classes,
                top_k=int(config["patch_top_k"]), ridge=float(config["ridge"]),
                device=device, chunk_size=int(config["patch_chunk_size"]),
            )
            del train_patches
        print("precomputing class-conditioned patch scores")
        scores = _precompute_scores(config, text_residual, device)
        score_width = (
            400
            if config["schema_version"] in (
                "gzsl-paper.dspe.v1", "gzsl-paper.dspe.v2", "gzsl-paper.pcme.v1"
            )
            else 200
        )
        if scores["train"].shape != (labels.shape[0], score_width):
            raise ValueError("CCPE train patch score形状错误。")
        if scores["seen"].shape != (official["seen_labels"].shape[0], score_width):
            raise ValueError("CCPE test-seen patch score形状错误。")
        if scores["unseen"].shape != (official["unseen_labels"].shape[0], score_width):
            raise ValueError("CCPE test-unseen patch score形状错误。")

        if config["schema_version"] == "gzsl-paper.crpe.v1":
            reliability = local_text_orthogonal_reliability(
                sentence, names, seen_classes.to(device)
            )
            model = ClassReliabilityPatchEvidence(
                reliability,
                max_absolute_beta=float(config["max_beta"]),
                max_delta_beta=float(config["delta_max_beta"]),
            ).to(device)
        elif config["schema_version"] == "gzsl-paper.pcme.v1":
            model = PatchConsensusMarginEvidence(
                max_absolute_beta=float(config["max_beta"]),
                max_gap_beta=float(config["gap_max_beta"]),
            ).to(device)
        elif config["schema_version"] in ("gzsl-paper.dspe.v1", "gzsl-paper.dspe.v2"):
            model = DualScalePatchEvidence(
                max_absolute_beta=float(config["max_beta"]),
                max_normalized_beta=float(config["normalized_max_beta"]),
            ).to(device)
        else:
            model = ClassConditionedPatchEvidence(
                max_beta=float(config["max_beta"])
            ).to(device)
        if config["schema_version"] in (
            "gzsl-paper.dspe.v2", "gzsl-paper.pcme.v1", "gzsl-paper.crpe.v1"
        ):
            ccpe_payload = torch.load(
                Path(config["ccpe_model"]), map_location="cpu", weights_only=False
            )
            model.raw_absolute_beta.data.copy_(
                ccpe_payload["ccpe_state_dict"]["raw_beta"].to(device)
            )
            model.raw_absolute_beta.requires_grad_(False)
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["learning_rate"]), weight_decay=0.0
        )
        mapping = torch.full((200,), -1, dtype=torch.long)
        mapping[seen_classes] = torch.arange(150)
        generator = torch.Generator().manual_seed(seed)
        prototypes = parent.prototypes().detach()
        scale = parent.scale().detach()
        class_ids = seen_classes.to(device)
        seen_mask = torch.ones(150, dtype=torch.bool, device=device)

        history = []
        best_metrics = evaluate(
            parent, sdrs, calibrator, model, official, scores,
            seen_classes, unseen_classes, device
        )
        expected_parent = config["parent_metrics_percent"]
        for key in ("U", "S", "H", "ZS"):
            if abs(best_metrics[key] - float(expected_parent[key])) > 1e-5:
                raise ValueError(f"CCPE关闭态未复现SEBC父指标：{key}。")
        best_h = best_metrics["H"]
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = -1
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "ccpe_state_dict": best_state,
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
            patch_scores = scores["train"].index_select(0, indices).to(device).float()
            targets = mapping[labels.index_select(0, indices)].to(device)
            base = F.normalize(images, dim=-1) @ prototypes.index_select(0, class_ids).T * scale
            logits = sdrs(base, images, class_ids)
            logits = calibrator(logits, seen_mask)
            logits = model(logits, patch_scores, class_ids)
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
                beta = (
                    {
                        "absolute": float(model.absolute_beta().detach()),
                        "normalized": float(model.normalized_beta().detach()),
                    }
                    if isinstance(model, DualScalePatchEvidence)
                    else (
                        {
                            "absolute": float(model.absolute_beta().detach()),
                            "gap": float(model.gap_beta().detach()),
                        }
                        if isinstance(model, PatchConsensusMarginEvidence)
                        else (
                            {
                                "absolute": float(model.absolute_beta().detach()),
                                "delta": float(model.delta_beta().detach()),
                            }
                            if isinstance(model, ClassReliabilityPatchEvidence)
                            else float(model.beta().detach())
                        )
                    )
                )
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
                            "ccpe_state_dict": best_state,
                            "best_metrics_percent": best_metrics,
                            "selected_iteration": best_iteration,
                            "config": config,
                            "code_commit": commit,
                            "reproducibility": reproducibility,
                        },
                    )
                print(
                    f"iter={iteration} H={metrics['H']:.6f} "
                    f"best_H={best_h:.6f} beta={beta}"
                )

        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "ccpe_state_dict": copy.deepcopy(model.state_dict()),
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
                "patch_files": config["patch_sha256"],
                "base_model": config["base_model_sha256"],
                "sdrs_model": config["sdrs_model_sha256"],
                "sebc_model": config["sebc_model_sha256"],
                "class_name_embeddings": config["class_name_embeddings_sha256"],
                **(
                    {"ccpe_model": config["ccpe_model_sha256"]}
                    if config["schema_version"] in (
                        "gzsl-paper.dspe.v2", "gzsl-paper.pcme.v1",
                        "gzsl-paper.crpe.v1"
                    )
                    else {}
                ),
            },
        )
        best_beta = (
            {
                "absolute": float(
                    torch.tanh(best_state["raw_absolute_beta"])
                    * float(config["max_beta"])
                ),
                "normalized": float(
                    torch.tanh(best_state["raw_normalized_beta"])
                    * float(config["normalized_max_beta"])
                ),
            }
            if config["schema_version"] in ("gzsl-paper.dspe.v1", "gzsl-paper.dspe.v2")
            else (
                {
                    "absolute": float(
                        torch.tanh(best_state["raw_absolute_beta"])
                        * float(config["max_beta"])
                    ),
                    "gap": float(
                        torch.tanh(best_state["raw_gap_beta"])
                        * float(config["gap_max_beta"])
                    ),
                }
                if config["schema_version"] == "gzsl-paper.pcme.v1"
                else (
                    {
                        "absolute": float(
                            torch.tanh(best_state["raw_absolute_beta"])
                            * float(config["max_beta"])
                        ),
                        "delta": float(
                            torch.tanh(best_state["raw_delta_beta"])
                            * float(config["delta_max_beta"])
                        ),
                    }
                    if config["schema_version"] == "gzsl-paper.crpe.v1"
                    else float(
                        torch.tanh(best_state["raw_beta"])
                        * float(config["max_beta"])
                    )
                )
            )
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
            "parent_metrics_percent": expected_parent,
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
