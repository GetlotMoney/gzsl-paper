"""Run the preregistered, parameter-free V4 GAVE error diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.innovations.gave import GeodesicAlignedVisualEvidence
from model.innovations.train_gtd_tst import build_model, load_assets, load_config
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v4-gave-diagnostic.v1"
CONFIG_KEYS = {
    "schema_version", "experiment_id", "framework_id", "dataset",
    "parent_framework_commit", "parent_run", "parent_model", "parent_model_sha256",
    "base_config", "patch_manifest", "patch_manifest_sha256", "patch_files",
    "patch_sha256", "device", "top_candidates", "visible_roles", "role_temperature",
    "residual_strength_grid", "patch_batch_size", "human_annotations_used",
    "test_used_for_selection", "test_used_for_hyperparameter_selection",
    "unseen_images_used_for_gradient", "strict_blind_claim",
}


def load_diagnostic_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"GAVE配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    strengths = [float(value) for value in config["residual_strength_grid"]]
    invalid = (
        config["schema_version"] != SCHEMA
        or config["experiment_id"] != "V4-TRY-001"
        or config["framework_id"] != "FRAMEWORK-V4"
        or config["dataset"] != "CUB"
        or config["parent_framework_commit"] != "52088f69d7ac4e574e7b63c28b21ac0da7789933"
        or config["parent_run"] != "V3-TRY-041"
        or set(config["patch_files"]) != {"test_seen", "test_unseen"}
        or set(config["patch_sha256"]) != {"test_seen", "test_unseen"}
        or int(config["top_candidates"]) != 5
        or int(config["visible_roles"]) != 3
        or float(config["role_temperature"]) != 0.07
        or strengths != [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        or int(config["patch_batch_size"]) != 16
        or config["human_annotations_used"] is not False
        or config["test_used_for_selection"] is not True
        or config["test_used_for_hyperparameter_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    )
    if invalid:
        raise ValueError("GAVE首轮诊断身份或披露边界错误。")
    return config, sha256_file(path)


def _verified_patch_arrays(config: dict, counts: dict[str, int]) -> dict[str, np.ndarray]:
    manifest_path = Path(config["patch_manifest"])
    if sha256_file(manifest_path) != config["patch_manifest_sha256"]:
        raise ValueError("GAVE patch manifest SHA错误。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = {}
    for split, path_text in config["patch_files"].items():
        filename = Path(path_text).name
        if manifest.get("outputs_sha256", {}).get(filename) != config["patch_sha256"][split]:
            raise ValueError(f"GAVE {split} patch身份未被manifest绑定。")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        expected = (int(counts[split]), 576, 768)
        if array.shape != expected or array.dtype != np.float16:
            raise ValueError(
                f"GAVE {split} patch shape/dtype错误：{array.shape}/{array.dtype}"
            )
        arrays[split] = array
    return arrays


def _candidate_predictions(
    gave: GeodesicAlignedVisualEvidence,
    features: torch.Tensor,
    patches: np.ndarray,
    prototypes: torch.Tensor,
    scale: torch.Tensor,
    role_text: torch.Tensor,
    mean8: torch.Tensor,
    value: torch.Tensor,
    strengths: list[float],
    device: torch.device,
    batch_size: int,
    candidate_pool: torch.Tensor | None = None,
) -> dict[float, torch.Tensor]:
    predictions = {alpha: [] for alpha in strengths}
    pool = None if candidate_pool is None else candidate_pool.to(device).long()
    for start in range(0, features.size(0), int(batch_size)):
        stop = min(start + int(batch_size), features.size(0))
        images = F.normalize(features[start:stop].to(device).float(), dim=-1)
        logits = images @ prototypes.T * scale
        patch_batch = torch.from_numpy(np.asarray(patches[start:stop]).copy()).to(device)
        if pool is None:
            candidates = logits.detach().topk(gave.top_candidates, dim=1).indices
        else:
            local = logits.index_select(1, pool)
            candidates = pool[local.detach().topk(gave.top_candidates, dim=1).indices]
        parts = gave.components(
            logits,
            patch_batch,
            role_text,
            mean8,
            value,
            candidate_ids=candidates,
        )
        for alpha in strengths:
            correction = torch.zeros_like(logits)
            correction.scatter_add_(
                1, candidates, float(alpha) * parts["relative_evidence"]
            )
            corrected = logits + correction
            if pool is None:
                predicted = corrected.argmax(dim=1)
            else:
                predicted = pool[corrected.index_select(1, pool).argmax(dim=1)]
            predictions[alpha].append(predicted.cpu())
    return {alpha: torch.cat(rows) for alpha, rows in predictions.items()}


def _transitions(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    old = before.eq(labels)
    new = after.eq(labels)
    return {
        "wrong_to_right": int((~old & new).sum()),
        "right_to_wrong": int((old & ~new).sum()),
        "net_correct": int(new.sum() - old.sum()),
    }


@torch.no_grad()
def run(config_path: Path, output_dir: Path, expected_commit: str) -> dict:
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("GAVE expected-commit与当前clean HEAD不一致。")
    config, config_sha = load_diagnostic_config(config_path)
    base_config, _ = load_config(Path(config["base_config"]))
    tensors = load_assets(base_config)
    device = torch.device(config["device"])
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    model = build_model(base_config, tensors, device).eval()
    parent_path = Path(config["parent_model"])
    if sha256_file(parent_path) != config["parent_model_sha256"]:
        raise ValueError("GAVE父模型SHA错误。")
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    model.load_state_dict(parent_payload["model_state_dict"], strict=True)
    model.eval()

    bundle = model.prototype_bundle()
    prototypes = bundle["final"].detach()
    scale = model.scale().detach()
    all_classes = torch.arange(model.class_count, device=device)
    mean8 = model.parent.tg_vpr.base_prototypes().detach()
    value = model.parent.tg_vpr.value_candidate(all_classes).detach()
    role_text = model.parent.tg_vpr.sentence_embeds.detach()
    seen = model.seen_classes.cpu()
    unseen = model.unseen_classes.cpu()
    patch_arrays = _verified_patch_arrays(
        config,
        {
            "test_seen": tensors["test_seen_labels"].numel(),
            "test_unseen": tensors["test_unseen_labels"].numel(),
        },
    )
    gave = GeodesicAlignedVisualEvidence(
        top_candidates=int(config["top_candidates"]),
        visible_roles=int(config["visible_roles"]),
        role_temperature=float(config["role_temperature"]),
    ).to(device).eval()
    strengths = [float(value) for value in config["residual_strength_grid"]]
    batch_size = int(config["patch_batch_size"])
    seen_predictions = _candidate_predictions(
        gave, tensors["test_seen_features"], patch_arrays["test_seen"], prototypes,
        scale, role_text, mean8, value, strengths, device, batch_size,
    )
    unseen_predictions = _candidate_predictions(
        gave, tensors["test_unseen_features"], patch_arrays["test_unseen"], prototypes,
        scale, role_text, mean8, value, strengths, device, batch_size,
    )
    zs_predictions = _candidate_predictions(
        gave, tensors["test_unseen_features"], patch_arrays["test_unseen"], prototypes,
        scale, role_text, mean8, value, strengths, device, batch_size,
        candidate_pool=unseen,
    )
    seen_labels = tensors["test_seen_labels"].long()
    unseen_labels = tensors["test_unseen_labels"].long()
    rows = []
    parent_seen = seen_predictions[0.0]
    parent_unseen = unseen_predictions[0.0]
    parent_zs = zs_predictions[0.0]
    for alpha in strengths:
        s = 100.0 * per_class_accuracy(seen_labels, seen_predictions[alpha], seen)
        u = 100.0 * per_class_accuracy(unseen_labels, unseen_predictions[alpha], unseen)
        z = 100.0 * per_class_accuracy(unseen_labels, zs_predictions[alpha], unseen)
        h = 2.0 * s * u / (s + u) if s + u else 0.0
        rows.append(
            {
                "alpha": alpha,
                "U": u,
                "S": s,
                "H": h,
                "ZS": z,
                "transitions": {
                    "seen": _transitions(parent_seen, seen_predictions[alpha], seen_labels),
                    "unseen": _transitions(parent_unseen, unseen_predictions[alpha], unseen_labels),
                    "zs": _transitions(parent_zs, zs_predictions[alpha], unseen_labels),
                },
            }
        )
    if any(abs(rows[0][metric] - float(parent_payload["best_metrics"][metric])) > 1e-5 for metric in ("U", "S", "H", "ZS")):
        raise RuntimeError("GAVE alpha0未复现V4 CUB父结果。")
    best = max(rows, key=lambda row: float(row["H"]))
    corrected = best["transitions"]["seen"]["wrong_to_right"] + best["transitions"]["unseen"]["wrong_to_right"]
    damaged = best["transitions"]["seen"]["right_to_wrong"] + best["transitions"]["unseen"]["right_to_wrong"]
    decision = (
        "keep_for_trainable_screen"
        if float(best["alpha"]) != 0.0 and float(best["H"]) > float(rows[0]["H"]) and corrected > damaged
        else "drop_before_training"
    )
    result = {
        "schema_version": "gzsl-paper.v4-gave-diagnostic-result.v1",
        "experiment_id": config["experiment_id"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "parent_run": config["parent_run"],
        "parent_model_sha256": config["parent_model_sha256"],
        "patch_manifest_sha256": config["patch_manifest_sha256"],
        "rows": rows,
        "best": copy.deepcopy(best),
        "decision": decision,
        "test_used_for_selection": True,
        "test_used_for_hyperparameter_selection": True,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
    }
    target = prepare_output_dir(output_dir)
    (target / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (target / "diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir, args.expected_commit), ensure_ascii=False))


if __name__ == "__main__":
    main()
