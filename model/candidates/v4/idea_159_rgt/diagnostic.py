"""Run the preregistered, parameter-free V4 RGT oracle diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.innovations.rgt import RefutationGatedTransport
from model.innovations.train_gtd_tst import build_model, load_assets, load_config
from tools.gzsl_data import per_class_accuracy
from tools.reproducibility import configure_reproducibility
from tools.run_contract import current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v4-rgt-diagnostic.v1"
CONFIG_KEYS = {
    "schema_version", "experiment_id", "framework_id", "dataset",
    "parent_framework_commit", "parent_run", "parent_model", "parent_model_sha256",
    "base_config", "patch_manifest", "patch_manifest_sha256", "patch_files",
    "patch_sha256", "device", "top_candidates", "visible_roles", "role_temperature",
    "attenuation_strength_grid", "patch_batch_size", "minimum_delta_h",
    "minimum_net_corrections", "human_annotations_used", "test_used_for_selection",
    "test_used_for_hyperparameter_selection", "unseen_images_used_for_gradient",
    "strict_blind_claim",
}


def load_diagnostic_config(path: Path) -> tuple[dict, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"RGT配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    strengths = [float(value) for value in config["attenuation_strength_grid"]]
    invalid = (
        config["schema_version"] != SCHEMA
        or config["experiment_id"] != "V4-TRY-002"
        or config["framework_id"] != "FRAMEWORK-V4"
        or config["dataset"] != "CUB"
        or config["parent_framework_commit"] != "52088f69d7ac4e574e7b63c28b21ac0da7789933"
        or config["parent_run"] != "V3-TRY-041"
        or set(config["patch_files"]) != {"test_seen", "test_unseen"}
        or set(config["patch_sha256"]) != {"test_seen", "test_unseen"}
        or config["device"] != "cpu"
        or int(config["top_candidates"]) != 5
        or int(config["visible_roles"]) != 3
        or float(config["role_temperature"]) != 0.07
        or strengths != [0.0, 0.25, 0.5, 0.75, 1.0]
        or int(config["patch_batch_size"]) != 16
        or float(config["minimum_delta_h"]) != 0.5
        or int(config["minimum_net_corrections"]) != 20
        or config["human_annotations_used"] is not False
        or config["test_used_for_selection"] is not True
        or config["test_used_for_hyperparameter_selection"] is not True
        or config["unseen_images_used_for_gradient"] is not False
        or config["strict_blind_claim"] is not False
    )
    if invalid:
        raise ValueError("RGT首轮诊断身份或披露边界错误。")
    return config, sha256_file(path)


def _verified_patches(config: dict, counts: dict[str, int]) -> dict[str, np.ndarray]:
    manifest_path = Path(config["patch_manifest"])
    if sha256_file(manifest_path) != config["patch_manifest_sha256"]:
        raise ValueError("RGT patch manifest SHA错误。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = {}
    for split, path_text in config["patch_files"].items():
        path = Path(path_text)
        if manifest.get("outputs_sha256", {}).get(path.name) != config["patch_sha256"][split]:
            raise ValueError(f"RGT {split} patch未被manifest绑定。")
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        expected = (int(counts[split]), 576, 768)
        if array.shape != expected or array.dtype != np.float16:
            raise ValueError(f"RGT {split} patch shape/dtype错误。")
        arrays[split] = array
    return arrays


def _predictions(
    rgt: RefutationGatedTransport,
    features: torch.Tensor,
    patches: np.ndarray,
    prototypes: torch.Tensor,
    scale: torch.Tensor,
    role_text: torch.Tensor,
    mean8: torch.Tensor,
    direction: torch.Tensor,
    theta: torch.Tensor,
    strengths: list[float],
    device: torch.device,
    batch_size: int,
    candidate_pool: torch.Tensor | None = None,
) -> dict[float, torch.Tensor]:
    predictions = {strength: [] for strength in strengths}
    pool = None if candidate_pool is None else candidate_pool.to(device).long()
    for start in range(0, features.size(0), int(batch_size)):
        stop = min(start + int(batch_size), features.size(0))
        images = F.normalize(features[start:stop].to(device).float(), dim=-1)
        logits = images @ prototypes.T * scale
        patch_batch = torch.from_numpy(np.asarray(patches[start:stop]).copy()).to(device)
        if pool is None:
            candidates = logits.detach().topk(rgt.top_candidates, dim=1).indices
        else:
            candidates = pool[
                logits.index_select(1, pool).detach().topk(rgt.top_candidates, dim=1).indices
            ]
        components = rgt.refutation_components(
            logits, patch_batch, role_text, mean8, direction, theta,
            candidate_ids=candidates,
        )
        for strength in strengths:
            corrected, _ = rgt.attenuated_logits(
                logits, images, prototypes, mean8, direction, theta, scale,
                components, strength=strength,
            )
            if pool is None:
                predicted = corrected.argmax(dim=1)
            else:
                predicted = pool[corrected.index_select(1, pool).argmax(dim=1)]
            predictions[strength].append(predicted.cpu())
    return {strength: torch.cat(rows) for strength, rows in predictions.items()}


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
        raise ValueError("RGT expected-commit与当前clean HEAD不一致。")
    config, config_sha = load_diagnostic_config(config_path)
    base_config, _ = load_config(Path(config["base_config"]))
    tensors = load_assets(base_config)
    device = torch.device(config["device"])
    configure_reproducibility(7, strict_determinism=True, deterministic_warn_only=False)
    model = build_model(base_config, tensors, device).eval()
    parent_path = Path(config["parent_model"])
    if sha256_file(parent_path) != config["parent_model_sha256"]:
        raise ValueError("RGT父模型SHA错误。")
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    model.load_state_dict(parent_payload["model_state_dict"], strict=True)
    model.eval()

    bundle = model.prototype_bundle()
    prototypes = bundle["final"].detach()
    scale = model.scale().detach()
    mean8 = model.parent.tg_vpr.base_prototypes().detach()
    role_text = model.parent.tg_vpr.sentence_embeds.detach()
    seen = model.seen_classes.cpu()
    unseen = model.unseen_classes.cpu()
    direction = torch.zeros_like(mean8)
    theta = mean8.new_zeros((model.class_count,))
    geometry = model._geometry(unseen, seen)
    direction[unseen] = geometry.direction
    theta[unseen] = bundle["theta"]
    patches = _verified_patches(
        config,
        {
            "test_seen": tensors["test_seen_labels"].numel(),
            "test_unseen": tensors["test_unseen_labels"].numel(),
        },
    )
    rgt = RefutationGatedTransport(
        top_candidates=int(config["top_candidates"]),
        visible_roles=int(config["visible_roles"]),
        role_temperature=float(config["role_temperature"]),
    ).to(device).eval()
    strengths = [float(value) for value in config["attenuation_strength_grid"]]
    batch_size = int(config["patch_batch_size"])
    seen_predictions = _predictions(
        rgt, tensors["test_seen_features"], patches["test_seen"], prototypes, scale,
        role_text, mean8, direction, theta, strengths, device, batch_size,
    )
    unseen_predictions = _predictions(
        rgt, tensors["test_unseen_features"], patches["test_unseen"], prototypes, scale,
        role_text, mean8, direction, theta, strengths, device, batch_size,
    )
    zs_predictions = _predictions(
        rgt, tensors["test_unseen_features"], patches["test_unseen"], prototypes, scale,
        role_text, mean8, direction, theta, strengths, device, batch_size,
        candidate_pool=unseen,
    )
    seen_labels = tensors["test_seen_labels"].long()
    unseen_labels = tensors["test_unseen_labels"].long()
    parent_seen = seen_predictions[0.0]
    parent_unseen = unseen_predictions[0.0]
    parent_zs = zs_predictions[0.0]
    rows = []
    for strength in strengths:
        s = 100.0 * per_class_accuracy(seen_labels, seen_predictions[strength], seen)
        u = 100.0 * per_class_accuracy(unseen_labels, unseen_predictions[strength], unseen)
        z = 100.0 * per_class_accuracy(unseen_labels, zs_predictions[strength], unseen)
        h = 2.0 * s * u / (s + u) if s + u else 0.0
        transitions = {
            "seen": _transitions(parent_seen, seen_predictions[strength], seen_labels),
            "unseen": _transitions(parent_unseen, unseen_predictions[strength], unseen_labels),
            "zs": _transitions(parent_zs, zs_predictions[strength], unseen_labels),
        }
        rows.append({"strength": strength, "U": u, "S": s, "H": h, "ZS": z, "transitions": transitions})
    if any(abs(rows[0][metric] - float(parent_payload["best_metrics"][metric])) > 1e-5 for metric in ("U", "S", "H", "ZS")):
        raise RuntimeError("RGT strength0未复现V4父结果。")
    best = max(rows, key=lambda row: float(row["H"]))
    parent = rows[0]
    net = best["transitions"]["seen"]["net_correct"] + best["transitions"]["unseen"]["net_correct"]
    delta_h = float(best["H"]) - float(parent["H"])
    zs_delta = float(best["ZS"]) - float(parent["ZS"])
    passed = (
        float(best["strength"]) != 0.0
        and delta_h >= float(config["minimum_delta_h"])
        and net >= int(config["minimum_net_corrections"])
        and zs_delta >= 0.0
    )
    result = {
        "schema_version": "gzsl-paper.v4-rgt-diagnostic-result.v1",
        "experiment_id": config["experiment_id"],
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "parent_run": config["parent_run"],
        "parent_model_sha256": config["parent_model_sha256"],
        "patch_manifest_sha256": config["patch_manifest_sha256"],
        "rows": rows,
        "best": copy.deepcopy(best),
        "delta_H": delta_h,
        "net_corrections": net,
        "delta_ZS": zs_delta,
        "decision": "keep_for_trainable_screen" if passed else "drop_before_training",
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
