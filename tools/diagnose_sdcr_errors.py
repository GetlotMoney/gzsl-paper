from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
from model.candidates.v2.modules.sdcr import SentenceDropoutConservativeRouting
from model.candidates.v2.trainers.train_chen_style import OFFICIAL_KEYS, resolve_paths
from model.candidates.v2.trainers.train_sebc import _load_main
from model.frameworks.v2 import train as h1
from tools.cub_data import load_cub_split
from tools.run_contract import atomic_write_json
from tools.runtime import sha256_file


def _unwrap_name(value) -> str:
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return ""
        current = current.reshape(-1)[0]
    return str(current)


def load_class_names(att_splits: Path) -> list[str]:
    payload = sio.loadmat(att_splits)
    raw = payload.get("allclasses_names")
    if raw is None or raw.size != 200:
        return [f"class_{index:03d}" for index in range(200)]
    return [_unwrap_name(value) for value in raw.reshape(-1)]


def summarize_predictions(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    logits: torch.Tensor,
    class_names: list[str],
    seen_mask: torch.Tensor,
    top_confusions: int = 20,
) -> dict[str, object]:
    labels = labels.long().cpu()
    predictions = predictions.long().cpu()
    logits = logits.float().cpu()
    correct = predictions.eq(labels)
    margins = logits.topk(2, dim=1).values
    margins = margins[:, 0] - margins[:, 1]
    confusion = Counter(
        (int(target), int(prediction))
        for target, prediction in zip(labels.tolist(), predictions.tolist())
        if target != prediction
    )
    per_class = []
    for class_id in labels.unique(sorted=True).tolist():
        mask = labels.eq(int(class_id))
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "count": int(mask.sum()),
                "accuracy_percent": float(correct[mask].float().mean() * 100),
            }
        )
    accuracies = torch.tensor(
        [item["accuracy_percent"] for item in per_class], dtype=torch.float32
    )
    predicted_seen = seen_mask.index_select(0, predictions)
    true_seen = seen_mask.index_select(0, labels)
    top_pairs = []
    for (target, prediction), count in confusion.most_common(int(top_confusions)):
        top_pairs.append(
            {
                "true_class_id": target,
                "true_class_name": class_names[target],
                "predicted_class_id": prediction,
                "predicted_class_name": class_names[prediction],
                "count": count,
                "true_domain": "seen" if bool(seen_mask[target]) else "unseen",
                "predicted_domain": "seen" if bool(seen_mask[prediction]) else "unseen",
            }
        )
    return {
        "sample_count": int(labels.numel()),
        "sample_accuracy_percent": float(correct.float().mean() * 100),
        "correct_count": int(correct.sum()),
        "wrong_count": int((~correct).sum()),
        "true_seen_predicted_unseen_rate_percent": float(
            (true_seen & ~predicted_seen).float().mean() * 100
        ),
        "true_unseen_predicted_seen_rate_percent": float(
            (~true_seen & predicted_seen).float().mean() * 100
        ),
        "correct_margin_mean": float(margins[correct].mean()) if bool(correct.any()) else None,
        "wrong_margin_mean": float(margins[~correct].mean()) if bool((~correct).any()) else None,
        "per_class_accuracy_percent": {
            "mean": float(accuracies.mean()),
            "min": float(accuracies.min()),
            "q25": float(torch.quantile(accuracies, 0.25)),
            "median": float(torch.quantile(accuracies, 0.5)),
            "q75": float(torch.quantile(accuracies, 0.75)),
            "max": float(accuracies.max()),
        },
        "worst_classes": sorted(
            per_class, key=lambda item: (item["accuracy_percent"], item["class_id"])
        )[:20],
        "top_confusions": top_pairs,
    }


@torch.no_grad()
def run(
    config_path: Path,
    sdcr_model_path: Path,
    output_json: Path,
    device_text: str,
) -> dict[str, object]:
    config_path = h1.repo_path(config_path)
    config = __import__(
        "model.candidates.v2.trainers.train_sdcr", fromlist=["load_config"]
    ).load_config(config_path)[0]
    paths = resolve_paths(config)
    device = torch.device(device_text)
    sentence = torch.load(paths["sentence_embeds"], map_location="cpu", weights_only=True)
    features = torch.load(paths["train_features"], map_location="cpu", weights_only=True)
    labels = torch.load(paths["train_labels"], map_location="cpu", weights_only=True).long()
    official = {
        name: torch.load(paths[name], map_location="cpu", weights_only=True)
        for name in OFFICIAL_KEYS
    }
    class_names_tensor = torch.load(
        Path(config["class_name_embeddings"]), map_location="cpu", weights_only=True
    ).to(device)
    sentence8 = torch.load(
        Path(config["eight_sentence_embeddings"]), map_location="cpu", weights_only=True
    ).to(device)
    seen_classes = torch.unique(labels, sorted=True)
    all_classes = torch.arange(200)
    unseen_classes = all_classes[~torch.isin(all_classes, seen_classes)]
    checked_seen, checked_unseen = load_cub_split(
        paths["res101"], paths["att_splits"], labels,
        official["seen_labels"], official["unseen_labels"], "cpu"
    )
    if not torch.equal(checked_seen, seen_classes) or not torch.equal(
        checked_unseen, unseen_classes
    ):
        raise ValueError("SDCR诊断类别边界错误。")
    parent, sdrs = _load_main(
        config, sentence, labels, features, class_names_tensor, seen_classes, device
    )
    calibrator_payload = torch.load(
        Path(config["sebc_model"]), map_location="cpu", weights_only=False
    )
    calibrator = EpisodicBiasCalibration(
        float(calibrator_payload["config"]["max_gamma"])
    ).to(device)
    calibrator.load_state_dict(calibrator_payload["calibrator_state_dict"], strict=True)
    calibrator.eval()
    casr_payload = torch.load(
        Path(config["casr_model"]), map_location="cpu", weights_only=False
    )
    sdcr_model_path = sdcr_model_path.resolve()
    sdcr_payload = torch.load(
        sdcr_model_path, map_location="cpu", weights_only=False
    )
    base_weights = torch.softmax(
        casr_payload["aosr_state_dict"]["raw_sentence_weights"].float(), dim=0
    ).to(device)
    fixed_beta = float(sdcr_payload["fixed_beta"])
    sdcr = SentenceDropoutConservativeRouting(
        sentence8,
        class_names_tensor,
        base_weights,
        fixed_beta,
        float(sdcr_payload["config"]["max_logit_residual"]),
        int(sdcr_payload["config"].get("drop_count", 1)),
    ).to(device)
    sdcr.load_state_dict(sdcr_payload["sdcr_state_dict"], strict=True)
    sdcr.eval()
    prototypes = parent.prototypes()

    def infer(split_features: torch.Tensor, class_ids: torch.Tensor | None = None):
        ids = torch.arange(200, device=device) if class_ids is None else class_ids.to(device)
        images = split_features.to(device).float()
        logits = F.normalize(images, dim=-1) @ prototypes.index_select(0, ids).T * parent.scale()
        logits = sdrs(logits, images, ids)
        mask = torch.isin(ids.cpu(), seen_classes).to(device)
        logits = calibrator(logits, mask)
        logits = sdcr(logits, images, ids)
        positions = logits.argmax(dim=1).cpu()
        predictions = positions if class_ids is None else class_ids.index_select(0, positions)
        return predictions, logits.cpu()

    seen_predictions, seen_logits = infer(official["seen_features"])
    unseen_predictions, unseen_logits = infer(official["unseen_features"])
    zsl_predictions, zsl_logits = infer(official["unseen_features"], unseen_classes)
    seen_mask = torch.zeros(200, dtype=torch.bool)
    seen_mask[seen_classes] = True
    class_names = load_class_names(paths["att_splits"])
    seen_accuracy = h1.per_class_accuracy(
        official["seen_labels"], seen_predictions, seen_classes
    )
    unseen_accuracy = h1.per_class_accuracy(
        official["unseen_labels"], unseen_predictions, unseen_classes
    )
    zsl_accuracy = h1.per_class_accuracy(
        official["unseen_labels"], zsl_predictions, unseen_classes
    )
    payload = {
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "sdcr_model": str(sdcr_model_path),
        "sdcr_model_sha256": sha256_file(sdcr_model_path),
        "metrics_percent": {
            "U": unseen_accuracy * 100,
            "S": seen_accuracy * 100,
            "H": 2 * seen_accuracy * unseen_accuracy / (seen_accuracy + unseen_accuracy) * 100,
            "ZS": zsl_accuracy * 100,
        },
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
        "seen_split": summarize_predictions(
            official["seen_labels"], seen_predictions, seen_logits,
            class_names, seen_mask
        ),
        "unseen_split_gzsl": summarize_predictions(
            official["unseen_labels"], unseen_predictions, unseen_logits,
            class_names, seen_mask
        ),
        "unseen_split_zsl": summarize_predictions(
            official["unseen_labels"], zsl_predictions, zsl_logits,
            class_names, seen_mask
        ),
    }
    output_json = output_json.resolve()
    if output_json.exists():
        raise FileExistsError(f"诊断输出已存在：{output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_json, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sdcr-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.config, args.sdcr_model, args.output_json, args.device)
    print(json.dumps(result["metrics_percent"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
