from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml

from model.tg_vpr_h1 import TGVPRH1FixedEqual
from tools.cub_data import load_cub_split
from tools.reproducibility import configure_reproducibility
from tools.run_contract import (
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    repo_path,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


FRAMEWORK_ID = "FRAMEWORK-V2"
MODULE_SOURCE_ID = "INNOVATION-MODULE-1"
MODULE_ID = MODULE_SOURCE_ID
EVALUATION_PROTOCOL = "test_selected_inductive_gzsl"
TRAINING_KEYS = ("sentence_embeds", "train_features", "train_labels", "res101", "att_splits")
OFFICIAL_KEYS = ("seen_features", "seen_labels", "unseen_features", "unseen_labels")
CONFIG_KEYS = {
    "schema_version",
    "framework_id",
    "module_id",
    "dataset",
    "evaluation_protocol",
    "test_used_for_selection",
    "unseen_images_used_for_gradient",
    "device",
    "random_seed",
    "batch_size",
    "epochs",
    "weight_decay",
    "dropout",
    "inner_ratio",
    "outer_ratio",
    "topology_weight",
    "temperature",
    "group_weights",
    "value_heads",
    "role_order",
    "lr_stages",
    "inputs",
    "expected_sha256",
    "class_order_sha256",
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
    path = repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        actual = set(config) if isinstance(config, dict) else set()
        raise ValueError(
            f"TG-VPR-H1配置字段不匹配；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if config["schema_version"] != "gzsl-paper.tg-vpr-h1.v1":
        raise ValueError("TG-VPR-H1配置schema错误。")
    if config["framework_id"] != FRAMEWORK_ID:
        raise ValueError("TG-VPR-H1只接受FRAMEWORK-V2身份。")
    if config["module_id"] != MODULE_SOURCE_ID or config["dataset"] != "CUB":
        raise ValueError("TG-VPR-H1来源模块身份或CUB数据集不匹配。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("TG-VPR-H1评估协议不匹配。")
    if config["test_used_for_selection"] is not True:
        raise ValueError("必须披露official test参与选择。")
    if config["unseen_images_used_for_gradient"] is not False:
        raise ValueError("unseen图像不得进入训练梯度。")
    if int(config["batch_size"]) != 64 or int(config["epochs"]) != 50:
        raise ValueError("正式配置固定batch_size=64、epochs=50。")
    if int(config["value_heads"]) != 1:
        raise ValueError("正式模块固定单一768维Value路径。")
    if config["group_weights"] != [1.0 / 3.0] * 3:
        raise ValueError("正式模块固定三组各1/3。")
    if float(config["inner_ratio"]) != 0.35:
        raise ValueError("正式模块固定inner_ratio=0.35。")
    if float(config["outer_ratio"]) != 0.65:
        raise ValueError("正式模块固定outer_ratio=0.65。")
    if float(config["topology_weight"]) != 0.1:
        raise ValueError("正式模块固定topology_weight=0.1。")
    if tuple(config["role_order"]) != (
        "beak",
        "head_features",
        "body_plumage",
        "wings",
        "tail",
        "legs",
        "overall_appearance",
        "unique_discriminative_features",
    ):
        raise ValueError("8句顺序与冻结缓存契约不一致。")
    if [int(stage["epochs"]) for stage in config["lr_stages"]] != [20, 20, 10]:
        raise ValueError("正式模块固定20/20/10训练日程。")
    return config, sha256_file(path)


def resolve_paths(config: dict) -> dict[str, Path]:
    paths = {name: repo_path(value) for name, value in config["inputs"].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少TG-VPR-H1输入：" + ", ".join(missing))
    return paths


def verify_inputs(config: dict, paths: dict[str, Path], keys) -> dict[str, str]:
    actual = {name: sha256_file(paths[name]) for name in keys}
    mismatch = [name for name in keys if actual[name] != config["expected_sha256"][name]]
    if mismatch:
        raise ValueError("输入SHA-256不匹配：" + ", ".join(mismatch))
    names = sio.loadmat(paths["att_splits"], variable_names=["allclasses_names"])[
        "allclasses_names"
    ]
    serialized = json.dumps(
        [str(item[0][0]) for item in names], ensure_ascii=False, separators=(",", ":")
    )
    if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != config["class_order_sha256"]:
        raise ValueError("CUB类别顺序不匹配。")
    return actual


def legacy_batch_indices(count: int, batch_size: int, generator: torch.Generator):
    return torch.randperm(count, generator=generator)[:batch_size]


def visual_centroids(features, labels, classes):
    normalized = F.normalize(features.detach().float(), dim=-1)
    return torch.stack(
        [F.normalize(normalized[labels == class_id].mean(dim=0), dim=0) for class_id in classes]
    )


def per_class_accuracy(labels, predictions, classes) -> float:
    values = []
    labels = labels.cpu().long()
    predictions = predictions.cpu().long()
    for class_id in classes.cpu().long():
        mask = labels == class_id
        if not mask.any():
            raise ValueError(f"评估缓存缺少类别{int(class_id)}。")
        values.append((predictions[mask] == labels[mask]).float().mean())
    return float(torch.stack(values).mean())


@torch.no_grad()
def evaluate(model, tensors, seenclasses, unseenclasses, device):
    model.eval()
    prototypes = model.prototypes()
    seen_logits = F.normalize(tensors["seen_features"].to(device).float(), dim=-1) @ prototypes.T * model.scale()
    unseen_logits = F.normalize(tensors["unseen_features"].to(device).float(), dim=-1) @ prototypes.T * model.scale()
    if not torch.isfinite(seen_logits).all() or not torch.isfinite(unseen_logits).all():
        raise ValueError("评估logits包含NaN/Inf。")
    seen_pred = seen_logits.argmax(dim=1).cpu()
    unseen_pred = unseen_logits.argmax(dim=1).cpu()
    unseen_only = unseenclasses[unseen_logits[:, unseenclasses.to(device)].argmax(dim=1).cpu()]
    seen = per_class_accuracy(tensors["seen_labels"], seen_pred, seenclasses)
    unseen = per_class_accuracy(tensors["unseen_labels"], unseen_pred, unseenclasses)
    zsl = per_class_accuracy(tensors["unseen_labels"], unseen_only, unseenclasses)
    harmonic = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    return {"U": unseen * 100, "S": seen * 100, "H": harmonic * 100, "ZS": zsl * 100}


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    config, config_sha = load_config(config_path)
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    paths = resolve_paths(config)
    input_sha = verify_inputs(config, paths, TRAINING_KEYS)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("正式TG-VPR-H1训练要求可见CUDA。")
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_handle)

    seed = int(config["random_seed"])
    reproducibility = configure_reproducibility(
        seed, strict_determinism=True, deterministic_warn_only=False
    )
    print(f"框架：{FRAMEWORK_ID}")
    print(f"来源模块：{MODULE_SOURCE_ID}")
    print(f"代码commit：{code_commit}")
    print(f"配置SHA-256：{config_sha}")
    print(f"RUN：{run_id} seed={seed}")

    tensors = {
        name: torch.load(paths[name], map_location="cpu", weights_only=True)
        for name in ("sentence_embeds", "train_features", "train_labels")
    }
    labels = tensors["train_labels"].long()
    seenclasses = torch.unique(labels, sorted=True)
    allclasses = torch.arange(200)
    unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
    if labels.numel() != 7057 or seenclasses.numel() != 150 or unseenclasses.numel() != 50:
        raise ValueError("CUB训练必须是7057样本和150/50类划分。")
    centroids = visual_centroids(tensors["train_features"], labels, seenclasses)
    model = TGVPRH1FixedEqual(
        tensors["sentence_embeds"],
        seenclasses,
        centroids,
        dropout=config["dropout"],
        inner_ratio=config["inner_ratio"],
        outer_ratio=config["outer_ratio"],
        temperature=config["temperature"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr_stages"][0]["lr"],
        weight_decay=config["weight_decay"],
    )
    stages = config["lr_stages"]
    boundaries = []
    total = 0
    for stage in stages:
        total += int(stage["epochs"])
        boundaries.append(total)
    active_stage = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stages[0]["epochs"], eta_min=stages[0]["eta_min"]
    )
    global_to_seen = torch.full((200,), -1, dtype=torch.long)
    global_to_seen[seenclasses] = torch.arange(150)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    best_state = None
    for epoch in range(1, 51):
        target_stage = next(i for i, boundary in enumerate(boundaries) if epoch <= boundary)
        if target_stage != active_stage:
            active_stage = target_stage
            stage = stages[active_stage]
            for group in optimizer.param_groups:
                group["lr"] = float(stage["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage["epochs"], eta_min=stage["eta_min"]
            )
        model.train()
        loss_sum = ce_sum = topology_sum = 0.0
        sample_count = 0
        for _ in range(labels.numel() // int(config["batch_size"])):
            indices = legacy_batch_indices(labels.numel(), config["batch_size"], generator)
            features = tensors["train_features"][indices].to(device).float()
            targets = global_to_seen[labels[indices]].to(device)
            optimizer.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model.logits(features, seenclasses), targets)
            topology = model.topology_loss()
            loss = ce + float(config["topology_weight"]) * topology
            if not torch.isfinite(loss):
                raise FloatingPointError("训练loss包含NaN/Inf。")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("模型梯度包含NaN/Inf。")
            optimizer.step()
            loss_sum += float(loss.detach()) * features.size(0)
            ce_sum += float(ce.detach()) * features.size(0)
            topology_sum += float(topology.detach()) * features.size(0)
            sample_count += features.size(0)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "train_ce": ce_sum / sample_count,
            "train_topology": topology_sum / sample_count,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"topology={row['train_topology']:.6f}"
        )
        if epoch == 50:
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    checkpoint = {
        "framework_id": FRAMEWORK_ID,
        "module_source_id": MODULE_SOURCE_ID,
        "module_id": MODULE_SOURCE_ID,
        "run_id": run_id,
        "code_commit": code_commit,
        "config": config,
        "config_sha256": config_sha,
        "seed": seed,
        "best_epoch": 50,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "history": history,
        "reproducibility": reproducibility,
    }
    torch.save(checkpoint, output_dir / "model_best.pth")
    torch.save(checkpoint, output_dir / "checkpoint_last.pth")

    input_sha.update(verify_inputs(config, paths, OFFICIAL_KEYS))
    tensors.update(
        {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in OFFICIAL_KEYS
        }
    )
    checked_seen, checked_unseen = load_cub_split(
        paths["res101"],
        paths["att_splits"],
        labels,
        tensors["seen_labels"],
        tensors["unseen_labels"],
        "cpu",
    )
    if not torch.equal(checked_seen, seenclasses) or not torch.equal(
        checked_unseen, unseenclasses
    ):
        raise RuntimeError("official split与训练类划分不一致。")
    metrics = evaluate(model, tensors, seenclasses, unseenclasses, device)
    atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
    atomic_write_json(
        output_dir / "metrics.json",
        {
            "framework_id": FRAMEWORK_ID,
            "module_source_id": MODULE_SOURCE_ID,
            "module_id": MODULE_SOURCE_ID,
            "run_id": run_id,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False,
            "code_commit": code_commit,
            "config_sha256": config_sha,
            "seed": seed,
            "best_epoch": 50,
            "inner_ratio": float(config["inner_ratio"]),
            "outer_ratio": float(config["outer_ratio"]),
            "topology_weight": float(config["topology_weight"]),
            "group_weights": model.semantic_group_weights().detach().cpu().tolist(),
            "value_heads": 1,
            "metrics_percent": metrics,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
        },
    )
    print("U={U:.6f}% S={S:.6f}% H={H:.6f}% ZS={ZS:.6f}%".format(**metrics))
    sys.stdout.flush()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/tg_vpr_h1.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.run_id)


if __name__ == "__main__":
    main()
