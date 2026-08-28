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

from model.candidates.v2.trainers.train_unified_seen import full_epoch_batches
from model.candidates.v2.modules.unified_expert import ExpertAttributeUnifiedModel
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
    require_finite_model,
)
from tools.runtime import sha256_file


EVALUATION_PROTOCOL = "xlsa17_ps_gzsl_after_validation_freeze"
TRAINING_KEYS = ("sentence_embeds", "train_features", "train_labels", "res101", "att_splits")
OFFICIAL_KEYS = ("seen_features", "seen_labels", "unseen_features", "unseen_labels")
INPUT_KEYS = TRAINING_KEYS + OFFICIAL_KEYS
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "condition_id",
    "framework_id",
    "dataset",
    "evaluation_protocol",
    "validation_selection",
    "test_used_for_selection",
    "official_test_evaluations",
    "expert_attributes_used",
    "feature_backbone",
    "feature_provenance_complete",
    "historical_test_informed_architecture",
    "strict_blind_claim_eligible",
    "owner_authorized_final_test",
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
    "transport_hidden_dim",
    "generator_hidden_dim",
    "max_transport_step",
    "max_generator_magnitude",
    "max_attribute_residual",
    "lr_stages",
    "inputs",
    "expected_sha256",
    "class_order_sha256",
}


def load_config(path: Path) -> tuple[dict, str]:
    path = h1.repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"标准最终配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"标准最终配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if config["schema_version"] not in (
        "gzsl-paper.standard-final.v1",
        "gzsl-paper.threefold-final.v1",
    ):
        raise ValueError("标准最终配置schema错误。")
    threefold_final = config["schema_version"] == "gzsl-paper.threefold-final.v1"
    expected_experiment = "V2-CONFIRM-008" if threefold_final else "V2-CONFIRM-003"
    if config["experiment_id"] != expected_experiment:
        raise ValueError("标准最终实验身份错误。")
    allowed_conditions = (
        ("THREEFOLD-NO-EXPERT",)
        if threefold_final
        else ("NO-EXPERT", "EXPERT")
    )
    if config["condition_id"] not in allowed_conditions:
        raise ValueError("最终condition只允许NO-EXPERT或EXPERT。")
    if config["framework_id"] != "FRAMEWORK-V2" or config["dataset"] != "CUB":
        raise ValueError("标准最终评估只接受FRAMEWORK-V2/CUB。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("标准最终评估协议身份错误。")
    if config["test_used_for_selection"] is not False:
        raise ValueError("official test不得用于最终配置选择。")
    if int(config["official_test_evaluations"]) != 1:
        raise ValueError("每条最终RUN固定只评估一次official test。")
    expected_expert = config["condition_id"] == "EXPERT"
    if config["expert_attributes_used"] is not expected_expert:
        raise ValueError("专家属性开关与最终condition不一致。")
    if config["feature_provenance_complete"] is not False:
        raise ValueError("必须披露遗留CLIP缓存来源不完整。")
    if config["historical_test_informed_architecture"] is not True:
        raise ValueError("必须披露方法结构受历史test探索影响。")
    if config["strict_blind_claim_eligible"] is not False:
        raise ValueError("当前结果不得标记为严格blind-test。")
    expected_authorization = "2026-08-25" if threefold_final else "2026-08-23"
    if config["owner_authorized_final_test"] != expected_authorization:
        raise ValueError("缺少owner当前最终test授权。")
    expected = (
        {
            "epochs": 17,
            "topology_weight": 0.1,
            "validation_experiment": "V2-TUNE-003",
            "validation_run": "RUN-001",
        }
        if threefold_final
        else {
            "NO-EXPERT": {
                "epochs": 24,
                "topology_weight": 0.1,
                "validation_experiment": "V2-TUNE-001",
                "validation_run": "RUN-001",
            },
            "EXPERT": {
                "epochs": 22,
                "topology_weight": 0.2,
                "validation_experiment": "V2-TUNE-001",
                "validation_run": "RUN-006",
            },
        }[config["condition_id"]]
    )
    if int(config["epochs"]) != expected["epochs"]:
        raise ValueError("最终epoch必须等于validation选择。")
    if float(config["topology_weight"]) != expected["topology_weight"]:
        raise ValueError("最终topology权重必须等于validation选择。")
    selection = config["validation_selection"]
    if selection.get("experiment_id") != expected[
        "validation_experiment"
    ] or selection.get("run_id") != expected["validation_run"]:
        raise ValueError("最终配置绑定的validation RUN错误。")
    if int(selection.get("selected_epoch", -1)) != expected["epochs"]:
        raise ValueError("validation selected_epoch绑定错误。")
    lr_horizon = sum(int(stage["epochs"]) for stage in config["lr_stages"])
    if (
        (not threefold_final and lr_horizon != int(config["epochs"]))
        or (threefold_final and lr_horizon != 50)
    ):
        raise ValueError("最终学习率阶段必须复现validation训练时的调度horizon。")
    if threefold_final and (
        float(config["max_transport_step"]) != 1.5
        or float(config["max_generator_magnitude"]) != 0.2
    ):
        raise ValueError("三折冻结最终配置的transport/CCGR参数错误。")
    if set(config["inputs"]) != set(INPUT_KEYS) or set(config["expected_sha256"]) != set(INPUT_KEYS):
        raise ValueError("最终输入或SHA字段不完整。")
    return config, sha256_file(path)


def resolve_paths(config: dict) -> dict[str, Path]:
    paths = {name: h1.repo_path(config["inputs"][name]) for name in INPUT_KEYS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少最终评估输入：" + ", ".join(missing))
    return paths


def verify_inputs(config: dict, paths: dict[str, Path], keys) -> dict[str, str]:
    actual = {name: sha256_file(paths[name]) for name in keys}
    mismatch = [name for name in keys if actual[name] != config["expected_sha256"][name]]
    if mismatch:
        raise ValueError("最终输入SHA不匹配：" + ", ".join(mismatch))
    names = sio.loadmat(paths["att_splits"], variable_names=["allclasses_names"])["allclasses_names"]
    serialized = json.dumps(
        [str(item[0][0]) for item in names], ensure_ascii=False, separators=(",", ":")
    )
    if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != config["class_order_sha256"]:
        raise ValueError("CUB类别顺序不匹配。")
    return actual


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config)
    input_sha = verify_inputs(config, paths, TRAINING_KEYS)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("最终训练要求可见CUDA。")
    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = h1.TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(
            seed, strict_determinism=True, deterministic_warn_only=False
        )
        sentence_embeds = torch.load(paths["sentence_embeds"], map_location="cpu", weights_only=True)
        features = torch.load(paths["train_features"], map_location="cpu", weights_only=True)
        labels = torch.load(paths["train_labels"], map_location="cpu", weights_only=True).long()
        seenclasses = torch.unique(labels, sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        if labels.numel() != 7057 or seenclasses.numel() != 150 or unseenclasses.numel() != 50:
            raise ValueError("最终训练必须使用trainval 7057图像和150/50类别。")
        centroids = h1.visual_centroids(features, labels, seenclasses)
        text_model = UnifiedSeenPrototypeModel(
            sentence_embeds,
            seenclasses,
            centroids,
            active_classes=allclasses,
            dropout=float(config["dropout"]),
            inner_ratio=float(config["inner_ratio"]),
            outer_ratio=float(config["outer_ratio"]),
            temperature=float(config["temperature"]),
            transport_hidden_dim=int(config["transport_hidden_dim"]),
            generator_hidden_dim=int(config["generator_hidden_dim"]),
            max_transport_step=float(config["max_transport_step"]),
            max_generator_magnitude=float(config["max_generator_magnitude"]),
        )
        if config["condition_id"] == "EXPERT":
            attribute_mat = sio.loadmat(paths["att_splits"], variable_names=["att"])["att"]
            model = ExpertAttributeUnifiedModel(
                text_model,
                torch.from_numpy(attribute_mat.T).float(),
                max_attribute_residual=float(config["max_attribute_residual"]),
            )
        else:
            model = text_model
        model = model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["lr_stages"][0]["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        stages = config["lr_stages"]
        boundaries = []
        total = 0
        for stage in stages:
            total += int(stage["epochs"])
            boundaries.append(total)
        active_stage = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(stages[0]["epochs"]),
            eta_min=float(stages[0]["eta_min"]),
        )
        global_to_seen = torch.full((200,), -1, dtype=torch.long)
        global_to_seen[seenclasses] = torch.arange(150)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        history = []
        print(f"实验：{config['experiment_id']} 条件：{config['condition_id']}")
        print(f"validation选择：{config['validation_selection']}")
        print(f"代码commit：{code_commit} 配置SHA：{config_sha}")
        print("official test将在最终checkpoint写入后加载一次")

        for epoch in range(1, int(config["epochs"]) + 1):
            target_stage = next(index for index, boundary in enumerate(boundaries) if epoch <= boundary)
            if target_stage != active_stage:
                active_stage = target_stage
                stage = stages[active_stage]
                for group in optimizer.param_groups:
                    group["lr"] = float(stage["lr"])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(stage["epochs"]),
                    eta_min=float(stage["eta_min"]),
                )
            model.train()
            loss_sum = 0.0
            sample_count = 0
            for indices in full_epoch_batches(labels.numel(), int(config["batch_size"]), generator):
                images = features.index_select(0, indices).to(device).float()
                targets = global_to_seen[labels.index_select(0, indices)].to(device)
                optimizer.zero_grad(set_to_none=True)
                ce = F.cross_entropy(model.logits(images, seenclasses), targets)
                topology = model.topology_loss()
                loss = ce + float(config["topology_weight"]) * topology
                if not torch.isfinite(loss):
                    raise FloatingPointError("最终训练loss包含NaN/Inf。")
                loss.backward()
                require_finite_gradients(model)
                optimizer.step()
                loss_sum += float(loss.detach()) * images.size(0)
                sample_count += images.size(0)
            if sample_count != 7057:
                raise RuntimeError("最终每个epoch必须完整且唯一遍历7057张图像。")
            scheduler.step()
            diagnostics = model.diagnostics()
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / sample_count,
                    "sample_count": sample_count,
                    "unique_sample_count": sample_count,
                    "diagnostics": diagnostics,
                }
            )
            print(
                f"epoch={epoch} samples={sample_count} loss={loss_sum/sample_count:.6f} "
                f"step={diagnostics['transport_step_mean']:.6f}"
            )

        model.eval()
        require_finite_model(model)
        final_state = copy.deepcopy(model.state_dict())
        checkpoint = {
            "experiment_id": config["experiment_id"],
            "condition_id": config["condition_id"],
            "run_id": run_id,
            "code_commit": code_commit,
            "config": config,
            "config_sha256": config_sha,
            "seed": seed,
            "reported_epoch": int(config["epochs"]),
            "model_state_dict": final_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "reproducibility": reproducibility,
        }
        atomic_torch_save(output_dir / "model_best.pth", checkpoint)
        atomic_torch_save(output_dir / "checkpoint_last.pth", checkpoint)

        # official test只在validation冻结的最终训练与checkpoint写入完成后加载一次。
        input_sha.update(verify_inputs(config, paths, OFFICIAL_KEYS))
        official = {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in OFFICIAL_KEYS
        }
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"],
            paths["att_splits"],
            labels,
            official["seen_labels"],
            official["unseen_labels"],
            "cpu",
        )
        if not torch.equal(checked_seen, seenclasses) or not torch.equal(checked_unseen, unseenclasses):
            raise RuntimeError("official split与最终训练划分不一致。")
        metrics = h1.evaluate(model, official, seenclasses, unseenclasses, device)
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        atomic_write_json(
            output_dir / "metrics.json",
            {
                "experiment_id": config["experiment_id"],
                "condition_id": config["condition_id"],
                "run_id": run_id,
                "framework_id": config["framework_id"],
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "validation_selection": config["validation_selection"],
                "test_used_for_selection": False,
                "official_test_evaluations": 1,
                "official_test_loaded_after_training": True,
                "expert_attributes_used": bool(config["expert_attributes_used"]),
                "feature_backbone": config["feature_backbone"],
                "feature_provenance_complete": False,
                "historical_test_informed_architecture": True,
                "strict_blind_claim_eligible": False,
                "code_commit": code_commit,
                "config_sha256": config_sha,
                "seed": seed,
                "reported_epoch": int(config["epochs"]),
                "train_samples_per_epoch": 7057,
                "metrics_percent": metrics,
                "diagnostics": model.diagnostics(),
                "model_sha256": sha256_file(output_dir / "model_best.pth"),
                "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
            },
        )
        print("U={U:.6f}% S={S:.6f}% H={H:.6f}% ZS={ZS:.6f}%".format(**metrics))
        return metrics
    finally:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.expected_commit, args.run_id)


if __name__ == "__main__":
    main()
