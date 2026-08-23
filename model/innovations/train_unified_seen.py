from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

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
    require_finite_model,
)
from tools.runtime import sha256_file


EVALUATION_PROTOCOL = "fixed_epoch_inductive_gzsl"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "framework_id",
    "dataset",
    "evaluation_protocol",
    "test_used_for_selection",
    "unseen_images_used_for_gradient",
    "unseen_text_used_during_training",
    "official_test_load_epoch",
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
    "lr_stages",
    "inputs",
    "expected_sha256",
    "class_order_sha256",
}


def load_config(path: Path) -> tuple[dict, str]:
    path = h1.repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"统一seen训练配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or actual != CONFIG_KEYS:
        raise ValueError(
            f"统一seen训练配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，"
            f"多出={sorted(actual-CONFIG_KEYS)}。"
        )
    if config["schema_version"] != "gzsl-paper.unified-seen.v1":
        raise ValueError("统一seen训练配置schema错误。")
    if config["experiment_id"] != "V2-CONFIRM-002":
        raise ValueError("统一seen训练实验身份错误。")
    if config["framework_id"] != "FRAMEWORK-V2" or config["dataset"] != "CUB":
        raise ValueError("统一seen训练只接受FRAMEWORK-V2/CUB。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("统一seen训练必须使用固定第50轮评估协议。")
    if config["test_used_for_selection"] is not False:
        raise ValueError("official test不得用于统一训练选模。")
    if config["unseen_images_used_for_gradient"] is not False:
        raise ValueError("真实unseen图像不得进入梯度。")
    if config["unseen_text_used_during_training"] is not True:
        raise ValueError("必须披露unseen文本参与共享原型与topology计算。")
    if config["official_test_load_epoch"] != "after_epoch_50":
        raise ValueError("official test只能在第50轮训练结束后加载。")
    if int(config["epochs"]) != 50 or int(config["batch_size"]) != 64:
        raise ValueError("统一seen训练固定50轮、batch size 64。")
    if float(config["topology_weight"]) != 0.1:
        raise ValueError("统一seen训练固定topology权重0.1。")
    if [int(stage["epochs"]) for stage in config["lr_stages"]] != [20, 20, 10]:
        raise ValueError("统一seen训练固定20/20/10学习率阶段。")
    return config, sha256_file(path)


def full_epoch_batches(
    sample_count: int,
    batch_size: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("sample_count和batch_size必须为正数。")
    permutation = torch.randperm(sample_count, generator=generator)
    batches = [
        permutation[start : start + batch_size]
        for start in range(0, sample_count, batch_size)
    ]
    joined = torch.cat(batches)
    if joined.numel() != sample_count or joined.unique().numel() != sample_count:
        raise RuntimeError("一个epoch必须让每个seen样本恰好出现一次。")
    return batches


def _gradient_group_norms(model: UnifiedSeenPrototypeModel) -> dict[str, float]:
    groups = {
        "tg_vpr": model.tg_vpr.parameters(),
        "transport": list(model.transport_trunk.parameters())
        + list(model.transport_head.parameters()),
        "generator": list(model.generator_trunk.parameters())
        + list(model.generator_weight_head.parameters())
        + list(model.generator_magnitude_head.parameters()),
    }
    result = {}
    for name, parameters in groups.items():
        values = [
            parameter.grad.detach().norm()
            for parameter in parameters
            if parameter.grad is not None
        ]
        result[name] = float(torch.stack(values).norm()) if values else 0.0
    return result


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    config, config_sha = load_config(config_path)
    paths = h1.resolve_paths(config)
    input_sha = h1.verify_inputs(config, paths, h1.TRAINING_KEYS)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("统一seen训练要求可见CUDA。")

    output_dir = prepare_output_dir(output_dir)
    with (output_dir / "config.snapshot.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    log_handle = (output_dir / "training.log").open(
        "x", encoding="utf-8", buffering=1
    )
    original_stdout = sys.stdout
    sys.stdout = h1.TeeStream(sys.stdout, log_handle)
    try:
        seed = int(config["random_seed"])
        reproducibility = configure_reproducibility(
            seed, strict_determinism=True, deterministic_warn_only=False
        )
        tensors = {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in ("sentence_embeds", "train_features", "train_labels")
        }
        labels = tensors["train_labels"].long()
        seenclasses = torch.unique(labels, sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        if labels.numel() != 7057 or seenclasses.numel() != 150:
            raise ValueError("统一训练必须使用7057张图像和150个seen类。")
        if unseenclasses.numel() != 50:
            raise ValueError("统一训练的文本类别空间必须包含50个unseen类。")

        centroids = h1.visual_centroids(
            tensors["train_features"], labels, seenclasses
        )
        model = UnifiedSeenPrototypeModel(
            tensors["sentence_embeds"],
            seenclasses,
            centroids,
            dropout=float(config["dropout"]),
            inner_ratio=float(config["inner_ratio"]),
            outer_ratio=float(config["outer_ratio"]),
            temperature=float(config["temperature"]),
            transport_hidden_dim=int(config["transport_hidden_dim"]),
            generator_hidden_dim=int(config["generator_hidden_dim"]),
            max_transport_step=float(config["max_transport_step"]),
            max_generator_magnitude=float(config["max_generator_magnitude"]),
        ).to(device)
        initial_model_state = copy.deepcopy(model.state_dict())
        model.eval()
        with torch.no_grad():
            initial = model.prototype_stages()
            if not torch.allclose(
                initial["tg_vpr"], initial["transported"], atol=1e-6, rtol=1e-6
            ):
                raise RuntimeError("统一迁移必须从TG-VPR数值等价初始化。")
            if not torch.allclose(
                initial["tg_vpr"], initial["final"], atol=1e-6, rtol=1e-6
            ):
                raise RuntimeError("统一生成器必须从TG-VPR数值等价初始化。")

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
        first_batch_gradient_norms = None

        print(f"实验：{config['experiment_id']}")
        print(f"代码commit：{code_commit}")
        print(f"配置SHA-256：{config_sha}")
        print("训练协议：50轮全seen遍历；official test仅训练结束后加载一次")

        for epoch in range(1, int(config["epochs"]) + 1):
            target_stage = next(
                index for index, boundary in enumerate(boundaries) if epoch <= boundary
            )
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
            epoch_batches = full_epoch_batches(
                labels.numel(), int(config["batch_size"]), generator
            )
            loss_sum = ce_sum = topology_sum = 0.0
            sample_count = 0
            for batch_index, indices in enumerate(epoch_batches):
                features = tensors["train_features"][indices].to(device).float()
                targets = global_to_seen[labels[indices]].to(device)
                optimizer.zero_grad(set_to_none=True)
                ce = F.cross_entropy(model.logits(features, seenclasses), targets)
                topology = model.topology_loss()
                loss = ce + float(config["topology_weight"]) * topology
                if not torch.isfinite(loss):
                    raise FloatingPointError("统一训练loss包含NaN/Inf。")
                loss.backward()
                require_finite_gradients(model)
                if epoch == 1 and batch_index == 0:
                    first_batch_gradient_norms = _gradient_group_norms(model)
                    if any(value <= 0.0 for value in first_batch_gradient_norms.values()):
                        raise RuntimeError(
                            "TG-VPR、迁移和生成模块都必须在首批收到非零梯度。"
                        )
                optimizer.step()
                loss_sum += float(loss.detach()) * features.size(0)
                ce_sum += float(ce.detach()) * features.size(0)
                topology_sum += float(topology.detach()) * features.size(0)
                sample_count += features.size(0)
            if sample_count != 7057:
                raise RuntimeError("每个epoch必须完整消费7057张seen图像。")
            scheduler.step()
            diagnostics = model.diagnostics()
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                "train_ce": ce_sum / sample_count,
                "train_topology": topology_sum / sample_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "sample_count": sample_count,
                "unique_sample_count": sample_count,
                "diagnostics": diagnostics,
            }
            history.append(row)
            print(
                f"epoch={epoch} samples={sample_count} "
                f"loss={row['train_loss']:.6f} "
                f"step={diagnostics['transport_step_mean']:.6f} "
                f"magnitude={diagnostics['generator_magnitude_mean']:.6f}"
            )

        model.eval()
        require_finite_model(model)
        final_state = copy.deepcopy(model.state_dict())
        checkpoint = {
            "experiment_id": config["experiment_id"],
            "run_id": run_id,
            "code_commit": code_commit,
            "config": config,
            "config_sha256": config_sha,
            "seed": seed,
            "reported_epoch": 50,
            "model_state_dict": final_state,
            "initial_model_state_dict": initial_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "reproducibility": reproducibility,
        }
        atomic_torch_save(output_dir / "model_best.pth", checkpoint)
        atomic_torch_save(output_dir / "checkpoint_last.pth", checkpoint)

        # official test只在全部50轮训练与checkpoint写入完成后加载一次。
        input_sha.update(h1.verify_inputs(config, paths, h1.OFFICIAL_KEYS))
        tensors.update(
            {
                name: torch.load(paths[name], map_location="cpu", weights_only=True)
                for name in h1.OFFICIAL_KEYS
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
        metrics = h1.evaluate(model, tensors, seenclasses, unseenclasses, device)
        diagnostics = model.diagnostics()
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        atomic_write_json(
            output_dir / "metrics.json",
            {
                "experiment_id": config["experiment_id"],
                "run_id": run_id,
                "framework_id": config["framework_id"],
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "test_used_for_selection": False,
                "official_test_evaluations": 1,
                "official_test_loaded_after_epoch": 50,
                "unseen_images_used_for_gradient": False,
                "unseen_text_used_during_training": True,
                "code_commit": code_commit,
                "config_sha256": config_sha,
                "seed": seed,
                "reported_epoch": 50,
                "train_samples_per_epoch": 7057,
                "unique_train_samples_per_epoch": 7057,
                "first_batch_gradient_norms": first_batch_gradient_norms,
                "diagnostics": diagnostics,
                "metrics_percent": metrics,
                "model_sha256": sha256_file(output_dir / "model_best.pth"),
                "checkpoint_last_sha256": sha256_file(
                    output_dir / "checkpoint_last.pth"
                ),
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
