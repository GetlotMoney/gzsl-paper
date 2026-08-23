from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path, PurePosixPath

import torch
import torch.nn.functional as F
import yaml

from model.innovations.elpt import VariableClassTGVPR, fixed_class_folds
from model.innovations.train_chen_stagewise import gradient_group_norms, set_trainable_stage
from model.innovations.train_chen_style import (
    INPUT_KEYS,
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
    require_finite_model,
)
from tools.runtime import sha256_file


EVALUATION_PROTOCOL = "chen_shiming_code_aligned_class_exclusive_gzsl"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "condition_id",
    "framework_id",
    "dataset",
    "evaluation_protocol",
    "test_used_for_selection",
    "unseen_images_used_for_gradient",
    "strict_blind_claim",
    "training_strategy",
    "selection_scope",
    "nested_official_test_selection",
    "feature_backbone",
    "feature_provenance_complete",
    "device",
    "random_seed",
    "batch_size",
    "batch_half",
    "main_tg_steps",
    "fold_tg_epochs",
    "transfer_steps",
    "joint_steps",
    "report_interval",
    "optimizer",
    "main_tg_lr",
    "fold_tg_lr",
    "transfer_lr",
    "joint_lr",
    "weight_decay",
    "dropout",
    "inner_ratio",
    "outer_ratio",
    "topology_weight",
    "pseudo_unseen_weight",
    "temperature",
    "transport_hidden_dim",
    "generator_hidden_dim",
    "max_transport_step",
    "max_generator_magnitude",
    "fold_count",
    "inputs",
    "expected_sha256",
    "class_order_sha256",
}
REUSE_CONFIG_KEYS = CONFIG_KEYS | {
    "reuse_parent_dir",
    "full_tg_parent_sha256",
    "fold_model_sha256",
}


def load_config(path: Path) -> tuple[dict, str]:
    path = h1.repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"class-exclusive配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = set(config) if isinstance(config, dict) else set()
    expected_keys = (
        REUSE_CONFIG_KEYS
        if isinstance(config, dict)
        and config.get("schema_version") == "gzsl-paper.chen-class-exclusive-reuse.v1"
        else CONFIG_KEYS
    )
    if not isinstance(config, dict) or actual != expected_keys:
        raise ValueError(
            f"class-exclusive配置字段错误；缺少={sorted(expected_keys-actual)}，"
            f"多出={sorted(actual-expected_keys)}。"
        )
    if config["schema_version"] not in (
        "gzsl-paper.chen-class-exclusive.v1",
        "gzsl-paper.chen-class-exclusive-reuse.v1",
    ):
        raise ValueError("class-exclusive配置schema错误。")
    if config["experiment_id"] != "V2-CONFIRM-007" or config["condition_id"] != "NO-EXPERT":
        raise ValueError("class-exclusive实验身份错误。")
    if config["framework_id"] != "FRAMEWORK-V2" or config["dataset"] != "CUB":
        raise ValueError("class-exclusive只接受FRAMEWORK-V2/CUB。")
    if config["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("class-exclusive协议身份错误。")
    required = {
        "test_used_for_selection": True,
        "unseen_images_used_for_gradient": False,
        "strict_blind_claim": False,
        "nested_official_test_selection": False,
    }
    for key, expected in required.items():
        if config[key] is not expected:
            raise ValueError(f"class-exclusive边界错误：{key}必须为{expected}。")
    if config["training_strategy"] != "full_tg_plus_three_class_exclusive_fold_parents":
        raise ValueError("class-exclusive训练策略身份错误。")
    if config["selection_scope"] != "inference_model_whole_run_only":
        raise ValueError("class-exclusive只允许最终推理模型全RUN一个best。")
    expected_numbers = {
        "batch_size": 50,
        "batch_half": 25,
        "main_tg_steps": 7050,
        "fold_tg_epochs": 50,
        "transfer_steps": 14100,
        "joint_steps": 7078,
        "report_interval": 141,
        "fold_count": 3,
    }
    for key, expected in expected_numbers.items():
        if int(config[key]) != expected:
            raise ValueError(f"class-exclusive固定{key}={expected}。")
    if config["optimizer"] != "Adam":
        raise ValueError("class-exclusive固定Adam。")
    if any(float(config[key]) != expected for key, expected in {
        "main_tg_lr": 1e-4,
        "fold_tg_lr": 1e-4,
        "transfer_lr": 1e-4,
        "joint_lr": 1e-5,
        "pseudo_unseen_weight": 0.25,
    }.items()):
        raise ValueError("class-exclusive学习率、pseudo权重或迁移上限错误。")
    if float(config["max_transport_step"]) not in (0.5, 1.5):
        raise ValueError("class-exclusive迁移上限只允许0.5或1.5。")
    config.setdefault("reuse_parent_dir", None)
    config.setdefault("full_tg_parent_sha256", None)
    config.setdefault("fold_model_sha256", None)
    if config["schema_version"] == "gzsl-paper.chen-class-exclusive-reuse.v1":
        if not (
            Path(config["reuse_parent_dir"]).is_absolute()
            or PurePosixPath(config["reuse_parent_dir"]).is_absolute()
        ):
            raise ValueError("复用父模型目录必须是绝对路径。")
        if set(config["fold_model_sha256"]) != {"0", "1", "2"}:
            raise ValueError("复用fold SHA必须包含0/1/2。")
    if set(config["inputs"]) != set(INPUT_KEYS) or set(config["expected_sha256"]) != set(INPUT_KEYS):
        raise ValueError("class-exclusive输入或SHA字段不完整。")
    return config, sha256_file(path)


def topology_loss_between(parent: torch.Tensor, adapted: torch.Tensor, class_ids: torch.Tensor):
    parent = F.normalize(parent.index_select(0, class_ids), dim=-1)
    adapted = F.normalize(adapted.index_select(0, class_ids), dim=-1)
    count = class_ids.numel()
    off_diag = ~torch.eye(count, dtype=torch.bool, device=parent.device)
    x = (parent @ parent.T).detach()[off_diag]
    y = (adapted @ adapted.T)[off_diag]
    x = x - x.mean()
    y = y - y.mean()
    correlation = (x * y).sum() / (
        torch.sqrt(x.square().sum() + 1e-8)
        * torch.sqrt(y.square().sum() + 1e-8)
    )
    return 1.0 - correlation


def balanced_fold_batch(
    labels: torch.Tensor,
    pseudo_seen: torch.Tensor,
    pseudo_unseen: torch.Tensor,
    half: int,
    generator: torch.Generator,
) -> torch.Tensor:
    seen_pool = torch.isin(labels, pseudo_seen).nonzero(as_tuple=False).flatten()
    unseen_pool = torch.isin(labels, pseudo_unseen).nonzero(as_tuple=False).flatten()
    seen_indices = seen_pool[torch.randperm(seen_pool.numel(), generator=generator)[:half]]
    unseen_indices = unseen_pool[torch.randperm(unseen_pool.numel(), generator=generator)[:half]]
    return torch.cat((seen_indices, unseen_indices))


def run(config_path: Path, output_dir: Path, expected_commit: str, run_id: str):
    require_clean_code_tree()
    code_commit = current_code_commit()
    if code_commit != expected_commit:
        raise ValueError("expected-commit与当前干净HEAD不一致。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须等于run-id。")
    config, config_sha = load_config(config_path)
    paths = resolve_paths(config)
    input_sha = verify_inputs(config, paths)
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("class-exclusive训练要求可见CUDA。")
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
        train_features = torch.load(paths["train_features"], map_location="cpu", weights_only=True)
        train_labels = torch.load(paths["train_labels"], map_location="cpu", weights_only=True).long()
        official = {
            name: torch.load(paths[name], map_location="cpu", weights_only=True)
            for name in OFFICIAL_KEYS
        }
        seenclasses = torch.unique(train_labels, sorted=True)
        allclasses = torch.arange(200)
        unseenclasses = allclasses[~torch.isin(allclasses, seenclasses)]
        checked_seen, checked_unseen = load_cub_split(
            paths["res101"], paths["att_splits"], train_labels,
            official["seen_labels"], official["unseen_labels"], "cpu"
        )
        if not torch.equal(checked_seen, seenclasses) or not torch.equal(checked_unseen, unseenclasses):
            raise RuntimeError("class-exclusive official split不一致。")
        global_to_seen = torch.full((200,), -1, dtype=torch.long)
        global_to_seen[seenclasses] = torch.arange(150)
        centroids = h1.visual_centroids(train_features, train_labels, seenclasses)
        model = UnifiedSeenPrototypeModel(
            sentence_embeds, seenclasses, centroids, active_classes=allclasses,
            dropout=float(config["dropout"]), inner_ratio=float(config["inner_ratio"]),
            outer_ratio=float(config["outer_ratio"]), temperature=float(config["temperature"]),
            transport_hidden_dim=int(config["transport_hidden_dim"]),
            generator_hidden_dim=int(config["generator_hidden_dim"]),
            max_transport_step=float(config["max_transport_step"]),
            max_generator_magnitude=float(config["max_generator_magnitude"]),
        ).to(device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        reuse_dir = Path(config["reuse_parent_dir"]) if config["reuse_parent_dir"] else None
        if reuse_dir is None:
            set_trainable_stage(model, "TG_ONLY")
            optimizer = torch.optim.Adam(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=float(config["main_tg_lr"]), weight_decay=float(config["weight_decay"]),
            )
            print(f"full_tg_pretrain steps={config['main_tg_steps']}")
            for iteration in range(int(config["main_tg_steps"])):
                indices = random_batch_indices(train_labels.numel(), int(config["batch_size"]), generator)
                images = train_features.index_select(0, indices).to(device).float()
                targets = global_to_seen[train_labels.index_select(0, indices)].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.logits(images, seenclasses), targets) + float(config["topology_weight"]) * model.topology_loss()
                loss.backward(); require_finite_gradients(model); optimizer.step()
            full_tg_state = copy.deepcopy(model.tg_vpr.state_dict())
        else:
            source = reuse_dir / "full_tg_parent.pth"
            if sha256_file(source) != config["full_tg_parent_sha256"]:
                raise ValueError("复用full TG父模型SHA不匹配。")
            full_tg_state = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]
            model.tg_vpr.load_state_dict(full_tg_state, strict=True)
            print(f"full_tg_reused_from={source}")
        atomic_torch_save(output_dir / "full_tg_parent.pth", {"state_dict": full_tg_state})

        folds = fixed_class_folds(seenclasses)
        fold_models = []
        fold_model_shas = {}
        for fold_id, (pseudo_seen, pseudo_unseen) in enumerate(folds):
            mask = torch.isin(train_labels, pseudo_seen)
            fold_positions = mask.nonzero(as_tuple=False).flatten()
            fold_labels = train_labels.index_select(0, fold_positions)
            fold_centroids = h1.visual_centroids(
                train_features.index_select(0, fold_positions), fold_labels, pseudo_seen
            )
            fold_model = VariableClassTGVPR(
                sentence_embeds, pseudo_seen, fold_centroids,
                dropout=float(config["dropout"]), inner_ratio=float(config["inner_ratio"]),
                outer_ratio=float(config["outer_ratio"]), temperature=float(config["temperature"]),
            ).to(device)
            fold_steps = fold_positions.numel() * int(config["fold_tg_epochs"]) // int(config["batch_size"])
            if reuse_dir is None:
                fold_optimizer = torch.optim.Adam(
                    fold_model.parameters(), lr=float(config["fold_tg_lr"]),
                    weight_decay=float(config["weight_decay"]),
                )
                fold_mapping = torch.full((200,), -1, dtype=torch.long)
                fold_mapping[pseudo_seen] = torch.arange(100)
                fold_generator = torch.Generator(device="cpu").manual_seed(seed * 100 + fold_id)
                for _ in range(fold_steps):
                    relative = random_batch_indices(fold_positions.numel(), int(config["batch_size"]), fold_generator)
                    positions = fold_positions.index_select(0, relative)
                    images = train_features.index_select(0, positions).to(device).float()
                    targets = fold_mapping[train_labels.index_select(0, positions)].to(device)
                    fold_optimizer.zero_grad(set_to_none=True)
                    fold_loss = F.cross_entropy(fold_model.logits(images, pseudo_seen), targets) + float(config["topology_weight"]) * fold_model.topology_loss()
                    fold_loss.backward(); require_finite_gradients(fold_model); fold_optimizer.step()
            else:
                source = reuse_dir / f"fold_{fold_id}.pth"
                if sha256_file(source) != config["fold_model_sha256"][str(fold_id)]:
                    raise ValueError(f"复用fold {fold_id} SHA不匹配。")
                payload = torch.load(source, map_location="cpu", weights_only=False)
                if not torch.equal(payload["pseudo_seen"], pseudo_seen) or not torch.equal(payload["pseudo_unseen"], pseudo_unseen):
                    raise ValueError(f"复用fold {fold_id}类别身份不匹配。")
                fold_model.load_state_dict(payload["state_dict"], strict=True)
            fold_model.eval()
            for parameter in fold_model.parameters():
                parameter.requires_grad_(False)
            target = output_dir / f"fold_{fold_id}.pth"
            atomic_torch_save(
                target,
                {
                    "fold_id": fold_id,
                    "pseudo_seen": pseudo_seen,
                    "pseudo_unseen": pseudo_unseen,
                    "steps": fold_steps,
                    "state_dict": copy.deepcopy(fold_model.state_dict()),
                },
            )
            fold_model_shas[str(fold_id)] = sha256_file(target)
            fold_models.append(fold_model)
            print(f"fold={fold_id} trained_classes=100 heldout_classes=50 steps={fold_steps} reused={reuse_dir is not None}")

        model.tg_vpr.load_state_dict(full_tg_state, strict=True)
        set_trainable_stage(model, "TRANSFER_CCGR")
        transfer_optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["transfer_lr"]), weight_decay=float(config["weight_decay"]),
        )
        transfer_generator = torch.Generator(device="cpu").manual_seed(seed * 1000)
        best_h = float("-inf")
        best_metrics = best_state = best_iteration = best_nominal_epoch = best_stage = None
        best_zs_observation = float("-inf")
        history = []

        def evaluate_and_track(global_iteration: int, nominal_epoch: int, stage: str, train_loss: float):
            nonlocal best_h, best_metrics, best_state, best_iteration, best_nominal_epoch, best_stage, best_zs_observation
            metrics = h1.evaluate(model, official, seenclasses, unseenclasses, device)
            history.append({
                "iteration": global_iteration,
                "nominal_epoch": nominal_epoch,
                "stage": stage,
                "train_loss": train_loss,
                "official_metrics_percent": metrics,
                "diagnostics": model.diagnostics(),
            })
            best_zs_observation = max(best_zs_observation, metrics["ZS"])
            if metrics["H"] > best_h:
                best_h = metrics["H"]
                best_metrics = metrics
                best_iteration = global_iteration
                best_nominal_epoch = nominal_epoch
                best_stage = stage
                best_state = copy.deepcopy(model.state_dict())
                atomic_torch_save(
                    output_dir / "model_best.pth",
                    {
                        "experiment_id": config["experiment_id"],
                        "run_id": run_id,
                        "code_commit": code_commit,
                        "config": config,
                        "config_sha256": config_sha,
                        "selected_iteration": best_iteration,
                        "selected_nominal_epoch": best_nominal_epoch,
                        "selected_stage": best_stage,
                        "best_metrics_percent": best_metrics,
                        "model_state_dict": best_state,
                        "reproducibility": reproducibility,
                    },
                )
            print(f"iter={global_iteration} epoch={nominal_epoch} stage={stage} H={metrics['H']:.6f} best_H={best_h:.6f}")

        baseline_metrics = h1.evaluate(model, official, seenclasses, unseenclasses, device)
        best_h = baseline_metrics["H"]
        best_metrics = baseline_metrics
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = int(config["main_tg_steps"]) - 1
        best_nominal_epoch = 49
        best_stage = "FULL_TG_PARENT"
        best_zs_observation = baseline_metrics["ZS"]
        history.append({
            "iteration": best_iteration,
            "nominal_epoch": 49,
            "stage": best_stage,
            "train_loss": None,
            "official_metrics_percent": baseline_metrics,
            "diagnostics": model.diagnostics(),
        })
        atomic_torch_save(
            output_dir / "model_best.pth",
            {
                "experiment_id": config["experiment_id"], "run_id": run_id,
                "code_commit": code_commit, "config": config, "config_sha256": config_sha,
                "selected_iteration": best_iteration, "selected_nominal_epoch": 49,
                "selected_stage": best_stage, "best_metrics_percent": best_metrics,
                "model_state_dict": best_state, "reproducibility": reproducibility,
            },
        )

        for step in range(int(config["transfer_steps"])):
            model.train()
            fold_id = step % int(config["fold_count"])
            pseudo_seen, pseudo_unseen = folds[fold_id]
            indices = balanced_fold_batch(
                train_labels, pseudo_seen, pseudo_unseen, int(config["batch_half"]), transfer_generator
            )
            images = train_features.index_select(0, indices).to(device).float()
            targets = global_to_seen[train_labels.index_select(0, indices)].to(device)
            stages = model.prototype_stages_from_tg(fold_models[fold_id], pseudo_seen)
            competition = stages["final"].index_select(0, seenclasses.to(device))
            logits = F.normalize(images, dim=-1) @ competition.T * fold_models[fold_id].scale()
            ce = F.cross_entropy(logits, targets)
            pseudo_ce = F.cross_entropy(logits[int(config["batch_half"]):], targets[int(config["batch_half"]):])
            topology = topology_loss_between(stages["tg_vpr"], stages["final"], seenclasses.to(device))
            loss = ce + float(config["pseudo_unseen_weight"]) * pseudo_ce + float(config["topology_weight"]) * topology
            transfer_optimizer.zero_grad(set_to_none=True)
            loss.backward(); require_finite_gradients(model); transfer_optimizer.step()
            if step % int(config["report_interval"]) == 0:
                evaluate_and_track(
                    int(config["main_tg_steps"]) + step,
                    50 + step // int(config["report_interval"]),
                    "CLASS_EXCLUSIVE_TRANSFER",
                    float(loss.detach()),
                )

        set_trainable_stage(model, "JOINT_FINETUNE")
        joint_optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["joint_lr"]), weight_decay=float(config["weight_decay"])
        )
        joint_generator = torch.Generator(device="cpu").manual_seed(seed * 10000)
        for step in range(int(config["joint_steps"])):
            model.train()
            indices = random_batch_indices(train_labels.numel(), int(config["batch_size"]), joint_generator)
            images = train_features.index_select(0, indices).to(device).float()
            targets = global_to_seen[train_labels.index_select(0, indices)].to(device)
            loss = F.cross_entropy(model.logits(images, seenclasses), targets) + float(config["topology_weight"]) * model.topology_loss()
            joint_optimizer.zero_grad(set_to_none=True)
            loss.backward(); require_finite_gradients(model); joint_optimizer.step()
            if step % int(config["report_interval"]) == 0:
                evaluate_and_track(
                    int(config["main_tg_steps"]) + int(config["transfer_steps"]) + step,
                    150 + step // int(config["report_interval"]),
                    "JOINT_FINETUNE",
                    float(loss.detach()),
                )

        require_finite_model(model)
        atomic_torch_save(
            output_dir / "checkpoint_last.pth",
            {
                "experiment_id": config["experiment_id"], "run_id": run_id,
                "code_commit": code_commit, "config": config, "config_sha256": config_sha,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "fold_model_state_dicts": [copy.deepcopy(fold.state_dict()) for fold in fold_models],
                "best_model_state_dict": best_state, "best_metrics_percent": best_metrics,
                "selected_iteration": best_iteration, "selected_nominal_epoch": best_nominal_epoch,
                "selected_stage": best_stage, "history": history, "reproducibility": reproducibility,
            },
        )
        atomic_write_json(output_dir / "data_fingerprints.json", {"files": input_sha})
        payload = {
            "experiment_id": config["experiment_id"], "run_id": run_id,
            "framework_id": config["framework_id"], "evaluation_protocol": EVALUATION_PROTOCOL,
            "training_strategy": config["training_strategy"], "selection_scope": config["selection_scope"],
            "nested_official_test_selection": False, "test_used_for_selection": True,
            "unseen_images_used_for_gradient": False, "strict_blind_claim": False,
            "code_commit": code_commit, "config_sha256": config_sha, "seed": seed,
            "main_tg_steps": int(config["main_tg_steps"]),
            "fold_parent_steps": {
                str(i): int(torch.isin(train_labels, folds[i][0]).sum()) * int(config["fold_tg_epochs"]) // int(config["batch_size"])
                for i in range(3)
            },
            "transfer_steps": int(config["transfer_steps"]), "joint_steps": int(config["joint_steps"]),
            "official_test_evaluation_count": len(history), "selected_iteration": best_iteration,
            "selected_nominal_epoch": best_nominal_epoch, "selected_stage": best_stage,
            "best_metrics_percent": best_metrics, "best_zs_observation_percent": best_zs_observation,
            "fold_model_sha256": fold_model_shas,
            "model_sha256": sha256_file(output_dir / "model_best.pth"),
            "checkpoint_last_sha256": sha256_file(output_dir / "checkpoint_last.pth"),
        }
        atomic_write_json(output_dir / "metrics.json", payload)
        print({"best": best_metrics, "iteration": best_iteration, "stage": best_stage})
        return best_metrics
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
