"""正式 RUN 的仓库身份、输出边界与 checkpoint 写入契约。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_code_commit() -> str:
    return git_text("rev-parse", "HEAD")


def require_clean_code_tree() -> None:
    if git_text("status", "--porcelain"):
        raise RuntimeError("正式 V1 训练要求 gzsl-paper 工作树干净。")


def prepare_output_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("output-dir 必须使用绝对路径。")
    output_dir = candidate.resolve()
    if output_dir == REPO_ROOT or REPO_ROOT in output_dir.parents:
        raise ValueError("output-dir 必须位于 Git 仓库外。")
    if output_dir.exists():
        raise FileExistsError(f"RUN 输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def is_new_best(metric: float, best_metric: float | None) -> bool:
    value = float(metric)
    if not math.isfinite(value):
        raise ValueError(f"用于选择模型的 H 必须有限，实际为 {value!r}。")
    if best_metric is None:
        return True
    previous = float(best_metric)
    if not math.isfinite(previous):
        raise ValueError(f"历史最佳 H 必须有限，实际为 {previous!r}。")
    return value > previous


def require_finite_metrics(metrics: dict[str, float]) -> None:
    for name in ("U", "S", "H", "ZS"):
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(f"指标 {name} 必须有限，实际为 {value!r}。")


def require_finite_tensor_tree(value, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{name} 包含 NaN/Inf。")
        return
    if isinstance(value, (float, complex)):
        if not math.isfinite(value.real) or (
            isinstance(value, complex) and not math.isfinite(value.imag)
        ):
            raise ValueError(f"{name} 包含 NaN/Inf 标量。")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_tensor_tree(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_tensor_tree(item, f"{name}[{index}]")


def require_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError(f"参数 {name} 的梯度包含 NaN/Inf。")


def require_finite_model(model: torch.nn.Module) -> None:
    require_finite_tensor_tree(model.state_dict(), "model_state_dict")


def validate_best_metrics_identity(
    best_h,
    best_metrics: dict,
    *,
    checkpoint_epoch,
    new_best: bool | None = None,
) -> None:
    require_finite_metrics(best_metrics)
    value = float(best_h)
    if not math.isfinite(value):
        raise ValueError(f"best_H 必须有限，实际为 {value!r}。")
    if value != float(best_metrics["H"]):
        raise ValueError("best_H 与 best_metrics.H 不一致。")
    epoch = best_metrics.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise ValueError("best_metrics.epoch 必须是正整数。")
    if (
        not isinstance(checkpoint_epoch, int)
        or isinstance(checkpoint_epoch, bool)
        or checkpoint_epoch <= 0
    ):
        raise ValueError("checkpoint.epoch 必须是正整数。")
    if epoch > checkpoint_epoch:
        raise ValueError("best_metrics.epoch 不能晚于 checkpoint.epoch。")
    if new_best is True and epoch != checkpoint_epoch:
        raise ValueError("new_best=true 时最佳 epoch 必须等于 checkpoint.epoch。")


def snapshot_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    require_finite_model(model)
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def validate_state_dict_identity(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    expected = model.state_dict()
    if set(state_dict) != set(expected):
        raise ValueError("best_model_state_dict 的参数键与当前模型不一致。")
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape:
            raise ValueError(f"best_model_state_dict[{name!r}] 的 shape 不一致。")
    require_finite_tensor_tree(state_dict, "best_model_state_dict")


def _atomic_torch_save(payload, target: Path) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(target: Path, payload: dict) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_best_model(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    best_state_dict: dict[str, torch.Tensor],
) -> None:
    validate_state_dict_identity(model, best_state_dict)
    _atomic_torch_save(best_state_dict, output_dir / "model_best.pth")


def save_epoch_artifacts(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    checkpoint: dict,
    new_best: bool,
) -> None:
    """最佳模型按需更新，完整续训 checkpoint 每个 epoch 都更新。"""
    require_finite_model(model)
    require_finite_tensor_tree(
        checkpoint["best_model_state_dict"], "checkpoint.best_model_state_dict"
    )
    require_finite_tensor_tree(
        checkpoint["model_state_dict"], "checkpoint.model_state_dict"
    )
    require_finite_tensor_tree(
        checkpoint["optimizer_state_dict"], "checkpoint.optimizer_state_dict"
    )
    require_finite_tensor_tree(
        checkpoint["scheduler_state_dict"], "checkpoint.scheduler_state_dict"
    )
    validate_best_metrics_identity(
        checkpoint["best_H"],
        checkpoint["best_metrics"],
        checkpoint_epoch=checkpoint["epoch"],
        new_best=new_best,
    )
    if new_best:
        materialize_best_model(
            output_dir=output_dir,
            model=model,
            best_state_dict=checkpoint["best_model_state_dict"],
        )
    _atomic_torch_save(checkpoint, output_dir / "checkpoint_last.pth")
