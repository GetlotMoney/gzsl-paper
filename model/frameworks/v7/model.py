"""Standalone graph-free deployment entry for FRAMEWORK-V7."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_COUNT = 200
EMBED_DIM = 768
HIDDEN_DIM = 64


class V7DeploymentModel(nn.Module):
    """Execute only ``h(x) Q^T + b`` from an exported C-PCLR checkpoint."""

    def __init__(
        self,
        *,
        q: torch.Tensor,
        bias: torch.Tensor,
        reader_in_weight: torch.Tensor,
        reader_in_bias: torch.Tensor,
        reader_out_weight: torch.Tensor,
        reader_out_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        tensors = {
            "q": torch.as_tensor(q).detach().cpu().float().clone(),
            "bias": torch.as_tensor(bias).detach().cpu().float().clone(),
            "reader_in_weight": torch.as_tensor(reader_in_weight).detach().cpu().float().clone(),
            "reader_in_bias": torch.as_tensor(reader_in_bias).detach().cpu().float().clone(),
            "reader_out_weight": torch.as_tensor(reader_out_weight).detach().cpu().float().clone(),
            "reader_out_bias": torch.as_tensor(reader_out_bias).detach().cpu().float().clone(),
        }
        expected = {
            "q": (CLASS_COUNT, 2 * EMBED_DIM),
            "bias": (CLASS_COUNT,),
            "reader_in_weight": (HIDDEN_DIM, EMBED_DIM),
            "reader_in_bias": (HIDDEN_DIM,),
            "reader_out_weight": (EMBED_DIM, HIDDEN_DIM),
            "reader_out_bias": (EMBED_DIM,),
        }
        for name, value in tensors.items():
            if tuple(value.shape) != expected[name] or not torch.isfinite(value).all():
                raise ValueError(f"FRAMEWORK-V7 export tensor错误：{name}")
            self.register_buffer(name, value, persistent=True)

    @classmethod
    def from_export(cls, export: dict[str, torch.Tensor]) -> "V7DeploymentModel":
        required = {
            "q",
            "bias",
            "reader_in_weight",
            "reader_in_bias",
            "reader_out_weight",
            "reader_out_bias",
        }
        if not isinstance(export, dict) or set(export) != required:
            raise ValueError("FRAMEWORK-V7 checkpoint export字段错误。")
        return cls(**export)

    def read_images(self, image_features: torch.Tensor) -> torch.Tensor:
        if (
            image_features.ndim != 2
            or image_features.size(0) == 0
            or image_features.size(1) != EMBED_DIM
            or not torch.isfinite(image_features).all()
        ):
            raise ValueError("FRAMEWORK-V7图像特征必须是有限非空[batch,768]。")
        values = image_features.float()
        hidden = F.linear(values, self.reader_in_weight, self.reader_in_bias)
        residual = F.linear(F.gelu(hidden), self.reader_out_weight, self.reader_out_bias)
        return F.normalize(values + residual, dim=-1)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image = F.normalize(image_features.float(), dim=-1)
        readout = self.read_images(image_features)
        logits = torch.cat((image, readout), dim=1) @ self.q.T + self.bias
        if tuple(logits.shape) != (len(image_features), CLASS_COUNT):
            raise RuntimeError("FRAMEWORK-V7 logits shape错误。")
        return logits


def load_v7_checkpoint(
    checkpoint_path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[V7DeploymentModel, dict]:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=map_location, weights_only=True
    )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("experiment_id") != "V6-TRY-006"
        or checkpoint.get("code_commit") != "8de7cebda0235ab12e1b4b8f669134c8f4e2c075"
        or checkpoint.get("config_sha256")
        != "73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b"
        or not isinstance(checkpoint.get("export"), dict)
    ):
        raise ValueError("FRAMEWORK-V7 checkpoint身份错误。")
    return V7DeploymentModel.from_export(checkpoint["export"]), checkpoint


@torch.no_grad()
def v7_logits(model: V7DeploymentModel, image_features: torch.Tensor) -> torch.Tensor:
    return model(image_features)
