from __future__ import annotations

import torch
import torch.nn as nn


def competition_confidence_features(
    logits: torch.Tensor,
    seen_mask: torch.Tensor,
) -> torch.Tensor:
    """从当前seen/unseen竞争分区提取每张图像的6维置信度摘要。"""
    if logits.ndim != 2:
        raise ValueError("SCCC logits必须是[B,C]。")
    mask = torch.as_tensor(seen_mask, device=logits.device, dtype=torch.bool)
    if mask.ndim != 1 or mask.numel() != logits.size(1) or not mask.any() or mask.all():
        raise ValueError("SCCC seen_mask必须把类别空间划分为非空seen/unseen。")
    seen = logits[:, mask]
    unseen = logits[:, ~mask]
    seen_top = seen.topk(min(5, seen.size(1)), dim=1).values
    unseen_top = unseen.topk(min(5, unseen.size(1)), dim=1).values
    seen_max = seen_top[:, 0]
    unseen_max = unseen_top[:, 0]
    seen_mean = seen_top.mean(dim=1)
    unseen_mean = unseen_top.mean(dim=1)
    return torch.stack(
        (
            seen_max,
            unseen_max,
            seen_mean,
            unseen_mean,
            seen_max - unseen_max,
            torch.logsumexp(seen, dim=1) - torch.logsumexp(unseen, dim=1),
        ),
        dim=1,
    )


class SampleConditionedCompetitionCalibration(nn.Module):
    """按样本置信度预测有界seen-logit扣减，零初始化严格返回父logits。"""

    def __init__(
        self,
        hidden_dim: int = 16,
        max_gamma: float = 2.0,
        gamma_mode: str = "signed",
    ):
        super().__init__()
        self.max_gamma = float(max_gamma)
        self.gamma_mode = str(gamma_mode)
        if self.max_gamma <= 0.0:
            raise ValueError("SCCC max_gamma必须为正数。")
        if self.gamma_mode not in ("signed", "nonnegative"):
            raise ValueError("SCCC gamma_mode只允许signed/nonnegative。")
        self.network = nn.Sequential(
            nn.Linear(6, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def gamma(self, logits: torch.Tensor, seen_mask: torch.Tensor) -> torch.Tensor:
        features = competition_confidence_features(logits, seen_mask)
        value = self.max_gamma * torch.tanh(self.network(features)).squeeze(-1)
        return value if self.gamma_mode == "signed" else value.clamp_min(0.0)

    def forward(self, logits: torch.Tensor, seen_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.as_tensor(seen_mask, device=logits.device, dtype=torch.bool)
        gamma = self.gamma(logits, mask)
        return logits - gamma.unsqueeze(1) * mask.to(logits.dtype).unsqueeze(0)

    @torch.no_grad()
    def stats(self, logits: torch.Tensor, seen_mask: torch.Tensor) -> dict[str, float]:
        value = self.gamma(logits, seen_mask)
        return {
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "min": float(value.min()),
            "max": float(value.max()),
            "max_abs": float(value.abs().max()),
        }
