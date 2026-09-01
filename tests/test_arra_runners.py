from pathlib import Path

import torch

from model.frameworks.v6.train_arra import (
    build_optimizer_and_scheduler,
    load_config,
    parameter_groups_for_optimizer,
)


def test_arra_official_config_contract():
    config, digest = load_config(Path("config/tries/v6_try_009_arra.yaml"))
    assert digest
    assert config["batch_size"] == 50
    assert config["nominal_epochs"] == 200
    assert config["total_updates"] == 28228
    assert config["eval_interval_steps"] == 141
    assert config["condition_id"] == "ARRA_ONE_STAGE_E2E"
    assert config["test_used_for_selection"] is True
    assert config["test_used_for_hyperparameter_selection"] is True
    assert config["nested_official_test_selection"] is True
    assert config["unseen_images_used_for_gradient"] is False
    assert config["seen_logit_gamma"] == 0.575
    assert config["initial_alpha"] == 1.0
    assert config["initial_delta"] == 0.0


class _FakeARRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.role = torch.nn.Parameter(torch.tensor(0.0))
        self.reader = torch.nn.Parameter(torch.tensor(1.0))
        self.visual = torch.nn.Parameter(torch.tensor(2.0))
        self.delta = torch.nn.Parameter(torch.tensor(3.0))

    def training_parameter_groups(self):
        return {
            "slow": (self.role, self.reader),
            "fast": (self.visual, self.delta),
        }


def test_optimizer_uses_two_simultaneous_nonoverlapping_lr_groups():
    config, _digest = load_config(Path("config/tries/v6_try_009_arra.yaml"))
    model = _FakeARRA()
    slow, fast = parameter_groups_for_optimizer(model)
    assert {id(parameter) for parameter in slow} == {id(model.role), id(model.reader)}
    assert {id(parameter) for parameter in fast} == {id(model.visual), id(model.delta)}

    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == config["slow_learning_rate"]
    assert optimizer.param_groups[1]["lr"] == config["fast_learning_rate"]
    total = config["total_updates"]
    assert abs(
        scheduler.lr_lambdas[0](total)
        - config["slow_min_learning_rate"] / config["slow_learning_rate"]
    ) < 1e-12
    assert abs(
        scheduler.lr_lambdas[1](total)
        - config["fast_min_learning_rate"] / config["fast_learning_rate"]
    ) < 1e-12
