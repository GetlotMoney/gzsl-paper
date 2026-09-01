from pathlib import Path

import pytest

from model.frameworks.v6.train_rgra import load_config


def test_rgra_official_config_contract():
    config, digest = load_config(Path("config/tries/v6_try_008_rgra.yaml"))
    assert digest
    assert config["batch_size"] == 50
    assert config["nominal_epochs"] == 200
    assert config["total_updates"] == 28228
    assert config["eval_interval_steps"] == 141
    assert config["condition_id"] == "RGRA_ONE_STAGE_E2E"
    assert config["test_used_for_selection"] is True
    assert config["unseen_images_used_for_gradient"] is False

