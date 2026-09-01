from pathlib import Path

import yaml

from model.frameworks.v6.train_ctpm import evaluation_updates, load_config


def test_ctpm_config_identity_is_parseable():
    path = Path("config/tries/v6_try_010_ctpm.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["schema_version"] == "gzsl-paper.v6-ctpm-train.v1"
    assert config["total_updates"] == 28228
    assert config["eval_interval_steps"] == 141
    assert config["required_module_delta_h"] == 1.0
    assert config["test_used_for_selection"] is True


def test_ctpm_evaluation_schedule_matches_fixed_200():
    updates = evaluation_updates(7057, 200, 50)
    assert updates[0] == 0
    assert updates[1] == 1
    assert updates[2] == 142
    assert updates[-1] == 28228
    assert len(updates) == 202
