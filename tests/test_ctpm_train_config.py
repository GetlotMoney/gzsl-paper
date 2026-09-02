from pathlib import Path

import yaml

from model.frameworks.v6.train_ctpm import evaluation_updates, load_config


def test_ctpm_config_identity_is_parseable():
    path = Path("config/tries/v6_try_010_r1_brpl.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["schema_version"] == "gzsl-paper.v6-brpl-train.v1"
    assert config["experiment_id"] == "V6-TRY-010-R1"
    assert config["rescue_of_experiment_id"] == "V6-TRY-010"
    assert config["total_updates"] == 28228
    assert config["eval_interval_steps"] == 141
    assert config["required_module_delta_h"] == 1.0
    assert config["test_used_for_selection"] is True
    loaded, digest = load_config(path)
    assert loaded == config
    assert len(digest) == 64


def test_ctpm_evaluation_schedule_matches_fixed_200():
    updates = sorted(evaluation_updates(28228, 141))
    assert updates[:3] == [0, 141, 282]
    assert updates[-2:] == [28200, 28228]
    assert len(updates) == 202
