from pathlib import Path

import pytest
import yaml

from model.innovations.train_gtd_tst import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "experiments/v4/tune/TUNE-002_tg_gtd_hparams/configs"


def test_stage1_tune_configs_are_accepted():
    configs = [load_config(path)[0] for path in sorted(CONFIG_ROOT.glob("RUN-*.yaml"))]
    assert len(configs) == 8
    assert {config["experiment_id"] for config in configs} == {
        f"TUNE-002-RUN-{index:03d}" for index in range(1, 9)
    }
    assert all(config["framework_id"] == "FRAMEWORK-V4" for config in configs)
    assert all(config["test_used_for_selection"] is True for config in configs)
    assert all(config["unseen_images_used_for_gradient"] is False for config in configs)


def test_tune_schema_rejects_out_of_range_parameter(tmp_path):
    source = yaml.safe_load((CONFIG_ROOT / "RUN-001.yaml").read_text(encoding="utf-8"))
    source["max_transport_step"] = 8.0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="GTD共享训练参数"):
        load_config(path)


def test_legacy_fixed_contract_is_unchanged():
    config, _ = load_config(ROOT / "config/tries/v3_try_041_gtd_scratch_fixed150.yaml")
    assert config["framework_id"] == "FRAMEWORK-V3-EXPLORATION"
    assert config["gate_loss_weight"] == 1.0
    assert config["max_transport_step"] == 1.5
