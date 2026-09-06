import csv
from pathlib import Path

import yaml

from model.frameworks.v5 import V5_DEPLOYMENT


ROOT = Path(__file__).resolve().parents[3]


def test_v5_framework_identity_and_parameters_are_consistent():
    framework = yaml.safe_load((ROOT / "experiments/v5/FRAMEWORK.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "config/framework_v5.yaml").read_text(encoding="utf-8"))
    assert framework["framework_id"] == "FRAMEWORK-V5"
    assert framework["framework_branch"] == "framework/v5"
    assert framework["framework_tag"] == "v5"
    assert framework["formal_config"] == "config/framework_v5.yaml"
    assert config["deployment"]["candidate_top_k"] == V5_DEPLOYMENT["candidate_top_k"]
    assert config["deployment"]["ridge_lambda"] == V5_DEPLOYMENT["ridge_lambda"]
    assert config["deployment"]["correction_scale"] == V5_DEPLOYMENT["correction_scale"]
    assert config["deployment"]["role0_weight"] == V5_DEPLOYMENT["role0_weight"]
    assert config["deployment"]["role6_weight"] == V5_DEPLOYMENT["role6_weight"]
    assert config["deployment"]["seen_logit_gamma"] == V5_DEPLOYMENT["seen_logit_gamma"]
    assert framework["formal_result"]["metrics_sha256"] == config["formal_metrics_sha256"]
    for metric in ("U", "S", "H", "ZS"):
        assert framework["formal_result"][metric] == config["metrics"][metric]


def test_v5_results_and_required_layout_are_complete():
    with (ROOT / "experiments/v5/evidence/RESULTS.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    promoted = next(row for row in rows if row["status"] == "promoted")
    assert promoted["dataset"] == "CUB"
    assert float(promoted["H"]) == 81.068777
    assert promoted["metrics_sha256"] == "efbdca19f8248b2e16c99baa7aa5a81d2279218db910a9a00e7303d45d2fc2bc"
    required = (
        "experiments/v5/framework_diagram.html",
        "experiments/v5/EXPERIMENT_QUEUE.csv",
        "experiments/v5/tune/INDEX.md",
        "experiments/v5/ablation/INDEX.md",
        "experiments/v5/innovation/INDEX.md",
        "experiments/v5/confirmation/INDEX.md",
        "experiments/v5/evidence/ARTIFACTS.yaml",
        "experiments/v5/evidence/AUDIT.md",
    )
    assert all((ROOT / path).is_file() for path in required)
