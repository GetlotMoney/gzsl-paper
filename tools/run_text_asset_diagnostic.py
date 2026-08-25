"""Run one SHA-bound, seen-only FRAMEWORK-V2 text-asset diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile

import yaml

from tools.diagnose_paper_text_assets import (
    OUTPUT_SCHEMA as DIAGNOSTIC_OUTPUT_SCHEMA,
    TRAINING_ASSET_FILES,
    run as run_seen_only_diagnostic,
)
from tools.run_contract import (
    REPO_ROOT,
    atomic_write_json,
    current_code_commit,
    prepare_output_dir,
    require_clean_code_tree,
)
from tools.runtime import sha256_file


CONFIG_SCHEMA = "gzsl-paper.text-diagnostic-run.v1"
FINGERPRINT_SCHEMA = "gzsl-paper.text-diagnostic-fingerprints.v1"
EXPERIMENT_ID = "V2-TUNE-005"
DATASETS = ("CUB", "AWA2", "SUN")
EXPECTED_ARTIFACTS = {
    "config.snapshot.yaml",
    "training.log",
    "metrics.json",
    "data_fingerprints.json",
}
REQUIRED_CONFIG_FIELDS = {
    "schema_version",
    "experiment_id",
    "run_id",
    "dataset",
    "asset_manifest",
    "asset_manifest_sha256",
    "base_role_name",
    "role_variants",
    "official_test_loaded",
    "seen_images_only",
    "unseen_images_used",
    "diagnostic_no_model",
}
REQUIRED_VARIANT_FIELDS = {
    "name",
    "path",
    "sha256",
    "class_order_sha256",
    "class_order_evidence",
}


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{field}必须是64位SHA256。")
    return value.lower()


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}必须是非空字符串。")
    return value.strip()


def _absolute_path(value: object, field: str) -> Path:
    text = _require_nonempty_string(value, field)
    path = Path(text)
    contract_is_absolute = (
        path.is_absolute()
        or PurePosixPath(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
    )
    if not contract_is_absolute:
        raise ValueError(f"{field}必须是绝对路径。")
    return path.resolve()


def validate_config(config: object) -> dict:
    """Validate the standalone YAML contract without touching any data files."""

    if not isinstance(config, dict):
        raise ValueError("诊断RUN配置必须是YAML mapping。")
    missing = REQUIRED_CONFIG_FIELDS - set(config)
    if missing:
        raise ValueError(f"诊断RUN配置缺少字段：{sorted(missing)}")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("诊断RUN配置schema错误。")
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError(f"experiment_id必须为{EXPERIMENT_ID}。")

    run_id = _require_nonempty_string(config["run_id"], "run_id")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id必须是单个安全路径段。")
    if config["dataset"] not in DATASETS:
        raise ValueError("dataset只允许CUB/AWA2/SUN。")
    _absolute_path(config["asset_manifest"], "asset_manifest")
    _validate_sha256(config["asset_manifest_sha256"], "asset_manifest_sha256")

    base_role_name = _require_nonempty_string(config["base_role_name"], "base_role_name")
    if base_role_name == "class-name":
        raise ValueError("base_role_name不能使用保留名class-name。")
    if config["official_test_loaded"] is not False:
        raise ValueError("seen-only诊断要求official_test_loaded=false。")
    if config["seen_images_only"] is not True:
        raise ValueError("seen-only诊断要求seen_images_only=true。")
    if config["unseen_images_used"] is not False:
        raise ValueError("seen-only诊断要求unseen_images_used=false。")
    if config["diagnostic_no_model"] is not True:
        raise ValueError("文本诊断要求diagnostic_no_model=true。")

    variants = config["role_variants"]
    if not isinstance(variants, list):
        raise ValueError("role_variants必须是列表。")
    names = {"class-name", base_role_name}
    for index, variant in enumerate(variants):
        prefix = f"role_variants[{index}]"
        if not isinstance(variant, dict):
            raise ValueError(f"{prefix}必须是mapping。")
        missing_variant = REQUIRED_VARIANT_FIELDS - set(variant)
        if missing_variant:
            raise ValueError(f"{prefix}缺少字段：{sorted(missing_variant)}")
        name = _require_nonempty_string(variant["name"], f"{prefix}.name")
        if name in names:
            raise ValueError(f"role variant名称重复或保留：{name}")
        names.add(name)
        _absolute_path(variant["path"], f"{prefix}.path")
        _validate_sha256(variant["sha256"], f"{prefix}.sha256")
        _validate_sha256(
            variant["class_order_sha256"], f"{prefix}.class_order_sha256"
        )
        _require_nonempty_string(
            variant["class_order_evidence"], f"{prefix}.class_order_evidence"
        )
    return config


def load_config(path: Path) -> tuple[dict, str, bytes]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"诊断RUN配置不存在：{path}")
    raw = path.read_bytes()
    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"诊断RUN配置不是有效UTF-8 YAML：{path}") from error
    return validate_config(parsed), sha256_file(path), raw


def _load_manifest_identity(config: dict) -> tuple[dict, Path, str]:
    manifest_path = _absolute_path(config["asset_manifest"], "asset_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"资产manifest不存在：{manifest_path}")
    actual_sha = sha256_file(manifest_path)
    expected_sha = _validate_sha256(
        config["asset_manifest_sha256"], "asset_manifest_sha256"
    )
    if actual_sha != expected_sha:
        raise ValueError(f"资产manifest SHA不匹配：{actual_sha}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"资产manifest不是有效UTF-8 JSON：{manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "gzsl-paper.clip-assets.v1":
        raise ValueError("资产manifest schema错误。")
    if manifest.get("dataset") != config["dataset"]:
        raise ValueError("配置dataset与资产manifest不一致。")
    manifest_class_order = _validate_sha256(
        manifest.get("class_order_sha256"), "asset manifest class_order_sha256"
    )
    for index, variant in enumerate(config["role_variants"]):
        variant_class_order = _validate_sha256(
            variant["class_order_sha256"],
            f"role_variants[{index}].class_order_sha256",
        )
        if variant_class_order != manifest_class_order:
            raise ValueError(
                f"role_variants[{index}]类别顺序SHA与资产manifest不一致。"
            )
        _require_nonempty_string(
            variant["class_order_evidence"],
            f"role_variants[{index}].class_order_evidence",
        )
    return manifest, manifest_path, manifest_class_order


def validate_output_path(path: Path, run_id: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("output-dir 必须使用绝对路径。")
    output_dir = candidate.resolve()
    if output_dir == REPO_ROOT or REPO_ROOT in output_dir.parents:
        raise ValueError("output-dir 必须位于 Git 仓库外。")
    if output_dir.name != run_id:
        raise ValueError("output-dir末级目录名必须与run_id完全一致。")
    if output_dir.exists():
        raise FileExistsError(f"RUN 输出目录已存在，拒绝覆盖：{output_dir}")
    return output_dir


def _validate_expected_commit(expected_commit: str, actual_commit: str) -> str:
    expected = _require_nonempty_string(expected_commit, "expected_commit").lower()
    actual = _require_nonempty_string(actual_commit, "current code commit").lower()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected):
        raise ValueError("expected-commit必须是完整Git commit SHA。")
    if actual != expected:
        raise RuntimeError(
            f"当前code commit与--expected-commit不一致：{actual} != {expected}"
        )
    return actual


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
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
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fingerprints(
    *,
    config_path: Path,
    config_sha256: str,
    code_commit: str,
    config: dict,
    manifest: dict,
    manifest_path: Path,
    class_order_sha256: str,
    diagnostics: dict,
) -> dict:
    asset_inputs = {}
    for filename, file_sha in diagnostics["input_sha256"].items():
        source_path = manifest_path if filename == "asset_manifest.json" else manifest_path.parent / filename
        asset_inputs[filename] = {
            "path": str(source_path.resolve()),
            "sha256": file_sha,
        }
    variants = []
    diagnostic_sources = diagnostics["role_variant_sources"]
    for variant in config["role_variants"]:
        source = diagnostic_sources[variant["name"]]
        variants.append(
            {
                "name": variant["name"],
                "path": source["path"],
                "sha256": source["sha256"],
                "class_order_sha256": source["class_order_sha256"],
                "class_order_evidence": variant["class_order_evidence"].strip(),
            }
        )
    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "dataset": config["dataset"],
        "code_commit": code_commit,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": config_sha256,
        },
        "asset_id": manifest.get("asset_id"),
        "asset_manifest": str(manifest_path),
        "asset_manifest_sha256": diagnostics["input_sha256"]["asset_manifest.json"],
        "class_order_sha256": class_order_sha256,
        "asset_inputs": asset_inputs,
        "role_variants": variants,
        "official_test_loaded": False,
        "seen_images_only": True,
        "unseen_images_used": False,
        "diagnostic_no_model": True,
    }


def run(config_path: Path, output_dir: Path, *, expected_commit: str) -> dict:
    """Execute and materialize one immutable four-artifact diagnostic RUN."""

    config_path = Path(config_path).resolve()
    config, config_sha256, raw_config = load_config(config_path)
    checked_output = validate_output_path(Path(output_dir), config["run_id"])
    manifest, manifest_path, class_order_sha256 = _load_manifest_identity(config)

    require_clean_code_tree()
    code_commit = _validate_expected_commit(expected_commit, current_code_commit())

    variants = [
        (
            variant["name"],
            _absolute_path(variant["path"], f"role_variants[{index}].path"),
            _validate_sha256(variant["sha256"], f"role_variants[{index}].sha256"),
        )
        for index, variant in enumerate(config["role_variants"])
    ]
    with tempfile.TemporaryDirectory(prefix="gzsl-text-diagnostic-") as temporary:
        diagnostic_path = Path(temporary) / "metrics.json"
        diagnostics = run_seen_only_diagnostic(
            manifest_path,
            diagnostic_path,
            base_role_name=config["base_role_name"],
            expected_manifest_sha256=config["asset_manifest_sha256"],
            role_variants=variants,
        )

    if diagnostics.get("schema_version") != DIAGNOSTIC_OUTPUT_SCHEMA:
        raise RuntimeError("底层文本诊断schema错误。")
    if diagnostics.get("dataset") != config["dataset"]:
        raise RuntimeError("底层文本诊断dataset与RUN配置不一致。")
    if diagnostics.get("class_order_sha256") != class_order_sha256:
        raise RuntimeError("底层文本诊断类别顺序与RUN配置不一致。")
    expected_boundary = {
        "official_test_loaded": False,
        "seen_images_only": True,
        "unseen_images_used": False,
    }
    if any(diagnostics.get(key) is not value for key, value in expected_boundary.items()):
        raise RuntimeError("底层文本诊断违反seen-only数据边界。")

    for variant in config["role_variants"]:
        diagnostics["role_variant_sources"][variant["name"]][
            "class_order_evidence"
        ] = variant["class_order_evidence"].strip()
    metrics = {
        **diagnostics,
        "run_config_schema_version": CONFIG_SCHEMA,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "code_commit": code_commit,
        "config_sha256": config_sha256,
        "base_role_name": config["base_role_name"],
        "diagnostic_no_model": True,
    }
    fingerprints = _fingerprints(
        config_path=config_path,
        config_sha256=config_sha256,
        code_commit=code_commit,
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
        class_order_sha256=class_order_sha256,
        diagnostics=diagnostics,
    )

    materialized = prepare_output_dir(checked_output)
    _atomic_write_bytes(materialized / "config.snapshot.yaml", raw_config)
    atomic_write_json(materialized / "metrics.json", metrics)
    atomic_write_json(materialized / "data_fingerprints.json", fingerprints)
    log = "\n".join(
        (
            "run_type=seen-only text asset diagnostic",
            f"experiment_id={config['experiment_id']}",
            f"run_id={config['run_id']}",
            f"dataset={config['dataset']}",
            f"code_commit={code_commit}",
            f"config_sha256={config_sha256}",
            "official_test_loaded=false",
            "seen_images_only=true",
            "unseen_images_used=false",
            "diagnostic_no_model=true",
            "status=complete",
            "",
        )
    ).encode("utf-8")
    _atomic_write_bytes(materialized / "training.log", log)

    actual_artifacts = {path.name for path in materialized.iterdir()}
    if actual_artifacts != EXPECTED_ARTIFACTS:
        raise RuntimeError(
            "诊断RUN产物集合错误："
            f"实际={sorted(actual_artifacts)}，预期={sorted(EXPECTED_ARTIFACTS)}"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    result = run(
        args.config,
        args.output_dir,
        expected_commit=args.expected_commit,
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "run_id": result["run_id"],
                "dataset": result["dataset"],
                "output_dir": str(args.output_dir.resolve()),
                "diagnostic_no_model": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
