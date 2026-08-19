"""gzsl-paper V1 正式训练的输入身份和断点续训工具。"""

import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _windows_file_id(path):
    import ctypes
    from ctypes import wintypes

    class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(BY_HANDLE_FILE_INFORMATION)]
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        0x80,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x02000000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    handle_value = handle if isinstance(handle, int) else handle.value
    if handle_value == invalid_handle:
        raise OSError(ctypes.get_last_error(), f"无法读取文件身份：{path}")
    try:
        info = BY_HANDLE_FILE_INFORMATION()
        if not get_information(handle, ctypes.byref(info)):
            raise OSError(ctypes.get_last_error(), f"无法读取文件身份：{path}")
        file_index = (info.nFileIndexHigh << 32) | info.nFileIndexLow
        return f"win:{info.dwVolumeSerialNumber}:{file_index}"
    finally:
        close_handle(handle)


def file_quick_identity(path):
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    if os.name == "nt":
        file_id = _windows_file_id(resolved)
    else:
        file_id = f"posix:{stat.st_dev}:{stat.st_ino}"
    return {
        "path": str(resolved),
        "file_id": file_id,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _valid_manifest(data, *, expected_names=None):
    if not isinstance(data, dict) or data.get("schema_version") != "gtpj.data_fingerprints.v1":
        return False
    files = data.get("files")
    if not isinstance(files, dict):
        return False
    if expected_names is not None and set(files) != set(expected_names):
        return False
    required = {"path", "file_id", "size_bytes", "mtime_ns", "sha256"}
    return all(
        isinstance(name, str)
        and isinstance(record, dict)
        and set(record) == required
        and isinstance(record["path"], str)
        and isinstance(record["file_id"], str)
        and isinstance(record["size_bytes"], int)
        and record["size_bytes"] >= 0
        and isinstance(record["mtime_ns"], int)
        and record["mtime_ns"] >= 0
        and isinstance(record["sha256"], str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]))
        for name, record in files.items()
    )


def _manifest_bytes(manifest):
    return (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_or_create_fingerprint_manifest(paths, manifest_path):
    manifest_path = Path(manifest_path)
    cached_files = {}
    try:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _valid_manifest(cached, expected_names=paths):
            cached_files = cached["files"]
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass

    identities = {name: file_quick_identity(path) for name, path in paths.items()}
    records = {}
    for name, identity in identities.items():
        cached_record = cached_files.get(name)
        if isinstance(cached_record, dict) and all(
            cached_record.get(field) == identity[field]
            for field in ("path", "file_id", "size_bytes", "mtime_ns")
        ):
            file_hash = cached_record["sha256"]
        else:
            file_hash = sha256_file(identity["path"])
            if file_quick_identity(identity["path"]) != identity:
                raise RuntimeError(f"输入 {name} 在计算 SHA-256 期间发生变化。")
        records[name] = {**identity, "sha256": file_hash}

    manifest = {
        "schema_version": "gtpj.data_fingerprints.v1",
        "files": records,
    }
    payload = _manifest_bytes(manifest)
    if not manifest_path.is_file() or manifest_path.read_bytes() != payload:
        _atomic_write(manifest_path, payload)
    final_payload = manifest_path.read_bytes()
    if final_payload != payload:
        raise RuntimeError("数据指纹清单在写入后发生变化，拒绝继续正式训练。")
    return manifest, hashlib.sha256(final_payload).hexdigest()


def input_record(path, tensor=None):
    record = file_quick_identity(path)
    record["sha256"] = sha256_file(record["path"])
    if tensor is not None:
        record["shape"] = list(tensor.shape)
        record["dtype"] = str(tensor.dtype)
    return record


def input_fingerprints(records):
    return {
        name: {
            key: value
            for key, value in record.items()
            if key not in {"path", "file_id", "mtime_ns"}
        }
        for name, record in records.items()
    }


def validate_stable_input_records(before, after):
    if set(before) != set(after):
        raise ValueError("输入清单在加载前后不一致。")
    for name in before:
        for field in ("path", "file_id", "size_bytes", "mtime_ns"):
            if before[name][field] != after[name][field]:
                raise RuntimeError(
                    f"输入 {name} 在加载期间发生变化，拒绝继续正式训练。"
                )


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("checkpoint 的 rng_state 不完整。")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint 的 CUDA RNG 设备数量与当前机器不一致。")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    elif cuda_states:
        raise ValueError("checkpoint 包含 CUDA RNG，但当前环境没有 CUDA。")


def validate_resume_identity(
    checkpoint,
    *,
    framework_id,
    code_commit,
    config_values,
    config_sha256,
    fingerprints,
    data_manifest_sha256,
    seenclasses,
    unseenclasses,
):
    expected = {
        "framework_id": framework_id,
        "code_commit": code_commit,
        "config": config_values,
        "config_sha256": config_sha256,
        "input_fingerprints": fingerprints,
        "data_manifest_sha256": data_manifest_sha256,
        "seenclasses": list(seenclasses),
        "unseenclasses": list(unseenclasses),
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"checkpoint 的 {field} 与当前正式运行身份不一致。")
