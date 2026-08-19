import os
import random

import numpy as np
import torch


def config_bool(config, name, default=False):
    return bool(getattr(config, name, default))


def config_int(config, name, default):
    value = getattr(config, name, default)
    if value is None or value == "":
        return int(default)
    return int(value)


def configure_reproducibility(seed, strict_determinism=False, deterministic_warn_only=True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    if strict_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=bool(deterministic_warn_only))
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        if torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(False)

    return reproducibility_state(seed, strict_determinism, deterministic_warn_only)


def reproducibility_state(seed, strict_determinism=False, deterministic_warn_only=True):
    return {
        "seed": int(seed),
        "strict_determinism": bool(strict_determinism),
        "deterministic_warn_only": bool(deterministic_warn_only),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }


def make_batch_generator(enabled, seed):
    if not enabled:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator
