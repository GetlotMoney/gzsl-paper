"""Backward-compatible FRAMEWORK-V1 command entry."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("model.frameworks.v1.train", run_name="__main__")

