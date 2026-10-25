"""CPU-only test configuration. GPU tests must be explicitly marked and skipped."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from instrumentation import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def _reproducible():
    set_seed(42, deterministic=True)
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA or MPS")


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="GPU not available in this process")
    has_gpu = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if has_gpu:
        return
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
