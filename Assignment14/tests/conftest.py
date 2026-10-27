"""Shared fixtures. Tests never download external datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))
matplotlib.use("Agg")

from src.config import tiny_config  # noqa: E402


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def cfg():
    return tiny_config()
