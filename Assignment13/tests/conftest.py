"""Shared fixtures. Tests never download external datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))

from helpers.config import tiny_config  # noqa: E402


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def base_cfg():
    return tiny_config()


@pytest.fixture
def midpoint_cfg():
    return tiny_config(variant="midpoint", experiment="B")


@pytest.fixture
def euler_cfg():
    return tiny_config(variant="euler", experiment="B")
