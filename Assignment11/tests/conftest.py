from __future__ import annotations

from pathlib import Path

import pytest

from utils import load_config, set_seed


@pytest.fixture
def default_cfg():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


@pytest.fixture
def fast_cfg():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "fast.yaml")


@pytest.fixture(autouse=True)
def _seed():
    set_seed(0, deterministic=True)
