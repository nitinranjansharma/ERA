"""Reversible language-model training helpers for ERA Session 13."""

from __future__ import annotations

from helpers.config import (
    TARGET_PARAM_COUNT,
    PARAM_COUNT_TOLERANCE,
    ExperimentConfig,
    get_runtime_info,
    set_seed,
    validate_config,
)

__all__ = [
    "TARGET_PARAM_COUNT",
    "PARAM_COUNT_TOLERANCE",
    "ExperimentConfig",
    "get_runtime_info",
    "set_seed",
    "validate_config",
]

__version__ = "1.0.0"
