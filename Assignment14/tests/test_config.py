"""Configuration validation."""

from __future__ import annotations

import pytest

from src.config import ExperimentConfig, tiny_config


def test_rejects_top_k_greater_than_experts():
    with pytest.raises(ValueError):
        tiny_config(num_experts=4, top_k=8)


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        tiny_config(initialization_strategy="partition")


def test_from_dict_round_trip():
    cfg = ExperimentConfig()
    restored = ExperimentConfig.from_dict(cfg.to_dict())
    assert restored.num_experts == 4
    assert restored.top_k == 2
    assert restored.architecture == "ffn"
