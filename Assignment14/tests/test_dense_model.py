"""2. Dense model forward-pass output shapes and finite predictions."""

from __future__ import annotations

import torch

from src.config import tiny_config
from src.dense_model import DenseRegressor, StrictLinearRegressor, build_dense_model, count_parameters


def test_ffn_dense_forward_shape_and_finite():
    cfg = tiny_config(architecture="ffn", batch_size=16)
    model = build_dense_model(cfg)
    x = torch.randn(16, cfg.input_dim)
    y = model(x)
    assert y.shape == (16, cfg.output_dim)
    assert torch.isfinite(y).all()
    assert count_parameters(model) > 0
    assert isinstance(model, DenseRegressor)


def test_strict_linear_forward_shape_and_finite():
    cfg = tiny_config(architecture="strict_linear")
    model = build_dense_model(cfg)
    assert isinstance(model, StrictLinearRegressor)
    x = torch.randn(8, cfg.input_dim)
    y = model(x)
    assert y.shape == (8, 1)
    assert torch.isfinite(y).all()


def test_dense_rejects_wrong_feature_width():
    model = build_dense_model(tiny_config())
    try:
        model(torch.randn(4, 3))
    except ValueError as exc:
        assert "expected" in str(exc).lower() or "got" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for wrong input width")
