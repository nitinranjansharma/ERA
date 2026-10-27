"""1. Dataset determinism and correct train/validation/test shapes."""

from __future__ import annotations

import torch

from src.config import tiny_config
from src.data import generating_function, make_synthetic_data


def test_dataset_shapes_and_split_sizes():
    cfg = tiny_config(n_train=64, n_val=32, n_test=32, input_dim=8)
    data = make_synthetic_data(cfg, seed=0)
    assert data.x_train.shape == (64, 8)
    assert data.y_train.shape == (64, 1)
    assert data.x_val.shape == (32, 8)
    assert data.y_val.shape == (32, 1)
    assert data.x_test.shape == (32, 8)
    assert data.y_test.shape == (32, 1)
    assert data.shapes()["x_train"] == (64, 8)


def test_dataset_determinism_same_seed():
    cfg = tiny_config(seed=7)
    a = make_synthetic_data(cfg, seed=7)
    b = make_synthetic_data(cfg, seed=7)
    assert torch.equal(a.x_train, b.x_train)
    assert torch.equal(a.y_train, b.y_train)
    assert torch.equal(a.x_val, b.x_val)
    assert torch.equal(a.y_test, b.y_test)


def test_dataset_changes_with_seed():
    cfg = tiny_config()
    a = make_synthetic_data(cfg, seed=1)
    b = make_synthetic_data(cfg, seed=2)
    assert not torch.equal(a.x_train, b.x_train)


def test_split_has_no_overlapping_rows():
    cfg = tiny_config(n_train=40, n_val=20, n_test=20)
    data = make_synthetic_data(cfg, seed=3)
    train = {tuple(row.tolist()) for row in data.x_train}
    val = {tuple(row.tolist()) for row in data.x_val}
    test = {tuple(row.tolist()) for row in data.x_test}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_targets_are_finite_and_match_generating_function_without_noise():
    cfg = tiny_config(noise_std=0.0, seed=4)
    data = make_synthetic_data(cfg, seed=4)
    assert torch.isfinite(data.y_train).all()
    expected = generating_function(data.x_train)
    assert torch.allclose(data.y_train, expected, atol=1e-6)
