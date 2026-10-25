"""Training-loop invariants: scalar loss, grads, parameter change, overfitting."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from models import TinyMLP
from safety import TrainingSafetyError, assert_loss_ok
from training_loop import (
    calculate_grad_norm,
    demonstrate_forgotten_zero_grad,
    load_checkpoint,
    overfit_tiny_dataset,
    save_checkpoint,
    training_step,
)


def test_loss_is_scalar_and_finite():
    model = TinyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(8, 8)
    y = torch.randn(8, 1)
    result = training_step(model, x, y, opt)
    assert math.isfinite(result.loss)
    rec = next(r for r in result.tracer.as_dicts() if r["name"] == "loss")
    assert rec["shape"] == []


def test_parameters_change_after_step():
    model = TinyMLP()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    training_step(model, torch.randn(4, 8), torch.randn(4, 1), opt)
    changed = any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
    assert changed


def test_overfit_tiny_dataset():
    payload = overfit_tiny_dataset(steps=400, threshold=1e-4, lr=0.2)
    assert payload["learned"], payload
    assert payload["final_loss"] <= payload["threshold"]
    # A merely decreasing loss is not enough; residual on the same points must be small.
    assert payload["residual_mse"] <= payload["threshold"]
    assert payload["final_loss"] < payload["initial_loss"]


def test_forgotten_zero_grad_approximately_doubles():
    payload = demonstrate_forgotten_zero_grad()
    assert payload["allclose_double"]
    assert abs(payload["mean_ratio"] - 2.0) < 1e-4


def test_nan_loss_is_rejected():
    with pytest.raises(TrainingSafetyError, match="not finite"):
        assert_loss_ok(torch.tensor(float("nan")))


def test_inf_loss_is_rejected():
    with pytest.raises(TrainingSafetyError, match="not finite"):
        assert_loss_ok(torch.tensor(float("inf")))


def test_non_scalar_loss_is_rejected():
    with pytest.raises(TrainingSafetyError, match="scalar"):
        assert_loss_ok(torch.tensor([1.0, 2.0]))


def test_incorrect_target_dtype():
    model = TinyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)
    y = torch.randint(0, 10, (4, 1))  # integer targets for a regression loss
    with pytest.raises(TrainingSafetyError, match="floating-point"):
        training_step(model, x, y, opt, task="regression")


def test_incorrect_shapes():
    model = TinyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)
    y = torch.randn(4, 3)
    with pytest.raises(TrainingSafetyError, match="shape mismatch"):
        training_step(model, x, y, opt, task="regression")
