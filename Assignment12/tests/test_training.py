"""Toy-model training, gradient averaging, and gradient accumulation."""

from __future__ import annotations

import pytest
import torch

from zero_sim.config import ToyTrainingConfig, get_device
from zero_sim.model import (
    compare_data_parallel_gradients,
    global_batch_size,
    train_toy_model,
    train_with_gradient_accumulation,
)


def test_global_batch_size_and_accumulation():
    assert global_batch_size(2, 4, 3) == 24
    assert global_batch_size(1, 32, 8) == 256
    with pytest.raises(ValueError):
        global_batch_size(0, 4, 3)
    result = train_with_gradient_accumulation()
    assert result["global_batch_size"] == 24
    assert result["microbatches"] == 12
    assert result["grad_error"] == pytest.approx(0.0, abs=1e-5)
    assert result["param_error"] == pytest.approx(0.0, abs=1e-5)


def test_gradient_equivalence_and_parameter_updates():
    result = compare_data_parallel_gradients()
    assert result.equal_batch_grad_error == pytest.approx(0.0, abs=1e-5)
    assert result.unequal_batch_grad_error == pytest.approx(0.0, abs=1e-5)
    assert result.param_error_after_step == pytest.approx(0.0, abs=1e-5)
    assert result.naive_mean_unequal_error > result.unequal_batch_grad_error
    assert result.global_batch == sum((5, 7, 10, 14))


def test_toy_model_training_finite_loss_and_gradients():
    result = train_toy_model()
    assert result["finite_loss"] is True
    assert torch.isfinite(torch.tensor(result["initial_loss"]))
    assert torch.isfinite(torch.tensor(result["final_loss"]))
    assert torch.isfinite(torch.tensor(result["grad_norm_last_step"]))
    assert result["final_loss"] < result["initial_loss"]
    assert result["parameter_count"] > 0
    assert result["device"] == str(get_device())


def test_deterministic_reproducibility():
    cfg = ToyTrainingConfig(steps=8, seed=123)
    first = train_toy_model(cfg)
    second = train_toy_model(cfg)
    assert first["history"] == second["history"]
    assert first["initial_loss"] == second["initial_loss"]
    third = train_toy_model(ToyTrainingConfig(steps=8, seed=124))
    assert third["history"] != first["history"]
