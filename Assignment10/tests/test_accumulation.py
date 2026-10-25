"""Incorrect vs correct micro-batch accumulation, including the equal-token control."""

from __future__ import annotations

import torch

from accumulation_experiment import (
    EXPECTED_CORRECT,
    EXPECTED_DISCREPANCY,
    EXPECTED_INCORRECT,
    arithmetic_losses,
    concatenated_reference,
    correct_accumulation,
    incorrect_accumulation,
    make_microbatches,
    padded_tokens_included_loss,
    run_accumulation_experiment,
)
from models import ScalarTokenModel


def test_assignment_arithmetic():
    a = arithmetic_losses()
    assert abs(a["correct_loss"] - EXPECTED_CORRECT) < 1e-12
    assert abs(a["incorrect_loss"] - EXPECTED_INCORRECT) < 1e-12
    assert abs(a["relative_discrepancy"] - EXPECTED_DISCREPANCY) < 1e-12
    assert abs(a["relative_discrepancy_percent"] - 15.384615384615385) < 1e-8


def test_actual_gradients_match_objectives():
    batches = make_microbatches()
    inc = incorrect_accumulation(ScalarTokenModel(1.0), batches)
    cor = correct_accumulation(ScalarTokenModel(1.0), batches)
    ref = concatenated_reference(ScalarTokenModel(1.0), batches)
    assert abs(inc["loss"] - 3.0) < 1e-12
    assert abs(cor["loss"] - 2.6) < 1e-12
    assert abs(inc["grad"] - 3.0) < 1e-12
    assert abs(cor["grad"] - 2.6) < 1e-12
    assert abs(cor["grad"] - ref["grad"]) < 1e-12
    assert abs(cor["loss"] - ref["loss"]) < 1e-12


def test_equal_token_discrepancy_vanishes():
    payload = run_accumulation_experiment()
    assert payload["equal_token"]["relative_discrepancy"] < 1e-12
    assert abs(payload["equal_token"]["incorrect"]["loss"] - payload["equal_token"]["correct"]["loss"]) < 1e-12


def test_padding_tokens_are_excluded():
    batches = make_microbatches()
    model = ScalarTokenModel(1.0)
    cor = correct_accumulation(model, batches)
    bug = padded_tokens_included_loss(ScalarTokenModel(1.0), batches)
    assert abs(bug - cor["loss"]) > 1.0


def test_full_experiment_passes():
    payload = run_accumulation_experiment()
    assert payload["passed"]
    assert payload["correct"]["total_tokens"] == 10
    running = payload["running_curves"]
    assert running[0]["incorrect_loss"] == running[0]["correct_loss"]
    assert running[-1]["incorrect_loss"] == 3.0
    assert running[-1]["correct_loss"] == 2.6
    assert running[-1]["gap"] > 0
    traj = payload["training_trajectories"]
    assert len(traj["incorrect"]) == len(traj["correct"]) == 25


def test_incorrect_scaling_is_not_the_token_mean():
    """Dividing each micro-batch mean by n_mb is the bug; it is not method B."""
    batches = make_microbatches()
    model = ScalarTokenModel(1.0)
    model.zero_grad(set_to_none=True)
    total = 10
    # Wrong scaling: mean_mb / total_tokens instead of sum_mb / total_tokens
    for coef, mask in batches:
        per_token = model.per_token_loss(coef)
        mb_mean = (per_token * mask).sum() / mask.sum()
        (mb_mean / total).backward()
    wrong = float(model.theta.grad)
    right = correct_accumulation(ScalarTokenModel(1.0), batches)["grad"]
    assert abs(wrong - right) > 0.1
