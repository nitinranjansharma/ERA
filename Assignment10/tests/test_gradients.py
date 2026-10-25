"""Analytical, autograd, and finite-difference agreement."""

from __future__ import annotations

import torch

from gradient_verification import (
    ANALYTICAL,
    analytical_gradients,
    autograd_gradients,
    central_finite_difference,
    nudge_one_weight_in_real_loop,
    sweep_finite_differences,
    verify,
)


def test_intermediates_match_assignment():
    a = analytical_gradients()
    for k, v in ANALYTICAL.items():
        assert abs(a[k] - v) < 1e-12, (k, a[k], v)


def test_autograd_matches_analytical_float64():
    a = analytical_gradients()
    g = autograd_gradients(dtype=torch.float64)
    assert abs(g["dL_dw1"] - a["dL_dw1"]) < 1e-12
    assert abs(g["dL_dw2"] - a["dL_dw2"]) < 1e-12
    assert abs(g["loss"] - 16.0) < 1e-12


def test_finite_difference_preferred_epsilon():
    numerical = central_finite_difference(1e-6, dtype=torch.float64)
    rel = abs(numerical - 64.0) / 64.0
    assert rel < 1e-6, rel


def test_epsilon_tradeoff_visible():
    rows = sweep_finite_differences(dtype=torch.float64)
    by_eps = {r["epsilon"]: r["relative_error"] for r in rows}
    # L(w1) is quadratic, so the central-difference truncation term is identically
    # zero. The remaining error is cancellation, which grows as ε shrinks.
    assert by_eps[1e-12] > by_eps[1e-6]
    assert by_eps[1e-12] > by_eps[1e-8]


def test_real_loop_nudge_agrees_with_backward():
    payload = nudge_one_weight_in_real_loop(epsilon=1e-4)
    assert payload["agrees_to_several_decimals"]
    assert payload["relative_error"] < 1e-6


def test_verify_payload_passes():
    payload = verify()
    assert payload["autograd_pass"]
    assert payload["numerical_pass"]
    assert payload["intermediates_pass"]
    assert payload["passed"]
    assert payload["analytical"]["dL_dw1"] == 64.0
