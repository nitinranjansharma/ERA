from __future__ import annotations

import torch

from metrics import aggregate_ratios, first_persistent_crossing, magnitude_ratio, missing_grad_record, update_to_weight_ratio


def test_relative_change_crossing_requires_consecutive_hold():
    steps = list(range(1, 8))
    values = [0.5, 0.4, 0.01, 0.2, 0.009, 0.008, 0.007]
    first = first_persistent_crossing(steps, values, tolerance=0.02, consecutive=3)
    assert first["met"] is True
    assert first["step"] == 5


def test_missing_grad_is_excluded():
    rec = missing_grad_record("head.weight")
    agg = aggregate_ratios([rec])
    assert agg["n_kept"] == 0
    assert agg["n_flagged"] == 1


def test_magnitude_ratio_clamps_near_zero_reference():
    payload = magnitude_ratio(1e-3, 0.0, eps=1e-12)
    assert payload["clamped"] is True
    assert payload["value"] == 1e-3 / 1e-12


def test_update_ratio_finite_for_identical_params():
    t = torch.ones(3)
    rec = update_to_weight_ratio(t, t.clone(), name="w")
    assert rec.ratio == 0.0
    assert not rec.non_finite
