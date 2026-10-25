"""MFU formula, unit hygiene, and the assignment sanity-check illustration."""

from __future__ import annotations

import pytest

from mfu import (
    ASSIGNMENT_EXAMPLE,
    assignment_example_mfu,
    estimated_training_flops_per_second,
    flops_to_tflops,
    milliseconds_to_seconds,
    mfu_percent,
    tflops_to_flops,
)
from models import TinyGPT, count_parameters
from safety import MFUError


def test_assignment_illustration():
    payload = assignment_example_mfu()
    assert payload["source"].startswith("assignment-provided")
    assert payload["matches_stated_648_tflops"]
    assert payload["matches_stated_8p2_percent"]
    assert abs(payload["computed_mfu_percent"] - ASSIGNMENT_EXAMPLE["approximate_mfu_percent"]) < 0.05


def test_six_n_d_formula():
    flops = estimated_training_flops_per_second(9_000_000_000, 12_000)
    assert abs(flops_to_tflops(flops) - 648.0) < 1e-9


def test_rejects_non_positive_peak():
    with pytest.raises(MFUError):
        mfu_percent(1.0, 0.0)


def test_tflop_vs_flop_conversion():
    # 648 TFLOP/s is 6.48e14 FLOP/s. Treating 648 as already FLOP/s would be a 1e12 error.
    tflops = 648.0
    flops = tflops_to_flops(tflops)
    assert flops == 648.0 * 1e12
    with pytest.raises(MFUError):
        tflops_to_flops(-1.0)
    # Using the unconverted number as the peak would make MFU ~1e12 times too large.
    wrong = mfu_percent(flops, tflops)  # denominator still in TFLOP/s by mistake
    right = mfu_percent(flops, flops)  # 100% if peak == achieved
    assert wrong > 1e11
    assert abs(right - 100.0) < 1e-9


def test_seconds_vs_milliseconds():
    assert milliseconds_to_seconds(1000.0) == 1.0
    tokens = 12_000
    elapsed_s = 1.0
    elapsed_ms = 1000.0
    tps_right = tokens / elapsed_s
    tps_wrong = tokens / elapsed_ms  # treating milliseconds as seconds
    assert tps_right == 12_000
    assert tps_wrong == 12.0
    assert tps_wrong != tps_right


def test_parameter_count_convention():
    model = TinyGPT()
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    assert trainable > 0
    assert total == trainable  # TinyGPT has no frozen params
    # Frozen params must not be silently included when trainable_only=True
    for p in model.parameters():
        p.requires_grad = False
        break
    assert count_parameters(model, trainable_only=True) < total


def test_incorrect_parameter_count_changes_mfu():
    tps = 12_000
    peak = tflops_to_flops(7_912)
    right = mfu_percent(estimated_training_flops_per_second(9_000_000_000, tps), peak)
    wrong = mfu_percent(estimated_training_flops_per_second(6_000_000_000, tps), peak)
    assert abs(right - wrong) > 1.0
