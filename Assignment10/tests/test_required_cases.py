"""The ten assignment test cases, one-to-one with the specification table.

These tests are the grading surface. Domain-specific extras live in the other
test modules; this file is the checklist.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from accumulation_experiment import (
    concatenated_reference,
    correct_accumulation,
    incorrect_accumulation,
    make_microbatches,
)
from gradient_monitor import clip_grad_norm_, run_spike_experiment
from gradient_verification import analytical_gradients, autograd_gradients, central_finite_difference
from mfu import assignment_example_mfu
from models import ScalarTokenModel, TinyMLP
from precision import BF16, FP8_E4M3FN, FP32, analyze_decimal_0p1, encode_rational
from fractions import Fraction
from safety import TrainingSafetyError, assert_grads_finite, assert_loss_ok
from training_loop import (
    calculate_grad_norm,
    demonstrate_forgotten_zero_grad,
    snapshot_params,
    training_step,
)


def test_01_end_to_end_training_step_prints_shapes_and_updates_params():
    model = TinyMLP(in_features=8, hidden=16, out_features=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(4, 8)
    y = torch.randn(4, 1)
    before = snapshot_params(model)

    result = training_step(model, x, y, opt, task="regression")
    records = {r["name"]: r for r in result.tracer.as_dicts()}

    assert records["inputs"]["shape"] == [4, 8]
    assert records["logits"]["shape"] == [4, 1]
    assert records["targets"]["shape"] == [4, 1]
    assert records["loss"]["shape"] == []
    assert records["inputs"]["dims"] == ["batch", "feature"]
    assert records["logits"]["dims"] == ["batch", "out"]
    assert math.isfinite(result.loss)
    assert result.grad_norm > 0.0
    assert any(n.startswith("grad.") for n in records)
    for name, p in model.named_parameters():
        assert p.grad is not None
        assert not torch.equal(before[name], p.detach())


def test_02_gradient_verification_three_methods_agree_on_64():
    analytic = analytical_gradients()
    auto = autograd_gradients(dtype=torch.float64)
    numerical = central_finite_difference(1e-6, dtype=torch.float64)
    assert abs(analytic["dL_dw1"] - 64.0) < 1e-12
    assert abs(auto["dL_dw1"] - 64.0) < 1e-12
    assert abs(numerical - 64.0) / 64.0 < 1e-6


def test_03_incorrect_gradient_accumulation_2p6_vs_3p0():
    batches = make_microbatches((4, 4, 2), (2.0, 2.0, 5.0))
    inc = incorrect_accumulation(ScalarTokenModel(1.0), batches)
    cor = correct_accumulation(ScalarTokenModel(1.0), batches)
    discrepancy = abs(inc["loss"] - cor["loss"]) / cor["loss"]
    assert abs(cor["loss"] - 2.6) < 1e-12
    assert abs(inc["loss"] - 3.0) < 1e-12
    assert abs(discrepancy - 0.1538461538461538) < 1e-12
    assert abs(100.0 * discrepancy - 15.38461538461538) < 1e-8


def test_04_accumulated_gradients_match_combined_batch():
    batches = make_microbatches((4, 4, 2), (2.0, 2.0, 5.0))
    accumulated = correct_accumulation(ScalarTokenModel(1.0), batches)
    combined = concatenated_reference(ScalarTokenModel(1.0), batches)
    assert abs(accumulated["grad"] - combined["grad"]) < 1e-12
    assert abs(accumulated["loss"] - combined["loss"]) < 1e-12


def test_05_gradient_reset_failure_detects_stale_grads():
    payload = demonstrate_forgotten_zero_grad()
    assert payload["allclose_double"], "uncleared second backward must equal 2× a cleared backward"
    assert abs(payload["mean_ratio"] - 2.0) < 1e-4
    assert payload["grad_norm_uncleared_twice"] > payload["grad_norm_single"]
    # Explicit detection, not a silent pass: stale grads are a 2× residue.
    stale = abs(payload["mean_ratio"] - 2.0) < 1e-4
    correctly_reset_differs = not math.isclose(
        payload["grad_norm_single"], payload["grad_norm_uncleared_twice"], rel_tol=1e-4
    )
    assert stale and correctly_reset_differs


def test_06_gradient_clipping_norm_8p4_to_1p0_preserves_direction():
    p = nn.Parameter(torch.zeros(4))
    direction = torch.tensor([1.0, 2.0, 0.5, -1.5], dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction)
    p.grad = direction * 8.4
    before = p.grad.detach().clone()
    pre = float(calculate_grad_norm([p]))
    assert abs(pre - 8.4) < 1e-5

    result = clip_grad_norm_([p], max_norm=1.0)
    assert abs(result.pre_clip_norm - 8.4) < 1e-5
    assert abs(result.post_clip_norm - 1.0) < 1e-5
    assert abs(result.scale - (1.0 / 8.4)) < 1e-6
    cosine = float(F.cosine_similarity(before.flatten(), p.grad.flatten(), dim=0))
    assert cosine > 0.999999
    # Direction preserved means the clipped grad is a positive multiple of the original.
    ratio = (p.grad / before)
    assert torch.allclose(ratio, torch.full_like(ratio, 1.0 / 8.4), rtol=1e-5, atol=1e-6)


def test_07_gradient_norm_anomaly_detection_logs_event_and_ordering():
    payload = run_spike_experiment(steps=40, spike_step=25, spike_factor=50.0, clip_threshold=5.0)
    rows = {r["step"]: r for r in payload["rows"]}
    assert all(
        k in payload["rows"][0]
        for k in ("step", "loss", "pre_clip_norm", "clip_threshold", "scale", "post_clip_norm")
    )
    q = payload["first_qualifying_step"]
    assert q == 25
    assert rows[q]["injected_spike"] is True
    assert rows[q]["pre_clip_norm"] > 5.0 * payload["baseline_pre_clip_norm"]
    assert payload["norm_changed_before_loss_in_this_run"] is True
    assert payload["next_loss_jump"]["step"] == 26
    assert payload["next_loss_jump"]["loss_moved_more_after"] is True


def test_08_mfu_assignment_illustration_648_tflops_8p2_percent():
    payload = assignment_example_mfu()
    assert payload["matches_stated_648_tflops"]
    assert abs(payload["computed_tflops_per_second"] - 648.0) < 1e-9
    assert abs(payload["computed_mfu_percent"] - 8.2) < 0.05
    assert payload["source"].startswith("assignment-provided")


def test_09_floating_point_bit_encoding_of_0p1():
    fp32 = encode_rational(Fraction(1, 10), FP32)
    bf16 = encode_rational(Fraction(1, 10), BF16)
    fp8 = encode_rational(Fraction(1, 10), FP8_E4M3FN)
    payload = analyze_decimal_0p1()
    assert fp32.integer == 0x3DCCCCCD
    assert fp32.grouped_binary == "0 01111011 10011001100110011001101"
    assert bf16.integer == 0x3DCD
    assert fp8.integer == 0x1D
    assert abs(fp8.represented_value - 0.1015625) < 1e-12
    assert payload["matches"]["fp32_matches_struct"]
    assert payload["matches"]["bf16_matches_torch"]
    assert payload["matches"]["fp8_matches_torch"]
    assert payload["passed"]


def test_10_non_finite_loss_and_grads_are_rejected_not_silenced():
    with pytest.raises(TrainingSafetyError, match="not finite"):
        assert_loss_ok(torch.tensor(float("nan")))
    with pytest.raises(TrainingSafetyError, match="not finite"):
        assert_loss_ok(torch.tensor(float("inf")))

    model = TinyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    def nan_loss(_logits, _targets):
        return torch.tensor(float("nan"))

    with pytest.raises(TrainingSafetyError, match="not finite"):
        training_step(model, torch.randn(4, 8), torch.randn(4, 1), opt, loss_fn=nan_loss)

    p = nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([1.0, float("inf")])
    with pytest.raises(TrainingSafetyError, match="Non-finite"):
        assert_grads_finite([p])
    with pytest.raises(TrainingSafetyError, match="Non-finite"):
        clip_grad_norm_([p], max_norm=1.0)
