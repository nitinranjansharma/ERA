"""Gradient-norm logging, clipping, and the controlled leading-indicator run."""

from __future__ import annotations

import math

import torch
from torch import nn

from gradient_monitor import clip_grad_norm_, run_spike_experiment
from training_loop import calculate_grad_norm


def test_global_norm_matches_manual():
    p1 = nn.Parameter(torch.zeros(3))
    p2 = nn.Parameter(torch.zeros(2, 2))
    p1.grad = torch.tensor([3.0, 0.0, 4.0])  # norm 5
    p2.grad = torch.ones(2, 2)  # norm 2
    got = float(calculate_grad_norm([p1, p2]))
    expected = math.sqrt(25.0 + 4.0)
    assert abs(got - expected) < 1e-6


def test_missing_grad_skip_vs_error():
    p = nn.Parameter(torch.ones(2), requires_grad=True)
    p.grad = None
    assert float(calculate_grad_norm([p], missing="skip")) == 0.0
    try:
        calculate_grad_norm([p], missing="error")
        assert False, "expected error"
    except Exception as exc:
        assert "Missing gradients" in str(exc)


def test_zero_norm_clip_scale_is_one():
    p = nn.Parameter(torch.ones(2))
    p.grad = torch.zeros(2)
    result = clip_grad_norm_([p], max_norm=1.0)
    assert result.pre_clip_norm == 0.0
    assert result.scale == 1.0
    assert result.post_clip_norm == 0.0


def test_clip_scale_formula():
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3.0, 4.0])  # norm 5
    result = clip_grad_norm_([p], max_norm=1.0)
    assert abs(result.pre_clip_norm - 5.0) < 1e-6
    assert abs(result.scale - 0.2) < 1e-6
    assert abs(result.post_clip_norm - 1.0) < 1e-5


def test_spike_experiment_identifies_step_from_logs():
    payload = run_spike_experiment(steps=40, spike_step=25, spike_factor=50.0, clip_threshold=5.0)
    assert payload["first_qualifying_step"] is not None
    rows_by_step = {r["step"]: r for r in payload["rows"]}
    q = payload["first_qualifying_step"]
    # The qualifying step is taken from recorded rows, not from a hardcoded assertion alone.
    assert rows_by_step[q]["pre_clip_norm"] > rows_by_step[q - 1]["pre_clip_norm"]
    assert payload["norm_changed_before_loss_in_this_run"]
    assert payload["next_loss_jump"]["step"] == q + 1
    for key in ("pre_clip_norm", "clip_threshold", "scale", "post_clip_norm", "loss", "step"):
        assert key in payload["rows"][0]
