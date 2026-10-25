"""Midpoint/Euler reconstruction, gradients, and numerical safeguards."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from helpers.config import tiny_config
from helpers.model import build_model
from helpers.reversible import (
    ReversibleError,
    midpoint_forward_reference,
    midpoint_stack,
    round_trip_relative_error,
    verify_euler_reconstruction,
    verify_midpoint_reconstruction,
)


def test_midpoint_reconstruction_fp32_and_fp64():
    config = tiny_config(experiment="B", variant="midpoint")
    model = build_model(config)
    model.eval()
    x = torch.randn(2, config.block_size, config.n_embd)
    report = verify_midpoint_reconstruction(x, model.blocks, two_h=config.two_h, atol=5e-4, rtol=1e-4)
    assert report.ok
    assert report.max_abs_error < 5e-4
    x64 = x.to(torch.float64)
    blocks64 = model.blocks.to(dtype=torch.float64)
    report64 = verify_midpoint_reconstruction(x64, blocks64, two_h=config.two_h, atol=1e-8, rtol=1e-8)
    assert report64.max_abs_error < 1e-7


def test_round_trip_relative_error_is_tiny_in_fp32():
    config = tiny_config(experiment="B", variant="midpoint")
    model = build_model(config).eval()
    x = torch.randn(2, config.block_size, config.n_embd)
    rel = round_trip_relative_error(x, model.blocks, "midpoint", config.midpoint_step_size)
    assert rel < 1e-5


def test_euler_reconstruction():
    config = tiny_config(experiment="B", variant="euler")
    model = build_model(config)
    model.eval()
    x = torch.randn(2, config.block_size, config.n_embd)
    report = verify_euler_reconstruction(x, model.blocks, h=config.euler_h, atol=5e-4, rtol=1e-4)
    assert report.ok


def test_euler_gradients_match_reference():
    from helpers.reversible import euler_forward_reference, euler_stack

    config = tiny_config(experiment="B", variant="euler", n_layer=2)
    model = build_model(config)
    x = torch.randn(2, 4, config.n_embd, requires_grad=True)
    y = euler_stack(x, model.blocks, h=config.euler_h)
    y.square().mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    model.zero_grad(set_to_none=True)
    x2 = x.detach().clone().requires_grad_(True)
    euler_forward_reference(x2, model.blocks, h=config.euler_h).square().mean().backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        assert torch.allclose(param.grad, grads[name], atol=1e-4, rtol=1e-4), name


def test_midpoint_gradients_match_reference_and_are_finite():
    config = tiny_config(experiment="B", variant="midpoint", n_layer=2)
    model = build_model(config)
    x = torch.randn(2, 4, config.n_embd, requires_grad=True)
    y_rev = midpoint_stack(x, model.blocks, two_h=config.two_h)
    loss_rev = y_rev.square().mean()
    loss_rev.backward()
    grads_rev = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    assert grads_rev, "reversible backward produced no parameter gradients"
    assert all(torch.isfinite(g).all() for g in grads_rev.values())
    assert x.grad is not None and torch.isfinite(x.grad).all()

    model.zero_grad(set_to_none=True)
    x2 = x.detach().clone().requires_grad_(True)
    y_ref = midpoint_forward_reference(x2, model.blocks, two_h=config.two_h)
    y_ref.square().mean().backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        assert torch.allclose(param.grad, grads_rev[name], atol=1e-4, rtol=1e-4), name
    assert torch.allclose(x2.grad, x.grad, atol=1e-4, rtol=1e-4)


def test_dropout_and_nonfinite_are_rejected():
    config = tiny_config(experiment="B", variant="midpoint")
    model = build_model(config)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.5
    x = torch.randn(1, config.block_size, config.n_embd)
    with pytest.raises(ReversibleError, match="dropout"):
        verify_midpoint_reconstruction(x, model.blocks, two_h=1.0)

    class Exploding(nn.Module):
        def forward(self, hidden):
            return hidden * float("nan")

    with pytest.raises(ReversibleError, match="non-finite"):
        verify_midpoint_reconstruction(
            x, nn.ModuleList([Exploding()]), two_h=1.0, atol=1.0, rtol=1.0
        )


def test_reconstruction_drift_raises():
    class Drift(nn.Module):
        def forward(self, hidden):
            return hidden + 1.0

    x = torch.randn(1, 4, 8)
    # Inverse uses the same f, so this actually reconstructs. Force a mismatch
    # by checking a report helper via a broken inverse tolerance on noise.
    noisy = x + 10.0
    with pytest.raises(ReversibleError, match="drift"):
        from helpers.reversible import _report_from_delta

        _report_from_delta("midpoint", noisy - x, x, atol=1e-8, rtol=0.0)
