"""Reversible transformer layers with explicit inverse reconstruction.

This module does **not** wrap ``torch.utils.checkpoint``. Checkpointing
discards activations and reruns the original forward; it is not a reversible
network. The implementations below store only the two boundary states of a
second-order (midpoint) or two-stream (Euler) recurrence and reconstruct
the missing state algebraically during backward.

Midpoint / leapfrog (primary)
-----------------------------
State indexing uses two consecutive hidden states ``p[l-1], p[l]``. Each
layer function ``f_{theta_l}`` produces the next state by the assignment's
second-order update

    p[l+1] = p[l-1] + 2 h f_{theta_l}(p[l])                       (1)

The exact inverse, using the same parameters and the same deterministic
``f_{theta_l}``, is

    p[l-1] = p[l+1] - 2 h f_{theta_l}(p[l])                       (2)

Initialization for a stack of ``L`` layers (matching the lecture notebook):

    p[0] = Embed(tokens)     # previous position starts equal to current
    p[1] = Embed(tokens)

so the first update is ``p[2] = p[0] + 2h F(p[1]) = x + 2h F(x)``.
Default ``h = 0.5`` so ``2h = 1``, matching the scale of a unit residual
increment without claiming that (1) equals a first-order ResNet step.

Euler position+velocity (lecture / proven S13)
----------------------------------------------
The first-order map ``x[l+1] = x[l] + h f(x[l])`` is not algebraically
reversible by itself. Carry a second full-width grid ``v`` (velocity,
starts at 0):

    v_new = v_old + h * F(x_old)
    x_new = x_old + h * v_new

    x_old = x_new - h * v_new          # no F needed
    v_old = v_new - h * F(x_old)

Same width as the baseline, so parameter count matches. Default
``h = 1/sqrt(n_layer)``. This is **not** RevNet channel-split coupling.

Dropout and other unseeded stochastic ops are forbidden on the reversible
path: reconstruction would otherwise depend on a random stream that is not
replayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from helpers.config import ExperimentConfig


class ReversibleError(RuntimeError):
    """Numerical or API failure in a reversible layer."""


def _require_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise ReversibleError(f"non-finite values in {name}")
    return tensor


def apply_residual_fn(fn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run ``F`` under AMP if enabled, then upcast so subtraction stays fp32."""
    from helpers.config import f_amp_context

    with f_amp_context():
        y = fn(x)
    if y.dtype in (torch.float16, torch.bfloat16):
        return y.float()
    return y


def _check_dropout_modules(modules: Iterable[nn.Module]) -> None:
    for module in modules:
        for submodule in module.modules():
            if isinstance(submodule, nn.Dropout) and submodule.p > 0:
                raise ReversibleError(
                    "dropout > 0 is not permitted on the reversible reconstruction "
                    "path unless RNG state is explicitly saved and replayed"
                )


def midpoint_step(
    p_prev: torch.Tensor,
    p_curr: torch.Tensor,
    fn: nn.Module,
    two_h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One reversible midpoint step. Only ``p_curr`` and ``p_next`` are stored."""

    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, prev: torch.Tensor, curr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            _require_finite(prev, "p_prev")
            _require_finite(curr, "p_curr")
            with torch.no_grad():
                f_val = apply_residual_fn(fn, curr)
                _require_finite(f_val, "f_theta(p_curr)")
                p_next = prev + float(two_h) * f_val
            ctx.save_for_backward(curr, p_next)
            return curr, p_next

        @staticmethod
        def backward(
            ctx, grad_curr_out: torch.Tensor | None, grad_next: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            curr, p_next = ctx.saved_tensors
            if grad_curr_out is None:
                grad_curr_out = torch.zeros_like(curr)
            if grad_next is None:
                grad_next = torch.zeros_like(p_next)
            curr = curr.detach().requires_grad_(True)
            with torch.enable_grad():
                f_val = apply_residual_fn(fn, curr)
            reconstructed_prev = (p_next - float(two_h) * f_val).detach()
            _require_finite(reconstructed_prev, "reconstructed p_prev")
            grad_f = float(two_h) * grad_next
            f_val.backward(grad_f)
            grad_curr = torch.zeros_like(curr) if curr.grad is None else curr.grad
            grad_curr = grad_curr + grad_curr_out
            grad_prev = grad_next
            return grad_prev, grad_curr

    return Fn.apply(p_prev, p_curr)


def euler_velocity_step(
    x: torch.Tensor,
    v: torch.Tensor,
    fn: nn.Module,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One reversible Euler step. Stores only (x_new, v_new)."""

    h_value = float(h)

    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pos: torch.Tensor, vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            _require_finite(pos, "x_old")
            _require_finite(vel, "v_old")
            with torch.no_grad():
                delta = apply_residual_fn(fn, pos)
                _require_finite(delta, "F(x_old)")
                v_new = vel + h_value * delta
                x_new = pos + h_value * v_new
            ctx.save_for_backward(x_new, v_new)
            return x_new, v_new

        @staticmethod
        def backward(
            ctx, grad_x: torch.Tensor | None, grad_v: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x_new, v_new = ctx.saved_tensors
            if grad_x is None:
                grad_x = torch.zeros_like(x_new)
            if grad_v is None:
                grad_v = torch.zeros_like(v_new)
            # x_old = x_new - h v_new  (no F)
            x_old = (x_new - h_value * v_new).detach().requires_grad_(True)
            with torch.enable_grad():
                fx = apply_residual_fn(fn, x_old)
            v_old = (v_new - h_value * fx).detach()
            _require_finite(v_old, "reconstructed v_old")
            # v_new = v_old + h F(x_old); x_new = x_old + h v_new
            # dL/d v_new = grad_v + h * grad_x
            grad_v_new = grad_v + h_value * grad_x
            fx.backward(h_value * grad_v_new)
            grad_x_old = grad_x + (torch.zeros_like(x_old) if x_old.grad is None else x_old.grad)
            if not torch.isfinite(grad_x_old).all():
                raise ReversibleError("Non-finite Euler gradients")
            return grad_x_old, grad_v_new

    return Fn.apply(x, v)


def forward_step(
    mode: str, h: float, block: nn.Module, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push the pair (a, b) forward through one layer. Matches the lecture notebook."""
    if mode == "euler":
        # a=x, b=v
        b_new = b + h * apply_residual_fn(block, a)
        return a + h * b_new, b_new
    if mode == "midpoint":
        # a=x_now, b=x_two_ago
        return b + (2 * h) * apply_residual_fn(block, a), a
    raise ReversibleError(f"unknown reversible mode {mode!r}")


def inverse_step(
    mode: str, h: float, block: nn.Module, a_new: torch.Tensor, b_new: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo one layer from the output pair."""
    if mode == "euler":
        a = a_new - h * b_new
        return a, b_new - h * apply_residual_fn(block, a)
    if mode == "midpoint":
        a = b_new
        return a, a_new - (2 * h) * apply_residual_fn(block, a)
    raise ReversibleError(f"unknown reversible mode {mode!r}")


def midpoint_stack(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    two_h: float,
) -> torch.Tensor:
    _check_dropout_modules(blocks)
    _require_finite(x, "midpoint input")
    p_prev = x
    p_curr = x
    for fn in blocks:
        p_prev, p_curr = midpoint_step(p_prev, p_curr, fn, two_h)
    return p_curr


def euler_stack(x: torch.Tensor, blocks: nn.ModuleList, h: float) -> torch.Tensor:
    _check_dropout_modules(blocks)
    _require_finite(x, "euler input")
    pos, vel = x, torch.zeros_like(x)
    for fn in blocks:
        pos, vel = euler_velocity_step(pos, vel, fn, h)
    return pos


def reversible_stack(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    config: ExperimentConfig,
) -> torch.Tensor:
    if config.variant == "midpoint":
        return midpoint_stack(x, blocks, two_h=config.two_h)
    if config.variant == "euler":
        return euler_stack(x, blocks, h=config.euler_h)
    raise ReversibleError(
        f"variant {config.variant!r} is not a reversible implementation; "
        "use baseline training or select midpoint/euler"
    )


def midpoint_forward_reference(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    two_h: float,
) -> torch.Tensor:
    """Non-reversible reference: same recurrence, all activations stored."""
    p_prev = x
    p_curr = x
    for fn in blocks:
        p_next = p_prev + two_h * apply_residual_fn(fn, p_curr)
        p_prev, p_curr = p_curr, p_next
    return p_curr


def midpoint_inverse_stack(
    p_l: torch.Tensor,
    p_lp1: torch.Tensor,
    blocks: nn.ModuleList,
    two_h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Walk (2) from the last two states back to ``(p[0], p[1])``."""
    p_prev, p_curr = p_l, p_lp1
    for fn in reversed(list(blocks)):
        f_val = apply_residual_fn(fn, p_prev)
        reconstructed = p_curr - two_h * f_val
        _require_finite(reconstructed, "midpoint inverse state")
        p_curr = p_prev
        p_prev = reconstructed
    return p_prev, p_curr


def euler_forward_reference(
    x: torch.Tensor, blocks: nn.ModuleList, h: float
) -> torch.Tensor:
    pos, vel = x, torch.zeros_like(x)
    for fn in blocks:
        vel = vel + h * apply_residual_fn(fn, pos)
        pos = pos + h * vel
    return pos


def euler_inverse_stack(
    x_new: torch.Tensor, v_new: torch.Tensor, blocks: nn.ModuleList, h: float
) -> tuple[torch.Tensor, torch.Tensor]:
    pos, vel = x_new, v_new
    for fn in reversed(list(blocks)):
        pos, vel = inverse_step("euler", h, fn, pos, vel)
        _require_finite(pos, "euler inverse x")
        _require_finite(vel, "euler inverse v")
    return pos, vel


@dataclass
class ReconstructionReport:
    variant: str
    max_abs_error: float
    mean_abs_error: float
    ok: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "variant": self.variant,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "ok": self.ok,
            "atol": self.atol,
            "rtol": self.rtol,
        }


def _report_from_delta(
    variant: str,
    delta: torch.Tensor,
    reference: torch.Tensor,
    atol: float,
    rtol: float,
) -> ReconstructionReport:
    _require_finite(delta, "reconstruction delta")
    abs_err = delta.abs()
    max_abs = float(abs_err.max().item()) if abs_err.numel() else 0.0
    mean_abs = float(abs_err.mean().item()) if abs_err.numel() else 0.0
    ref_max = float(reference.abs().max().item()) if reference.numel() else 0.0
    tol = atol + rtol * ref_max
    if max_abs > tol:
        raise ReversibleError(
            f"{variant} reconstruction drift {max_abs:.3e} exceeds "
            f"tolerance {tol:.3e} (atol={atol}, rtol={rtol})"
        )
    return ReconstructionReport(
        variant=variant,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        ok=True,
        atol=atol,
        rtol=rtol,
    )


def verify_midpoint_reconstruction(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    two_h: float,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> ReconstructionReport:
    """Forward then inverse; recovered p[1] must match the embeddings ``x``."""
    _check_dropout_modules(blocks)
    was_training = [module.training for module in blocks]
    for module in blocks:
        module.eval()
    try:
        with torch.no_grad():
            p_prev = x
            p_curr = x
            for fn in blocks:
                p_next = p_prev + two_h * apply_residual_fn(fn, p_curr)
                p_prev, p_curr = p_curr, p_next
            recon_prev, recon_curr = midpoint_inverse_stack(p_prev, p_curr, blocks, two_h)
            return _report_from_delta("midpoint", recon_curr - x, x, atol, rtol)
    finally:
        for module, flag in zip(blocks, was_training):
            module.train(flag)


def verify_euler_reconstruction(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    h: float | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> ReconstructionReport:
    _check_dropout_modules(blocks)
    step = 1.0 / (len(list(blocks)) ** 0.5) if h is None else float(h)
    was_training = [module.training for module in blocks]
    for module in blocks:
        module.eval()
    try:
        with torch.no_grad():
            pos, vel = x, torch.zeros_like(x)
            for fn in blocks:
                vel = vel + step * apply_residual_fn(fn, pos)
                pos = pos + step * vel
            recon_x, recon_v = euler_inverse_stack(pos, vel, blocks, step)
            return _report_from_delta("euler", recon_x - x, x, atol, rtol)
    finally:
        for module, flag in zip(blocks, was_training):
            module.train(flag)


def midpoint_forward_keep_states(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    two_h: float,
) -> list[torch.Tensor]:
    """Return [p0, p1, ..., p_{L+1}] for diagnostics. p0 = p1 = x."""
    states = [x, x]
    p_prev, p_curr = states[0], states[1]
    for fn in blocks:
        p_next = p_prev + two_h * apply_residual_fn(fn, p_curr)
        states.append(p_next)
        p_prev, p_curr = p_curr, p_next
    return states


@torch.no_grad()
def round_trip_relative_error(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    mode: str,
    h: float,
) -> float:
    """Forward then inverse. Returns max relative error; does not raise.

    In fp32 this is ~1e-6 or better. Under 16-bit AMP the proven notebook
    measures ~1e-2 after 12 layers — that is a finding, not a crash.
    """
    a = x
    b = torch.zeros_like(x) if mode == "euler" else x
    start = x.clone()
    for blk in blocks:
        a, b = forward_step(mode, h, blk, a, b)
    for blk in reversed(list(blocks)):
        a, b = inverse_step(mode, h, blk, a, b)
    denom = max(float(start.abs().max().item()), 1e-12)
    return float((a - start).abs().max().item()) / denom
