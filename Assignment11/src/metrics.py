"""Numerical metrics: ratios, relative differences, finite-value policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


class NonFiniteError(ValueError):
    """Raised when a loss, gradient, parameter, or ratio is not finite."""


def require_finite(value: float | int, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteError(f"{name} is not finite: {number}")
    return number


def tensor_norm(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    value = float(torch.linalg.vector_norm(tensor.detach()).item())
    return require_finite(value, "tensor_norm")


def relative_difference(actual: float, reference: float, eps: float = 1e-12) -> dict[str, float | bool]:
    """Relative difference with a clamped denominator.

    Definition
    ----------
        rel_diff(a, b; eps) = |a - b| / max(|b|, eps)

    ``b`` is the reference (bias-corrected update by default). If ``|b| < eps``
    the denominator is replaced by ``eps`` and ``clamped`` is True so a zero
    reference cannot produce a division error.
    """
    a = require_finite(actual, "relative_difference.actual")
    b = require_finite(reference, "relative_difference.reference")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    abs_ref = abs(b)
    clamped = abs_ref < eps
    denom = eps if clamped else abs_ref
    value = abs(a - b) / denom
    return {
        "value": require_finite(value, "relative_difference"),
        "numerator": abs(a - b),
        "denominator": denom,
        "clamped": clamped,
        "eps": eps,
    }


def magnitude_ratio(actual: float, reference: float, eps: float = 1e-12) -> dict[str, float | bool]:
    """|actual| / max(|reference|, eps)."""
    a = require_finite(actual, "magnitude_ratio.actual")
    b = require_finite(reference, "magnitude_ratio.reference")
    abs_ref = abs(b)
    clamped = abs_ref < eps
    denom = eps if clamped else abs_ref
    value = abs(a) / denom
    return {
        "value": require_finite(value, "magnitude_ratio"),
        "clamped": clamped,
        "denominator": denom,
        "eps": eps,
    }


@dataclass
class RatioRecord:
    name: str
    param_norm: float
    update_norm: float
    ratio: float | None
    near_zero_weight: bool
    missing_grad: bool
    non_finite: bool
    excluded_from_aggregate: bool
    policy: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "param_norm": self.param_norm,
            "update_norm": self.update_norm,
            "ratio": self.ratio,
            "near_zero_weight": self.near_zero_weight,
            "missing_grad": self.missing_grad,
            "non_finite": self.non_finite,
            "excluded_from_aggregate": self.excluded_from_aggregate,
            "policy": self.policy,
        }


def update_to_weight_ratio(
    before: torch.Tensor,
    after: torch.Tensor,
    eps: float = 1e-12,
    near_zero_eps: float = 1e-8,
    name: str = "param",
) -> RatioRecord:
    """ratio = ||theta_after - theta_before|| / max(||theta_before||, eps).

    This uses the actual parameter displacement, not the gradient norm.

    Near-zero policy
    ----------------
    If ``||theta_before|| < near_zero_eps`` the ratio is still computed with
    ``max(||theta_before||, eps)`` but marked ``near_zero_weight``. Callers
    must keep those rows out of the headline aggregate (see
    ``aggregate_ratios``).
    """
    if before.shape != after.shape:
        raise ValueError(f"Shape mismatch for {name}: {tuple(before.shape)} vs {tuple(after.shape)}")
    delta = after.detach() - before.detach()
    try:
        param_norm = tensor_norm(before)
        update_norm = tensor_norm(delta)
    except NonFiniteError:
        return RatioRecord(
            name=name,
            param_norm=float("nan"),
            update_norm=float("nan"),
            ratio=None,
            near_zero_weight=False,
            missing_grad=False,
            non_finite=True,
            excluded_from_aggregate=True,
            policy="exclude_non_finite",
        )
    near_zero = param_norm < near_zero_eps
    denom = max(param_norm, eps)
    ratio = update_norm / denom
    if not math.isfinite(ratio):
        return RatioRecord(
            name=name,
            param_norm=param_norm,
            update_norm=update_norm,
            ratio=None,
            near_zero_weight=near_zero,
            missing_grad=False,
            non_finite=True,
            excluded_from_aggregate=True,
            policy="exclude_non_finite",
        )
    return RatioRecord(
        name=name,
        param_norm=param_norm,
        update_norm=update_norm,
        ratio=ratio,
        near_zero_weight=near_zero,
        missing_grad=False,
        non_finite=False,
        excluded_from_aggregate=near_zero,
        policy="clamp_denom_and_flag" if near_zero else "standard",
    )


def missing_grad_record(name: str) -> RatioRecord:
    return RatioRecord(
        name=name,
        param_norm=0.0,
        update_norm=0.0,
        ratio=None,
        near_zero_weight=False,
        missing_grad=True,
        non_finite=False,
        excluded_from_aggregate=True,
        policy="exclude_missing_grad",
    )


def aggregate_ratios(records: Sequence[RatioRecord]) -> dict:
    kept = [r for r in records if not r.excluded_from_aggregate and r.ratio is not None]
    flagged = [r.as_dict() for r in records if r.excluded_from_aggregate]
    if not kept:
        return {
            "n_kept": 0,
            "mean_ratio": None,
            "max_ratio": None,
            "n_flagged": len(flagged),
            "flagged": flagged,
        }
    values = [float(r.ratio) for r in kept]
    return {
        "n_kept": len(kept),
        "mean_ratio": sum(values) / len(values),
        "max_ratio": max(values),
        "n_flagged": len(flagged),
        "flagged": flagged,
    }


def first_persistent_crossing(
    steps: Sequence[int],
    values: Sequence[float],
    tolerance: float,
    consecutive: int,
) -> dict:
    """First index where ``consecutive`` successive |values| stay <= tolerance.

    The search uses the provided series as-is. Callers that need to look past
    step 20 must pass a longer series. A result from a long search must be
    labelled separately from a 20-step plot.
    """
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    if len(steps) != len(values):
        raise ValueError("steps and values must have the same length")
    n = len(values)
    for i in range(0, n - consecutive + 1):
        window = [float(v) for v in values[i : i + consecutive]]
        if all(math.isfinite(v) and abs(v) <= tolerance for v in window):
            return {
                "met": True,
                "step": int(steps[i]),
                "index": i,
                "consecutive": consecutive,
                "tolerance": tolerance,
                "window_end_step": int(steps[i + consecutive - 1]),
                "window_values": list(window),
            }
    return {
        "met": False,
        "step": None,
        "index": None,
        "consecutive": consecutive,
        "tolerance": tolerance,
        "window_end_step": None,
        "window_values": [],
        "searched_through_step": int(steps[-1]) if steps else None,
    }


def describe_trend(values: Sequence[float]) -> str:
    """Classify a short series as decreasing, increasing, oscillating, or mixed."""
    if len(values) < 3:
        return "insufficient"
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    if neg == len(diffs):
        return "decreasing"
    if pos == len(diffs):
        return "increasing"
    # Peak then fall, or valley then rise, counts as mixed-with-turn.
    peak_at = max(range(len(values)), key=lambda i: values[i])
    trough_at = min(range(len(values)), key=lambda i: values[i])
    if 0 < peak_at < len(values) - 1 and peak_at > trough_at:
        return "rises_then_falls"
    if 0 < trough_at < len(values) - 1 and trough_at > peak_at:
        return "falls_then_rises"
    return "oscillating"


def assert_grads_finite(params: Iterable[torch.nn.Parameter]) -> None:
    for param in params:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise NonFiniteError("Non-finite gradient")


def assert_params_finite(params: Iterable[torch.nn.Parameter]) -> None:
    for param in params:
        if not torch.isfinite(param).all():
            raise NonFiniteError("Non-finite parameter")


def global_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    assert_grads_finite(params)
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().pow(2).sum().item())
    if total < 0.0:
        raise NonFiniteError("squared gradient sum is negative")
    return require_finite(math.sqrt(total), "global_grad_norm")
