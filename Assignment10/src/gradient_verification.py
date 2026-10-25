"""Independent gradient verification: chain rule, autograd, and finite differences.

Computation
-----------
    x = 2,  w1 = 3,  w2 = 4,  target = 20
    h = w1 * x          →  6
    y = w2 * h          →  24
    L = (y - target)^2  →  16

Chain rule
----------
    dL/dy  = 2 (y - target) = 8
    dL/dw2 = dL/dy * h      = 48
    dL/dh  = dL/dy * w2     = 32
    dL/dw1 = dL/dh * x      = 64

Central finite difference for w1
--------------------------------
    (L(w1+ε) - L(w1-ε)) / (2ε)

Truncation error of the central difference is O(ε²). Floating-point cancellation
in the numerator is O(ε_machine / ε). Those errors trade off: too large an ε
truncates the Taylor series, too small an ε subtracts nearly equal losses.
float64 is used to push ε_machine down to ~2e-16 so an ε near 1e-6 is in the
sweet spot (roughly ε_machine^{1/3} for central differences).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.nn import functional as F

from models import TinyMLP


ANALYTICAL = {
    "h": 6.0,
    "y": 24.0,
    "loss": 16.0,
    "dL_dy": 8.0,
    "dL_dw2": 48.0,
    "dL_dh": 32.0,
    "dL_dw1": 64.0,
}

DEFAULT_EPSILONS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)


def _forward_values(x, w1, w2, target):
    h = w1 * x
    y = w2 * h
    loss = (y - target) ** 2
    return h, y, loss


def analytical_gradients(x=2.0, w1=3.0, w2=4.0, target=20.0) -> dict[str, float]:
    h, y, loss = _forward_values(x, w1, w2, target)
    dL_dy = 2.0 * (y - target)
    dL_dw2 = dL_dy * h
    dL_dh = dL_dy * w2
    dL_dw1 = dL_dh * x
    return {
        "h": float(h),
        "y": float(y),
        "loss": float(loss),
        "dL_dy": float(dL_dy),
        "dL_dw2": float(dL_dw2),
        "dL_dh": float(dL_dh),
        "dL_w1": float(dL_dw1),
        "dL_dw1": float(dL_dw1),
    }


def autograd_gradients(
    x=2.0,
    w1=3.0,
    w2=4.0,
    target=20.0,
    dtype=torch.float64,
) -> dict[str, float]:
    x_t = torch.tensor(x, dtype=dtype)
    w1_t = torch.tensor(w1, dtype=dtype, requires_grad=True)
    w2_t = torch.tensor(w2, dtype=dtype, requires_grad=True)
    target_t = torch.tensor(target, dtype=dtype)
    h, y, loss = _forward_values(x_t, w1_t, w2_t, target_t)
    loss.backward()
    return {
        "h": float(h.detach()),
        "y": float(y.detach()),
        "loss": float(loss.detach()),
        "dL_dw1": float(w1_t.grad.detach()),
        "dL_dw2": float(w2_t.grad.detach()),
    }


def loss_at_w1(w1: float, x=2.0, w2=4.0, target=20.0, dtype=torch.float64) -> float:
    w1_t = torch.tensor(w1, dtype=dtype)
    x_t = torch.tensor(x, dtype=dtype)
    w2_t = torch.tensor(w2, dtype=dtype)
    target_t = torch.tensor(target, dtype=dtype)
    _, _, loss = _forward_values(x_t, w1_t, w2_t, target_t)
    return float(loss)


def central_finite_difference(
    epsilon: float,
    w1: float = 3.0,
    x: float = 2.0,
    w2: float = 4.0,
    target: float = 20.0,
    dtype=torch.float64,
) -> float:
    if epsilon == 0:
        raise ValueError("epsilon must be non-zero")
    plus = loss_at_w1(w1 + epsilon, x=x, w2=w2, target=target, dtype=dtype)
    minus = loss_at_w1(w1 - epsilon, x=x, w2=w2, target=target, dtype=dtype)
    return (plus - minus) / (2.0 * epsilon)


def sweep_finite_differences(
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    *,
    analytical_dL_dw1: float = 64.0,
    dtype=torch.float64,
) -> list[dict]:
    rows = []
    for eps in epsilons:
        numerical = central_finite_difference(eps, dtype=dtype)
        abs_err = abs(numerical - analytical_dL_dw1)
        rel_err = abs_err / abs(analytical_dL_dw1) if analytical_dL_dw1 != 0 else float("inf")
        rows.append(
            {
                "epsilon": eps,
                "numerical_dL_dw1": numerical,
                "absolute_error": abs_err,
                "relative_error": rel_err,
            }
        )
    return rows


def cubic_truncation_demo(
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    *,
    w: float = 3.0,
    dtype=torch.float64,
) -> dict:
    """Auxiliary cubic L = w³ whose central-difference error is ε² plus cancellation.

    This is *not* the assignment's (y-target)² graph. It exists only to make the
    truncation-vs-cancellation trade-off numerically visible, because the
    assignment loss is quadratic and therefore truncation-free.
    """
    analytical = 3.0 * w * w  # d/dw w^3 = 3 w^2

    def loss(val: float) -> float:
        return float(torch.tensor(val, dtype=dtype) ** 3)

    rows = []
    for eps in epsilons:
        numerical = (loss(w + eps) - loss(w - eps)) / (2.0 * eps)
        truncation_term = eps * eps  # exact for this cubic in infinite precision
        rows.append(
            {
                "epsilon": eps,
                "numerical": numerical,
                "analytical": analytical,
                "absolute_error": abs(numerical - analytical),
                "relative_error": abs(numerical - analytical) / analytical,
                "infinite_precision_truncation": truncation_term,
            }
        )
    return {
        "function": "L(w) = w^3 at w=3, dL/dw = 27",
        "purpose": "show O(ε²) truncation vs cancellation; not the assignment graph",
        "rows": rows,
    }


def nudge_one_weight_in_real_loop(
    *,
    epsilon: float = 1e-4,
    dtype=torch.float64,
) -> dict:
    """Nudge one scalar weight in TinyMLP and compare ΔL/(2ε) to backward().

    This is the assignment graph's cousin on a real training loop: pick
    ``fc1.weight[0, 0]``, run MSE, read autograd, then perturb the same entry.
    """
    torch.manual_seed(0)
    model = TinyMLP(in_features=4, hidden=8, out_features=1).double()
    x = torch.randn(6, 4, dtype=dtype)
    y = torch.randn(6, 1, dtype=dtype)
    param = model.fc1.weight
    model.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    autograd_g = float(param.grad[0, 0].detach())
    orig = float(param.detach()[0, 0])

    with torch.no_grad():
        param[0, 0] = orig + epsilon
    loss_plus = float(F.mse_loss(model(x), y).detach())
    with torch.no_grad():
        param[0, 0] = orig - epsilon
    loss_minus = float(F.mse_loss(model(x), y).detach())
    with torch.no_grad():
        param[0, 0] = orig
    numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
    abs_err = abs(numerical - autograd_g)
    rel_err = abs_err / max(abs(autograd_g), 1e-12)
    return {
        "parameter": "fc1.weight[0, 0]",
        "epsilon": epsilon,
        "dtype": str(dtype),
        "autograd": autograd_g,
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "numerical": numerical,
        "absolute_error": abs_err,
        "relative_error": rel_err,
        "agrees_to_several_decimals": rel_err < 1e-6,
    }


def verify(
    *,
    abs_tol_autograd: float = 1e-12,
    rel_tol_numerical: float = 1e-6,
    preferred_epsilon: float = 1e-6,
    epsilons: Iterable[float] = DEFAULT_EPSILONS,
    dtype=torch.float64,
) -> dict:
    analytic = analytical_gradients()
    auto = autograd_gradients(dtype=dtype)
    fd_rows = sweep_finite_differences(tuple(epsilons), analytical_dL_dw1=analytic["dL_dw1"], dtype=dtype)

    autograd_abs = {
        "dL_dw1": abs(auto["dL_dw1"] - analytic["dL_dw1"]),
        "dL_dw2": abs(auto["dL_dw2"] - analytic["dL_dw2"]),
        "loss": abs(auto["loss"] - analytic["loss"]),
    }
    autograd_rel = {
        k: (autograd_abs[k] / abs(analytic[k]) if analytic[k] != 0 else autograd_abs[k])
        for k in autograd_abs
    }
    preferred = next(r for r in fd_rows if abs(r["epsilon"] - preferred_epsilon) < 1e-30)
    best = min(fd_rows, key=lambda r: r["relative_error"])

    autograd_pass = all(v <= abs_tol_autograd for v in autograd_abs.values())
    numerical_pass = preferred["relative_error"] <= rel_tol_numerical
    intermediates_pass = all(
        abs(analytic[k] - ANALYTICAL[k]) < 1e-12 for k in ("h", "y", "loss", "dL_dy", "dL_dw2", "dL_dh", "dL_dw1")
    )
    passed = autograd_pass and numerical_pass and intermediates_pass
    cubic = cubic_truncation_demo(tuple(epsilons), dtype=dtype)
    nudge = nudge_one_weight_in_real_loop(epsilon=1e-4, dtype=torch.float64)
    passed = passed and nudge["agrees_to_several_decimals"]

    return {
        "analytical": analytic,
        "autograd": auto,
        "autograd_absolute_error": autograd_abs,
        "autograd_relative_error": autograd_rel,
        "finite_differences": fd_rows,
        "preferred_epsilon": preferred_epsilon,
        "preferred_row": preferred,
        "best_epsilon_row": best,
        "tolerances": {
            "abs_tol_autograd": abs_tol_autograd,
            "rel_tol_numerical": rel_tol_numerical,
            "dtype": str(dtype),
        },
        "tradeoff": (
            "In general a central difference has O(ε²) truncation error and "
            "roughly O(ε_machine/ε) cancellation error. This particular L(w1) = "
            "(8 w1 - 20)^2 is a quadratic, so the O(ε²) term is identically zero "
            "and moderate ε (1e-2 … 1e-6) recover 64 to nearly machine precision "
            "in float64. At ε ≤ 1e-12 the two nearby losses collide and cancellation "
            "dominates. float64 (ε_machine ≈ 2e-16) is still required: the same "
            "sweep in float32 would start losing digits much earlier."
        ),
        "passed": passed,
        "autograd_pass": autograd_pass,
        "numerical_pass": numerical_pass,
        "intermediates_pass": intermediates_pass,
        "cubic_truncation_demo": cubic,
        "real_loop_nudge": nudge,
    }
