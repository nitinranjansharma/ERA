"""Scalar Adam implemented by hand, with a separate no-bias-correction path.

Step convention
---------------
The first applied gradient is step ``t = 1``. Moments start at zero. Python
``float`` (IEEE-754 binary64) is used for the manual algebra so the comparison
with a ``float64`` ``torch.optim.Adam`` scalar measures the update equations,
not float32 rounding.

PyTorch epsilon placement
-------------------------
Textbook (this module's default corrected update):

    delta = -lr * m_hat / (sqrt(v_hat) + eps)

with ``m_hat = m / (1 - beta1**t)`` and ``v_hat = v / (1 - beta2**t)``.

PyTorch's default single-tensor Adam computes the algebraically equivalent

    delta = -(lr / (1 - beta1**t)) * m / (sqrt(v) / sqrt(1 - beta2**t) + eps)

The two agree because ``sqrt(v / (1 - beta2**t)) = sqrt(v) / sqrt(1 - beta2**t)``
and epsilon is added *after* that scaling. Adding epsilon before the bias
correction of ``v`` is a different algorithm.

The no-bias-correction implementation never materialises ``m_hat`` / ``v_hat``.
It applies ``delta = -lr * m / (sqrt(v) + eps)`` to the raw moments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch


DEFAULT_GRADIENTS: tuple[float, ...] = (0.50, 0.40, 0.60, 0.45, 0.55)


@dataclass
class AdamStepRecord:
    """One optimizer step, including every intermediate the assignment asks for."""

    t: int
    g: float
    m: float
    v: float
    m_hat: float | None
    v_hat: float | None
    delta_w: float
    w: float
    bias_correction: bool
    denom: float
    lr: float

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "t": self.t,
            "g": self.g,
            "m": self.m,
            "v": self.v,
            "m_hat": self.m_hat,
            "v_hat": self.v_hat,
            "delta_w": self.delta_w,
            "w": self.w,
            "bias_correction": self.bias_correction,
            "denom": self.denom,
            "lr": self.lr,
        }


class ScalarAdam:
    """Scalar Adam with optional bias correction.

    Parameters
    ----------
    bias_correction:
        If True, use the conventional hat-moments. If False, update from the
        raw moments and leave ``m_hat`` / ``v_hat`` as ``None``.
    epsilon_placement:
        ``"after_vhat"`` is the textbook/PyTorch-equivalent form.
        ``"inside_vhat"`` adds epsilon before dividing ``v`` by the bias term
        and is recorded only as a contrast; it is not used as the default.
    """

    def __init__(
        self,
        w0: float = 1.0,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        bias_correction: bool = True,
        epsilon_placement: str = "after_vhat",
    ) -> None:
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if epsilon_placement not in {"after_vhat", "pytorch_form", "inside_vhat"}:
            raise ValueError(f"Unknown epsilon_placement {epsilon_placement!r}")
        if epsilon_placement != "after_vhat" and not bias_correction:
            raise ValueError("epsilon_placement variants apply only with bias correction")

        self.w0 = float(w0)
        self.w = float(w0)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.bias_correction = bool(bias_correction)
        self.epsilon_placement = epsilon_placement
        self.m = 0.0
        self.v = 0.0
        self.t = 0
        self.history: list[AdamStepRecord] = []

    def step(self, gradient: float) -> AdamStepRecord:
        g = float(gradient)
        if not math.isfinite(g):
            raise ValueError(f"Non-finite gradient at t={self.t + 1}: {g}")

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)

        m_hat: float | None
        v_hat: float | None
        if self.bias_correction:
            bc1 = 1.0 - self.beta1**self.t
            bc2 = 1.0 - self.beta2**self.t
            m_hat = self.m / bc1
            v_hat = self.v / bc2
            if self.epsilon_placement == "after_vhat":
                denom = math.sqrt(v_hat) + self.eps
                delta = -self.lr * m_hat / denom
            elif self.epsilon_placement == "pytorch_form":
                denom = math.sqrt(self.v) / math.sqrt(bc2) + self.eps
                delta = -(self.lr / bc1) * self.m / denom
            else:
                # Contrast: epsilon rides inside the bias-corrected second moment.
                denom = math.sqrt(self.v) + self.eps
                denom = denom / math.sqrt(bc2)
                delta = -self.lr * m_hat / denom
        else:
            m_hat = None
            v_hat = None
            denom = math.sqrt(self.v) + self.eps
            delta = -self.lr * self.m / denom

        if not math.isfinite(delta):
            raise ValueError(f"Non-finite Adam update at t={self.t}: {delta}")

        self.w = self.w + delta
        record = AdamStepRecord(
            t=self.t,
            g=g,
            m=self.m,
            v=self.v,
            m_hat=m_hat,
            v_hat=v_hat,
            delta_w=delta,
            w=self.w,
            bias_correction=self.bias_correction,
            denom=denom,
            lr=self.lr,
        )
        self.history.append(record)
        return record

    def run(self, gradients: Sequence[float]) -> list[AdamStepRecord]:
        return [self.step(g) for g in gradients]


class ScalarAdamNoBiasCorrection(ScalarAdam):
    """Explicit uncorrected Adam. Does not compute or store hat-moments."""

    def __init__(self, **kwargs) -> None:
        kwargs["bias_correction"] = False
        kwargs.pop("epsilon_placement", None)
        super().__init__(epsilon_placement="after_vhat", **kwargs)


def run_manual_adam(
    gradients: Sequence[float] = DEFAULT_GRADIENTS,
    w0: float = 1.0,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
    epsilon_placement: str = "after_vhat",
) -> list[AdamStepRecord]:
    opt = ScalarAdam(
        w0=w0,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        bias_correction=bias_correction,
        epsilon_placement=epsilon_placement,
    )
    return opt.run(gradients)


def run_torch_adam_scalar(
    gradients: Sequence[float] = DEFAULT_GRADIENTS,
    w0: float = 1.0,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> list[AdamStepRecord]:
    """Reference trajectory from ``torch.optim.Adam`` on one float64 parameter."""

    param = torch.nn.Parameter(torch.tensor(float(w0), dtype=torch.float64))
    optimizer_kwargs = {
        "lr": float(lr),
        "betas": (float(beta1), float(beta2)),
        "eps": float(eps),
        "weight_decay": 0.0,
        "amsgrad": False,
    }
    # Disable fused/foreach kernels so the comparison is against the documented
    # single-tensor update, not an implementation shortcut.
    for extra in ("foreach", "fused", "capturable", "differentiable"):
        optimizer_kwargs[extra] = False
    try:
        opt = torch.optim.Adam([param], **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("foreach", None)
        optimizer_kwargs.pop("fused", None)
        optimizer_kwargs.pop("capturable", None)
        optimizer_kwargs.pop("differentiable", None)
        opt = torch.optim.Adam([param], **optimizer_kwargs)

    records: list[AdamStepRecord] = []
    w_prev = float(param.detach())
    for g in gradients:
        opt.zero_grad(set_to_none=True)
        param.grad = torch.tensor(float(g), dtype=torch.float64)
        opt.step()
        state = opt.state[param]
        step_raw = state["step"]
        t = int(step_raw.item()) if torch.is_tensor(step_raw) else int(step_raw)
        m = float(state["exp_avg"].detach())
        v = float(state["exp_avg_sq"].detach())
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        w_now = float(param.detach())
        delta = w_now - w_prev
        bc2 = 1.0 - beta2**t
        denom = math.sqrt(v) / math.sqrt(bc2) + eps
        records.append(
            AdamStepRecord(
                t=t,
                g=float(g),
                m=m,
                v=v,
                m_hat=m_hat,
                v_hat=v_hat,
                delta_w=delta,
                w=w_now,
                bias_correction=True,
                denom=denom,
                lr=float(lr),
            )
        )
        w_prev = w_now
    return records


def _field_error(a: float | None, b: float | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(float(a) - float(b))


def compare_records(
    manual: Sequence[AdamStepRecord],
    reference: Sequence[AdamStepRecord],
    fields: Sequence[str] = ("g", "m", "v", "m_hat", "v_hat", "delta_w", "w"),
) -> dict[str, float | int | dict[str, float]]:
    if len(manual) != len(reference):
        raise ValueError("Record lengths differ")
    worst = 0.0
    per_field = {name: 0.0 for name in fields}
    per_step = []
    for left, right in zip(manual, reference):
        step_err = 0.0
        for name in fields:
            err = _field_error(getattr(left, name), getattr(right, name))
            per_field[name] = max(per_field[name], err)
            step_err = max(step_err, err)
        worst = max(worst, step_err)
        per_step.append({"t": left.t, "max_abs_error": step_err})
    return {
        "n_steps": len(manual),
        "worst_abs_error": worst,
        "per_field_worst": per_field,
        "per_step": per_step,
    }


def verify_against_pytorch(
    gradients: Sequence[float] = DEFAULT_GRADIENTS,
    w0: float = 1.0,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tolerance: float = 1e-12,
) -> dict:
    textbook = run_manual_adam(
        gradients, w0, lr, beta1, beta2, eps, True, "after_vhat"
    )
    pytorch_form = run_manual_adam(
        gradients, w0, lr, beta1, beta2, eps, True, "pytorch_form"
    )
    torch_ref = run_torch_adam_scalar(gradients, w0, lr, beta1, beta2, eps)
    vs_torch = compare_records(textbook, torch_ref)
    vs_form = compare_records(textbook, pytorch_form)
    inside = run_manual_adam(gradients, w0, lr, beta1, beta2, eps, True, "inside_vhat")
    vs_inside = compare_records(textbook, inside)
    uncorrected = ScalarAdamNoBiasCorrection(w0=w0, lr=lr, beta1=beta1, beta2=beta2, eps=eps).run(
        gradients
    )
    return {
        "step_convention": (
            "t starts at 1 for the first gradient. Moments are initialized to 0. "
            "The parameter update is w_t = w_{t-1} + delta_w_t with "
            "delta_w_t = -lr * m_hat / (sqrt(v_hat) + eps) when bias correction is on."
        ),
        "epsilon_placement": (
            "Default matches PyTorch: epsilon is added after the bias-corrected "
            "second-moment square root, not inside it."
        ),
        "dtype": "float64",
        "hyperparameters": {
            "w0": w0,
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "gradients": list(gradients),
        },
        "manual_corrected": [r.as_dict() for r in textbook],
        "manual_pytorch_form": [r.as_dict() for r in pytorch_form],
        "manual_uncorrected": [r.as_dict() for r in uncorrected],
        "torch_adam": [r.as_dict() for r in torch_ref],
        "manual_vs_torch": vs_torch,
        "textbook_vs_pytorch_form": vs_form,
        "textbook_vs_inside_eps": vs_inside,
        "tolerance": tolerance,
        "agrees_with_torch": vs_torch["worst_abs_error"] <= tolerance,
        "textbook_matches_pytorch_form": vs_form["worst_abs_error"] <= tolerance,
        "inside_eps_differs": vs_inside["worst_abs_error"] > tolerance,
    }


def closed_form_uncorrected_ratio(t: int, beta1: float, beta2: float) -> float:
    """|delta_off| / |delta_on| for a constant gradient, neglecting epsilon.

    For constant g the bias-corrected update is ``-lr * sign(g)`` (eps ignored)
    while the uncorrected update scales by ``(1 - beta1**t) / sqrt(1 - beta2**t)``.
    """
    if t < 1:
        raise ValueError("t is 1-indexed")
    return (1.0 - beta1**t) / math.sqrt(1.0 - beta2**t)


def bias_correction_factor_series(
    n_steps: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> list[dict[str, float | int]]:
    rows = []
    for t in range(1, n_steps + 1):
        ratio = closed_form_uncorrected_ratio(t, beta1, beta2)
        rows.append({"t": t, "uncorrected_over_corrected": ratio, "rel_to_one": abs(ratio - 1.0)})
    return rows
