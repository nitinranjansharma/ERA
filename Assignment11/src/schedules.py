"""Learning-rate schedules as pure functions of a 1-indexed optimizer step."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

ScheduleName = Literal["constant", "warmup_stable", "cosine", "wsd"]


def _require_positive_int(value: int, name: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise ValueError(f"{name} must be >= 0")
    return ivalue


def linear_warmup_lr(t: int, warmup_steps: int, peak_lr: float, start_lr: float = 0.0) -> float:
    """Linear ramp. At t=warmup_steps the rate equals peak_lr. t is 1-indexed."""
    if t < 1:
        return float(start_lr)
    if warmup_steps <= 0:
        return float(peak_lr)
    if t >= warmup_steps:
        return float(peak_lr)
    frac = t / warmup_steps
    return float(start_lr + frac * (peak_lr - start_lr))


def cosine_decay_lr(
    t: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup, then cosine from peak_lr down to min_lr at total_steps."""
    total_steps = max(1, int(total_steps))
    warmup_steps = min(_require_positive_int(warmup_steps, "warmup_steps"), total_steps)
    if t < 1:
        return 0.0
    if t <= warmup_steps:
        return linear_warmup_lr(t, warmup_steps, peak_lr)
    if t >= total_steps:
        return float(min_lr)
    span = max(1, total_steps - warmup_steps)
    progress = (t - warmup_steps) / span
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (peak_lr - min_lr) * cosine)


def wsd_lr(
    t: int,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Warmup, stable peak, then linear decay over the final decay_steps.

    Decay occupies the last ``decay_steps`` indices, inclusive of ``total_steps``.
    The first decay step is ``total_steps - decay_steps + 1``.
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = min(_require_positive_int(warmup_steps, "warmup_steps"), total_steps)
    decay_steps = min(_require_positive_int(decay_steps, "decay_steps"), max(0, total_steps - warmup_steps))
    if t < 1:
        return 0.0
    if t <= warmup_steps:
        return linear_warmup_lr(t, warmup_steps, peak_lr)
    decay_start = total_steps - decay_steps + 1 if decay_steps > 0 else total_steps + 1
    if t < decay_start:
        return float(peak_lr)
    if decay_steps <= 0:
        return float(peak_lr)
    progress = (t - decay_start + 1) / decay_steps
    progress = min(1.0, max(0.0, progress))
    return float(peak_lr + (min_lr - peak_lr) * progress)


@dataclass(frozen=True)
class ScheduleSpec:
    name: ScheduleName
    total_steps: int
    warmup_steps: int
    decay_steps: int
    peak_lr: float
    min_lr: float

    def lr(self, t: int) -> float:
        if self.name == "constant":
            return float(self.peak_lr)
        if self.name == "warmup_stable":
            return linear_warmup_lr(t, self.warmup_steps, self.peak_lr) if t <= self.warmup_steps else float(self.peak_lr)
        if self.name == "cosine":
            return cosine_decay_lr(t, self.total_steps, self.warmup_steps, self.peak_lr, self.min_lr)
        if self.name == "wsd":
            return wsd_lr(t, self.total_steps, self.warmup_steps, self.decay_steps, self.peak_lr, self.min_lr)
        raise ValueError(f"Unknown schedule {self.name}")

    def lr_peak_step(self) -> int:
        """First 1-indexed step at which the schedule attains the peak rate."""
        if self.name == "constant":
            return 1
        return max(1, self.warmup_steps)

    def mean_lr(self, until: int | None = None) -> float:
        last = int(until or self.total_steps)
        if last < 1:
            return 0.0
        return sum(self.lr(t) for t in range(1, last + 1)) / last

    def series(self, until: int | None = None) -> list[dict[str, float | int | str]]:
        last = int(until or self.total_steps)
        return [{"t": t, "lr": self.lr(t), "name": self.name} for t in range(1, last + 1)]


def make_schedule(
    name: ScheduleName,
    total_steps: int,
    peak_lr: float,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.2,
    min_lr_fraction: float = 0.1,
    warmup_steps: int | None = None,
    decay_steps: int | None = None,
    min_lr: float | None = None,
) -> ScheduleSpec:
    total_steps = max(1, int(total_steps))
    if warmup_steps is None:
        warmup_steps = int(round(float(warmup_fraction) * total_steps))
    if decay_steps is None:
        decay_steps = int(round(float(decay_fraction) * total_steps))
    if min_lr is None:
        min_lr = float(min_lr_fraction) * float(peak_lr)
    warmup_steps = max(0, min(int(warmup_steps), total_steps))
    decay_steps = max(0, min(int(decay_steps), total_steps - warmup_steps))
    return ScheduleSpec(
        name=name,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        peak_lr=float(peak_lr),
        min_lr=float(min_lr),
    )


def apply_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def as_callable(spec: ScheduleSpec) -> Callable[[int], float]:
    return spec.lr
