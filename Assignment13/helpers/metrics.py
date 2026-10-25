"""Metric calculations, token accounting, and report validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import math


class MetricsError(ValueError):
    """Raised when a reported metric is internally inconsistent."""


@dataclass
class TokenAccounting:
    """Explicit token-counting convention used by every experiment.

    Training tokens are **non-padding target-token positions** processed during
    the training loop. Validation tokens are recorded separately and never
    counted toward the 50M training budget.

    Packed sequences have no padding, so each example of length ``T``
    contributes ``T`` target tokens. Gradient accumulation does not
    double-count: tokens are counted once per micro-batch forward.
    """

    input_tokens: int = 0
    target_tokens: int = 0
    padding_tokens: int = 0
    discarded_tokens: int = 0
    validation_tokens: int = 0

    def add_micro_batch(
        self,
        *,
        input_tokens: int,
        target_tokens: int,
        padding_tokens: int = 0,
    ) -> None:
        if min(input_tokens, target_tokens, padding_tokens) < 0:
            raise MetricsError("token counts must be non-negative")
        self.input_tokens += int(input_tokens)
        self.target_tokens += int(target_tokens)
        self.padding_tokens += int(padding_tokens)

    @property
    def training_target_tokens(self) -> int:
        return int(self.target_tokens)

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "target_tokens": self.target_tokens,
            "padding_tokens": self.padding_tokens,
            "discarded_tokens": self.discarded_tokens,
            "validation_tokens": self.validation_tokens,
            "training_target_tokens": self.training_target_tokens,
        }


@dataclass
class ExperimentMetrics:
    """Machine-readable metrics for one completed or interrupted run."""

    experiment: str
    variant: str
    status: str
    trainable_parameters: int
    micro_batch_size: int
    accumulation_steps: int
    effective_batch_size: int
    sequence_length: int
    training_target_tokens: int
    token_budget: int
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    tokens_per_second: float | None = None
    wall_clock_seconds: float | None = None
    time_per_optimizer_step: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    device: str = "cpu"
    precision: str = "fp32"
    gpu_name: str | None = None
    gpu_total_memory_bytes: int | None = None
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    reconstruction_error: float | None = None
    reconstruction_ok: bool | None = None
    reversible_method: str | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_REPORT_FIELDS = (
    "experiment",
    "variant",
    "status",
    "trainable_parameters",
    "micro_batch_size",
    "accumulation_steps",
    "effective_batch_size",
    "sequence_length",
    "training_target_tokens",
    "token_budget",
    "final_train_loss",
    "final_val_loss",
    "tokens_per_second",
    "wall_clock_seconds",
    "time_per_optimizer_step",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "device",
    "precision",
)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise MetricsError("mean() requires a non-empty sequence")
    return float(sum(values) / len(values))


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise MetricsError("division by zero in metric calculation")
    return float(numerator) / float(denominator)


def tokens_per_second(target_tokens: int, wall_clock_seconds: float) -> float:
    """Throughput numerator is non-padding training target tokens."""
    if target_tokens < 0:
        raise MetricsError(f"target_tokens must be >= 0, got {target_tokens}")
    if wall_clock_seconds <= 0:
        raise MetricsError(
            f"wall_clock_seconds must be > 0, got {wall_clock_seconds}"
        )
    return safe_div(target_tokens, wall_clock_seconds)


def bytes_to_mib(num_bytes: int | None) -> float | None:
    if num_bytes is None:
        return None
    return float(num_bytes) / (1024.0 ** 2)


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fail when a serialized experiment report is missing required fields
    or confuses micro-batch, effective batch, and token budget."""
    missing = [name for name in REQUIRED_REPORT_FIELDS if name not in report]
    if missing:
        raise MetricsError(f"report is missing required fields: {missing}")
    status = str(report["status"])
    if status not in {"completed", "smoke", "failed", "interrupted", "pending", "oom"}:
        raise MetricsError(f"unknown status {status!r}")
    if status == "pending":
        return dict(report)
    micro = int(report["micro_batch_size"])
    accum = int(report["accumulation_steps"])
    effective = int(report["effective_batch_size"])
    if micro < 1:
        raise MetricsError(f"micro_batch_size must be >= 1, got {micro}")
    if accum < 1:
        raise MetricsError(f"accumulation_steps must be >= 1, got {accum}")
    if effective != micro * accum:
        raise MetricsError(
            "effective_batch_size must equal micro_batch_size * accumulation_steps "
            f"({micro} * {accum} != {effective}). Gradient accumulation is not a "
            "larger per-device micro-batch."
        )
    seq = int(report["sequence_length"])
    if seq < 2:
        raise MetricsError(f"sequence_length must be >= 2, got {seq}")
    budget = int(report["token_budget"])
    trained = int(report["training_target_tokens"])
    if budget <= 0:
        raise MetricsError(f"token_budget must be positive, got {budget}")
    if trained < 0:
        raise MetricsError(f"training_target_tokens must be >= 0, got {trained}")
    if report.get("validation_tokens", 0) and trained == report.get("validation_tokens"):
        # Not automatically illegal, but a common accounting bug when the
        # same counter is reused. Flag only when extra says they were merged.
        extra = report.get("extra") or {}
        if extra.get("counted_validation_as_training"):
            raise MetricsError("validation tokens must not count as training tokens")
    tps = report.get("tokens_per_second")
    extra = report.get("extra") or {}
    measured_s = extra.get("measured_training_seconds")
    measured_tok = extra.get("measured_training_tokens", trained)
    if tps is not None and measured_s not in (None, 0) and measured_tok:
        expected = tokens_per_second(int(measured_tok), float(measured_s))
        if abs(float(tps) - expected) > max(1e-6, 0.05 * abs(expected)):
            raise MetricsError(
                f"tokens_per_second {tps} disagrees with measured target tokens / "
                f"measured training seconds ({expected})"
            )
    return dict(report)


def finite_or_none(values: Iterable[float]) -> list[float]:
    finite = [float(v) for v in values if is_finite_number(v)]
    return finite
