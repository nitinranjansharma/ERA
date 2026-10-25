"""Numerical-safety checks. Invalid measurements raise; they are never coerced into plausible values."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


class TrainingSafetyError(ValueError):
    """Raised when a training invariant is violated."""


class PrecisionError(ValueError):
    """Raised for unsupported numeric formats or devices."""


class MFUError(ValueError):
    """Raised when MFU inputs use inconsistent units or impossible values."""


def assert_loss_ok(loss: torch.Tensor, *, name: str = "loss") -> None:
    if not torch.is_tensor(loss):
        raise TrainingSafetyError(f"{name} must be a tensor, got {type(loss)}")
    if loss.ndim != 0:
        raise TrainingSafetyError(
            f"{name} must be a scalar (0-d) tensor, got shape {tuple(loss.shape)}"
        )
    if not torch.isfinite(loss).item():
        raise TrainingSafetyError(f"{name} is not finite: {loss.detach().cpu().item()!r}")


def assert_grads_finite(parameters: Iterable[nn.Parameter], *, where: str = "backward") -> None:
    bad = []
    for name, p in _named(parameters):
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
    if bad:
        raise TrainingSafetyError(f"Non-finite gradients after {where}: {bad}")


def assert_params_finite(parameters: Iterable[nn.Parameter], *, where: str = "optimizer.step") -> None:
    bad = []
    for name, p in _named(parameters):
        if not torch.isfinite(p).all():
            bad.append(name)
    if bad:
        raise TrainingSafetyError(f"Non-finite parameters after {where}: {bad}")


def assert_expected_grads(
    module: nn.Module,
    *,
    allow_missing: Optional[set[str]] = None,
) -> None:
    allow_missing = allow_missing or set()
    missing = []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None and name not in allow_missing:
            missing.append(name)
    if missing:
        raise TrainingSafetyError(
            "Trainable parameters missing gradients (disconnected from the loss graph?): "
            + ", ".join(missing)
        )


def assert_parameters_changed(
    before: dict[str, torch.Tensor],
    module: nn.Module,
    *,
    atol: float = 0.0,
) -> None:
    unchanged = []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        prev = before[name]
        if torch.allclose(prev, p.detach(), atol=atol, rtol=0.0):
            unchanged.append(name)
    if unchanged:
        raise TrainingSafetyError(
            "Optimizer step did not change parameters: " + ", ".join(unchanged)
        )


def snapshot_params(module: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def assert_shapes_compatible(logits: torch.Tensor, targets: torch.Tensor, *, task: str) -> None:
    if task == "regression":
        if logits.shape != targets.shape:
            raise TrainingSafetyError(
                f"Regression logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"
            )
    elif task == "token_lm":
        if logits.ndim != 3:
            raise TrainingSafetyError(f"LM logits must be (B, T, V), got {tuple(logits.shape)}")
        if targets.shape != logits.shape[:2]:
            raise TrainingSafetyError(
                f"LM targets must be (B, T) matching logits, got {tuple(targets.shape)} vs {tuple(logits.shape)}"
            )
    else:
        raise TrainingSafetyError(f"Unknown task {task!r}")


def assert_target_dtype(targets: torch.Tensor, *, task: str) -> None:
    if task == "regression":
        if targets.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TrainingSafetyError(
                f"Regression targets must be floating-point, got dtype={targets.dtype}"
            )
    elif task == "token_lm":
        if targets.dtype not in (torch.int32, torch.int64, torch.long):
            raise TrainingSafetyError(
                f"Classification/LM targets must be integer class indices, got dtype={targets.dtype}"
            )


def assert_reduction(reduction: str, *, allowed: tuple[str, ...] = ("mean", "sum", "none")) -> None:
    if reduction not in allowed:
        raise TrainingSafetyError(
            f"Unsupported loss reduction {reduction!r}; expected one of {allowed}"
        )


def warn_or_raise_zero_norm(norm: float, *, strict: bool = False) -> None:
    if norm == 0.0:
        msg = (
            "Global gradient norm is exactly 0. This may indicate a disconnected graph, "
            "a zero learning signal, or all gradients being None."
        )
        if strict:
            raise TrainingSafetyError(msg)


def assert_supported_precision(precision: str, device: torch.device) -> None:
    precision = precision.lower()
    known = {"fp32", "float32", "fp64", "float64", "bf16", "bfloat16", "fp16", "float16", "fp8", "fp8_e4m3fn"}
    if precision not in known:
        raise PrecisionError(f"Unsupported precision {precision!r}")
    if precision in {"fp16", "float16"} and device.type == "cpu":
        # float16 matmul on CPU is often emulated / incomplete; refuse silent fallbacks.
        raise PrecisionError(
            "float16 training is not supported on CPU in this project; use fp32 or bfloat16"
        )
    if precision in {"fp8", "fp8_e4m3fn"} and device.type == "cpu":
        raise PrecisionError(
            "FP8 E4M3 compute is not supported as a training dtype on this CPU device; "
            "the encoder/decoder is available for analysis only"
        )


def _named(parameters: Iterable[nn.Parameter]) -> list[tuple[str, nn.Parameter]]:
    out = []
    for i, p in enumerate(parameters):
        name = getattr(p, "_param_name", None) or f"param[{i}]"
        out.append((name, p))
    return out
