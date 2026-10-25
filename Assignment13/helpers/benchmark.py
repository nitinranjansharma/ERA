"""Warm-up, timing, peak-memory measurement, and maximum-batch search.

Throughput numerator
--------------------
Non-padding training target tokens processed in the *measured* window,
divided by CUDA-synchronized wall time of that window. Warm-up steps,
dataset download, model construction, and evaluation are excluded.

Micro-batch vs effective batch
------------------------------
``micro_batch_size`` is the per-forward sequence count. ``effective_batch_size``
is ``micro_batch_size * accumulation_steps``. A larger accumulated batch is
not reported as a larger per-device micro-batch.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from helpers.config import ExperimentConfig, assignment_config, get_runtime_info, set_seed, tiny_config
from helpers.metrics import tokens_per_second


class BenchmarkError(RuntimeError):
    """Timing or batch-search failure."""


@dataclass
class BenchmarkResult:
    warmup_steps: int
    measured_steps: int
    measured_seconds: float
    target_tokens: int
    tokens_per_second: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    device: str
    micro_batch_size: int
    effective_batch_size: int
    sequence_length: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchSearchResult:
    best_micro_batch: int | None
    tried: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    notes: str = ""
    headroom_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda malloc" in message


def _cuda_device_or_none(device: torch.device | str | None = None) -> torch.device | None:
    """Return a CUDA device only when stats APIs will accept it.

    Colab has CUDA, but Unit 7 / CPU tests pass ``device='cpu'``. Passing that
    object to ``torch.cuda.reset_peak_memory_stats`` raises
    ``ValueError: Expected a cuda device, but got: cpu``.
    """
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.device("cuda")
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type != "cuda":
        return None
    return resolved


def reset_peak_memory(device: torch.device | str | None = None) -> None:
    resolved = _cuda_device_or_none(device)
    if resolved is None:
        return
    torch.cuda.reset_peak_memory_stats(resolved)
    torch.cuda.empty_cache()


def read_peak_memory(device: torch.device | str | None = None) -> tuple[int | None, int | None]:
    resolved = _cuda_device_or_none(device)
    if resolved is None:
        return None, None
    return int(torch.cuda.max_memory_allocated(resolved)), int(
        torch.cuda.max_memory_reserved(resolved)
    )


def synchronize(device: torch.device | str | None = None) -> None:
    resolved = _cuda_device_or_none(device)
    if resolved is None:
        return
    torch.cuda.synchronize(resolved)


def measure_training_step(
    step_fn: Callable[[], int],
    *,
    warmup_steps: int = 2,
    measured_steps: int = 5,
    device: torch.device | str = "cpu",
    micro_batch_size: int = 1,
    accumulation_steps: int = 1,
    sequence_length: int = 8,
) -> BenchmarkResult:
    """Time ``step_fn`` after warm-up. ``step_fn`` returns target tokens processed."""
    if warmup_steps < 0 or measured_steps < 1:
        raise BenchmarkError("warmup_steps >= 0 and measured_steps >= 1 are required")
    resolved = torch.device(device) if not isinstance(device, torch.device) else device
    for _ in range(warmup_steps):
        step_fn()
    synchronize(resolved)
    reset_peak_memory(resolved)
    synchronize(resolved)
    start = torch.cuda.Event(enable_timing=True) if resolved.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if resolved.type == "cuda" else None
    import time

    if start is not None:
        start.record()
        wall0 = time.perf_counter()
    else:
        wall0 = time.perf_counter()
    total_tokens = 0
    for _ in range(measured_steps):
        total_tokens += int(step_fn())
    if end is not None:
        end.record()
        synchronize(resolved)
        elapsed = float(start.elapsed_time(end)) / 1000.0
    else:
        elapsed = time.perf_counter() - wall0
    if elapsed <= 0:
        raise BenchmarkError("measured elapsed time must be positive")
    alloc, reserved = read_peak_memory(resolved)
    tps = tokens_per_second(total_tokens, elapsed)
    return BenchmarkResult(
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        measured_seconds=elapsed,
        target_tokens=total_tokens,
        tokens_per_second=tps,
        peak_allocated_bytes=alloc,
        peak_reserved_bytes=reserved,
        device=str(resolved),
        micro_batch_size=micro_batch_size,
        effective_batch_size=micro_batch_size * accumulation_steps,
        sequence_length=sequence_length,
        notes="throughput uses non-padding target tokens / synchronized measured time",
    )


def _clear_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def search_max_micro_batch(
    probe_fn: Callable[[int], None],
    *,
    start: int = 1,
    max_batch: int = 128,
    confirm_steps: int = 3,
    headroom: bool = True,
) -> BatchSearchResult:
    """Double until failure, then keep the largest size that survives ``confirm_steps``.

    ``probe_fn(batch_size)`` must run one representative training step and raise
    on OOM or numerical failure. Non-OOM exceptions propagate. The returned
    micro-batch is never described as an effective batch.
    """
    if start < 1 or max_batch < start or confirm_steps < 1:
        raise BenchmarkError("invalid batch-search bounds")
    result = BatchSearchResult(best_micro_batch=None)
    candidate = start
    successful: list[int] = []

    def _try(batch: int, repeats: int) -> bool:
        try:
            for _ in range(repeats):
                probe_fn(batch)
            result.tried.append({"micro_batch_size": batch, "status": "ok", "repeats": repeats})
            return True
        except Exception as exc:  # noqa: BLE001 — probe must distinguish OOM
            if is_oom(exc):
                _clear_oom()
                result.failed.append(
                    {
                        "micro_batch_size": batch,
                        "status": "oom",
                        "error": str(exc),
                    }
                )
                return False
            result.failed.append(
                {
                    "micro_batch_size": batch,
                    "status": "error",
                    "error": str(exc),
                }
            )
            raise

    while candidate <= max_batch:
        if _try(candidate, repeats=1):
            successful.append(candidate)
            result.best_micro_batch = candidate
            next_batch = candidate * 2
            if next_batch > max_batch:
                break
            candidate = next_batch
        else:
            break

    if result.best_micro_batch is None:
        result.status = "failed"
        result.notes = "no micro-batch size completed a representative step"
        return result

    # Confirm the largest successful size with repeated steps.
    confirmed = []
    for batch in reversed(successful):
        if _try(batch, repeats=confirm_steps):
            confirmed.append(batch)
            break
    if not confirmed:
        result.status = "failed"
        result.notes = "largest successful sizes failed confirmation repeats"
        result.best_micro_batch = None
        return result

    best = confirmed[0]
    if headroom and len(successful) >= 2:
        # Use the second-largest confirmed doubling rather than the memory limit.
        safer = successful[-2]
        if safer < best:
            result.headroom_applied = True
            result.notes = (
                f"headroom: using micro-batch {safer} instead of limit {best} "
                "so the run is not pinned to a single successful allocation"
            )
            best = safer
    result.best_micro_batch = best
    result.status = "completed"
    return result


def make_step_probe(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> Callable[[int], None]:
    """Build a probe that runs one AdamW step at a requested micro-batch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def probe(micro_batch: int) -> None:
        model.train()
        seq = int(config.sequence_length)
        idx = torch.randint(0, config.vocab_size, (micro_batch, seq), device=device)
        targets = torch.randint(0, config.vocab_size, (micro_batch, seq), device=device)
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(idx, targets)
        if loss is None or not torch.isfinite(loss):
            raise BenchmarkError("non-finite loss during batch probe")
        loss.backward()
        optimizer.step()

    return probe


def run_pilot_and_search(config: ExperimentConfig) -> dict[str, Any]:
    from helpers.model import build_model

    set_seed(config.seed)
    device = config.resolved_device()
    runtime = get_runtime_info(device)
    model = build_model(config).to(device)
    probe = make_step_probe(model, config, device)

    def timed_step() -> int:
        seq = int(config.sequence_length)
        idx = torch.randint(
            0, config.vocab_size, (config.micro_batch_size, seq), device=device
        )
        targets = torch.randint(
            0, config.vocab_size, (config.micro_batch_size, seq), device=device
        )
        opt = getattr(timed_step, "_opt", None)
        if opt is None:
            opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            timed_step._opt = opt  # type: ignore[attr-defined]
        opt.zero_grad(set_to_none=True)
        _logits, loss = model(idx, targets)
        assert loss is not None
        loss.backward()
        opt.step()
        return config.micro_batch_size * seq

    timing = measure_training_step(
        timed_step,
        warmup_steps=1,
        measured_steps=2,
        device=device,
        micro_batch_size=config.micro_batch_size,
        accumulation_steps=config.accumulation_steps,
        sequence_length=int(config.sequence_length),
    )
    search = search_max_micro_batch(
        probe,
        start=config.batch_search_start,
        max_batch=config.max_search_batch,
        confirm_steps=config.confirm_batch_steps,
        headroom=config.headroom,
    )
    return {
        "runtime": runtime,
        "timing": timing.to_dict(),
        "batch_search": search.to_dict(),
        "config": config.to_dict(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pilot timing and maximum micro-batch search for ERA Session 13."
    )
    parser.add_argument("--pilot-batch", action="store_true", help="time a few steps at the configured batch")
    parser.add_argument("--search-max-batch", action="store_true", help="search the largest reliable micro-batch")
    parser.add_argument("--variant", choices=["baseline", "midpoint", "euler"], default="midpoint")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-search-batch", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--proofs", action="store_true", help="run invert/gradient proofs (and GPU memory-vs-depth if CUDA)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.pilot_batch and not args.search_max_batch:
        args.pilot_batch = True
        args.search_max_batch = True
    if args.mode == "smoke":
        config = tiny_config(variant=args.variant, experiment="C" if args.variant != "baseline" else "A")
    else:
        config = assignment_config("C" if args.variant != "baseline" else "A", mode="full", variant=args.variant)
    if args.device:
        config = ExperimentConfig.from_dict({**config.to_dict(), "device": args.device})
    if args.max_search_batch:
        config = ExperimentConfig.from_dict({**config.to_dict(), "max_search_batch": args.max_search_batch})
    if args.micro_batch_size:
        config = ExperimentConfig.from_dict({**config.to_dict(), "micro_batch_size": args.micro_batch_size})
    if args.proofs:
        from helpers.proofs import proof_gradients_match, proof_invert_stack, proof_memory_vs_depth

        proof_report = {
            "proof1_invert": proof_invert_stack(),
            "proof2_grads": proof_gradients_match(),
            "proof3_memory_vs_depth": proof_memory_vs_depth(),
        }
        print(json.dumps(proof_report, indent=2, default=str))
        if args.output:
            Path = __import__("pathlib").Path
            Path(args.output).write_text(json.dumps(proof_report, indent=2, default=str) + "\n")
        if not args.pilot_batch and not args.search_max_batch:
            return 0
    report = run_pilot_and_search(config)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
