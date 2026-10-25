"""Benchmark timing, memory reporting, and max-batch search tests."""

from __future__ import annotations

import pytest
import torch

from helpers.benchmark import (
    BenchmarkError,
    is_oom,
    measure_training_step,
    read_peak_memory,
    search_max_micro_batch,
)
from helpers.config import tiny_config
from helpers.metrics import MetricsError, tokens_per_second, validate_report
from helpers.reporting import export_run_artifacts, pending_results_template


def test_warmup_excluded_from_timing_and_cpu_memory_is_none():
    calls = {"n": 0}

    def step():
        calls["n"] += 1
        torch.randn(8, 8).sum().item()
        return 16

    result = measure_training_step(
        step,
        warmup_steps=2,
        measured_steps=3,
        device="cpu",
        micro_batch_size=2,
        sequence_length=8,
    )
    assert calls["n"] == 5
    assert result.warmup_steps == 2
    assert result.measured_steps == 3
    assert result.target_tokens == 48
    assert result.tokens_per_second == tokens_per_second(48, result.measured_seconds)
    alloc, reserved = read_peak_memory("cpu")
    assert alloc is None and reserved is None
    assert result.peak_allocated_bytes is None
    assert result.device == "cpu"


def test_throughput_rejects_zero_time():
    with pytest.raises(MetricsError):
        tokens_per_second(10, 0.0)


def test_search_handles_simulated_oom_and_headroom():
    def probe(batch: int) -> None:
        if batch > 8:
            raise torch.cuda.OutOfMemoryError("simulated OOM")

    result = search_max_micro_batch(
        probe, start=1, max_batch=32, confirm_steps=3, headroom=True
    )
    assert result.status == "completed"
    assert result.best_micro_batch == 4  # headroom below 8
    assert result.headroom_applied is True
    assert any(row["status"] == "oom" for row in result.failed)
    assert result.best_micro_batch != 16  # must not confuse with a larger size


def test_search_propagates_non_oom_errors():
    def probe(batch: int) -> None:
        raise ValueError("not oom")

    with pytest.raises(ValueError, match="not oom"):
        search_max_micro_batch(probe, start=1, max_batch=4, confirm_steps=1)


def test_is_oom_detection():
    assert is_oom(torch.cuda.OutOfMemoryError("oom"))
    assert is_oom(RuntimeError("CUDA out of memory"))
    assert not is_oom(RuntimeError("shape mismatch"))


def test_report_export_and_pending_template(tmp_path):
    pending = pending_results_template()
    assert len(pending) == 4
    assert all(row["status"] == "pending" for row in pending)
    # Completed synthetic report for export (not a 50M result).
    report = {
        "experiment": "A",
        "variant": "baseline",
        "status": "smoke",
        "trainable_parameters": 1000,
        "micro_batch_size": 2,
        "accumulation_steps": 1,
        "effective_batch_size": 2,
        "sequence_length": 8,
        "training_target_tokens": 32,
        "token_budget": 32,
        "final_train_loss": 2.5,
        "final_val_loss": 2.6,
        "tokens_per_second": 16.0,
        "wall_clock_seconds": 2.0,
        "time_per_optimizer_step": 1.0,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "device": "cpu",
        "precision": "fp32",
        "reversible_method": "none",
    }
    validate_report(report)
    artifacts = export_run_artifacts(tmp_path, [report], histories={"A": [{"tokens": 16, "train_loss": 2.7}]})
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert "loss_plot" in artifacts
