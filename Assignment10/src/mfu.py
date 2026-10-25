"""Model FLOPs Utilization.

Assignment formula (the 6ND transformer-training approximation)
--------------------------------------------------------------
    estimated_training_flops_per_second = 6 * parameter_count * tokens_per_second
    MFU = estimated_training_flops_per_second / available_hardware_flops_per_second

The factor 6 is a *transformer* rule of thumb (≈ 2N forward + 4N backward for
the dominant matmuls). It is not an exact FLOP count of an arbitrary MLP.

Units
-----
All internal arithmetic uses FLOP/s (not TFLOP/s) and seconds (not milliseconds).
Callers that pass TFLOP/s or milliseconds must do so through the explicit
conversion helpers; mixing units raises MFUError rather than silently scaling.

Assignment-provided illustrative scenario (NOT a local measurement)
------------------------------------------------------------------
    9e9 params, 12_000 tok/s, 648 TFLOP/s achieved, 7_912 TFLOP/s peak → ≈ 8.2% MFU
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from models import TinyGPT, count_parameters
from safety import MFUError


TFLOP = 1e12  # decimal tera, matching vendor GPU peak quotes
ASSIGNMENT_EXAMPLE = {
    "source": "assignment-provided illustrative figures, not a local benchmark",
    "parameter_count": 9_000_000_000,
    "tokens_per_second": 12_000,
    "estimated_tflops_per_second": 648.0,
    "hardware_peak_tflops_per_second": 7_912.0,
    "approximate_mfu_percent": 8.2,
}


def estimated_training_flops_per_second(parameter_count: int, tokens_per_second: float) -> float:
    if parameter_count < 0:
        raise MFUError(f"parameter_count must be non-negative, got {parameter_count}")
    if tokens_per_second < 0:
        raise MFUError(f"tokens_per_second must be non-negative, got {tokens_per_second}")
    return 6.0 * float(parameter_count) * float(tokens_per_second)


def mfu_ratio(flops_per_second: float, peak_flops_per_second: float) -> float:
    if peak_flops_per_second <= 0:
        raise MFUError(
            f"available_hardware_flops_per_second must be positive, got {peak_flops_per_second}"
        )
    if flops_per_second < 0:
        raise MFUError(f"estimated FLOP/s must be non-negative, got {flops_per_second}")
    return flops_per_second / peak_flops_per_second


def mfu_percent(flops_per_second: float, peak_flops_per_second: float) -> float:
    return 100.0 * mfu_ratio(flops_per_second, peak_flops_per_second)


def tflops_to_flops(tflops: float) -> float:
    if tflops < 0:
        raise MFUError(f"TFLOP/s must be non-negative, got {tflops}")
    return tflops * TFLOP


def flops_to_tflops(flops: float) -> float:
    return flops / TFLOP


def milliseconds_to_seconds(ms: float) -> float:
    if ms < 0:
        raise MFUError(f"milliseconds must be non-negative, got {ms}")
    return ms / 1000.0


def seconds_to_milliseconds(seconds: float) -> float:
    return seconds * 1000.0


def assignment_example_mfu() -> dict:
    """Reproduce the assignment's 9B / 12k tok/s illustration."""
    params = ASSIGNMENT_EXAMPLE["parameter_count"]
    tps = ASSIGNMENT_EXAMPLE["tokens_per_second"]
    achieved = estimated_training_flops_per_second(params, tps)
    peak = tflops_to_flops(ASSIGNMENT_EXAMPLE["hardware_peak_tflops_per_second"])
    percent = mfu_percent(achieved, peak)
    return {
        **ASSIGNMENT_EXAMPLE,
        "computed_flops_per_second": achieved,
        "computed_tflops_per_second": flops_to_tflops(achieved),
        "computed_mfu_percent": percent,
        "matches_stated_648_tflops": abs(flops_to_tflops(achieved) - 648.0) < 1e-9,
        "matches_stated_8p2_percent": abs(percent - 8.2) < 0.05,
    }


def theoretical_neon_fp32_peak_flops(
    n_performance_cores: int,
    p_core_ghz: float,
    n_efficiency_cores: int,
    e_core_ghz: float,
    neon_fp32_lanes: int = 4,
    fma_flops_per_lane: int = 2,
) -> dict:
    """Documented NEON FP32 peak: cores × Hz × 4 FP32 lanes × 2 FMA flops.

    This is a *theoretical vector peak*, not a measured GEMM number. Apple M-series
    chips also have AMX, which can exceed NEON on large GEMMs.
    """
    p = n_performance_cores * p_core_ghz * 1e9 * neon_fp32_lanes * fma_flops_per_lane
    e = n_efficiency_cores * e_core_ghz * 1e9 * neon_fp32_lanes * fma_flops_per_lane
    return {
        "kind": "theoretical_neon_fp32_peak",
        "n_performance_cores": n_performance_cores,
        "p_core_ghz": p_core_ghz,
        "n_efficiency_cores": n_efficiency_cores,
        "e_core_ghz": e_core_ghz,
        "neon_fp32_lanes": neon_fp32_lanes,
        "fma_flops_per_lane": fma_flops_per_lane,
        "p_core_flops": p,
        "e_core_flops": e,
        "peak_flops_per_second": p + e,
        "peak_tflops_per_second": (p + e) / TFLOP,
        "notes": (
            "FMA counts as two FLOPs. This ignores AMX, the GPU, and the Neural Engine. "
            "A tiny PyTorch loop will not approach this number."
        ),
    }


def apple_m2_peaks() -> dict:
    """Peaks for the machine this project was developed on: Apple M2, 4P+4E.

    Clocks are publicly documented typical maxima (≈ 3.49 GHz P / 2.42 GHz E).
    The 1.09 TFLOP/s CPU GEMM figure is a literature *measured* Accelerate/AMX
    peak from Banchelli et al., arXiv:2502.05317, not measured in this repo.
    """
    neon = theoretical_neon_fp32_peak_flops(4, 3.49, 4, 2.42)
    return {
        "hardware": "Apple M2 (4 performance + 4 efficiency cores), 8 GB unified memory",
        "precision": "fp32",
        "accelerator_count": 1,
        "theoretical_neon_fp32": neon,
        "literature_amx_gemm_tflops": {
            "tflops_per_second": 1.09,
            "flops_per_second": 1.09 * TFLOP,
            "source": "arXiv:2502.05317 (measured vDSP/Accelerate SGEMM on M2 CPU)",
            "kind": "literature_measured_effective_peak",
            "notes": "Not reproduced in this assignment; used only as a published effective-peak reference.",
        },
        "gpu": {
            "available_in_this_process": bool(torch.cuda.is_available())
            or bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "notes": "No CUDA. MPS may or may not be visible to this PyTorch build; GPU MFU is not reported unless a device actually runs.",
        },
    }


def measure_tokens_per_second(
    *,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
    timed_steps: int,
    vocab_size: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 32,
    block_size: int = 32,
    device: torch.device | str = "cpu",
    lr: float = 1e-3,
) -> dict:
    if seq_len > block_size:
        raise MFUError(f"seq_len {seq_len} exceeds block_size {block_size}")
    device = torch.device(device)
    model = TinyGPT(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=0.0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    param_count = count_parameters(model, trainable_only=True)
    tokens_per_step = batch_size * seq_len

    def one_step() -> None:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), targets.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(warmup_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(timed_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    if elapsed_s <= 0:
        raise MFUError("timed interval was non-positive; cannot compute tokens/s")
    tokens = tokens_per_step * timed_steps
    tps = tokens / elapsed_s
    return {
        "device": str(device),
        "precision": "fp32",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "elapsed_s": elapsed_s,
        "tokens_processed": tokens,
        "tokens_per_second": tps,
        "parameter_count_trainable": param_count,
        "parameter_count_total": count_parameters(model, trainable_only=False),
        "model": {
            "class": "TinyGPT",
            "vocab_size": vocab_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "block_size": block_size,
        },
        "timing": "perf_counter around the timed_steps loop; warmup excluded; CUDA synchronize if CUDA",
    }


def mfu_from_measurement(
    measurement: dict,
    peak_flops_per_second: float,
    *,
    peak_kind: str,
) -> dict:
    flops_s = estimated_training_flops_per_second(
        measurement["parameter_count_trainable"],
        measurement["tokens_per_second"],
    )
    percent = mfu_percent(flops_s, peak_flops_per_second)
    return {
        **measurement,
        "formula": "6 * N * tokens_per_second",
        "estimated_training_flops_per_second": flops_s,
        "estimated_training_tflops_per_second": flops_to_tflops(flops_s),
        "peak_flops_per_second": peak_flops_per_second,
        "peak_tflops_per_second": flops_to_tflops(peak_flops_per_second),
        "peak_kind": peak_kind,
        "mfu": mfu_ratio(flops_s, peak_flops_per_second),
        "mfu_percent": percent,
        "gap_to_40_percent_points": 40.0 - percent,
    }


def run_mfu_experiment(
    *,
    warmup_steps: int = 5,
    timed_steps: int = 20,
    configs: Optional[list[dict]] = None,
    device: torch.device | str = "cpu",
) -> dict:
    configs = configs or [
        {"name": "small", "batch_size": 4, "seq_len": 16},
        {"name": "large", "batch_size": 16, "seq_len": 32},
    ]
    peaks = apple_m2_peaks()
    neon_peak = peaks["theoretical_neon_fp32"]["peak_flops_per_second"]
    amx_peak = peaks["literature_amx_gemm_tflops"]["flops_per_second"]
    example = assignment_example_mfu()
    results = []
    for cfg in configs:
        meas = measure_tokens_per_second(
            batch_size=int(cfg["batch_size"]),
            seq_len=int(cfg["seq_len"]),
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            device=device,
        )
        meas["config_name"] = cfg["name"]
        results.append(
            {
                "config_name": cfg["name"],
                "measurement": meas,
                "vs_theoretical_neon": mfu_from_measurement(
                    meas, neon_peak, peak_kind="theoretical_neon_fp32_peak"
                ),
                "vs_literature_amx": mfu_from_measurement(
                    meas, amx_peak, peak_kind="literature_measured_amx_gemm"
                ),
            }
        )
    return {
        "assignment_example": example,
        "hardware": peaks,
        "software": {
            "torch": torch.__version__,
            "device": str(device),
        },
        "configs": results,
        "gap_discussion": [
            "The 6ND formula itself is an LLM-training approximation; TinyGPT's embeddings and tiny width make the true FLOP mix different.",
            "Workloads are tiny: kernel-launch and Python/autograd overhead dominate arithmetic.",
            "CPU PyTorch eager mode has no fused transformer kernel; each Linear/softmax is a separate launch.",
            "No GPU tensor cores are used; the assignment's 40% class of utilisation is a large-accelerator, large-batch regime.",
            "Memory traffic for small GEMMs is not hidden; the working set is cache-resident but launch-bound.",
            "No distributed communication or activation checkpointing is present here — those would *lower* MFU further on a multi-GPU run, they are not the cause of this particular gap.",
        ],
        "gpu_mfu_reported": False,
        "gpu_mfu_reason": "CUDA unavailable; MPS not used. GPU MFU is not invented.",
    }
