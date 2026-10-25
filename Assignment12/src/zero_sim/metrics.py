"""Tables and derived metrics for the notebook and tests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .collectives import communication_estimate, estimate_compute_seconds
from .config import (
    COMM_VOLUME_MULTIPLIER,
    STATE_LAYOUT,
    STRATEGY_LABELS,
    STRATEGIES,
    SimulationConfig,
)
from .memory import ladder_rows, memory_breakdown, round_gib


def strategy_comparison_table(
    config: SimulationConfig | None = None,
    world_size: int | None = None,
    tokens_per_step: int = 32 * 2048,
    peak_tflops: float = 312.0,
    utilization: float = 0.35,
) -> pd.DataFrame:
    """Per-strategy memory, fit, and communication snapshot.

    ``peak_tflops`` defaults to a conservative BF16-class figure used only
    for the communication-to-compute sketch. It is an assumption, not a
    measurement of the Colab runtime.
    """
    cfg = config or SimulationConfig()
    n = world_size or cfg.world_size
    rows: list[dict[str, Any]] = []
    compute_s = estimate_compute_seconds(
        num_params=cfg.num_params,
        tokens_per_step=tokens_per_step,
        peak_tflops=peak_tflops,
        utilization=utilization,
        world_size=n,
    )
    for strategy in STRATEGIES:
        mem = memory_breakdown(
            strategy,
            n,
            num_params=cfg.num_params,
            activation_gib=cfg.activation_gib,
            runtime_overhead_gib=cfg.runtime_overhead_gib,
            gpu_capacity_gb=cfg.gpu_capacity_gb,
            weight_bytes=cfg.weight_bytes,
            grad_bytes=cfg.grad_bytes,
            master_bytes=cfg.master_bytes,
            moment_bytes=cfg.moment_bytes,
        )
        comm = communication_estimate(
            strategy,
            p_gb=cfg.param_copy_p_gb,
            intra_node_bw_gb_s=cfg.intra_node_bw_gb_s,
            inter_node_bw_gb_s=cfg.inter_node_bw_gb_s,
        )
        layout = STATE_LAYOUT[strategy]
        sharded = [name for name, kind in layout.items() if kind == "sharded"] or ["none"]
        rows.append(
            {
                "strategy": STRATEGY_LABELS[strategy],
                "strategy_id": strategy,
                "world_size": n,
                "memory_per_gpu_gib": round_gib(mem.training_state_gib),
                "gpu_capacity_gib": round_gib(mem.capacity_gib),
                "headroom_gib": round_gib(mem.headroom_gib),
                "fit_status": "Fits (training state)" if mem.fits else "Does not fit",
                "communication_volume_gb": comm.volume_gb,
                "intra_node_seconds": round(comm.intra_node_seconds, 2),
                "inter_node_seconds": round(comm.inter_node_seconds, 2),
                "comm_to_compute_inter": comm.inter_node_seconds / compute_s,
                "compute_seconds_assumed": compute_s,
                "sharded_states": ", ".join(sharded),
                "assumptions": (
                    "Training-state only unless activations/overhead > 0; "
                    f"P={cfg.param_copy_p_gb:.0f} GB; "
                    f"simplified {int(COMM_VOLUME_MULTIPLIER[strategy])}P volume; "
                    "80 GB ≈ 74.5 GiB; analytical, not measured."
                ),
            }
        )
    return pd.DataFrame(rows)


def memory_ladder_table(
    world_sizes: tuple[int, ...] = (8, 16, 32, 64),
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SimulationConfig()
    rows = []
    for breakdown in ladder_rows(world_sizes, cfg):
        rows.append(
            {
                "strategy": STRATEGY_LABELS[breakdown.strategy],
                "strategy_id": breakdown.strategy,
                "world_size": breakdown.world_size,
                "bytes_per_param": breakdown.bytes_per_param,
                "memory_per_gpu_gib": round_gib(breakdown.training_state_gib),
                "capacity_gib": round_gib(breakdown.capacity_gib),
                "headroom_gib": round_gib(breakdown.headroom_gib),
                "fit_status": "Fits (training state)" if breakdown.fits else "Does not fit",
            }
        )
    return pd.DataFrame(rows)


def memory_breakdown_table(
    world_size: int,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SimulationConfig()
    rows = []
    for strategy in STRATEGIES:
        mem = memory_breakdown(
            strategy,
            world_size,
            num_params=cfg.num_params,
            activation_gib=cfg.activation_gib,
            runtime_overhead_gib=cfg.runtime_overhead_gib,
            gpu_capacity_gb=cfg.gpu_capacity_gb,
        )
        rows.append(
            {
                "strategy": STRATEGY_LABELS[strategy],
                "weights_gib": round_gib(mem.weights_gib),
                "gradients_gib": round_gib(mem.gradients_gib),
                "master_weights_gib": round_gib(mem.master_weights_gib),
                "optimizer_moments_gib": round_gib(mem.optimizer_moments_gib),
                "training_state_gib": round_gib(mem.training_state_gib),
                "activation_gib": round_gib(mem.activation_gib),
                "runtime_overhead_gib": round_gib(mem.runtime_overhead_gib),
                "total_gib": round_gib(mem.total_gib),
                "headroom_gib": round_gib(mem.headroom_gib),
                "fits": mem.fits,
            }
        )
    return pd.DataFrame(rows)


def communication_table(config: SimulationConfig | None = None) -> pd.DataFrame:
    cfg = config or SimulationConfig()
    rows = []
    for strategy in STRATEGIES:
        comm = communication_estimate(
            strategy,
            p_gb=cfg.param_copy_p_gb,
            intra_node_bw_gb_s=cfg.intra_node_bw_gb_s,
            inter_node_bw_gb_s=cfg.inter_node_bw_gb_s,
        )
        rows.append(
            {
                "strategy": STRATEGY_LABELS[strategy],
                "multiplier": f"{int(comm.multiplier)}P",
                "volume_gb": comm.volume_gb,
                "intra_node_450GBs_s": round(comm.intra_node_seconds, 2),
                "inter_node_50GBs_s": round(comm.inter_node_seconds, 2),
            }
        )
    return pd.DataFrame(rows)
