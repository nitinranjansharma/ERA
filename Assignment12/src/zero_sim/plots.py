"""Matplotlib figures for the Colab notebook.

Plots are generated in-memory. No external image files are required.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import STRATEGY_LABELS, STRATEGIES, SimulationConfig, gpu_capacity_gib
from .metrics import (
    communication_table,
    memory_ladder_table,
    strategy_comparison_table,
)

STRATEGY_COLORS = {
    "dp": "#4C72B0",
    "zero1": "#DD8452",
    "zero2": "#55A868",
    "zero3": "#C44E52",
}


def _style(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_memory_by_strategy(
    world_size: int = 32,
    config: SimulationConfig | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    cfg = config or SimulationConfig()
    table = strategy_comparison_table(cfg, world_size=world_size)
    fig_ax = ax or plt.subplots(figsize=(9, 5))[1]
    fig = fig_ax.figure
    colors = [STRATEGY_COLORS[sid] for sid in table["strategy_id"]]
    fig_ax.bar(table["strategy"], table["memory_per_gpu_gib"], color=colors)
    capacity = gpu_capacity_gib(cfg.gpu_capacity_gb)
    fig_ax.axhline(
        capacity,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"GPU capacity ({capacity:.1f} GiB)",
    )
    fig_ax.set_ylabel("Training-state memory per GPU (GiB)")
    fig_ax.set_title(f"Per-GPU training-state memory at {world_size} GPUs")
    fig_ax.legend()
    _style(fig_ax)
    fig.tight_layout()
    return fig


def plot_memory_ladder(
    world_sizes: tuple[int, ...] = (8, 16, 32, 64),
    config: SimulationConfig | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    cfg = config or SimulationConfig()
    table = memory_ladder_table(world_sizes, cfg)
    fig_ax = ax or plt.subplots(figsize=(10, 5.5))[1]
    fig = fig_ax.figure
    x = np.arange(len(world_sizes))
    width = 0.18
    for i, strategy in enumerate(STRATEGIES):
        subset = table[table["strategy_id"] == strategy]
        values = [
            float(subset.loc[subset["world_size"] == n, "memory_per_gpu_gib"].iloc[0])
            for n in world_sizes
        ]
        fig_ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
        )
    capacity = gpu_capacity_gib(cfg.gpu_capacity_gb)
    fig_ax.axhline(
        capacity,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"GPU capacity ({capacity:.1f} GiB)",
    )
    fig_ax.set_xticks(x, [str(n) for n in world_sizes])
    fig_ax.set_xlabel("World size (virtual GPUs)")
    fig_ax.set_ylabel("Training-state memory per GPU (GiB)")
    fig_ax.set_title("Memory ladder across GPU counts")
    fig_ax.legend(ncols=2)
    _style(fig_ax)
    fig.tight_layout()
    return fig


def plot_communication(
    config: SimulationConfig | None = None,
) -> plt.Figure:
    table = communication_table(config)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = [STRATEGY_COLORS[sid] for sid in STRATEGIES]
    axes[0].bar(table["strategy"], table["volume_gb"], color=colors)
    axes[0].set_ylabel("Volume (decimal GB)")
    axes[0].set_title("Communication volume per step")
    _style(axes[0])

    x = np.arange(len(table))
    width = 0.38
    axes[1].bar(x - width / 2, table["intra_node_450GBs_s"], width, label="Intra-node 450 GB/s")
    axes[1].bar(x + width / 2, table["inter_node_50GBs_s"], width, label="Inter-node 50 GB/s")
    axes[1].set_xticks(x, table["strategy"])
    axes[1].set_ylabel("Estimated time (seconds)")
    axes[1].set_title("Estimated transfer time (analytical)")
    axes[1].legend()
    _style(axes[1])
    fig.suptitle("Simplified 2P / 3P communication model (not a Colab measurement)")
    fig.tight_layout()
    return fig


def plot_comm_to_compute(
    config: SimulationConfig | None = None,
    world_size: int = 32,
) -> plt.Figure:
    table = strategy_comparison_table(config, world_size=world_size)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [STRATEGY_COLORS[sid] for sid in table["strategy_id"]]
    ax.bar(table["strategy"], table["comm_to_compute_inter"], color=colors)
    ax.set_ylabel("Inter-node communication / compute")
    ax.set_title(
        f"Communication-to-compute ratio at {world_size} GPUs\n"
        "(inter-node 50 GB/s vs assumed transformer FLOPs)"
    )
    _style(ax)
    fig.tight_layout()
    return fig


def plot_loss_curve(
    history: Sequence[float],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    fig_ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    fig = fig_ax.figure
    fig_ax.plot(range(1, len(history) + 1), list(history), marker="o", linewidth=1.8)
    fig_ax.set_xlabel("Training step")
    fig_ax.set_ylabel("Cross-entropy loss")
    fig_ax.set_title("Toy-model loss over training steps (not a 30B run)")
    _style(fig_ax)
    fig.tight_layout()
    return fig


def display_ready_tables() -> dict[str, pd.DataFrame]:
    """Convenience bundle used by the notebook's results section."""
    cfg = SimulationConfig()
    return {
        "comparison_32": strategy_comparison_table(cfg, world_size=32),
        "ladder": memory_ladder_table((8, 16, 32, 64), cfg),
        "communication": communication_table(cfg),
    }
