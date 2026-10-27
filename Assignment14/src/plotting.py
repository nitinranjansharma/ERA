"""Reusable visualizations. Figures are written under ``outputs/figures/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset(
    features: torch.Tensor,
    targets: torch.Tensor,
    path: str | Path,
    title: str = "Synthetic regression data",
) -> Path:
    x = features.detach().cpu().numpy()
    y = targets.detach().cpu().numpy().reshape(-1)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    axes = axes.ravel()
    for i in range(4):
        axes[i].scatter(x[:, i], y, s=8, alpha=0.55, c=y, cmap="viridis")
        axes[i].set_xlabel(f"x{i}")
        axes[i].set_ylabel("y")
        axes[i].set_title(f"target vs x{i}")
    axes[4].hist(y, bins=30, color="#3b6ea5", edgecolor="white")
    axes[4].set_xlabel("y")
    axes[4].set_ylabel("count")
    axes[4].set_title("target histogram")
    axes[5].scatter(x[:, 0], x[:, 1], s=8, alpha=0.55, c=y, cmap="viridis")
    axes[5].set_xlabel("x0")
    axes[5].set_ylabel("x1")
    axes[5].set_title("x0 vs x1 (colored by y)")
    fig.suptitle(title)
    return _save(fig, Path(path))


def plot_loss_curves(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
    title: str,
    train_key: str = "train_mse",
    val_key: str = "val_mse",
) -> Path:
    epochs = [row["epoch"] for row in history]
    train = [row[train_key] for row in history]
    val = [row[val_key] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, label="train MSE", color="#1f77b4")
    ax.plot(epochs, val, label="validation MSE", color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, Path(path))


def plot_dense_vs_moe_losses(
    dense_history: Sequence[Mapping[str, Any]],
    moe_history: Sequence[Mapping[str, Any]],
    path: str | Path,
    moe_epoch_offset: int = 0,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    d_epochs = [row["epoch"] for row in dense_history]
    ax.plot(d_epochs, [r["train_mse"] for r in dense_history], label="dense train", color="#1f77b4")
    ax.plot(
        d_epochs,
        [r["val_mse"] for r in dense_history],
        label="dense val",
        color="#1f77b4",
        linestyle="--",
    )
    m_epochs = [row["epoch"] + moe_epoch_offset for row in moe_history]
    ax.plot(m_epochs, [r["train_mse"] for r in moe_history], label="MoE train", color="#ff7f0e")
    ax.plot(
        m_epochs,
        [r["val_mse"] for r in moe_history],
        label="MoE val",
        color="#ff7f0e",
        linestyle="--",
    )
    if moe_epoch_offset:
        ax.axvline(moe_epoch_offset, color="gray", linestyle=":", label="upcycling")
    ax.set_xlabel("epoch (MoE epochs offset after dense training)")
    ax.set_ylabel("mean squared error")
    ax.set_title("Dense pretraining vs MoE fine-tuning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, Path(path))


def plot_expert_utilization(
    selected_load_fractions: Sequence[float],
    ideal_load: float,
    path: str | Path,
    title: str = "Expert selected-load fractions",
    dead_experts: int | None = None,
    max_vio: float | None = None,
) -> Path:
    fractions = np.asarray(selected_load_fractions, dtype=float)
    ids = np.arange(len(fractions))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(ids, fractions, color="#4c78a8", label="selected-load fraction")
    ax.axhline(ideal_load, color="#d62728", linestyle="--", label=f"ideal = {ideal_load:.2f}")
    ax.set_xticks(ids)
    ax.set_xticklabels([f"E{i}" for i in ids])
    ax.set_ylabel("fraction of samples selecting the expert")
    ax.set_xlabel("expert")
    subtitle = title
    extras = []
    if dead_experts is not None:
        extras.append(f"dead experts = {dead_experts}")
    if max_vio is not None and max_vio == max_vio:
        extras.append(f"MaxVio = {max_vio:.3f}")
    if extras:
        subtitle = title + "\n" + ", ".join(extras)
    ax.set_title(subtitle)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, Path(path))


def plot_balancing_comparison(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    path: str | Path,
    before_label: str = "no aux loss",
    after_label: str = "with aux loss",
) -> Path:
    ids = np.arange(len(before["selected_load_fractions"]))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(
        ids - width / 2,
        before["selected_load_fractions"],
        width=width,
        label=before_label,
        color="#9ecae1",
    )
    ax.bar(
        ids + width / 2,
        after["selected_load_fractions"],
        width=width,
        label=after_label,
        color="#3182bd",
    )
    ax.axhline(
        before["ideal_selected_load"],
        color="#d62728",
        linestyle="--",
        label="ideal selected load",
    )
    ax.set_xticks(ids)
    ax.set_xticklabels([f"E{i}" for i in ids])
    ax.set_ylabel("selected-load fraction")
    ax.set_title("Expert utilization with and without load balancing")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, Path(path))


def plot_three_run_story(
    dense_history: Sequence[Mapping[str, Any]],
    upcycled_history: Sequence[Mapping[str, Any]],
    scratch_history: Sequence[Mapping[str, Any]] | None,
    selected_load_fractions: Sequence[float],
    ideal_load: float,
    path: str | Path,
    dense_epochs: int,
) -> Path:
    """Three-panel summary: conversion join, upcycled vs scratch, expert load."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    d_epochs = [row["epoch"] for row in dense_history]
    axes[0].plot(d_epochs, [r["train_mse"] for r in dense_history], color="#5B7FDB", lw=1.7, label="A dense")
    m_epochs = [row["epoch"] + dense_epochs for row in upcycled_history]
    axes[0].plot(
        m_epochs,
        [r["train_mse"] for r in upcycled_history],
        color="#E0823D",
        lw=1.7,
        label="B upcycled MoE",
    )
    axes[0].axvline(dense_epochs, color="#999", ls="--", lw=1)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train MSE")
    axes[0].set_title("1 — it continues")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        [row["epoch"] for row in upcycled_history],
        [r["train_mse"] for r in upcycled_history],
        color="#E0823D",
        lw=1.7,
        label="B upcycled",
    )
    if scratch_history:
        axes[1].plot(
            [row["epoch"] for row in scratch_history],
            [r["train_mse"] for r in scratch_history],
            color="#3FA37A",
            lw=1.7,
            label="C from scratch",
        )
    axes[1].set_xlabel("MoE epoch")
    axes[1].set_ylabel("train MSE")
    axes[1].set_title("2 — upcycling vs starting over")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    fractions = np.asarray(selected_load_fractions, dtype=float)
    ids = np.arange(len(fractions))
    axes[2].bar(ids, fractions, color="#9A62C4", width=0.7, label="selected-load fraction")
    axes[2].axhline(ideal_load, color="#333", ls="--", lw=1.2, label=f"ideal = {ideal_load:.2f}")
    axes[2].set_xticks(ids)
    axes[2].set_xticklabels([f"E{i}" for i in ids])
    axes[2].set_xlabel("expert")
    axes[2].set_ylabel("selected-load fraction")
    axes[2].set_title("3 — expert load")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Dense → MoE: continue, compare, check load", y=1.02)
    return _save(fig, Path(path))
