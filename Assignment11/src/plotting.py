"""Plots generated from saved experiment outputs, not from hard-coded numbers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import ensure_dir


def _style(ax) -> None:
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_bias_correction(task_b: dict, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = Path(out) / "bias_correction.png"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex="col")
    names = (("constant", "Constant gradient"), ("varying", "Sign-changing gradient"))
    for col, (key, title) in enumerate(names):
        rows = task_b["scenarios"][key]["plot_rows"]
        t = [r["t"] for r in rows]
        axes[0, col].plot(t, [abs(r["delta_corrected"]) for r in rows], marker="o", label="bias corrected")
        axes[0, col].plot(t, [abs(r["delta_uncorrected"]) for r in rows], marker="s", label="no correction")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("|update|")
        axes[0, col].legend(frameon=False)
        _style(axes[0, col])
        axes[1, col].plot(t, [r["relative_difference"] for r in rows], color="tab:purple", marker="o")
        axes[1, col].set_xlabel("step t")
        axes[1, col].set_ylabel("relative difference")
        _style(axes[1, col])
    fig.suptitle("Adam bias correction, first 20 steps")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_update_to_weight(task_c: dict, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = Path(out) / "update_to_weight_warmup.png"
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.4), sharex=True)
    wu = task_c["with_warmup"]["logs"]
    nw = task_c["without_warmup"]["logs"]
    t = [r["global_step"] for r in wu]
    peak = task_c["lr_peak_step"]
    ratio_peak = task_c.get("ratio_peak_step")
    stab = task_c.get("ratio_stabilize_step")
    axes[0].plot(t, [r["learning_rate"] for r in wu], label="warmup")
    axes[0].plot(t, [r["learning_rate"] for r in nw], label="no warmup", linestyle="--")
    axes[0].axvline(peak, color="tab:red", linestyle=":", label=f"LR peak step {peak}")
    axes[0].set_ylabel("learning rate")
    axes[0].legend(frameon=False)
    _style(axes[0])
    axes[1].plot(t, [r["update_to_weight_ratio"] for r in wu], label="warmup")
    axes[1].plot(
        [r["global_step"] for r in nw],
        [r["update_to_weight_ratio"] for r in nw],
        label="no warmup",
        linestyle="--",
    )
    axes[1].axvline(peak, color="tab:red", linestyle=":", label="LR peak")
    if ratio_peak is not None:
        axes[1].axvline(ratio_peak, color="tab:purple", linestyle="-.", label=f"ratio peak step {ratio_peak}")
    if stab is not None:
        axes[1].axvline(stab, color="tab:green", linestyle="--", label=f"rel-change hold step {stab}")
    axes[1].set_xlabel("global step")
    axes[1].set_ylabel("mean update/weight")
    axes[1].legend(frameon=False)
    _style(axes[1])
    fig.suptitle("Warmup vs no-warmup: learning rate and update-to-weight ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_cosine_vs_wsd(task_d: dict, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = Path(out) / "cosine_vs_wsd.png"
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    for key, label, color in (
        ("cosine_300", "cosine (300-step)", "tab:blue"),
        ("wsd_300", "WSD (300-step)", "tab:orange"),
    ):
        hist = task_d[key]["history"]
        t = [r["global_step"] for r in hist]
        axes[0].plot(t, [r["learning_rate"] for r in hist], label=label, color=color)
        val_t = [r["global_step"] for r in hist if r["validation_loss"] is not None]
        val = [r["validation_loss"] for r in hist if r["validation_loss"] is not None]
        axes[1].plot(t, [r["training_loss"] for r in hist], color=color, alpha=0.35)
        axes[1].plot(val_t, val, label=label, color=color)
        late = [(r["global_step"], r["validation_loss"]) for r in hist if r["validation_loss"] is not None and r["global_step"] >= 80]
        if late:
            axes[2].plot([p[0] for p in late], [p[1] for p in late], label=label, color=color)
    axes[0].axvline(200, color="grey", linestyle=":", label="step 200")
    axes[1].axvline(200, color="grey", linestyle=":")
    axes[0].set_ylabel("learning rate")
    axes[1].set_ylabel("loss")
    axes[2].set_ylabel("val loss (step ≥ 80)")
    axes[2].set_xlabel("optimizer step")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, title="validation (solid) / train (faint)")
    labeled = [line for line in axes[2].lines if line.get_label() and not str(line.get_label()).startswith("_")]
    if labeled:
        axes[2].axvline(200, color="grey", linestyle=":")
        axes[2].legend(frameon=False)
    _style(axes[0])
    _style(axes[1])
    _style(axes[2])
    fig.suptitle("Cosine vs WSD on a matched tiny sequence model")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_width_lr_sweep(task_e: dict, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = Path(out) / "lr_sweep_widths.png"
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    colors = {256: "tab:blue", 512: "tab:orange", 1024: "tab:green"}
    for key, payload in task_e["per_width"].items():
        width = payload["width"]
        cells = [c for c in payload["cells"] if c.get("finite")]
        xs = [c["lr"] for c in cells]
        ys = [c["val_loss"] for c in cells]
        ax.plot(xs, ys, marker="o", label=f"width {width}", color=colors.get(width))
        best = payload["best"]
        ax.scatter(
            [best["lr"]],
            [best["val_loss"]],
            s=90,
            marker="*",
            zorder=5,
            color=colors.get(width),
            edgecolor="black",
        )
        marker = "boundary" if best.get("on_boundary") else "interior"
        offset = {"256": (8, 12), "512": (8, -18), "1024": (-52, 10)}.get(str(width), (6, 8))
        ax.annotate(
            f"w={width}\n{best['lr']:.2e} ({marker})",
            (best["lr"], best["val_loss"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("validation MSE")
    ax.legend(frameon=False)
    _style(ax)
    fig.suptitle("Standard-parameterization LR sweep vs width (stars = observed minima)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_all(results: dict, out_dir: str | Path) -> dict[str, str]:
    paths = {}
    if "B" in results:
        paths["bias_correction"] = str(plot_bias_correction(results["B"], out_dir))
    if "C" in results:
        paths["update_to_weight"] = str(plot_update_to_weight(results["C"], out_dir))
    if "D" in results:
        paths["cosine_vs_wsd"] = str(plot_cosine_vs_wsd(results["D"], out_dir))
    if "E" in results:
        paths["lr_sweep"] = str(plot_width_lr_sweep(results["E"], out_dir))
    return paths
