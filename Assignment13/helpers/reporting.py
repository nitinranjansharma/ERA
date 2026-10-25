"""CSV/JSON export, plot generation, and experiment comparison.

Reference S13 figures (saved under ``outputs/plots/`` and ``outputs/images/``):

* ``results.png`` — loss vs tokens, peak memory bars, throughput bars
* ``euler_vs_midpoint.png`` — zoomed Euler vs midpoint + gap
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from helpers.config import ensure_dir
from helpers.metrics import (
    REQUIRED_REPORT_FIELDS,
    bytes_to_mib,
    validate_report,
)

PLOT_COLOR = {
    "A": "#5B7FDB",
    "normal": "#5B7FDB",
    "baseline": "#5B7FDB",
    "B_euler": "#E0823D",
    "euler": "#E0823D",
    "B_midpoint": "#3FA37A",
    "midpoint": "#3FA37A",
    "C": "#9A62C4",
    "pushed": "#9A62C4",
}

COMPARISON_COLUMNS = [
    "experiment",
    "trainable_parameters",
    "reversible_method",
    "sequence_length",
    "micro_batch_size",
    "effective_batch_size",
    "training_target_tokens",
    "final_train_loss",
    "final_val_loss",
    "tokens_per_second",
    "wall_clock_seconds",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "status",
]


class ReportingError(ValueError):
    """Invalid or incomplete experiment artifacts."""


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = list(columns or COMPARISON_COLUMNS)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return path
    fieldnames = list(columns or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def comparison_row(report: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_report(report)
    method = validated.get("reversible_method")
    if not method:
        method = "none" if validated.get("variant") == "baseline" else validated.get("variant")
    row = {column: validated.get(column) for column in COMPARISON_COLUMNS}
    row["reversible_method"] = method
    return row


def write_comparison_table(path: str | Path, reports: Sequence[Mapping[str, Any]]) -> Path:
    rows = [comparison_row(report) for report in reports]
    return write_csv(path, rows, COMPARISON_COLUMNS)


def markdown_table(reports: Sequence[Mapping[str, Any]]) -> str:
    if not reports:
        return (
            "| experiment | parameters | variant | seq | micro-batch | effective batch "
            "| train tokens | train loss | val loss | tok/s | wall (s) | peak alloc | peak reserved | status |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| _pending_ |  |  |  |  |  |  |  |  |  |  |  |  |  |\n"
        )
    header = (
        "| experiment | parameters | variant | seq | micro-batch | effective batch "
        "| train tokens | train loss | val loss | tok/s | wall (s) | peak alloc MiB | peak reserved MiB | status |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for report in reports:
        row = comparison_row(report)
        alloc = bytes_to_mib(row.get("peak_allocated_bytes"))
        reserved = bytes_to_mib(row.get("peak_reserved_bytes"))
        lines.append(
            "| {experiment} | {trainable_parameters} | {reversible_method} | {sequence_length} | "
            "{micro_batch_size} | {effective_batch_size} | {training_target_tokens} | "
            "{final_train_loss} | {final_val_loss} | {tokens_per_second} | {wall_clock_seconds} | "
            "{alloc} | {reserved} | {status} |".format(
                alloc=_fmt(alloc, 1) if alloc is not None else "n/a (cpu)",
                reserved=_fmt(reserved, 1) if reserved is not None else "n/a (cpu)",
                **{k: _fmt(row.get(k)) for k in (
                    "experiment",
                    "trainable_parameters",
                    "reversible_method",
                    "sequence_length",
                    "micro_batch_size",
                    "effective_batch_size",
                    "training_target_tokens",
                    "final_train_loss",
                    "final_val_loss",
                    "tokens_per_second",
                    "wall_clock_seconds",
                    "status",
                )},
            )
        )
    return "\n".join(lines) + "\n"


def report_display_key(report: Mapping[str, Any]) -> str:
    exp = str(report.get("experiment") or "?")
    variant = str(report.get("variant") or "")
    if exp == "A" or variant in ("baseline", "normal"):
        return "A"
    if variant == "euler" and exp in ("B", "?"):
        return "B_euler"
    if variant == "midpoint" and exp == "B":
        return "B_midpoint"
    if exp == "C":
        return "C"
    return f"{exp}_{variant}".strip("_")


def named_reports(reports: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    named: dict[str, Mapping[str, Any]] = {}
    if isinstance(reports, Mapping):
        items = list(reports.items())
    else:
        items = [(report_display_key(row), row) for row in reports]
    for key, report in items:
        named[str(key)] = report
        named[report_display_key(report)] = report
        variant = str(report.get("variant") or "")
        if variant:
            named[variant] = report
    return named


def _history(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    extra = report.get("extra") if isinstance(report.get("extra"), Mapping) else {}
    hist = extra.get("history") if extra else None
    if hist:
        return list(hist)
    return list(report.get("history") or [])


def _peak_mb(report: Mapping[str, Any]) -> float:
    extra = report.get("extra") if isinstance(report.get("extra"), Mapping) else {}
    if extra and extra.get("peak_memory_mb") is not None:
        return float(extra["peak_memory_mb"])
    raw = report.get("peak_allocated_bytes")
    if raw in (None, 0):
        return 0.0
    return float(raw) / 2**20


def _setup_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def plot_loss_vs_tokens(histories: Mapping[str, Sequence[Mapping[str, Any]]], path: str | Path) -> Path:
    plt = _setup_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, history in histories.items():
        tokens = [row["tokens"] for row in history if "tokens" in row and "train_loss" in row]
        losses = [row["train_loss"] for row in history if "tokens" in row and "train_loss" in row]
        if tokens:
            ax.plot(tokens, losses, color=PLOT_COLOR.get(str(name)), lw=1.5, label=f"{name} train")
        val_t = [row["tokens"] for row in history if row.get("val_loss") is not None]
        val_l = [row["val_loss"] for row in history if row.get("val_loss") is not None]
        if val_t:
            ax.plot(val_t, val_l, linestyle="--", label=f"{name} val")
    ax.set_xlabel("Training target tokens")
    ax.set_ylabel("Loss")
    ax.set_title("Loss versus training tokens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_throughput(reports: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    plt = _setup_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [report_display_key(r) for r in reports]
    values = [float(r["tokens_per_second"]) if r.get("tokens_per_second") is not None else 0.0 for r in reports]
    colors = [PLOT_COLOR.get(label, "#888") for label in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Training throughput")
    ax.margins(y=0.18)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_peak_memory(reports: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    plt = _setup_pyplot()
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [report_display_key(r) for r in reports]
    alloc = [bytes_to_mib(r.get("peak_allocated_bytes")) or 0.0 for r in reports]
    reserved = [bytes_to_mib(r.get("peak_reserved_bytes")) or 0.0 for r in reports]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, alloc, width, label="allocated")
    ax.bar(x + width / 2, reserved, width, label="reserved")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MiB")
    ax.set_title("Peak GPU memory (0 if CPU / unavailable)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def export_run_artifacts(
    output_dir: str | Path,
    reports: Sequence[Mapping[str, Any]],
    histories: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, str]:
    root = ensure_dir(output_dir)
    json_path = write_json(root / "results.json", list(reports))
    csv_path = write_comparison_table(root / "results.csv", reports)
    table_path = root / "results.md"
    table_path.write_text(markdown_table(reports))
    artifacts = {
        "results_json": str(json_path),
        "results_csv": str(csv_path),
        "results_md": str(table_path),
    }
    if not histories:
        histories = {report_display_key(report): _history(report) for report in reports if _history(report)}
    artifacts.update(save_reference_figures(reports, root, histories=histories))
    return artifacts


def pending_results_template() -> list[dict[str, Any]]:
    """Empty rows for the four-run protocol. Not fabricated measurements."""
    rows = []
    for exp, variant, batch_note in (
        ("A", "baseline", "batch 32, normal residual"),
        ("B", "euler", "batch 32, velocity Euler"),
        ("B", "midpoint", "batch 32, leapfrog midpoint"),
        ("C", "winner", "batch 128 then 64 then 32 on OOM"),
    ):
        rows.append(
            {
                "experiment": exp,
                "variant": variant,
                "reversible_method": "none" if variant == "baseline" else variant,
                "status": "pending",
                "trainable_parameters": 19_876_608,
                "micro_batch_size": 32 if exp != "C" else None,
                "accumulation_steps": 1,
                "effective_batch_size": 32 if exp != "C" else None,
                "sequence_length": 512,
                "training_target_tokens": 50_000_000,
                "token_budget": 50_000_000,
                "final_train_loss": None,
                "final_val_loss": None,
                "tokens_per_second": None,
                "wall_clock_seconds": None,
                "time_per_optimizer_step": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "device": None,
                "precision": "auto",
                "notes": batch_note,
            }
        )
    return rows


def plot_four_run_panel(
    reports: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    path: str | Path,
    keys: Sequence[str] | None = None,
) -> Path:
    """Loss / peak memory / throughput panel used by the proven S13 write-up."""
    plt = _setup_pyplot()
    named = named_reports(reports)
    keys = list(keys or [k for k in ("A", "B_euler", "B_midpoint", "C") if k in named])
    if not keys:
        keys = [k for k in named if k in ("A", "B_euler", "B_midpoint", "C") or k == report_display_key(named[k])]
        # unique in insertion order
        seen: set[str] = set()
        ordered = []
        for key in keys:
            if key not in seen and key in named:
                seen.add(key)
                ordered.append(key)
        keys = ordered or list(dict.fromkeys(named))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for key in keys:
        hist = _history(named[key])
        ax[0].plot(
            [float(row.get("tokens") or 0) / 1e6 for row in hist],
            [row.get("smooth_loss", row.get("train_loss")) for row in hist],
            color=PLOT_COLOR.get(key, PLOT_COLOR.get(str(named[key].get("variant")), "#444444")),
            lw=1.5,
            label=key,
        )
    ax[0].set(xlabel="millions of tokens read", ylabel="loss (lower is better)", title="Did it learn?")
    ax[0].legend(fontsize=7)

    mem = [_peak_mb(named[k]) for k in keys]
    tps = [float(named[k].get("tokens_per_second") or 0) for k in keys]
    tick_labels = []
    for key in keys:
        report = named[key]
        mode = "normal" if report_display_key(report) == "A" else str(report.get("variant") or key)
        batch = report.get("micro_batch_size", "")
        tick_labels.append(f"{mode}\nbatch {batch}")
    for axis, title, vals in (
        (ax[1], "Peak GPU memory (MB)", mem),
        (ax[2], "Speed (tokens per second)", tps),
    ):
        axis.bar(range(len(keys)), vals, color=[PLOT_COLOR.get(k, "#888888") for k in keys], width=0.6)
        axis.set_xticks(range(len(keys)))
        axis.set_xticklabels(tick_labels, fontsize=7)
        axis.set_title(title)
        axis.margins(y=0.18)
        for i, value in enumerate(vals):
            axis.text(i, value, f"{value:,.0f}" if value else "n/a", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_euler_vs_midpoint(
    euler: Mapping[str, Any],
    midpoint: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Zoomed Euler vs midpoint trajectories (proven S13 figure 2)."""
    plt = _setup_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eh = _history(euler)
    mh = _history(midpoint)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(
        [r.get("step") for r in eh],
        [r.get("smooth_loss", r.get("train_loss")) for r in eh],
        color="#E0823D",
        label="euler",
    )
    ax[0].plot(
        [r.get("step") for r in mh],
        [r.get("smooth_loss", r.get("train_loss")) for r in mh],
        color="#3FA37A",
        label="midpoint",
    )
    ax[0].set(xlabel="step", ylabel="loss", title="Euler vs midpoint")
    ax[0].legend()
    n = min(len(eh), len(mh))
    if n:
        gap = [
            (mh[i].get("smooth_loss", mh[i].get("train_loss")) or 0)
            - (eh[i].get("smooth_loss", eh[i].get("train_loss")) or 0)
            for i in range(n)
        ]
        ax[1].plot([eh[i].get("step") for i in range(n)], gap, color="#555555")
        ax[1].axhline(0.0, color="#aaaaaa", lw=0.8)
        ax[1].set(xlabel="step", ylabel="midpoint − euler", title="Gap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_run_loss_plot(report: Mapping[str, Any], path: str | Path) -> Path | None:
    """Per-run loss curve, written next to that experiment's metrics.json."""
    hist = _history(report)
    if not hist:
        return None
    key = report_display_key(report)
    return plot_loss_vs_tokens({key: hist}, path)


def _copy_png(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def save_reference_figures(
    reports: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    histories: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, str]:
    """Write the proven S13 figures into ``plots/`` and ``images/`` under output_dir."""
    root = ensure_dir(output_dir)
    plots = ensure_dir(root / "plots")
    images = ensure_dir(root / "images")
    named = named_reports(reports)
    ordered = [named[k] for k in ("A", "B_euler", "B_midpoint", "C") if k in named]
    if not ordered:
        seen: set[int] = set()
        ordered = []
        for report in named.values():
            ident = id(report)
            if ident not in seen:
                seen.add(ident)
                ordered.append(report)
    artifacts: dict[str, str] = {}
    if histories:
        artifacts["loss_plot"] = str(plot_loss_vs_tokens(histories, plots / "loss_vs_tokens.png"))
        _copy_png(plots / "loss_vs_tokens.png", images / "loss_vs_tokens.png")
    elif ordered:
        hist_map = {report_display_key(r): _history(r) for r in ordered if _history(r)}
        if hist_map:
            artifacts["loss_plot"] = str(plot_loss_vs_tokens(hist_map, plots / "loss_vs_tokens.png"))
            _copy_png(plots / "loss_vs_tokens.png", images / "loss_vs_tokens.png")
    if ordered:
        artifacts["throughput_plot"] = str(plot_throughput(ordered, plots / "tokens_per_second.png"))
        artifacts["memory_plot"] = str(plot_peak_memory(ordered, plots / "peak_memory.png"))
        artifacts["results_png"] = str(plot_four_run_panel(named, plots / "results.png"))
        artifacts["results_png_images"] = str(_copy_png(plots / "results.png", images / "results.png"))
        _copy_png(plots / "tokens_per_second.png", images / "tokens_per_second.png")
        _copy_png(plots / "peak_memory.png", images / "peak_memory.png")
        if "B_euler" in named and "B_midpoint" in named:
            artifacts["euler_vs_midpoint"] = str(
                plot_euler_vs_midpoint(named["B_euler"], named["B_midpoint"], plots / "euler_vs_midpoint.png")
            )
            artifacts["euler_vs_midpoint_images"] = str(
                _copy_png(plots / "euler_vs_midpoint.png", images / "euler_vs_midpoint.png")
            )
    return artifacts


def refresh_output_plots(output_dir: str | Path) -> dict[str, str]:
    """Rebuild comparison figures from every ``exp_*/metrics.json`` under output_dir."""
    root = Path(output_dir)
    reports: list[dict[str, Any]] = []
    for metrics in sorted(root.glob("exp_*/metrics.json")):
        try:
            payload = json.loads(metrics.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("status") not in {"pending"}:
            reports.append(payload)
    if not reports:
        return {}
    return save_reference_figures(reports, root)
