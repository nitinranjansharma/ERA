"""CSV/JSON export, plot generation, and experiment comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from helpers.config import ensure_dir
from helpers.metrics import (
    REQUIRED_REPORT_FIELDS,
    bytes_to_mib,
    validate_report,
)

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


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def plot_loss_vs_tokens(histories: Mapping[str, Sequence[Mapping[str, Any]]], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, history in histories.items():
        tokens = [row["tokens"] for row in history if "tokens" in row and "train_loss" in row]
        losses = [row["train_loss"] for row in history if "tokens" in row and "train_loss" in row]
        if tokens:
            ax.plot(tokens, losses, label=f"{name} train")
        val_t = [row["tokens"] for row in history if row.get("val_loss") is not None]
        val_l = [row["val_loss"] for row in history if row.get("val_loss") is not None]
        if val_t:
            ax.plot(val_t, val_l, linestyle="--", label=f"{name} val")
    ax.set_xlabel("Training target tokens")
    ax.set_ylabel("Loss")
    ax.set_title("Loss versus training tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_throughput(reports: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    labels = [str(r.get("experiment", "?")) + "/" + str(r.get("variant", "?")) for r in reports]
    values = [float(r["tokens_per_second"]) if r.get("tokens_per_second") is not None else 0.0 for r in reports]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Training throughput")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_peak_memory(reports: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    labels = [str(r.get("experiment", "?")) + "/" + str(r.get("variant", "?")) for r in reports]
    alloc = [bytes_to_mib(r.get("peak_allocated_bytes")) or 0.0 for r in reports]
    reserved = [bytes_to_mib(r.get("peak_reserved_bytes")) or 0.0 for r in reports]
    import numpy as np

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
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def export_run_artifacts(
    output_dir: str | Path,
    reports: Sequence[Mapping[str, Any]],
    histories: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, str]:
    root = ensure_dir(output_dir)
    plots = ensure_dir(root / "plots")
    json_path = write_json(root / "results.json", list(reports))
    csv_path = write_comparison_table(root / "results.csv", reports)
    table_path = root / "results.md"
    table_path.write_text(markdown_table(reports))
    artifacts = {
        "results_json": str(json_path),
        "results_csv": str(csv_path),
        "results_md": str(table_path),
    }
    if histories:
        artifacts["loss_plot"] = str(plot_loss_vs_tokens(histories, plots / "loss_vs_tokens.png"))
    if reports:
        artifacts["throughput_plot"] = str(plot_throughput(reports, plots / "tokens_per_second.png"))
        artifacts["memory_plot"] = str(plot_peak_memory(reports, plots / "peak_memory.png"))
        named = {}
        for report in reports:
            key = f"{report.get('experiment')}_{report.get('variant')}"
            named[key] = report
        artifacts["four_run_panel"] = str(plot_four_run_panel(named, plots / "results.png"))
        if "B_euler" in named and "B_midpoint" in named:
            artifacts["euler_vs_midpoint"] = str(
                plot_euler_vs_midpoint(named["B_euler"], named["B_midpoint"], plots / "euler_vs_midpoint.png")
            )
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
    reports: Mapping[str, Mapping[str, Any]],
    path: str | Path,
    keys: Sequence[str] | None = None,
) -> Path:
    """Loss / peak memory / throughput panel used by the proven S13 write-up."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(keys or [k for k in ("A", "B_euler", "B_midpoint", "C") if k in reports])
    if not keys:
        keys = list(reports)
    color = {"A": "#5B7FDB", "B_euler": "#E0823D", "B_midpoint": "#3FA37A", "C": "#9A62C4",
             "baseline": "#5B7FDB", "euler": "#E0823D", "midpoint": "#3FA37A", "pushed": "#9A62C4"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for k in keys:
        hist = reports[k].get("extra", {}).get("history") or []
        ax[0].plot(
            [row.get("tokens", 0) / 1e6 for row in hist],
            [row.get("smooth_loss", row.get("train_loss")) for row in hist],
            color=color.get(k, color.get(str(reports[k].get("variant")), "#444")),
            lw=1.5,
            label=k,
        )
    ax[0].set(xlabel="millions of tokens", ylabel="loss", title="Did it learn?")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, alpha=0.25)

    def _mb(report: Mapping[str, Any]) -> float:
        raw = report.get("peak_allocated_bytes")
        return (raw / 2**20) if raw else 0.0

    mem = [_mb(reports[k]) for k in keys]
    tps = [float(reports[k].get("tokens_per_second") or 0) for k in keys]
    ax[1].bar(range(len(keys)), mem, color=[color.get(k, "#888") for k in keys], width=0.6)
    ax[1].set_xticks(range(len(keys)))
    ax[1].set_xticklabels(keys, fontsize=7)
    ax[1].set_title("Peak GPU memory (MB)")
    ax[2].bar(range(len(keys)), tps, color=[color.get(k, "#888") for k in keys], width=0.6)
    ax[2].set_xticks(range(len(keys)))
    ax[2].set_xticklabels(keys, fontsize=7)
    ax[2].set_title("Speed (tokens per second)")
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eh = euler.get("extra", {}).get("history") or []
    mh = midpoint.get("extra", {}).get("history") or []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([r.get("step") for r in eh], [r.get("smooth_loss", r.get("train_loss")) for r in eh],
               color="#E0823D", label="euler")
    ax[0].plot([r.get("step") for r in mh], [r.get("smooth_loss", r.get("train_loss")) for r in mh],
               color="#3FA37A", label="midpoint")
    ax[0].set(xlabel="step", ylabel="loss", title="Euler vs midpoint")
    ax[0].legend()
    n = min(len(eh), len(mh))
    if n:
        gap = [
            (mh[i].get("smooth_loss", mh[i].get("train_loss")) or 0)
            - (eh[i].get("smooth_loss", eh[i].get("train_loss")) or 0)
            for i in range(n)
        ]
        ax[1].plot([eh[i].get("step") for i in range(n)], gap, color="#555")
        ax[1].axhline(0.0, color="#aaa", lw=0.8)
        ax[1].set(xlabel="step", ylabel="midpoint − euler", title="Gap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
