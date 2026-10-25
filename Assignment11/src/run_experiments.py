#!/usr/bin/env python3
"""CLI for Session 11 experiments.

    python -m run_experiments
    python -m run_experiments --tasks A,B --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import run_all_tasks  # noqa: E402
from plotting import plot_all  # noqa: E402
from reporting import write_report  # noqa: E402
from utils import (  # noqa: E402
    ensure_dir,
    environment_record,
    load_config,
    project_root,
    resolve_device,
    set_seed,
    utc_stamp,
    write_csv,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERA Session 11 optimizer experiments")
    parser.add_argument("--config", default=str(project_root() / "configs" / "default.yaml"))
    parser.add_argument("--tasks", default="A,B,C,D,E", help="Comma-separated task letters")
    parser.add_argument("--output-dir", default=None, help="Override artifacts directory")
    parser.add_argument("--overwrite", action="store_true", help="Refresh the stable artifact copies")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args(argv)


def _flatten_adam_rows(task_a: dict) -> list[dict]:
    rows = []
    for rec in task_a.get("manual_corrected") or []:
        rows.append({"source": "manual_corrected", **rec})
    for rec in task_a.get("manual_uncorrected") or []:
        rows.append({"source": "manual_uncorrected", **rec})
    for rec in task_a.get("torch_adam") or []:
        rows.append({"source": "torch_adam", **rec})
    return rows


def _flatten_bias_rows(task_b: dict) -> list[dict]:
    rows = []
    for name, payload in (task_b.get("scenarios") or {}).items():
        for rec in payload.get("plot_rows") or []:
            rows.append({"scenario": name, **rec})
    return rows


def _flatten_ratio_rows(task_c: dict) -> list[dict]:
    rows = []
    for arm in ("with_warmup", "without_warmup"):
        for rec in (task_c.get(arm) or {}).get("logs") or []:
            rows.append(
                {
                    "arm": arm,
                    "global_step": rec["global_step"],
                    "learning_rate": rec["learning_rate"],
                    "parameter_norm": rec["parameter_norm"],
                    "update_norm": rec["update_norm"],
                    "update_to_weight_ratio": rec["update_to_weight_ratio"],
                    "gradient_norm": rec["gradient_norm"],
                    "training_loss": rec["training_loss"],
                    "n_near_zero_weight": rec["n_near_zero_weight"],
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    device = resolve_device(cfg.get("device", "cpu"))
    tasks = [part.strip().upper() for part in args.tasks.split(",") if part.strip()]

    artifacts = Path(args.output_dir) if args.output_dir else project_root() / cfg.get("paths", {}).get("artifacts", "artifacts")
    run_dir = ensure_dir(artifacts / "runs" / utc_stamp())
    plots_dir = ensure_dir(run_dir / "plots")
    logs_dir = ensure_dir(run_dir / "logs")

    results = run_all_tasks(cfg, tasks=tasks)
    results["environment"] = environment_record(seed, device)
    results["run_dir"] = str(run_dir)

    if "A" in results:
        write_csv(logs_dir / "adam_steps.csv", _flatten_adam_rows(results["A"]))
    if "B" in results:
        write_csv(logs_dir / "bias_correction.csv", _flatten_bias_rows(results["B"]))
    if "C" in results:
        write_csv(logs_dir / "update_to_weight.csv", _flatten_ratio_rows(results["C"]))
    if "D" in results:
        hist = []
        for arm in ("cosine_300", "wsd_300"):
            for rec in results["D"][arm]["history"]:
                hist.append({"arm": arm, **rec})
        write_csv(logs_dir / "schedules.csv", hist)
    if "E" in results:
        cells = []
        for payload in results["E"]["per_width"].values():
            for cell in payload["cells"]:
                cells.append(cell)
        write_csv(logs_dir / "width_lr_sweep.csv", cells)

    plot_paths = {}
    if not args.skip_plots:
        plot_paths = plot_all(results, plots_dir)
    results["plots"] = plot_paths

    write_json(run_dir / "results.json", results)
    write_json(run_dir / "config.json", cfg)

    report_path = run_dir / "final_report.md"
    if not args.skip_report:
        write_report(results, report_path, plot_paths)

    write_stable = args.overwrite or args.output_dir is None
    if write_stable:
        stable_plots = ensure_dir(artifacts / "plots")
        stable_logs = ensure_dir(artifacts / "logs")
        write_json(artifacts / "results.json", results)
        import shutil

        for src in plots_dir.glob("*.png"):
            shutil.copy2(src, stable_plots / src.name)
        for src in logs_dir.glob("*.csv"):
            shutil.copy2(src, stable_logs / src.name)
        stable_plot_paths = {k: str(stable_plots / Path(v).name) for k, v in plot_paths.items()}
        (project_root() / "report").mkdir(exist_ok=True)
        write_report(results, project_root() / "report" / "final_report.md", stable_plot_paths)

    print(f"wrote {run_dir / 'results.json'}")
    for name, path in plot_paths.items():
        print(f"plot {name}: {path}")
    if "A" in results:
        vs = results["A"]["manual_vs_torch"]["worst_abs_error"]
        print(f"task A worst abs error vs torch: {vs:.3e} agrees={results['A']['agrees_with_torch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
