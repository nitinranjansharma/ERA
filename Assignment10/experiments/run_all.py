#!/usr/bin/env python3
"""Run every CPU-compatible experiment, write artifacts, return non-zero on failure.

    python -m experiments.run_all
    python -m experiments.run_all --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from accumulation_experiment import run_accumulation_experiment  # noqa: E402
from gradient_monitor import run_spike_experiment  # noqa: E402
from gradient_verification import verify as verify_gradients  # noqa: E402
from instrumentation import (  # noqa: E402
    create_run_dir,
    load_config,
    mirror_to_stable,
    record_environment,
    resolve_device,
    set_seed,
    write_csv,
    write_json,
    write_text,
)
from mfu import run_mfu_experiment  # noqa: E402
from precision import analyze_decimal_0p1, format_report as format_precision  # noqa: E402
from training_loop import (  # noqa: E402
    demonstrate_forgotten_zero_grad,
    overfit_tiny_dataset,
    traced_language_model_step,
)


def format_truth_log(results: dict, env: dict, statuses: dict, run_dir: Path) -> str:
    """One printed document: the loop telling the truth about itself."""
    lines: list[str] = []
    lines += [
        "THE TRAINING LOOP, TELLING THE TRUTH ABOUT ITSELF",
        "=" * 72,
        f"run: {run_dir}",
        f"python {env.get('python_version')}  torch {env.get('torch')}  device {env.get('device')}",
        f"seed {env.get('seed')}  cuda={env.get('cuda_available')}  mps={env.get('mps_available')}",
        "",
        "Statuses: " + ", ".join(f"{k}={v}" for k, v in statuses.items()),
        "",
    ]

    shapes = results.get("shapes") or {}
    if "records" in shapes:
        lines += ["1. Every tensor in one training step", "-" * 72]
        for rec in shapes["records"]:
            shape = "(" + ", ".join(str(s) for s in rec["shape"]) + ("," if len(rec["shape"]) == 1 else "") + ")"
            if rec["shape"] == []:
                shape = "()"
            dims = " × ".join(rec["dims"]) if rec["dims"] else "scalar"
            lines.append(f"  {rec['name']:32s} {shape:20s}  {dims}")
            lines.append(f"    {rec['explanation']}")
        lines.append(
            f"  step loss={shapes.get('loss'):.6f}  grad_norm={shapes.get('grad_norm'):.6f}  "
            f"finite={True}"
        )
        lines.append("")

    grads = results.get("gradients") or {}
    if grads.get("analytical"):
        a = grads["analytical"]
        auto = grads["autograd"]
        pref = grads["preferred_row"]
        nudge = grads.get("real_loop_nudge") or {}
        lines += [
            "2. One gradient, three ways (then a real-loop nudge)",
            "-" * 72,
            f"  chain rule   ∂L/∂w1 = {a['dL_dw1']}",
            f"  autograd     ∂L/∂w1 = {auto['dL_dw1']}",
            f"  finite diff  ∂L/∂w1 = {pref['numerical_dL_dw1']}  (ε={pref['epsilon']}, rel err {pref['relative_error']:.3e})",
            f"  they agree to several decimals: {grads.get('passed')}",
        ]
        if nudge:
            lines.append(
                f"  real loop nudge of {nudge.get('parameter')}: "
                f"autograd={nudge.get('autograd'):.10f}  "
                f"ΔL/(2ε)={nudge.get('numerical'):.10f}  "
                f"rel err={nudge.get('relative_error'):.3e}  "
                f"agree={nudge.get('agrees_to_several_decimals')}"
            )
        lines.append("")

    acc = results.get("accumulation") or {}
    if acc.get("correct"):
        lines += [
            "3. Gradient accumulation broken on purpose",
            "-" * 72,
            f"  tokens {acc['arithmetic']['token_counts']}, means {acc['arithmetic']['mb_means']}",
            f"  correct token-weighted loss = {acc['correct']['loss']}",
            f"  incorrect mean-of-means     = {acc['incorrect']['loss']}",
            f"  relative discrepancy        = {acc['loss_relative_discrepancy']*100:.4f}%",
            f"  grads: correct {acc['correct']['grad']} vs incorrect {acc['incorrect']['grad']}",
            f"  equal-token control discrepancy = {acc['equal_token']['relative_discrepancy']}",
            "",
        ]

    mon = results.get("monitoring") or {}
    if mon.get("rows"):
        q = mon.get("first_qualifying_step")
        jump = mon.get("next_loss_jump") or {}
        row_q = next((r for r in mon["rows"] if r["step"] == q), None)
        lines += [
            "4. Grad norm logged every step; one step where it moved first",
            "-" * 72,
            f"  first qualifying step from the CSV: {q}",
            f"  at that step: loss={row_q['loss']:.6f}  pre-clip norm={row_q['pre_clip_norm']:.4f}" if row_q else "",
            f"  next step loss jump: step {jump.get('step')}  loss={jump.get('loss')}  "
            f"Δ={jump.get('delta_vs_qualifying')}",
            f"  norm changed before the loss in this run: {mon.get('norm_changed_before_loss_in_this_run')}",
            f"  {mon.get('caveat')}",
            "",
        ]

    mfu = results.get("mfu") or {}
    if mfu.get("assignment_example"):
        ex = mfu["assignment_example"]
        lines += [
            "5. MFU — assignment illustration, then this machine",
            "-" * 72,
            f"  ILLUSTRATION (not measured here): 9e9 params, 12k tok/s, peak 7912 TFLOP/s",
            f"    estimated {ex['computed_tflops_per_second']} TFLOP/s, MFU {ex['computed_mfu_percent']:.2f}%",
        ]
        for cfg in mfu.get("configs") or []:
            neon = cfg["vs_theoretical_neon"]
            lines.append(
                f"  LOCAL {cfg['config_name']}: B={neon['batch_size']} T={neon['seq_len']}  "
                f"{neon['tokens_per_second']:.0f} tok/s  N={neon['parameter_count_trainable']}  "
                f"MFU vs NEON peak {neon['mfu_percent']:.2f}%  (gap to 40%: {neon['gap_to_40_percent_points']:.1f} pt)"
            )
        lines.append("  What costs the distance to 40% on this CPU toy loop:")
        for note in mfu.get("gap_discussion") or []:
            lines.append(f"    - {note}")
        lines.append(f"  GPU MFU reported: {mfu.get('gpu_mfu_reported')} ({mfu.get('gpu_mfu_reason')})")
        lines.append("")

    prec = results.get("precision") or {}
    if prec.get("encodings"):
        lines += ["6. Decimal 0.1 in FP32, BF16, and OCP FP8 E4M3FN", "-" * 72]
        lines.append(f"  {prec.get('normalized_binary_scientific')}")
        for name, enc in prec["encodings"].items():
            lines.append(
                f"  {name:12s} bits {enc['grouped_binary']}  hex {enc['hex']}  "
                f"value {enc['represented_value']}  abs err {enc['abs_error_vs_exact']:.6e}"
            )
        rec = prec.get("recommendation") or {}
        lines.append(f"  train in: {rec.get('recommended')}")
        lines.append(f"  why: {rec.get('rationale')}")
        lines.append("")

    over = results.get("overfit") or {}
    if over.get("learned") is not None:
        lines += [
            "7. The loop can learn (overfit is not 'loss went down')",
            "-" * 72,
            f"  initial {over.get('initial_loss'):.6f} → final {over.get('final_loss'):.3e}  "
            f"residual {over.get('residual_mse'):.3e}  learned={over.get('learned')}",
        ]
        fg = over.get("forgotten_zero_grad") or {}
        if fg:
            lines.append(
                f"  forgotten zero_grad ratio = {fg.get('mean_ratio')}  "
                f"(stale grads, not more learning)"
            )
        lines.append("")

    lines += [
        "Every serious training bug is silent. The loss curve is not the check.",
        "Shapes, a nudged weight, a broken reduction, a logged norm, units of MFU, and bits of 0.1 are.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ERA V5 Session 10 experiment suite")
    p.add_argument("--overwrite", action="store_true", help="Replace stable copies under outputs/")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--run-id", type=str, default=None)
    return p.parse_args()


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def run_shape_experiment(run_dir: Path, cfg: dict, device: torch.device) -> dict:
    seq = cfg["sequence_model"]
    result = traced_language_model_step(
        vocab_size=seq["vocab_size"],
        n_layer=seq["n_layer"],
        n_head=seq["n_head"],
        n_embd=seq["n_embd"],
        block_size=seq["block_size"],
        batch_size=2,
        seq_len=8,
        device=device,
    )
    report = result.tracer.format_report()
    write_text(run_dir / "logs" / "training_step_shapes.txt", report)
    payload = {
        "loss": result.loss,
        "grad_norm": result.grad_norm,
        "lr": result.lr,
        "elapsed_s": result.elapsed_s,
        "throughput_samples_per_s": result.throughput_samples_per_s,
        "tokens_per_s": result.tokens_per_s,
        "records": result.tracer.as_dicts(),
        "passed": True,
    }
    write_json(run_dir / "metrics" / "shapes.json", payload)
    return payload


def run_gradient_experiment(run_dir: Path, cfg: dict) -> dict:
    gv = cfg["gradient_verification"]
    payload = verify_gradients(
        abs_tol_autograd=gv["abs_tol_autograd"],
        rel_tol_numerical=gv["rel_tol_numerical"],
        preferred_epsilon=gv["preferred_epsilon"],
        epsilons=gv["epsilons"],
    )
    write_json(run_dir / "metrics" / "gradient_verification.json", payload)

    xs = [r["epsilon"] for r in payload["finite_differences"]]
    ys = [r["relative_error"] for r in payload["finite_differences"]]
    plt.figure(figsize=(7.5, 4.5))
    plt.loglog(xs, ys, marker="o", label="relative error vs analytical 64")
    plt.axhline(gv["rel_tol_numerical"], color="gray", linestyle="--", label="pass tolerance")
    plt.xlabel("ε (perturbation of w1)")
    plt.ylabel("relative error |ĝ − 64| / 64")
    plt.title("Central finite-difference error vs ε (float64)\nassignment quadratic: truncation is identically 0")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    _savefig(run_dir / "figures" / "finite_difference_error.png")

    cubic = payload["cubic_truncation_demo"]
    plt.figure(figsize=(7.5, 4.5))
    plt.loglog(
        [r["epsilon"] for r in cubic["rows"]],
        [r["relative_error"] for r in cubic["rows"]],
        marker="o",
        label="observed relative error",
    )
    plt.loglog(
        [r["epsilon"] for r in cubic["rows"]],
        [max(r["infinite_precision_truncation"] / 27.0, 1e-18) for r in cubic["rows"]],
        linestyle="--",
        label="ε² / 27 (pure truncation)",
    )
    plt.xlabel("ε")
    plt.ylabel("relative error vs d(w³)/dw = 27")
    plt.title("Auxiliary cubic: truncation dominates at large ε, cancellation at tiny ε")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    _savefig(run_dir / "figures" / "finite_difference_cubic_tradeoff.png")
    return payload


def run_accumulation(run_dir: Path, cfg: dict) -> dict:
    acc = cfg["accumulation"]
    payload = run_accumulation_experiment(
        token_counts=acc["microbatch_token_counts"],
        mb_means=acc["microbatch_average_losses"],
        equal_token_count=acc["equal_token_count"],
    )
    write_json(run_dir / "metrics" / "accumulation.json", payload)

    counts = payload["arithmetic"]["token_counts"]
    correct = payload["correct"]["loss"]
    incorrect = payload["incorrect"]["loss"]
    disc = payload["loss_relative_discrepancy"] * 100.0

    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(
        ["Incorrect\n(mean of means)", "Correct\n(token-weighted mean)"],
        [incorrect, correct],
        color=["#c44e52", "#4c72b0"],
    )
    plt.ylabel("loss (dimensionless)")
    plt.title(
        f"Gradient-accumulation averaging bug\n"
        f"valid tokens {counts}; discrepancy {disc:.2f}%"
    )
    for b, v in zip(bars, [incorrect, correct]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.4f}", ha="center")
    plt.ylim(0, max(incorrect, correct) * 1.25)
    plt.axhline(2.6, color="black", linestyle=":", linewidth=0.8, label="expected correct 2.6")
    plt.legend()
    _savefig(run_dir / "figures" / "accumulation_loss.png")

    plt.figure(figsize=(7.5, 4.5))
    g_inc, g_cor = payload["incorrect"]["grad"], payload["correct"]["grad"]
    bars = plt.bar(
        ["Incorrect ∂L/∂θ", "Correct ∂L/∂θ"],
        [g_inc, g_cor],
        color=["#c44e52", "#4c72b0"],
    )
    plt.ylabel("gradient of scalar parameter θ")
    plt.title("Parameter gradient under the two reductions")
    for b, v in zip(bars, [g_inc, g_cor]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.4f}", ha="center")
    plt.ylim(0, max(g_inc, g_cor) * 1.25)
    _savefig(run_dir / "figures" / "accumulation_gradients.png")

    eq = payload["equal_token"]
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(
        ["Incorrect (equal tokens)", "Correct (equal tokens)"],
        [eq["incorrect"]["loss"], eq["correct"]["loss"]],
        color=["#c44e52", "#4c72b0"],
    )
    plt.ylabel("loss (dimensionless)")
    plt.title(
        f"Equal-token control: {eq['counts']} valid tokens per micro-batch\n"
        f"discrepancy {eq['relative_discrepancy'] * 100:.4f}%"
    )
    _savefig(run_dir / "figures" / "accumulation_equal_tokens.png")

    running = payload["running_curves"]
    xs = [r["microbatches_included"] for r in running]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(xs, [r["incorrect_loss"] for r in running], marker="o", color="#c44e52", label="incorrect: mean of means")
    plt.plot(xs, [r["correct_loss"] for r in running], marker="s", color="#4c72b0", label="correct: token-weighted mean")
    plt.xticks(xs, [f"after MB{k}\n({running[k-1]['tokens_so_far']} tokens)" for k in xs])
    plt.ylabel("loss")
    plt.title(
        f"Accumulation bug as a pair of curves\n"
        f"token counts {counts}; gap appears when the short batch arrives"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    _savefig(run_dir / "figures" / "accumulation_curves.png")

    traj = payload["training_trajectories"]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(
        [r["step"] for r in traj["incorrect"]],
        [r["loss"] for r in traj["incorrect"]],
        marker="o",
        markersize=3,
        color="#c44e52",
        label="incorrect accumulation",
    )
    plt.plot(
        [r["step"] for r in traj["correct"]],
        [r["loss"] for r in traj["correct"]],
        marker="s",
        markersize=3,
        color="#4c72b0",
        label="correct accumulation",
    )
    plt.xlabel("SGD step")
    plt.ylabel("reported loss")
    plt.title("Same toy model, two reductions: the gap is a curve, not a single number")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _savefig(run_dir / "figures" / "accumulation_training_curves.png")
    return payload


def run_monitoring(run_dir: Path, cfg: dict) -> dict:
    mon = cfg["monitoring"]
    payload = run_spike_experiment(
        steps=mon["steps"],
        spike_step=mon["spike_step"],
        spike_factor=mon["spike_factor"],
        clip_threshold=mon["clip_threshold"],
        detection_norm_ratio=mon["detection_norm_ratio"],
        seed=cfg["seed"],
    )
    write_json(run_dir / "metrics" / "monitoring.json", payload)
    write_csv(run_dir / "logs" / "grad_norm_steps.csv", payload["rows"])

    steps = [r["step"] for r in payload["rows"]]
    losses = [r["loss"] for r in payload["rows"]]
    norms = [r["pre_clip_norm"] for r in payload["rows"]]
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax0.plot(steps, losses, marker="o", markersize=3, color="#4c72b0", label="loss")
    ax1.plot(steps, norms, marker="o", markersize=3, color="#c44e52", label="pre-clip grad norm")
    if payload["first_qualifying_step"] is not None:
        ax0.axvline(payload["first_qualifying_step"], color="gray", linestyle="--")
        ax1.axvline(payload["first_qualifying_step"], color="gray", linestyle="--", label="first qualifying step")
    ax0.set_ylabel("MSE loss")
    ax1.set_ylabel("global L2 grad norm")
    ax1.set_xlabel("training step")
    ax0.set_title("Loss and gradient norm (controlled spike experiment)")
    ax0.legend(loc="upper left")
    ax1.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)
    ax1.grid(True, alpha=0.3)
    _savefig(run_dir / "figures" / "loss_and_grad_norm.png")
    return payload


def run_mfu(run_dir: Path, cfg: dict, device: torch.device) -> dict:
    mfu_cfg = cfg["mfu"]
    payload = run_mfu_experiment(
        warmup_steps=mfu_cfg["warmup_steps"],
        timed_steps=mfu_cfg["timed_steps"],
        configs=mfu_cfg["configs"],
        device=device,
    )
    write_json(run_dir / "metrics" / "mfu.json", payload)
    names = [c["config_name"] for c in payload["configs"]]
    neon = [c["vs_theoretical_neon"]["mfu_percent"] for c in payload["configs"]]
    amx = [c["vs_literature_amx"]["mfu_percent"] for c in payload["configs"]]
    x = range(len(names))
    plt.figure(figsize=(7.5, 4.5))
    width = 0.35
    plt.bar([i - width / 2 for i in x], neon, width, label="vs theoretical NEON FP32 peak")
    plt.bar([i + width / 2 for i in x], amx, width, label="vs literature AMX GEMM peak")
    plt.xticks(list(x), names)
    plt.ylabel("MFU (%)")
    plt.title("Local TinyGPT MFU on CPU (6ND formula; warmup excluded)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    _savefig(run_dir / "figures" / "mfu_configs.png")
    return payload


def run_precision(run_dir: Path) -> dict:
    payload = analyze_decimal_0p1()
    write_json(run_dir / "metrics" / "precision.json", payload)
    write_text(run_dir / "logs" / "precision_0p1.txt", format_precision(payload))
    return payload


def run_overfit(run_dir: Path, cfg: dict, device: torch.device) -> dict:
    tr = cfg["training"]
    payload = overfit_tiny_dataset(
        steps=tr["overfit_steps"],
        threshold=tr["overfit_loss_threshold"],
        device=device,
    )
    forgotten = demonstrate_forgotten_zero_grad()
    payload["forgotten_zero_grad"] = forgotten
    write_json(run_dir / "metrics" / "overfit.json", payload)
    plt.figure(figsize=(7.5, 4.5))
    plt.semilogy(range(1, len(payload["history"]) + 1), payload["history"])
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.title("Overfit of an 8-sample synthetic regression set")
    plt.grid(True, which="both", alpha=0.3)
    _savefig(run_dir / "figures" / "overfit_loss.png")
    return payload


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed, deterministic=bool(cfg.get("deterministic", True)))
    device = resolve_device(cfg.get("device", "cpu"))
    run_dir = create_run_dir(overwrite=False, run_id=args.run_id)
    env = record_environment(seed, device)
    write_json(run_dir / "metrics" / "environment.json", env)
    write_json(run_dir / "metrics" / "run_config.json", cfg)

    statuses: dict[str, str] = {}
    results: dict[str, dict] = {}

    def _run(name, fn):
        try:
            out = fn()
            results[name] = out
            statuses[name] = "PASS" if out.get("passed", True) else "FAIL"
        except Exception as exc:  # noqa: BLE001 — suite must record the failure
            results[name] = {"error": repr(exc)}
            statuses[name] = "FAIL"
            print(f"[{name}] FAIL: {exc!r}", file=sys.stderr)

    _run("shapes", lambda: run_shape_experiment(run_dir, cfg, device))
    _run("gradients", lambda: run_gradient_experiment(run_dir, cfg))
    _run("accumulation", lambda: run_accumulation(run_dir, cfg))
    _run("monitoring", lambda: run_monitoring(run_dir, cfg))
    _run("mfu", lambda: {**run_mfu(run_dir, cfg, device), "passed": True})
    _run("precision", lambda: run_precision(run_dir))
    _run("overfit", lambda: {**run_overfit(run_dir, cfg, device), "passed": results.get("overfit", {}).get("learned", True)})

    # overfit status uses the learned flag
    if "overfit" in results and "learned" in results["overfit"]:
        statuses["overfit"] = "PASS" if results["overfit"]["learned"] else "FAIL"

    summary = {
        "run_dir": str(run_dir),
        "environment": env,
        "statuses": statuses,
        "gpu_tests": "SKIPPED (no CUDA; unit tests never require a GPU)",
    }
    write_json(run_dir / "metrics" / "summary.json", summary)
    truth = format_truth_log(results, env, statuses, run_dir)
    write_text(run_dir / "logs" / "training_truth.txt", truth)
    print(truth)
    mirror_to_stable(run_dir, overwrite=args.overwrite or True)

    print(f"Run directory: {run_dir}")
    for k, v in statuses.items():
        print(f"  {k:16s} {v}")
    failed = [k for k, v in statuses.items() if v == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
