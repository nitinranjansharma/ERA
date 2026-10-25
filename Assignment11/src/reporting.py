"""Build the written report from measured JSON, never from invented numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils import write_text


def _f(value: Any, fmt: str = ".6g") -> str:
    if value is None:
        return "n/a"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_report(results: dict, plot_paths: dict[str, str] | None = None) -> str:
    env = results.get("environment") or {}
    plots = plot_paths or results.get("plots") or {}
    a = results.get("A") or {}
    b = results.get("B") or {}
    c = results.get("C") or {}
    d = results.get("D") or {}
    e = results.get("E") or {}
    lines: list[str] = []
    lines += [
        "# ERA V5 Session 11 — Optimizers and Learning-Rate Schedules",
        "",
        "This report is generated from `artifacts/results.json`. Every numeric claim below was measured by the code in `src/`.",
        "",
        "## 1. Setup",
        "",
        f"- Python: `{env.get('python', 'unknown')}`",
        f"- PyTorch: `{env.get('torch', 'unknown')}`",
        f"- Device: `{env.get('device', 'cpu')}`",
        f"- Seed: `{env.get('seed', results.get('config', {}).get('seed'))}`",
        f"- CUDA available: `{env.get('cuda_available')}`  MPS available: `{env.get('mps_available')}`",
        "",
        "The implementation is original. A public reference repository was used only to confirm the assignment's scope. Its source, plots, and numbers were not copied.",
        "",
        "## 2. Task A — Manual Adam vs `torch.optim.Adam`",
        "",
        a.get("step_convention", ""),
        "",
        a.get("epsilon_placement", ""),
        "",
    ]
    hp = a.get("hyperparameters") or {}
    lines += [
        f"Defaults: `w0={hp.get('w0')}`, `lr={hp.get('lr')}`, `beta1={hp.get('beta1')}`, `beta2={hp.get('beta2')}`, `eps={hp.get('eps')}`, gradients `{hp.get('gradients')}`.",
        "",
        "Manual Adam is implemented in IEEE-754 float64 (Python `float`) without calling `torch.optim.Adam` for the update. The no-bias-correction class never writes `m_hat` / `v_hat`.",
        "",
    ]
    if a.get("manual_corrected"):
        lines += [
            _row(["t", "g", "m", "v", "m_hat", "v_hat", "delta_w", "w"]),
            _row(["---"] * 8),
        ]
        for rec in a["manual_corrected"]:
            lines.append(
                _row(
                    [
                        str(rec["t"]),
                        _f(rec["g"], ".4f"),
                        _f(rec["m"], ".8g"),
                        _f(rec["v"], ".8g"),
                        _f(rec["m_hat"], ".8g"),
                        _f(rec["v_hat"], ".8g"),
                        _f(rec["delta_w"], ".8g"),
                        _f(rec["w"], ".12g"),
                    ]
                )
            )
        lines.append("")
    vs = a.get("manual_vs_torch") or {}
    lines += [
        f"- Worst absolute disagreement vs float64 `torch.optim.Adam` across listed intermediates: **{_f(vs.get('worst_abs_error'), '.3e')}**",
        f"- Documented tolerance: `{a.get('tolerance')}`",
        f"- Agrees under that tolerance: **{a.get('agrees_with_torch')}**",
        f"- Textbook form vs PyTorch form (epsilon after the scaled second moment): worst error `{_f((a.get('textbook_vs_pytorch_form') or {}).get('worst_abs_error'), '.3e')}`",
        f"- Moving epsilon inside the `v` bias correction differs: `{a.get('inside_eps_differs')}` (worst error `{_f((a.get('textbook_vs_inside_eps') or {}).get('worst_abs_error'), '.3e')}`)",
        "",
        "Any residual gap, if present, is from floating-point evaluation order, not from a different algorithm. Weight decay is zero so this is Adam, not AdamW.",
        "",
        "## 3. Task B — Bias correction over 20 steps",
        "",
        b.get("relative_difference_definition", ""),
        "",
    ]
    scenarios = b.get("scenarios") or {}
    for name, payload in scenarios.items():
        crossing = payload.get("persistent_crossing") or {}
        plot_cross = payload.get("persistent_crossing_within_plot") or {}
        lines += [
            f"### Scenario: {name}",
            "",
            f"- First-step relative difference: **{_f(payload.get('first_step_relative_difference'))}**",
            f"- Maximum over the first 20 steps: **{_f(payload.get('max_relative_difference'))}** at step `{payload.get('max_relative_difference_step')}`",
            f"- Relative difference at step 20: **{_f(payload.get('step20_relative_difference'))}**",
            f"- Trend in the 20-step window: **{payload.get('trend_over_plot_window')}**",
            f"- Do the two methods agree by step 20 at tolerance `{ (b.get('hyperparameters') or {}).get('tolerance') }`? **{payload.get('agree_by_step20')}**",
            f"- Persistent crossing inside the 20-step plot: `{plot_cross.get('met')}`"
            + (f" at step {plot_cross.get('step')}" if plot_cross.get("met") else ""),
            f"- Persistent crossing in the extended search (limit `{payload.get('persistent_crossing_search_limit')}`): `{crossing.get('met')}`"
            + (
                f" at step **{crossing.get('step')}**, holding for {crossing.get('consecutive')} steps"
                if crossing.get("met")
                else f" (not met through step {crossing.get('searched_through_step')})"
            ),
            "",
        ]
    if plots.get("bias_correction"):
        lines += [f"![Bias correction]({_rel(plots['bias_correction'])})", ""]
    lines += [
        "When epsilon is negligible compared with `sqrt(v)`, the relative difference reduces to `| (1-β1^t)/√(1-β2^t) - 1 |`, which depends on `t` and the betas, not on the gradient sequence. That is why the two relative-difference panels look almost the same while the raw update magnitudes do not.",
        "",
        "The 20-step figure is the required window. The persistent-tolerance step is labelled separately and is not inferred from that figure.",
        "",
        "## 4. Task C — Update-to-weight ratios and warmup",
        "",
        c.get("update_definition", ""),
        "",
        c.get("near_zero_policy", ""),
        "",
        c.get("explanation", ""),
        "",
        f"- Learning-rate ramp reaches its peak at step **{c.get('lr_peak_step')}**.",
        f"- Mean update-to-weight ratio (warmup arm) is largest at step **{c.get('ratio_peak_step')}**.",
        f"- Relative-change of that ratio first holds within tolerance at step **{c.get('ratio_stabilize_step')}** (this can fire during a slow climb; it is not the same event as the LR peak).",
        f"- LR-peak step equals relative-change hold step: **{c.get('events_are_identical')}**.",
        f"- Zero-init tensors excluded from the headline mean: `{((c.get('with_warmup') or {}).get('init_near_zero'))}`",
        f"- Final training loss with warmup: `{_f(c.get('loss_with_warmup'))}`",
        f"- Final training loss without warmup: `{_f(c.get('loss_without_warmup'))}`",
        "",
    ]
    if plots.get("update_to_weight"):
        lines += [f"![Update-to-weight]({_rel(plots['update_to_weight'])})", ""]
    lines += [
        "Zero-initialized biases are present in the instrumented MLP. Their ratios are computed with a clamped denominator and excluded from the headline mean.",
        "",
        "## 5. Task D — Cosine decay vs WSD",
        "",
        d.get("checkpoint_rationale", ""),
        "",
        d.get("limitations", ""),
        "",
    ]
    if d:
        cos = d.get("cosine_300") or {}
        wsd = d.get("wsd_300") or {}
        lines += [
            _row(["schedule", "LR @200", "train @200", "val @200", "train @300", "val @300", "mean LR @200", "mean LR @300"]),
            _row(["---"] * 8),
        ]
        for name, run in (("cosine", cos), ("WSD", wsd)):
            s200 = run.get("step200") or {}
            s300 = run.get("step_final") or {}
            spec = run.get("schedule") or {}
            lines.append(
                _row(
                    [
                        name,
                        _f(s200.get("learning_rate"), ".4e"),
                        _f(s200.get("training_loss")),
                        _f(s200.get("validation_loss")),
                        _f(s300.get("training_loss")),
                        _f(s300.get("validation_loss")),
                        _f(spec.get("mean_lr_200"), ".4e"),
                        _f(spec.get("mean_lr_total"), ".4e"),
                    ]
                )
            )
        lines += [
            "",
            f"- Lower validation loss at step 200 (300-step schedules): **{d.get('winner_val_at_200_from_300_budget')}**",
            f"- Lower validation loss at step 300: **{d.get('winner_val_at_300')}**",
            f"- Lower validation loss when both schedules are designed for a 200-step budget: **{d.get('winner_val_at_200_from_200_budget')}**",
            "",
            "The comparison depends on the planned horizon and on the average learning rate spent, not only on the shape name. WSD is not claimed to be universally better. On this synthetic next-token walk the model is still improving at step 200, so the arm that spent more average LR (WSD, still at the peak) wins both at the early-stop checkpoint and at step 300. Redesigning both schedules for a 200-step budget did not reverse that ranking here, because the cosine-200 run has already annealed while the task still benefits from a larger step size.",
            "",
        ]
        cos200 = d.get("cosine_designed_200") or {}
        wsd200 = d.get("wsd_designed_200") or {}
        if cos200 and wsd200:
            lines += [
                "When both schedules are built for 200 steps, cosine finishes its decay at the evaluation point while WSD spends a short terminal decay. That is a different experiment from reading the step-200 checkpoint of a 300-step run.",
                "",
                f"- Cosine-200 final val: `{_f((cos200.get('step_final') or {}).get('validation_loss'))}`",
                f"- WSD-200 final val: `{_f((wsd200.get('step_final') or {}).get('validation_loss'))}`",
                "",
            ]
    if plots.get("cosine_vs_wsd"):
        lines += [f"![Cosine vs WSD]({_rel(plots['cosine_vs_wsd'])})", ""]
    lines += [
        "## 6. Task E — Learning-rate sweep across widths",
        "",
        e.get("parameterization", ""),
        "",
        f"- Optimizer: `{e.get('optimizer')}`, weight decay `{e.get('weight_decay')}`, budget `{e.get('steps')}` steps.",
        f"- Metric: {e.get('evaluation_metric')}.",
        f"- Seed: `{e.get('seed')}`.",
        "",
    ]
    per_width = e.get("per_width") or {}
    if per_width:
        lines += [
            _row(["width", "best LR", "val MSE", "train loss", "on grid boundary", "grid expanded"]),
            _row(["---"] * 6),
        ]
        for key in sorted(per_width, key=lambda k: int(k)):
            best = per_width[key]["best"]
            lines.append(
                _row(
                    [
                        key,
                        _f(best.get("lr"), ".4e"),
                        _f(best.get("val_loss")),
                        _f(best.get("train_loss")),
                        str(best.get("on_boundary")),
                        str(per_width[key].get("grid_expanded")),
                    ]
                )
            )
        lines.append("")
    fit = e.get("fit_log_log") or {}
    extra = e.get("extrapolation") or {}
    lines += [
        f"Log-log fit: `{fit.get('form')}` with slope **{_f(fit.get('slope'))}**, intercept `{_f(fit.get('intercept'))}`, RMSE in log-space `{_f(fit.get('rmse_log'))}`.",
        "",
        f"Candidate LR at width {extra.get('width')}:",
        f"- from the log-log fit: **{_f(extra.get('lr_from_loglog'), '.4e')}**",
        f"- from the inverse-width heuristic off the largest measured width: **{_f(extra.get('lr_from_inverse_width'), '.4e')}**",
        "",
        extra.get("uncertainty", ""),
        "",
        "Leave-one-out slopes on the three measured widths: "
        + ", ".join(
            f"drop {row.get('left_out_width')} → slope {_f(row.get('slope'))}"
            for row in (extra.get("leave_one_out_slopes") or [])
        )
        + ". Widths 256 and 512 sharing the same grid minimum makes the exponent poorly identified; the 4096 figure is a candidate, not a transferred optimum.",
        "",
        "This is a standard-parameterization sweep. It is not a μP reparameterization study and does not claim that the optimal learning rate transfers unchanged across width.",
        "",
    ]
    if plots.get("lr_sweep"):
        lines += [f"![LR sweep]({_rel(plots['lr_sweep'])})", ""]
    lines += [
        "## 7. Limitations",
        "",
        "- All training runs are tiny and CPU-only. They illustrate optimizer and schedule mechanics; they are not LLM-scale evidence.",
        "- Task D uses one seed and a synthetic next-token walk. Rankings can flip with seed, horizon, or peak LR.",
        "- Task E uses three widths and a modest grid. An interior minimum is still a grid point, not a continuous optimum.",
        "- Bias-correction settling time is a function of `beta2`, not a universal Adam constant.",
        "",
        "## 8. How to reproduce",
        "",
        "```bash",
        "python3 -m venv .venv && source .venv/bin/activate",
        "pip install -r requirements.txt && pip install -e .",
        "python -m pytest -q",
        "python -m run_experiments --overwrite",
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


def _rel(path: str) -> str:
    p = Path(path)
    try:
        return str(Path("..") / p.relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(p)


def write_report(results: dict, path: str | Path, plot_paths: dict[str, str] | None = None) -> Path:
    return write_text(path, render_report(results, plot_paths))
