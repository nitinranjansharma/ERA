"""Task A–E experiment runners. Each function returns a JSON-serialisable dict."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from adam_manual import (
    ScalarAdamNoBiasCorrection,
    closed_form_uncorrected_ratio,
    run_manual_adam,
    verify_against_pytorch,
)
from data import classification_tensors, cycle_dataset, regression_from_teacher, repeating_stream
from metrics import (
    NonFiniteError,
    assert_grads_finite,
    describe_trend,
    first_persistent_crossing,
    global_grad_norm,
    magnitude_ratio,
    missing_grad_record,
    relative_difference,
    require_finite,
    tensor_norm,
    update_to_weight_ratio,
)
from model import InstrumentedMLP, StandardWidthMLP, TinySequenceLM, named_trainable, snapshot_params
from schedules import ScheduleSpec, apply_lr, make_schedule
from utils import resolve_device, resolve_dtype, set_seed


def run_task_a(cfg: dict) -> dict:
    adam_cfg = cfg["adam"]
    payload = verify_against_pytorch(
        gradients=list(adam_cfg["gradients"]),
        w0=float(adam_cfg["w0"]),
        lr=float(adam_cfg["lr"]),
        beta1=float(adam_cfg["beta1"]),
        beta2=float(adam_cfg["beta2"]),
        eps=float(adam_cfg["eps"]),
        tolerance=float(adam_cfg.get("match_tolerance", 1e-12)),
    )
    # First-step closed form, used by tests and the report.
    g0 = float(adam_cfg["gradients"][0])
    beta1 = float(adam_cfg["beta1"])
    beta2 = float(adam_cfg["beta2"])
    m1 = (1.0 - beta1) * g0
    v1 = (1.0 - beta2) * g0 * g0
    payload["first_step_closed_form"] = {
        "g": g0,
        "m": m1,
        "v": v1,
        "m_hat": m1 / (1.0 - beta1),
        "v_hat": v1 / (1.0 - beta2),
    }
    return payload


def _run_bias_pair(
    gradients: Sequence[float],
    w0: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    relative_eps: float,
) -> list[dict]:
    on = run_manual_adam(gradients, w0, lr, beta1, beta2, eps, bias_correction=True)
    off = ScalarAdamNoBiasCorrection(w0=w0, lr=lr, beta1=beta1, beta2=beta2, eps=eps).run(gradients)
    rows = []
    for a, b in zip(on, off):
        rel = relative_difference(b.delta_w, a.delta_w, eps=relative_eps)
        mag = magnitude_ratio(b.delta_w, a.delta_w, eps=relative_eps)
        rows.append(
            {
                "t": a.t,
                "g": a.g,
                "delta_corrected": a.delta_w,
                "delta_uncorrected": b.delta_w,
                "abs_difference": abs(a.delta_w - b.delta_w),
                "relative_difference": rel["value"],
                "relative_clamped": rel["clamped"],
                "magnitude_ratio_off_over_on": mag["value"],
                "w_corrected": a.w,
                "w_uncorrected": b.w,
                "m_hat": a.m_hat,
                "v_hat": a.v_hat,
                "m_raw_uncorrected": b.m,
                "v_raw_uncorrected": b.v,
            }
        )
    return rows


def _summarise_bias_rows(rows: list[dict], n_plot: int, tolerance: float, consecutive: int) -> dict:
    plot = rows[:n_plot]
    rel = [r["relative_difference"] for r in plot]
    first = plot[0]["relative_difference"] if plot else None
    last = plot[-1]["relative_difference"] if plot else None
    max_row = max(plot, key=lambda r: r["relative_difference"]) if plot else None
    trend = describe_trend(rel)
    crossing_plot = first_persistent_crossing(
        [r["t"] for r in plot], rel, tolerance=tolerance, consecutive=consecutive
    )
    return {
        "first_step_relative_difference": first,
        "step20_relative_difference": last,
        "max_relative_difference": None if max_row is None else max_row["relative_difference"],
        "max_relative_difference_step": None if max_row is None else max_row["t"],
        "trend_over_plot_window": trend,
        "agree_by_step20": bool(last is not None and last <= tolerance),
        "persistent_crossing_within_plot": crossing_plot,
        "plot_rows": plot,
    }


def run_task_b(cfg: dict) -> dict:
    bcfg = cfg["bias_correction"]
    n_plot = int(bcfg["n_plot_steps"])
    search_limit = int(bcfg["search_limit"])
    relative_eps = float(bcfg["relative_eps"])
    tolerance = float(bcfg["tolerance"])
    consecutive = int(bcfg["consecutive"])
    kwargs = dict(
        w0=float(bcfg["w0"]),
        lr=float(bcfg["lr"]),
        beta1=float(bcfg["beta1"]),
        beta2=float(bcfg["beta2"]),
        eps=float(bcfg["eps"]),
        relative_eps=relative_eps,
    )
    constant_g = [float(bcfg["constant_gradient"])] * max(n_plot, search_limit)
    # Sign-changing, deterministic: a sampled-looking but closed-form wave.
    amp = float(bcfg["varying_amplitude"])
    varying_g = [amp * math.cos(math.pi * t / 3.0) for t in range(1, max(n_plot, search_limit) + 1)]

    scenarios = {}
    for name, grads in (("constant", constant_g), ("varying", varying_g)):
        rows = _run_bias_pair(grads, **kwargs)
        summary = _summarise_bias_rows(rows, n_plot, tolerance, consecutive)
        long_rel = [r["relative_difference"] for r in rows]
        long_steps = [r["t"] for r in rows]
        crossing = first_persistent_crossing(long_steps, long_rel, tolerance, consecutive)
        peak = max(rows[:n_plot], key=lambda r: r["relative_difference"])
        extra = {
            "persistent_crossing_search_limit": search_limit,
            "persistent_crossing": crossing,
            "note": (
                "The persistent-tolerance crossing is searched up to search_limit "
                "and is reported separately from the 20-step plot."
            ),
        }
        if name == "constant":
            extra["closed_form_ratio_at_plot"] = [
                {
                    "t": t,
                    "uncorrected_over_corrected": closed_form_uncorrected_ratio(
                        t, kwargs["beta1"], kwargs["beta2"]
                    ),
                }
                for t in range(1, n_plot + 1)
            ]
        scenarios[name] = {**summary, **extra, "peak_in_plot": peak}
    return {
        "relative_difference_definition": (
            "rel_diff(t) = |delta_uncorrected(t) - delta_corrected(t)| "
            "/ max(|delta_corrected(t)|, eps_rel). "
            "If the corrected update is smaller than eps_rel the denominator is "
            "clamped and the row is flagged."
        ),
        "hyperparameters": {
            "beta1": kwargs["beta1"],
            "beta2": kwargs["beta2"],
            "lr": kwargs["lr"],
            "eps": kwargs["eps"],
            "tolerance": tolerance,
            "consecutive": consecutive,
            "n_plot_steps": n_plot,
            "search_limit": search_limit,
        },
        "scenarios": scenarios,
    }


def _move_batch(batch, device, dtype=None):
    x = batch.x.to(device)
    y = batch.y.to(device)
    if dtype is not None and x.is_floating_point():
        x = x.to(dtype)
        if y.is_floating_point():
            y = y.to(dtype)
    return x, y


def _ratio_rows_for_step(
    model: nn.Module,
    before: dict[str, torch.Tensor],
    missing: set[str],
    ratio_eps: float,
    near_zero_eps: float,
) -> list[dict]:
    rows = []
    for name, param in named_trainable(model):
        if name in missing:
            rec = missing_grad_record(name)
        else:
            rec = update_to_weight_ratio(
                before[name], param, eps=ratio_eps, near_zero_eps=near_zero_eps, name=name
            )
        rows.append(rec.as_dict())
    return rows


def _group_ratios(param_rows: list[dict], groups: dict[str, list[str]]) -> list[dict]:
    by_name = {r["name"]: r for r in param_rows}
    grouped = []
    for group, names in groups.items():
        present = [by_name[n] for n in names if n in by_name]
        if not present:
            continue
        # Group update/weight uses summed squared norms, excluding flagged rows
        # from the headline number but listing them.
        flagged = [r for r in present if r.get("excluded_from_aggregate")]
        kept = [r for r in present if not r.get("excluded_from_aggregate")]
        if kept:
            # Reconstruct from stored norms: not exact for concatenated tensors,
            # but each row already used the true per-tensor delta. Report mean
            # of kept tensors plus max.
            ratios = [float(r["ratio"]) for r in kept if r["ratio"] is not None]
            grouped.append(
                {
                    "name": group,
                    "mean_ratio": sum(ratios) / len(ratios) if ratios else None,
                    "max_ratio": max(ratios) if ratios else None,
                    "n_tensors": len(kept),
                    "n_flagged": len(flagged),
                }
            )
        else:
            grouped.append(
                {
                    "name": group,
                    "mean_ratio": None,
                    "max_ratio": None,
                    "n_tensors": 0,
                    "n_flagged": len(flagged),
                }
            )
    return grouped


def _train_ratio_run(
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    schedule: ScheduleSpec,
    batch_size: int,
    input_dim: int,
    hidden_dim: int,
    n_classes: int,
    n_train: int,
    ratio_eps: float,
    near_zero_eps: float,
    stabilize_tol: float,
    stabilize_k: int,
    data_seed: int,
) -> dict:
    set_seed(seed)
    model = InstrumentedMLP(input_dim, hidden_dim, n_classes).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=schedule.peak_lr, weight_decay=0.0)
    x, y = classification_tensors(n_train, input_dim, n_classes, seed=data_seed)
    x = x.to(device=device, dtype=dtype)
    y = y.to(device)
    batches = cycle_dataset(x, y, batch_size, steps, seed=data_seed + 1)
    init_norms = {name: tensor_norm(param) for name, param in named_trainable(model)}
    init_near_zero = {name for name, nrm in init_norms.items() if nrm < near_zero_eps}
    logs = []
    global_ratios = []
    for t, batch in enumerate(batches, start=1):
        lr = schedule.lr(t)
        apply_lr(opt, lr)
        model.train()
        opt.zero_grad(set_to_none=True)
        xb, yb = batch.x.to(device), batch.y.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        require_finite(float(loss.item()), "training_loss")
        loss.backward()
        missing = {name for name, p in named_trainable(model) if p.grad is None}
        gnorm = global_grad_norm(model.parameters())
        before = snapshot_params(model)
        opt.step()
        param_rows = _ratio_rows_for_step(model, before, missing, ratio_eps, near_zero_eps)
        for row in param_rows:
            if row["name"] in init_near_zero:
                row["near_zero_weight"] = True
                row["excluded_from_aggregate"] = True
                row["policy"] = "exclude_init_near_zero"
        kept_ratios = [
            float(r["ratio"])
            for r in param_rows
            if r["ratio"] is not None and not r["excluded_from_aggregate"]
        ]
        mean_ratio = sum(kept_ratios) / len(kept_ratios) if kept_ratios else None
        if mean_ratio is not None:
            global_ratios.append(mean_ratio)
        groups = _group_ratios(param_rows, model.parameter_groups())
        pnorm = math.sqrt(sum(tensor_norm(p) ** 2 for p in before.values()))
        logs.append(
            {
                "global_step": t,
                "learning_rate": lr,
                "param_rows": param_rows,
                "layer_groups": groups,
                "parameter_norm": pnorm,
                "update_norm": math.sqrt(
                    sum(float(r["update_norm"]) ** 2 for r in param_rows if math.isfinite(r["update_norm"]))
                ),
                "update_to_weight_ratio": mean_ratio,
                "gradient_norm": gnorm,
                "training_loss": float(loss.item()),
                "n_missing_grad": len(missing),
                "n_near_zero_weight": sum(1 for r in param_rows if r["near_zero_weight"]),
            }
        )

    lr_peak_step = schedule.lr_peak_step()
    ratio_series = [row["update_to_weight_ratio"] for row in logs]
    peak_pairs = [(i + 1, r) for i, r in enumerate(ratio_series) if r is not None]
    ratio_peak_step = max(peak_pairs, key=lambda p: p[1])[0] if peak_pairs else None
    # Stabilization: relative change of the global ratio.
    rel_changes = []
    for i in range(1, len(ratio_series)):
        prev = ratio_series[i - 1]
        cur = ratio_series[i]
        if prev is None or cur is None:
            rel_changes.append(float("inf"))
        else:
            rel_changes.append(abs(cur - prev) / max(abs(prev), ratio_eps))
    # Align rel_changes with steps 2..T so a crossing at step t uses change t vs t-1.
    crossing = first_persistent_crossing(
        list(range(2, steps + 1)),
        rel_changes,
        tolerance=stabilize_tol,
        consecutive=stabilize_k,
    )
    keep_detail = {1, lr_peak_step, steps}
    slim_logs = []
    for row in logs:
        slim = dict(row)
        if row["global_step"] not in keep_detail:
            slim.pop("param_rows", None)
        slim_logs.append(slim)
    return {
        "logs": slim_logs,
        "lr_peak_step": lr_peak_step,
        "ratio_peak_step": ratio_peak_step,
        "ratio_stabilize_step": crossing.get("step"),
        "ratio_stabilize": crossing,
        "events_are_identical": crossing.get("step") == lr_peak_step,
        "final_loss": logs[-1]["training_loss"] if logs else None,
        "schedule": {
            "name": schedule.name,
            "warmup_steps": schedule.warmup_steps,
            "peak_lr": schedule.peak_lr,
            "total_steps": schedule.total_steps,
        },
        "init_near_zero": sorted(init_near_zero),
        "n_near_zero_init": len(init_near_zero),
        "n_near_zero_last": logs[-1]["n_near_zero_weight"] if logs else 0,
    }


def run_task_c(cfg: dict) -> dict:
    ccfg = cfg["warmup_ratio"]
    seed = int(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))
    dtype = resolve_dtype(cfg.get("dtype", "float32"))
    steps = int(ccfg["steps"])
    warmup_steps = int(ccfg["warmup_steps"])
    shared = dict(
        seed=seed,
        device=device,
        dtype=dtype,
        steps=steps,
        batch_size=int(ccfg["batch_size"]),
        input_dim=int(ccfg["input_dim"]),
        hidden_dim=int(ccfg["hidden_dim"]),
        n_classes=int(ccfg["n_classes"]),
        n_train=int(ccfg["n_train"]),
        ratio_eps=float(ccfg["ratio_eps"]),
        near_zero_eps=float(ccfg["near_zero_weight_eps"]),
        stabilize_tol=float(ccfg["stabilize_tolerance"]),
        stabilize_k=int(ccfg["stabilize_consecutive"]),
        data_seed=seed + 17,
    )
    warmup = make_schedule(
        "warmup_stable",
        total_steps=steps,
        peak_lr=float(ccfg["peak_lr"]),
        warmup_steps=warmup_steps,
        min_lr=float(ccfg.get("min_lr", 0.0)),
        decay_fraction=0.0,
    )
    no_warmup = make_schedule(
        "constant",
        total_steps=steps,
        peak_lr=float(ccfg["peak_lr"]),
        warmup_steps=0,
        min_lr=float(ccfg.get("min_lr", 0.0)),
        decay_fraction=0.0,
    )
    with_wu = _train_ratio_run(schedule=warmup, **shared)
    without = _train_ratio_run(schedule=no_warmup, **shared)
    return {
        "near_zero_policy": (
            "ratio = ||delta|| / max(||w_before||, ratio_eps). Tensors whose "
            "*initial* ||w|| is below near_zero_weight_eps (zero-init biases) stay "
            "flagged for the whole run and are excluded from the headline mean; "
            "otherwise a tiny bias that moved on step 1 would dominate later ratios. "
            "Missing gradients are excluded. Non-finite values raise."
        ),
        "update_definition": "||parameter_after - parameter_before||, not the gradient norm.",
        "with_warmup": with_wu,
        "without_warmup": without,
        "lr_peak_step": with_wu["lr_peak_step"],
        "ratio_peak_step": with_wu["ratio_peak_step"],
        "ratio_stabilize_step": with_wu["ratio_stabilize_step"],
        "events_are_identical": with_wu["events_are_identical"],
        "loss_with_warmup": with_wu["final_loss"],
        "loss_without_warmup": without["final_loss"],
        "explanation": (
            "The schedule reaches its peak at warmup_steps by construction. "
            "The observed update-to-weight ratio can level off earlier or later "
            "because Adam's effective step size also depends on the moment "
            "estimates and the shrinking residual, not only on the nominal LR."
        ),
    }


def controlled_training_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer,
    *,
    inject_nan_loss: bool = False,
    inject_inf_grad: bool = False,
) -> float:
    """One classification step that refuses non-finite loss or gradients."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    if inject_nan_loss:
        loss = loss * float("nan")
    require_finite(float(loss.detach().item()), "training_loss")
    loss.backward()
    if inject_inf_grad:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.view(-1)[0] = float("inf")
                break
    assert_grads_finite(model.parameters())
    optimizer.step()
    return float(loss.detach().item())


def _checkpoint_fingerprint(state: dict[str, torch.Tensor]) -> dict:
    pieces = []
    n_params = 0
    for name in sorted(state):
        tensor = state[name].detach().float().contiguous().cpu().reshape(-1)
        n_params += int(tensor.numel())
        pieces.append(tensor)
    flat = torch.cat(pieces) if pieces else torch.zeros(1)
    digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
    return {
        "n_tensors": len(state),
        "n_params": n_params,
        "param_norm": float(torch.linalg.vector_norm(flat)),
        "finite": bool(torch.isfinite(flat).all()),
        "sha256_16": digest,
    }


def _evaluate_lm(model: TinySequenceLM, batches, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in batches:
            _logits, loss = model(inputs.to(device), targets.to(device))
            require_finite(float(loss.item()), "val_loss")
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _train_schedule_run(
    *,
    schedule: ScheduleSpec,
    init_state: dict[str, torch.Tensor],
    train_stream: list[tuple[torch.Tensor, torch.Tensor]],
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    model_kwargs: dict,
    device: torch.device,
    seed: int,
    eval_every: int,
    early_stop_step: int,
    weight_decay: float,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    set_seed(seed)
    model = TinySequenceLM(**model_kwargs).to(device)
    model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=schedule.peak_lr, weight_decay=weight_decay)
    history = []
    ckpt_200 = None
    train_loss_200 = None
    val_loss_200 = None
    lr_200 = None
    for t, (inputs, targets) in enumerate(train_stream[: schedule.total_steps], start=1):
        lr = schedule.lr(t)
        apply_lr(opt, lr)
        model.train()
        opt.zero_grad(set_to_none=True)
        _logits, loss = model(inputs.to(device), targets.to(device))
        require_finite(float(loss.item()), "training_loss")
        loss.backward()
        assert_grads_finite(model.parameters())
        opt.step()
        record = {
            "global_step": t,
            "learning_rate": lr,
            "training_loss": float(loss.item()),
            "validation_loss": None,
        }
        if t % eval_every == 0 or t == schedule.total_steps or t == early_stop_step:
            record["validation_loss"] = _evaluate_lm(model, val_batches, device)
        if t == early_stop_step:
            ckpt_200 = _clone_state(model)
            train_loss_200 = record["training_loss"]
            val_loss_200 = record["validation_loss"]
            lr_200 = lr
        history.append(record)
    final = history[-1]
    ckpt_meta = None
    ckpt_path = None
    if ckpt_200 is not None:
        ckpt_meta = _checkpoint_fingerprint(ckpt_200)
        if checkpoint_dir is not None:
            dest = Path(checkpoint_dir)
            dest.mkdir(parents=True, exist_ok=True)
            ckpt_path = dest / f"{schedule.name}_horizon{schedule.total_steps}_step{early_stop_step}.pt"
            torch.save(
                {
                    "step": int(early_stop_step),
                    "schedule": schedule.name,
                    "train_loss": train_loss_200,
                    "validation_loss": val_loss_200,
                    "learning_rate": lr_200,
                    "state_dict": ckpt_200,
                },
                ckpt_path,
            )
    return {
        "schedule": {
            "name": schedule.name,
            "total_steps": schedule.total_steps,
            "warmup_steps": schedule.warmup_steps,
            "decay_steps": schedule.decay_steps,
            "peak_lr": schedule.peak_lr,
            "min_lr": schedule.min_lr,
            "mean_lr_200": schedule.mean_lr(until=early_stop_step),
            "mean_lr_total": schedule.mean_lr(),
        },
        "history": history,
        "step200": {
            "step": int(early_stop_step) if ckpt_200 is not None else None,
            "training_loss": train_loss_200,
            "validation_loss": val_loss_200,
            "learning_rate": lr_200,
            "checkpoint_saved": ckpt_200 is not None,
            "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
            "checkpoint": ckpt_meta,
        },
        "step_final": {
            "step": schedule.total_steps,
            "training_loss": final["training_loss"],
            "validation_loss": final["validation_loss"],
            "learning_rate": final["learning_rate"],
        },
    }


def run_task_d(cfg: dict) -> dict:
    scfg = cfg["schedules"]
    seed = int(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))
    total = int(scfg["total_steps"])
    early = int(scfg["early_stop_step"])
    warmup_frac = float(scfg["warmup_fraction"])
    decay_frac = float(scfg["wsd_decay_fraction"])
    peak_lr = float(scfg["peak_lr"])
    min_frac = float(scfg["min_lr_fraction"])
    eval_every = int(scfg["eval_every"])
    model_kwargs = dict(
        vocab_size=int(scfg["vocab_size"]),
        n_embd=int(scfg["n_embd"]),
        n_head=int(scfg["n_head"]),
        n_layer=int(scfg["n_layer"]),
        block_size=int(scfg["seq_len"]),
        dropout=float(scfg["dropout"]),
    )
    set_seed(seed)
    proto = TinySequenceLM(**model_kwargs)
    init_state = _clone_state(proto)
    train_stream = repeating_stream(
        n_steps=total,
        batch_size=int(scfg["batch_size"]),
        seq_len=int(scfg["seq_len"]),
        vocab_size=int(scfg["vocab_size"]),
        seed=seed + 3,
    )
    val_batches = repeating_stream(
        n_steps=int(scfg["n_val_batches"]),
        batch_size=int(scfg["batch_size"]),
        seq_len=int(scfg["seq_len"]),
        vocab_size=int(scfg["vocab_size"]),
        seed=seed + 99,
    )
    cosine_300 = make_schedule(
        "cosine", total, peak_lr, warmup_fraction=warmup_frac, min_lr_fraction=min_frac
    )
    wsd_300 = make_schedule(
        "wsd",
        total,
        peak_lr,
        warmup_fraction=warmup_frac,
        decay_fraction=decay_frac,
        min_lr_fraction=min_frac,
    )
    cosine_200 = make_schedule(
        "cosine", early, peak_lr, warmup_fraction=warmup_frac, min_lr_fraction=min_frac
    )
    wsd_200 = make_schedule(
        "wsd",
        early,
        peak_lr,
        warmup_fraction=warmup_frac,
        decay_fraction=decay_frac,
        min_lr_fraction=min_frac,
    )
    shared = dict(
        init_state=init_state,
        train_stream=train_stream,
        val_batches=val_batches,
        model_kwargs=model_kwargs,
        device=device,
        seed=seed,
        eval_every=eval_every,
        early_stop_step=early,
        weight_decay=float(scfg["weight_decay"]),
        checkpoint_dir=cfg.get("_checkpoint_dir"),
    )
    run_cos = _train_schedule_run(schedule=cosine_300, **shared)
    run_wsd = _train_schedule_run(schedule=wsd_300, **shared)
    # Fairness pair: both schedules designed for a known 200-step budget.
    shared_200 = {**shared, "early_stop_step": early}
    run_cos_200 = _train_schedule_run(schedule=cosine_200, **shared_200)
    run_wsd_200 = _train_schedule_run(schedule=wsd_200, **shared_200)

    def _pick(a: dict, b: dict, key_path: tuple[str, ...]) -> str:
        node_a: Any = a
        node_b: Any = b
        for k in key_path:
            node_a = node_a[k]
            node_b = node_b[k]
        if node_a is None or node_b is None:
            return "undetermined"
        if node_a < node_b:
            return "cosine"
        if node_b < node_a:
            return "wsd"
        return "tie"

    return {
        "seed": seed,
        "limitations": (
            "Single seed, synthetic modular-increment sequences, tiny Transformer. "
            "Results are not a universal ranking of cosine vs WSD."
        ),
        "matched": {
            "initialization": "cloned state_dict before either run",
            "data_stream": "pre-generated identical batches",
            "optimizer": "Adam, identical peak lr and weight decay",
            "compute_budget": total,
        },
        "cosine_300": run_cos,
        "wsd_300": run_wsd,
        "cosine_designed_200": run_cos_200,
        "wsd_designed_200": run_wsd_200,
        "winner_val_at_200_from_300_budget": _pick(run_cos, run_wsd, ("step200", "validation_loss")),
        "winner_val_at_300": _pick(run_cos, run_wsd, ("step_final", "validation_loss")),
        "winner_val_at_200_from_200_budget": _pick(
            run_cos_200, run_wsd_200, ("step_final", "validation_loss")
        ),
        "checkpoint_rationale": (
            "Step 200 is an early-stop snapshot on a 300-step schedule, not a "
            "finished cosine run. Cosine has already started decaying; WSD is "
            "still at the peak. A 200-step-designed pair is reported separately "
            "because the ranking can change with the planned horizon."
        ),
    }


def _fit_log_log(widths: Sequence[float], lrs: Sequence[float]) -> dict:
    xs = np.log(np.asarray(widths, dtype=np.float64))
    ys = np.log(np.asarray(lrs, dtype=np.float64))
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = intercept + slope * xs
    resid = ys - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse_log": rmse,
        "form": "log(best_lr) = intercept + slope * log(width)",
    }


def _is_boundary(index: int, n: int) -> bool:
    return index == 0 or index == n - 1


def _train_width_cell(
    *,
    width: int,
    lr: float,
    steps: int,
    batch_size: int,
    input_dim: int,
    n_classes: int,
    hidden_layers: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    weight_decay: float,
) -> dict:
    set_seed(seed)
    model = StandardWidthMLP(input_dim, width, n_classes, hidden_layers).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    batches = cycle_dataset(x_train, y_train, batch_size, steps, seed=seed + 5)
    last_loss = None
    diverged = False
    for batch in batches:
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(batch.x.to(device=device, dtype=dtype))
        loss = F.mse_loss(pred, batch.y.to(device=device, dtype=dtype))
        value = float(loss.item())
        if not math.isfinite(value):
            diverged = True
            last_loss = value
            break
        try:
            loss.backward()
            assert_grads_finite(model.parameters())
            opt.step()
        except NonFiniteError:
            diverged = True
            last_loss = value
            break
        last_loss = value
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val.to(device=device, dtype=dtype))
        val_loss = float(F.mse_loss(val_pred, y_val.to(device=device, dtype=dtype)).item())
    if not math.isfinite(val_loss):
        diverged = True
    return {
        "width": width,
        "lr": lr,
        "train_loss": last_loss,
        "val_loss": val_loss,
        "diverged": diverged,
        "finite": math.isfinite(val_loss) and last_loss is not None and math.isfinite(last_loss),
    }


def _choose_best(cells: list[dict]) -> dict:
    finite = [c for c in cells if c.get("finite")]
    if not finite:
        raise RuntimeError("No finite results in LR grid")
    best = min(finite, key=lambda c: c["val_loss"])
    lrs = [c["lr"] for c in cells]
    idx = lrs.index(best["lr"])
    return {
        **best,
        "grid_index": idx,
        "on_boundary": _is_boundary(idx, len(cells)),
        "n_grid": len(cells),
    }


def run_task_e(cfg: dict) -> dict:
    ecfg = cfg["width_sweep"]
    seed = int(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))
    dtype = resolve_dtype(cfg.get("dtype", "float32"))
    widths = [int(w) for w in ecfg["widths"]]
    lr_grid = [float(v) for v in ecfg["lr_grid"]]
    steps = int(ecfg["steps"])
    input_dim = int(ecfg["input_dim"])
    n_classes = int(ecfg["n_classes"])
    hidden_layers = int(ecfg["hidden_layers"])
    set_seed(seed)
    x_train, y_train = regression_from_teacher(
        int(ecfg["n_train"]), input_dim, n_classes, seed=seed + 11
    )
    x_val, y_val = regression_from_teacher(
        int(ecfg["n_val"]), input_dim, n_classes, seed=seed + 23
    )
    x_train = x_train.to(dtype)
    y_train = y_train.to(dtype)
    x_val = x_val.to(dtype)
    y_val = y_val.to(dtype)

    per_width = {}
    for width in widths:
        cells = []
        grid = list(lr_grid)
        expansions = 0
        while True:
            cells = [
                _train_width_cell(
                    width=width,
                    lr=lr,
                    steps=steps,
                    batch_size=int(ecfg["batch_size"]),
                    input_dim=input_dim,
                    n_classes=n_classes,
                    hidden_layers=hidden_layers,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    device=device,
                    dtype=dtype,
                    seed=seed,
                    weight_decay=float(ecfg["weight_decay"]),
                )
                for lr in grid
            ]
            best = _choose_best(cells)
            if not best["on_boundary"] or expansions >= int(ecfg.get("max_expansions", 1)):
                break
            expansions += 1
            factor = float(ecfg.get("expand_factor", 3.0))
            if best["grid_index"] == 0:
                grid = [grid[0] / factor] + grid
            else:
                grid = grid + [grid[-1] * factor]
        if bool(ecfg.get("refine", True)) and not best["on_boundary"]:
            idx = best["grid_index"]
            left, right = grid[idx - 1], grid[idx + 1]
            geo_left = math.sqrt(left * best["lr"])
            geo_right = math.sqrt(best["lr"] * right)
            for extra in (geo_left, geo_right):
                if extra not in grid:
                    cells.append(
                        _train_width_cell(
                            width=width,
                            lr=extra,
                            steps=steps,
                            batch_size=int(ecfg["batch_size"]),
                            input_dim=input_dim,
                            n_classes=n_classes,
                            hidden_layers=hidden_layers,
                            x_train=x_train,
                            y_train=y_train,
                            x_val=x_val,
                            y_val=y_val,
                            device=device,
                            dtype=dtype,
                            seed=seed,
                            weight_decay=float(ecfg["weight_decay"]),
                        )
                    )
            cells.sort(key=lambda c: c["lr"])
            best = _choose_best(cells)
        per_width[str(width)] = {
            "width": width,
            "cells": cells,
            "best": best,
            "grid_expanded": expansions > 0,
            "n_expansions": expansions,
        }

    best_pairs = [(int(k), v["best"]["lr"], v["best"]["val_loss"]) for k, v in per_width.items()]
    best_pairs.sort()
    fit = _fit_log_log([p[0] for p in best_pairs], [p[1] for p in best_pairs])
    target = int(ecfg["extrapolate_width"])
    log_pred = fit["intercept"] + fit["slope"] * math.log(target)
    lr_fit = math.exp(log_pred)
    # Inverse-width heuristic anchored at the largest measured width.
    largest_w, largest_lr, _ = best_pairs[-1]
    lr_inv = largest_lr * (largest_w / target)
    # Leave-one-out exponent sensitivity.
    loo = []
    for i in range(len(best_pairs)):
        keep = [p for j, p in enumerate(best_pairs) if j != i]
        if len(keep) < 2:
            continue
        sub = _fit_log_log([p[0] for p in keep], [p[1] for p in keep])
        loo.append({"left_out_width": best_pairs[i][0], "slope": sub["slope"]})

    return {
        "parameterization": "standard (Kaiming hidden, 1/sqrt(fan_in) head, zero biases). Not μP.",
        "seed": seed,
        "optimizer": "Adam",
        "weight_decay": float(ecfg["weight_decay"]),
        "steps": steps,
        "evaluation_metric": "validation MSE against a frozen width-64 teacher",
        "per_width": per_width,
        "fit_log_log": fit,
        "extrapolation": {
            "width": target,
            "method": "OLS of log(best_lr) on log(width), plus inverse-width heuristic from the largest measured width",
            "lr_from_loglog": lr_fit,
            "lr_from_inverse_width": lr_inv,
            "leave_one_out_slopes": loo,
            "uncertainty": (
                "Only three measured widths. The exponent is sensitive to which "
                "grid point is called the minimum and to the training horizon. "
                "This is an SP observation, not a μP transfer result: we do not "
                "claim the optimum learning rate is width-invariant."
            ),
        },
        "protocol": {
            "init": "Kaiming normal hidden weights, N(0, 1/fan_in) head, zero biases",
            "data": "same teacher-generated tensors for every width and LR",
            "optimizer": "Adam, weight_decay as configured",
            "budget": steps,
        },
    }


def run_all_tasks(cfg: dict, tasks: Sequence[str] | None = None) -> dict:
    selected = [t.upper() for t in (tasks or ["A", "B", "C", "D", "E"])]
    out: dict[str, Any] = {"tasks": selected, "config": cfg}
    dispatch = {
        "A": run_task_a,
        "B": run_task_b,
        "C": run_task_c,
        "D": run_task_d,
        "E": run_task_e,
    }
    for name in selected:
        if name not in dispatch:
            raise ValueError(f"Unknown task {name}")
        out[name] = dispatch[name](cfg)
    return out
