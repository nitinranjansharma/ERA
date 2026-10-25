"""Training loop, validation, checkpointing, and CLI entry point.

Usage::

    python -m helpers.train --help
    python -m helpers.train --mode smoke --experiment A
    python -m helpers.train --experiment B --mode full --download-data
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW

from helpers.benchmark import reset_peak_memory
from helpers.config import (
    ExperimentConfig,
    assignment_config,
    ensure_dir,
    get_runtime_info,
    new_grad_scaler,
    require_gpu_for_full_run,
    set_amp_enabled,
    set_seed,
    tiny_config,
)
from helpers.data import PreparedData, infinite_loader, prepare_data
from helpers.metrics import ExperimentMetrics, TokenAccounting, is_finite_number, tokens_per_second
from helpers.model import CausalLM, build_model
from helpers.reversible import (
    ReversibleError,
    round_trip_relative_error,
    verify_euler_reconstruction,
    verify_midpoint_reconstruction,
)


class TrainError(RuntimeError):
    """Unrecoverable training failure."""


@dataclass
class TrainState:
    step: int = 0
    tokens: int = 0
    best_val_loss: float = float("inf")
    history: list[dict[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


def cosine_lr(step_tokens: int, config: ExperimentConfig) -> float:
    warmup = max(int(config.warmup_tokens), 0)
    total = max(int(config.token_budget), 1)
    if warmup > 0 and step_tokens < warmup:
        return config.learning_rate * float(step_tokens) / float(max(warmup, 1))
    progress = (step_tokens - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "ln" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    kwargs = dict(
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if torch.cuda.is_available():
        try:
            return AdamW(groups, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return AdamW(groups, **kwargs)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: CausalLM,
    data: PreparedData,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    budget = int(config.eval_token_budget)
    for batch in data.val_loader:
        if total_tokens >= budget > 0:
            break
        batch = _move_batch(batch, device)
        _logits, loss = model(batch["input_ids"], batch["labels"])
        if loss is None or not torch.isfinite(loss):
            raise TrainError("non-finite validation loss")
        n_tok = int(batch["n_target_tokens"].item())
        total_loss += float(loss.item()) * n_tok
        total_tokens += n_tok
        if budget and total_tokens >= budget:
            break
    model.train()
    if total_tokens == 0:
        return float("nan"), 0
    return total_loss / total_tokens, total_tokens


def reconstruction_probe(model: CausalLM, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    if model.config.variant == "baseline":
        return {"reconstruction_error": None, "reconstruction_ok": None, "variant": "baseline"}
    model.eval()
    with torch.no_grad():
        hidden = model.embed(batch["input_ids"])
        if model.config.variant == "midpoint":
            report = verify_midpoint_reconstruction(
                hidden,
                model.blocks,
                two_h=model.config.two_h,
                atol=model.config.reconstruction_atol,
                rtol=model.config.reconstruction_rtol,
            )
        else:
            report = verify_euler_reconstruction(
                hidden,
                model.blocks,
                h=model.config.euler_h,
                atol=model.config.reconstruction_atol,
                rtol=model.config.reconstruction_rtol,
            )
    model.train()
    return {
        "reconstruction_error": report.max_abs_error,
        "reconstruction_ok": report.ok,
        "variant": report.variant,
    }


def save_checkpoint(
    path: Path,
    model: CausalLM,
    optimizer: AdamW,
    state: TrainState,
    config: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": {"step": state.step, "tokens": state.tokens, "best_val_loss": state.best_val_loss},
            "history": state.history,
            "config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: CausalLM,
    optimizer: AdamW | None,
    state: TrainState,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    saved = payload.get("state") or {}
    state.step = int(saved.get("step", 0))
    state.tokens = int(saved.get("tokens", 0))
    state.best_val_loss = float(saved.get("best_val_loss", float("inf")))
    state.history = list(payload.get("history") or [])


def train(config: ExperimentConfig, data: PreparedData | None = None) -> dict[str, Any]:
    """Run one experiment until the token budget or max_steps is reached."""
    set_seed(config.seed)
    device = config.resolved_device()
    runtime = get_runtime_info(device)
    precision = config.resolved_precision()
    use_amp = bool(config.use_autocast and device.type == "cuda")
    set_amp_enabled(use_amp, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = ensure_dir(config.output_dir)
    run_dir = ensure_dir(out_dir / f"exp_{config.experiment}_{config.variant}_{config.mode}")
    config.save(run_dir / "config.json")

    if data is None:
        data = prepare_data(config, download=config.dataset_name not in ("synthetic",))
    data.tokenizer.save(run_dir / "tokenizer.json")

    # Weights stay fp32. AMP (if enabled) only wraps F, matching the proven notebook.
    model = build_model(config).to(device=device)
    optimizer = build_optimizer(model, config)
    scaler = new_grad_scaler(device, precision)
    state = TrainState()
    ckpt_path = run_dir / config.checkpoint_name
    if config.resume_path:
        load_checkpoint(Path(config.resume_path), model, optimizer, state)

    param_report = model.param_report()
    accounting = TokenAccounting(discarded_tokens=data.discarded_train_tokens)
    recon: dict[str, Any] = {
        "reconstruction_error": None,
        "reconstruction_ok": None,
        "variant": config.variant,
    }

    reset_peak_memory(device)
    train_start = time.perf_counter()
    iterator = infinite_loader(data.train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    status = "completed"
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    val_tokens_seen = 0
    measured_tokens = 0
    measured_time = 0.0
    optimizer_steps_timed = 0
    smooth_loss: float | None = None
    planned_steps = max(1, int(config.token_budget) // max(config.tokens_per_optimizer_step, 1))

    try:
        while state.tokens < config.token_budget:
            if config.max_steps is not None and state.step >= config.max_steps:
                status = "smoke" if config.mode == "smoke" else "completed"
                break
            accum_loss = 0.0
            step_tokens = 0
            step_t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for micro in range(config.accumulation_steps):
                batch = _move_batch(next(iterator), device)
                logits, loss = model(batch["input_ids"], batch["labels"])
                if loss is None or not torch.isfinite(loss):
                    raise TrainError("non-finite training loss")
                scaled = loss / config.accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                n_tok = int(batch["n_target_tokens"].reshape(-1).sum().item())
                n_pad = int(batch["n_padding_tokens"].reshape(-1).sum().item())
                n_in = int(batch["n_input_tokens"].reshape(-1).sum().item())
                accounting.add_micro_batch(
                    input_tokens=n_in, target_tokens=n_tok, padding_tokens=n_pad
                )
                accum_loss += float(loss.item())
                step_tokens += n_tok
                if state.step == 0 and micro == 0 and config.is_reversible and not use_amp:
                    recon = reconstruction_probe(model, batch)
            if scaler is not None:
                scaler.unscale_(optimizer)
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            lr = cosine_lr(state.tokens, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_dt = time.perf_counter() - step_t0
            state.step += 1
            state.tokens += step_tokens
            measured_tokens += step_tokens
            measured_time += step_dt
            optimizer_steps_timed += 1
            last_train_loss = accum_loss / config.accumulation_steps
            smooth_loss = (
                last_train_loss
                if smooth_loss is None
                else 0.95 * smooth_loss + 0.05 * last_train_loss
            )
            record = {
                "step": float(state.step),
                "tokens": float(state.tokens),
                "train_loss": last_train_loss,
                "smooth_loss": smooth_loss,
                "lr": lr,
                "step_seconds": step_dt,
            }
            log_now = state.step <= 3 or (
                config.log_every_steps and state.step % config.log_every_steps == 0
            )
            if log_now:
                state.history.append(record)
                rate = measured_tokens / max(measured_time, 1e-9)
                remain = max(config.token_budget - state.tokens, 0)
                eta_min = (remain / rate / 60.0) if rate > 0 else float("nan")
                print(
                    f"  step {state.step:>6}/{planned_steps}  loss {smooth_loss:6.3f}  "
                    f"{rate:>9,.0f} tokens/s  eta {eta_min:5.1f} min",
                    flush=True,
                )
            if (
                config.eval_every_tokens
                and state.tokens > 0
                and (state.tokens // max(config.eval_every_tokens, 1))
                != ((state.tokens - step_tokens) // max(config.eval_every_tokens, 1))
            ):
                last_val_loss, val_n = evaluate(model, data, config, device)
                val_tokens_seen += val_n
                record["val_loss"] = last_val_loss
                state.history.append(record)
                if is_finite_number(last_val_loss) and last_val_loss < state.best_val_loss:
                    state.best_val_loss = last_val_loss
                    save_checkpoint(ckpt_path, model, optimizer, state, config)
            if state.tokens >= config.token_budget:
                break
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        save_checkpoint(run_dir / "interrupt.pt", model, optimizer, state, config)
        _write_partial(run_dir, config, status, str(exc))
        set_amp_enabled(False)
        raise
    except (TrainError, ReversibleError, KeyboardInterrupt) as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        save_checkpoint(run_dir / "interrupt.pt", model, optimizer, state, config)
        _write_partial(run_dir, config, status, str(exc))
        set_amp_enabled(False)
        raise

    wall = time.perf_counter() - train_start
    last_val_loss, val_n = evaluate(model, data, config, device)
    val_tokens_seen += val_n
    accounting.validation_tokens = val_tokens_seen
    if config.is_reversible:
        try:
            probe_batch = _move_batch(next(iterator), device)
            hidden = model.embed(probe_batch["input_ids"])
            mode = config.variant
            h = config.euler_h if mode == "euler" else config.midpoint_step_size
            rel = round_trip_relative_error(hidden, model.blocks, mode, h)
            recon["round_trip_error"] = rel
            recon["reconstruction_error"] = rel
            if not use_amp:
                recon["reconstruction_ok"] = rel < 1e-4
        except Exception as exc:  # noqa: BLE001 — report, don't fail the run
            recon["round_trip_error"] = None
            recon["notes"] = f"round-trip probe failed: {type(exc).__name__}"
    save_checkpoint(ckpt_path, model, optimizer, state, config)

    peak_alloc, peak_reserved = _peak_memory(device)
    tps = tokens_per_second(measured_tokens, measured_time) if measured_time > 0 else None
    time_per_step = measured_time / optimizer_steps_timed if optimizer_steps_timed else None
    metrics = ExperimentMetrics(
        experiment=config.experiment,
        variant=config.variant,
        status=status,
        trainable_parameters=int(param_report["trainable_parameters"]),
        micro_batch_size=config.micro_batch_size,
        accumulation_steps=config.accumulation_steps,
        effective_batch_size=config.effective_batch_size,
        sequence_length=int(config.sequence_length),
        training_target_tokens=accounting.training_target_tokens,
        token_budget=config.token_budget,
        final_train_loss=last_train_loss if is_finite_number(last_train_loss) else None,
        final_val_loss=last_val_loss if is_finite_number(last_val_loss) else None,
        tokens_per_second=tps,
        wall_clock_seconds=wall,
        time_per_optimizer_step=time_per_step,
        peak_allocated_bytes=peak_alloc,
        peak_reserved_bytes=peak_reserved,
        device=str(device),
        precision=precision,
        gpu_name=runtime.get("gpu_name"),
        gpu_total_memory_bytes=runtime.get("gpu_total_memory_bytes"),
        optimizer=config.optimizer_name,
        learning_rate=config.learning_rate,
        reconstruction_error=recon.get("reconstruction_error"),
        reconstruction_ok=recon.get("reconstruction_ok"),
        reversible_method=None if config.variant == "baseline" else config.variant,
        notes=config.notes,
        extra={
            "param_report": param_report,
            "runtime": runtime,
            "data_source": data.source,
            "tokenizer": data.tokenizer_info(),
            "token_accounting": accounting.to_dict(),
            "optimizer_steps": state.step,
            "history": state.history,
            "two_h": config.two_h,
            "euler_h": config.euler_h,
            "perplexity": (
                math.exp(min(last_val_loss, 20.0))
                if is_finite_number(last_val_loss)
                else None
            ),
            "tie_embeddings": config.tie_embeddings,
            "measured_training_seconds": measured_time,
            "measured_training_tokens": measured_tokens,
            "weight_decay": config.weight_decay,
            "amp": use_amp,
            "amp_dtype": str(runtime.get("recommended_precision")),
            "smooth_loss": smooth_loss,
            "round_trip_error": recon.get("round_trip_error"),
        },
    )
    report = metrics.to_dict()
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    _append_history_csv(run_dir / "history.csv", state.history)
    from helpers.reporting import refresh_output_plots, save_run_loss_plot

    save_run_loss_plot(report, run_dir / "loss.png")
    refresh_output_plots(out_dir)
    set_amp_enabled(False)
    return report


def _peak_memory(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    return int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device))


def _write_partial(run_dir: Path, config: ExperimentConfig, status: str, message: str) -> None:
    payload = {
        "status": status,
        "experiment": config.experiment,
        "variant": config.variant,
        "error": message,
    }
    (run_dir / "failure.json").write_text(json.dumps(payload, indent=2) + "\n")


def _append_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        path.write_text("step,tokens,train_loss,val_loss,lr,step_seconds\n")
        return
    keys = ["step", "tokens", "train_loss", "smooth_loss", "val_loss", "lr", "step_seconds"]
    lines = [",".join(keys)]
    for row in history:
        lines.append(",".join("" if row.get(k) is None else str(row.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def smoke_train(variant: str = "baseline", **overrides: Any) -> dict[str, Any]:
    overrides.setdefault("variant", variant)
    overrides.setdefault("experiment", "A" if overrides.get("variant") == "baseline" else "B")
    config = tiny_config(**overrides)
    data = prepare_data(config, download=False)
    return train(config, data=data)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline or reversible causal LMs for ERA Session 13."
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--experiment", choices=["A", "B", "C", "all"], default="A")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument(
        "--dataset",
        default=None,
        help="wikitext2, synthetic, or sangraha (Indic BPE; needs datasets+tokenizers)",
    )
    parser.add_argument("--config", default=None, help="JSON config to load")
    parser.add_argument("--variant", choices=["baseline", "midpoint", "euler"], default=None)
    parser.add_argument(
        "--allow-cpu-full",
        action="store_true",
        help="Allow a full 50M-token run on CPU (very slow; Colab GPU is the target).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        config = ExperimentConfig.load(args.config)
    elif args.mode == "smoke" and args.experiment != "all":
        variant = args.variant or {"A": "baseline", "B": "midpoint", "C": "midpoint"}[args.experiment]
        config = tiny_config(experiment=args.experiment, variant=variant, mode="smoke")
    else:
        config = assignment_config(args.experiment if args.experiment != "all" else "A", mode=args.mode)
    updates: dict[str, Any] = {
        "output_dir": args.output_dir,
        "device": args.device,
        "seed": args.seed,
        "resume_path": args.resume,
    }
    if args.max_steps is not None:
        updates["max_steps"] = args.max_steps
    if args.token_budget is not None:
        updates["token_budget"] = args.token_budget
    if args.micro_batch_size is not None:
        updates["micro_batch_size"] = args.micro_batch_size
    if args.accumulation_steps is not None:
        updates["accumulation_steps"] = args.accumulation_steps
    if args.dataset is not None:
        dataset = str(args.dataset).lower().replace("-", "")
        if dataset.startswith("wikitext"):
            updates["dataset"] = "wikitext2"
        elif dataset in ("indic", "sangraha", "sangrahaindic", "ai4bharat"):
            updates["dataset"] = "sangraha"
        else:
            updates["dataset"] = dataset
    elif args.download_data:
        updates["dataset"] = "wikitext2"
    if args.variant is not None:
        updates["variant"] = args.variant
    payload = config.to_dict()
    payload.update({k: v for k, v in updates.items() if v is not None})
    return ExperimentConfig.from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    experiments = ["A", "B", "C"] if args.experiment == "all" else [args.experiment]
    for name in experiments:
        args.experiment = name
        config = config_from_args(args)
        if args.mode == "full" and not args.allow_cpu_full:
            require_gpu_for_full_run("full", config.resolved_device())
        download = bool(args.download_data and config.dataset_name != "synthetic")
        data = prepare_data(config, download=download)
        report = train(config, data=data)
        print(json.dumps({k: report[k] for k in ("experiment", "variant", "status", "trainable_parameters", "training_target_tokens", "final_train_loss")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
