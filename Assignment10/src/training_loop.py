"""Minimal real training loop with shape tracing, logging, and sanity checks.

PyTorch accumulates gradients by default: ``p.grad`` is *added to* on each
``backward()`` if it already exists. That is the feature that makes micro-batch
gradient accumulation possible. In a conventional single-micro-batch loop the
correct action is therefore to clear gradients *before* ``backward()``, which
this module does with ``optimizer.zero_grad(set_to_none=True)``. Forgetting
that clear is incorrect for a non-accumulation loop: successive batches' grads
are summed, so the update is silently scaled by the number of uncleared steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from instrumentation import ShapeTracer, StepLog, now_seconds
from models import TinyGPT, TinyMLP, count_parameters, parameter_shape_records
from safety import (
    TrainingSafetyError,
    assert_expected_grads,
    assert_grads_finite,
    assert_loss_ok,
    assert_parameters_changed,
    assert_params_finite,
    assert_shapes_compatible,
    assert_target_dtype,
    snapshot_params,
)


def calculate_grad_norm(
    parameters,
    *,
    missing: str = "skip",
) -> torch.Tensor:
    """Global L2 norm: sqrt(sum_i ||g_i||^2).

    missing:
      - "skip": parameters with grad is None contribute nothing (same as a zero tensor)
      - "zero": identical numerically to skip; documented alias
      - "error": raise if a requires_grad parameter has a missing grad
    """
    if missing not in {"skip", "zero", "error"}:
        raise TrainingSafetyError(f"Unknown missing-grad policy {missing!r}")
    grads = []
    missing_names = []
    for i, p in enumerate(parameters):
        if not getattr(p, "requires_grad", True):
            continue
        if p.grad is None:
            missing_names.append(getattr(p, "_param_name", f"param[{i}]"))
            continue
        grads.append(p.grad.detach().reshape(-1))
    if missing_names and missing == "error":
        raise TrainingSafetyError(f"Missing gradients for: {missing_names}")
    if not grads:
        return torch.tensor(0.0)
    flat = torch.cat(grads).float()
    return torch.linalg.vector_norm(flat, ord=2)


@dataclass
class TrainingStepResult:
    loss: float
    grad_norm: float
    lr: float
    elapsed_s: float
    throughput_samples_per_s: float
    tokens_per_s: Optional[float]
    tracer: ShapeTracer


def training_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    loss_fn=None,
    task: str = "regression",
    tracer: Optional[ShapeTracer] = None,
    record_optimizer_state: bool = True,
    tokens_in_batch: Optional[int] = None,
) -> TrainingStepResult:
    """One conventional training step: zero_grad → forward → loss → backward → step."""
    tracer = tracer or ShapeTracer()
    loss_fn = loss_fn or (nn.MSELoss(reduction="mean") if task == "regression" else None)

    t0 = now_seconds()
    logits = model(inputs, tracer=tracer) if _accepts_tracer(model) else model(inputs)
    if task == "regression":
        assert_target_dtype(targets, task=task)
        assert_shapes_compatible(logits, targets, task=task)
        tracer.add("targets", targets, list(_labels_for(targets, ["batch", "out"])), "Regression targets aligned with logits.")
        loss = loss_fn(logits, targets)
    elif task == "token_lm":
        assert_target_dtype(targets, task=task)
        assert_shapes_compatible(logits, targets, task=task)
        tracer.add(
            "targets",
            targets,
            ["batch", "sequence"],
            "Next-token class indices in [0, vocab). Same layout as input_ids.",
        )
        # Flatten token dimension; mean over B*T predicted positions.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
    else:
        raise TrainingSafetyError(f"Unknown task {task!r}")

    tracer.add("loss", loss, [], "Scalar mean loss; a 0-d tensor, not a 1-vector.")
    assert_loss_ok(loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert_expected_grads(model)
    assert_grads_finite(model.parameters())

    for name, p in model.named_parameters():
        if p.grad is not None:
            tracer.add(
                f"grad.{name}",
                p.grad,
                list(_guess_param_dims(name, p.grad)),
                f"Gradient of the scalar loss w.r.t. {name}; same shape as the parameter.",
            )

    grad_norm = calculate_grad_norm(model.parameters(), missing="error")
    tracer.add("grad_norm", grad_norm, [], "Scalar global L2 norm of all parameter gradients.")

    before = snapshot_params(model)
    optimizer.step()
    assert_params_finite(model.parameters())
    assert_parameters_changed(before, model)

    if record_optimizer_state:
        _record_optimizer_state(optimizer, tracer)

    elapsed = now_seconds() - t0
    n_samples = int(inputs.shape[0])
    throughput = n_samples / elapsed if elapsed > 0 else float("inf")
    tok_s = None
    if tokens_in_batch is not None:
        tok_s = tokens_in_batch / elapsed if elapsed > 0 else float("inf")
    lr = float(optimizer.param_groups[0]["lr"])
    parameter_shape_records(model, tracer)
    return TrainingStepResult(
        loss=float(loss.detach().cpu()),
        grad_norm=float(grad_norm.detach().cpu()),
        lr=lr,
        elapsed_s=elapsed,
        throughput_samples_per_s=throughput,
        tokens_per_s=tok_s,
        tracer=tracer,
    )


def run_regression_training(
    *,
    steps: int,
    lr: float,
    batch_size: int,
    in_features: int,
    hidden: int,
    seed_offset: int = 0,
    device: torch.device | str = "cpu",
    clip_threshold: Optional[float] = None,
) -> list[StepLog]:
    device = torch.device(device)
    model = TinyMLP(in_features=in_features, hidden=hidden).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Fixed synthetic mapping: y = sum(x) / F, easy to overfit.
    g = torch.Generator(device="cpu").manual_seed(0 + seed_offset)
    logs: list[StepLog] = []
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, in_features, generator=g, device="cpu").to(device)
        y = x.sum(dim=-1, keepdim=True) / in_features
        t0 = now_seconds()
        logits = model(x)
        loss = F.mse_loss(logits, y)
        assert_loss_ok(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = calculate_grad_norm(model.parameters())
        if clip_threshold is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
        optimizer.step()
        elapsed = now_seconds() - t0
        logs.append(
            StepLog(
                step=step,
                loss=float(loss.detach().cpu()),
                grad_norm=float(grad_norm.detach().cpu()),
                lr=lr,
                elapsed_s=elapsed,
                throughput_samples_per_s=batch_size / elapsed if elapsed > 0 else float("inf"),
            )
        )
    return logs


def overfit_tiny_dataset(
    *,
    n_samples: int = 8,
    in_features: int = 8,
    hidden: int = 32,
    steps: int = 400,
    lr: float = 0.2,
    threshold: float = 1e-4,
    device: torch.device | str = "cpu",
) -> dict:
    """Prove the loop can *learn* by interpolating a tiny fixed dataset.

    A decreasing loss is not sufficient: we also require that the final loss
    falls below ``threshold`` and that held-out-from-batch training residuals
    on the same points are small.
    """
    device = torch.device(device)
    g = torch.Generator().manual_seed(123)
    x = torch.randn(n_samples, in_features, generator=g)
    y = torch.randn(n_samples, 1, generator=g)
    model = TinyMLP(in_features, hidden, 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    x_d, y_d = x.to(device), y.to(device)
    for step in range(1, steps + 1):
        logits = model(x_d)
        loss = F.mse_loss(logits, y_d)
        assert_loss_ok(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert_expected_grads(model)
        opt.step()
        history.append(float(loss.detach().cpu()))
    final = history[-1]
    with torch.no_grad():
        residual = F.mse_loss(model(x_d), y_d).item()
    learned = final <= threshold and residual <= threshold
    return {
        "n_samples": n_samples,
        "steps": steps,
        "initial_loss": history[0],
        "final_loss": final,
        "residual_mse": residual,
        "threshold": threshold,
        "learned": learned,
        "history": history,
        "parameter_count": count_parameters(model),
    }


def demonstrate_forgotten_zero_grad(
    *,
    in_features: int = 4,
    hidden: int = 8,
    batch_size: int = 4,
) -> dict:
    """Show that skipping zero_grad sums successive backward passes.

    Two identical minibatches without a clear produce ~2× the gradient of a
    correctly-cleared step. That is not "more learning"; it is an implicit
    and usually unwanted rescaling of the update.
    """
    x = torch.ones(batch_size, in_features)
    y = torch.zeros(batch_size, 1)

    def one_backward(clear: bool, twice: bool) -> torch.Tensor:
        model = TinyMLP(in_features, hidden, 1)
        # Identical weights.
        torch.manual_seed(0)
        for p in model.parameters():
            torch.nn.init.constant_(p, 0.1)
        loss_fn = nn.MSELoss()
        if clear:
            model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if twice:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])

    g_once = one_backward(clear=True, twice=False)
    g_twice_uncleared = one_backward(clear=True, twice=True)
    ratio = (g_twice_uncleared / g_once).mean().item()
    return {
        "grad_norm_single": float(torch.linalg.vector_norm(g_once)),
        "grad_norm_uncleared_twice": float(torch.linalg.vector_norm(g_twice_uncleared)),
        "mean_ratio": ratio,
        "allclose_double": bool(torch.allclose(g_twice_uncleared, 2 * g_once, rtol=1e-5, atol=1e-6)),
        "explanation": (
            "PyTorch's autograd *adds* into p.grad when it already exists. "
            "That is the accumulation mechanism. In an ordinary loop you must "
            "clear grads before backward; otherwise two identical batches yield "
            "approximately 2× the intended gradient."
        ),
    }


def traced_language_model_step(
    *,
    vocab_size: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 32,
    block_size: int = 32,
    batch_size: int = 2,
    seq_len: int = 8,
    device: torch.device | str = "cpu",
) -> TrainingStepResult:
    device = torch.device(device)
    model = TinyGPT(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Next-token targets: a shifted copy, last position wraps (toy, not a real LM objective).
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return training_step(
        model,
        input_ids,
        targets,
        optimizer,
        task="token_lm",
        tokens_in_batch=batch_size * seq_len,
    )


def save_checkpoint(model: nn.Module, path) -> None:
    torch.save({"model": model.state_dict(), "arch": type(model).__name__}, path)


def load_checkpoint(model: nn.Module, path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored = payload["model"]
    current = model.state_dict()
    if stored.keys() != current.keys():
        missing = set(current) - set(stored)
        unexpected = set(stored) - set(current)
        raise TrainingSafetyError(
            f"Checkpoint/architecture mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    for k, v in stored.items():
        if current[k].shape != v.shape:
            raise TrainingSafetyError(
                f"Checkpoint shape mismatch for {k}: checkpoint {tuple(v.shape)} vs model {tuple(current[k].shape)}"
            )
        if current[k].dtype != v.dtype:
            raise TrainingSafetyError(
                f"Checkpoint dtype mismatch for {k}: checkpoint {v.dtype} vs model {current[k].dtype}"
            )
    model.load_state_dict(stored)


def _accepts_tracer(model: nn.Module) -> bool:
    try:
        import inspect

        return "tracer" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _labels_for(tensor: torch.Tensor, labels: list[str]) -> list[str]:
    if tensor.ndim != len(labels):
        return [f"dim{i}" for i in range(tensor.ndim)]
    return labels


def _guess_param_dims(name: str, t: torch.Tensor) -> list[str]:
    if t.ndim == 0:
        return []
    if t.ndim == 1:
        return ["feature"] if t.numel() > 1 else ["parameter"]
    if t.ndim == 2:
        return ["out_feature", "in_feature"]
    return [f"dim{i}" for i in range(t.ndim)]


def _record_optimizer_state(optimizer: torch.optim.Optimizer, tracer: ShapeTracer) -> None:
    for i, group in enumerate(optimizer.param_groups):
        for j, p in enumerate(group["params"]):
            state = optimizer.state.get(p, {})
            for key, val in state.items():
                if torch.is_tensor(val):
                    tracer.add(
                        f"adam.state[{i},{j}].{key}",
                        val,
                        list(_guess_param_dims(key, val)),
                        f"Optimizer state '{key}' has the same shape as the corresponding parameter.",
                    )

