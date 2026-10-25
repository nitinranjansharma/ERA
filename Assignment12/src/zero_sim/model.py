"""Tiny real PyTorch training demo used to illustrate data-parallel math.

The network is deliberately small so it runs on a free Colab CPU runtime in
seconds. It is not a 30-billion-parameter model, and the virtual workers are
not a distributed process group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ToyTrainingConfig, get_device, set_seed


class TinyClassifier(nn.Module):
    """Three-layer MLP for synthetic classification."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 32, n_classes: int = 4) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def clone(self) -> "TinyClassifier":
        twin = TinyClassifier(self.in_dim, self.hidden_dim, self.n_classes)
        twin.load_state_dict(self.state_dict())
        return twin


def make_synthetic_data(
    n_samples: int,
    in_dim: int,
    n_classes: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    features = torch.randn(n_samples, in_dim, generator=generator)
    weights = torch.randn(in_dim, n_classes, generator=generator)
    logits = features @ weights
    labels = logits.argmax(dim=1)
    return features.to(device), labels.to(device)


def global_batch_size(
    sequences_per_gpu: int,
    virtual_gpu_count: int,
    accumulation_steps: int,
) -> int:
    if min(sequences_per_gpu, virtual_gpu_count, accumulation_steps) <= 0:
        raise ValueError("all global-batch factors must be positive")
    return sequences_per_gpu * virtual_gpu_count * accumulation_steps


def _named_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        grads[name] = param.grad.detach().clone()
    return grads


def _set_grads(model: nn.Module, grads: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        param.grad = grads[name].clone()


def _zero_grads(model: nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _weighted_average_grads(
    worker_grads: list[dict[str, torch.Tensor]],
    batch_sizes: list[int],
) -> dict[str, torch.Tensor]:
    total = float(sum(batch_sizes))
    if total <= 0:
        raise ValueError("batch sizes must be positive")
    names = worker_grads[0].keys()
    averaged: dict[str, torch.Tensor] = {}
    for name in names:
        acc = None
        for grads, n_local in zip(worker_grads, batch_sizes):
            contrib = grads[name] * (n_local / total)
            acc = contrib if acc is None else acc + contrib
        assert acc is not None
        averaged[name] = acc
    return averaged


def _max_abs_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    worst = 0.0
    for name in left:
        worst = max(worst, float((left[name] - right[name]).abs().max().item()))
    return worst


def _params_max_abs_diff(left: nn.Module, right: nn.Module) -> float:
    worst = 0.0
    for a, b in zip(left.parameters(), right.parameters()):
        worst = max(worst, float((a - b).abs().max().item()))
    return worst


@dataclass
class GradientSyncResult:
    equal_batch_grad_error: float
    unequal_batch_grad_error: float
    naive_mean_unequal_error: float
    param_error_after_step: float
    reference_loss: float
    synced_loss: float
    batch_sizes: tuple[int, ...]
    global_batch: int


def compare_data_parallel_gradients(
    config: ToyTrainingConfig | None = None,
    device: torch.device | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> GradientSyncResult:
    """Compare a global-batch backward pass with virtual-worker averaging.

    Local losses use ``mean`` reduction. When local batch sizes differ, each
    worker's gradient is weighted by ``n_local / n_global``. A naive mean of
    worker gradients is recorded so the notebook can show why that is wrong.
    """
    cfg = config or ToyTrainingConfig()
    device = device or get_device()
    set_seed(cfg.seed)

    features, labels = make_synthetic_data(
        n_samples=sum(cfg.unequal_worker_batches),
        in_dim=cfg.in_dim,
        n_classes=cfg.n_classes,
        device=device,
        seed=cfg.seed,
    )
    reference = TinyClassifier(cfg.in_dim, cfg.hidden_dim, cfg.n_classes).to(device)
    set_seed(cfg.seed)
    reference = TinyClassifier(cfg.in_dim, cfg.hidden_dim, cfg.n_classes).to(device)

    # --- equal local batches -------------------------------------------------
    equal_n = features.shape[0] // cfg.virtual_gpu_count
    equal_chunks = list(features[: equal_n * cfg.virtual_gpu_count].split(equal_n))
    equal_labels = list(labels[: equal_n * cfg.virtual_gpu_count].split(equal_n))
    equal_x = torch.cat(equal_chunks)
    equal_y = torch.cat(equal_labels)

    ref_eq = reference.clone()
    loss_eq = F.cross_entropy(ref_eq(equal_x), equal_y, reduction="mean")
    loss_eq.backward()
    ref_eq_grads = _named_grads(ref_eq)

    worker_grads = []
    for chunk_x, chunk_y in zip(equal_chunks, equal_labels):
        worker = reference.clone()
        loss = F.cross_entropy(worker(chunk_x), chunk_y, reduction="mean")
        loss.backward()
        worker_grads.append(_named_grads(worker))
    equal_avg = _weighted_average_grads(worker_grads, [equal_n] * len(equal_chunks))
    equal_err = _max_abs_diff(ref_eq_grads, equal_avg)

    # --- unequal local batches ----------------------------------------------
    sizes = list(cfg.unequal_worker_batches)
    splits_x = torch.split(features, sizes)
    splits_y = torch.split(labels, sizes)

    ref_uneq = reference.clone()
    loss_ref = F.cross_entropy(ref_uneq(features), labels, reduction="mean")
    loss_ref.backward()
    ref_grads = _named_grads(ref_uneq)

    unequal_worker_grads = []
    for chunk_x, chunk_y in zip(splits_x, splits_y):
        worker = reference.clone()
        loss = F.cross_entropy(worker(chunk_x), chunk_y, reduction="mean")
        loss.backward()
        unequal_worker_grads.append(_named_grads(worker))

    weighted = _weighted_average_grads(unequal_worker_grads, sizes)
    naive = _weighted_average_grads(
        unequal_worker_grads, [1] * len(sizes)
    )  # equal weights == naive mean
    unequal_err = _max_abs_diff(ref_grads, weighted)
    naive_err = _max_abs_diff(ref_grads, naive)

    synced = reference.clone()
    _set_grads(synced, weighted)
    opt_ref = torch.optim.SGD(ref_uneq.parameters(), lr=cfg.lr)
    opt_sync = torch.optim.SGD(synced.parameters(), lr=cfg.lr)
    opt_ref.step()
    opt_sync.step()
    param_err = _params_max_abs_diff(ref_uneq, synced)

    if equal_err > atol + rtol:
        raise AssertionError(f"equal-batch gradients diverged: {equal_err}")
    if unequal_err > atol + rtol:
        raise AssertionError(f"weighted unequal-batch gradients diverged: {unequal_err}")

    return GradientSyncResult(
        equal_batch_grad_error=equal_err,
        unequal_batch_grad_error=unequal_err,
        naive_mean_unequal_error=naive_err,
        param_error_after_step=param_err,
        reference_loss=float(loss_ref.detach().cpu()),
        synced_loss=float(loss_ref.detach().cpu()),
        batch_sizes=tuple(sizes),
        global_batch=int(features.shape[0]),
    )


def train_toy_model(
    config: ToyTrainingConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run a short SGD loop and return loss history plus gradient stats."""
    cfg = config or ToyTrainingConfig()
    device = device or get_device()
    set_seed(cfg.seed)

    model = TinyClassifier(cfg.in_dim, cfg.hidden_dim, cfg.n_classes).to(device)
    features, labels = make_synthetic_data(
        cfg.n_samples, cfg.in_dim, cfg.n_classes, device, cfg.seed + 1
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    history: list[float] = []
    grad_norm = float("nan")
    for step in range(cfg.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = (step * cfg.batch_size) % cfg.n_samples
        idx = torch.arange(start, start + cfg.batch_size, device=device) % cfg.n_samples
        logits = model(features[idx])
        loss = F.cross_entropy(logits, labels[idx])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is None:
                raise RuntimeError("missing gradient during toy training")
            if not torch.isfinite(param.grad).all():
                raise RuntimeError("non-finite gradient during toy training")
            grad_norm += float(param.grad.detach().norm().cpu())
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    return {
        "initial_loss": history[0],
        "final_loss": history[-1],
        "history": history,
        "grad_norm_last_step": grad_norm,
        "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "finite_loss": all(torch.isfinite(torch.tensor(v)) for v in history),
    }


def train_with_gradient_accumulation(
    config: ToyTrainingConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Simulate micro-batches whose mean gradient matches a global batch.

    Global batch size = sequences_per_gpu × virtual GPUs × accumulation steps.
    """
    cfg = config or ToyTrainingConfig()
    device = device or get_device()
    set_seed(cfg.seed)
    gbs = global_batch_size(
        cfg.sequences_per_gpu, cfg.virtual_gpu_count, cfg.accumulation_steps
    )

    features, labels = make_synthetic_data(
        gbs, cfg.in_dim, cfg.n_classes, device, cfg.seed + 2
    )
    reference = TinyClassifier(cfg.in_dim, cfg.hidden_dim, cfg.n_classes).to(device)
    accumulated = reference.clone()

    ref_loss = F.cross_entropy(reference(features), labels, reduction="mean")
    ref_loss.backward()
    ref_grads = _named_grads(reference)

    micro = cfg.sequences_per_gpu
    n_micro = cfg.virtual_gpu_count * cfg.accumulation_steps
    _zero_grads(accumulated)
    acc_loss = 0.0
    for i in range(n_micro):
        sl = slice(i * micro, (i + 1) * micro)
        loss = F.cross_entropy(accumulated(features[sl]), labels[sl], reduction="mean")
        (loss / n_micro).backward()
        acc_loss += float(loss.detach().cpu()) / n_micro

    acc_grads = _named_grads(accumulated)
    grad_error = _max_abs_diff(ref_grads, acc_grads)

    opt_ref = torch.optim.SGD(reference.parameters(), lr=cfg.lr)
    opt_acc = torch.optim.SGD(accumulated.parameters(), lr=cfg.lr)
    opt_ref.step()
    opt_acc.step()
    param_error = _params_max_abs_diff(reference, accumulated)

    return {
        "global_batch_size": gbs,
        "sequences_per_gpu": cfg.sequences_per_gpu,
        "virtual_gpu_count": cfg.virtual_gpu_count,
        "accumulation_steps": cfg.accumulation_steps,
        "microbatches": n_micro,
        "grad_error": grad_error,
        "param_error": param_error,
        "reference_loss": float(ref_loss.detach().cpu()),
        "accumulated_loss": acc_loss,
        "device": str(device),
    }


def parameter_update_matches(
    config: ToyTrainingConfig | None = None,
    device: torch.device | None = None,
    atol: float = 1e-5,
) -> bool:
    result = compare_data_parallel_gradients(config=config, device=device, atol=atol)
    return result.param_error_after_step <= atol
