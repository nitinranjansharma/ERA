"""Training loops, optimizer configuration, and best-checkpoint selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.checkpoint import save_checkpoint
from src.config import ExperimentConfig
from src.losses import total_loss, update_expert_biases
from src.moe_model import is_moe_model


def build_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
    learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    lr = config.learning_rate if learning_rate is None else learning_rate
    kwargs = dict(lr=lr, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unknown optimizer {config.optimizer!r}")


def _forward(
    model: nn.Module, features: torch.Tensor
) -> tuple[torch.Tensor, Any]:
    if is_moe_model(model):
        predictions, routing = model(features, return_routing=True)
        return predictions, routing
    return model(features), None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    """One pass over ``loader`` with gradient updates."""
    model.train()
    totals = {"task": 0.0, "aux": 0.0, "z": 0.0, "loss": 0.0, "samples": 0.0}
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions, routing = _forward(model, features)
        parts = total_loss(
            predictions,
            targets,
            routing=routing,
            num_experts=config.num_experts,
            aux_loss_coef=config.aux_loss_coef if routing is not None else 0.0,
            z_loss_coef=config.z_loss_coef if routing is not None else 0.0,
        )
        parts["loss"].backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        if (
            routing is not None
            and config.use_bias_balancing
            and hasattr(model, "router")
        ):
            update_expert_biases(
                bias=model.router.expert_bias,
                selected_indices=routing.selected_indices,
                num_experts=config.num_experts,
                update_rate=config.bias_update_rate,
                dropped_mask=routing.dropped_mask,
            )
        batch = features.size(0)
        totals["task"] += float(parts["task"].detach().item()) * batch
        totals["aux"] += float(parts["aux"].detach().item()) * batch
        totals["z"] += float(parts["z"].detach().item()) * batch
        totals["loss"] += float(parts["loss"].detach().item()) * batch
        totals["samples"] += batch
    n = max(totals["samples"], 1.0)
    return {
        "train_mse": totals["task"] / n,
        "train_aux": totals["aux"] / n,
        "train_z": totals["z"] / n,
        "train_loss": totals["loss"] / n,
        "train_samples": int(totals["samples"]),
    }


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str = "val",
) -> dict[str, float]:
    """Full-dataset MSE/MAE in eval mode. No gradient tracking."""
    model.eval()
    sse = 0.0
    sae = 0.0
    n = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        predictions, _routing = _forward(model, features)
        delta = predictions - targets
        sse += float(delta.square().sum().item())
        sae += float(delta.abs().sum().item())
        n += features.size(0)
    n = max(n, 1)
    mse = sse / n
    mae = sae / n
    return {
        f"{split}_mse": mse,
        f"{split}_mae": mae,
        f"{split}_rmse": mse**0.5,
        f"{split}_samples": n,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    learning_rate: float | None = None,
    checkpoint_path: str | Path | None = None,
    log_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train for ``epochs``, keep the lowest-validation-MSE checkpoint, reload it."""
    optimizer = build_optimizer(model, config, learning_rate=learning_rate)
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    best_path = Path(checkpoint_path) if checkpoint_path is not None else None

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, config, device)
        val_stats = evaluate_loss(model, val_loader, device, split="val")
        row: dict[str, Any] = {"epoch": epoch, **train_stats, **val_stats}
        history.append(row)
        if log_fn is not None:
            log_fn(row)
        if val_stats["val_mse"] < best_val:
            best_val = val_stats["val_mse"]
            best_epoch = epoch
            if best_path is not None:
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    metrics={"val_mse": best_val, "train_mse": train_stats["train_mse"]},
                )

    restored = False
    if best_path is not None and best_path.is_file():
        from src.checkpoint import load_checkpoint

        blob = load_checkpoint(best_path, map_location=device, model=model, optimizer=None)
        restored = True
        _ = blob

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "final_train_mse": history[-1]["train_mse"] if history else None,
        "final_val_mse": history[-1]["val_mse"] if history else None,
        "epochs": epochs,
        "steps": sum(row["train_samples"] for row in history) // max(
            history[0]["train_samples"], 1
        )
        if history
        else 0,
        "checkpoint_path": str(best_path) if best_path is not None else None,
        "restored_best": restored,
        "optimizer": config.optimizer,
        "learning_rate": learning_rate if learning_rate is not None else config.learning_rate,
    }
