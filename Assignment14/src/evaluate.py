"""Held-out evaluation and dense-versus-MoE comparison."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import aggregate_index_batches, regression_metrics
from src.moe_model import is_moe_model
from src.train import evaluate_loss


@torch.no_grad()
def predict(model: nn.Module, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    outputs = model(features.to(device))
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=False)
    preds: list[torch.Tensor] = []
    seen: list[torch.Tensor] = []
    for batch_x, batch_y in loader:
        pred = predict(model, batch_x, device)
        preds.append(pred.detach().cpu())
        seen.append(batch_y.detach().cpu())
    prediction = torch.cat(preds, dim=0)
    target = torch.cat(seen, dim=0)
    return regression_metrics(prediction, target)


@torch.no_grad()
def compare_outputs(
    dense: nn.Module,
    moe: nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Pointwise discrepancy between dense and MoE predictions on the same inputs."""
    dense_out = predict(dense, features, device).detach().cpu()
    moe_out = predict(moe, features, device).detach().cpu()
    delta = moe_out - dense_out
    return {
        "mean_abs_diff": float(delta.abs().mean().item()),
        "max_abs_diff": float(delta.abs().max().item()),
        "mse": float(delta.square().mean().item()),
        "n": int(features.size(0)),
    }


@torch.no_grad()
def collect_routing_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_experts: int,
    top_k: int,
) -> dict[str, Any]:
    if not is_moe_model(model):
        raise TypeError("collect_routing_metrics requires an MoE model")
    model.eval()
    batches: list[dict[str, torch.Tensor]] = []
    targets: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for features, target in loader:
        features = features.to(device)
        _pred, routing = model(features, return_routing=True)
        batches.append(
            {
                "indices": routing.selected_indices.detach().cpu(),
                "dropped_mask": routing.dropped_mask.detach().cpu(),
            }
        )
        targets.append(target.detach().cpu())
        indices.append(routing.selected_indices.detach().cpu())
    metrics = aggregate_index_batches(batches, num_experts=num_experts, top_k=top_k)
    all_targets = torch.cat(targets, dim=0).squeeze(-1)
    all_indices = torch.cat(indices, dim=0)
    metrics["by_target_quartile"] = _routing_by_target_quartile(
        all_targets, all_indices, num_experts
    )
    return metrics


def _routing_by_target_quartile(
    targets: torch.Tensor,
    selected_indices: torch.Tensor,
    num_experts: int,
) -> dict[str, Any]:
    """Optional diagnostic: expert usage in each target-value quartile."""
    if targets.numel() == 0:
        return {}
    qs = torch.quantile(targets.float(), torch.tensor([0.25, 0.5, 0.75]))
    bins = [
        targets <= qs[0],
        (targets > qs[0]) & (targets <= qs[1]),
        (targets > qs[1]) & (targets <= qs[2]),
        targets > qs[2],
    ]
    labels = ["Q1 (lowest y)", "Q2", "Q3", "Q4 (highest y)"]
    out: dict[str, Any] = {}
    for label, mask in zip(labels, bins):
        if not bool(mask.any()):
            continue
        idx = selected_indices[mask]
        counts = [(idx == e).sum().item() for e in range(num_experts)]
        n = int(mask.sum().item())
        out[label] = {
            "n": n,
            "selected_load_fractions": [c / n for c in counts],
        }
    return out


def split_eval_bundle(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    return {
        "train": evaluate_loss(model, train_loader, device, split="train"),
        "val": evaluate_loss(model, val_loader, device, split="val"),
        "test": evaluate_loss(model, test_loader, device, split="test"),
    }
