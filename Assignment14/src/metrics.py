"""Loss history helpers and chapter-inspired routing diagnostics.

Load convention used throughout (unless a function name says otherwise):

* ``selected_load_fraction_i = (samples that selected expert i) / batch_size``
* These fractions sum to ``top_k``, not 1, because every sample selects
  ``top_k`` experts. For the default top-2-of-4 setup the ideal fraction is
  ``2 / 4 = 0.5``.
* Assignment fractions ``count_i / (batch * top_k)`` sum to 1 and are the
  ``f_i`` used inside the Switch auxiliary loss. Both are reported; plots use
  selected-load fractions.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch

from src.dense_model import count_parameters
from src.router import RoutingResult


def selected_load_counts(
    selected_indices: torch.Tensor,
    num_experts: int,
    dropped_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert selected-assignment counts. Shape: [num_experts]."""
    if selected_indices.ndim != 2:
        raise ValueError(
            f"selected_indices must be [batch, top_k], got {tuple(selected_indices.shape)}"
        )
    match = selected_indices.unsqueeze(-1) == torch.arange(
        num_experts, device=selected_indices.device
    )
    if dropped_mask is not None:
        match = match & ~dropped_mask.unsqueeze(-1)
    return match.sum(dim=(0, 1))  # [num_experts]


def max_violation(load: torch.Tensor) -> float:
    """``(max_load - mean_load) / mean_load`` when mean load is positive."""
    load_f = load.to(dtype=torch.float64)
    mean = float(load_f.mean().item())
    if mean <= 0:
        return float("nan")
    maximum = float(load_f.max().item())
    return (maximum - mean) / mean


def routing_metrics_from_indices(
    selected_indices: torch.Tensor,
    num_experts: int,
    top_k: int,
    dropped_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute load statistics for one concatenated index tensor [N, top_k]."""
    batch_size = int(selected_indices.size(0))
    counts = selected_load_counts(selected_indices, num_experts, dropped_mask)
    counts_list = [int(v) for v in counts.tolist()]
    selected_fractions = [c / batch_size if batch_size else 0.0 for c in counts_list]
    assignment_total = batch_size * top_k
    dropped = int(dropped_mask.sum().item()) if dropped_mask is not None else 0
    effective_assignments = assignment_total - dropped
    assignment_fractions = [
        c / effective_assignments if effective_assignments else 0.0 for c in counts_list
    ]
    ideal_selected = top_k / num_experts
    dead = sum(1 for c in counts_list if c == 0)
    return {
        "batch_size": batch_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_counts": counts_list,
        "selected_load_fractions": selected_fractions,
        "assignment_fractions": assignment_fractions,
        "ideal_selected_load": ideal_selected,
        "max_vio": max_violation(counts),
        "dead_experts": dead,
        "dropped_assignments": dropped,
        "dropped_fraction": dropped / assignment_total if assignment_total else 0.0,
        "selected_load_sum": float(sum(selected_fractions)),
        "load_convention": (
            "selected_load_fraction = (samples selecting expert i) / N; "
            "fractions sum to top_k"
        ),
    }


def routing_metrics_from_result(
    routing: RoutingResult,
    num_experts: int,
    top_k: int,
) -> dict[str, Any]:
    return routing_metrics_from_indices(
        selected_indices=routing.selected_indices,
        num_experts=num_experts,
        top_k=top_k,
        dropped_mask=routing.dropped_mask,
    )


def aggregate_index_batches(
    batches: Sequence[Mapping[str, torch.Tensor]],
    num_experts: int,
    top_k: int,
) -> dict[str, Any]:
    """Concatenate per-batch routing indices collected under ``torch.no_grad``."""
    if not batches:
        raise ValueError("Cannot aggregate an empty routing collection")
    indices = torch.cat([item["indices"].detach().cpu() for item in batches], dim=0)
    masks = [item.get("dropped_mask") for item in batches]
    dropped = None
    if all(mask is not None for mask in masks):
        dropped = torch.cat([mask.detach().cpu() for mask in masks], dim=0)  # type: ignore[union-attr]
    return routing_metrics_from_indices(indices, num_experts, top_k, dropped)


def regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"regression_metrics expected matching shapes, got {tuple(predictions.shape)} vs "
            f"{tuple(targets.shape)}"
        )
    delta = predictions - targets
    mse = float(torch.mean(delta.square()).item())
    mae = float(torch.mean(delta.abs()).item())
    return {
        "mse": mse,
        "mae": mae,
        "rmse": mse**0.5,
    }


def history_to_columns(history: Iterable[Mapping[str, Any]]) -> dict[str, list[Any]]:
    rows = list(history)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: [row[key] for row in rows] for key in keys}


def jsonable(value: Any) -> Any:
    """Convert tensors / numpy scalars into JSON-friendly Python objects."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bytearray)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def parameter_counts_for(module: torch.nn.Module) -> int:
    return count_parameters(module)
