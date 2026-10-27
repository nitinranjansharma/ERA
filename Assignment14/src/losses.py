"""Task loss, Switch-style auxiliary load-balancing loss, and router z-loss.

Auxiliary terms are implemented as separate, configurable add-ons. The primary
experiment uses mean squared error only; callers opt into balancing losses by
setting the corresponding coefficients above zero.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.router import RoutingResult


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"mse_loss expected matching shapes, got {tuple(predictions.shape)} vs "
            f"{tuple(targets.shape)}"
        )
    return F.mse_loss(predictions, targets)


def mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"mae_loss expected matching shapes, got {tuple(predictions.shape)} vs "
            f"{tuple(targets.shape)}"
        )
    return torch.mean(torch.abs(predictions - targets))


def switch_auxiliary_loss(
    routing: RoutingResult,
    num_experts: int,
    alpha: float,
) -> torch.Tensor:
    """Switch Transformer auxiliary loss.

    ``L_aux = alpha * N * sum_i (f_i * P_i)``

    * ``P_i`` is the mean softmax router probability for expert ``i``
      (differentiable).
    * ``f_i`` is the fraction of *routed assignments* sent to expert ``i``
      (hard counts, treated as a constant). Assignments dropped by capacity
      are excluded. The fractions sum to 1.

    For top-k routing the assignment denominator is ``batch * top_k`` when
    nothing is dropped, not ``batch``. That matches "fraction of routed
    assignments" rather than the selected-load fraction used in utilization
    plots (those fractions sum to ``top_k``).
    """
    if alpha == 0:
        return routing.scores.new_zeros(())
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    scores = routing.scores  # [batch, num_experts], differentiable
    if scores.size(-1) != num_experts:
        raise ValueError(
            f"scores has {scores.size(-1)} experts but num_experts={num_experts}"
        )
    mean_prob = scores.mean(dim=0)  # P_i, [num_experts]

    with torch.no_grad():
        one_hot = F.one_hot(routing.selected_indices, num_classes=num_experts).to(
            dtype=scores.dtype
        )  # [batch, top_k, num_experts]
        if routing.dropped_mask is not None:
            one_hot = one_hot.masked_fill(routing.dropped_mask.unsqueeze(-1), 0)
        counts = one_hot.sum(dim=(0, 1))  # [num_experts]
        total = counts.sum().clamp_min(1.0)
        assignment_fraction = counts / total  # f_i, sums to 1

    # f_i is a constant; gradient flows through P_i only.
    return alpha * num_experts * torch.sum(assignment_fraction * mean_prob)


def router_z_loss(logits: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Penalty on squared log-sum-exp of router logits.

    ``L_z = coefficient * mean_t (logsumexp_e logits_{t,e})^2``

    ``torch.logsumexp`` is used for numerical stability. The loss discourages
    the router logits from drifting to extremely large magnitudes, which can
    make softmax saturate in float32.
    """
    if coefficient == 0:
        return logits.new_zeros(())
    if coefficient < 0:
        raise ValueError(f"z-loss coefficient must be >= 0, got {coefficient}")
    if logits.ndim != 2:
        raise ValueError(f"router logits must be [batch, experts], got {tuple(logits.shape)}")
    log_z = torch.logsumexp(logits, dim=-1)  # [batch]
    return coefficient * torch.mean(log_z.square())


def total_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    routing: RoutingResult | None,
    num_experts: int,
    aux_loss_coef: float,
    z_loss_coef: float,
) -> dict[str, torch.Tensor]:
    """Combine task MSE with optional auxiliary terms."""
    task = mse_loss(predictions, targets)
    aux = predictions.new_zeros(())
    z_loss = predictions.new_zeros(())
    if routing is not None:
        aux = switch_auxiliary_loss(routing, num_experts=num_experts, alpha=aux_loss_coef)
        z_loss = router_z_loss(routing.logits, z_loss_coef)
    combined = task + aux + z_loss
    return {"loss": combined, "task": task, "aux": aux, "z": z_loss}


@torch.no_grad()
def update_expert_biases(
    bias: torch.Tensor,
    selected_indices: torch.Tensor,
    num_experts: int,
    update_rate: float,
    dropped_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Auxiliary-loss-free load balancing (sign rule).

    Increase the selection bias of underloaded experts and decrease it for
    overloaded experts:

        bias <- bias + u * sign(mean_load - load_i)

    The bias is used only for expert *selection*, never for output weights.
    Updates are in-place on the buffer and also returned.
    """
    if update_rate < 0:
        raise ValueError(f"update_rate must be >= 0, got {update_rate}")
    if update_rate == 0:
        return bias
    match = selected_indices.unsqueeze(-1) == torch.arange(
        num_experts, device=selected_indices.device
    )
    if dropped_mask is not None:
        match = match & ~dropped_mask.unsqueeze(-1)
    load = match.sum(dim=(0, 1)).to(dtype=bias.dtype)  # [num_experts]
    mean_load = load.mean()
    delta = update_rate * torch.sign(mean_load - load)
    bias.add_(delta)
    return bias
