"""Auxiliary load-balancing loss, router z-loss, and bias updates."""

from __future__ import annotations

import torch

from src.losses import router_z_loss, switch_auxiliary_loss, update_expert_biases
from src.router import RoutingResult


def _routing(scores: torch.Tensor, indices: torch.Tensor) -> RoutingResult:
    batch, experts = scores.shape
    top_k = indices.size(-1)
    selected = torch.gather(scores, -1, indices)
    return RoutingResult(
        logits=torch.log(scores.clamp_min(1e-8)),
        scores=scores,
        selection_scores=scores,
        selected_indices=indices,
        selected_scores=selected,
        selected_weights=selected / selected.sum(-1, keepdim=True),
        dropped_mask=torch.zeros(batch, top_k, dtype=torch.bool),
        dropped_assignments=0,
        capacity=None,
    )


def test_switch_auxiliary_loss_matches_closed_form():
    # Two samples, two experts, top-1. Both pick expert 0.
    scores = torch.tensor([[0.8, 0.2], [0.7, 0.3]], requires_grad=True)
    indices = torch.tensor([[0], [0]])
    routing = _routing(scores, indices)
    loss = switch_auxiliary_loss(routing, num_experts=2, alpha=0.1)
    # f = [1, 0], P = [0.75, 0.25], N * sum(fP) = 2 * 0.75 = 1.5, * alpha = 0.15
    assert torch.allclose(loss, torch.tensor(0.15), atol=1e-6)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_router_z_loss_matches_logsumexp_squared():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    loss = router_z_loss(logits, coefficient=0.5)
    manual = 0.5 * torch.mean(torch.logsumexp(logits, dim=-1).square())
    assert torch.allclose(loss, manual)
    assert torch.allclose(router_z_loss(logits, 0.0), torch.tensor(0.0))


def test_bias_update_increases_underloaded_expert():
    bias = torch.zeros(2)
    # Both samples select expert 0 (top-1).
    indices = torch.tensor([[0], [0], [0], [0]])
    update_expert_biases(bias, indices, num_experts=2, update_rate=0.1)
    # Expert 0 overloaded -> bias decreases; expert 1 underloaded -> increases.
    assert bias[0] < 0
    assert bias[1] > 0
