"""5. MoE forward-pass shape, finite outputs, and correct weighted combination."""

from __future__ import annotations

import torch
from torch import nn

from src.config import tiny_config
from src.moe_model import MoERegressor, build_moe_model, sparse_expert_combine
from src.router import RoutingResult


def test_moe_forward_shape_and_finite():
    cfg = tiny_config(architecture="ffn", num_experts=4, top_k=2)
    model = build_moe_model(cfg)
    x = torch.randn(10, cfg.input_dim)
    y, routing = model(x, return_routing=True)
    assert y.shape == (10, 1)
    assert torch.isfinite(y).all()
    assert routing.selected_indices.shape == (10, 2)
    assert isinstance(model, MoERegressor)


def test_strict_linear_moe_forward_shape():
    cfg = tiny_config(architecture="strict_linear", num_experts=4, top_k=2)
    model = build_moe_model(cfg)
    y = model(torch.randn(5, cfg.input_dim))
    assert y.shape == (5, 1)
    assert torch.isfinite(y).all()


def test_weighted_combination_matches_manual_mix():
    """Two constant experts mixed with known weights must reproduce the convex combination."""

    class ConstExpert(nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden.new_full((hidden.size(0), 1), self.value)

    hidden = torch.zeros(2, 3)
    experts = nn.ModuleList([ConstExpert(1.0), ConstExpert(3.0), ConstExpert(99.0)])
    indices = torch.tensor([[0, 1], [0, 1]])
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    routing = RoutingResult(
        logits=torch.zeros(2, 3),
        scores=torch.ones(2, 3) / 3,
        selection_scores=torch.ones(2, 3) / 3,
        selected_indices=indices,
        selected_scores=weights,
        selected_weights=weights,
        dropped_mask=torch.zeros(2, 2, dtype=torch.bool),
        dropped_assignments=0,
        capacity=None,
    )
    combined = sparse_expert_combine(hidden, experts, routing)
    assert combined.shape == (2, 1)
    assert torch.allclose(combined[0], torch.tensor([0.25 * 1.0 + 0.75 * 3.0]))
    assert torch.allclose(combined[1], torch.tensor([0.5 * 1.0 + 0.5 * 3.0]))


def test_unselected_expert_forward_is_not_called():
    calls = {0: 0, 1: 0, 2: 0, 3: 0}

    class CountingExpert(nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            self.idx = idx
            self.fc = nn.Linear(4, 4)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            calls[self.idx] += 1
            return hidden

    hidden = torch.randn(3, 4)
    experts = nn.ModuleList([CountingExpert(i) for i in range(4)])
    indices = torch.tensor([[0, 1], [0, 1], [1, 0]])
    weights = torch.ones(3, 2) * 0.5
    routing = RoutingResult(
        logits=torch.zeros(3, 4),
        scores=torch.ones(3, 4) / 4,
        selection_scores=torch.ones(3, 4) / 4,
        selected_indices=indices,
        selected_scores=weights,
        selected_weights=weights,
        dropped_mask=torch.zeros(3, 2, dtype=torch.bool),
        dropped_assignments=0,
        capacity=None,
    )
    sparse_expert_combine(hidden, experts, routing)
    assert calls[0] == 1
    assert calls[1] == 1
    assert calls[2] == 0
    assert calls[3] == 0
