"""Optional capacity-based dropping."""

from __future__ import annotations

import torch

from src.config import tiny_config
from src.moe_model import build_moe_model
from src.router import TopKRouter, apply_capacity_drop


def test_capacity_drop_keeps_highest_scores_and_counts_drops():
    indices = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    scores = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.3],
            [0.2, 0.4],
        ]
    )
    dropped, n_drop = apply_capacity_drop(indices, scores, num_experts=2, capacity=2)
    # Expert 0 has 4 assignments, capacity 2 -> drop 2. Expert 1 likewise.
    assert n_drop == 4
    assert dropped.shape == (4, 2)
    assert int(dropped.sum()) == 4


def test_moe_with_capacity_tracks_dropped_assignments():
    cfg = tiny_config(num_experts=4, top_k=2, capacity_factor=0.25)
    model = build_moe_model(cfg)
    x = torch.randn(16, cfg.input_dim)
    _y, routing = model(x, return_routing=True)
    assert routing.capacity is not None
    assert routing.dropped_assignments >= 0
    assert routing.dropped_mask.shape == (16, 2)


def test_router_wrong_hidden_width():
    router = TopKRouter(hidden_dim=8, num_experts=4, top_k=2)
    try:
        router(torch.randn(3, 5))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
