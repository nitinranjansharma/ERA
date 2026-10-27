"""3–4. Router shapes, distinct top-k, unit weights, and invalid configs."""

from __future__ import annotations

import pytest
import torch

from src.router import TopKRouter, expert_capacity


def test_router_shapes_distinct_topk_and_normalized_weights():
    router = TopKRouter(hidden_dim=16, num_experts=4, top_k=2)
    hidden = torch.randn(12, 16)
    routing = router(hidden)
    assert routing.logits.shape == (12, 4)
    assert routing.scores.shape == (12, 4)
    assert routing.selected_indices.shape == (12, 2)
    assert routing.selected_weights.shape == (12, 2)
    assert torch.allclose(routing.scores.sum(dim=-1), torch.ones(12), atol=1e-5)
    assert torch.allclose(routing.selected_weights.sum(dim=-1), torch.ones(12), atol=1e-5)
    for row in routing.selected_indices:
        assert len(set(row.tolist())) == 2
    assert torch.isfinite(routing.logits).all()


def test_router_invalid_top_k():
    with pytest.raises(ValueError):
        TopKRouter(hidden_dim=8, num_experts=4, top_k=0)
    with pytest.raises(ValueError):
        TopKRouter(hidden_dim=8, num_experts=4, top_k=-1)
    with pytest.raises(ValueError):
        TopKRouter(hidden_dim=8, num_experts=4, top_k=5)


def test_router_top_k_equals_num_experts_is_allowed():
    router = TopKRouter(hidden_dim=8, num_experts=3, top_k=3)
    routing = router(torch.randn(5, 8))
    assert routing.selected_indices.shape == (5, 3)
    assert torch.allclose(routing.selected_weights.sum(-1), torch.ones(5), atol=1e-5)


def test_selection_bias_does_not_change_output_weights_formula():
    """Bias may change *who* is selected; weights always come from unbiased softmax."""
    router = TopKRouter(hidden_dim=8, num_experts=4, top_k=2, use_bias_for_selection=True)
    torch.nn.init.zeros_(router.gate.weight)
    torch.nn.init.zeros_(router.gate.bias)
    hidden = torch.ones(6, 8)
    unbiased = router(hidden)
    router.expert_bias.copy_(torch.tensor([0.0, 0.0, 5.0, 5.0]))
    biased = router(hidden)
    # Unbiased logits are all equal, so scores are uniform 0.25.
    assert torch.allclose(unbiased.scores, torch.full_like(unbiased.scores, 0.25), atol=1e-5)
    assert torch.allclose(biased.scores, unbiased.scores, atol=1e-5)
    # Selection should prefer experts 2 and 3 after the bias is applied.
    selected = set(biased.selected_indices[0].tolist())
    assert selected == {2, 3}


def test_sigmoid_selected_weights_still_sum_to_one():
    router = TopKRouter(hidden_dim=8, num_experts=4, top_k=2, score="sigmoid")
    routing = router(torch.randn(7, 8))
    assert routing.selected_indices.shape == (7, 2)
    assert torch.allclose(routing.selected_weights.sum(-1), torch.ones(7), atol=1e-5)


def test_capacity_formula():
    # tokens=8, top_k=2, experts=4, factor=1.0 -> capacity = (8*2/4)*1 = 4
    assert expert_capacity(8, 2, 4, 1.0) == 4
    assert expert_capacity(8, 2, 4, 2.0) == 8
