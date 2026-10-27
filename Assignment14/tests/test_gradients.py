"""8. Finite gradients for selected experts and the router; none for unselected."""

from __future__ import annotations

import torch

from src.config import tiny_config
from src.dense_model import build_dense_model
from src.losses import mse_loss
from src.upcycling import upcycle_dense_to_moe


def test_selected_experts_and_router_get_finite_grads_unselected_do_not():
    cfg = tiny_config(architecture="ffn", num_experts=4, top_k=2, seed=0)
    dense = build_dense_model(cfg)
    moe, _report = upcycle_dense_to_moe(dense, cfg, initialization_strategy="copy", seed=0)

    # Force logits so experts 0 and 1 always win top-2.
    with torch.no_grad():
        moe.router.gate.weight.zero_()
        moe.router.gate.bias.copy_(torch.tensor([8.0, 8.0, -8.0, -8.0]))

    x = torch.randn(6, cfg.input_dim)
    y = torch.randn(6, 1)
    moe.train()
    moe.zero_grad(set_to_none=True)
    pred, routing = moe(x, return_routing=True)
    assert set(routing.selected_indices[0].tolist()) == {0, 1}
    loss = mse_loss(pred, y)
    loss.backward()

    assert moe.router.gate.weight.grad is not None
    assert torch.isfinite(moe.router.gate.weight.grad).all()
    assert moe.input_proj.weight.grad is not None
    assert torch.isfinite(moe.input_proj.weight.grad).all()
    assert moe.output_proj.weight.grad is not None

    for idx in (0, 1):
        grad = moe.experts[idx].fc1.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0

    for idx in (2, 3):
        grad = moe.experts[idx].fc1.weight.grad
        assert grad is None or torch.count_nonzero(grad).item() == 0
