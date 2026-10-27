"""6–7. Upcycling copies weights, keeps independent storage, preserves dense output."""

from __future__ import annotations

import torch

from src.config import tiny_config
from src.dense_model import build_dense_model
from src.upcycling import get_strategy, upcycle_dense_to_moe


def _trained_like_dense(seed: int = 0):
    torch.manual_seed(seed)
    cfg = tiny_config(architecture="ffn", num_experts=4, top_k=2, seed=seed)
    dense = build_dense_model(cfg)
    # Nudge weights away from the default init so a copy is actually detectable.
    with torch.no_grad():
        for param in dense.parameters():
            param.add_(0.35)
    return cfg, dense


def test_upcycling_copies_weights_independent_storage_and_leaves_dense_unchanged():
    cfg, dense = _trained_like_dense(seed=11)
    snapshot = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    moe, report = upcycle_dense_to_moe(dense, cfg, initialization_strategy="copy", seed=11)

    assert report.dense_unchanged
    assert report.independent_storage
    assert report.all_copied_exactly
    assert report.num_experts == 4
    assert report.top_k == 2
    for key, tensor in dense.state_dict().items():
        assert torch.equal(tensor, snapshot[key])

    source = dense.ffn.state_dict()
    for expert in moe.experts:
        for key, tensor in expert.state_dict().items():
            assert torch.equal(tensor, source[key])
            assert tensor.data_ptr() != source[key].data_ptr()

    # Mutating an expert must not write through to the dense FFN.
    with torch.no_grad():
        moe.experts[0].fc1.weight[0, 0] = 123.456
    assert dense.ffn.fc1.weight[0, 0] != 123.456


def test_identical_copied_experts_reproduce_dense_output():
    cfg, dense = _trained_like_dense(seed=21)
    moe, report = upcycle_dense_to_moe(dense, cfg, initialization_strategy="copy", seed=21)
    assert report.all_copied_exactly
    x = torch.randn(32, cfg.input_dim)
    dense.eval()
    moe.eval()
    with torch.no_grad():
        dense_y = dense(x)
        moe_y = moe(x)
    assert dense_y.shape == moe_y.shape
    assert torch.allclose(dense_y, moe_y, rtol=1e-5, atol=1e-5)


def test_perturb_strategy_breaks_exact_equality():
    cfg, dense = _trained_like_dense(seed=3)
    moe, report = upcycle_dense_to_moe(
        dense, cfg.replaced(perturb_std=0.05), initialization_strategy="perturb", seed=3
    )
    assert report.strategy == "perturb"
    assert not report.all_copied_exactly
    x = torch.randn(16, cfg.input_dim)
    with torch.no_grad():
        delta = (dense(x) - moe(x)).abs().mean().item()
    assert delta > 0


def test_unimplemented_strategy_is_rejected():
    try:
        get_strategy("partition")
    except ValueError as exc:
        assert "not implemented" in str(exc).lower() or "partition" in str(exc).lower()
    else:
        raise AssertionError("partition should be rejected")


def test_strict_linear_upcycling_preserves_output():
    torch.manual_seed(0)
    cfg = tiny_config(architecture="strict_linear", num_experts=4, top_k=2)
    dense = build_dense_model(cfg)
    with torch.no_grad():
        dense.linear.weight.add_(0.5)
    moe, report = upcycle_dense_to_moe(dense, cfg, initialization_strategy="copy", seed=0)
    assert report.all_copied_exactly
    x = torch.randn(20, cfg.input_dim)
    with torch.no_grad():
        assert torch.allclose(dense(x), moe(x), rtol=1e-5, atol=1e-5)
