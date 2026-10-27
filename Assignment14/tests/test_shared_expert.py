"""Shared-expert conversion and scrambled-router invariance."""

from __future__ import annotations

import torch

from src.config import tiny_config
from src.dense_model import build_dense_model
from src.upcycling import conversion_invariance_report, upcycle_dense_to_moe


def test_shared_expert_preserves_dense_output_and_survives_router_scramble():
    torch.manual_seed(0)
    cfg = tiny_config(use_shared_expert=True, num_experts=4, top_k=2, seed=0)
    dense = build_dense_model(cfg)
    with torch.no_grad():
        for param in dense.parameters():
            param.add_(0.4)
    moe, report = upcycle_dense_to_moe(dense, cfg, initialization_strategy="copy", seed=0)
    assert report.used_shared_expert
    assert report.output_scale == 0.5
    assert moe.shared_expert is not None
    x = torch.randn(16, cfg.input_dim)
    info = conversion_invariance_report(dense, moe, x, rtol=1e-5, atol=1e-5)
    assert info["preserved"], info
    assert info["router_irrelevant"], info


def test_scrambled_router_still_matches_without_shared_expert():
    torch.manual_seed(1)
    cfg = tiny_config(use_shared_expert=False, num_experts=8, top_k=2, seed=1)
    dense = build_dense_model(cfg)
    moe, _ = upcycle_dense_to_moe(dense, cfg, initialization_strategy="copy", seed=1)
    x = torch.randn(12, cfg.input_dim)
    info = conversion_invariance_report(dense, moe, x)
    assert info["preserved"]
    assert info["router_irrelevant"]
