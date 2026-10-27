"""Load fractions, MaxVio, dead experts, and parameter accounting."""

from __future__ import annotations

import torch

from src.config import tiny_config
from src.dense_model import build_dense_model, count_parameters
from src.metrics import routing_metrics_from_indices
from src.moe_model import parameter_report
from src.upcycling import upcycle_dense_to_moe


def test_selected_load_fractions_sum_to_top_k_and_ideal_is_half():
    # 4 samples, top-2, perfectly balanced over 4 experts: each expert selected twice.
    indices = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
        ]
    )
    metrics = routing_metrics_from_indices(indices, num_experts=4, top_k=2)
    assert metrics["ideal_selected_load"] == 0.5
    assert abs(metrics["selected_load_sum"] - 2.0) < 1e-9
    assert metrics["selected_load_fractions"] == [0.5, 0.5, 0.5, 0.5]
    assert metrics["dead_experts"] == 0
    assert metrics["max_vio"] == 0.0 or abs(metrics["max_vio"]) < 1e-12


def test_dead_experts_and_max_vio():
    # All mass on expert 0 (top-1 of 4).
    indices = torch.tensor([[0], [0], [0], [0]])
    metrics = routing_metrics_from_indices(indices, num_experts=4, top_k=1)
    assert metrics["dead_experts"] == 3
    assert metrics["expert_counts"][0] == 4
    # mean load = 1, max = 4, MaxVio = 3
    assert abs(metrics["max_vio"] - 3.0) < 1e-9


def test_parameter_counts_active_less_than_total():
    cfg = tiny_config(architecture="ffn", num_experts=4, top_k=2)
    dense = build_dense_model(cfg)
    moe, _ = upcycle_dense_to_moe(dense, cfg, initialization_strategy="copy", seed=0)
    report = parameter_report(dense, moe)
    assert report["moe_total"] > report["dense_total"]
    assert report["moe_active_per_sample"] < report["moe_total"]
    breakdown = moe.parameter_breakdown()
    assert (
        breakdown["per_expert"] * 4
        + breakdown["router"]
        + breakdown["shared_layers"]
        + breakdown["shared_expert"]
        == breakdown["total"]
    )
    assert breakdown["active_per_sample"] == (
        breakdown["shared"] + breakdown["router"] + 2 * breakdown["per_expert"]
    )
    assert count_parameters(dense) == report["dense_total"]
