"""Numerical collective operations and analytical communication estimates."""

from __future__ import annotations

import pytest
import torch

from zero_sim.collectives import (
    all_gather,
    all_reduce,
    all_reduce_via_reduce_scatter_all_gather,
    communication_estimate,
    communication_volume_gb,
    estimate_transfer_seconds,
    reconstruct_from_shards,
    reduce_scatter,
    shard_tensor,
)


def test_all_reduce_sum_and_mean_equivalence():
    torch.manual_seed(0)
    tensors = [torch.randn(6, 4) for _ in range(4)]
    summed = all_reduce(tensors, op="sum")
    averaged = all_reduce(tensors, op="mean")
    expected_sum = torch.stack(tensors).sum(dim=0)
    expected_mean = torch.stack(tensors).mean(dim=0)
    for replica in summed:
        torch.testing.assert_close(replica, expected_sum)
    for replica in averaged:
        torch.testing.assert_close(replica, expected_mean)
        torch.testing.assert_close(replica, summed[0] / 4)


def test_all_reduce_matches_reduce_scatter_then_all_gather():
    torch.manual_seed(1)
    tensors = [torch.arange(16, dtype=torch.float32) + rank for rank in range(8)]
    direct = all_reduce(tensors, op="sum")
    composed = all_reduce_via_reduce_scatter_all_gather(tensors)
    torch.testing.assert_close(direct[0], composed[0])
    for replica in composed:
        torch.testing.assert_close(replica, composed[0])


def test_reduce_scatter_and_all_gather_reconstruct():
    full = torch.arange(24, dtype=torch.float32)
    shards = shard_tensor(full, world_size=6)
    assert len(shards) == 6
    torch.testing.assert_close(reconstruct_from_shards(shards), full)
    gathered = all_gather(shards)
    for replica in gathered:
        torch.testing.assert_close(replica, full)

    rank_tensors = [full + rank for rank in range(6)]
    scattered = reduce_scatter(rank_tensors)
    reconstructed = reconstruct_from_shards(scattered)
    torch.testing.assert_close(reconstructed, torch.stack(rank_tensors).sum(dim=0))


def test_communication_volumes_and_simplified_times():
    assert communication_volume_gb("dp") == 120.0
    assert communication_volume_gb("zero1") == 120.0
    assert communication_volume_gb("zero2") == 120.0
    assert communication_volume_gb("zero3") == 180.0

    two_p_intra = estimate_transfer_seconds(120.0, 450.0)
    three_p_intra = estimate_transfer_seconds(180.0, 450.0)
    two_p_inter = estimate_transfer_seconds(120.0, 50.0)
    three_p_inter = estimate_transfer_seconds(180.0, 50.0)
    assert two_p_intra == pytest.approx(0.27, abs=0.01)
    assert three_p_intra == pytest.approx(0.40, abs=0.01)
    assert two_p_inter == pytest.approx(2.40, abs=0.01)
    assert three_p_inter == pytest.approx(3.60, abs=0.01)

    z3 = communication_estimate("zero3")
    assert z3.volume_gb == 180.0
    assert z3.intra_node_seconds == pytest.approx(0.40, abs=0.01)
    assert z3.inter_node_seconds == pytest.approx(3.60, abs=0.01)
