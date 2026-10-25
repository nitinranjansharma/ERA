"""Virtual cluster topology, shard ownership, and ZeRO stage layout."""

from __future__ import annotations

import torch

from zero_sim.config import STATE_LAYOUT, SimulationConfig
from zero_sim.simulator import VirtualCluster, coverage_is_complete, shard_ranges


def test_default_cluster_has_32_virtual_gpus_on_four_nodes():
    cluster = VirtualCluster(strategy="zero3")
    assert len(cluster.gpus) == 32
    assert cluster.num_nodes == 4
    mapping = cluster.rank_to_node_map()
    assert set(mapping["node_id"]) == {0, 1, 2, 3}
    assert list(mapping["local_gpu_id"]) == list(range(8)) * 4
    gpu0 = cluster.inspect_gpu(0)
    gpu31 = cluster.inspect_gpu(31)
    assert gpu0["node_id"] == 0 and gpu0["local_gpu_id"] == 0
    assert gpu31["node_id"] == 3 and gpu31["local_gpu_id"] == 7


def test_shard_ownership_covers_full_state_without_overlap():
    ranges = shard_ranges(32, 32)
    assert coverage_is_complete(ranges, 32)
    for strategy in STATE_LAYOUT:
        cluster = VirtualCluster(strategy=strategy)
        if "weights" in cluster.sharded_states:
            assert coverage_is_complete(cluster.param_shard_ranges(), 32)
        if "optimizer_moments" in cluster.sharded_states:
            assert coverage_is_complete(cluster.optimizer_shard_ranges(), 32)
        owned = {gpu.rank: gpu.optimizer_shard for gpu in cluster.gpus}
        assert set(owned) == set(range(32))


def test_stage_layout_replicated_versus_sharded():
    dp = VirtualCluster(strategy="dp")
    z1 = VirtualCluster(strategy="zero1")
    z2 = VirtualCluster(strategy="zero2")
    z3 = VirtualCluster(strategy="zero3")
    assert dp.replicated_states == (
        "weights",
        "gradients",
        "master_weights",
        "optimizer_moments",
    )
    assert z1.sharded_states == ("master_weights", "optimizer_moments")
    assert z1.replicated_states == ("weights", "gradients")
    assert z2.sharded_states == ("gradients", "master_weights", "optimizer_moments")
    assert z2.replicated_states == ("weights",)
    assert z3.replicated_states == ()
    assert z3.sharded_states == (
        "weights",
        "gradients",
        "master_weights",
        "optimizer_moments",
    )


def test_toy_shards_reconstruct_original_parameters():
    torch.manual_seed(0)
    params = torch.arange(64, dtype=torch.float32)
    cfg = SimulationConfig(world_size=8, gpus_per_node=4, num_params=30_000_000_000)
    cluster = VirtualCluster(config=cfg, strategy="zero3", toy_params=params)
    rebuilt = cluster.reconstruct_toy_params()
    torch.testing.assert_close(rebuilt, params)
    assert cluster.gpus[0].owned_param_shard.numel() == 8
    replica = VirtualCluster(config=cfg, strategy="dp", toy_params=params)
    torch.testing.assert_close(replica.reconstruct_toy_params(), params)
    assert replica.gpus[0].owned_param_shard.numel() == 64
