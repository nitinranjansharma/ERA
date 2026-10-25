"""Lightweight virtual GPU cluster for ZeRO stage demonstrations.

Ranks, nodes, shard ranges, and communication counters are ordinary Python
objects plus tiny tensors. This module never creates 32 OS processes or 32
CUDA devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from .collectives import (
    all_gather,
    communication_estimate,
    reconstruct_from_shards,
    reduce_scatter,
    shard_tensor,
)
from .config import (
    STATE_LAYOUT,
    STRATEGY_LABELS,
    SimulationConfig,
    validate_strategy,
)
from .memory import memory_breakdown


def shard_ranges(n_items: int, world_size: int) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping ranges that cover ``[0, n_items)``."""
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    base, rem = divmod(n_items, world_size)
    ranges: list[tuple[int, int]] = []
    start = 0
    for rank in range(world_size):
        extra = 1 if rank < rem else 0
        end = start + base + extra
        ranges.append((start, end))
        start = end
    if ranges[-1][1] != n_items:
        raise RuntimeError("shard ranges do not cover the full state")
    return ranges


def coverage_is_complete(ranges: list[tuple[int, int]], n_items: int) -> bool:
    """True when ranges are disjoint and union to ``[0, n_items)``."""
    covered: set[int] = set()
    for start, end in ranges:
        if start < 0 or end > n_items or start > end:
            return False
        piece = set(range(start, end))
        if covered & piece:
            return False
        covered |= piece
    return covered == set(range(n_items))


@dataclass
class VirtualGPU:
    rank: int
    node_id: int
    local_gpu_id: int
    strategy: str
    param_shard: tuple[int, int]
    grad_shard: tuple[int, int]
    optimizer_shard: tuple[int, int]
    replicated_states: tuple[str, ...]
    sharded_states: tuple[str, ...]
    memory_gib: float
    comm_counters: dict[str, float] = field(default_factory=dict)
    owned_param_shard: torch.Tensor | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "node_id": self.node_id,
            "local_gpu_id": self.local_gpu_id,
            "strategy": self.strategy,
            "param_shard": self.param_shard,
            "grad_shard": self.grad_shard,
            "optimizer_shard": self.optimizer_shard,
            "replicated_states": list(self.replicated_states),
            "sharded_states": list(self.sharded_states),
            "memory_gib": self.memory_gib,
            "comm_volume_gb": self.comm_counters.get("volume_gb", 0.0),
            "comm_steps": self.comm_counters.get("steps", 0.0),
            "owned_shard_numel": 0
            if self.owned_param_shard is None
            else int(self.owned_param_shard.numel()),
        }


class VirtualCluster:
    """A metadata cluster plus optional tiny parameter shards."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        strategy: str = "zero3",
        toy_params: torch.Tensor | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.strategy = validate_strategy(strategy)
        self.layout = STATE_LAYOUT[self.strategy]
        self.replicated_states = tuple(
            name for name, kind in self.layout.items() if kind == "replicated"
        )
        self.sharded_states = tuple(
            name for name, kind in self.layout.items() if kind == "sharded"
        )
        self.toy_params = toy_params
        self.gpus = self._build_gpus()
        if toy_params is not None:
            self._assign_toy_shards(toy_params)

    def _build_gpus(self) -> list[VirtualGPU]:
        cfg = self.config
        n = cfg.world_size
        # Shard index space is abstract: N units, one per rank, so ownership
        # is well-defined even though we never materialize 30B parameters.
        abstract_ranges = shard_ranges(n, n)
        full_range = (0, n)
        breakdown = memory_breakdown(
            self.strategy,
            n,
            num_params=cfg.num_params,
            activation_gib=cfg.activation_gib,
            runtime_overhead_gib=cfg.runtime_overhead_gib,
            gpu_capacity_gb=cfg.gpu_capacity_gb,
            weight_bytes=cfg.weight_bytes,
            grad_bytes=cfg.grad_bytes,
            master_bytes=cfg.master_bytes,
            moment_bytes=cfg.moment_bytes,
        )
        gpus: list[VirtualGPU] = []
        for rank in range(n):
            node_id = rank // cfg.gpus_per_node
            local_gpu_id = rank % cfg.gpus_per_node
            param_shard = abstract_ranges[rank] if self.layout["weights"] == "sharded" else full_range
            grad_shard = abstract_ranges[rank] if self.layout["gradients"] == "sharded" else full_range
            opt_shard = abstract_ranges[rank] if self.layout["optimizer_moments"] == "sharded" else full_range
            gpus.append(
                VirtualGPU(
                    rank=rank,
                    node_id=node_id,
                    local_gpu_id=local_gpu_id,
                    strategy=self.strategy,
                    param_shard=param_shard,
                    grad_shard=grad_shard,
                    optimizer_shard=opt_shard,
                    replicated_states=self.replicated_states,
                    sharded_states=self.sharded_states,
                    memory_gib=breakdown.training_state_gib,
                    comm_counters={"volume_gb": 0.0, "steps": 0.0},
                )
            )
        return gpus

    def _assign_toy_shards(self, tensor: torch.Tensor) -> None:
        shards = shard_tensor(tensor, self.config.world_size)
        if self.layout["weights"] == "sharded":
            for gpu, shard in zip(self.gpus, shards):
                gpu.owned_param_shard = shard
        else:
            for gpu in self.gpus:
                gpu.owned_param_shard = tensor.clone()

    @property
    def num_nodes(self) -> int:
        return self.config.num_nodes

    def rank_to_node_map(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "rank": gpu.rank,
                    "node_id": gpu.node_id,
                    "local_gpu_id": gpu.local_gpu_id,
                }
                for gpu in self.gpus
            ]
        )

    def inspect_gpu(self, rank: int) -> dict[str, Any]:
        return self.gpus[rank].describe()

    def gpu_table(self) -> pd.DataFrame:
        return pd.DataFrame([gpu.describe() for gpu in self.gpus])

    def shard_ownership_table(self) -> pd.DataFrame:
        rows = []
        for gpu in self.gpus:
            rows.append(
                {
                    "rank": gpu.rank,
                    "node_id": gpu.node_id,
                    "weights": _fmt_range(gpu.param_shard, self.layout["weights"]),
                    "gradients": _fmt_range(gpu.grad_shard, self.layout["gradients"]),
                    "optimizer": _fmt_range(gpu.optimizer_shard, self.layout["optimizer_moments"]),
                    "replicated": ", ".join(gpu.replicated_states) or "—",
                    "sharded": ", ".join(gpu.sharded_states) or "—",
                }
            )
        return pd.DataFrame(rows)

    def param_shard_ranges(self) -> list[tuple[int, int]]:
        return [gpu.param_shard for gpu in self.gpus]

    def optimizer_shard_ranges(self) -> list[tuple[int, int]]:
        return [gpu.optimizer_shard for gpu in self.gpus]

    def reconstruct_toy_params(self) -> torch.Tensor:
        if self.toy_params is None:
            raise RuntimeError("no toy parameter tensor was attached to this cluster")
        if self.layout["weights"] == "sharded":
            shards = [gpu.owned_param_shard for gpu in self.gpus]
            if any(shard is None for shard in shards):
                raise RuntimeError("missing parameter shards")
            return reconstruct_from_shards(shards)  # type: ignore[arg-type]
        assert self.gpus[0].owned_param_shard is not None
        return self.gpus[0].owned_param_shard.clone()

    def simulate_step(self) -> dict[str, float]:
        """Account for one analytical communication step on every virtual GPU."""
        estimate = communication_estimate(
            self.strategy,
            p_gb=self.config.param_copy_p_gb,
            intra_node_bw_gb_s=self.config.intra_node_bw_gb_s,
            inter_node_bw_gb_s=self.config.inter_node_bw_gb_s,
        )
        for gpu in self.gpus:
            gpu.comm_counters["volume_gb"] += estimate.volume_gb
            gpu.comm_counters["steps"] += 1
        return {
            "strategy": self.strategy,
            "volume_gb": estimate.volume_gb,
            "intra_node_seconds": estimate.intra_node_seconds,
            "inter_node_seconds": estimate.inter_node_seconds,
        }

    def aggregate_metrics(self) -> dict[str, Any]:
        breakdown = memory_breakdown(
            self.strategy,
            self.config.world_size,
            num_params=self.config.num_params,
            activation_gib=self.config.activation_gib,
            runtime_overhead_gib=self.config.runtime_overhead_gib,
            gpu_capacity_gb=self.config.gpu_capacity_gb,
        )
        estimate = communication_estimate(self.strategy, p_gb=self.config.param_copy_p_gb)
        return {
            "strategy": self.strategy,
            "label": STRATEGY_LABELS[self.strategy],
            "world_size": self.config.world_size,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.config.gpus_per_node,
            "memory_per_gpu_gib": breakdown.training_state_gib,
            "capacity_gib": breakdown.capacity_gib,
            "headroom_gib": breakdown.headroom_gib,
            "fits": breakdown.fits,
            "comm_volume_gb": estimate.volume_gb,
            "intra_node_seconds": estimate.intra_node_seconds,
            "inter_node_seconds": estimate.inter_node_seconds,
            "replicated_states": list(self.replicated_states),
            "sharded_states": list(self.sharded_states),
        }

    def demonstrate_collectives_on_toy_grads(
        self,
        per_rank_grads: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run the stage-appropriate gradient collective on small tensors."""
        if len(per_rank_grads) != self.config.world_size:
            raise ValueError("expected one gradient tensor per virtual GPU")
        if self.layout["gradients"] == "replicated":
            reduced = all_gather(reduce_scatter(per_rank_grads))
            return {"op": "all_reduce", "result": reduced}
        scattered = reduce_scatter(per_rank_grads)
        return {"op": "reduce_scatter", "result": scattered}


def _fmt_range(shard: tuple[int, int], kind: str) -> str:
    if kind == "replicated":
        return "replicated (full)"
    return f"sharded [{shard[0]}, {shard[1]})"


def compare_strategies(config: SimulationConfig | None = None) -> pd.DataFrame:
    cfg = config or SimulationConfig()
    rows = []
    for strategy in STATE_LAYOUT:
        cluster = VirtualCluster(config=cfg, strategy=strategy)
        rows.append(cluster.aggregate_metrics())
    return pd.DataFrame(rows)
