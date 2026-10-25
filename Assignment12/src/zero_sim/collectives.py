"""Collective operations on small tensors plus analytical large-model estimates.

The tensor helpers demonstrate numerical correctness of all-reduce,
reduce-scatter, and all-gather. They operate on toy tensors that fit in
a Colab CPU runtime.

Large-model communication volumes (multiples of P = 60 GB) are computed
analytically. They are *not* measured from the Colab runtime and do not
reflect real NVLink or InfiniBand performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import (
    COMM_VOLUME_MULTIPLIER,
    DEFAULT_PARAM_COPY_P_GB,
    INTER_NODE_BANDWIDTH_GB_S,
    INTRA_NODE_BANDWIDTH_GB_S,
    validate_strategy,
)


def all_reduce(tensors: list[torch.Tensor], op: str = "sum") -> list[torch.Tensor]:
    """Return a replica of the reduced tensor on every rank.

    ``op='sum'`` matches a sum all-reduce. ``op='mean'`` divides by world size
    after the sum. The input list is not mutated.
    """
    if not tensors:
        raise ValueError("all_reduce requires at least one tensor")
    _check_same_shape(tensors)
    stacked = torch.stack([t.detach().to(dtype=torch.float64) for t in tensors])
    if op == "sum":
        reduced = stacked.sum(dim=0)
    elif op == "mean":
        reduced = stacked.mean(dim=0)
    else:
        raise ValueError(f"unsupported all-reduce op: {op!r}")
    out_dtype = tensors[0].dtype
    out_device = tensors[0].device
    reduced = reduced.to(dtype=out_dtype, device=out_device)
    return [reduced.clone() for _ in tensors]


def reduce_scatter(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Sum matching shards across ranks, then keep only the local shard.

    Each input tensor is treated as a concatenation of ``world_size`` equal
    (or nearly equal) chunks along dim 0. Rank ``r`` receives chunk ``r`` of
    the reduced tensor.
    """
    if not tensors:
        raise ValueError("reduce_scatter requires at least one tensor")
    _check_same_shape(tensors)
    world = len(tensors)
    stacked = torch.stack([t.detach().to(dtype=torch.float64) for t in tensors])
    reduced = stacked.sum(dim=0)
    chunks = _split_dim0(reduced, world)
    out_dtype = tensors[0].dtype
    out_device = tensors[0].device
    return [chunk.to(dtype=out_dtype, device=out_device).contiguous().clone() for chunk in chunks]


def all_gather(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """Concatenate shards in rank order and replicate the full tensor."""
    if not shards:
        raise ValueError("all_gather requires at least one tensor")
    gathered = torch.cat([s.detach() for s in shards], dim=0)
    return [gathered.clone() for _ in shards]


def all_reduce_via_reduce_scatter_all_gather(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Reference identity: all-reduce(sum) == all-gather(reduce-scatter)."""
    return all_gather(reduce_scatter(tensors))


def _check_same_shape(tensors: list[torch.Tensor]) -> None:
    shape = tensors[0].shape
    for i, tensor in enumerate(tensors):
        if tensor.shape != shape:
            raise ValueError(f"tensor {i} has shape {tuple(tensor.shape)}, expected {tuple(shape)}")


def _split_dim0(tensor: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    n = tensor.shape[0]
    if n < world_size:
        raise ValueError(
            f"dim 0 length {n} is smaller than world_size {world_size}; "
            "use a larger toy tensor"
        )
    base, rem = divmod(n, world_size)
    chunks: list[torch.Tensor] = []
    start = 0
    for rank in range(world_size):
        extra = 1 if rank < rem else 0
        end = start + base + extra
        chunks.append(tensor[start:end])
        start = end
    return chunks


def shard_tensor(tensor: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """Split ``tensor`` along dim 0 into ``world_size`` contiguous shards."""
    return [chunk.contiguous().clone() for chunk in _split_dim0(tensor, world_size)]


def reconstruct_from_shards(shards: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(shards, dim=0)


def communication_volume_gb(
    strategy: str,
    p_gb: float = DEFAULT_PARAM_COPY_P_GB,
) -> float:
    """Simplified per-step communication volume as a multiple of P.

    * Data parallelism: 2P  (ring all-reduce of gradients)
    * ZeRO-1: 2P           (still all-reduces gradients)
    * ZeRO-2: 2P           (reduce-scatter grads + all-gather params)
    * ZeRO-3: 3P           (param all-gather for FWD and BWD + reduce-scatter)
    """
    strategy = validate_strategy(strategy)
    if p_gb <= 0:
        raise ValueError("P must be positive")
    return COMM_VOLUME_MULTIPLIER[strategy] * p_gb


def estimate_transfer_seconds(volume_gb: float, bandwidth_gb_s: float) -> float:
    if bandwidth_gb_s <= 0:
        raise ValueError("bandwidth must be positive")
    if volume_gb < 0:
        raise ValueError("volume cannot be negative")
    return volume_gb / bandwidth_gb_s


@dataclass(frozen=True)
class CommunicationEstimate:
    strategy: str
    p_gb: float
    volume_gb: float
    intra_node_seconds: float
    inter_node_seconds: float
    intra_node_bw_gb_s: float
    inter_node_bw_gb_s: float
    multiplier: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "strategy": self.strategy,
            "p_gb": self.p_gb,
            "volume_gb": self.volume_gb,
            "intra_node_seconds": self.intra_node_seconds,
            "inter_node_seconds": self.inter_node_seconds,
            "intra_node_bw_gb_s": self.intra_node_bw_gb_s,
            "inter_node_bw_gb_s": self.inter_node_bw_gb_s,
            "multiplier": self.multiplier,
        }


def communication_estimate(
    strategy: str,
    p_gb: float = DEFAULT_PARAM_COPY_P_GB,
    intra_node_bw_gb_s: float = INTRA_NODE_BANDWIDTH_GB_S,
    inter_node_bw_gb_s: float = INTER_NODE_BANDWIDTH_GB_S,
) -> CommunicationEstimate:
    strategy = validate_strategy(strategy)
    volume = communication_volume_gb(strategy, p_gb)
    return CommunicationEstimate(
        strategy=strategy,
        p_gb=p_gb,
        volume_gb=volume,
        intra_node_seconds=estimate_transfer_seconds(volume, intra_node_bw_gb_s),
        inter_node_seconds=estimate_transfer_seconds(volume, inter_node_bw_gb_s),
        intra_node_bw_gb_s=intra_node_bw_gb_s,
        inter_node_bw_gb_s=inter_node_bw_gb_s,
        multiplier=COMM_VOLUME_MULTIPLIER[strategy],
    )


def communication_to_compute_ratio(
    comm_seconds: float,
    compute_seconds: float,
) -> float:
    if compute_seconds <= 0:
        raise ValueError("compute_seconds must be positive")
    if comm_seconds < 0:
        raise ValueError("comm_seconds cannot be negative")
    return comm_seconds / compute_seconds


def estimate_compute_seconds(
    num_params: int,
    tokens_per_step: int,
    peak_tflops: float,
    utilization: float = 0.4,
    flops_per_token_factor: float = 6.0,
    world_size: int = 1,
) -> float:
    """Rough transformer compute time: ``6ND`` FLOPs / (N_gpu * peak * util).

    This is an order-of-magnitude sketch used only for the communication-to-
    compute ratio plot. It is not a profiler measurement.
    """
    if min(num_params, tokens_per_step, world_size) <= 0:
        raise ValueError("num_params, tokens_per_step, and world_size must be positive")
    if peak_tflops <= 0 or utilization <= 0:
        raise ValueError("peak_tflops and utilization must be positive")
    flops = flops_per_token_factor * num_params * tokens_per_step
    available = world_size * peak_tflops * 1e12 * utilization
    return flops / available
