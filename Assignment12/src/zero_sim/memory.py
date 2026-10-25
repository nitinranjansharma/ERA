"""Closed-form training-state memory accounting.

All 30-billion-parameter results are scalar arithmetic. Nothing in this
module allocates a tensor, let alone a 30B-parameter tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    BYTES_PER_GB,
    BYTES_PER_GIB,
    COMM_VOLUME_MULTIPLIER,
    DEFAULT_BYTES_PER_PARAM,
    DEFAULT_GPU_CAPACITY_GB,
    DEFAULT_GRAD_BYTES,
    DEFAULT_MASTER_BYTES,
    DEFAULT_MOMENT_BYTES,
    DEFAULT_NUM_PARAMS,
    DEFAULT_WEIGHT_BYTES,
    SimulationConfig,
    gpu_capacity_gib,
    validate_num_params,
    validate_strategy,
    validate_world_size,
)


def gb_to_bytes(gb: float) -> float:
    return gb * BYTES_PER_GB


def gib_to_bytes(gib: float) -> float:
    return gib * BYTES_PER_GIB


def bytes_to_gb(num_bytes: float) -> float:
    return num_bytes / BYTES_PER_GB


def bytes_to_gib(num_bytes: float) -> float:
    return num_bytes / BYTES_PER_GIB


def round_gib(value: float, digits: int = 1) -> float:
    return round(float(value), digits)


def bytes_per_parameter(
    strategy: str,
    world_size: int,
    weight_bytes: int = DEFAULT_WEIGHT_BYTES,
    grad_bytes: int = DEFAULT_GRAD_BYTES,
    master_bytes: int = DEFAULT_MASTER_BYTES,
    moment_bytes: int = DEFAULT_MOMENT_BYTES,
) -> float:
    """Bytes of training state stored on each GPU, per parameter.

    Assignment formulas (defaults: 2 + 2 + 4 + 8 = 16 bytes/parameter):

    * Data parallelism: ``16``
    * ZeRO-1: ``4 + 12 / N``  (replicated weights+grads, sharded optimizer)
    * ZeRO-2: ``2 + 14 / N``  (replicated weights, sharded grads+optimizer)
    * ZeRO-3: ``16 / N``      (everything sharded)
    """
    strategy = validate_strategy(strategy)
    world_size = validate_world_size(world_size)
    w, g, m, o = weight_bytes, grad_bytes, master_bytes, moment_bytes
    if strategy == "dp":
        return float(w + g + m + o)
    if strategy == "zero1":
        return float(w + g) + float(m + o) / world_size
    if strategy == "zero2":
        return float(w) + float(g + m + o) / world_size
    return float(w + g + m + o) / world_size


def memory_per_gpu_bytes(
    strategy: str,
    world_size: int,
    num_params: int = DEFAULT_NUM_PARAMS,
    **byte_widths: int,
) -> float:
    validate_num_params(num_params)
    return num_params * bytes_per_parameter(strategy, world_size, **byte_widths)


def memory_per_gpu_gib(
    strategy: str,
    world_size: int,
    num_params: int = DEFAULT_NUM_PARAMS,
    **byte_widths: int,
) -> float:
    return bytes_to_gib(memory_per_gpu_bytes(strategy, world_size, num_params, **byte_widths))


def memory_per_gpu_gb(
    strategy: str,
    world_size: int,
    num_params: int = DEFAULT_NUM_PARAMS,
    **byte_widths: int,
) -> float:
    return bytes_to_gb(memory_per_gpu_bytes(strategy, world_size, num_params, **byte_widths))


@dataclass(frozen=True)
class MemoryBreakdown:
    strategy: str
    world_size: int
    num_params: int
    weights_gib: float
    gradients_gib: float
    master_weights_gib: float
    optimizer_moments_gib: float
    training_state_gib: float
    activation_gib: float
    runtime_overhead_gib: float
    total_gib: float
    capacity_gib: float
    headroom_gib: float
    fits: bool
    bytes_per_param: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "strategy": self.strategy,
            "world_size": self.world_size,
            "num_params": self.num_params,
            "weights_gib": self.weights_gib,
            "gradients_gib": self.gradients_gib,
            "master_weights_gib": self.master_weights_gib,
            "optimizer_moments_gib": self.optimizer_moments_gib,
            "training_state_gib": self.training_state_gib,
            "activation_gib": self.activation_gib,
            "runtime_overhead_gib": self.runtime_overhead_gib,
            "total_gib": self.total_gib,
            "capacity_gib": self.capacity_gib,
            "headroom_gib": self.headroom_gib,
            "fits": self.fits,
            "bytes_per_param": self.bytes_per_param,
        }


def _state_bytes_on_rank(
    total_bytes: float,
    world_size: int,
    sharded: bool,
) -> float:
    return total_bytes / world_size if sharded else total_bytes


def memory_breakdown(
    strategy: str,
    world_size: int,
    num_params: int = DEFAULT_NUM_PARAMS,
    activation_gib: float = 0.0,
    runtime_overhead_gib: float = 0.0,
    gpu_capacity_gb: float = DEFAULT_GPU_CAPACITY_GB,
    weight_bytes: int = DEFAULT_WEIGHT_BYTES,
    grad_bytes: int = DEFAULT_GRAD_BYTES,
    master_bytes: int = DEFAULT_MASTER_BYTES,
    moment_bytes: int = DEFAULT_MOMENT_BYTES,
) -> MemoryBreakdown:
    """Per-GPU training-state breakdown plus optional extras.

    Activations and runtime overhead are opt-in estimates. They are *not*
    part of the assignment's 16-bytes-per-parameter identity; they exist to
    show why a theoretical training-state fit is not a deployment guarantee.
    """
    strategy = validate_strategy(strategy)
    world_size = validate_world_size(world_size)
    validate_num_params(num_params)
    if activation_gib < 0 or runtime_overhead_gib < 0:
        raise ValueError("activation_gib and runtime_overhead_gib must be >= 0")

    shard_weights = strategy == "zero3"
    shard_grads = strategy in {"zero2", "zero3"}
    shard_optimizer = strategy != "dp"

    weights = _state_bytes_on_rank(num_params * weight_bytes, world_size, shard_weights)
    grads = _state_bytes_on_rank(num_params * grad_bytes, world_size, shard_grads)
    master = _state_bytes_on_rank(num_params * master_bytes, world_size, shard_optimizer)
    moments = _state_bytes_on_rank(num_params * moment_bytes, world_size, shard_optimizer)

    training_state = weights + grads + master + moments
    training_state_gib = bytes_to_gib(training_state)
    capacity = gpu_capacity_gib(gpu_capacity_gb)
    total = training_state_gib + activation_gib + runtime_overhead_gib
    headroom = capacity - total
    return MemoryBreakdown(
        strategy=strategy,
        world_size=world_size,
        num_params=num_params,
        weights_gib=bytes_to_gib(weights),
        gradients_gib=bytes_to_gib(grads),
        master_weights_gib=bytes_to_gib(master),
        optimizer_moments_gib=bytes_to_gib(moments),
        training_state_gib=training_state_gib,
        activation_gib=activation_gib,
        runtime_overhead_gib=runtime_overhead_gib,
        total_gib=total,
        capacity_gib=capacity,
        headroom_gib=headroom,
        fits=total <= capacity,
        bytes_per_param=bytes_per_parameter(
            strategy,
            world_size,
            weight_bytes=weight_bytes,
            grad_bytes=grad_bytes,
            master_bytes=master_bytes,
            moment_bytes=moment_bytes,
        ),
    )


def memory_fit(
    strategy: str,
    world_size: int,
    num_params: int = DEFAULT_NUM_PARAMS,
    activation_gib: float = 0.0,
    runtime_overhead_gib: float = 0.0,
    gpu_capacity_gb: float = DEFAULT_GPU_CAPACITY_GB,
    **byte_widths: int,
) -> tuple[bool, float]:
    """Return ``(fits, headroom_gib)`` for the requested configuration."""
    breakdown = memory_breakdown(
        strategy,
        world_size,
        num_params=num_params,
        activation_gib=activation_gib,
        runtime_overhead_gib=runtime_overhead_gib,
        gpu_capacity_gb=gpu_capacity_gb,
        **byte_widths,
    )
    return breakdown.fits, breakdown.headroom_gib


def activation_memory_gib(
    tokens_per_gpu: int,
    hidden_size: int,
    num_layers: int,
    bytes_per_activation: int = 2,
    overhead_factor: float = 1.0,
) -> float:
    """Optional, very rough activation-memory estimate.

    ``tokens_per_gpu * hidden_size * num_layers * bytes`` is only a sketch.
    Real transformers also keep residuals, attention scores, and checkpointed
    tensors. The result is reported separately from training-state memory.
    """
    if min(tokens_per_gpu, hidden_size, num_layers, bytes_per_activation) <= 0:
        raise ValueError("activation estimate inputs must be positive")
    if overhead_factor < 0:
        raise ValueError("overhead_factor must be >= 0")
    return bytes_to_gib(
        tokens_per_gpu * hidden_size * num_layers * bytes_per_activation * overhead_factor
    )


def ladder_rows(
    world_sizes: tuple[int, ...] | list[int],
    config: SimulationConfig | None = None,
) -> list[MemoryBreakdown]:
    cfg = config or SimulationConfig()
    rows: list[MemoryBreakdown] = []
    for n in world_sizes:
        for strategy in COMM_VOLUME_MULTIPLIER:
            rows.append(
                memory_breakdown(
                    strategy,
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
            )
    return rows


# Sanity-check identity used by tests and the notebook smoke test.
assert DEFAULT_BYTES_PER_PARAM == 16
