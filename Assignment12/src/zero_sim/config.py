"""Validated configuration, unit conventions, and runtime helpers."""

from __future__ import annotations

import platform
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Unit conventions
# ---------------------------------------------------------------------------
# Decimal GB (SI) is used for communication volumes and advertised GPU
# capacity in gigabytes: 1 GB = 10^9 bytes.
# Binary GiB (IEC) is used for memory-capacity accounting: 1 GiB = 2^30 bytes.
# An 80 GB GPU therefore holds 80e9 / 2^30 ≈ 74.51 GiB of addressable memory.
BYTES_PER_GB = 10**9
BYTES_PER_GIB = 1024**3
BYTES_PER_MB = 10**6
BYTES_PER_MIB = 1024**2

# ---------------------------------------------------------------------------
# Assignment baseline: 30B-parameter training-state accounting
# ---------------------------------------------------------------------------
DEFAULT_NUM_PARAMS = 30_000_000_000
DEFAULT_WEIGHT_BYTES = 2  # FP16 / BF16 weights
DEFAULT_GRAD_BYTES = 2  # FP16 / BF16 gradients
DEFAULT_MASTER_BYTES = 4  # FP32 master weights
DEFAULT_MOMENT_BYTES = 8  # two FP32 Adam moments
DEFAULT_BYTES_PER_PARAM = (
    DEFAULT_WEIGHT_BYTES + DEFAULT_GRAD_BYTES + DEFAULT_MASTER_BYTES + DEFAULT_MOMENT_BYTES
)  # 16

DEFAULT_GPU_CAPACITY_GB = 80.0
DEFAULT_PARAM_COPY_P_GB = 60.0  # one FP16/BF16 parameter copy, "P"
DEFAULT_WORLD_SIZE = 32
DEFAULT_GPUS_PER_NODE = 8
DEFAULT_SEED = 42

INTRA_NODE_BANDWIDTH_GB_S = 450.0
INTER_NODE_BANDWIDTH_GB_S = 50.0

VALID_WORLD_SIZES = (1, 2, 4, 8, 16, 32, 64)
STRATEGIES = ("dp", "zero1", "zero2", "zero3")
STRATEGY_LABELS = {
    "dp": "Data parallelism",
    "zero1": "ZeRO-1",
    "zero2": "ZeRO-2",
    "zero3": "ZeRO-3",
}

# Which training states are replicated vs sharded under the simplified model.
STATE_LAYOUT = {
    "dp": {
        "weights": "replicated",
        "gradients": "replicated",
        "master_weights": "replicated",
        "optimizer_moments": "replicated",
    },
    "zero1": {
        "weights": "replicated",
        "gradients": "replicated",
        "master_weights": "sharded",
        "optimizer_moments": "sharded",
    },
    "zero2": {
        "weights": "replicated",
        "gradients": "sharded",
        "master_weights": "sharded",
        "optimizer_moments": "sharded",
    },
    "zero3": {
        "weights": "sharded",
        "gradients": "sharded",
        "master_weights": "sharded",
        "optimizer_moments": "sharded",
    },
}

# Simplified communication volume as a multiple of P.
COMM_VOLUME_MULTIPLIER = {
    "dp": 2.0,
    "zero1": 2.0,
    "zero2": 2.0,
    "zero3": 3.0,
}


def validate_world_size(world_size: int) -> int:
    """Reject world sizes outside the assignment ladder."""
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise ValueError(f"world_size must be an int, got {type(world_size)!r}")
    if world_size not in VALID_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {VALID_WORLD_SIZES}, got {world_size}"
        )
    return world_size


def validate_num_params(num_params: int) -> int:
    """Reject non-positive or non-integer parameter counts."""
    if not isinstance(num_params, int) or isinstance(num_params, bool):
        raise ValueError(f"num_params must be an int, got {type(num_params)!r}")
    if num_params <= 0:
        raise ValueError(f"num_params must be positive, got {num_params}")
    return num_params


def validate_strategy(strategy: str) -> str:
    key = strategy.lower().strip()
    aliases = {
        "data_parallel": "dp",
        "data-parallel": "dp",
        "data parallelism": "dp",
        "zero-1": "zero1",
        "zero_1": "zero1",
        "zero-2": "zero2",
        "zero_2": "zero2",
        "zero-3": "zero3",
        "zero_3": "zero3",
    }
    key = aliases.get(key, key)
    if key not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    return key


def gpu_capacity_gib(capacity_gb: float = DEFAULT_GPU_CAPACITY_GB) -> float:
    """Convert advertised decimal-GB capacity into binary GiB."""
    return capacity_gb * BYTES_PER_GB / BYTES_PER_GIB


def get_device() -> torch.device:
    """Prefer CUDA when a GPU is present; otherwise use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy, and PyTorch for deterministic toy experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    return seed


def collect_runtime_info(device: torch.device | None = None) -> dict[str, Any]:
    """Gather interpreter, library, and accelerator diagnostics."""
    selected = device or get_device()
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(selected),
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    try:
        import pandas as pd

        info["pandas"] = pd.__version__
    except ImportError:
        info["pandas"] = "not installed"
    try:
        import matplotlib

        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        info["matplotlib"] = "not installed"

    if torch.cuda.is_available():
        idx = selected.index if selected.type == "cuda" and selected.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = props.total_memory / BYTES_PER_GB
    return info


@dataclass(frozen=True)
class SimulationConfig:
    """Closed-form 30B-scale accounting configuration.

    All 30B-parameter figures are computed analytically. This object never
    allocates tensors proportional to ``num_params``.
    """

    num_params: int = DEFAULT_NUM_PARAMS
    world_size: int = DEFAULT_WORLD_SIZE
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE
    gpu_capacity_gb: float = DEFAULT_GPU_CAPACITY_GB
    param_copy_p_gb: float = DEFAULT_PARAM_COPY_P_GB
    weight_bytes: int = DEFAULT_WEIGHT_BYTES
    grad_bytes: int = DEFAULT_GRAD_BYTES
    master_bytes: int = DEFAULT_MASTER_BYTES
    moment_bytes: int = DEFAULT_MOMENT_BYTES
    intra_node_bw_gb_s: float = INTRA_NODE_BANDWIDTH_GB_S
    inter_node_bw_gb_s: float = INTER_NODE_BANDWIDTH_GB_S
    activation_gib: float = 0.0
    runtime_overhead_gib: float = 0.0
    seed: int = DEFAULT_SEED
    world_sizes: tuple[int, ...] = VALID_WORLD_SIZES

    def __post_init__(self) -> None:
        validate_num_params(self.num_params)
        validate_world_size(self.world_size)
        if self.gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        if self.world_size % self.gpus_per_node != 0:
            raise ValueError(
                "world_size must be divisible by gpus_per_node "
                f"(got {self.world_size} and {self.gpus_per_node})"
            )
        if self.gpu_capacity_gb <= 0 or self.param_copy_p_gb <= 0:
            raise ValueError("GPU capacity and P must be positive")
        if min(self.weight_bytes, self.grad_bytes, self.master_bytes, self.moment_bytes) <= 0:
            raise ValueError("byte widths must be positive")
        if self.activation_gib < 0 or self.runtime_overhead_gib < 0:
            raise ValueError("activation and overhead estimates cannot be negative")
        for size in self.world_sizes:
            validate_world_size(size)

    @property
    def bytes_per_param(self) -> int:
        return self.weight_bytes + self.grad_bytes + self.master_bytes + self.moment_bytes

    @property
    def gpu_capacity_gib(self) -> float:
        return gpu_capacity_gib(self.gpu_capacity_gb)

    @property
    def num_nodes(self) -> int:
        return self.world_size // self.gpus_per_node


@dataclass(frozen=True)
class ToyTrainingConfig:
    """Tiny real PyTorch experiment — not a 30B model."""

    in_dim: int = 16
    hidden_dim: int = 32
    n_classes: int = 4
    n_samples: int = 48
    batch_size: int = 8
    steps: int = 25
    lr: float = 0.08
    sequences_per_gpu: int = 2
    virtual_gpu_count: int = 4
    accumulation_steps: int = 3
    seed: int = DEFAULT_SEED
    unequal_worker_batches: tuple[int, ...] = (5, 7, 10, 14)

    def __post_init__(self) -> None:
        if min(self.in_dim, self.hidden_dim, self.n_classes, self.n_samples) <= 0:
            raise ValueError("model and data sizes must be positive")
        if min(self.batch_size, self.steps, self.virtual_gpu_count, self.accumulation_steps) <= 0:
            raise ValueError("batch, steps, workers, and accumulation must be positive")
        if self.sequences_per_gpu <= 0:
            raise ValueError("sequences_per_gpu must be positive")
        if sum(self.unequal_worker_batches) <= 0:
            raise ValueError("unequal_worker_batches must sum to a positive count")
