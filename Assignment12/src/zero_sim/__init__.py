"""Educational ZeRO / data-parallel training-state simulator.

This package is a teaching tool for ERA V5 Session 12. It accounts for
30-billion-parameter memory using closed-form arithmetic, simulates a
virtual GPU cluster with lightweight metadata, and runs a tiny real
PyTorch model to illustrate gradient averaging.

It is not DeepSpeed, not FSDP, and not a substitute for real
multi-GPU training.
"""

from .config import (
    SimulationConfig,
    ToyTrainingConfig,
    collect_runtime_info,
    get_device,
    set_seed,
)
from .memory import (
    bytes_per_parameter,
    bytes_to_gb,
    bytes_to_gib,
    gb_to_bytes,
    gib_to_bytes,
    memory_breakdown,
    memory_fit,
    memory_per_gpu_bytes,
    memory_per_gpu_gib,
)
from .collectives import (
    all_gather,
    all_reduce,
    communication_volume_gb,
    estimate_transfer_seconds,
    reduce_scatter,
)
from .simulator import VirtualCluster, VirtualGPU
from .model import TinyClassifier, global_batch_size, train_toy_model
from .metrics import strategy_comparison_table

__all__ = [
    "SimulationConfig",
    "ToyTrainingConfig",
    "TinyClassifier",
    "VirtualCluster",
    "VirtualGPU",
    "all_gather",
    "all_reduce",
    "bytes_per_parameter",
    "bytes_to_gb",
    "bytes_to_gib",
    "collect_runtime_info",
    "communication_volume_gb",
    "estimate_transfer_seconds",
    "gb_to_bytes",
    "get_device",
    "gib_to_bytes",
    "global_batch_size",
    "memory_breakdown",
    "memory_fit",
    "memory_per_gpu_bytes",
    "memory_per_gpu_gib",
    "reduce_scatter",
    "set_seed",
    "strategy_comparison_table",
    "train_toy_model",
]

__version__ = "0.1.0"
