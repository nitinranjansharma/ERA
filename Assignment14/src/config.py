"""Validated experiment configuration, seeds, and runtime diagnostics.

Default architecture is a compact MLP whose inner feed-forward block is the
source of MoE experts (chapter-aligned sparse upcycling). A second
``strict_linear`` mode converts a single ``nn.Linear`` layer, matching a
literal reading of "train a Linear model".
"""

from __future__ import annotations

import json
import platform
import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch

DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "outputs"

ArchitectureName = Literal["ffn", "strict_linear"]
StrategyName = Literal["copy", "perturb"]
OptimizerName = Literal["adam", "adamw"]
ActivationName = Literal["gelu", "relu", "silu", "tanh"]

VALID_ARCHITECTURES = ("ffn", "strict_linear")
VALID_STRATEGIES = ("copy", "perturb")
VALID_OPTIMIZERS = ("adam", "adamw")
VALID_ACTIVATIONS = ("gelu", "relu", "silu", "tanh")
VALID_ROUTER_SCORES = ("softmax", "sigmoid")
DOCUMENTED_UNIMPLEMENTED_STRATEGIES = ("partition", "drop_upcycling")

# Primary success check: MoE task loss must fall from its own pre-fine-tune
# baseline by at least this absolute amount (full-dataset eval MSE).
LOSS_REDUCTION_ABS_TOL = 1.0e-4


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _positive_float(name: str, value: Any, minimum: float = 0.0, strict: bool = True) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if strict and number <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {value}")
    if not strict and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value}")
    return number


def _non_negative_float(name: str, value: Any) -> float:
    return _positive_float(name, value, minimum=0.0, strict=False)


def _choice(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


@dataclass
class ExperimentConfig:
    """Single validated configuration object. No hidden globals."""

    # Data
    input_dim: int = 8
    n_train: int = 1024
    n_val: int = 256
    n_test: int = 256
    noise_std: float = 0.05
    batch_size: int = 64

    # Dense / expert architecture
    hidden_dim: int = 32
    ffn_hidden_dim: int = 32
    output_dim: int = 1
    activation: str = "gelu"
    architecture: str = "ffn"

    # MoE
    num_experts: int = 4
    top_k: int = 2
    routed_scaling_factor: float = 1.0
    initialization_strategy: str = "copy"
    perturb_std: float = 0.02
    router_init_std: float = 0.02
    use_shared_expert: bool = False
    router_score: str = "softmax"
    bias_init_noise: float = 0.0

    # Balancing / stability (all optional, off by default for the primary run)
    aux_loss_coef: float = 0.0
    z_loss_coef: float = 0.0
    use_bias_balancing: bool = False
    bias_update_rate: float = 1.0e-3
    capacity_factor: float | None = None  # None => dropless routing

    # Training
    dense_epochs: int = 40
    moe_epochs: int = 40
    ablation_epochs: int = 20
    learning_rate: float = 1.0e-3
    moe_learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    grad_clip_norm: float | None = None
    run_ablation: bool = True
    run_scratch_control: bool = True

    # Runtime
    seed: int = DEFAULT_SEED
    device: str = "auto"
    output_dir: str = DEFAULT_OUTPUT_DIR
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.input_dim = _positive_int("input_dim", self.input_dim)
        self.n_train = _positive_int("n_train", self.n_train)
        self.n_val = _positive_int("n_val", self.n_val)
        self.n_test = _positive_int("n_test", self.n_test)
        self.noise_std = _non_negative_float("noise_std", self.noise_std)
        self.batch_size = _positive_int("batch_size", self.batch_size)
        if self.batch_size > self.n_train:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed n_train ({self.n_train})"
            )

        self.hidden_dim = _positive_int("hidden_dim", self.hidden_dim)
        self.ffn_hidden_dim = _positive_int("ffn_hidden_dim", self.ffn_hidden_dim)
        self.output_dim = _positive_int("output_dim", self.output_dim)
        self.activation = _choice("activation", self.activation, VALID_ACTIVATIONS)
        self.architecture = _choice("architecture", self.architecture, VALID_ARCHITECTURES)

        self.num_experts = _positive_int("num_experts", self.num_experts)
        self.top_k = _positive_int("top_k", self.top_k)
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})"
            )
        self.routed_scaling_factor = _positive_float(
            "routed_scaling_factor", self.routed_scaling_factor, strict=True
        )
        if self.initialization_strategy in DOCUMENTED_UNIMPLEMENTED_STRATEGIES:
            raise ValueError(
                f"initialization_strategy={self.initialization_strategy!r} is documented "
                "in the chapter but not implemented. Use 'copy' or 'perturb'."
            )
        self.initialization_strategy = _choice(
            "initialization_strategy", self.initialization_strategy, VALID_STRATEGIES
        )
        self.perturb_std = _non_negative_float("perturb_std", self.perturb_std)
        self.router_init_std = _positive_float("router_init_std", self.router_init_std)
        if not isinstance(self.use_shared_expert, bool):
            raise ValueError("use_shared_expert must be a bool")
        self.router_score = _choice("router_score", self.router_score, VALID_ROUTER_SCORES)
        self.bias_init_noise = _non_negative_float("bias_init_noise", self.bias_init_noise)

        self.aux_loss_coef = _non_negative_float("aux_loss_coef", self.aux_loss_coef)
        self.z_loss_coef = _non_negative_float("z_loss_coef", self.z_loss_coef)
        if not isinstance(self.use_bias_balancing, bool):
            raise ValueError("use_bias_balancing must be a bool")
        self.bias_update_rate = _non_negative_float("bias_update_rate", self.bias_update_rate)
        if self.capacity_factor is not None:
            self.capacity_factor = _positive_float("capacity_factor", self.capacity_factor)

        self.dense_epochs = _positive_int("dense_epochs", self.dense_epochs)
        self.moe_epochs = _positive_int("moe_epochs", self.moe_epochs)
        self.ablation_epochs = _positive_int("ablation_epochs", self.ablation_epochs)
        self.learning_rate = _positive_float("learning_rate", self.learning_rate)
        self.moe_learning_rate = _positive_float("moe_learning_rate", self.moe_learning_rate)
        self.weight_decay = _non_negative_float("weight_decay", self.weight_decay)
        self.optimizer = _choice("optimizer", self.optimizer, VALID_OPTIMIZERS)
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = _positive_float("grad_clip_norm", self.grad_clip_norm)
        if not isinstance(self.run_ablation, bool):
            raise ValueError("run_ablation must be a bool")
        if not isinstance(self.run_scratch_control, bool):
            raise ValueError("run_scratch_control must be a bool")

        self.seed = _non_negative_int("seed", self.seed)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string such as 'auto', 'cpu', or 'cuda'")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")
        self.num_workers = _non_negative_int("num_workers", self.num_workers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**dict(data))

    def replaced(self, **changes: Any) -> "ExperimentConfig":
        return replace(self, **changes)


def tiny_config(**overrides: Any) -> ExperimentConfig:
    """Small deterministic config for unit tests and smoke runs."""
    values: dict[str, Any] = dict(
        n_train=64,
        n_val=32,
        n_test=32,
        batch_size=16,
        hidden_dim=16,
        ffn_hidden_dim=16,
        dense_epochs=2,
        moe_epochs=2,
        ablation_epochs=1,
        run_ablation=False,
        run_scratch_control=False,
        seed=0,
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. Enables deterministic CuDNN when present."""
    seed = _non_negative_int("seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(spec)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device spec {spec!r}") from exc


def get_runtime_info(device: torch.device | None = None) -> dict[str, Any]:
    resolved = device if device is not None else resolve_device("auto")
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(resolved),
        "cuda_available": torch.cuda.is_available(),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(resolved)
    return info


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "figures": root / "figures",
        "metrics": root / "metrics",
        "checkpoints": root / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
