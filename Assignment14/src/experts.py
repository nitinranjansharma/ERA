"""Independent expert feed-forward modules.

Each expert is a standalone ``nn.Module``. The default expert reproduces the
dense model's inner feed-forward block so sparse upcycling can copy weights
parameter-for-parameter.

In a transformer, a router scores *tokens*. This educational experiment
routes *samples* of a tabular regression problem: one row is one routed
unit. The tensor shapes and the top-k algorithm are the same; only the
interpretation of the batch axis changes.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from src.config import ExperimentConfig


def get_activation(name: str) -> nn.Module:
    """Return a stateless activation module by name."""
    key = name.lower()
    mapping: dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if key not in mapping:
        raise ValueError(f"Unknown activation {name!r}. Expected one of {tuple(mapping)}")
    return mapping[key]()


def apply_activation(name: str, hidden: torch.Tensor) -> torch.Tensor:
    """Functional activation used when a module would be awkward."""
    key = name.lower()
    if key == "gelu":
        return F.gelu(hidden)
    if key == "relu":
        return F.relu(hidden)
    if key == "silu":
        return F.silu(hidden)
    if key == "tanh":
        return torch.tanh(hidden)
    raise ValueError(f"Unknown activation {name!r}")


class ExpertFFN(nn.Module):
    """Two-layer feed-forward expert: Linear -> activation -> Linear.

    Shapes:
        x: [batch, hidden_dim]
        fc1: [batch, ffn_hidden_dim]
        fc2: [batch, hidden_dim]
    """

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        if hidden_dim < 1 or ffn_hidden_dim < 1:
            raise ValueError("hidden_dim and ffn_hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.activation_name = activation
        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.activation = get_activation(activation)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [batch, hidden_dim]
        if hidden.ndim != 2 or hidden.size(-1) != self.hidden_dim:
            raise ValueError(
                f"ExpertFFN expected [batch, {self.hidden_dim}], got {tuple(hidden.shape)}"
            )
        intermediate = self.activation(self.fc1(hidden))  # [batch, ffn_hidden_dim]
        return self.fc2(intermediate)  # [batch, hidden_dim]


class ExpertLinear(nn.Module):
    """Single linear expert used by ``architecture='strict_linear'``.

    Shapes:
        x: [batch, input_dim]
        out: [batch, output_dim]
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.input_dim:
            raise ValueError(
                f"ExpertLinear expected [batch, {self.input_dim}], got {tuple(features.shape)}"
            )
        return self.linear(features)  # [batch, output_dim]


def build_expert(config: ExperimentConfig) -> nn.Module:
    """Construct one expert matching the configured dense block."""
    if config.architecture == "strict_linear":
        return ExpertLinear(config.input_dim, config.output_dim)
    return ExpertFFN(config.hidden_dim, config.ffn_hidden_dim, config.activation)
