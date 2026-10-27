"""Dense baseline regressors.

Two interpretations of "train a Linear model" live here:

* ``ffn`` (default, chapter-aligned): an MLP whose inner two-layer feed-forward
  block is the block that sparse upcycling copies into every expert.
* ``strict_linear``: a single ``nn.Linear``. Upcycling copies that one layer
  into every expert. There is no inner activation, so this is a weaker
  scientific analogue of transformer FFN upcycling, but it matches the
  assignment wording literally.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.config import ExperimentConfig
from src.experts import ExpertFFN, get_activation


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of scalar parameters in ``module``."""
    if trainable_only:
        return sum(param.numel() for param in module.parameters() if param.requires_grad)
    return sum(param.numel() for param in module.parameters())


class DenseRegressor(nn.Module):
    """Compact MLP: input projection, inner FFN (upcycled), output projection.

    Forward:
        x: [batch, input_dim]
        h = activation(input_proj(x))   # [batch, hidden_dim]
        z = ffn(h)                      # [batch, hidden_dim]  <-- converted block
        y = output_proj(z)              # [batch, output_dim]
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        if config.architecture != "ffn":
            raise ValueError("DenseRegressor requires architecture='ffn'")
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.pre_ffn_activation = get_activation(config.activation)
        self.ffn = ExpertFFN(config.hidden_dim, config.ffn_hidden_dim, config.activation)
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Map raw inputs to the FFN hidden space. Shape: [batch, hidden_dim]."""
        self._check_input(features)
        return self.pre_ffn_activation(self.input_proj(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features)  # [batch, hidden_dim]
        transformed = self.ffn(hidden)  # [batch, hidden_dim]
        return self.output_proj(transformed)  # [batch, output_dim]

    def _check_input(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.size(-1) != self.input_dim:
            raise ValueError(
                f"DenseRegressor expected [batch, {self.input_dim}], got {tuple(features.shape)}"
            )

    def parameter_breakdown(self) -> dict[str, int]:
        return {
            "total": count_parameters(self),
            "input_proj": count_parameters(self.input_proj),
            "ffn": count_parameters(self.ffn),
            "output_proj": count_parameters(self.output_proj),
        }


class StrictLinearRegressor(nn.Module):
    """A single ``nn.Linear`` mapping inputs to the regression target."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        if config.architecture != "strict_linear":
            raise ValueError("StrictLinearRegressor requires architecture='strict_linear'")
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.linear = nn.Linear(config.input_dim, config.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.input_dim:
            raise ValueError(
                f"StrictLinearRegressor expected [batch, {self.input_dim}], "
                f"got {tuple(features.shape)}"
            )
        return self.linear(features)  # [batch, output_dim]

    def parameter_breakdown(self) -> dict[str, int]:
        return {
            "total": count_parameters(self),
            "linear": count_parameters(self.linear),
        }


def build_dense_model(config: ExperimentConfig) -> nn.Module:
    """Factory for the configured dense baseline."""
    if config.architecture == "strict_linear":
        return StrictLinearRegressor(config)
    if config.architecture == "ffn":
        return DenseRegressor(config)
    raise ValueError(f"Unsupported architecture {config.architecture!r}")


def dense_block_module(dense_model: nn.Module) -> nn.Module:
    """Return the submodule that upcycling copies into experts."""
    if isinstance(dense_model, DenseRegressor):
        return dense_model.ffn
    if isinstance(dense_model, StrictLinearRegressor):
        return dense_model
    raise TypeError(f"Unsupported dense model type {type(dense_model)!r}")


def model_kind(model: nn.Module) -> str:
    return type(model).__name__


def summarize_dense(model: nn.Module) -> dict[str, Any]:
    breakdown = model.parameter_breakdown() if hasattr(model, "parameter_breakdown") else {}
    return {
        "kind": model_kind(model),
        "parameters": breakdown,
    }
