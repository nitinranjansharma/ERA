"""Sparse top-k Mixture-of-Experts regressors.

The forward pass evaluates only the experts that a sample actually selected.
Unselected experts are skipped, not run "for simplicity".

This is a pedagogical single-device implementation. Production MoE training
dispatches tokens across expert-parallel ranks with AllToAll; that communication
path is not implemented here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.config import ExperimentConfig
from src.dense_model import count_parameters
from src.experts import ExpertFFN, ExpertLinear, build_expert, get_activation
from src.router import RoutingResult, TopKRouter, expert_capacity


def sparse_expert_combine(
    hidden: torch.Tensor,
    experts: nn.ModuleList,
    routing: RoutingResult,
) -> torch.Tensor:
    """Combine selected expert outputs with normalized router weights.

    ``hidden``: [batch, expert_input_dim]
    Each expert maps [n_selected, expert_input_dim] -> [n_selected, expert_output_dim]
    Returns: [batch, expert_output_dim]
    """
    if hidden.ndim != 2:
        raise ValueError(f"sparse_expert_combine expected a rank-2 tensor, got {tuple(hidden.shape)}")
    batch_size = hidden.size(0)
    sample_output: torch.Tensor | None = None

    for expert_id, expert in enumerate(experts):
        match = routing.selected_indices == expert_id  # [batch, top_k]
        if routing.dropped_mask is not None:
            match = match & ~routing.dropped_mask
        sample_mask = match.any(dim=-1)  # [batch]
        if not bool(sample_mask.any()):
            continue
        # A sample selects an expert at most once (distinct top-k). Sum is the weight.
        weights = (routing.selected_weights * match.to(dtype=routing.selected_weights.dtype)).sum(
            dim=-1
        )  # [batch]
        selected_hidden = hidden[sample_mask]  # [n_sel, dim]
        expert_out = expert(selected_hidden)  # [n_sel, out_dim]
        weighted = expert_out * weights[sample_mask].unsqueeze(-1)
        if sample_output is None:
            sample_output = hidden.new_zeros(batch_size, expert_out.size(-1))
        sample_output[sample_mask] = sample_output[sample_mask] + weighted

    if sample_output is None:
        # Every assignment was dropped. Return zeros with a best-effort output dim.
        out_dim = _infer_expert_output_dim(experts[0], hidden)
        return hidden.new_zeros(batch_size, out_dim)
    return sample_output


def _infer_expert_output_dim(expert: nn.Module, hidden: torch.Tensor) -> int:
    if isinstance(expert, ExpertFFN):
        return expert.hidden_dim
    if isinstance(expert, ExpertLinear):
        return expert.output_dim
    return hidden.size(-1)


class MoERegressor(nn.Module):
    """FFN-upcycled MoE: shared projections plus a sparse expert block."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        if config.architecture != "ffn":
            raise ValueError("MoERegressor requires architecture='ffn'")
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.pre_ffn_activation = get_activation(config.activation)
        self.use_shared_expert = config.use_shared_expert
        self.router = TopKRouter(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            routed_scaling_factor=config.routed_scaling_factor,
            use_bias_for_selection=config.use_bias_balancing,
            score=config.router_score,
        )
        self.experts = nn.ModuleList([build_expert(config) for _ in range(config.num_experts)])
        self.shared_expert = build_expert(config) if config.use_shared_expert else None
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.input_dim:
            raise ValueError(
                f"MoERegressor expected [batch, {self.input_dim}], got {tuple(features.shape)}"
            )
        return self.pre_ffn_activation(self.input_proj(features))

    def moe_block(self, hidden: torch.Tensor) -> tuple[torch.Tensor, RoutingResult]:
        capacity = None
        if self.capacity_factor is not None:
            capacity = expert_capacity(
                batch_size=hidden.size(0),
                top_k=self.top_k,
                num_experts=self.num_experts,
                capacity_factor=self.capacity_factor,
            )
        routing = self.router(hidden, capacity=capacity)
        combined = sparse_expert_combine(hidden, self.experts, routing)
        if self.shared_expert is not None:
            combined = combined + self.shared_expert(hidden)
        return combined, routing

    def forward(
        self,
        features: torch.Tensor,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RoutingResult]:
        hidden = self.encode(features)  # [batch, hidden_dim]
        transformed, routing = self.moe_block(hidden)
        predictions = self.output_proj(transformed)  # [batch, output_dim]
        if return_routing:
            return predictions, routing
        return predictions

    def parameter_breakdown(self) -> dict[str, int]:
        per_expert = count_parameters(self.experts[0]) if len(self.experts) else 0
        shared_layers = count_parameters(self.input_proj) + count_parameters(self.output_proj)
        shared_expert = count_parameters(self.shared_expert) if self.shared_expert is not None else 0
        router = count_parameters(self.router)
        total = count_parameters(self)
        active = shared_layers + shared_expert + router + self.top_k * per_expert
        return {
            "total": total,
            "shared": shared_layers,
            "shared_layers": shared_layers,
            "shared_expert": shared_expert,
            "router": router,
            "per_expert": per_expert,
            "all_experts": per_expert * self.num_experts,
            "active_per_sample": active,
            "inactive_expert_params": per_expert * (self.num_experts - self.top_k),
        }


class StrictLinearMoE(nn.Module):
    """MoE whose experts are copies of a single ``nn.Linear``."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        if config.architecture != "strict_linear":
            raise ValueError("StrictLinearMoE requires architecture='strict_linear'")
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.use_shared_expert = config.use_shared_expert
        self.router = TopKRouter(
            hidden_dim=config.input_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            routed_scaling_factor=config.routed_scaling_factor,
            use_bias_for_selection=config.use_bias_balancing,
            score=config.router_score,
        )
        self.experts = nn.ModuleList([build_expert(config) for _ in range(config.num_experts)])
        self.shared_expert = build_expert(config) if config.use_shared_expert else None

    def forward(
        self,
        features: torch.Tensor,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RoutingResult]:
        if features.ndim != 2 or features.size(-1) != self.input_dim:
            raise ValueError(
                f"StrictLinearMoE expected [batch, {self.input_dim}], got {tuple(features.shape)}"
            )
        capacity = None
        if self.capacity_factor is not None:
            capacity = expert_capacity(
                batch_size=features.size(0),
                top_k=self.top_k,
                num_experts=self.num_experts,
                capacity_factor=self.capacity_factor,
            )
        routing = self.router(features, capacity=capacity)
        predictions = sparse_expert_combine(features, self.experts, routing)
        if self.shared_expert is not None:
            predictions = predictions + self.shared_expert(features)
        if return_routing:
            return predictions, routing
        return predictions

    def parameter_breakdown(self) -> dict[str, int]:
        per_expert = count_parameters(self.experts[0]) if len(self.experts) else 0
        shared_expert = count_parameters(self.shared_expert) if self.shared_expert is not None else 0
        router = count_parameters(self.router)
        total = count_parameters(self)
        active = shared_expert + router + self.top_k * per_expert
        return {
            "total": total,
            "shared": 0,
            "shared_layers": 0,
            "shared_expert": shared_expert,
            "router": router,
            "per_expert": per_expert,
            "all_experts": per_expert * self.num_experts,
            "active_per_sample": active,
            "inactive_expert_params": per_expert * (self.num_experts - self.top_k),
        }


def build_moe_model(config: ExperimentConfig) -> nn.Module:
    if config.architecture == "strict_linear":
        return StrictLinearMoE(config)
    if config.architecture == "ffn":
        return MoERegressor(config)
    raise ValueError(f"Unsupported architecture {config.architecture!r}")


def is_moe_model(model: nn.Module) -> bool:
    return isinstance(model, (MoERegressor, StrictLinearMoE))


def parameter_report(dense: nn.Module, moe: nn.Module) -> dict[str, Any]:
    dense_total = count_parameters(dense)
    moe_break = moe.parameter_breakdown() if hasattr(moe, "parameter_breakdown") else {}
    moe_total = int(moe_break.get("total", count_parameters(moe)))
    active = int(moe_break.get("active_per_sample", moe_total))
    ratio = (moe_total / active) if active else None
    return {
        "dense_total": dense_total,
        "moe_total": moe_total,
        "moe_active_per_sample": active,
        "total_to_active_ratio": ratio,
        "moe_minus_dense": moe_total - dense_total,
        "moe_breakdown": moe_break,
    }
