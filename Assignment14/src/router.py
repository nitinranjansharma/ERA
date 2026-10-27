"""Learnable top-k router with softmax scores and selected-weight normalization.

The router consumes sample (or token) representations of shape
``[batch_size, hidden_dim]`` and produces logits of shape
``[batch_size, num_experts]``.

Softmax scores used as *output weights* are computed from the raw logits.
Optional per-expert selection biases are added only to the scores used for
top-k *selection*, matching auxiliary-loss-free balancing (DeepSeek-V3 /
Qwen3 style). Capacity-based dropping is optional and off by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


TIE_BREAK_EPS = 1.0e-6


@dataclass
class RoutingResult:
    """Inspectable routing tensors for one batch.

    Shapes:
        logits:            [batch, num_experts]
        scores:            [batch, num_experts]  (softmax of logits, no bias)
        selection_scores:  [batch, num_experts]  (softmax of logits + bias)
        selected_indices:  [batch, top_k]
        selected_scores:   [batch, top_k]        (raw softmax scores of selected experts)
        selected_weights:  [batch, top_k]        (renormalized, * routed_scaling_factor)
        dropped_mask:      [batch, top_k]        (True where capacity dropped an assignment)
    """

    logits: torch.Tensor
    scores: torch.Tensor
    selection_scores: torch.Tensor
    selected_indices: torch.Tensor
    selected_scores: torch.Tensor
    selected_weights: torch.Tensor
    dropped_mask: torch.Tensor
    dropped_assignments: int
    capacity: int | None


class TopKRouter(nn.Module):
    """Linear router with top-k selection and selected-weight renormalization."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float = 1.0,
        use_bias_for_selection: bool = False,
        score: str = "softmax",
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
        if routed_scaling_factor <= 0:
            raise ValueError(f"routed_scaling_factor must be > 0, got {routed_scaling_factor}")
        if score not in {"softmax", "sigmoid"}:
            raise ValueError(f"score must be 'softmax' or 'sigmoid', got {score!r}")

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.use_bias_for_selection = use_bias_for_selection
        self.score = score
        self.gate = nn.Linear(hidden_dim, num_experts)
        # Selection bias is not a trained parameter; it is updated from load stats.
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(
        self,
        hidden: torch.Tensor,
        capacity: int | None = None,
    ) -> RoutingResult:
        """Route a batch of representations.

        ``hidden``: [batch_size, hidden_dim]
        """
        if hidden.ndim != 2 or hidden.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Router expected [batch, {self.hidden_dim}], got {tuple(hidden.shape)}"
            )
        batch_size = hidden.size(0)
        # Router logits stay in fp32: 16-bit routers have been observed to diverge
        # (Switch Transformer). This experiment is usually fp32 already.
        logits = self.gate(hidden.float())  # [batch, num_experts]
        scores = _score_fn(logits, self.score)
        if self.use_bias_for_selection:
            selection_scores = scores + self.expert_bias.to(dtype=scores.dtype)
        else:
            selection_scores = scores
        logits = logits.to(dtype=hidden.dtype)
        scores = scores.to(dtype=hidden.dtype)
        selection_scores = selection_scores.to(dtype=hidden.dtype)

        selected_indices = self._topk_indices(selection_scores)  # [batch, top_k]
        selected_scores = torch.gather(scores, dim=-1, index=selected_indices)
        selected_weights = _normalize_selected(selected_scores) * self.routed_scaling_factor

        dropped_mask = torch.zeros(
            batch_size, self.top_k, dtype=torch.bool, device=hidden.device
        )
        dropped_assignments = 0
        if capacity is not None:
            dropped_mask, dropped_assignments = apply_capacity_drop(
                selected_indices=selected_indices,
                selected_scores=selected_scores,
                num_experts=self.num_experts,
                capacity=capacity,
            )
            selected_weights = selected_weights.masked_fill(dropped_mask, 0.0)

        return RoutingResult(
            logits=logits,
            scores=scores,
            selection_scores=selection_scores,
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            selected_weights=selected_weights,
            dropped_mask=dropped_mask,
            dropped_assignments=dropped_assignments,
            capacity=capacity,
        )

    def _topk_indices(self, selection_scores: torch.Tensor) -> torch.Tensor:
        """Top-k expert ids. Ties prefer the lower expert index."""
        # Subtract a tiny index-dependent epsilon so equal scores sort stably.
        expert_ids = torch.arange(
            self.num_experts, device=selection_scores.device, dtype=selection_scores.dtype
        )
        tie_broken = selection_scores - expert_ids * TIE_BREAK_EPS
        _values, indices = torch.topk(tie_broken, k=self.top_k, dim=-1, largest=True, sorted=True)
        return indices  # [batch, top_k]


def _score_fn(logits: torch.Tensor, score: str) -> torch.Tensor:
    if score == "sigmoid":
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def _normalize_selected(selected_scores: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Renormalize the k selected softmax scores so each row sums to 1."""
    denom = selected_scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    return selected_scores / denom


def expert_capacity(batch_size: int, top_k: int, num_experts: int, capacity_factor: float) -> int:
    """Switch-style expert capacity for one batch.

    ``capacity = (tokens * top_k / num_experts) * capacity_factor``

    Capacity applies to the number of *assignments* an expert may receive,
    not to unique samples. With top-2 routing a sample occupies one slot in
    each of two experts.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be > 0, got {capacity_factor}")
    raw = (batch_size * top_k / num_experts) * capacity_factor
    return max(1, int(math.ceil(raw)))


def apply_capacity_drop(
    selected_indices: torch.Tensor,
    selected_scores: torch.Tensor,
    num_experts: int,
    capacity: int,
) -> tuple[torch.Tensor, int]:
    """Drop overflow assignments, keeping the highest-score slots per expert.

    Returns
        dropped_mask: [batch, top_k]
        dropped_assignments: int
    """
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    batch_size, top_k = selected_indices.shape
    dropped = torch.zeros(batch_size, top_k, dtype=torch.bool, device=selected_indices.device)
    for expert_id in range(num_experts):
        match = selected_indices == expert_id  # [batch, top_k]
        assigned = int(match.sum().item())
        if assigned <= capacity:
            continue
        scores = selected_scores.masked_fill(~match, float("-inf")).reshape(-1)
        keep = torch.topk(scores, k=capacity, largest=True, sorted=False).indices
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        keep_mask[keep] = True
        keep_mask = keep_mask.view(batch_size, top_k)
        dropped |= match & ~keep_mask
    return dropped, int(dropped.sum().item())
