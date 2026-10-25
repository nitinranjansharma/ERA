"""Decoder-only causal Transformer and parameter-count utilities.

Architecture (GPT-2 Pre-LN style)
---------------------------------
* Learned token embeddings and learned positional embeddings.
* ``n_layer`` residual transformer functions ``F_l``.
* Causal self-attention with an explicit upper-triangular mask.
* GELU MLP with a 4x inner width by default.
* Final LayerNorm and a language-model head.
* Token embeddings and the output head are **tied** by default.

A standard baseline block is the Euler residual

    x_{l+1} = x_l + F_l(x_l)

where ``F_l`` is implemented by :class:`TransformerResidualFn`. The reversible
midpoint stack in ``helpers.reversible`` reuses the same ``F_l`` modules.

Parameter-count rule
--------------------
Trainable parameters are the sum of ``p.numel()`` over ``p.requires_grad``.
Tied ``lm_head.weight`` is the same storage as ``wte.weight`` and is counted
once. The closed-form helper :func:`helpers.config.expected_param_count`
must match that sum.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.config import ExperimentConfig, expected_param_count


class ModelError(ValueError):
    """Raised when tensor shapes, vocab range, or config are invalid."""


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention. Never attends to future tokens."""

    def __init__(self, n_embd: int, n_head: int, bias: bool, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ModelError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.attn_dropout_p = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        # Explicit matmul attention: bitwise-deterministic recomputation for
        # reversible reconstruction. Flash/SDPA kernels are not used.
        scale = self.head_dim ** -0.5
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.ones(seq, seq, device=x.device, dtype=torch.bool).triu(1)
        att = att.masked_fill(causal, torch.finfo(att.dtype).min)
        att = torch.softmax(att, dim=-1)
        if self.attn_dropout_p > 0 and self.training:
            att = F.dropout(att, p=self.attn_dropout_p, training=True)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq, self.n_embd)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, n_inner: int, bias: bool, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, n_inner, bias=bias)
        self.proj = nn.Linear(n_inner, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class TransformerResidualFn(nn.Module):
    """Residual function ``F`` of a Pre-LN transformer block.

    The standard block is ``x + F(x)`` where

        a = Attn(LN1(x))
        F(x) = a + MLP(LN2(x + a))

    Reversible layers apply the same ``F`` inside the midpoint/Euler update
    instead of the Euler skip ``x + F(x)``.
    """

    def __init__(self, config: ExperimentConfig, n_embd: int | None = None) -> None:
        super().__init__()
        d = int(n_embd if n_embd is not None else config.n_embd)
        if d % config.n_head != 0:
            # For Euler half-width blocks, n_head must still divide width.
            n_head = 1
            for candidate in range(config.n_head, 0, -1):
                if d % candidate == 0:
                    n_head = candidate
                    break
        else:
            n_head = config.n_head
        inner = 4 * d if n_embd is not None else int(config.n_inner or 4 * d)
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, config.bias, config.dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = FeedForward(d, inner, config.bias, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_delta = self.attn(self.ln1(x))
        y = x + attn_delta
        mlp_delta = self.mlp(self.ln2(y))
        return attn_delta + mlp_delta


class CausalLM(nn.Module):
    """Decoder-only language model with baseline or reversible stacks."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerResidualFn(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        # GPT residual scaling: each block starts out contributing little.
        scale = 0.02 / math.sqrt(2 * max(config.n_layer, 1))
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        validate_token_batch(idx, self.config)
        batch, seq = idx.shape
        pos = torch.arange(seq, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        return self.drop(x)

    def apply_blocks(self, x: torch.Tensor, plain: bool = False) -> torch.Tensor:
        variant = self.config.variant
        if variant == "baseline":
            from helpers.reversible import apply_residual_fn

            for block in self.blocks:
                x = x + apply_residual_fn(block, x)
            return x
        from helpers.reversible import (
            euler_forward_reference,
            midpoint_forward_reference,
            reversible_stack,
        )

        if plain:
            if variant == "midpoint":
                return midpoint_forward_reference(x, self.blocks, two_h=self.config.two_h)
            if variant == "euler":
                return euler_forward_reference(x, self.blocks, h=self.config.euler_h)
        return reversible_stack(x, self.blocks, self.config)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        plain: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.apply_blocks(self.embed(idx), plain=plain)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ModelError(
                    f"targets shape {tuple(targets.shape)} != input shape {tuple(idx.shape)}"
                )
            logits_f = logits.float() if logits.dtype != torch.float32 else logits
            loss = F.cross_entropy(
                logits_f.reshape(-1, logits_f.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def param_count(self) -> int:
        return count_trainable_parameters(self)

    def param_report(self) -> dict[str, Any]:
        trainable = self.param_count()
        expected = expected_param_count(self.config)
        return {
            "trainable_parameters": trainable,
            "expected_parameters": expected,
            "target_parameters": 20_000_000,
            "variant": self.config.variant,
            "vocab_size": self.config.vocab_size,
            "n_embd": self.config.n_embd,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_inner": self.config.n_inner,
            "block_size": self.config.block_size,
            "tie_embeddings": self.config.tie_embeddings,
            "bias": self.config.bias,
            "within_tolerance": _within_target(trainable),
        }


def _within_target(count: int, target: int = 20_000_000, tol: float = 0.15) -> bool:
    return abs(count - target) <= tol * target


def count_trainable_parameters(model: nn.Module) -> int:
    """Sum of numel() over parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def validate_token_batch(idx: torch.Tensor, config: ExperimentConfig) -> None:
    if idx.ndim != 2:
        raise ModelError(f"token batch must be rank-2 (B, T), got shape {tuple(idx.shape)}")
    batch, seq = idx.shape
    if batch < 1:
        raise ModelError("batch size must be >= 1")
    if seq < 1:
        raise ModelError("sequence length must be >= 1")
    if seq > config.block_size:
        raise ModelError(
            f"sequence length {seq} exceeds block_size {config.block_size}"
        )
    if idx.dtype not in (torch.int32, torch.int64, torch.long):
        raise ModelError(f"token ids must be integer, got {idx.dtype}")
    if idx.numel() == 0:
        raise ModelError("token batch is empty")
    vmin = int(idx.min().item())
    vmax = int(idx.max().item())
    if vmin < 0 or vmax >= config.vocab_size:
        raise ModelError(
            f"token id out of range [{vmin}, {vmax}] for vocab_size={config.vocab_size}"
        )


def shift_input_target(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal LM shift: inputs are tokens[:-1], targets are tokens[1:]."""
    if tokens.ndim == 1:
        if tokens.numel() < 2:
            raise ModelError("need at least 2 tokens to form an input/target pair")
        return tokens[:-1], tokens[1:]
    if tokens.ndim != 2:
        raise ModelError(f"tokens must be rank-1 or rank-2, got {tuple(tokens.shape)}")
    if tokens.size(1) < 2:
        raise ModelError("need at least 2 tokens to form an input/target pair")
    return tokens[:, :-1], tokens[:, 1:]


def build_model(config: ExperimentConfig) -> CausalLM:
    return CausalLM(config)


def future_token_influence(
    model: CausalLM,
    idx: torch.Tensor,
    position: int,
    new_token: int,
) -> torch.Tensor:
    """Absolute change in logits at positions strictly before ``position``.

    A causal model in eval mode must produce a numerically zero delta.
    """
    model.eval()
    if not 0 <= position < idx.size(1):
        raise ModelError("position out of range")
    with torch.no_grad():
        baseline, _ = model(idx)
        mutated = idx.clone()
        mutated[:, position] = int(new_token)
        updated, _ = model(mutated)
        if position == 0:
            return torch.zeros_like(baseline[:, :1])
        return (updated[:, :position] - baseline[:, :position]).abs()
