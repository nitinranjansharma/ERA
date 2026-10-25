"""Minimal deterministic models used by the training-loop experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from instrumentation import ShapeTracer


class TinyMLP(nn.Module):
    """Regression MLP: inputs (B, F) → logits (B, out)."""

    def __init__(self, in_features: int = 8, hidden: int = 16, out_features: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor, tracer: Optional[ShapeTracer] = None) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        logits = self.fc2(h)
        if tracer is not None:
            tracer.add("inputs", x, ["batch", "feature"], "B examples, each with F input features.")
            tracer.add("hidden", h, ["batch", "hidden"], "B examples after the first affine+tanh layer.")
            tracer.add(
                "logits",
                logits,
                ["batch", "out"],
                "B predicted scalars (named logits to match the assignment training step).",
            )
        return logits


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, tracer: Optional[ShapeTracer] = None, prefix: str = "attn") -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        out = self.c_proj(y)
        if tracer is not None:
            tracer.add(f"{prefix}.x", x, ["batch", "sequence", "feature"], "Residual stream entering attention.")
            tracer.add(f"{prefix}.qkv", qkv, ["batch", "sequence", "3*feature"], "Packed Q,K,V projections.")
            tracer.add(
                f"{prefix}.q",
                q,
                ["batch", "head", "sequence", "head_dim"],
                "Queries split across H heads; head_dim = n_embd / n_head.",
            )
            tracer.add(
                f"{prefix}.k",
                k,
                ["batch", "head", "sequence", "head_dim"],
                "Keys with the same layout as queries.",
            )
            tracer.add(
                f"{prefix}.v",
                v,
                ["batch", "head", "sequence", "head_dim"],
                "Values with the same layout as queries.",
            )
            tracer.add(
                f"{prefix}.scores",
                att,
                ["batch", "head", "sequence", "sequence"],
                "Causal attention weights over key positions for each query position.",
            )
            tracer.add(
                f"{prefix}.out",
                out,
                ["batch", "sequence", "feature"],
                "Attention output projected back to d_model.",
            )
        return out


class MLPBlock(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tracer: Optional[ShapeTracer] = None, prefix: str = "mlp") -> torch.Tensor:
        h = F.gelu(self.fc(x))
        out = self.dropout(self.proj(h))
        if tracer is not None:
            tracer.add(
                f"{prefix}.hidden",
                h,
                ["batch", "sequence", "4*feature"],
                "Position-wise FFN hidden layer, typically 4× d_model.",
            )
            tracer.add(f"{prefix}.out", out, ["batch", "sequence", "feature"], "FFN output in residual-stream width.")
        return out


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLPBlock(n_embd, dropout)

    def forward(self, x: torch.Tensor, tracer: Optional[ShapeTracer] = None, prefix: str = "block") -> torch.Tensor:
        x = x + self.attn(self.ln1(x), tracer=tracer, prefix=f"{prefix}.attn")
        x = x + self.mlp(self.ln2(x), tracer=tracer, prefix=f"{prefix}.mlp")
        return x


class TinyGPT(nn.Module):
    """A tiny causal LM used for shape tracing and MFU token accounting.

    input_ids: (batch, sequence) integer token ids
    logits:    (batch, sequence, vocabulary)
    """

    def __init__(
        self,
        vocab_size: int = 64,
        n_layer: int = 2,
        n_head: int = 4,
        n_embd: int = 32,
        block_size: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        tracer: Optional[ShapeTracer] = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be (batch, sequence), got {tuple(input_ids.shape)}")
        b, t = input_ids.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.block_size}")
        pos = torch.arange(0, t, device=input_ids.device)
        tok = self.tok_emb(input_ids)
        pos_e = self.pos_emb(pos)
        x = self.drop(tok + pos_e)
        if tracer is not None:
            tracer.add(
                "input_ids",
                input_ids,
                ["batch", "sequence"],
                "B independent sequences, each of T token ids in [0, vocab).",
            )
            tracer.add(
                "tok_emb",
                tok,
                ["batch", "sequence", "feature"],
                "Each token id is replaced by a d_model-dimensional embedding vector.",
            )
            tracer.add(
                "pos_emb",
                pos_e,
                ["sequence", "feature"],
                "Learned absolute position vectors broadcast over the batch.",
            )
            tracer.add(
                "residual_stream",
                x,
                ["batch", "sequence", "feature"],
                "Token + position embeddings entering the transformer stack.",
            )
        for i, block in enumerate(self.blocks):
            x = block(x, tracer=tracer, prefix=f"layer{i}")
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if tracer is not None:
            tracer.add(
                "final_hidden",
                x,
                ["batch", "sequence", "feature"],
                "LayerNorm'd residual stream before the language-model head.",
            )
            tracer.add(
                "logits",
                logits,
                ["batch", "sequence", "vocabulary"],
                "Unnormalised scores over the vocabulary at every token position.",
            )
        return logits


class ScalarTokenModel(nn.Module):
    """Toy model for accumulation: per-token loss contribution is theta * coefficient.

    At theta = 1 the per-token loss equals the coefficient, so constructed
    micro-batch average losses (2, 2, 5) are realised exactly.
    """

    def __init__(self, init: float = 1.0, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(init, dtype=dtype))

    def per_token_loss(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.theta * coefficients


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def parameter_shape_records(module: nn.Module, tracer: ShapeTracer) -> None:
    for name, p in module.named_parameters():
        dims = _param_dims(name, p)
        tracer.add(
            f"param.{name}",
            p,
            dims,
            f"Trainable parameter {name}: {p.numel()} scalars; gradient will match this shape.",
        )


def _param_dims(name: str, p: torch.Tensor) -> list[str]:
    if p.ndim == 0:
        return []
    if p.ndim == 1:
        if "bias" in name or "ln" in name:
            return ["feature"]
        return ["parameter"]
    if "tok_emb" in name or "lm_head" in name:
        return ["vocabulary", "feature"]
    if "pos_emb" in name:
        return ["sequence", "feature"]
    if "c_attn" in name and p.ndim == 2:
        return ["3*feature", "feature"]
    if p.ndim == 2:
        return ["out_feature", "in_feature"]
    return [f"dim{i}" for i in range(p.ndim)]
