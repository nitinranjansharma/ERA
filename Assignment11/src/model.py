"""Small original models used by the Session 11 experiments.

Standard parameterization (SP) for the width sweep
--------------------------------------------------
Hidden ``nn.Linear`` layers use Kaiming/He normal initialization
``N(0, 2 / fan_in)`` with ReLU, matching the usual PyTorch default for ReLU
MLPs. Biases are zeros. The last layer uses a smaller ``N(0, 1 / fan_in)``
draw so the untrained readout does not explode with width. This is *not* μP:
learning rates are not width-multiplied per tensor, and residual branches are
not 1/width-scaled. Optimal LRs are therefore expected to move with width.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def zero_biases(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear) and child.bias is not None:
            nn.init.zeros_(child.bias)
        if isinstance(child, nn.LayerNorm) and child.bias is not None:
            nn.init.zeros_(child.bias)


class InstrumentedMLP(nn.Module):
    """Three-layer classifier with LayerNorm, used for update/weight logging."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in (self.stem, self.hidden):
            nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.head.weight)
        zero_biases(self)
        # Keep LayerNorm scale at 1, bias at 0 (already default / zeroed).
        nn.init.ones_(self.norm1.weight)
        nn.init.ones_(self.norm2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.stem(x)))
        h = F.relu(self.norm2(self.hidden(h)))
        return self.head(h)

    def parameter_groups(self) -> dict[str, list[str]]:
        return {
            "stem": ["stem.weight", "stem.bias"],
            "hidden": ["hidden.weight", "hidden.bias"],
            "norm": ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"],
            "head": ["head.weight", "head.bias"],
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.proj = nn.Linear(n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = self.dropout(torch.softmax(att, dim=-1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinySequenceLM(nn.Module):
    """Compact causal Transformer for the cosine vs WSD comparison."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _b, t = idx.shape
        if t > self.block_size:
            raise ValueError("sequence longer than block_size")
        pos = torch.arange(0, t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class StandardWidthMLP(nn.Module):
    """Width-parameterized MLP under standard (not μP) parameterization."""

    def __init__(self, input_dim: int, width: int, n_classes: int, hidden_layers: int = 2) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)
        self.width = int(width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=1.0 / math.sqrt(self.head.in_features))
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def named_trainable(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param
