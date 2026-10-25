"""The three proofs from the lecture / proven S13 notebook.

1. The stack inverts (round-trip of both state grids).
2. Memory-free gradients match ordinary autograd, tensor by tensor.
3. Activation memory stops growing with depth (CUDA only).
"""

from __future__ import annotations

from typing import Any

import torch

from helpers.config import ExperimentConfig, tiny_config
from helpers.model import CausalLM, build_model
from helpers.reversible import forward_step, inverse_step


def _small(mode: str, depth: int, dtype: torch.dtype = torch.float64) -> CausalLM:
    experiment = "A" if mode == "baseline" else "B"
    variant = "baseline" if mode == "normal" else mode
    cfg = tiny_config(
        experiment=experiment,
        variant=variant,
        n_layer=depth,
        n_embd=16,
        n_head=4,
        n_inner=32,
        block_size=8,
        vocab_size=32,
        bias=False,
    )
    model = build_model(cfg)
    return model.to(dtype=dtype)


def proof_invert_stack(
    modes: tuple[str, ...] = ("euler", "midpoint"),
    depths: tuple[int, ...] = (2, 4),
    atol: float = 1e-10,
) -> list[dict[str, Any]]:
    """Proof 1: forward then inverse recovers the starting pair in float64."""
    rows: list[dict[str, Any]] = []
    torch.manual_seed(0)
    for mode in modes:
        for depth in depths:
            model = _small(mode, depth, torch.float64).eval()
            ids = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
            x0 = model.embed(ids).double()
            h = model.config.euler_h if mode == "euler" else model.config.midpoint_step_size
            a, b = x0, (torch.zeros_like(x0) if mode == "euler" else x0)
            start = (a.clone(), b.clone())
            with torch.no_grad():
                for blk in model.blocks:
                    a, b = forward_step(mode, h, blk, a, b)
                for blk in reversed(list(model.blocks)):
                    a, b = inverse_step(mode, h, blk, a, b)
            rel = max(
                (a - start[0]).abs().max().item(),
                (b - start[1]).abs().max().item(),
            ) / max(start[0].abs().max().item(), 1e-12)
            rows.append(
                {
                    "mode": mode,
                    "depth": depth,
                    "relative_error": rel,
                    "ok": rel < atol,
                }
            )
    return rows


def proof_gradients_match(modes: tuple[str, ...] = ("euler", "midpoint")) -> list[dict[str, Any]]:
    """Proof 2: reversible backward vs storing-all-activations reference."""
    rows: list[dict[str, Any]] = []
    torch.manual_seed(0)
    for mode in modes:
        model = _small(mode, 2, torch.float64)
        ids = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
        targets = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
        model.zero_grad(set_to_none=True)
        _, loss_ours = model(ids, targets, plain=False)
        assert loss_ours is not None
        loss_ours.backward()
        ours = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        _, loss_ref = model(ids, targets, plain=True)
        assert loss_ref is not None
        loss_ref.backward()
        worst = 0.0
        where = ""
        for name, param in model.named_parameters():
            if param.grad is None or name not in ours:
                continue
            denom = max(param.grad.abs().max().item(), 1e-30)
            rel = (ours[name] - param.grad).abs().max().item() / denom
            if rel > worst:
                worst, where = rel, name
        rows.append(
            {
                "mode": mode,
                "loss_delta": abs(float(loss_ours.item()) - float(loss_ref.item())),
                "max_rel_grad": worst,
                "tensor": where,
                "ok": worst < 1e-6,
            }
        )
    return rows


def proof_memory_vs_depth(
    depths: tuple[int, ...] = (4, 8, 16, 32),
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Proof 3: peak extra CUDA memory vs depth. Skipped on CPU."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return {
            "skipped": True,
            "reason": "PROOF 3 needs a CUDA GPU (Colab T4/L4/A100).",
            "table": {},
        }
    table: dict[int, dict[str, float]] = {}
    for depth in depths:
        row: dict[str, float] = {}
        for mode in ("baseline", "euler", "midpoint"):
            torch.cuda.empty_cache()
            experiment = "A" if mode == "baseline" else "B"
            cfg = ExperimentConfig(
                experiment=experiment,
                variant="baseline" if mode == "baseline" else mode,
                vocab_size=512,
                n_embd=256,
                n_layer=depth,
                n_head=4,
                n_inner=1024,
                block_size=256,
                bias=False,
                dropout=0.0,
                micro_batch_size=8,
                token_budget=1024,
                dataset="synthetic",
                mode="smoke",
            )
            model = build_model(cfg).to(device)
            ids = torch.randint(0, cfg.vocab_size, (8, cfg.block_size), device=device)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            _, loss = model(ids, ids)
            assert loss is not None
            loss.backward()
            torch.cuda.synchronize()
            row[mode] = (torch.cuda.max_memory_allocated(device) - before) / 2**20
            del model, ids, loss
            torch.cuda.empty_cache()
        table[depth] = row
    growth = {
        mode: table[depths[-1]][mode] / max(table[depths[0]][mode], 1e-6)
        for mode in ("baseline", "euler", "midpoint")
    }
    return {"skipped": False, "table": table, "growth": growth, "depths": list(depths)}
