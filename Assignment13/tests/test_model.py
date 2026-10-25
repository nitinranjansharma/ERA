"""Parameter count, causal masking, and tensor-shape tests."""

from __future__ import annotations

import pytest
import torch

from helpers.config import (
    PARAM_COUNT_TOLERANCE,
    TARGET_PARAM_COUNT,
    assignment_config,
    expected_param_count,
    tiny_config,
)
from helpers.model import (
    ModelError,
    build_model,
    count_trainable_parameters,
    future_token_influence,
    shift_input_target,
    validate_token_batch,
)


def test_tiny_param_count_matches_formula_and_pytorch():
    config = tiny_config()
    model = build_model(config)
    live = count_trainable_parameters(model)
    formula = expected_param_count(config)
    assert live == formula == model.param_count()
    assert model.config.tie_embeddings
    # Tied head must not double-count embedding weights.
    n_named = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_named == live


def test_assignment_model_near_20m():
    config = assignment_config("A")
    model = build_model(config)
    live = count_trainable_parameters(model)
    formula = expected_param_count(config)
    assert live == formula
    assert abs(live - TARGET_PARAM_COUNT) <= PARAM_COUNT_TOLERANCE * TARGET_PARAM_COUNT
    report = model.param_report()
    assert report["within_tolerance"]
    assert report["tie_embeddings"] is True
    # Proven S13 sizing: 12 x 384 x ff=1024 x vocab=8192 x ctx=512, no Linear bias.
    assert 18_000_000 <= live <= 22_000_000


def test_forward_shapes_and_shift():
    config = tiny_config()
    model = build_model(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size + 1))
    idx, targets = shift_input_target(tokens)
    logits, loss = model(idx, targets)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    with pytest.raises(ModelError):
        validate_token_batch(torch.tensor([[config.vocab_size]]), config)
    with pytest.raises(ModelError):
        model(torch.randint(0, config.vocab_size, (2, config.block_size + 1)))


def test_future_tokens_do_not_change_earlier_logits():
    config = tiny_config()
    model = build_model(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, config.block_size))
    position = config.block_size - 1
    new_token = int((idx[0, position] + 1) % config.vocab_size)
    delta = future_token_influence(model, idx, position, new_token)
    assert torch.max(delta) == 0
