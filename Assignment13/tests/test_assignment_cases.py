"""The ten required acceptance scenarios for Session 13."""

from __future__ import annotations

import math

import pytest
import torch

from helpers.benchmark import measure_training_step, search_max_micro_batch
from helpers.config import (
    PARAM_COUNT_TOLERANCE,
    TARGET_PARAM_COUNT,
    assignment_config,
    expected_param_count,
    tiny_config,
)
from helpers.data import prepare_data, tokenize_text
from helpers.metrics import validate_report
from helpers.model import build_model, count_trainable_parameters, future_token_influence, shift_input_target
from helpers.reporting import export_run_artifacts
from helpers.reversible import midpoint_forward_reference, midpoint_stack, verify_midpoint_reconstruction
from helpers.train import smoke_train


def test_case_01_configuration_validation():
    tiny_config()
    assignment_config("A")
    assignment_config("B")
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(n_embd=15, n_head=4)
    with pytest.raises(ValueError, match="token_budget"):
        tiny_config(token_budget=0)
    with pytest.raises(ValueError, match="precision"):
        tiny_config(precision="fp8")
    with pytest.raises(ValueError, match="micro_batch_size"):
        tiny_config(micro_batch_size=0)
    with pytest.raises(ValueError, match="block_size"):
        tiny_config(block_size=1)
    with pytest.raises(ValueError, match="dropout"):
        tiny_config(experiment="B", variant="midpoint", dropout=0.1)
    with pytest.raises(ValueError, match="Experiment A"):
        tiny_config(experiment="A", variant="midpoint")
    with pytest.raises(ValueError, match="euler_step_size"):
        tiny_config(experiment="B", variant="euler", euler_step_size=0.0)
    assignment_config("B", variant="euler", dataset="sangraha", precision="auto")
    with pytest.raises(ValueError, match="dataset"):
        tiny_config(dataset="not-a-dataset")


def test_case_02_parameter_count_verification():
    model = build_model(assignment_config("A"))
    live = count_trainable_parameters(model)
    assert live == expected_param_count(assignment_config("A"))
    assert abs(live - TARGET_PARAM_COUNT) / TARGET_PARAM_COUNT <= PARAM_COUNT_TOLERANCE
    tiny = build_model(tiny_config())
    assert count_trainable_parameters(tiny) == expected_param_count(tiny_config())


def test_case_03_causal_language_model_correctness():
    config = tiny_config()
    model = build_model(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size + 1))
    idx, targets = shift_input_target(tokens)
    logits, loss = model(idx, targets)
    assert idx.shape == targets.shape == (2, config.block_size)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None
    delta = future_token_influence(
        model,
        idx[:1],
        position=config.block_size - 1,
        new_token=int((idx[0, -1] + 3) % config.vocab_size),
    )
    assert float(delta.max()) == 0.0


def test_case_04_dataset_and_token_accounting(tmp_path):
    assert tokenize_text("") == []
    config = tiny_config(output_dir=str(tmp_path))
    data = prepare_data(config, download=False)
    batch = next(iter(data.train_loader))
    assert int(batch["n_padding_tokens"].item()) == 0
    assert int(batch["n_target_tokens"].item()) == config.micro_batch_size * config.sequence_length
    assert data.discarded_train_tokens >= 0
    assert data.val_target_tokens >= 0
    # Validation tokens are stored separately from the training counter.
    assert data.accounting.validation_tokens == 0


def test_case_05_reversible_forward_inverse_reconstruction():
    config = tiny_config(experiment="B", variant="midpoint")
    model = build_model(config).eval()
    x = torch.randn(2, config.block_size, config.n_embd)
    report = verify_midpoint_reconstruction(x, model.blocks, two_h=config.two_h, atol=5e-4)
    assert report.ok
    assert report.max_abs_error <= 5e-4


def test_case_06_gradient_correctness():
    config = tiny_config(experiment="B", variant="midpoint")
    model = build_model(config)
    x = torch.randn(2, 4, config.n_embd, requires_grad=True)
    midpoint_stack(x, model.blocks, two_h=1.0).sum().backward()
    rev_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    assert rev_grads and all(torch.isfinite(g).all() for g in rev_grads)
    model.zero_grad(set_to_none=True)
    x_ref = x.detach().clone().requires_grad_(True)
    midpoint_forward_reference(x_ref, model.blocks, two_h=1.0).sum().backward()
    ref_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    for g_rev, g_ref in zip(rev_grads, ref_grads):
        assert torch.allclose(g_rev, g_ref, atol=1e-4, rtol=1e-4)


def test_case_07_numerical_stability():
    from helpers.reversible import ReversibleError, _require_finite, _check_dropout_modules
    import torch.nn as nn

    with pytest.raises(ReversibleError):
        _require_finite(torch.tensor([1.0, float("nan")]), "probe")
    drop = nn.Dropout(0.3)
    with pytest.raises(ReversibleError, match="dropout"):
        _check_dropout_modules([drop])


def test_case_08_training_smoke_test(tmp_path):
    base = smoke_train("baseline", output_dir=str(tmp_path / "a"), max_steps=2, token_budget=48)
    rev = smoke_train("midpoint", output_dir=str(tmp_path / "b"), max_steps=2, token_budget=48)
    for report in (base, rev):
        assert report["training_target_tokens"] > 0
        assert report["final_train_loss"] is not None
        assert math.isfinite(report["final_train_loss"])
        assert report["status"] in {"smoke", "completed"}


def test_case_09_benchmark_and_memory_reporting():
    n = {"i": 0}

    def step():
        n["i"] += 1
        return 8

    result = measure_training_step(step, warmup_steps=1, measured_steps=2, device="cpu")
    assert n["i"] == 3
    assert result.warmup_steps == 1
    assert result.peak_allocated_bytes is None  # CPU degrades gracefully


def test_case_10_max_batch_search_and_end_to_end_report(tmp_path):
    def probe(batch: int) -> None:
        if batch >= 16:
            raise torch.cuda.OutOfMemoryError("simulated")

    search = search_max_micro_batch(probe, start=1, max_batch=32, confirm_steps=2, headroom=True)
    assert search.best_micro_batch == 4
    report = {
        "experiment": "C",
        "variant": "midpoint",
        "status": "smoke",
        "trainable_parameters": 2048,
        "micro_batch_size": int(search.best_micro_batch),
        "accumulation_steps": 1,
        "effective_batch_size": int(search.best_micro_batch) * 1,
        "sequence_length": 8,
        "training_target_tokens": 64,
        "token_budget": 64,
        "final_train_loss": 1.2,
        "final_val_loss": 1.3,
        "tokens_per_second": 32.0,
        "wall_clock_seconds": 2.0,
        "time_per_optimizer_step": 0.5,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "device": "cpu",
        "precision": "fp32",
        "reversible_method": "midpoint",
        "extra": {"batch_search": search.to_dict()},
    }
    validate_report(report)
    with pytest.raises(Exception):
        validate_report({**report, "effective_batch_size": 99})
    artifacts = export_run_artifacts(tmp_path, [report])
    assert artifacts["results_json"].endswith("results.json")
