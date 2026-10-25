"""Training smoke tests for baseline and reversible models."""

from __future__ import annotations

import math

import torch

from helpers.config import tiny_config
from helpers.data import prepare_data
from helpers.train import cosine_lr, smoke_train, train


def test_smoke_baseline_and_midpoint(tmp_path):
    for variant, experiment in (("baseline", "A"), ("midpoint", "B")):
        report = smoke_train(
            variant,
            output_dir=str(tmp_path / variant),
            max_steps=2,
            token_budget=64,
            log_interval_steps=1,
        )
        assert report["status"] in {"smoke", "completed"}
        assert report["variant"] == variant
        assert report["experiment"] == experiment
        assert report["training_target_tokens"] > 0
        assert report["final_train_loss"] is not None
        assert math.isfinite(report["final_train_loss"])
        assert report["micro_batch_size"] * report["accumulation_steps"] == report["effective_batch_size"]
        if variant == "midpoint":
            assert report["reconstruction_ok"] is True
            assert report["reversible_method"] == "midpoint"


def test_optimizer_updates_parameters(tmp_path):
    config = tiny_config(
        experiment="B",
        variant="midpoint",
        output_dir=str(tmp_path),
        max_steps=1,
        token_budget=32,
    )
    data = prepare_data(config, download=False)
    from helpers.model import build_model

    model = build_model(config)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # train() builds its own model; compare loss finiteness instead via a manual step.
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = next(iter(data.train_loader))
    opt.zero_grad()
    _logits, loss = model(batch["input_ids"], batch["labels"])
    loss.backward()
    opt.step()
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert changed
    assert torch.isfinite(loss)


def test_cosine_warmup_then_decay():
    config = tiny_config(learning_rate=1e-3, warmup_tokens=10, token_budget=110, min_lr_ratio=0.1)
    assert cosine_lr(0, config) == 0.0 or cosine_lr(0, config) < config.learning_rate
    mid_warmup = cosine_lr(5, config)
    assert 0 < mid_warmup < config.learning_rate
    peakish = cosine_lr(10, config)
    late = cosine_lr(110, config)
    assert late <= peakish
    assert late <= config.learning_rate * config.min_lr_ratio + 1e-12
