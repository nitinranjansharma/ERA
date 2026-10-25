"""Every tensor in a training step has a shape and a dimension explanation."""

from __future__ import annotations

import torch

from training_loop import traced_language_model_step, training_step
from models import TinyMLP


def test_complete_step_records_core_tensors():
    result = traced_language_model_step(batch_size=2, seq_len=8)
    names = {r["name"] for r in result.tracer.as_dicts()}
    for required in (
        "input_ids",
        "tok_emb",
        "logits",
        "targets",
        "loss",
        "grad_norm",
    ):
        assert required in names, f"missing {required}"
    records = result.tracer.as_dicts()
    loss_rec = next(r for r in records if r["name"] == "loss")
    assert loss_rec["shape"] == []
    logits_rec = next(r for r in records if r["name"] == "logits")
    assert logits_rec["shape"] == [2, 8, 64]
    assert logits_rec["dims"] == ["batch", "sequence", "vocabulary"]


def test_every_record_explains_its_axes():
    result = traced_language_model_step(batch_size=2, seq_len=4)
    for rec in result.tracer.as_dicts():
        assert rec["explanation"], rec["name"]
        if rec["shape"] == []:
            assert rec["dims"] == []
        else:
            assert len(rec["dims"]) == len(rec["shape"]), rec


def test_attention_scores_are_causal_square():
    result = traced_language_model_step(batch_size=2, seq_len=8)
    scores = next(r for r in result.tracer.as_dicts() if r["name"].endswith(".attn.scores"))
    assert scores["shape"][-2:] == [8, 8]
    assert scores["dims"] == ["batch", "head", "sequence", "sequence"]


def test_parameter_and_grad_shapes_match():
    result = traced_language_model_step(batch_size=2, seq_len=4)
    by_name = {r["name"]: r for r in result.tracer.as_dicts()}
    grad_names = [n for n in by_name if n.startswith("grad.")]
    assert grad_names, "no gradients recorded"
    for gname in grad_names:
        pname = "param." + gname[len("grad.") :]
        if pname in by_name:
            assert by_name[gname]["shape"] == by_name[pname]["shape"]


def test_optimizer_state_shapes_recorded_for_adam():
    result = traced_language_model_step(batch_size=2, seq_len=4)
    state = [r for r in result.tracer.as_dicts() if r["name"].startswith("adam.state")]
    assert state, "Adam state tensors should be recorded after the first step"
    assert any("exp_avg" in r["name"] for r in state)


def test_regression_step_shapes():
    model = TinyMLP(8, 16, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)
    y = torch.randn(4, 1)
    result = training_step(model, x, y, opt, task="regression")
    logits = next(r for r in result.tracer.as_dicts() if r["name"] == "logits")
    assert logits["shape"] == [4, 1]
    assert logits["dims"] == ["batch", "out"]
