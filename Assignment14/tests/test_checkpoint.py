"""10. Checkpoint round-trip and seeded reproducibility."""

from __future__ import annotations

import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import set_seed, tiny_config
from src.data import make_synthetic_data
from src.dense_model import build_dense_model
from src.train import build_optimizer


def test_checkpoint_round_trip_reproduces_predictions(tmp_path, cpu_device):
    cfg = tiny_config(seed=5)
    set_seed(5)
    model = build_dense_model(cfg).to(cpu_device)
    opt = build_optimizer(model, cfg)
    x = torch.randn(9, cfg.input_dim)
    with torch.no_grad():
        before = model(x).clone()
    path = tmp_path / "dense.pt"
    save_checkpoint(path, model, opt, cfg, epoch=3, metrics={"val_mse": 0.1})

    fresh = build_dense_model(tiny_config(seed=99)).to(cpu_device)
    with torch.no_grad():
        scrambled = fresh(x)
    assert not torch.allclose(before, scrambled)
    load_checkpoint(path, map_location=cpu_device, model=fresh)
    with torch.no_grad():
        after = fresh(x)
    assert torch.allclose(before, after, rtol=1e-6, atol=1e-6)


def test_seeded_data_and_init_are_reproducible():
    cfg = tiny_config(seed=12)
    set_seed(12)
    a = make_synthetic_data(cfg, seed=12)
    m1 = build_dense_model(cfg)
    w1 = m1.input_proj.weight.detach().clone()

    set_seed(12)
    b = make_synthetic_data(cfg, seed=12)
    m2 = build_dense_model(cfg)
    w2 = m2.input_proj.weight.detach().clone()

    assert torch.equal(a.x_train, b.x_train)
    assert torch.equal(w1, w2)
