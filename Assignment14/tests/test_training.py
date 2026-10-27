"""9. Dense pretraining and MoE fine-tuning reduce task loss on a toy problem."""

from __future__ import annotations

from src.config import LOSS_REDUCTION_ABS_TOL, tiny_config
from src.data import make_synthetic_data
from src.dense_model import build_dense_model
from src.experiment import loss_reduced
from src.train import evaluate_loss, train_model
from src.upcycling import upcycle_dense_to_moe


def test_dense_then_moe_reduce_task_loss(cpu_device):
    cfg = tiny_config(
        n_train=128,
        n_val=32,
        n_test=32,
        batch_size=32,
        hidden_dim=16,
        ffn_hidden_dim=16,
        dense_epochs=25,
        moe_epochs=25,
        learning_rate=1e-2,
        moe_learning_rate=1e-2,
        seed=0,
        run_ablation=False,
    )
    data = make_synthetic_data(cfg, seed=cfg.seed)
    dense = build_dense_model(cfg).to(cpu_device)
    dense_start = evaluate_loss(dense, data.train_loader, cpu_device, split="train")["train_mse"]
    dense_result = train_model(
        dense,
        data.train_loader,
        data.val_loader,
        cfg,
        cpu_device,
        epochs=cfg.dense_epochs,
        learning_rate=cfg.learning_rate,
        checkpoint_path=None,
    )
    dense_end = dense_result["history"][-1]["train_mse"]
    assert loss_reduced(dense_start, dense_end, abs_tol=1e-3), (
        f"dense train MSE did not drop enough: {dense_start:.6f} -> {dense_end:.6f}"
    )

    moe, report = upcycle_dense_to_moe(dense, cfg, initialization_strategy="copy", seed=cfg.seed)
    assert report.dense_unchanged
    moe = moe.to(cpu_device)
    moe_start = evaluate_loss(moe, data.train_loader, cpu_device, split="train")["train_mse"]
    moe_result = train_model(
        moe,
        data.train_loader,
        data.val_loader,
        cfg,
        cpu_device,
        epochs=cfg.moe_epochs,
        learning_rate=cfg.moe_learning_rate,
        checkpoint_path=None,
    )
    moe_end = moe_result["history"][-1]["train_mse"]
    assert moe_end < moe_start - LOSS_REDUCTION_ABS_TOL, (
        f"MoE train MSE did not drop from its own baseline: {moe_start:.6f} -> {moe_end:.6f}"
    )
