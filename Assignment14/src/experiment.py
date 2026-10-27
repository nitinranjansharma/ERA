"""End-to-end dense pretraining, upcycling, and MoE fine-tuning."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from src.checkpoint import write_csv, write_json
from src.config import (
    LOSS_REDUCTION_ABS_TOL,
    ExperimentConfig,
    get_runtime_info,
    output_paths,
    resolve_device,
    set_seed,
)
from src.data import make_synthetic_data
from src.dense_model import build_dense_model, count_parameters
from src.evaluate import collect_routing_metrics, compare_outputs, split_eval_bundle
from src.metrics import jsonable
from src.moe_model import build_moe_model
from src.plotting import (
    plot_balancing_comparison,
    plot_dataset,
    plot_dense_vs_moe_losses,
    plot_expert_utilization,
    plot_loss_curves,
    plot_three_run_story,
)
from src.session_figures import fetch_session_figures
from src.train import train_model
from src.upcycling import clone_model, conversion_invariance_report, upcycle_dense_to_moe


def loss_reduced(start: float, end: float, abs_tol: float = LOSS_REDUCTION_ABS_TOL) -> bool:
    """Primary acceptance check: end task loss is meaningfully below start."""
    return end < start - abs_tol


def run_full_experiment(
    config: ExperimentConfig | None = None,
    output_dir: str | Path | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Run the assignment pipeline and write artifacts.

    Order:
        1. Generate and split data.
        2. Train the dense baseline.
        3. Save the best dense checkpoint.
        4. Convert the trained dense model into an MoE.
        5. Evaluate the MoE before fine-tuning.
        6. Continue training the converted MoE.
        7. Evaluate validation and test metrics.
        8. Compare learning curves and routing statistics.
    """
    cfg = config or ExperimentConfig()
    if output_dir is not None:
        cfg = cfg.replaced(output_dir=str(output_dir))
    paths = output_paths(cfg.output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(paths["root"] / ".mplconfig"))
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    started = time.perf_counter()
    runtime = get_runtime_info(device)
    if log:
        print(f"device={device} seed={cfg.seed} architecture={cfg.architecture}")
        print(f"runtime={runtime}")

    try:
        session_images = fetch_session_figures("images")
        if log:
            print("session figures:", session_images)
    except OSError as exc:
        session_images = {"error": str(exc)}
        if log:
            print("session figure download skipped:", exc)

    data = make_synthetic_data(cfg, seed=cfg.seed)
    plot_dataset(
        data.x_train,
        data.y_train,
        paths["figures"] / "dataset.png",
        title=f"Synthetic train set (n={cfg.n_train}, seed={cfg.seed})",
    )

    dense = build_dense_model(cfg).to(device)
    dense_before = split_eval_bundle(
        dense, data.train_loader, data.val_loader, data.test_loader, device
    )
    dense_ckpt = paths["checkpoints"] / "dense_best.pt"
    if log:
        print(
            f"dense params={count_parameters(dense)} "
            f"init train MSE={dense_before['train']['train_mse']:.6f}"
        )

    dense_result = train_model(
        dense,
        data.train_loader,
        data.val_loader,
        cfg,
        device,
        epochs=cfg.dense_epochs,
        learning_rate=cfg.learning_rate,
        checkpoint_path=dense_ckpt,
        log_fn=(lambda row: print(
            f"dense epoch {row['epoch']:03d}  "
            f"train={row['train_mse']:.6f} val={row['val_mse']:.6f}"
        ))
        if log
        else None,
    )
    dense_after = split_eval_bundle(
        dense, data.train_loader, data.val_loader, data.test_loader, device
    )
    plot_loss_curves(
        dense_result["history"],
        paths["figures"] / "dense_loss.png",
        title="Dense baseline training",
    )

    moe, conversion = upcycle_dense_to_moe(dense, cfg, seed=cfg.seed)
    moe = moe.to(device)
    discrepancy_before = compare_outputs(dense, moe, data.x_val, device)
    invariance = conversion_invariance_report(dense, moe, data.x_val)
    moe_before = split_eval_bundle(
        moe, data.train_loader, data.val_loader, data.test_loader, device
    )
    routing_before = collect_routing_metrics(
        moe, data.val_loader, device, cfg.num_experts, cfg.top_k
    )
    if log:
        print(
            f"upcycling strategy={conversion.strategy} "
            f"dense unchanged={conversion.dense_unchanged} "
            f"exact copies={conversion.all_copied_exactly}"
        )
        print(
            f"pre-finetune |y_moe - y_dense| mean={discrepancy_before['mean_abs_diff']:.3e} "
            f"max={discrepancy_before['max_abs_diff']:.3e}"
        )
        print(
            "conversion invariance preserved="
            f"{invariance['preserved']} router_irrelevant={invariance['router_irrelevant']} "
            f"scrambled max abs={invariance['scrambled_max_abs_diff']:.3e}"
        )
        print(
            f"MoE train MSE before FT={moe_before['train']['train_mse']:.6f}"
        )

    moe_ckpt = paths["checkpoints"] / "moe_best.pt"
    moe_result = train_model(
        moe,
        data.train_loader,
        data.val_loader,
        cfg,
        device,
        epochs=cfg.moe_epochs,
        learning_rate=cfg.moe_learning_rate,
        checkpoint_path=moe_ckpt,
        log_fn=(lambda row: print(
            f"moe   epoch {row['epoch']:03d}  "
            f"train={row['train_mse']:.6f} val={row['val_mse']:.6f}"
        ))
        if log
        else None,
    )
    moe_after = split_eval_bundle(
        moe, data.train_loader, data.val_loader, data.test_loader, device
    )
    discrepancy_after = compare_outputs(dense, moe, data.x_val, device)
    routing_after = collect_routing_metrics(
        moe, data.val_loader, device, cfg.num_experts, cfg.top_k
    )
    plot_loss_curves(
        moe_result["history"],
        paths["figures"] / "moe_loss.png",
        title="MoE fine-tuning (from upcycled dense weights)",
    )
    plot_dense_vs_moe_losses(
        dense_result["history"],
        moe_result["history"],
        paths["figures"] / "dense_vs_moe_loss.png",
        moe_epoch_offset=cfg.dense_epochs,
    )
    plot_expert_utilization(
        routing_after["selected_load_fractions"],
        routing_after["ideal_selected_load"],
        paths["figures"] / "expert_utilization.png",
        title="MoE expert utilization after fine-tuning (validation)",
        dead_experts=routing_after["dead_experts"],
        max_vio=routing_after["max_vio"],
    )

    scratch = None
    if cfg.run_scratch_control:
        if log:
            print(f"run C: MoE from scratch for {cfg.moe_epochs} epochs")
        scratch_model = build_moe_model(cfg).to(device)
        scratch_ckpt = paths["checkpoints"] / "scratch_best.pt"
        scratch_result = train_model(
            scratch_model,
            data.train_loader,
            data.val_loader,
            cfg,
            device,
            epochs=cfg.moe_epochs,
            learning_rate=cfg.moe_learning_rate,
            checkpoint_path=scratch_ckpt,
            log_fn=(lambda row: print(
                f"scratch epoch {row['epoch']:03d}  "
                f"train={row['train_mse']:.6f} val={row['val_mse']:.6f}"
            ))
            if log
            else None,
        )
        scratch_eval = split_eval_bundle(
            scratch_model, data.train_loader, data.val_loader, data.test_loader, device
        )
        scratch = {
            "trained": scratch_eval,
            "best_epoch": scratch_result["best_epoch"],
            "best_val_mse": scratch_result["best_val_mse"],
            "history": scratch_result["history"],
            "checkpoint": str(scratch_ckpt),
            "parameters": count_parameters(scratch_model),
        }
        write_csv(paths["metrics"] / "scratch_history.csv", scratch_result["history"])

    plot_three_run_story(
        dense_result["history"],
        moe_result["history"],
        None if scratch is None else scratch["history"],
        routing_after["selected_load_fractions"],
        routing_after["ideal_selected_load"],
        paths["figures"] / "three_run_story.png",
        dense_epochs=cfg.dense_epochs,
    )

    ablation = None
    if cfg.run_ablation:
        ablation = _run_ablation(cfg, dense, data, device, paths, log=log)

    moe_train_start = moe_before["train"]["train_mse"]
    moe_train_end_last_epoch = moe_result["history"][-1]["train_mse"]
    moe_train_end = moe_after["train"]["train_mse"]
    moe_val_start = moe_before["val"]["val_mse"]
    moe_val_end = moe_after["val"]["val_mse"]
    # Last-epoch train MSE is the continued-training signal; restored-best
    # train MSE can be slightly higher if the best checkpoint is chosen by val.
    success = loss_reduced(moe_train_start, moe_train_end_last_epoch)
    elapsed = time.perf_counter() - started

    results: dict[str, Any] = {
        "config": cfg.to_dict(),
        "runtime": runtime,
        "elapsed_seconds": elapsed,
        "data_shapes": data.shapes(),
        "dense": {
            "initial": dense_before,
            "trained": dense_after,
            "best_epoch": dense_result["best_epoch"],
            "best_val_mse": dense_result["best_val_mse"],
            "history": dense_result["history"],
            "checkpoint": str(dense_ckpt),
            "parameters": count_parameters(dense),
        },
        "conversion": conversion.to_dict(),
        "conversion_invariance": invariance,
        "discrepancy_before_finetune": discrepancy_before,
        "discrepancy_after_finetune": discrepancy_after,
        "moe": {
            "before_finetune": moe_before,
            "trained": moe_after,
            "best_epoch": moe_result["best_epoch"],
            "best_val_mse": moe_result["best_val_mse"],
            "history": moe_result["history"],
            "checkpoint": str(moe_ckpt),
            "parameters": count_parameters(moe),
            "parameter_breakdown": moe.parameter_breakdown(),
        },
        "routing_before_finetune": routing_before,
        "routing_after_finetune": routing_after,
        "scratch": scratch,
        "session_figures": session_images,
        "ablation": ablation,
        "acceptance": {
            "moe_train_mse_before": moe_train_start,
            "moe_train_mse_after_last_epoch": moe_train_end_last_epoch,
            "moe_train_mse_after_best_ckpt": moe_train_end,
            "moe_val_mse_before": moe_val_start,
            "moe_val_mse_after": moe_val_end,
            "abs_tol": LOSS_REDUCTION_ABS_TOL,
            "train_loss_reduced": success,
            "val_loss_reduced": loss_reduced(moe_val_start, moe_val_end),
            "criterion": (
                "MoE full-dataset train MSE after fine-tuning is at least "
                f"{LOSS_REDUCTION_ABS_TOL} below its own pre-fine-tune train MSE"
            ),
        },
        "figures": {
            "dataset": str(paths["figures"] / "dataset.png"),
            "dense_loss": str(paths["figures"] / "dense_loss.png"),
            "moe_loss": str(paths["figures"] / "moe_loss.png"),
            "dense_vs_moe": str(paths["figures"] / "dense_vs_moe_loss.png"),
            "expert_utilization": str(paths["figures"] / "expert_utilization.png"),
            "three_run_story": str(paths["figures"] / "three_run_story.png"),
        },
    }
    if ablation is not None:
        results["figures"]["balancing"] = str(paths["figures"] / "balancing_ablation.png")

    write_json(paths["metrics"] / "experiment_results.json", jsonable(results))
    write_csv(paths["metrics"] / "dense_history.csv", dense_result["history"])
    write_csv(paths["metrics"] / "moe_history.csv", moe_result["history"])
    write_json(paths["metrics"] / "config.json", cfg.to_dict())

    if log:
        print(
            f"ACCEPTANCE train_loss_reduced={success} "
            f"{moe_train_start:.6f} -> {moe_train_end_last_epoch:.6f}"
        )
        print(f"elapsed_seconds={elapsed:.2f}")
        print(_five_questions(dense_after, moe_before, moe_after, routing_after, scratch, success))
        print(f"wrote {paths['metrics'] / 'experiment_results.json'}")
    return results


def _five_questions(
    dense_after: dict[str, Any],
    moe_before: dict[str, Any],
    moe_after: dict[str, Any],
    routing_after: dict[str, Any],
    scratch: dict[str, Any] | None,
    train_loss_reduced: bool,
) -> str:
    dense_val = dense_after["val"]["val_mse"]
    moe_val = moe_after["val"]["val_mse"]
    moe_train_before = moe_before["train"]["train_mse"]
    moe_train_after = moe_after["train"]["train_mse"]
    fracs = routing_after["selected_load_fractions"]
    lines = [
        "=" * 72,
        "1. DID IT CONTINUE?     upcycled train MSE "
        f"{moe_train_before:.6f} -> {moe_train_after:.6f}  reduced={train_loss_reduced}",
        "2. DID THE LOSS DROP?   dense val "
        f"{dense_val:.6f} -> MoE val {moe_val:.6f}  ({moe_val - dense_val:+.6f})",
    ]
    if scratch is not None:
        scratch_val = scratch["trained"]["val"]["val_mse"]
        lines.append(
            "3. WAS UPCYCLING WORTH IT?  upcycled val "
            f"{moe_val:.6f} vs from-scratch val {scratch_val:.6f}  "
            f"(gap {scratch_val - moe_val:+.6f})"
        )
    else:
        lines.append("3. WAS UPCYCLING WORTH IT?  scratch control not run")
    lines.append(
        "4. DID ANY EXPERT DIE?  dead="
        f"{routing_after['dead_experts']}  MaxVio={routing_after['max_vio']:.4f}  "
        f"load={['%.3f' % f for f in fracs]}"
    )
    lines.append(
        "5. THE MoE BARGAIN      extra total parameters vs extra active compute "
        "are reported in parameter_breakdown; this is a single-device toy, not EP speedup."
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def _run_ablation(
    cfg: ExperimentConfig,
    trained_dense: torch.nn.Module,
    data: Any,
    device: torch.device,
    paths: dict[str, Path],
    log: bool,
) -> dict[str, Any]:
    """Short comparison of baseline routing vs Switch auxiliary loss."""
    from src.upcycling import upcycle_dense_to_moe as convert

    dense_cpu = clone_model(trained_dense).cpu()
    baseline_cfg = cfg.replaced(
        aux_loss_coef=0.0,
        z_loss_coef=0.0,
        use_bias_balancing=False,
        run_ablation=False,
    )
    aux_cfg = cfg.replaced(
        aux_loss_coef=0.02,
        z_loss_coef=0.0,
        use_bias_balancing=False,
        run_ablation=False,
    )
    moe_base, _ = convert(dense_cpu, baseline_cfg, seed=cfg.seed)
    moe_aux, _ = convert(clone_model(dense_cpu), aux_cfg, seed=cfg.seed)
    moe_base = moe_base.to(device)
    moe_aux = moe_aux.to(device)
    if log:
        print(f"ablation: {cfg.ablation_epochs} epochs, aux=0 vs aux={aux_cfg.aux_loss_coef}")
    train_model(
        moe_base,
        data.train_loader,
        data.val_loader,
        baseline_cfg,
        device,
        epochs=cfg.ablation_epochs,
        learning_rate=cfg.moe_learning_rate,
        checkpoint_path=None,
    )
    train_model(
        moe_aux,
        data.train_loader,
        data.val_loader,
        aux_cfg,
        device,
        epochs=cfg.ablation_epochs,
        learning_rate=cfg.moe_learning_rate,
        checkpoint_path=None,
    )
    route_base = collect_routing_metrics(
        moe_base, data.val_loader, device, cfg.num_experts, cfg.top_k
    )
    route_aux = collect_routing_metrics(
        moe_aux, data.val_loader, device, cfg.num_experts, cfg.top_k
    )
    plot_balancing_comparison(
        route_base,
        route_aux,
        paths["figures"] / "balancing_ablation.png",
        before_label="no aux loss",
        after_label=f"aux={aux_cfg.aux_loss_coef}",
    )
    return {
        "epochs": cfg.ablation_epochs,
        "aux_loss_coef": aux_cfg.aux_loss_coef,
        "baseline_routing": route_base,
        "aux_routing": route_aux,
        "note": (
            "Balancing scope is the current training batch on one device "
            "(the whole batch; there are no micro-batches or expert-parallel ranks)."
        ),
    }


if __name__ == "__main__":
    run_full_experiment()
