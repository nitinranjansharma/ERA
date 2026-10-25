# ERA V5 Session 11 — Optimizers and Learning-Rate Schedules

CPU-first, original implementation of the five required experiments: manual Adam, bias-correction analysis, update-to-weight ratios under warmup, cosine vs WSD, and a standard-parameterization learning-rate sweep across width.

Numbers in the report come from `artifacts/results.json`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m pytest -q
python -m run_experiments --overwrite
```

Without the editable install, the same CLI is `PYTHONPATH=src python -m run_experiments --overwrite` or `python src/run_experiments.py --overwrite`.

`--overwrite` refreshes `artifacts/{results.json,plots,logs}` and `report/final_report.md`. Every invocation also writes a timestamped copy under `artifacts/runs/<UTC stamp>/`.

## Commands

| Action | Command |
|---|---|
| Unit + acceptance tests | `python -m pytest -q` |
| All five experiments | `python -m run_experiments --overwrite` |
| Subset | `python -m run_experiments --tasks A,B` |
| Custom config | `python -m run_experiments --config configs/fast.yaml --output-dir /tmp/s11` |

Default hyperparameters live in `configs/default.yaml`.

## Layout

```
configs/          default.yaml, fast.yaml, traceability.yaml
src/              adam, schedules, models, metrics, experiments, CLI
tests/            pytest, including the ten required cases
artifacts/        results.json, logs/, plots/
report/           final_report.md (generated from JSON)
notebooks/        walkthrough
```

## What is measured

1. **Task A.** Scalar Adam in float64 vs `torch.optim.Adam` on `w0=1.0` and gradients `[0.50, 0.40, 0.60, 0.45, 0.55]`. A separate class implements the no-bias-correction update and does not fill hat-moments.
2. **Task B.** Corrected vs uncorrected updates for a constant gradient and a sign-changing sequence. The 20-step plot is separate from the persistent-tolerance search.
3. **Task C.** Instrumented MLP. `update_to_weight_ratio` uses `||θ_after − θ_before||`, not the gradient norm. Warmup and no-warmup share initialization and data order.
4. **Task D.** Tiny causal Transformer, 300 optimizer steps, checkpoint at 200, plus a 200-step-designed pair so the ranking is not tied to one horizon.
5. **Task E.** Standard-parameterization MLP at widths 256, 512, 1024. The width-4096 learning rate is an extrapolation, not a μP transfer result.

## Tests

`tests/test_required_cases.py` is the ten-case acceptance table:

1. Manual Adam vs `torch.optim.Adam` (intermediates + `atol=1e-10`, `rtol=1e-8`)
2. First-step bias correction (`|δ_on|≈lr`, uncorrected ≈ 3.16×)
3. Zero / near-zero gradients stay finite
4. Warmup: step 0, midpoint, boundary, plateau
5. Update-to-weight ratio vs a hand-computed delta, including zero-init bias
6. Seeded rerun reproducibility
7. Cosine vs WSD: 300-step runs, valid step-200 checkpoints
8. Width sweep 256/512/1024 on a small grid; reported min is the true min; boundaries flagged
9. Non-finite loss/grad is raised, never marked successful
10. Lightweight pipeline writes the required JSON, CSVs, and four plots

