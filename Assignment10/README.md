# ERA V5 Session 10 — The Training Loop

A CPU-first, evidence-driven implementation of the six mandatory experiments: tensor-shape tracing, independent gradient verification, the gradient-accumulation averaging bug, gradient-norm monitoring, MFU, and the FP32 / BF16 / FP8 E4M3 bit-level encoding of decimal 0.1.

Do not treat a falling loss as proof that the loop is correct. The tests check scalar finite loss, parameter updates, overfitting of a tiny set, forgotten `zero_grad`, padded-token exclusion, unit mistakes in MFU, and independent floating-point encodings.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q
python -m experiments.run_all --overwrite
```

`--overwrite` replaces the stable copies under `outputs/{figures,logs,metrics}/`. Every run is also kept in a timestamped directory `outputs/runs/<UTC stamp>/`, so older results are never deleted.

Exit codes are non-zero if a mandatory experiment or test fails.

## What the suite does

| Experiment | Command path | Primary artifacts |
|---|---|---|
| Tensor shapes for one full step | `experiments/run_all.py::run_shape_experiment` | `outputs/logs/training_step_shapes.txt` |
| Analytical vs autograd vs finite difference | `src/gradient_verification.py` | `outputs/figures/finite_difference_error.png` |
| Incorrect vs correct accumulation | `src/accumulation_experiment.py` | `outputs/figures/accumulation_loss.png` |
| Grad-norm spike vs loss | `src/gradient_monitor.py` | `outputs/figures/loss_and_grad_norm.png` |
| MFU (6ND formula) | `src/mfu.py` | `outputs/metrics/mfu.json` |
| 0.1 in FP32 / BF16 / OCP E4M3FN | `src/precision.py` | `outputs/logs/precision_0p1.txt` |

The written report is `report/final_report.md`.

## Layout

```
configs/          default.yaml, traceability matrix
src/              implementation
tests/            pytest, CPU-only by default
experiments/      run_all.py
outputs/          figures, logs, metrics, timestamped runs
report/           final_report.md
```

The ten test cases live in `tests/test_required_cases.py` (`test_01` … `test_10`). GPU tests are marked `@pytest.mark.gpu` and are skipped when CUDA/MPS is absent.

## Hardware note

Developed and measured on an Apple M2 (4P+4E, 8 GB), PyTorch CPU, seed 42. CUDA is unavailable.
