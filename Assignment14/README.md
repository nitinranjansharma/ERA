# Dense-to-MoE Sparse Upcycling (ERA V5 Session 14)

Train a small dense regressor on a synthetic nonlinear task, **copy its trained feed-forward weights into a top-2-of-4 Mixture-of-Experts**, then keep training. The assignment is a pass if the converted MoE’s train MSE falls from its own pre-fine-tune baseline.

Primary deliverable: [`moe_upcycling_assignment.ipynb`](moe_upcycling_assignment.ipynb)

This is a **pedagogical single-device** experiment. It is not distributed expert-parallel training and does not reproduce Qwen3-30B-A3B (or any production MoE) in scale or speed.

---

## Assignment objective

> Train a Linear model and convert that into an MoE. Your call on model size and data trained on, but must show they continue to train and reduce loss.

Two readings of “Linear model” are supported:

| `architecture` | Meaning |
| --- | --- |
| `ffn` (default) | Compact MLP. The inner two-layer FFN is copied into every expert. This matches Session 14 sparse upcycling. |
| `strict_linear` | A single `nn.Linear`. That one layer is copied into every expert. Literal wording; weaker analogue of transformer FFN upcycling. |

Default run: CPU-friendly synthetic regression, 4 experts, top-2, AdamW, MSE.

---

## What the notebook does

1. Train a small dense MLP on synthetic regression.
2. Copy the trained FFN into a top-2-of-4 MoE.
3. Keep training the MoE and show train MSE falling from the conversion baseline.

`src/` holds the helpers and extra knobs (aux loss, shared expert, from-scratch control). The notebook keeps the default path only.

Copied experts start identical, so they preserve the dense FFN for any normalized top-k mix until fine-tuning actually changes them. That is checked numerically before MoE training starts.

---

## Project structure

```
Assignment14/
├── moe_upcycling_assignment.ipynb
├── README.md
├── requirements.txt
├── pytest.ini
├── src/
│   ├── config.py          # validated ExperimentConfig
│   ├── data.py            # synthetic regression + loaders
│   ├── dense_model.py     # dense MLP / strict Linear
│   ├── experts.py         # ExpertFFN / ExpertLinear
│   ├── router.py          # top-k router, capacity drop
│   ├── moe_model.py       # sparse combine + MoE regressors
│   ├── upcycling.py       # dense → MoE conversion
│   ├── losses.py          # MSE, aux, z-loss, bias update
│   ├── metrics.py         # load, MaxVio, dead experts
│   ├── train.py           # loops, AdamW, best checkpoint
│   ├── evaluate.py        # test metrics, dense vs MoE
│   ├── plotting.py        # figures
│   ├── checkpoint.py      # save / load + RNG
│   ├── session_figures.py # download Session 14 illustration images
│   └── experiment.py      # full pipeline (runs A, B, C)
├── tests/                 # 10+ required categories
├── images/                # session illustrations (auto-fetched if missing)
└── outputs/
    ├── figures/
    ├── metrics/
    └── checkpoints/
```

---

## Python and dependencies

- Python 3.10+ (measured run: 3.11.8)
- `torch`, `numpy`, `matplotlib`, `pytest`

`requirements.txt` does **not** pin `torch`, so Colab can keep its CUDA build.

```bash
python -m pip install -r requirements.txt
```

CUDA is used automatically when `torch.cuda.is_available()` is true. CPU is the guaranteed path.

---

## Google Colab setup

1. Upload this folder, or clone the ERA repo and open `Assignment14/moe_upcycling_assignment.ipynb`.
2. Runtime → CPU is enough. A GPU is optional.
3. Run all cells top to bottom. The bootstrap cell locates `src/` after a checkout or upload.
4. Optional clone (if `src/` is not next to the notebook):

```bash
!git clone --depth 1 https://github.com/nitinranjansharma/ERA.git /content/ERA
%cd /content/ERA/Assignment14
```

---

## How to run the notebook

Open `moe_upcycling_assignment.ipynb` and run every cell in order. The notebook imports `src/` rather than inlining the model. It trains the dense model, converts it to an MoE, fine-tunes, and prints whether train MSE dropped. Figures go under `outputs/figures/`.

Session 14’s three-panel language-model illustration is stored at `images/results.png`. A fresh Colab runtime can pull it with:

```python
from src.session_figures import fetch_session_figures
fetch_session_figures("images")  # no-ops if the file is already present
```

That picture is a **session illustration**, not this synthetic run. This experiment’s own three-panel figure is `outputs/figures/three_run_story.png`.

A one-shot CLI equivalent (same seed, same pipeline, plus the aux-loss ablation):

```bash
python -m src
```

---

## How to run helper scripts locally

```bash
python -m src                    # full experiment
python -c "from src.experiment import run_full_experiment; run_full_experiment()"
```

Change settings through `ExperimentConfig` (no hidden globals):

```python
from src.config import ExperimentConfig
from src.experiment import run_full_experiment

run_full_experiment(ExperimentConfig(num_experts=8, top_k=2, moe_epochs=20))
```

---

## How to execute tests

From `Assignment14/`:

```bash
python -m pytest tests -q
```

Measured result on this machine: **38 passed** in about 36 seconds (includes the deterministic training-convergence test).

---

## How to reproduce with fixed seeds

`ExperimentConfig.seed` defaults to **42**. That seed is used for data generation, dense init, training RNG (`set_seed`), and router init at conversion. DataLoader shuffle uses a `torch.Generator` seeded from the same value. `num_workers=0` to keep the run deterministic on CPU.

```python
from src.config import set_seed, ExperimentConfig
set_seed(42)
```

---

## Model architecture and hyperparameters

Default `ExperimentConfig`:

| Field | Default |
| --- | --- |
| input / hidden / FFN / output | 8 / 32 / 32 / 1 |
| activation | GELU |
| experts / top-k | 4 / 2 |
| routed scaling | 1.0 |
| n_train / n_val / n_test | 1024 / 256 / 256 |
| noise std | 0.05 |
| batch size | 64 |
| dense / MoE epochs | 40 / 40 |
| optimizer | AdamW, lr `1e-3`, weight decay 0 |
| aux / z-loss / bias balancing | off (primary run) |
| capacity factor | `None` (dropless) |
| upcycling | `copy` |

Generating function (known, fixed):

```text
y = sin(x0) + 0.5*x1^2 + 0.3*x0*x2 + 0.2*x3 - 0.4*tanh(x4) + 0.15*x5*x6 + 0.1*x7 + noise
```

---

## How dense-to-MoE upcycling works

1. Train `DenseRegressor` and keep the lowest-validation-MSE checkpoint.
2. Build `MoERegressor` with the same shared projections and \(N\) independent `ExpertFFN`s.
3. Copy `input_proj` and `output_proj` from dense.
4. Copy `dense.ffn` into **each** expert (`copy`), or copy then add \(N(0,\\sigma)\) (`perturb`).
5. Initialize the router from \(N(0, 0.02)\) with the conversion seed.
6. Refuse to skip mismatched parameters; abort if the dense module would be mutated or if expert tensors alias dense storage.

After `copy`, any normalized top-k mix of identical experts equals the original FFN. The router does not change the function until experts or the router actually train.

---

## How to interpret training and validation losses

- **Train MSE** is the assignment’s primary “continued training” signal. Compare MoE train MSE **after** fine-tuning to MoE train MSE **before** fine-tuning (the cloned dense function), not only to the dense model.
- **Val MSE** chooses the checkpoint and guards overfitting. The MoE is not required to beat dense on val/test.
- **Test MSE / MAE** are held-out and are not used for checkpointing.
- Success check: `moe_train_mse_last_epoch < moe_train_mse_before - 1e-4`.

---

## Expected artifacts

| Path | Contents |
| --- | --- |
| `outputs/figures/dataset.png` | train-set visualization |
| `outputs/figures/dense_loss.png` | dense train/val curves |
| `outputs/figures/moe_loss.png` | MoE fine-tune curves |
| `outputs/figures/dense_vs_moe_loss.png` | overlay (MoE epochs offset) |
| `outputs/figures/expert_utilization.png` | selected-load bars |
| `outputs/figures/three_run_story.png` | A continues, B vs C, expert load |
| `images/results.png` | Session 14 language-model illustration (downloaded) |
| `outputs/figures/balancing_ablation.png` | aux vs no-aux utilization |
| `outputs/metrics/experiment_results.json` | machine-readable run |
| `outputs/metrics/dense_history.csv` | per-epoch dense losses |
| `outputs/metrics/moe_history.csv` | per-epoch MoE losses |
| `outputs/checkpoints/dense_best.pt` | trained dense + RNG |
| `outputs/checkpoints/moe_best.pt` | fine-tuned MoE + RNG |

---

## Actual results (executed locally)

These numbers are from `python -m src` on this machine, **not** invented. If you re-run on another device or PyTorch build, expect small floating-point differences; the qualitative story should hold.

| Item | Value |
| --- | --- |
| Runtime | Python 3.11.8, PyTorch 2.7.0, CPU, macOS arm64 |
| Wall time | 8.28 s (including 20+20 epoch ablation) |
| Dense parameters | 2433 |
| MoE total / active per sample | 8901 / 4677 (ratio ≈ 1.90) |
| Dense train MSE init → best | 1.532225 → 0.016999 |
| Dense val MSE init → best | 1.814579 → 0.019994 (best epoch 40) |
| Dense test MSE / MAE | 0.041580 / 0.116240 |
| Copy validation | exact copies, independent storage, dense unchanged |
| Conversion invariance | preserved; scrambled-router max abs 4.77×10⁻⁷ |
| Pre-FT \|y_MoE − y_dense\| | mean 3.22×10⁻⁸, max 4.77×10⁻⁷ |
| MoE train MSE pre-FT → last epoch | **0.016999 → 0.006713** |
| MoE val MSE pre-FT → best ckpt | 0.019994 → 0.011628 (best epoch 39) |
| MoE test MSE / MAE | 0.021066 / 0.094685 |
| From-scratch MoE (run C) val MSE | 0.027057 (worse than upcycled 0.011628) |
| **Acceptance `train_loss_reduced`** | **True** (tol 1e-4) |
| Val loss also reduced | True |
| Dead experts after FT (val) | 0 |
| Selected-load fractions after FT | [0.605, 0.543, 0.684, 0.168] (ideal 0.50, sum 2.0) |
| MaxVio after FT | 0.367 |
| Ablation MaxVio no-aux vs aux=0.02 | 0.250 → 0.055 (20 epochs, from the same dense checkpoint) |
| `pytest` | 38 passed |

The MoE **did** continue training and reduced both train and validation MSE. It also reached a **lower test MSE** than the dense checkpoint on this seed (0.021 vs 0.042). Run C (same MoE, random init, same 40 epochs) finished at val MSE **0.027**, so upcycling beat starting over on this seed. That is a measured outcome of this synthetic fit, not a general claim that MoE must win.

Expert 3 was under-used after unconstrained fine-tuning (selected-load 0.17 vs ideal 0.50) but was not dead. The short aux-loss ablation moved utilization closer to uniform. The primary assignment run keeps aux loss **off** so the dense-to-MoE story stays easy to read.

---

## Common failures and troubleshooting

| Symptom | What to check |
| --- | --- |
| `src/` not found in Colab | `cd` into `Assignment14` so `src/__init__.py` is visible; run the bootstrap cell. |
| `top_k` errors | Must satisfy `1 <= top_k <= num_experts`. |
| MoE train loss does not fall | Confirm you started from the **trained** dense checkpoint (`copy`), not a random MoE. Try more `moe_epochs` or `moe_learning_rate=1e-3`. Report a miss honestly before changing the recipe. |
| Pre-FT dense vs MoE outputs differ | `copy` + `routed_scaling_factor=1.0` should match within ~1e-6. `perturb` is *supposed* to differ. |
| Matplotlib cache warnings | The experiment sets `MPLCONFIGDIR` under `outputs/.mplconfig`. |
| `pytest` training test is slow | It runs 25+25 tiny epochs on CPU; tens of seconds is expected. |
| CUDA vs CPU mismatch | Checkpoints load with `map_location`; default device is `auto`. |

---

## Limitations and extensions

- Single device, sample-level routing, synthetic 8-d regression.
- Default experts start identical; specialization is learned only during fine-tuning.
- No expert-parallel communication, so no EP scaling claims.
- Capacity dropping and z-loss are implemented but not used in the primary run.
- Partition / drop-upcycling are documented, not implemented.
- Do not treat this as evidence about LLM MoE quality.

Possible extensions: enable bias balancing on the main run, a token-level sequence toy, perturb-vs-copy fine-tune comparison, or a larger FFN width.

---

## Reproduce this README’s numbers

```bash
cd Assignment14
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m src
```

Then inspect `outputs/metrics/experiment_results.json`.
