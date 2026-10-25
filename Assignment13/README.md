# Reversible Language-Model Training (ERA V5 Session 13)

Train a ~20 million-parameter decoder-only Transformer on a 50 million **target-token** budget, then compare:

1. **Experiment A** — ordinary residual (non-reversible) training at a fixed micro-batch.
2. **Experiment B** — the same model width/depth with a **midpoint / leapfrog** reversible stack at that same micro-batch.
3. **Experiment C** — the reversible stack again at the largest micro-batch that repeatedly survives on the selected hardware.

This is a working experimental comparison, not a claim that reversible training must win.

Primary deliverable: [`reversible_llm_training.ipynb`](reversible_llm_training.ipynb)

Repository: [https://github.com/nitinranjansharma/ERA](https://github.com/nitinranjansharma/ERA) (`Assignment13/`)

---

## Research questions

1. Does algebraic reconstruction of hidden states reduce **peak activation memory** enough to matter at ~20M parameters?
2. What **compute overhead** (tokens/second) does recomputing \(f_{\theta_l}\) on the backward pass introduce at a matched micro-batch?
3. If a larger micro-batch becomes possible, how much of any loss or throughput change is **batch size**, not reversibility?

---

## Standard vs reversible training

A Pre-LN transformer block is the first-order residual

\[
x_{l+1} = x_l + F_l(x_l)
\]

Autograd stores the activations of \(F_l\) (or **checkpoints** them and reruns the original forward). Checkpointing is not a reversible network.

### Midpoint / leapfrog (primary, Experiments B and C)

\[
p[l+1] = p[l-1] + 2 h\, f_{\theta_l}(p[l])
\]

Exact inverse with the same parameters and a deterministic \(f_{\theta_l}\):

\[
p[l-1] = p[l+1] - 2 h\, f_{\theta_l}(p[l])
\]

Initialization (lecture notebook): \(p[0] = p[1] = \mathrm{Embed}(\text{tokens})\), so the first update is \(x + 2h F(x)\). Default \(h = 0.5\) so \(2h = 1\).

Only the last two states are stored. Backward reconstructs \(p[l-1]\) with the inverse, recomputes \(f_{\theta_l}(p[l])\), and backpropagates through that recomputation. Dropout is rejected on this path.

### Euler position + velocity (same width as the baseline)

The map \(x \leftarrow x + h F(x)\) is not invertible by itself. Carry a second full-width grid \(v\) (starts at 0):

```text
FORWARD                         BACKWARD
v_new = v_old + h * F(x_old)    x_old = x_new - h * v_new
x_new = x_old + h * v_new       v_old = v_new - h * F(x_old)
```

Default \(h = 1/\sqrt{L}\) (\(\approx 0.289\) at 12 layers). Parameter count matches the residual baseline. This is the lecture Euler, **not** RevNet channel-split coupling.

Hidden states stay **fp32** for the subtraction; \(F\) may run under autocast. On a T4 do **not** use `torch.cuda.is_bf16_supported()` — it is True while bf16 matmuls are emulated. Use compute capability \(\ge 8.0\) for bf16, else fp16.

No `memcnn` / `revtorch` dependency. `torch.utils.checkpoint` is not reported as reversibility.

The proven S13 write-up trains **four** runs at 50M tokens: normal, Euler, and midpoint at the same batch, then the winner at a larger batch. This repo keeps the assignment's A/B/C names and adds Euler as a first-class variant (`--variant euler`) plus the three proofs.

---

## Project structure

```
Assignment13/
├── reversible_llm_training.ipynb
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── helpers/
│   ├── __init__.py
│   ├── config.py          # validated configs, seeds, device
│   ├── data.py            # tokenizer, WikiText-2 / synthetic / Sangraha, packing
│   ├── sangraha.py        # optional Indic BPE path (proven S13)
│   ├── model.py           # causal Transformer, param count
│   ├── reversible.py      # midpoint leapfrog + Euler velocity
│   ├── proofs.py          # invert, gradient-match, memory-vs-depth
│   ├── train.py           # loop, CLI: python -m helpers.train
│   ├── benchmark.py       # timing, memory, max-batch search CLI
│   ├── metrics.py         # token accounting, report validation
│   └── reporting.py       # CSV/JSON/plots
├── tests/                 # ten acceptance cases + focused unit tests
└── outputs/               # run artifacts (gitignored except .gitkeep)
```

---

## Architecture (~20M parameters)

| Setting | Value |
| --- | --- |
| Type | Decoder-only causal Transformer (GPT-2 Pre-LN) |
| `vocab_size` | 8192 |
| `n_embd` | 384 |
| `n_layer` | 12 |
| `n_head` | 6 |
| `n_inner` | 1024 |
| `block_size` / sequence length | 512 |
| Bias on linears | no (LayerNorm still affine) |
| Embeddings | learned token + learned position |
| LM head | **tied** to token embeddings |
| Dropout | 0.0 (required for reversible reconstruction; matched on baseline) |
| Weight decay | 0.0 (matched on baseline so the comparison is fair) |
| Trainable parameters | **19,876,608** (closed-form = live PyTorch) |
| Distance to 20M | −0.6% |

Attention is explicit matmul + causal mask (no Flash/SDPA) so reversible recomputation is deterministic.

Optimizer: AdamW, \(\beta=(0.9, 0.95)\), weight decay 0.0, grad clip 1.0, cosine decay after ~5M-token warmup (about 300 steps at batch 32). Weights stay **fp32**; `F` runs under AMP keyed by compute capability (bf16 on Ampere+, fp16 on T4). Seed: **1337**. Fixed comparison batch is 32; run 4 uses 128 with 64/32 fallback on OOM.

---

## Dataset and token budget

- **Full runs (proven S13 path):** AI4Bharat Sangraha Indic mixture + 8192-piece byte-level BPE (`--dataset sangraha`). Needs optional `datasets` and `tokenizers`. Token ids are uint16 `.bin` files loaded with `np.fromfile` (RAM, not memmap).
- **Full runs (no HuggingFace):** WikiText-2 word-level files (`--dataset wikitext2`), cached under `outputs/data/`. Official train/valid/test files are separate.
- **Tests and CPU smoke:** deterministic synthetic corpus. Tests never download.
- Word tokenizer: regex word/punctuation trained on the **train** split only. BPE tokenizer: HuggingFace `tokenizers` JSON saved next to each run.
- Packing: non-overlapping blocks of `T+1` ids → input `[:T]`, target `[1:]`. Remainder tokens are discarded, not trained. Packed data has **zero padding**.
- **50M budget** = 50,000,000 non-padding **training target** positions, including repeats if the corpus is shorter than 50M. Validation tokens are a separate counter.

---

## Environment

```bash
python -m pip install -r requirements.txt
```

`torch` is unpinned so Colab keeps its CUDA build. Local development used Python 3.11.8 and PyTorch 2.7.0 **CPU** (no NVIDIA GPU on the authoring machine).

| Item | Authoring machine | Intended full-run hardware |
| --- | --- | --- |
| Device | CPU | Colab T4 / L4 / A100 |
| CUDA | unavailable | required for 50M-token A/B/C |
| Peak GPU memory | `None` (graceful) | `max_memory_allocated` / `reserved` |

---

## Google Colab

1. Runtime → Change runtime type → GPU (T4 is enough for the 20M model).
2. Clone or sparse-checkout this folder:

```bash
!git clone --depth 1 https://github.com/nitinranjansharma/ERA.git /content/ERA
%cd /content/ERA/Assignment13
```

3. Open `reversible_llm_training.ipynb`.
4. The bootstrap cell locates `helpers/` from `cwd`, `/content`, or a glob.
5. Leave `RUN_MODE = "smoke"` until the GPU check prints a device name.
6. Set `RUN_MODE = "full"` and `RUN_EXPERIMENT_A/B/C = True` only when you intend to spend the 50M-token budget.
7. Outputs land in `outputs/` (not only notebook RAM). Checkpoints are resumable with `--resume`.

Do not use **Runtime → Run all** expecting 50M-token results on a CPU runtime. Full mode raises if CUDA is missing.

---

## Local commands

All commands are from `Assignment13/`.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m helpers.train --help
python -m helpers.benchmark --proofs
python -m helpers.benchmark --help
```

### Smoke tests (tiny model, synthetic data, CPU OK)

```bash
python -m helpers.train --mode smoke --experiment A
python -m helpers.train --mode smoke --experiment B
python -m helpers.train --mode smoke --experiment B --variant euler
python -m helpers.train --mode smoke --experiment C
python -m helpers.benchmark --pilot-batch --search-max-batch --mode smoke --variant midpoint
```

### Full experiments (50M target tokens, CUDA required)

```bash
python -m helpers.train --mode full --experiment A --download-data --dataset sangraha
python -m helpers.train --mode full --experiment B --download-data --dataset sangraha --variant euler
python -m helpers.train --mode full --experiment B --download-data --dataset sangraha --variant midpoint
python -m helpers.train --mode full --experiment C --download-data --dataset sangraha --variant euler --micro-batch-size 128
# WikiText-2 fallback if HuggingFace datasets cannot be streamed:
python -m helpers.train --mode full --experiment A --download-data --dataset wikitext2
# if that OOMs, retry 64 then 32; scale lr by sqrt(new_batch/32) as in the lecture notebook
```

`--allow-cpu-full` exists only as an explicit override; a 50M-token CPU run is not practical.

Resume:

```bash
python -m helpers.train --mode full --experiment A --resume outputs/exp_A_baseline_full/checkpoint.pt
```

---

## Acceptance tests

```bash
python -m pytest -q
python -m pytest -q tests/test_assignment_cases.py
```

The ten required scenarios (all implemented, all passing on CPU):

| # | Scenario | Module |
| --- | --- | --- |
| 1 | Configuration validation | `test_case_01_configuration_validation` |
| 2 | Parameter-count verification (~20M ± 15%) | `test_case_02_parameter_count_verification` |
| 3 | Causal LM shapes and no future-token leakage | `test_case_03_causal_language_model_correctness` |
| 4 | Dataset / packing / token accounting | `test_case_04_dataset_and_token_accounting` |
| 5 | Midpoint forward/inverse reconstruction | `test_case_05_reversible_forward_inverse_reconstruction` |
| 6 | Reversible gradients vs stored-activation reference | `test_case_06_gradient_correctness` |
| 7 | Non-finite values and dropout rejected | `test_case_07_numerical_stability` |
| 8 | Baseline and reversible training smoke | `test_case_08_training_smoke_test` |
| 9 | Warm-up vs measured timing; CPU memory degrades | `test_case_09_benchmark_and_memory_reporting` |
| 10 | Simulated OOM batch search + machine-readable report | `test_case_10_max_batch_search_and_end_to_end_report` |

**Authoring-machine outcome:** pytest passing on CPU. CUDA peak-memory population, Proof 3, and 50M-token A/B/C are hardware-dependent and were **not executed** here. Do not copy another notebook's T4 numbers into this table.

---

## How to read the metrics

| Metric | Meaning |
| --- | --- |
| `trainable_parameters` | `sum(p.numel() for p in model.parameters() if p.requires_grad)`; tied head counted once |
| `micro_batch_size` | Sequences per forward on this device |
| `accumulation_steps` | Micro-batches per optimizer step |
| `effective_batch_size` | `micro_batch_size * accumulation_steps` (sequences, not tokens) |
| `training_target_tokens` | Non-padding target positions in the training loop only |
| `token_budget` | Requested training target tokens (50,000,000 for full runs) |
| `final_train_loss` | Mean token-weighted CE at the last training step |
| `final_val_loss` | Token-weighted CE on the validation split (separate counter) |
| `tokens_per_second` | Measured training target tokens / CUDA-synchronized train time (warm-up, download, eval excluded) |
| `wall_clock_seconds` | End-to-end train call including final eval |
| `peak_allocated_bytes` / `peak_reserved_bytes` | `torch.cuda.max_memory_*` after a reset; `None` on CPU |
| `reconstruction_error` | Max abs error of midpoint inverse vs embeddings (Experiment B/C) |

Throughput is **not** `training_target_tokens / wall_clock_seconds` when eval is included in the wall clock. Reports store `extra.measured_training_seconds` for the throughput denominator.

---

## Results

Full 50M-token Experiments A/B/C have **not** been executed on the authoring CPU. The table below is a **pending template**, not fabricated measurements.

| experiment | parameters | variant | seq | micro-batch | effective batch | train tokens | train loss | val loss | tok/s | wall (s) | peak alloc | peak reserved | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 19876608 | none (baseline) | 512 | 32 | 32 | 50000000 |  |  |  |  |  |  | pending |
| B euler | 19876608 | euler \(h=1/\sqrt{12}\) | 512 | 32 | 32 | 50000000 |  |  |  |  |  |  | pending |
| B midpoint | 19876608 | midpoint \(h=0.5\) | 512 | 32 | 32 | 50000000 |  |  |  |  |  |  | pending |
| C | 19876608 | winner, max batch | 512 | 128→64→32 |  | 50000000 |  |  |  |  |  |  | pending |

### CPU smoke (tiny model, not the 50M assignment runs)

Executed on the authoring machine (PyTorch 2.7.0, CPU, no GPU). Model is the unit-test Transformer (5,120 parameters, `T=8`), 32 target tokens, 2 optimizer steps. These rows are **not** Experiments A/B/C.

| experiment | parameters | variant | seq | micro-batch | effective batch | train tokens | train loss | val loss | tok/s | wall (s) | peak alloc | peak reserved | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5120 | none | 8 | 2 | 2 | 32 | 3.412 | 3.357 | 5767 | 0.006 | n/a (cpu) | n/a (cpu) | smoke |
| B | 5120 | midpoint | 8 | 2 | 2 | 32 | 3.422 | 3.368 | 3928 | 0.008 | n/a (cpu) | n/a (cpu) | smoke |

Midpoint reconstruction error on the first batch: \(7.5\times 10^{-9}\). Throughput dropped vs baseline at the same micro-batch, which is the expected extra \(F_l\) recomputation. Peak GPU memory is unavailable on CPU.

After a Colab GPU run, replace the pending 50M table from `outputs/results.csv` / `outputs/results.json`. Plots are written to `outputs/plots/`:

- `loss_vs_tokens.png`
- `tokens_per_second.png`
- `peak_memory.png`

---

## Memory-versus-compute trade-off (what to look for)

At matched micro-batch (A vs B), reversibility should **lower activation memory** and **lower tokens/second** because each \(F_l\) is computed twice. Experiment C asks whether that memory headroom converts into a larger micro-batch. A larger batch can improve utilization and change optimization noise; a better final loss in C is not by itself evidence that midpoint is a better function class.

On a T4-class GPU a 20M model may already be optimizer-state bound, so reversibility might not increase the max batch much. Report that honestly if it happens.

---

## Deviations from the assignment

- Implemented reversible blocks in PyTorch instead of a missing `memcnn`/`revtorch` install.
- Both lecture integrators are first-class: Euler (velocity, \(h=1/\sqrt{L}\)) and midpoint (\(h=0.5\)). A/B/C still name the assignment's three slots; B should be run twice (both variants) as in the proven notebook.
- Default CLI dataset is WikiText-2 so a full run does not require HuggingFace. The notebook's full mode uses Sangraha + byte-level BPE like the proven write-up (`DATASET = "sangraha"`).
- Baseline dropout is 0.0 to match reversible determinism.
- Full 50M-token numbers are pending GPU execution.

---

## Reproducibility

- Config JSON is saved per run (`outputs/exp_<letter>_<variant>_<mode>/config.json`).
- Tokenizer JSON is saved beside it.
- `--seed` sets Python, NumPy, and PyTorch (including CUDA if present).
- `cudnn.deterministic = True` for reconstruction-sensitive runs.
- Stale artifacts: each run uses `exp_{experiment}_{variant}_{mode}/`. Pending templates are labeled `status=pending` and must not be mixed with `completed` rows.

Known gaps: FlashAttention is intentionally unused; mixed-precision reversible reconstruction is allowed by config (`fp16`/`bf16`) but not the default because inverse drift grows. MPS is resolved to a device string if present but full benchmarks still prefer CUDA.
