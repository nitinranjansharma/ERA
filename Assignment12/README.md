# ZeRO Training Simulator (ERA V5 Session 12)

A **Google Colab–first** educational simulator for data parallelism and DeepSpeed-style ZeRO stages 1, 2, and 3. The notebook accounts for a **30-billion-parameter** training state using closed-form arithmetic, walks a **32-GPU virtual cluster** made of lightweight metadata, and trains a **tiny real PyTorch model** so gradient averaging can be checked numerically.

This is a simulator, not a production distributed-training stack. It does **not** install DeepSpeed, does **not** spawn 32 processes, does **not** download a 30B checkpoint, and does **not** allocate tensors proportional to 30 billion parameters.

Primary deliverable: [`zero_simulation.ipynb`](zero_simulation.ipynb)

Repository: [https://github.com/nitinranjansharma/ERA](https://github.com/nitinranjansharma/ERA) (`Assignment12/`)

---

## Assignment objective

Given a 30B-parameter model whose training state costs **16 bytes per parameter** and an **80 GB** GPU (about **74.5 GiB**), determine which parallelism strategies can theoretically hold that training state on a 32-GPU cluster, and at what communication cost.

The assignment baseline is:

| State | Precision | Bytes / parameter |
| --- | --- | --- |
| Weights | FP16 / BF16 | 2 |
| Gradients | FP16 / BF16 | 2 |
| FP32 master weights | FP32 | 4 |
| Two Adam moments | FP32 × 2 | 8 |
| **Total** | | **16** |

30e9 × 16 bytes = 480e9 bytes ≈ **447.0 GiB** if every GPU stores a full replica. That is far beyond 74.5 GiB, which is why optimizer-state, gradient, and parameter sharding exist.

---

## Architecture

```
ERA/Assignment12/
├── zero_simulation.ipynb   # main Colab entry point
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/zero_sim/
│   ├── config.py           # units, validation, device, seed
│   ├── memory.py           # GB/GiB and ZeRO memory formulas
│   ├── collectives.py      # all-reduce / reduce-scatter / all-gather
│   ├── simulator.py        # 32-GPU virtual cluster
│   ├── model.py            # tiny trainable network
│   ├── metrics.py          # comparison tables
│   └── plots.py            # in-notebook figures
└── tests/                  # pytest suite used by the notebook
```

The notebook is the main entry point. Helper modules live under `src/zero_sim/` so the same code runs in Colab, in `pytest`, and in a local Jupyter kernel. The bootstrap cell does not assume that the current working directory is the repository root.

| Module | Role |
| --- | --- |
| `config.py` | Assignment constants, world-size validation, GB vs GiB, runtime diagnostics |
| `memory.py` | Bytes-per-parameter formulas, per-GPU GiB, fit/headroom, optional activations |
| `collectives.py` | Small-tensor collectives plus analytical 2P / 3P volumes and times |
| `simulator.py` | Virtual ranks, nodes, shard ownership, reconstruction |
| `model.py` | Tiny MLP, data-parallel gradient check, gradient accumulation |
| `metrics.py` | Pandas tables for the 32-GPU comparison and the memory ladder |
| `plots.py` | Memory, communication, ratio, and loss figures |

---

## Memory derivations

Let \(N\) be the world size and 16 the unreplicated bytes per parameter.

- **Data parallelism:** every GPU keeps weights, gradients, master weights, and moments.
  \[
  16 \text{ bytes/parameter}
  \]
- **ZeRO-1:** shard the 12-byte optimizer state (4-byte master + 8-byte moments); keep 4-byte FP16 weights + gradients replicated.
  \[
  4 + 12/N
  \]
- **ZeRO-2:** also shard the 2-byte gradients.
  \[
  2 + 14/N
  \]
- **ZeRO-3:** shard all 16 bytes.
  \[
  16/N
  \]

Memory capacity is reported in **binary GiB** (\(2^{30}\) bytes). Communication volumes use **decimal GB** (\(10^9\) bytes). An 80 GB GPU therefore holds \(80 \times 10^9 / 2^{30} \approx 74.5\) GiB.

---

## Collective operations

The simulator implements three collectives on **small PyTorch tensors**:

- **All-reduce** — every rank ends with the same reduced tensor.
- **Reduce-scatter** — ranks reduce, then each keeps one shard.
- **All-gather** — shards are concatenated and replicated.

Ring all-reduce of a parameter-sized gradient is modeled as **2P**. ZeRO-1 still all-reduces gradients (**2P**). ZeRO-2 replaces the all-reduce with reduce-scatter of gradients plus all-gather of updated parameters (**2P**). ZeRO-3 adds a parameter all-gather for the forward and backward passes plus a gradient reduce-scatter (**3P**).

With \(P = 60\) GB:

| Path | Bandwidth | 2P time | 3P time |
| --- | --- | --- | --- |
| Intra-node | 450 GB/s | 0.27 s | 0.40 s |
| Inter-node | 50 GB/s | 2.40 s | 3.60 s |

These numbers are **simplified estimates**, not measurements of the Colab VM, NVLink, or InfiniBand. The Colab T4 run reproduced them exactly from the closed-form model.

---

## ZeRO stage comparison

| Strategy | Replicated | Sharded | Comm / step | 32-GPU fit? |
| --- | --- | --- | --- | --- |
| Data parallel | weights, grads, master, moments | — | 2P | No |
| ZeRO-1 | weights, grads | master, moments | 2P | No |
| ZeRO-2 | weights | grads, master, moments | 2P | Yes* |
| ZeRO-3 | — | weights, grads, master, moments | 3P | Yes* |

\*Theoretical training-state fit only.

Ownership is contiguous. The union of shards is the full state; shards do not overlap. On the Colab run, a 128-element toy parameter vector reconstructed exactly from 32 ZeRO-3 shards (rank 0 owned 4 values).

---

## Colab execution

The notebook is designed for a **fresh Google Colab runtime**. CPU is the guaranteed baseline. A single Colab GPU is used for diagnostics when CUDA is available. The 30B memory analysis never needs that GPU.

Do not install DeepSpeed, flash-attn, or other CUDA-only distributed packages. Do not upgrade PyTorch: Colab's preinstalled `torch` is the build you want.

### Workflow A — clone this repository (recommended)

`Assignment12` lives inside the ERA monorepo. A GitHub `/tree/...` URL is **not** a valid `git clone` target (that form failed in Colab). Sparse-checkout the subdirectory:

```bash
mkdir -p /content/zero-training-simulator
cd /content/zero-training-simulator
git init
git remote add origin https://github.com/nitinranjansharma/ERA.git
git config core.sparseCheckout true
echo "Assignment12/*" >> .git/info/sparse-checkout
git fetch --depth 1 origin main
git checkout FETCH_HEAD
```

The helper package is then at `/content/zero-training-simulator/Assignment12`. The bootstrap cell searches `cwd`, `/content/zero-training-simulator`, `/content`, and parent directories for `src/zero_sim/__init__.py`.

A full clone also works if you do not mind downloading the rest of ERA:

```bash
git clone https://github.com/nitinranjansharma/ERA.git /content/ERA
# helpers are at /content/ERA/Assignment12
```

### Workflow B — ZIP upload

1. Zip `Assignment12` so that `src/zero_sim/` and `tests/` are inside the archive.
2. In Colab: **Runtime → Run all**, or run the bootstrap cell.
3. If the helper package is not found, run the **ZIP upload** cell, choose the archive, and re-run the locator cell.

---

## Local tests

From `Assignment12/` (a local Jupyter/Python environment, not required for Colab):

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

The notebook's section 15 runs the same suite with `pytest.main(...)`.

---

## Observed Colab run

The notebook in this tree includes outputs from a **Google Colab T4** session. That run is not a substitute for re-running on your own runtime, but it is the source of the numbers below.

| Item | Observed value |
| --- | --- |
| Runtime | Google Colab, GPU accelerator, Tesla T4 |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| Matplotlib | 3.10.0 |
| CUDA | available |
| GPU memory | 15.64 GB (decimal) |
| Selected device | `cuda` |
| Project root | `/content/zero-training-simulator/Assignment12` |
| Seed | 42 |

Dependencies already present: numpy, pandas, matplotlib, pytest. PyTorch was **not** upgraded. Smoke test of the 8-GPU memory ladder passed on scalar arithmetic (no 30B tensors).

### Memory ladder (30B training state, rounded to 0.1 GiB)

| World size | Data parallel | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- | --- |
| 8 | 447.0, no | 153.7, no | 104.8, no | **55.9, yes** (+18.6) |
| 16 | 447.0, no | 132.7, no | 80.3, no | **27.9, yes** (+46.6) |
| 32 | 447.0, no | 122.2, no | **68.1, yes** (+6.4) | **14.0, yes** (+60.5) |
| 64 | 447.0, no | 117.0, no | **62.0, yes** (+12.5) | **7.0, yes** (+67.5) |

ZeRO-3 is the only strategy that fits at 8 and 16 GPUs. ZeRO-2 first fits at **32** GPUs, with only **6.4 GiB** of training-state headroom. Data parallelism never moves. ZeRO-1 is still 117.0 GiB even at 64 GPUs because the replicated FP16 weights + gradients alone are ~111.8 GiB.

### 32-GPU training-state breakdown (GiB / GPU)

| Strategy | Weights | Grads | Master | Moments | Total | Headroom | Fit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Data parallelism | 55.9 | 55.9 | 111.8 | 223.5 | 447.0 | −372.5 | No |
| ZeRO-1 | 55.9 | 55.9 | 3.5 | 7.0 | 122.2 | −47.7 | No |
| ZeRO-2 | 55.9 | 1.7 | 3.5 | 7.0 | 68.1 | +6.4 | Yes* |
| ZeRO-3 | 1.7 | 1.7 | 3.5 | 7.0 | 14.0 | +60.5 | Yes* |

\*Training-state only. Capacity = 74.5 GiB.

### Communication (analytical, \(P = 60\) GB)

| Strategy | Volume | Intra-node 450 GB/s | Inter-node 50 GB/s | Comm / compute (assumed) |
| --- | --- | --- | --- | --- |
| Data parallelism | 120 GB (2P) | 0.27 s | 2.40 s | 0.71 |
| ZeRO-1 | 120 GB (2P) | 0.27 s | 2.40 s | 0.71 |
| ZeRO-2 | 120 GB (2P) | 0.27 s | 2.40 s | 0.71 |
| ZeRO-3 | 180 GB (3P) | 0.40 s | 3.60 s | 1.07 |

The communication-to-compute column uses an assumed transformer FLOP sketch, not a T4 profiler reading. ZeRO-3 is cheaper in memory and more expensive in communication.

### Activations and overhead (why a paper fit is not enough)

At 32 GPUs, training-state-only ZeRO-2 has 6.4 GiB of slack. Adding a 5.0 GiB activation sketch plus 4.0 GiB of runtime/allocator overhead **flips ZeRO-2 to a miss** (−2.6 GiB) while ZeRO-3 still fits (+51.5 GiB). A 20 GiB activation tensor plus 4 GiB overhead leaves ZeRO-2 at −17.6 GiB and ZeRO-3 at +36.5 GiB.

### Toy-model training (CPU)

The T4 runtime selected `cuda` for diagnostics, but the tiny training demo was pinned to **CPU** so cloned worker models and data stayed on one device. Results:

| Metric | Value |
| --- | --- |
| Parameters | 1,732 |
| Initial loss | 1.3655 |
| Final loss | 1.2729 |
| Last-step grad L1 | 1.4534 |
| Equal-batch grad error | 1.49e-08 |
| Weighted unequal-batch grad error | 2.98e-08 |
| Naive mean (unequal batches) | 8.88e-02 |
| Param error after one synced step | 1.49e-08 |
| Global batch | 2 × 4 × 3 = **24** |
| Accumulation micro-batches | 12 |
| Accumulated vs global-batch grad error | 3.73e-09 |
| Reference / accumulated loss | 1.3644 / 1.3644 |

Weighted averaging matched the global-batch backward pass. A naive mean of worker gradients did **not**, once local batch sizes were (5, 7, 10, 14). This MLP is not a 30B model, and the virtual workers are not `torch.distributed` ranks.

### Automated tests on that Colab GPU runtime

Section 15 ran `pytest` against `/content/zero-training-simulator/Assignment12/tests`:

- **37 passed**
- **2 failed** (`pytest` exit code 1, notebook cell raised)

Both failures are the same CUDA device mismatch in `TinyClassifier.clone()`: the clone is constructed on CPU while features already live on `cuda:0` when `get_device()` returns CUDA.

| Test | Result |
| --- | --- |
| `test_global_batch_size_and_accumulation` | Failed on CUDA (`mat1` on `cuda:0`, weights on CPU) |
| `test_gradient_equivalence_and_parameter_updates` | Failed on CUDA (same mismatch) |
| Remaining memory, collective, shard, fit, and training tests | Passed |

On a CPU-only runtime those two tests pass. The notebook training cells avoid the failure by passing `device="cpu"` explicitly. The 30B tables, plots, and smoke test are independent of this issue.

---

## Interpretation

At **32 virtual GPUs**, only **ZeRO-2** (~68.1 GiB) and **ZeRO-3** (~14.0 GiB) put 30B training state under the **74.5 GiB** line. Data parallelism stays at 447.0 GiB because it replicates everything. ZeRO-1 still replicates FP16 weights and gradients, so 32-way optimizer sharding is not enough.

The cost of going further is communication. ZeRO-3 is the only strategy whose simplified model pays **3P** instead of **2P**. Lower memory is not a free lunch, and a theoretical training-state fit still ignores activations, CUDA context, allocator fragmentation, and framework scratch.

---

## Known limitations

- The 30B analysis is **arithmetic**, not a model load.
- Collectives on tiny tensors prove algebra, not network performance.
- Bandwidth numbers (450 GB/s intra-node, 50 GB/s inter-node) are assignment assumptions. The T4 did not measure NVLink or InfiniBand.
- Activation memory is an optional estimate with a crude formula.
- The toy MLP is a few thousand parameters. Virtual workers are Python objects, not `torch.distributed` ranks.
- `TinyClassifier.clone()` currently stays on CPU, so default CUDA test paths fail until the clone is moved onto the source module's device.
- This is **not** DeepSpeed ZeRO, **not** PyTorch FSDP2, and **not** a cluster job.

---

## Simulation versus real distributed training

| Simulator | Real ZeRO / FSDP |
| --- | --- |
| One process, CPU or one GPU | Many processes, NCCL/Gloo |
| Metadata + formulas for 30B | Real parameter shards on device memory |
| Tiny SGD demo | Mixed precision, fused Adam, prefetch |
| Instant "cluster" of 32 ranks | Actual nodes, NICs, topology |

Use the notebook to reason about **why** the stages look the way they do. Use DeepSpeed or FSDP when you need to train a real model.

---

## Submit the assignment

1. Open `Assignment12/zero_simulation.ipynb` in Google Colab.
2. Clone helpers with the sparse-checkout commands above (or upload a ZIP).
3. **Runtime → Run all**.
4. Confirm the pytest cell. On a GPU runtime, expect the two clone-device failures noted above unless that helper is patched; on CPU the suite is green.
5. In Colab: **File → Download → Download .ipynb**.
6. Submit the downloaded `.ipynb` **and** the GitHub link [https://github.com/nitinranjansharma/ERA](https://github.com/nitinranjansharma/ERA) (folder `Assignment12`).

---

## Student reflection prompts

1. Why does ZeRO-1 still miss a 74.5 GiB budget at 32 GPUs even though the optimizer is sharded?
2. If activation memory were 20 GiB per GPU, would ZeRO-2 still fit? What about ZeRO-3?
3. Why is a naive mean of worker gradients wrong when local batch sizes differ?
4. When would you accept ZeRO-3's extra all-gather instead of stopping at ZeRO-2?
5. Which parts of this notebook would break first if you tried to drop in a real 30B checkpoint?
6. How would NVLink versus Ethernet change the communication-to-compute plot without changing the memory table?

---

## License

Educational material for ERA V5 Session 12. Treat the helper package as a teaching model, not as an optimizer implementation.
