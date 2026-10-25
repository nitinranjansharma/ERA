# ZeRO Training Simulator (ERA V5 Session 12)

A **Google Colab–first** educational simulator for data parallelism and DeepSpeed-style ZeRO stages 1, 2, and 3. The notebook accounts for a **30-billion-parameter** training state using closed-form arithmetic, walks a **32-GPU virtual cluster** made of lightweight metadata, and trains a **tiny real PyTorch model** so gradient averaging can be checked numerically.

This is a simulator, not a production distributed-training stack. It does **not** install DeepSpeed, does **not** spawn 32 processes, does **not** download a 30B checkpoint, and does **not** allocate tensors proportional to 30 billion parameters.

Primary deliverable: [`zero_simulation.ipynb`](zero_simulation.ipynb)

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
zero-training-simulator/
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

The notebook is the main entry point. Helper modules live under `src/zero_sim/` so the same code runs in Colab, in `pytest`, and in a local Jupyter kernel. Nothing in the notebook depends on files that exist only on one developer's machine.

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

Representative training-state results for 30B parameters (rounded to 0.1 GiB):

| Strategy | 8 GPUs | 32 GPUs |
| --- | --- | --- |
| Data parallelism | 447.0 GiB | 447.0 GiB |
| ZeRO-1 | 153.7 GiB | 122.2 GiB |
| ZeRO-2 | 104.8 GiB | 68.1 GiB |
| ZeRO-3 | 55.9 GiB | 14.0 GiB |

At 32 GPUs, **ZeRO-2 and ZeRO-3** sit under the 74.5 GiB training-state budget. Data parallelism and ZeRO-1 do not. That statement is only about optimizer / gradient / parameter storage. Activations, workspace buffers, the CUDA allocator, and framework state still have to fit.

---

## Collective operations

The simulator implements three collectives on **small PyTorch tensors**:

- **All-reduce** — every rank ends with the same reduced tensor.
- **Reduce-scatter** — ranks reduce, then each keeps one shard.
- **All-gather** — shards are concatenated and replicated.

Ring all-reduce of a parameter-sized gradient is modeled as **2P**. ZeRO-1 still all-reduces gradients (**2P**). ZeRO-2 replace the all-reduce with reduce-scatter of gradients plus all-gather of updated parameters (**2P**). ZeRO-3 adds a parameter all-gather for the forward and backward passes plus a gradient reduce-scatter (**3P**).

With \(P = 60\) GB:

| Path | Bandwidth | 2P time | 3P time |
| --- | --- | --- | --- |
| Intra-node | 450 GB/s | 0.27 s | 0.40 s |
| Inter-node | 50 GB/s | 2.40 s | 3.60 s |

These numbers are **simplified estimates**, not measurements of the Colab VM, NVLink, or InfiniBand.

---

## ZeRO stage comparison

| Strategy | Replicated | Sharded | Comm / step | 32-GPU fit? |
| --- | --- | --- | --- | --- |
| Data parallel | weights, grads, master, moments | — | 2P | No |
| ZeRO-1 | weights, grads | master, moments | 2P | No |
| ZeRO-2 | weights | grads, master, moments | 2P | Yes* |
| ZeRO-3 | — | weights, grads, master, moments | 3P | Yes* |

\*Theoretical training-state fit only.

Ownership is contiguous: rank \(r\) owns slice \([r \cdot \lceil P/N \rceil, \ldots)\). The union of shards is the full state; shards do not overlap. A tiny tensor demo reconstructs the original parameter vector with `all_gather`.

---

## Colab execution

The notebook is designed for a **fresh Google Colab runtime**. CPU is the guaranteed baseline. A single Colab GPU is used automatically for the tiny training demo when CUDA is available, and ignored when it is not.

Do not install DeepSpeed, flash-attn, or other CUDA-only distributed packages. Do not upgrade PyTorch unless you know you need to: Colab's preinstalled `torch` is the build you want.

### Workflow A — GitHub clone (after you publish)

Replace the placeholder with **your** repository URL. This README does not invent one.

```python
# In Colab
!git clone https://github.com/YOUR_USERNAME/zero-training-simulator.git /content/zero-training-simulator
```

Then open `/content/zero-training-simulator/zero_simulation.ipynb`, or upload the notebook and let the bootstrap cell find `/content/zero-training-simulator`.

### Workflow B — ZIP upload (use this if the repo is not public yet)

1. Zip the project so that `src/zero_sim/` and `tests/` are inside the archive.
2. In Colab: **Runtime → Run all**, or run the bootstrap cell.
3. If the helper package is not found, run the **ZIP upload** cell, choose the archive, and re-run the locator cell.

The notebook does **not** assume that the current working directory is the repository root. The bootstrap cell walks `cwd`, `/content/zero-training-simulator`, `/content`, and parent directories looking for `src/zero_sim/__init__.py`.

---

## Local tests

From the project root (a local Jupyter/Python environment, not required for Colab):

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

The notebook's section 15 runs the same suite with `pytest.main(...)` and raises if any test fails.

---

## Representative results (32 virtual GPUs, training state only)

| Strategy | Memory / GPU | Capacity | Headroom | Fit | Volume | Inter-node time |
| --- | --- | --- | --- | --- | --- | --- |
| Data parallelism | 447.0 GiB | 74.5 GiB | −372.5 GiB | No | 120 GB | 2.40 s |
| ZeRO-1 | 122.2 GiB | 74.5 GiB | −47.7 GiB | No | 120 GB | 2.40 s |
| ZeRO-2 | 68.1 GiB | 74.5 GiB | +6.4 GiB | Yes* | 120 GB | 2.40 s |
| ZeRO-3 | 14.0 GiB | 74.5 GiB | +60.5 GiB | Yes* | 180 GB | 3.60 s |

ZeRO-2 and ZeRO-3 satisfy the assignment's **theoretical 74.5 GiB training-state limit** at 32 GPUs because they stop replicating the expensive optimizer state (and, for ZeRO-3, the parameters). Data parallelism never shards anything. ZeRO-1 still keeps a full FP16 parameter copy **and** a full FP16 gradient copy on every GPU (4 bytes × 30e9 ≈ 111.8 GiB) plus a 1/32 optimizer shard, which remains above 74.5 GiB.

The trade-off is communication. ZeRO-3 is the only strategy whose simplified model pays **3P** instead of **2P**: parameters must be gathered for forward and backward, then gradients reduced and scattered. Lower memory does not mean a free lunch.

---

## Known limitations

- The 30B analysis is **arithmetic**, not a model load.
- Collectives on tiny tensors prove algebra, not network performance.
- Bandwidth numbers (450 GB/s intra-node, 50 GB/s inter-node) are assignment assumptions.
- Activation memory is an optional estimate with a crude formula.
- The toy MLP is a few thousand parameters. Virtual workers are Python objects, not `torch.distributed` ranks.
- Theoretical fit does not include CUDA context, allocator fragmentation, fused kernels, or checkpointed activations.
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

## Create and push a GitHub repository

This project does not come with a remote URL. After you create an empty GitHub repository named `zero-training-simulator`:

```bash
cd zero-training-simulator
git init
git add .
git commit -m "Add ZeRO training simulator for ERA Session 12."
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/zero-training-simulator.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub account. Do not commit `.env` files or API keys; this project does not need any.

---

## Submit the assignment

1. Run `zero_simulation.ipynb` **top to bottom in Google Colab**.
2. Confirm the pytest cell reports a passing suite.
3. In Colab: **File → Download → Download .ipynb**.
4. Push this repository to GitHub using the commands above.
5. Submit the downloaded `.ipynb` **and** the GitHub repository link.

---

## Student reflection prompts

1. Why does ZeRO-1 still miss a 74.5 GiB budget at 32 GPUs even though the optimizer is sharded?
2. If activation memory were 20 GiB per GPU, would ZeRO-2 still fit? What about ZeRO-3?
3. Why must mean-reduced local gradients be weighted by local batch size when those sizes differ?
4. When would you accept ZeRO-3's extra all-gather instead of stopping at ZeRO-2?
5. Which parts of this notebook would break first if you tried to drop in a real 30B checkpoint?
6. How would NVLink versus Ethernet change the communication-to-compute plot without changing the memory table?

---

## License

Educational material for ERA V5 Session 12. Treat the helper package as a teaching model, not as an optimizer implementation.
