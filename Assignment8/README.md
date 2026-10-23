# How Attention Works Now

A chronological web explainer of modern attention mechanisms.

Live site: *(add URL after deploy)*  
This repo: static HTML/CSS/JS — open `index.html` or host on any static host (GitHub Pages, Netlify, Vercel).

## What this is

The page starts with **scaled dot-product attention** and then walks techniques **in the order they were launched**, not the order they were taught.

Each section answers:

1. What already existed  
2. What problem people hit  
3. What the new mechanism changed  
4. What it bought / gave up / when to choose it  

## Local preview

```bash
# from this directory
python3 -m http.server 8080
# then open http://localhost:8080
```

## Chronology sources

Dates used for ordering are the **first public appearance** (arXiv *submitted* date from the arXiv API, or the original post/release when that is the primary source). Verified 2026-03 against primary links below.

| Order | Technique | Date used | Primary source |
|------:|-----------|-----------|----------------|
| 1 | Scaled dot-product attention | 2017-06-12 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 2 | Sinusoidal PE | 2017-06-12 | same paper |
| 3 | Absolute learned positions | 2018-06-11 | [GPT-1 PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| 4 | Sparse / top-k attention | 2019-04-23 | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| 5 | Multi-query attention (MQA) | 2019-11-06 | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 6 | Sliding-window attention | 2020-04-10 | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) (Longformer); later [Mistral 7B](https://arxiv.org/abs/2310.06825) |
| 7 | Linear attention | 2020-06-29 | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 8 | RoPE | 2021-04-20 | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 9 | ALiBi | 2021-08-27 | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 10 | Grouped-query attention (GQA) | 2023-05-22 | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 11 | NTK-aware RoPE scaling | 2023-06-29 | [bloc97 / r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/); formula in [YaRN appendix](https://arxiv.org/abs/2309.00071) |
| 12 | YaRN | 2023-08-31 | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| 13 | Attention sinks | 2023-09-29 | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| 14 | Multi-head latent attention (MLA) | 2024-05-07 | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| 15 | Delta rule / DeltaNet | 2024-06-10 | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) (rule lineage: Schlag et al. 2021) |
| 16 | Gated DeltaNet | 2024-12-09 | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 17 | Native Sparse Attention (extra) | 2025-02-16 | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| 18 | DeepSeek Sparse Attention (DSA) | 2025-09-30 (V3.2-Exp) · paper 2025-12-02 | [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) |
| 19 | DroPE | 2025-12-13 | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) |
| 20 | Compressed Sparse Attention (CSA) | 2026-04-26 | [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) |

### Dating notes (where agents often go wrong)

- **Sinusoidal vs learned absolute:** both appear as options in the 2017 Transformer paper; GPT-1 (2018) is used as the clear productization of *learned* absolute tables.
- **YaRN arXiv id `2309.00071`:** submitted **2023-08-31** (September id, August submission).
- **NTK-aware:** community post first (2023-06-29), not a standalone arXiv paper; YaRN cites and records the base-change formula.
- **Delta rule:** associative delta updates are older (e.g. Schlag et al. 2021); the timeline entry is the **scalable DeltaNet** training paper (2024-06).
- **DeepSeek DSA vs CSA:** DSA (sparse selection on latent/token entries) ships with V3.2-Exp (Sep 2025). **CSA** (compress blocks, then DSA) is specified in DeepSeek-V4 (2026-04). They are separate timeline entries because compression is a later cost lever.
- **DroPE vs session “8K→256K” anecdotes:** DroPE is the public algorithm (drop PE after pretraining + recalibration). Specific V4 cookbook numbers are treated as run evidence, not as DroPE’s paper abstract.

## Design choices

- Interactions only where they change understanding (mask on/off, √dₖ collapse, RoPE relative angle, ALiBi slopes, KV head sharing, sinks, delta overwrite, frequency stretch, CSA top-k).
- Pros/cons forced for every mechanism — including defaults like GQA and RoPE.
- Extra mechanism beyond the minimum list: **NSA (2025)** as the trainable/hardware-aligned sparse precursor to DeepSeek’s later compression stack.

## License

Course assignment material. Paper titles and equations remain with their authors; explanations are original.
