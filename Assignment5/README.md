# V5 Data Mixture and Curriculum

A 3T-token mixture for an Indic-capable coding and agentic model. Every share is checked against inventory; where a share exceeds unique supply, that is a named production programme, not a percentage that hides the gap.

Numbers are labelled **stated** (course inventory widget or a cited source), **derived** (arithmetic on stated rows), **measured** (computed or observed in this repo), or **hypothesis** (a planning partition, not a recount). Everything below is produced by [`scripts/`](scripts/). A passing `pytest` run is the evidence that the invariants hold.

**Status.** Specification v2. Twelve decisions taken ([`docs/DECISIONS.md`](docs/DECISIONS.md)), each with a reversal condition. Selector, escrow, Indic cap, SFT holdout and epoch ceilings are executable. **1B and 3B proxy training has not been run.** Those tables are empty on purpose.

---

## 1. Executive decision

**3T total** (session range 2.4–4T) · **97% main run (2,910B) / 3% anneal (90B)**. Anneal size follows OLMo 2's 50/100/300B mixes.

| lane | main | anneal | demand | unique available to that pool | epochs | status |
|---|---|---|---|---|---|---|
| General web | 34% | 8% | 997B | 4,691B | 0.21 | funded |
| Code | 24% | 20% | 716B | 1,103B | 0.65 | funded |
| **Indic** | **17%** | **28%** | 520B | 187B partitioned | **2.8–3.3** | funded, tightest real lane after audit |
| STEM | 12% | 10% | 358B | 146B | 2.45 | funded |
| Reasoning traces | 9% | 18% | 278B | 77.1B after SFT holdout | 3.61 | funded — 78B of 85.1B is V4 AON |
| Long-context | 2% | 8% | 65B | 100B | 0.65 | funded |
| **Agentic** | **2%** | **8%** | 65B | **0.627B** | **main 218× / anneal 20×** | **programme — §4** |

Feasibility test: **epochs ≤ 4** (arXiv:2305.16264). Constraint, not a preference.

**98% of the main run is funded from supply the inventory lists today. 2% is a named synthesis programme.** All delivery risk sits in one lane — the lane the session's protected floor exists to defend.

**Changes from the course composer `24/2/6/6/16/12/34`:**

- **Long-context 6% → 2%**, freeing 116B. arXiv:2402.10171 finds 0.5–5B tokens suffice to retrieve anywhere in a 128K context. 2% still allocates 58B in the main run, 12× the published upper bound.
- **Reasoning 6% → 9%.** Llama 3 spends ~25% on math+reasoning; 9% traces + 12% STEM = 21%.
- **Indic 16% → 17%.** The tier split matters more than the headline — §3.

**Three inventory corrections that move epochs, not adjectives:**

1. 12.76B of "verified Indic" is an **English** Sangraha split. Verified Indic is 51.5B, not 64.3B.
2. Sangraha synthetic native/Latin splits have **identical row counts**. Only the native half (81.4B) is scored by MILU / IndicGenBench.
3. **Samanantar is inside BPCC.** Counting both is double-counting. Tier C unique is 6.8B, not 5B+2B.

**SFT holdout, stated as a partition not a footnote.** 85.1B unique reasoning = 8B SFT holdout (never in pretraining) + 8B anneal escrow + 69.1B main-run PT. The old 12B "real reasoning supply" figure is not used: it cannot be both the parent pool and the PT remainder after an 8B holdout.

**Tier-A is a partition, not a subset.** 51.5B verified Indic = 15.0B anneal escrow + 36.5B main-run remainder. A unique token is not counted twice.

---

## 2. Inventory and feasibility

Full row-level map: [`data/inventory.json`](data/inventory.json), audit [`docs/SUPPLY.md`](docs/SUPPLY.md). Reproduce with `python scripts/ledger.py` and `python scripts/supply_audit.py`.

| claim as given (widget) | used here | how |
|---|---|---|
| Indic verified 64.3B | **51.5B** | 12.76B English split removed (**derived**) |
| Indic synthetic 162.7B | **81.4B** native-script | 1:1 transliteration (**derived**, card row counts) |
| Tier C = BPCC 3B + Samanantar 2B | **6.8B BPCC Indic-side; Samanantar = 0** | subset, not extra supply |
| STEM 250B (composer) / web 4,500B | **STEM 146B, web 4,691B** | peS2o 42 + proof-pile-2 55 + D4 49; remainder is web |
| Reasoning 85.1B all available to PT | **77.1B PT** | 8B SFT holdout disjoint (**hypothesis partition**) |

**Measured in this repo, 2026-10-23.** HuggingFace datasets-server:

- `ai4bharat/sangraha`: partial conversion, 177,355,164 / 268,356,447 rows, 65 splits, 1.018 TB.
- `ai4bharat/IndicCorpV2`: 37.6% converted; extrapolated 167GB against the published 20.9B tokens → **8.0 bytes/token**.
- `ai4bharat/BPCC`: HTTP 401. Ledger keeps the stated 6.8B Indic-side figure.

**B-overlap sensitivity (executable).** Indic B unique is 45.2B if IndicCorpV2 ∩ Sangraha-unverified = ∅. If the overlap exceeds 5.0B, B demand is 6.6 epochs and the B share must fall. The ledger prints this every run; it does not assume the overlap is zero.

**What a public-only rebuild supports.** 78B of reasoning is V4 AON. Without it, reasoning unique is 7.1B → **1.0% of the run, not 9%**. Losing V4 corpora opens an 8% hole. That dependency is load-bearing and invisible in a flat percentage table.

---

## 3. Mixture allocation

Stage mixtures **integrate to the lane budget**. The course curriculum widget does not enforce this. Solved by iterative proportional fitting: every stage sums to 100%, every lane's token-weighted mean equals its share.

### 3.1 Named lane shares

Long-context is a **named lane inside each stage pool, not added on top**. S4's 44B of long-context is part of S4's 698B.

| lane | S1 233B | S2 931B | S3 1,048B | S4 698B | anneal 90B | run tokens |
|---|---|---|---|---|---|---|
| web | 55.6% | 45.1% | 28.2% | 20.7% | 8.0% | 997B |
| code | 14.4% | 19.1% | 26.8% | 29.5% | 20.0% | 716B |
| reason | 1.9% | 4.8% | 12.5% | 11.8% | 18.0% | 278B |
| longctx | 0.2% | 0.4% | 0.9% | 6.4% | 8.0% | 65B |
| **indic** | **17.0%** | **16.8%** | **16.9%** | **17.4%** | **28.0%** | **520B** |
| stem | 8.9% | 11.8% | 12.8% | 12.1% | 10.0% | 358B |
| agentic | 2.0% | 2.0% | 2.0% | 2.0% | 8.0% | 65B |

Indic is deliberately flat in the main run. A protected capability that ramps is still leftover budget. Agentic stays at its floor in the main run; concentration is the anneal, which is the only scale at which 8% is arithmetically feasible.

### 3.2 Long-context overlay (inside the pool)

Inventory composition of the longctx lane: **60% repo-packed code, 40% packed books**. Folded into capabilities without changing the token total.

| from longctx | S1 | S2 | S3 | S4 | anneal |
|---|---|---|---|---|---|
| → code | 0.11% | 0.26% | 0.53% | 3.82% | 4.80% |
| → web | 0.07% | 0.18% | 0.35% | 2.54% | 3.20% |
| longctx tokens | 0.4B | 4.1B | 9.3B | **44.4B** | 7.2B |

### 3.3 Effective capability mix (longctx folded)

| lane | S1 | S2 | S3 | S4 | anneal |
|---|---|---|---|---|---|
| web | 55.7% | 45.3% | 28.5% | 23.2% | 11.2% |
| code | 14.5% | 19.3% | 27.3% | **33.4%** | 24.8% |
| reason | 1.9% | 4.8% | 12.5% | 11.8% | 18.0% |
| indic | 17.0% | 16.8% | 16.9% | 17.4% | 28.0% |
| stem | 8.9% | 11.8% | 12.8% | 12.1% | 10.0% |
| agentic | 2.0% | 2.0% | 2.0% | 2.0% | 8.0% |

S4 **named** code is 29.5%. S4 **effective** code is 33.4% after the overlay. Both numbers are real; they answer different questions. The 32B-at-S3 confusion from the previous draft does not arise here: long-context is 2% of the main run, staged late, and already inside the 3T.

### 3.4 Indic tiers — hard sequence-token cap at 40%

The cap is **translated + synthetic as a percentage of Indic sequence tokens**, in every stage and run-wide. It is not a native-score-weighted view.

Composer default A40/B25/**C20**/D15 demands 99B of parallel text against 6.8B unique → **14 epochs**. Impossible. The feasible split is C at 2% (existing parallel only) and D at 38% (synthetic, already conservative vs D's 59% of Indic supply).

| stage | Indic B | A verified | B unverified | C translated | D synthetic | **C+D** |
|---|---|---|---|---|---|---|
| S1 | 39.5 | 11.1 | 12.6 | 0.8 | 15.0 | **40.0%** |
| S2 | 156.8 | 43.9 | 50.2 | 3.1 | 59.6 | **40.0%** |
| S3 | 177.0 | 49.6 | 56.6 | 3.5 | 67.3 | **40.0%** |
| S4 | 121.4 | 34.0 | 38.8 | 2.4 | 46.1 | **40.0%** |
| anneal | 25.2 | 12.6 | 2.5 | 0.5 | 9.6 | **40.0%** |
| **run** | **519.9** | 151.1 | 160.8 | 10.4 | 197.6 | **40.0%** |

Anneal is 50% verified, **and still 40% C+D**. Those two statements are both true. "50% verified" does not imply a lower non-native cap.

Generated translation is routed to D, not C. C is existing parallel corpora only.

**Per language.** Temperature sampling `p ∝ S^α` against a 4-epoch per-language cap. `α* = 0.72` is the largest temperature at which the weakest MILU language (Punjabi, 6.20B) already hits its cap; lower α buys MILU nothing and pins more languages. **14 languages funded, 8 gated** at 1.30B unique (the 24× cliff between Assamese 6.16B and Sindhi 0.26B). MILU 10/10. IndicGenBench **14/29**, not 29/29. Full tables: [`docs/INDIC.md`](docs/INDIC.md).

---

## 4. Selector, floors, anneal reserve

### Floors

Two kinds, both tested.

| kind | Indic | Agentic | Reasoning | Web |
|---|---|---|---|---|
| **Run-integral** (composer clamp) | 12% | 2% | 6% | 25% |
| **Per-batch S1** | 12% | 1.5% | 1.2% | 25% |
| Per-batch S2 | 12% | 1.5% | 3.0% | 25% |
| Per-batch S3 | 12% | 1.5% | 8.0% | 18% |
| Per-batch S4 | 12% | 1.5% | 7.5% | 18% |
| Per-batch anneal | 20% | 8% | 12% | 8% |

Reasoning ramps, so S1/S2 sit below the 6% run floor on purpose; the integral is 9%. Agentic's run floor **equals** its allocation — there is no slack to concede. Ceiling is **1.25 × that stage's target**. If eligible supply cannot meet a floor, the loader **raises** (`RuntimeError`); it does not silently undershoot. Declared fallback: `steal_web`, then fail.

### Loss-token accounting

Sequence share ≠ gradient share. Agentic observations carry no loss.

| stage | Agentic seq → loss | Reasoning seq → loss |
|---|---|---|
| S2 | 2.0% → 1.1% | 4.8% → 4.4% |
| S4 | 2.0% → 1.1% | 11.8% → 10.9% |
| anneal | 8.0% → **4.5%** | 18.0% → 17.2% |

Agentic 0.53 is **stated** (session widget 356/668). Reasoning 0.90 is a **hypothesis** until measured on packed examples. Anneal agentic loss share has a floor of 3.5% (tested).

### Selector simulation — measured here

`python scripts/selector_sim.py`. Candidate stream = the stage mixture, oversampled by 1/0.40. Eligible lanes ranked; protected lanes bypass; floors inject; ceilings clip.

**Starvation arm (english proxy, no floors), measured, seed 7, n=4000:** Indic **0.00%**, agentic **0.00%**. With V5 floors on the same stream: Indic 16.5%, agentic 1.6% at S2. Rebalancing the proxy without floors does not rescue agentic in this sim either. Only a protected floor works. This is a **selector simulation**, not a 1B training run.

### Anneal escrow — tested, not asserted

Held in [`data/escrow_manifest.json`](data/escrow_manifest.json). Shard hashes match files on disk. [`data/loader_s1_s3.json`](data/loader_s1_s3.json) names every escrow shard as forbidden. `tests/test_escrow.py` checks uniqueness, hashes, isolation, and that Indic/reasoning scheduled usage fits unique × 4 epochs. Agentic anneal **does not** fit — 7.2B / 0.36B = 20 epochs — and the test asserts the synthesis gap rather than pretending the reserve is sufficient.

| escrowed unique | size | anneal use | epochs |
|---|---|---|---|
| Verified Indic Tier-A partition | 15.0B | 12.6B | 0.84 |
| AON long traces (disjoint from PT-main and SFT) | 8.0B | 16.2B | 2.02 |
| Long agentic trajectories (SWE-Gym + SWE-smith + OpenHands) | 0.36B | 7.2B | 20 → synthesize 1.44B |

Fixtures under `data/escrow_shards/` are audit artifacts, not 15B of Indic text.

---

## 5. Dataset → benchmark mapping

The chain is **lane → benchmark → required training shape → inventory row → unique tokens**. A benchmark name is not a training set. Full table: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Each row of [`data/inventory.json`](data/inventory.json) carries name, source, shard ID, unique tokens, languages, slots, tier, verification, license, contamination check, native/translated/synthetic, and a stated/derived/hypothesis label.

| slot | inventory evidence (unique) | why it is in the mix | primary benchmarks |
|---|---|---|---|
| web | DCLM 2.6T, FineWeb-Edu 1.3T, D1/D2 V4 791B | breadth; the slack that absorbs every fallback | MMLU (common sense is unmeasured) |
| code | Stack v2 900B, D3 V4 199B, **CommitPackFT 4B** | generation + **diffs** | LiveCodeBench, Aider Polyglot, Codeforces |
| stem | peS2o 42B, proof-pile-2 55B, D4 V4 49B | papers and formal math | GPQA, AIME, MMLU-science |
| reason | AON PT 62B + public 7.1B; 8B escrow; 8B SFT | short/medium traces so SFT/RLVR can land | AIME, GPQA Diamond, HLE, FrontierMath |
| agentic | FC 0.267B + traj 0.360B | tool calls vs long recovery traces — two shapes | BFCL v3, τ-bench, SWE-bench Verified |
| indic | A 51.5B, B 45.2B (overlap risk), C 6.8B, D 81.4B | native fluency vs capability transfer | MILU 10/10, IndicGenBench **14/29** |
| longctx | repo-packed 60B, books 40B | retrieval at target context | long-eval |

**CommitPackFT** is the only row that teaches search/replace diffs. Natural weight is 0.56% of the code lane. Upsampled to the **4-epoch ceiling = 16B = 2.2%**. An 8× schedule would be 8 epochs and would fail the feasibility test; we do not use one.

**Not claimed:** Terminal-Bench, WebArena, OSWorld. They need a live shell, hosted sites, a desktop. V4 never built that. No token budget buys them.

**Loss shape, agentic (the thing most often got wrong):** green = assistant planning, tool calls, final answer, *including a call that fails*. Grey = user turns, tool returns, repo files, shell output. Violet = verifier outcome. Applying loss to observations teaches the model to invent tool results.

---

## 6. Curriculum and stability

Full schedule: [`docs/CURRICULUM.md`](docs/CURRICULUM.md).

**Difficulty bands — assignment procedure, not just examples.**

An example is scored 0/1 on five factors, summed to 0–5, then mapped B0–B5. Factors: (1) dependent reasoning or execution steps ≥3 / ≥8 / ≥15; (2) files, tools, or evidence sources ≥2 / ≥4; (3) error recovery required; (4) context ≥4k / ≥32k or cross-document; (5) independently measured solve rate <30% on a reference model. Bands are validated on a 200-example manual sample per lane before the run; an LLM label is not sufficient. Concrete examples and stage ranges are in the curriculum doc.

**Reasoning-length bands** are shares of the *reasoning lane*. Length = generated reasoning tokens, not total example tokens. R0/R1 boundary is `<64` vs `≥64`. Long traces require a verified answer; verbose incorrect traces are rejected, not counted toward R3/R4. R3/R4 are only legal from S3 (32k context available). The anneal is 8k context, so R4 is **not** an anneal band — B5 difficulty at R2 length is.

| band | generated trace | S2 | S3 | S4 | anneal |
|---|---|---|---|---|---|
| R0 ≤64 | one-line work | 25% | 20% | 5% | 5% |
| R1 64–512 | two-step | 45% | 35% | 15% | 15% |
| R2 0.5k–4k | derive + check | 30% | 30% | 40% | 50% |
| R3 4k–16k | alternatives + self-correction | 0 | 15% | 30% | 30% |
| R4 16k–32k | only with verified answer | 0 | 0 | 10% | 0 |

R0–R4 sum to 100% per stage. R3+R4 capped because HIGH→ULTRA is **stated** by the session widget as 2.15× tokens for +4 accuracy points and labelled wasted effort. Teaching the dial requires the short bands to dominate.

**Seam stability.** Embeddings stay trainable across every seam (the V4 Hindi-share incident is a frozen-embedding story more than a mixture-shift story; session widget default is a 151× spike frozen vs ~8× unfrozen — **stated**, not re-measured here). Blend every seam over ≥2B tokens; the anneal seam gets 4B.

---

## 7. Proxy experiments

Protocol: Llama 3 annealing-as-evaluation / OLMo 2 microannealing — **not** from-scratch 3T. Two questions are separated:

- **Mixture quality:** same model size, same training tokens, same optimizer. 1B / 20B tokens (20 tok/param).
- **Scale robustness:** 3B / 60B tokens (also 20 tok/param). A 1B/30B vs 3B/60B ladder would confound mixture with tokens-per-parameter; we do not use it.

Uncertainty: 3 seeds on A0 and A1 at 1B. σ is the seed standard deviation of the metric. A result counts only if the gap is **> 2σ and ≥ the predeclared minimum effect**. Primary metric = per-lane held-out loss/bpb on a **fixed, decontaminated** eval set. Task metrics (MILU, BFCL, SWE-bench) are secondary and are ignored when they sit near chance.

Full pre-registration: [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

| ID | claim | metric | confirmed if | refuted if |
|---|---|---|---|---|
| H1 floors | english-only selector, no floors, starves scarce lanes | native-Indic bpb; agentic assistant-token loss | A2 worse by ≥3% Indic **and** ≥8% agentic, each >2σ | either gap <1% → lower that floor |
| H2 escrow | reserve beats early-spend at equal tokens | same + tool-call EM | A1 better by ≥2σ on both, no English regression >1.5% | no gap → drop escrow |
| H3 non-native cap | 40% C+D does not hurt native fluency | native-verified bpb; MGSM-hi/bn | bpb within +1% **and** probe ≥ +2 points, both >2σ | bpb worse >1% → cut C+D to ≤20% |
| H4 share size | diminishing returns past 20% Indic | Indic bpb; English val loss | gain(10→20) ≥ 2× gain(20→30) **and** English penalty ≤1.5% | English penalty >1.5% → Indic 15% |
| H5 agentic loss | ~4.5% anneal gradient is enough | tool-call EM, trajectory val loss | 2× arm gain <2σ **or** <3% relative | gain ≥5% relative **and** >2σ, no other-lane regression → S4 agentic 18% |
| H6 blend | hard switch spikes grad-norm ≥3×; blended ≤1.5× | grad-norm / EMA | both thresholds | blended >2× → 8B band |

**What has actually been run**

| ID | result |
|---|---|
| E0 ledger invariants | **35/35 pytest passed**, 2026-10-23. Floors, caps, disjoint partitions, epoch ceilings, escrow hashes. |
| E1 budget arithmetic | Agentic unique needed 14.6B main / 1.8B anneal vs 0.267B / 0.360B. Composer C20 is impossible. |
| E2 Indic per-language | Usable Sangraha native 157.2B; α*=0.72; 14 funded / 8 gated; MILU 10/10; IGB 14/29. |
| E3 selector sim | **Measured:** english no-floor Indic 0.00%, agentic 0.00%. V5 floors restore both. |
| E4 escrow tests | Isolation, unique IDs, sha256 match, S1–S3 cannot name escrow shards. |
| E5 live HF audit | IndicCorpV2 **8.0 bytes/token** measured; Sangraha 65 splits; BPCC 401. |
| T1–T5 1B/3B training | **Not run. No numbers. Do not invent any.** |

Arms: A0 web-heavy 80%, no floors · **A1 this plan** · A2 english-only proxy, no floors · A3 escrow spent in S1 · A4 C+D = 15% · A5/A6 Indic 10%/30% · A7 hard seam, frozen embeddings · A8 agentic 2×.

---

## 8. Data gate and reproduction

Counting unit: **cleaned AND documented unique tokens**. Raw tokens, repeated training tokens, and generated tokens do not count. The course's numeric threshold is not in the files this plan started from; this table does not invent one.

| lane | required unique | raw stated | documented | cleaned (this repo) | eligible | shortfall | % |
|---|---|---|---|---|---|---|---|
| Indic A escrow | 15.0B | 15.0B | 15.0B | 0 | 0 | 15.0B | 0 |
| Indic A main | 36.5B | 51.5B | 51.5B | 0 | 0 | 36.5B | 0 |
| Indic B | 45.2B | 45.2B | 45.2B | 0 | 0 | 45.2B | 0 |
| Indic C | 6.8B | 6.8B | 6.8B | 0 | 0 | 6.8B | 0 |
| Indic D native | 81.4B | 81.4B | 81.4B | 0 | 0 | 81.4B | 0 |
| Agentic traj | 0.36B | 0.36B | 0.36B | 0 | 0 | 0.36B | 0 |
| Agentic FC | 0.27B | 0.27B | 0.27B | 0 | 0 | 0.27B | 0 |
| Reason PT main | 69.1B | 69.1B | 69.1B | 0 | 0 | 69.1B | 0 |
| Reason escrow | 8.0B | 8.0B | 8.0B | 0 | 0 | 8.0B | 0 |
| Reason SFT | 8.0B | 8.0B | 8.0B | 0 | 0 | 8.0B | 0 |

**Gate verdict: NOT MET** on the cleaned-token unit. Session 4 provenance-stamped shards are not in this repository. Documented-from-widget coverage is high; that is a different unit. Cleaning priority is the starved unique pools: verified Indic, real agentic trajectories, verified reasoning traces.

```bash
python scripts/ledger.py          # specification checker
python scripts/agentic.py         # synthesis programme in samples
python scripts/indic.py           # per-language allocation
python scripts/indic_gated.py     # 14 funded / 8 gated
python scripts/selector_sim.py    # floors vs starvation arm
python scripts/build_escrow.py    # manifest + shard hashes
python scripts/data_gate.py       # this table
python scripts/supply_audit.py    # live HF queries
python -m pytest tests/ -q        # 35 invariants; this is the evidence
python scripts/render_all.py      # everything above
```

**Supporting documents:** [BUDGET](docs/BUDGET.md) · [SUPPLY](docs/SUPPLY.md) · [INDIC](docs/INDIC.md) · [CURRICULUM](docs/CURRICULUM.md) · [BENCHMARKS](docs/BENCHMARKS.md) · [AGENTIC](docs/AGENTIC.md) · [DECISIONS](docs/DECISIONS.md) · [EXPERIMENT_LOG](docs/EXPERIMENT_LOG.md) · [CROSS_LANE](docs/CROSS_LANE.md)

What would change the plan: verified Indic smaller than 51.5B → Indic share falls. H1 or H2 refuted → drop that mechanism. IndicCorpV2 overlap >5B → B share falls. 4T run scale → Indic must fall to ~15.7% or it breaches 4 epochs. T1–T5 numbers, when they exist, override every sentence above.
