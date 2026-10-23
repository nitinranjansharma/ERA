# Experiment log

Append-only. Pre-register before running. Never fill a result with an invented measurement.

## E0 — Ledger invariants (COMPLETE, 2026-10-23)

**Question.** Do the README claims hold as executable assertions?

**Method.** `python -m pytest tests/ -q`

**Result (measured).** 35 passed. Covers share sums, IPF integrals, run and per-batch floors, 40% Indic sequence-token cap, disjoint SFT/escrow/PT partitions, disjoint Tier-A partition, epoch caps, longctx-inside-pool, Samanantar = 0, English Sangraha not slotted as Indic, escrow hashes and S1–S3 isolation, selector floors/ceilings/loud failure.

## E1 — Budget arithmetic (COMPLETE, 2026-10-23)

**Method.** `python scripts/ledger.py`

**Result (measured).** Agentic unique needed at 4 epochs: 14.6B main vs 0.267B, 1.8B anneal vs 0.360B. Composer C20 = 14 epochs on 6.8B. Long-context 6%→2% frees 116B. Reasoning PT-main 261.9B / 69.1B = 3.79 epochs after the 8B SFT holdout.

## E2 — Indic per language (COMPLETE, 2026-10-23)

**Method.** `python scripts/indic.py` on the Sangraha per-language card table.

**Result (derived from stated card rows).** Verified Indic 51.5B. Native synthetic 81.4B. Usable Sangraha native 157.2B. α* = 0.72. 14 funded / 8 gated at 1.30B. MILU 10/10. IndicGenBench 14/29.

## E3 — Selector simulation (COMPLETE, 2026-10-23)

**Method.** `python scripts/selector_sim.py --n 4000 --seed 7`

**Result (measured).** English proxy, no floors, S2 candidate mix: Indic **0.00%**, agentic **0.00%**. V5 proxy with floors: S2 Indic 16.5%, agentic 1.6%; anneal Indic 24.2%, agentic 8.0%. Floors hold; 1.25× ceilings hold within 2 pp on a 4000-token batch. This is a selector sim, not a trained 1B model.

## E4 — Escrow isolation (COMPLETE, 2026-10-23)

**Method.** `python scripts/build_escrow.py && pytest tests/test_escrow.py`

**Result (measured).** Five shard fixtures, unique IDs, sha256 matches, S1–S3 forbidden list contains every escrow ID, S5 cannot read the SFT holdout prefix, Indic/reason scheduled usage ≤ 4 epochs, agentic anneal **exceeds** 4 epochs and the test records the synthesis gap.

## E5 — Live HF audit (COMPLETE, 2026-10-23)

**Method.** `python scripts/supply_audit.py`

**Result (measured).** Sangraha partial, 65 splits, 1.018 TB. IndicCorpV2 37.6% converted, **8.0 bytes/token**. BPCC HTTP 401; stated 6.8B retained.

## T1 — Long-context 2% vs 6% (PRE-REGISTERED, not run)

1B/20B matched tok/param, two anneal arms, long-eval retrieval. Refuted if 2% degrades >2σ. **Result: pending. No number.**

## T2 — 40% C+D vs verified-heavy (PRE-REGISTERED, not run)

MILU per subject (India-specific vs general) + native-verified bpb. Refuted if India-specific subjects degrade >2σ. Prediction: D hurts India-specific, holds on general. **Result: pending.**

## T3 — Escrow vs early-spend (PRE-REGISTERED, not run)

SWE-bench Verified pass@1, BFCL v3, plus H2 non-regression on English val loss. **Result: pending.** Known limit: Llama 3 anneal gains were negligible at 405B.

## T4 — α* = 0.72 vs proportional (PRE-REGISTERED, not run)

MILU mean **and** minimum. Refuted if proportional matches, or the mean rises via Hindi alone.

## T5 — English-internal vs native CoT vs neither (PRE-REGISTERED, not run)

MILU by subject + script detection on CoT and answer. Arm A is the default; C is included because both A and B could be worse than nothing.

## Scale protocol (not yet executed)

| | 1B | 3B |
|---|---|---|
| tokens | 20B | 60B |
| tok/param | 20 | 20 |
| job | screen all hypotheses | confirm survivors |
| seeds | 3 on A0/A1 | 1 unless 1B was marginal |

Do not interpret a 1B/30B vs 3B/60B difference as a mixture effect.
