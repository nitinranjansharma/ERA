# Curriculum

Reproduce: `python scripts/curriculum.py` (wraps the ledger IPF).

## Budget-integral constraint

Per-stage mixtures must integrate to the lane budget. IPF enforces (1) each stage sums to 100% and (2) each lane's token-weighted mean equals `MAIN_SHARE`. Named, overlay, and effective tables are in the README.

Shape rationale: web falls; reasoning enters late; long-context is absent until S4; Indic is flat; agentic equals its floor in the main run.

## Seam bands

| seam | at | band |
|---|---|---|
| S1 → S2 | ~233B | 2B |
| S2 → S3 | ~1,164B | 2B |
| S3 → S4 | ~2,212B | 2B |
| S4 → anneal | ~2,910B | **4B** |

Rule 1: embeddings stay trainable. Rule 2: never hard-switch. Guard: halt if grad-norm > 2.5× EMA for ≥50 steps, roll back one checkpoint, double the band. Session widget default of ~151× frozen vs ~8× unfrozen is **stated**, not re-measured here.

## Difficulty assignment

Each example is scored 0/1 on five factors, summed 0–5 → B0–B5.

| factor | 1 if |
|---|---|
| Steps | dependent reasoning or execution steps ≥3 (B2), ≥8 (B4), ≥15 (B5) — take the highest that applies as the step contribution, then OR with the others |
| Sources | ≥2 files/tools/evidence sources (one point); ≥4 (still one point — the step factor already grades complexity) |
| Recovery | error recovery or verification is required |
| Context | ≥4k tokens or cross-document |
| Hardness | independently measured solve rate <30% on a frozen reference model |

Mapping: 0 → B0, 1 → B1, 2 → B2, 3 → B3, 4 → B4, 5 → B5. Validate on a 200-example manual sample per lane; Cohen's κ vs two annotators must be ≥0.6 before the label is used in sampling. LLM-only labels are not accepted.

Concrete examples:

| band | code | math | Indic | agentic |
|---|---|---|---|---|
| B0 | `print("hi")` | 2+3 | news sentence | — |
| B1 | `is_palindrome` | "Ravi has 5 pens…" | Hindi/Tamil sentence | `get_weather("Pune")` |
| B2 | MBPP+ with tests | MATH 1–3 | agri advisory | 2–3 dependent calls |
| B3 | 3-file bugfix | AMC, 3+ steps | Hinglish word problem | 6–10 steps, one recovery |
| B4 | SWE-bench-Lite-like | AIME | Marathi legal QA 8k | 15-step investigation |
| B5 | SWE-bench Verified + hidden tests | FrontierMath | long government doc QA | multi-repo, failed-then-recovered |

Stage ranges: S1 B0–B2 · S2 B1–B3 · S3 B2–B4 · S4 B3–B5 · anneal B4–B5. Weights shift 70/25/5/0 → 5/20/40/35 on (B1,B2,B3,B4+) across S1→anneal.

## Reasoning-length bands

Length = **generated reasoning tokens**. R0 is `<64`, R1 is `≥64 and <512`. R3/R4 require a verified answer; traces that are long and wrong are discarded, not up-weighted. R4 needs 32k context (S4 only). Anneal context is 8k, so R4 share there is 0. Shares in the README sum to 100% per stage.

Domain split of the reasoning lane: math 40%, code 35%, general 15%, Indic-language traces 10%. Indic-language traces sit inside the reasoning lane; a language-tag audit tracks total Indic exposure ~2–3 points above the 17% lane figure. That audit is a coverage tag, not a second Indic budget.
