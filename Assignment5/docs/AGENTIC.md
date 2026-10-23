# Agentic synthesis programme

Reproduce: `python scripts/agentic.py`. The lane cannot be filled from real data at any scale in 2.4–4T.

## Two shapes

| shape | samples | tokens | mean | role |
|---|---|---|---|---|
| Short function-calling | 458,000 | 267M | 583 tok | main-run 2% — schema-generable, execution-verifiable |
| Long trajectories | 38,400 | 360M | 9,375 tok | anneal 8% — needs real repos |

SWE-Gym is 2,400 samples at 62,500 tok each. xLAM is 60,000 samples at 417 tok. Sample counts misjudge training cost by two orders of magnitude.

## Gap at 4 epochs

| | main (FC) | anneal (traj) |
|---|---|---|
| demand | 58.2B seen (30.8B supervised at 53%) | 7.2B seen (3.8B supervised) |
| unique needed | 14.6B | 1.8B |
| have | 0.267B | 0.360B |
| synthesize | **14.3B = 54×** | **1.44B = 4×** |
| production target | **≈ 24.5M new calls** | **≈ 153,600 new trajectories** |

153,600 trajectories is 5.9× SWE-smith's 26,000. Putting the expensive shape in the main run would demand ~1.5M trajectories. The anneal placement cuts the hard half by an order of magnitude.

## Method and QA gate

FC: APIGen/ToolACE. Schema pool → intent → call → **execute** → format check → semantic check. Mask: green on the call only.

Trajectories: SWE-smith/SWE-Gym. Mine repos, inject faults, roll out, keep hidden-test passes. Retain failed-then-recovered traces. Failed call is supervised; environment error text is not.

Hard gates before admission: (1) every tool return from a real invocation; (2) verifier-labelled success; (3) mask audited on a sample; (4) decontaminated vs SWE-bench Verified/Live/Pro. Unverified synthetic agentic data is worse than none.

If the execution budget binds, the lane is cut. Shipping unverified samples is not a fallback.
