# Lane → benchmark → data shape → inventory

A benchmark is not a training set. Each inventory row is in `data/inventory.json` with license, shard ID, unique tokens, contamination check, and a stated/derived/hypothesis label.

Loss colours: **green** supervised, **grey** masked, **violet** reward only.

## Code — 24% (716B)

| benchmark | scores | required shape |
|---|---|---|
| LiveCodeBench | pass@1, rolling window | green: solution + reasoning · grey: problem, hidden tests |
| Aider Polyglot | strict search/replace that must apply | green: **the diff only** · grey: files, tests |
| Codeforces | ELO | green: program + reasoning · grey: problem |

Datasets: Stack v2 900B (B) · D3 V4 199B (B) · CommitPackFT 4B (B). CommitPackFT is the only diff-following row; upsampled to 16B = 4 epochs = 2.2% of the lane. Contamination: LiveCodeBench cutoff; Aider's ~225 exercises held out.

## Agentic — 2% main + 8% anneal (65B)

| benchmark | fed by |
|---|---|
| BFCL v3 | short function-calling |
| τ-bench | short FC + policy dialogue |
| SWE-bench Verified / Live | long trajectories |
| GAIA / BrowseComp | long trajectories + harness |

**Not claimed:** Terminal-Bench, WebArena, OSWorld (live environments V4 never built).

Green = assistant plan, calls, final answer, including failed calls. Grey = user, tool returns, repo, shell. Violet = verifier. Stated supervision yield 356/668 = 53%.

Datasets FC 267M: Glaive, ToolBench, ToolACE, Nexus, xLAM, Hermes. Trajectories 360M, escrowed: SWE-Gym, SWE-smith, OpenHands. Contamination: overlapping SWE-bench repos and issue IDs dropped.

## Reasoning — 9% (278B)

AIME, GPQA Diamond, HLE, FrontierMath. Green: CoT + answer. Grey: problem.

Unique: AON PT 62B + OpenThoughts2 3B + OpenMathReasoning 2B + OpenR1-Math 1.6B + NuminaMath 0.5B = 69.1B PT-main. Plus 8B anneal escrow and 8B SFT holdout, disjoint. 78B of 85.1B is V4. Public-only rebuild is 7.1B.

## STEM — 12% (358B)

peS2o 42 · proof-pile-2 55 · D4 V4 49 = 146B (stated rows). Composer 250B is unused.

## Long-context — 2% (65B)

Repo-packed code 60B + packed books 40B. long-eval retrieval. Staged 4k→8k→16k→32k; one length per batch.

## Indic — 17% (520B)

MILU 10/10. IndicGenBench 14/29 (10 zero-token languages, 5 gated). Dakshina claimed on a 1% romanized sub-lane, not on 81B of transliteration.

## Web — 34% (997B)

DCLM 2.6T · FineWeb-Edu 1.3T · D1/D2 V4 791B. Buys MMLU. Real product is common sense, which has no clean benchmark. Hard floor 25%.
