# Decisions

A decision without a reversal condition is a preference.

| # | decision | reverses if |
|---|---|---|
| 1 | run scale **3T** | 4T is chosen — Indic then breaches 4 epochs and must fall to ~15.7% |
| 2 | anneal **3% (90B)** | trajectory synthesis is cheaper than costed, making 5% affordable |
| 3 | agentic **2% floor, fund the programme** | H2/T3: anneal reserve earns nothing |
| 4 | reasoning **English-internal CoT default** | T5: native CoT wins on India-specific MILU subjects |
| 5 | tail languages **gated at 1.30B** | a gated language crosses the threshold — automatic |
| 6 | **prioritise** NCERT/SCERT/NPTEL/IGNOU | it fails; Indic-STEM requirement drops to ~3.4% |
| 7 | execution environments **not claimed** | V4 infra delivers a live harness |
| 8 | romanized Indic **1% claiming Dakshina** | a generation-fluency eval is added to the rubric |
| 9 | CommitPackFT **4-epoch upsample (16B)** | more diff-following unique data arrives |
| 10 | IndicGenBench **14/29** | follows #5 |
| 11 | Indic **17%, C+D cap 40% sequence tokens** | H3 refutes the cap, or scale moves to 4T |
| 12 | STEM unique **146B** | course confirms 250B |
| 13 | SFT holdout **8B disjoint of 85.1B** | Sessions 17–18 do not need a pretraining holdout |
| 14 | Tier-A **15B escrow + 36.5B main, disjoint** | quality partition is not separable in the stamps |

## Notes

- **#3** is not optional given E3: english no-floor selector measured Indic 0% and agentic 0%.
- **#4** is a default on verifiability (question/answer translations are exact-match checkable; free-form native CoT is not). T5 still runs with a neither-arm, because both arms could be worse than nothing. The slice is 10% of the reasoning lane, a re-presentation of AON, and does not relieve 3.79 PT epochs.
- **#9** was 8× in an earlier draft, which is 8 epochs of 4B unique and fails the cap. Corrected to 4× unique = 16B.
- **#11** was a 40% cap that S2–S4 violated. The split is now A28/B32/C2/D38 so the cap holds as a sequence-token rule.
- **#13/#14** close the two double-count bugs: SFT holdout vs PT pool, and Tier-A-as-subset vs partition.
