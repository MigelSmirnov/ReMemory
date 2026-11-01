# Phase 4D results

The preregistered CPU experiment ran from commit
`df80c567d3e4cb103abdc0d790ec59b215b36afd`: three disjoint pilots and 30
fresh held-out models for each width 4, 8, 16, 32 and 64. Each public candidate
pool contained 4,096 standard-normal inputs. All 150 held-out models exactly
memorized the 64-token target with the correct input.

## Candidate occurrence

The target occurred at least once in 22/30 pools at width 4, 14/30 at width 8,
1/30 at width 16, and 0/30 at widths 32 and 64. Mean per-input occurrence fell
from 10.69% at width 4 to 0.915% at width 8 and 0.00163% at width 16. These are
fixed-budget empirical rates, not asymptotic probabilities.

Candidate diversity moved in the opposite direction. Mean distinct rounded arrays
out of 4,096 samples were 3,511, 3,939, 4,095.8, 4,096 and 4,096 as width grew.
Mean distinct activation patterns were 52, 1,257, 3,983, 4,096 and 4,096.
Thus the sampled output map changes from clustered to effectively unique across
the tested inputs.

## Target-free identification

Frequency mode Recall@1/Recall@10 exactly matched the occurrence upper bound:
22/30, 14/30, 1/30, 0/30 and 0/30. Local-stability Recall@10 matched the same
counts; its Recall@1 was 22/30, 13/30, 1/30, 0/30 and 0/30. There were no held-out
models where the target occurred but mode failed to place it in top 10. All
remaining failures were `target never observed`.

This does not mean mode recovers a key. At narrow widths, if the memorized output
appears, it occupies a high-frequency constant region and is easy to label from
public samples. At width 16 the one success occurred among essentially unique
arrays and should be treated as a sparse finite-sample event. Median-medoid,
analytic moment, random-first and zero baselines did not identify targets beyond
the narrow-width special cases recorded in `ranking_summary.csv`.

## Mechanistic control and scaling limits

Untrained controls also clustered strongly at width 4: their mean maximum array
fraction was 10.91%, close to the trained pool's clustered scale. At width 8 it
fell to 0.52%; at width 16 and above every sampled untrained output was unique.
This confirms that clustering is substantially caused by narrow ReLU geometry.
Training determines whether the memorized target becomes one of those clusters.

Sampling runtime rose only from about 0.054 seconds at width 4 to 0.076 seconds
at width 64; the experiment therefore provides no evidence for GPU-days or an
exponential complexity law. With a fixed 4,096-sample budget, width 32/64 results
only establish non-observation, not impossibility or security.

Machine-readable outputs include `aggregate.json`, all 150 held-out rows in
`per_trial.csv`, `ranking_summary.csv`, disjoint `pilot_summary.json`, and four
SVG plots. No secret input or target is stored. Review is required before any
next phase.
