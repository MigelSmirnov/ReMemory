# Phase 5B results — Known-Token Reconstruction

## Setup

Run on CPU (4 threads) from commit `ec3fd6c70990c6e3ec3b39288483612f2a36321d`
(recorded in `config.json`); methodology frozen earlier in `8344486`
(`PHASE_5B_METHOD.md`). Python 3.13.18, PyTorch 2.15.0, NumPy 2.5.4. Total
runtime 3,334 s; peak process RSS 699 MiB for the whole run.

- Architecture `384 -> 320 -> 320 -> 256`, Adam lr=0.01, MSE on `tokens/S`.
  Scales S=4095 and S=10000 (primary) and S=1 (raw-scale positive control).
- Stopping rule: token-space MSE ≤ 1e-5 on the saved model, cap 5,000 steps.
- 3 pilot models per scale (settings selection only), 30 held-out models per scale,
  each with its own OS-CSPRNG secret input and target.
- k ∈ {0, 4, 8, 16, 32, 64} known positions out of 256 (uniform without
  replacement, public seed), 3 position draws per k>0 on held-out models, plus the
  k=256 diagnostic.

**Deviation from the task text.** The task says that for k=64 no tokens remain
unknown; that holds only for a 64-token output, but the prescribed architecture
has 256 outputs. The architecture was kept, so k=64 leaves 192 unknown tokens and
k=256 serves as the "everything known" diagnostic (`PHASE_5B_METHOD.md`).

## Training

Every held-out model met the stopping rule and reproduced all 256 tokens from its
correct key: 30/30 per scale (Wilson 95% [0.89, 1.00]). No model was excluded.
Median steps: 266 (S=1), 227 (S=4095), 222 (S=10000).

## Pilot and frozen settings

The pre-registered rule picked `input_fit` setting 3 — lr=0.05, unit-sphere
projection, 2,000 steps, 8 restarts — with pilot mean unknown exact fraction
0.568, against 0.489, 0.518 and 0.456 for the other grid entries
(`pilot_summary.json`). It was written before any held-out model was created.

Pilot versus held-out, `input_fit`, whole unknown array (pilot: 3 models, 1 draw):

| Scale | k | Pilot | Held-out (model × draw) |
|---|---|---|---|
| S=4095 | 4–64 | 0/3 at every k | 0/90 at every k |
| S=10000 | 4, 8 | 0/3 | 0/90, 1/90 |
| S=10000 | 16 | 3/3 [0.44, 1.00] | 36/90 [0.30, 0.50] |
| S=10000 | 32 | 3/3 [0.44, 1.00] | 65/90 [0.62, 0.80] |
| S=10000 | 64 | 3/3 [0.44, 1.00] | 64/90 [0.61, 0.79] |

The pilot overestimated the S=10000 rate at k=16; held-out values are the ones to use.

## Main results (held-out, 30 models per scale)

Whole unknown array recovered, as model × draw pairs with Wilson 95%; in brackets
after it, the number of models recovered on at least one / on all three draws.
Mean exact fraction is computed per model (mean over draws), then over models.

**S=10000**

| k | input_fit whole | input_fit exact | input_fit MAE | plane_fit exact | output_mean exact |
|---|---|---|---|---|---|
| 0 (blind PCA) | 0/30 [0.00, 0.11] | 0.001 | 521 | — | 0.001 |
| 4 | 0/90 [0.00, 0.04] | 0.057 | 6.3 | 0.011 | 0.001 |
| 8 | 1/90 [0.00, 0.06] (1/0) | 0.079 | 5.2 | 0.014 | 0.001 |
| 16 | **36/90 [0.30, 0.50]** (18/6) | 0.476 | 4.4 | 0.016 | 0.001 |
| 32 | **65/90 [0.62, 0.80]** (24/20) | 0.821 | 0.7 | 0.015 | 0.001 |
| 64 | **64/90 [0.61, 0.79]** (24/18) | 0.842 | 0.3 | 0.015 | 0.001 |

**S=4095**

| k | input_fit whole | input_fit exact | input_fit MAE | plane_fit exact | output_mean exact |
|---|---|---|---|---|---|
| 0 (blind PCA) | 0/30 [0.00, 0.11] | 0.001 | 833 | — | 0.004 |
| 4 | 0/90 [0.00, 0.04] | 0.046 | 9.1 | 0.009 | 0.004 |
| 8 | 0/90 [0.00, 0.04] | 0.054 | 7.3 | 0.010 | 0.004 |
| 16 | 0/90 [0.00, 0.04] | 0.053 | 6.9 | 0.011 | 0.004 |
| 32 | 0/90 [0.00, 0.04] | 0.081 | 4.8 | 0.011 | 0.004 |
| 64 | 0/90 [0.00, 0.04] | 0.187 | 2.1 | 0.011 | 0.004 |

**S=1 (positive control).** Blind PCA/lattice 30/30; `input_fit` and `plane_fit`
90/90 [0.96, 1.00] at every k ≥ 4. This matches Phase 5A.

No `plane_fit` or `output_mean` candidate recovered a whole unknown array for
S=4095 or S=10000.

## Approximate leakage

Exact recovery understates what leaks. On unknown positions `input_fit` has
within-64 accuracy ≥ 0.99 at every k ≥ 4 for both normalized scales, and mean
absolute error of 0.3–9.1 tokens out of 4,096 (falling as k grows). `plane_fit` reaches within-64
0.80–0.92 (MAE 26–42). Even the blind `output_mean`, which uses no known tokens,
has MAE 134 and within-64 0.41 for S=4095 (500 and 0.10 for S=10000). For
comparison, a uniformly random guess has expected MAE ≈ 1,365.

## Evaluation questions

**1. Consistency with known positions.** `input_fit` produced candidates that
match every known token in 87/90 (k=16), 73/90 (k=32) and 66/90 (k=64) S=10000
pairs. Consistency is weak evidence at small k: at k=16, 51 consistent
candidates were wrong on unknown positions, while at k=64 only 2 were. No
inconsistent candidate was correct. For S=4095 consistency fell with k (78/90 at
k=16, 4/90 at k=64). `plane_fit` was never consistent for the normalized scales.

**2–3. Unknown-token recovery versus k, scale and positions.** For S=10000
whole-array recovery appears at k=16 and plateaus around 70% for k=32–64; for
S=4095 it never appears within this search budget. Known-position selection
matters most at the transition: at S=10000, k=16 the mean between-draw standard
deviation of exact fraction is 0.19, and 18 models were recovered on at least one
draw but only 6 on all three. At k ≥ 32 it falls to 0.05–0.07.

**4. Genuine reconstruction or a permissive rule?** The incorrect-known-token
control (true positions, random values; draw 0) recovered 0/30 whole arrays at every
k and scale, with unknown exact fraction ≤ 0.002. Restarts were selected by
known-position loss only, settings were frozen on pilot models, and training used
the strict R2 rule. Success therefore depends on the true known values, not on a
selection artefact.

The S=4095 failure is a search limitation, not evidence that no solution exists.
At k=256 (every token known) `input_fit` matched all known tokens in 0/30 S=4095
models — median final known-position MSE 0.87 token² — even though the true key
reproduces them exactly. For S=10000 it succeeded in 22/30 (median loss 0.0).

**5. Cost.** `input_fit` costs 16,008 forward and 16,000 backward passes (8
restarts × 2,000 steps, batched), median 1.3–1.5 s per run, independent of k.
`plane_fit` reuses 1,024 forward passes and is negligible. The blind
PCA/lattice baseline takes about 2.7 s. RAM was measured only for the whole
process (699 MiB peak), not per attack.

## Output data versus the secret input

Recovering the output did not recover the secret input. For successful
S=10000 runs, the cosine between the found input and the secret input had median
0.805 (0.27–0.84), against 0.775 for failures; in S=1 every success had
0.59–0.72. Many inputs produce the memorized output. The cosine is still far from
the ≈0 expected for an unrelated unit vector in 384 dimensions. For normalized
scales the wrong-values control gives cosine ≈ 0, so the true known tokens pull
the search towards the secret direction: partial information about the input,
not key recovery. For S=1 the wrong-values control still reaches cosine 0.39–0.47,
so there it partly reflects model geometry rather than the known tokens.

## Changes relative to Phase 5A

- Phase 5A found normalized targets resistant to the blind attack (0/30). Phase 5B
  confirms that for k=0, but at S=10000 16–64 known tokens out of 256 recover the
  remaining tokens in 40–72% of model × draw pairs. Normalization defeats the
  blind plane attack, not attacks with partial knowledge.
- The dependence on scale is not monotonic: S=10000 leaked more than S=4095 under
  known tokens, although both resisted the blind attack.
- The preliminary single normalized success (16 known tokens) is replicated as a
  regular effect at S=10000, k=16 (18/30 models on at least one draw), in the
  ReMemory architecture rather than the KNC-0 architecture where it was first seen.

## Limitations

- One architecture, one input dimension (384), one output length (256), one
  search configuration. S=4095 results are a statement about this budget only.
- Draws within a model are correlated; Wilson intervals over model × draw pairs
  are therefore optimistic. Model-level counts are reported alongside.
- The wrong-values control ran on draw 0 only. It also ran at k=256, which the
  methodology did not list; those rows are diagnostic only.
- `known_consistent` is trivially true at k=0.
- Unsuccessful trials are not evidence of practical security.

## Files

`config.json`, `pilot_summary.json`, `aggregate.json` (training summaries and every
scale × k × variant × method condition), `per_trial.csv` (5,868 rows: 288 pilot,
5,580 held-out), `exact_fraction_vs_k.svg`, `whole_unknown_vs_k.svg` (produced by
`experiments/knc_0/phase_5b_plots.py`; each point has a hover tooltip). No secret
inputs or targets are stored. Review is required before any further phase.
