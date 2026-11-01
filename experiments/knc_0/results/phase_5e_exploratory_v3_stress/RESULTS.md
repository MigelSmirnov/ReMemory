# Phase 5E (exploratory) — stress test of the variant that held in Phase 5D

**Exploratory.** Unlike Phases 5A–5D there was no methodology committed in advance:
the attacks and settings were chosen in conversation, there were no pilot or held-out
splits, and the evaluator was run in the same script. Treat these numbers as
indications for a future pre-registered phase, not as results of the same standing.

## Question

In Phase 5D, variant V3 (S=10000, Adam lr=0.001) let no attack read a text. Does it
hold against (1) the output-line/plane attack that broke ReMemory and `latent_recall`,
and (2) input search with a much larger budget and more known tokens?

## Setup

- Script: `experiments/knc_0/exploratory/v3_stress.py 10`; log:
  `v3_stress_10models.log` (one line per model plus the summary).
- 10 models: architecture `384 -> 320 -> 320 -> 256`, MSE on `tokens/10000`, Adam
  lr=0.001, token-space MSE ≤ 1e-5 (Phase 5D `train`, cap 5,000 steps). Inputs are
  unit vectors and targets are uniform 12-bit tokens, both from fixed public seeds
  (3000 + t), not CSPRNG — the run is reproducible but its secrets are not secret.
- Geometry: 2,048 random unit inputs; the evaluator measures the target's maximum
  distance from the affine span of the mean and the top 1/2/8/32 principal components.
- `dead_l1` readout `S·(W3·relu(b2)+b3)`.
- `input_fit` (Phase 5C `adam_fit`) with 32 restarts × 8,000 steps (16× the
  Phase 5B–5D budget), known positions k ∈ {64, 128, 256} out of 256, one draw each.

## Results

All 10 models reproduced their 256 tokens exactly from the key (median 249 steps).

| Check | Result (median over 10 models) |
|---|---|
| RMS along PC1 / PC2 / PC3 / PC4 | 952 / 129 / 124 / 119 tokens |
| Target distance from the 1-D / 2-D / 8-D / 32-D principal subspace (max over tokens) | 326 / 321 / 272 / 195 tokens |
| `dead_l1` exact tokens | 0% |

| Known tokens | Known reproduced exactly | Unknown exact | Full recovery | Unknown MAE |
|---|---|---|---|---|
| 64 (¼) | 7/10 | 1.6% | 0/10 | **14.6** |
| 128 (½) | 0/10 | 12.5% | 0/10 | **2.3** |
| 256 (all) | 2/10 | — | — | — |

About 8 s per `input_fit` run on 4 CPU threads.

## Reading

- **No low-dimensional output structure.** The outputs spread over many directions,
  and the target lies hundreds of tokens away from any low-dimensional principal
  subspace. The line/plane attack has nothing to work with, and the first layer was
  never fully inactive.
- **A 16× larger search budget did not help.** At k=64 the unknown-token MAE (14.6) is
  in the range Phase 5D measured with the standard budget (12.8–14.5). Even with every
  token known, the search reproduced the stored array exactly in only 2 of 10 models,
  although the true key reproduces it. The limit is the function's landscape for
  this optimizer, not compute.
- **Leakage grows with what the attacker knows.** With half of the tokens known,
  the rest are recovered to about ±2 units. Phase 5D found that an error of about 0.7
  units leaves English text 75% byte-correct; at 2.3 units part of an English text is
  likely readable. Text readability at k=128 was not measured.

## Limitations

- 10 models, one draw per k, random tokens rather than text, public seeds.
- One search method (projected Adam with restarts). Levenberg–Marquardt, other
  objectives and language-model decoding were not tried on this variant.
- Not evidence of security: the variant resisted these attacks under these budgets.
