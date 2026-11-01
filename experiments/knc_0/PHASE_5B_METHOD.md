# Phase 5B methodology — Known-Token Reconstruction

Written and committed before any Phase 5B model was trained.

## Deviation from the task text
The task fixes the architecture `384 -> 320 -> 320 -> 256`, i.e. `m=256` output
tokens, but also says "for k=64 there are no unknown tokens", which holds only
for a 64-token output. The architecture is kept. Therefore k=64 is an ordinary
condition with 192 unknown tokens, and an extra diagnostic condition k=256 (every
token known, nothing to recover) takes the role of the task's k=64 diagnostic.
k=256 is never counted as recovery.

## Models
- Architecture `384 -> 320 -> 320 -> 256`, ReLU, Adam lr=0.01, MSE on
  `tokens / S`. Scales S=4095 and S=10000 (primary), S=1 (raw-scale positive control).
- Stopping rule: token-space MSE `mean((out*S - tokens)^2) <= 1e-5`, checked on the
  saved (post-step) model; cap 5,000 steps. Models that miss the rule or do not
  reproduce every token from the correct key are recorded and reported; they are
  excluded only from the primary recovery tables, with their count stated.
- Secret input (unit vector, d=384) and target tokens come from OS entropy, are
  never written to disk and are independent of all public seeds.
- 3 pilot models per scale (settings selection only) and 30 held-out models per scale.

## Known-token conditions
k ∈ {0, 4, 8, 16, 32, 64} plus the k=256 diagnostic. Known positions are drawn
uniformly without replacement from a public RNG seeded by (phase seed, scale,
trial, k, draw), independent of the secret. Held-out models get 3 independent
position draws per k>0 (1 for k=0 and k=256). The analyst receives public weights,
architecture, S, and the selected indices and their values — nothing else.

## Analyst methods (public data + known tokens only)
1. `pca_lattice` (k=0 only): the Phase 5A blind attack, unchanged.
2. `plane_fit` (k>=4): the Phase 5A output plane (mean + 2 principal components of
   1,024 random unit-input outputs, in token units); plane coordinates fitted to the
   known values by least squares; candidate = rounded plane point.
3. `input_fit` (k>=4): gradient search over the input `z`. Objective: token-space MSE
   on the known positions only. R=8 restarts from random unit vectors, batched;
   Adam; the restart with the lowest known-position loss is selected; candidate =
   rounded, clamped output at that input. Settings chosen on pilot models from the
   grid lr ∈ {0.01, 0.05} × projection ∈ {free, unit sphere}, 2,000 steps. Selection
   rule (fixed now): highest mean unknown-position exact fraction over all pilot
   models, scales and k ∈ {4..64}; ties go to the earlier grid entry. The chosen
   setting is written to `pilot_summary.json` before held-out models are created.
4. `output_mean` (control): rounded mean of the random-input outputs; uses no known tokens.

## Controls
- k=0 blind baselines: `pca_lattice` and `output_mean`.
- Incorrect known tokens: for draw 0 of every k>0, `input_fit` and `plane_fit` are
  re-run with the true known values replaced by independent random tokens at the
  same positions. Success there would mean the method is not reconstructing from the
  known tokens.
- `output_mean` for every condition.

## Metrics (evaluator only, after the candidate is frozen)
On unknown positions only: exact fraction, whole-unknown-array recovery, MAE,
within-1/8/64 accuracy. Also: whether the candidate matches every known token
(question 1, attacker-checkable), cosine similarity of the found input to the
secret input (`input_fit`, evaluator-only, distinguishing output recovery from
secret-input recovery), runtime, forward and backward passes, process peak RSS.

Unit of analysis is the model. Whole-unknown recovery is reported as Wilson 95%
over model×draw pairs (draws are correlated within a model), plus the number of
models recovered on at least one draw and on all draws. Between-draw variability
is the mean over models of the standard deviation of exact fraction across draws.
Continuous metrics: mean with a normal 95% interval over models, and min/median/max.

## Integrity
- Analyst functions take no target, secret or evaluator argument; tests enforce
  the signatures, the public-bundle contents, and that two targets agreeing on the
  known positions yield identical candidates.
- Restarts, stopping points and hyperparameters are never chosen with unknown
  target values on held-out models.
- Unsuccessful trials are not evidence of practical security.

Run:

```bash
python experiments/knc_0/phase_5b.py --out /tmp/knc5b
pytest -q -p no:cacheprovider tests/test_knc_0_phase_5b.py
```
