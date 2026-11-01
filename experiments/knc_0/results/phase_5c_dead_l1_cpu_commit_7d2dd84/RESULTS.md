# Phase 5C results (reduced scope) — weight-only readout of the dead first layer

## Scope

By the owner's decision, Phase 5C was cut to one test (`PHASE_5C_METHOD.md`,
amendment 2). The Jacobian, observability, Levenberg–Marquardt, trajectory and
control experiments were **not run**, so the decision gate was not evaluated. This
report does not say whether local observability explains the Phase 5B known-token
transition.

## Setup

Run on CPU (4 threads) from commit `7d2dd841f7b943b4996cf9246fde85224d1d11c0`
(`config.json`). Python 3.13.18, PyTorch 2.15.0, NumPy 2.5.4; runtime 26 s, peak RSS
667 MiB.

- 30 trials; in each, an S=4095 and an S=10000 model share one OS-CSPRNG secret
  `(x, tokens)` and one initialization seed.
- Architecture `384 -> 320 -> 320 -> 256`, ReLU, Adam lr=0.01, MSE on `tokens/S`,
  token-space MSE ≤ 1e-5 on the saved model, cap 5,000 steps (as in Phase 5B).
- Analyst: `S·(W3·relu(b2) + b3)` from public weights only, rounded and clamped to
  0–4095. No input, no known tokens, no search; about 2 ms per model.
- Evaluator: active units in each layer at the true key, and the candidate's score
  on all 256 tokens.

## Results

| | S=4095 | S=10000 |
|---|---|---|
| Correct key reproduces all tokens | 30/30 [0.89, 1.00] | 30/30 [0.89, 1.00] |
| Key's first layer fully inactive | 0/30 [0.00, 0.11] | **19/30** [0.46, 0.78] |
| First-layer active units at the key, median (min–max) | 50.5 (12–62) | **0** (0–6) |
| **Full recovery from weights only** | 0/30 [0.00, 0.11] | **19/30** [0.46, 0.78] |
| … when the key's first layer is fully inactive | — | **19/19** [0.83, 1.00] |
| … when it is not | 0/30 [0.00, 0.11] | 0/11 [0.00, 0.26] |
| Candidate MAE, median (min–max), tokens | 854 (179–1,160) | 0 for inactive keys; 9.2 (4.2–216) otherwise |
| Candidate within ±64 tokens, median | 0.055 | 1.0 in both groups |

Wilson 95% intervals in brackets; the model is the unit.

The exploratory run in the conversation before amendment 2 (not saved) gave
S=10000 16/30 full recoveries, all 15 inactive-key models recovered, and S=4095 0/30.
It agrees with this run.

## Interpretation

- **S=10000 is often unprotected against a trivial blind readout.** In 19 of 30
  models the memorized output is reproduced exactly from public weights in
  milliseconds. Phase 5B's blind result for S=10000 (0/30) holds only for the
  methods tested there (PCA/lattice and output mean), not for the model.
- **Mechanism.** When the key's first layer is entirely inactive, the output in
  that region is constant and equals `W3·relu(b2) + b3`. The memorized array is then
  stored in the downstream weights and biases, independently of the input. Full
  recovery happened if and only if the key's first layer was fully inactive (19/19
  versus 0/41 across both scales).
- **Near-inactive keys still leak approximately.** The 11 S=10000 models whose
  key kept 1–6 active first-layer units were not recovered exactly, but the same
  weight-only candidate was within ±64 tokens on the median model (median MAE 9.2).
- **S=4095 behaves differently.** Keys keep 12–62 active first-layer units, and the
  candidate is no better than a rough guess (median MAE 854; uniform guessing
  ≈ 1,365). This says nothing about S=4095's security against other attacks: Phase
  5B already showed approximate leakage of 2–9 tokens with known tokens.
- **Link to Phase 5B (hypothesis, not tested here).** Known-token recovery at
  S=10000 may consist of finding inputs in or near this same inactive region. The
  smoke test before amendment 1 hinted at it (successful candidates had low-rank
  Jacobians and unobservable gain ≈ 1e-4), but those data were discarded and the
  experiments that would test it were cut.

## Limitations

- One architecture, input dimension, output length, optimizer and stopping rule.
  Why smaller target scales drive the first layer inactive at the key was not studied.
- Only one analyst method; failures are not evidence of security.
- 30 models per scale; the Wilson intervals are wide (S=10000: 0.46–0.78).
- No plots in this reduced phase; the plotting script `phase_5c_plots.py` belongs to
  the full protocol and was not used.

## Files

`config.json`, `aggregate.json` (counts with Wilson intervals, medians, partial
recoveries of active-key models) and `per_trial.csv` (60 rows: 30 per scale). No
secret inputs or targets are stored. Review is required before any further phase.
