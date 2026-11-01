# Phase 5C methodology — Jacobian geometry and known-token observability

Written and committed before any Phase 5C model was trained. All thresholds,
tolerances, budgets and selection rules below are fixed here; pilot models are
used only for a smoke test and timing, never to choose settings.

## Models
- Architecture `384 -> 320 -> 320 -> 256`, ReLU, Adam lr=0.01, MSE on `tokens/S`,
  stopping rule token-space MSE ≤ 1e-5 on the saved model, cap 5,000 steps
  (identical to Phase 5B).
- Scales S=4095 and S=10000, **paired**: in each trial both scales share one
  OS-CSPRNG secret `(x, tokens)` and one initialization seed.
- 3 pilot trials and 30 held-out trials (60 held-out models). Pilot uses one
  position draw per k; held-out uses 3 draws per k.

## Notation and units
`f(z)` is the network output in normalized units; `F(z) = S·f(z)` in token units.
`J = ∂f/∂z` (256×384) and `J_tok = S·J`. Every spectrum is stored in both units;
tables state which. `K` = known positions, `U` = unknown positions.

## Analyst (public weights + S + known positions/values only)
- `adam_fit`: Phase 5B frozen setting — 8 restarts from random unit vectors,
  Adam lr=0.05, unit-sphere projection, 2,000 steps, objective = token-space MSE on
  `K` only, restart selected by known loss.
- `lm_fit`: Levenberg–Marquardt on the same objective and the same 8 starts:
  step `Δ = -J_Kᵀ (J_K J_Kᵀ + λI)⁻¹ r` in token units, unit-sphere projection after
  each accepted step, λ₀ = 1e-3·trace(J_K J_Kᵀ)/k, λ×3 on rejection (≤10 tries),
  λ/3 on acceptance, 100 iterations or known MSE < 1e-6. Restart selected by known loss.
- k=0 baselines: Phase 5A blind PCA/lattice and the random-output mean.
- Positions: uniform without replacement from a public RNG seeded by
  (phase seed, trial, k, draw), independent of the secret; the same positions are
  used for both scales of a trial. k ∈ {4, 8, 16, 32, 64}, plus the k=256 diagnostic.

## Experiment A — Jacobian geometry
`J` is computed (autograd) at: the true key (evaluator-only, never passed to the
analyst), 16 random unit inputs (public seed), and every frozen analyst candidate.
For each point: singular values (normalized and token units), effective rank at
relative thresholds σ_i ≥ 1e-2·σ₁ and ≥ 1e-3·σ₁, absolute count σ_i,tok ≥ 50
(an input step of 0.01 moves an output by ≥ 0.5 token), stable rank ‖J‖²_F/σ₁²,
entropy rank exp(H(σ/Σσ)), and active ReLU counts. Principal angles: mean cosine
of principal angles between the top-8 left singular subspaces at the candidate and
at the key (evaluator-only), and between random inputs and the key as a baseline.
Global output PCA (1,024 random unit inputs, token units): components for 99% of
variance and entropy rank, to contrast local rank with global dimension.

## Experiment B — Known-coordinate observability
At each frozen candidate `z*` (analyst-computable): `J_K = J_tok[K]`, its numerical
rank at relative tolerance 1e-3 and condition number on that rank; the numerical
nullspace `N_K` (right singular vectors below the tolerance, completed by the
orthogonal complement of the row space). Primary statistic, the **unobservable
gain** `g = σ_max(J_tok[U] · N_K)`, in tokens of unknown-output change per unit
input step that leaves known outputs unchanged to first order; also `g_rel = g /
σ_max(J_tok[U])`.

Nonlinear check: along `v_worst` (top right singular vector of `J_tok[U]·N_K`) and
one random unit vector in span(`N_K`), step ε ∈ {1e-3, 1e-2}; record the actual max
change of known and unknown outputs (tokens), the linear prediction `ε·‖J_tok[U]v‖∞`,
and the number of ReLU units whose activity flips.

Failures are split into: (i) known positions not matched (`known_consistent` false,
an optimization failure), and (ii) known matched but unknown wrong (non-identifiability).

## Experiment C — Optimization landscape and scale
For draw 0 at k ∈ {16, 64, 256}, the Adam trajectory of every restart is logged at
steps {0, 10, 50, 100, 250, 500, 1000, 2000}: known MSE (tokens²), gradient norm
w.r.t. `z`, ReLU flips since the previous checkpoint, ‖z‖, and — evaluator-only,
logged after the run, never used — cosine to the key and unknown-token MAE. LM logs
iterations, final known MSE and accepted steps. Adam and LM from the same starts
separate optimizer choice from landscape.

## Experiment D — Controls
- Wrong known values (true positions, random tokens), draw 0, both methods.
- k=0 baselines (above).
- Shuffled outputs: a public permutation π reorders the last layer; `adam_fit` runs
  with permuted positions and the same values and seeds (draw 0, k ∈ {16, 64}); the
  un-permuted candidate is compared with the original candidate.
- Multiple inputs: for each `adam_fit` run, the number of restarts whose full rounded
  output equals the selected candidate's, and the minimum pairwise input cosine among them.

## Evaluation and statistics
Evaluator scores frozen candidates on `U` only: exact fraction, whole-unknown
recovery, MAE, within-1/8/64, cosine to the key. Model is the unit of analysis:
rates and means are per-model means over draws, with percentile bootstrap 95%
intervals over models (2,000 resamples, public seed). The ability of `g` to separate
successes from failures is reported as the Mann–Whitney AUC over pairs with a
model-level bootstrap interval.

## Decision gate (fixed now)
Local observability is judged to **explain** the Phase 5B transition only if all hold:
1. Among known-consistent candidates, `g` separates whole-unknown successes from
   failures with AUC ≥ 0.8 and bootstrap lower bound > 0.5, in every scale × k cell
   with at least 5 successes and 5 failures, pooled over k within each scale;
2. median `0.01·g` < 0.5 token for successes and > 0.5 token for failures;
3. at ε=1e-3 the nonlinear unknown change along `v_worst` agrees with the linear
   prediction within a factor of 2 in ≥ 80% of candidates.
Otherwise the results are reported as counterexamples or partial evidence, with
alternative hypotheses; no explanation is claimed.

Run:

```bash
python experiments/knc_0/phase_5c.py --out /tmp/knc5c
pytest -q -p no:cacheprovider tests/test_knc_0_phase_5c.py
```

## Amendment 1 (before any held-out model was trained)

Prompted by the smoke test (1 pilot + 1 held-out trial, 1 draw; output discarded) and
by 3 scratch models outside the experiment. No held-out data had been produced.

1. **Constant first-layer region.** In the smoke test the S=10000 key had every
   first-layer unit inactive, so `J = 0` at the key and the output there equals
   `S·(W3·relu(b2) + b3)`. In 2 of 3 scratch S=10000 models this weight-only
   expression reproduced the target exactly; in 3 of 3 S=4095 models it did not.
   Added: the analyst candidate `dead_l1` (k=0, blind, weights only); per-layer
   active-unit counts at the key, random inputs and candidates; and recovery rates
   split by whether the key's first layer is fully inactive (`by_key_l1_dead`).
2. **Linearity noise floor.** Successful candidates have unobservable gains near
   1e-4, so the first-order prediction at ε=1e-3 is ~1e-7 token while float32
   evaluation and ReLU flips move outputs by ~1e-2 token. A ratio of two values
   below noise is meaningless. Criterion 3 now also counts a candidate as consistent
   when both the predicted and the actual change are below 0.05 token.
3. **Zero spectrum.** When σ₁ = 0 every rank statistic is 0 (previously the relative
   thresholds counted all 256 zero singular values).

Criteria 1 and 2, all budgets, thresholds and the analyst methods of the original
methodology are unchanged.

## Amendment 2 — reduced scope (owner decision, 2026-11-01)

The full protocol (~75 CPU minutes) was stopped by the owner before any held-out
model was evaluated; one aborted pilot attempt produced no stored data. Phase 5C is
reduced to a single test of amendment 1's mechanism:

- 30 fresh paired trials (S=4095 and S=10000 share secret and initialization),
  trained exactly as above (R2, cap 5,000 steps).
- Analyst: `dead_l1` only — `S·(W3·relu(b2) + b3)` from public weights, no input
  and no known tokens.
- Evaluator: memorization from the key, first- and second-layer active units at the
  key, whether the first layer is fully inactive, and the candidate's exact fraction,
  full recovery, MAE and within-64 accuracy.
- Reported: Wilson 95% intervals for full recovery overall and split by whether the
  key's first layer is fully inactive. The model is the unit; there are no draws.

Not run: Experiments A–D (Jacobian spectra, observability `g`, Levenberg–Marquardt,
trajectories, shuffled and wrong-value controls). The decision gate is therefore not
evaluated, and no claim is made about whether local observability explains the
Phase 5B known-token transition. A single exploratory run of the same test, done in
the conversation before this amendment and not saved, gave S=10000 16/30 full
recoveries (15/15 dead keys) and S=4095 0/30; it is reported as preliminary only.

Runner: `experiments/knc_0/phase_5c_dead_l1.py`.
