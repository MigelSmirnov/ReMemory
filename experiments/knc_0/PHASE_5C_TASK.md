# Phase 5C — Why Known Tokens Reveal Unknown Outputs

## Objective
Explain the mechanism behind Phase 5B known-token reconstruction in normalized ReMemory MLPs. Investigate whether a small subset of output coordinates constrains the output manifold sufficiently to predict remaining coordinates. Do not infer cryptographic security from failures.

## Evidence
Phase 5B, architecture 384→320→320→256, ReLU, Adam/MSE, token-space MSE <=1e-5:
- S=10000: full unknown-array recovery at k=16: 36/90 model×draw; k=32: 65/90; k=64: 64/90.
- S=4095: no full recovery in those budgets, but at k=64 MAE 2.1 tokens on unknown coordinates.
- Optimization at S=4095 failed even with all 256 known tokens; failure cannot be interpreted as privacy.
- Draws within each model are correlated. Report model-level uncertainty.

## Main questions
1. Is there a low-dimensional *local* output manifold even when global 2D PCA fails?
2. Does the Jacobian of output with respect to input have low effective rank near candidate inputs?
3. Can known output coordinates constrain the active local degrees of freedom, explaining why 16–32 coordinates sometimes predict 200+ others?
4. Why does S=10000 yield easier optimization than S=4095, despite identical target information?
5. Are successful reconstructed outputs associated with recovery of the true input, or with alternative inputs yielding the same output?

## Experiment A — Jacobian geometry
For independently trained S=4095 and S=10000 models, evaluate J=∂f/∂x at: true input (evaluator-only), random inputs, and attacker-generated candidate inputs. Compute singular spectrum, effective rank under multiple thresholds, stable rank, and principal-angle comparisons. Express singular values in token units and in normalized units, clearly distinguishing them. Do not expose secret-input Jacobians to the attacker.

## Experiment B — Known-coordinate observability
For k=4,8,16,32,64, select known positions with a preregistered random rule. Compute rank and conditioning of the restricted Jacobian J_known at attacker-generated inputs. Compare to full J and determine whether variations that preserve known coordinates also preserve unknown coordinates. Test this by constructing small local perturbations in the numerical nullspace of J_known, checking actual nonlinear outputs and ReLU pattern changes. Use evaluator-only unknown-token errors after candidates are frozen.

## Experiment C — Optimization landscape and scale
Use matched secrets and initializations across S=4095 and S=10000. Track known-token loss, gradient norms, activation-pattern changes, input norm/cosine (cosine evaluator-only), and held-out unknown-token error during evaluation only. Separate landscape geometry from optimizer choice. Compare at least Adam and a deterministic least-squares/Gauss–Newton-style local baseline where computationally feasible. Avoid tuning against unknown tokens. Record failures and convergence diagnostics.

## Experiment D — Controls
- Include wrong-known-token controls and k=0 baselines.
- Check that a model with shuffled output coordinates preserves the results under correspondingly shuffled known positions.
- Check whether successful outputs can be generated from multiple distinct inputs.
- Distinguish local Jacobian rank from global PCA dimension; neither alone proves unique reconstruction.

## Protocol
- Start with a CPU smoke test on 2–3 pilot models per scale; preregister thresholds, numerical tolerances, selection rules and budgets.
- Then run at least 30 fresh held-out models per scale if feasible, with 3 known-position draws per k.
- Keep attacker and evaluator APIs separate. Secrets generated independently from public seeds; no unknown target values in candidate ranking or stopping.
- Report model-level bootstrap confidence intervals; do not treat model×draw pairs as independent.
- Profile CPU time before requesting any paid GPU. No RunPod without approval.
- Preserve existing ReMemory code and experiments; work only in experiment/knc-0.

## Deliverables
Independent Phase 5C runner, unit tests, methodology, RESULTS.md, aggregate.json, per_trial.csv, spectra and accuracy/conditioning plots. Compare explicitly with Phase 5B. Commit and push to experiment/knc-0, then stop for review.

## Decision gate
Conclude whether local low-rank structure and known-coordinate observability explain the Phase 5B recovery transition. If not, document counterexamples and alternative hypotheses; do not claim an explanation without direct evidence.
