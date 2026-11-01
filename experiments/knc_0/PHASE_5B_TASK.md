# Phase 5B — Known-Token Reconstruction

## Objective
Evaluate how much a normalized single-association ReLU MLP reveals when an analyst knows a small subset of its memorized 12-bit output tokens. The key question is whether partial known output allows recovery of the remaining tokens from public model weights. This is an empirical privacy/security evaluation, not a cryptographic-security claim.

## Starting evidence
Phase 5A found that a two-component PCA/lattice method recovered raw-scale ReMemory outputs in 30/30 strict-training trials, but recovered normalized targets (divided by 4095 or 10000) in 0/30 under the same strict token-space loss criterion. A preliminary separate test reported one successful normalized-model recovery with 16 known tokens out of 64, but this needs independent replication.

## Model and conditions
- Architecture: 384 -> 320 -> 320 -> 256, ReLU, Adam lr=0.01, MSE.
- Compare scales S=4095 and S=10000; optionally raw-scale S=1 as a positive control.
- Require exact correct-key token recovery in all included primary evaluations. Use token-space MSE <= 1e-5 with a documented training cap. Record convergence failures without silently excluding them.
- Generate secret input and random target tokens with OS CSPRNG independently of public experiment seeds.
- Use at least 30 held-out models per scale; use separate pilot models for choosing fixed analysis settings.
- CPU first; do not spend RunPod budget without explicit approval.

## Known-token conditions
Evaluate k=0,4,8,16,32,64 known token positions (k=0 is the blind baseline; k=64 is a diagnostic). Pre-register how positions are selected, e.g. uniformly without replacement. Repeat position selections where practical and report variance. For each condition the analyst receives public weights, architecture, normalization scale, and only the selected token indices and values. The analyst must not receive the secret input, remaining tokens, or evaluator-only diagnostics.

## Evaluation questions
1. Can the analyst construct a candidate output array consistent with known positions?
2. How many unknown tokens are exactly recovered, and how often is the entire unknown suffix/subset correct?
3. How does accuracy depend on k, target scale, and selection of known positions?
4. Is success attributable to genuine reconstruction or to an overly permissive stopping/selection rule?
5. How does the cost in time, evaluations, and memory grow with k?

Compare with the Phase 5A blind PCA baseline and trivial output-statistics controls. Any gradient-based input fitting must use only known token positions in its objective. Freeze each candidate before the independent evaluator scores unknown positions. Never select restarts, stopping points, or hyperparameters using unknown target values.

## Metrics
- Correct-key exact memorization rate.
- Exact recovery rate on unknown positions; whole-unknown-array recovery.
- MAE and within-1/8/64 accuracy on unknown positions only.
- Attack runtime, number of forward/backward passes, CPU RAM.
- Pilot vs held-out confidence intervals and per-model distributions.
- Distinguish recovery of output data from recovery of the actual secret input.

## Controls and integrity
- Separate attacker and evaluator APIs, with tests enforcing information boundaries.
- Keep public model bundles separate from evaluation secrets.
- Include k=0 blind controls, shuffled/incorrect known-token controls, and a simple public-weight output estimator.
- For k=64, there are no unknown tokens: do not count this as successful recovery.
- Do not infer practical security from unsuccessful trials.

## Deliverables
- Standalone Phase 5B runner, unit/smoke tests, and methodology.
- RESULTS.md, aggregate.json, per_trial.csv, machine-readable configuration, and accuracy-versus-known-token plots.
- Brief review of what changes relative to Phase 5A.
- Commit and push to experiment/knc-0; stop for review before any further phase.
- Do not modify main or existing ReMemory functionality.
