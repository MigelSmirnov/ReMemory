# Phase 4D — Secret Identification Complexity

## Research question
As ReLU MLP hidden width grows, how do (a) the chance that an independently sampled input yields the memorized 64-token array and (b) the ability to identify that array without knowing the key or target change? Distinguish candidate generation from candidate identification. Do not equate the number of activation patterns with attack complexity or claim exponential hardness.

## Baseline and architecture
- d=256, output m=64 uniformly random 12-bit tokens.
- Two hidden ReLU layers with widths h=4,8,16,32,64.
- Adam + MSE; follow prior exact-memorization protocol.
- At least 30 independently generated models per width; use CPU and profile before any GPU request.
- Keep existing ReMemory code and previous experiments unchanged.

## Experiment A — Candidate occurrence
For each model, sample a fixed, predeclared number of independent inputs from the documented key prior. Measure the evaluator-only proportion that yields the exact memorized array, number of distinct rounded output arrays, frequency distribution, activation-pattern diversity, and runtime. Report per-model values as well as across-model averages; a mean occurrence rate alone can hide many models with zero probability in the tested sample.

## Experiment B — Target-free candidate identification
The attacker receives only public weights, architecture, input prior and sampled outputs. Implement pre-registered, target-independent rankings such as output frequency/mode, cluster density, low output variance under local perturbation, and analytic public-weight output estimates. Freeze a top-1 candidate and a fixed top-k ranking before evaluation. Evaluate Recall@1, Recall@k, MRR, exact full-array recovery, and time. Include random-candidate and fixed-zero baselines. No hidden target, secret input, correct activation pattern, checksum, or oracle selection may be used to rank candidates.

## Experiment C — Identification versus occurrence
For each model, evaluator records whether the correct output occurred anywhere in the sampled candidate pool. Compare that oracle upper bound with target-free Recall@1/k, and decompose failures into 'target never observed' versus 'target observed but not identified'. Clearly label the oracle metric as evaluator-only. Do not treat success in an output-frequency heuristic as key recovery.

## Experiment D — Scaling
Plot occurrence probability, target-free identification rate, candidate diversity, and runtime against width and parameter count. Use Wilson intervals for success rates and uncertainty intervals across independent models. Test simple scaling fits only as exploratory descriptions; do not extrapolate to large widths or GPU-days without evidence. Account for fixed sampling budget and correlated output tokens.

## Experimental safeguards
- Fresh OS-generated secret input and target, independent of public seeds.
- Separate attacker/evaluator APIs; candidate rankings fixed before truth is revealed.
- Store no secrets in public bundles or reports.
- Pre-register ranking rules using disjoint pilot models.
- Report failures to memorize rather than excluding them.
- Use at least one synthetic negative control and check whether candidate clustering is caused by ReLU constant regions.

## Deliverables
Standalone Phase 4D runner, unit tests, methodology, RESULTS.md, aggregate.json, per_trial.csv, ranking summaries, and plots under experiments/knc_0/. First perform a CPU smoke test, then full held-out evaluation. Commit and push to experiment/knc-0, stop for review. Do not modify main or rent RunPod without approval.
