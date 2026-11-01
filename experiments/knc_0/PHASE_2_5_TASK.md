# KNC-0 Phase 2.5 — Leakage Analysis (Codex task)

## Context
Baseline results: `experiments/knc_0/results/phase1_2_cpu_10_commit_65d3e07/`. Correct-key full recovery 10/10; cheap baselines 0/10 full recovery. Mean output over random probes has mean token MAE ~492.5; investigate whether this is a meaningful information leak, not a security guarantee.

## Scope
Implement a **separate** Phase 2.5 runner and tests in `experiments/knc_0/`. Do not modify `main` or original ReMemory pipeline. CPU first, no RunPod yet. Use the same 256→64→64→64 ReLU model and compare against the published Phase 1–2 baseline.

### Required experiments
1. Random-probe mean convergence: N=100, 1k, 10k, 100k, at least 10 independently trained models. Measure MAE, RMSE, exact token fraction, full recovery, runtime; stream batches for large N.
2. Per-coordinate output statistics: mean, std, median, quantiles (e.g. 5/25/75/95%), and distribution of deviations from target. Report aggregate across trials; avoid dumping target values.
3. Hidden activation diagnostics: compare patterns on correct input vs random inputs, including layerwise active fraction and activation-pattern agreement. **Correct input is evaluator-only**, never available to attack code.
4. Parameter-only baseline: estimate outputs using last-layer bias and public-weight-derived hidden activation statistics; compare to simple random-output mean and untrained network.
5. Calibration controls: compare trained model with untrained model and with a shuffled/random target under same training protocol. Report baseline MAE expected from trivial predictors (e.g. midpoint 2047.5) to avoid overstating leakage.

### Threat model and experiment hygiene
- Public attacker bundle: final state_dict, architecture, nonsecret config.
- Secret input and target generated with OS CSPRNG, not reconstructible from public training seed. Training randomness may be reproducible independently.
- Attack implementations may not import or read evaluator secrets, access hidden target for optimization or choose the best trial, or select probes based on target. Use clean API boundaries.
- Evaluation may use hidden truth **after** an attacker candidate is frozen. Label any oracle-assisted diagnostics explicitly and keep out of attack success rates.
- Do not call random-probe mean an attack on the secret key: it is an estimator of target information leakage.
- Do not conclude security from failed tests.

### Outputs
- Script, unit/smoke tests, documented CLI.
- Per-trial CSV, aggregate JSON, machine-readable config, charts for MAE vs probe count and time vs probe count.
- Methodology note explaining which quantities are attacker-observable vs evaluator-only.
- A concise RESULTS.md with actual measured findings and uncertainty across seeds.
- Verify runtime and tests on CPU, commit to `experiment/knc-0`, and push.
- Stop after Phase 2.5; request review before Phase 3 adaptive optimization.

## Acceptance criteria
- 10 independent trials, correct-key exactness confirmed for each.
- No synthetic secret regeneration possible from public metadata.
- No ground-truth leakage into attack candidate selection.
- Clear comparison of estimator vs trivial baseline, with uncertainty.
- Existing Phase 1–2 tests remain green.
