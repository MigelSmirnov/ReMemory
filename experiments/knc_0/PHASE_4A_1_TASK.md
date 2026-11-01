# Phase 4A.1 — Exact Token Recovery and Constant-Function Diagnostics

## Goal
Clarify whether the single-token results in Phase 4A represent exact recovery without the memorized input or merely near-misses. Determine whether very small ReLU networks behave approximately as constant functions after training.

## Baseline
Use Phase 4A results and its runner. Keep the original ReMemory code and prior results unchanged. Work on experiment/knc-0, CPU only.

## Configurations
Focus first on input dimensions d=1,4,256 and hidden widths h=1,2,4, with one 12-bit output token. Use 100 independent fresh examples per configuration if CPU time permits; otherwise 30 with uncertainty intervals. Include linear controls. Maintain the original training protocol and report all convergence failures.

## Diagnostics
1. Report exact equality after rounding separately from error <=1, <=8, <=64. Show numerators, denominators and Wilson 95% intervals.
2. Evaluate fixed zero input, random-input mean, random-input median and a single random input. Freeze each candidate before scoring against the hidden target. Compare with midpoint and uniform random-token controls.
3. For each trained model, sample many independent inputs and report output standard deviation, range, fraction of samples producing the same rounded token, and sensitivity to input perturbations. Identify approximately constant-output models using a predefined threshold; never choose the threshold based on hidden target.
4. Compare exact-recovery rates between approximately constant and nonconstant models.
5. Check whether output biases alone or the last linear layer account for apparent constant behavior; report ReLU activation rates.
6. Do not confuse one-token output recovery with secret-input recovery. A single token contains only 12 bits.

## Reproducibility
Keep secret generation independent of public seeds. Attacker-side diagnostics consume only public model/config; evaluator uses hidden truth only after candidate generation. Run tests for these boundaries. Store aggregated statistics and per-trial metrics, not secret values.

## Deliverables
Add an independent diagnostic runner, tests, RESULTS.md, aggregate.json, per_run.csv, and plots of exact recovery vs width and output variance vs width. Commit and push to experiment/knc-0. Stop for review. Do not rent GPU or proceed to Phase 4B yet.
