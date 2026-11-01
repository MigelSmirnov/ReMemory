# Phase 4C — ReLU Activation Pattern Analysis

## Objective
Explain the high off-key full-array recovery observed in Phase 4B for narrow ReLU MLPs, especially 256 -> 1 -> 1 -> 64. Determine which activation patterns produce the memorized output on broad input regions. This is a controlled analysis of neural-memory behavior, not a cryptographic-security claim.

## Starting evidence
Phase 4B: with h=1, m=64, zero-input recovery was 24/30 full arrays; h=2 gave 17/30, h=4 gave 7/30, and h=8 gave 0/30. All models memorized targets exactly at the correct input. Results are in experiments/knc_0/results/phase_4b_cpu_commit_e3a9846/.

## Study design
Keep d=256, m=64, ReLU, Adam and MSE. Analyze h=1 first, then h=2,4,8. Use 30 independent fresh trials per width initially; reuse prior public models only where available, otherwise train new models under the same protocol. Keep evaluator secrets separate from public model inspection.

### Part A: Exact piecewise-linear structure (h=1)
For a two-hidden-layer width-1 network, enumerate the four formal binary ReLU activation combinations. Identify which are feasible and derive each region's affine output mapping from public weights and biases. Distinguish regions with exactly constant outputs from those varying with input. Measure output stability within feasible regions.

### Part B: Memorized-input pattern
In a strictly evaluator-only analysis, record the activation pattern at the correct secret input. Compare it with patterns at zero input and independent random inputs. Measure how often the same pattern yields exactly the full memorized 64-token vector after rounding. Never use the secret pattern to select an attacker candidate.

### Part C: Width comparison
For h=2,4,8, measure the number of distinct observed activation patterns, their frequency under a declared input distribution, and the frequency of output arrays matching the memorized target. Avoid claiming exhaustive enumeration for larger widths. Compare output variance and fixed-input recovery across widths.

### Part D: Mechanistic controls
Analyze the roles of output bias, inactive ReLU units, and piecewise-constant regions. Include untrained and linear-network controls. Verify whether exact recovery results from a constant region, a shared activation pattern, or a rounding coincidence. Report these explanations separately.

## Metrics
- Feasible activation patterns and their empirical frequencies.
- Exact full-array recovery and exact-token fraction for zero and random inputs.
- Pattern agreement with the correct input (evaluator-only).
- Output variance within regions and across inputs.
- Percentage of models with constant-output regions.
- Runtime, model parameter count, and confidence intervals.

## Experimental integrity
Generate secret inputs and targets from OS entropy independent of public seeds. Attacker-side computations use only public weights, architecture and nonsecret configuration. Evaluator-only functions may use the secret input/target after candidates are frozen. Do not infer key recovery from output recovery.

## Deliverables
Create a standalone Phase 4C runner, unit tests, methodology, RESULTS.md, aggregate.json, per_trial.csv, and clear plots or tables of activation-pattern frequencies and recovery rates. Keep original ReMemory and previous experiment artifacts unchanged. Run on CPU, do not use RunPod. Commit and push to experiment/knc-0; stop for review before Phase 4D.
