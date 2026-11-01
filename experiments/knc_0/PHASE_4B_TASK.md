# Phase 4B — Multi-Token Capacity and Output Leakage

## Goal
Test whether the exact off-key token recovery observed in Phase 4A.1 persists when a single ReLU MLP memorizes multiple 12-bit tokens. Study scaling of memorization, output sensitivity, and numerical leakage; do not interpret these results as cryptographic security.

## Model and matrix
Use d=256, architecture 256 -> h -> h -> m, ReLU, Adam, MSE.
Hidden widths h = 1, 2, 4, 8.
Output token counts m = 1, 2, 4, 8, 16, 64.
Run 30 independent trials per configuration (720 total), starting with a small CPU smoke test. Use fresh independent CSPRNG-generated inputs and targets; public seeds must not reconstruct secrets. Keep the original ReMemory implementation and earlier experiments unchanged.

## Training and capacity
Keep a consistent protocol across configurations. Record whether all tokens are exactly recovered with the correct input, training steps, final loss, elapsed time, parameter count, serialized model size, and failures. Do not omit failed runs. Compare leakage among successfully memorized models separately.

## Off-key behavior
For each trained public model, evaluate fixed zero input, single random input, mean and median of random outputs, last-layer bias, and a trivial midpoint control. Candidate generation must not use hidden target data. Freeze candidates before evaluator scoring.

Report:
- exact-token fraction and exact-token counts;
- full-array exact recovery rate;
- token MAE and within-1/8/64 fractions;
- output variance and dominant-token fractions across random inputs;
- layerwise activation rates and near-constant-output frequency;
- time, RAM and model size.

Define approximately constant behavior before evaluation, without reference to the hidden target. Distinguish exact token recovery from recovery of the secret input.

## Analysis
Plot token exact fraction and whole-array recovery against output length for each width. Plot memorization success and parameter count against output length. Report Wilson intervals for recovery proportions and uncertainty for numerical errors. Compare against Phase 4A.1 m=1 baseline, while clearly noting independent datasets/trials. Look for trends, not a presumed universal scaling law.

A single token has 12 bits; increasing output length does not imply independent token errors or exponential security. Report empirical full-array recovery directly rather than multiplying per-token probabilities.

## Deliverables
Create a standalone Phase 4B runner, tests, methodology, RESULTS.md, aggregate.json, per_run.csv, and plots under experiments/knc_0/. First run on CPU; no RunPod without approval. Commit and push to experiment/knc-0, stop for review before further architecture changes.
