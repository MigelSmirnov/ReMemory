# Phase 4A: Single-Token Scaling Experiment

## Goal
Measure how accurately a tiny ReLU MLP memorizes one random 12-bit token and how model width, input dimension, and parameter count affect numerical output behavior. This is an exploratory scaling study, not a claim of security.

## Experiment design
- Model: d -> h -> h -> 1 with ReLU activations, Adam optimizer and MSE.
- Input dimensions d: 1, 4, 16, 256.
- Hidden widths h: 1, 2, 4, 8, 16, 64.
- Target: one integer from 0 to 4095, scaled to [0,1] for training.
- 30 independent runs per configuration: 720 MLP runs.
- Add a linear d -> 1 baseline.
- Run a small CPU smoke test before the complete matrix.

## Measurements
Record per run: input dimension, width, parameter count, exact memorization after rounding, training loss, training steps, elapsed time, model size, and failure to converge. Retain failed runs in the summary.

For analysis of model outputs, compare fixed zero inputs, random-input averages, and the output distribution of independently initialized models. Report absolute token error and proportions within 1, 8, and 64 token units. Keep reference targets separate from candidate generation and score candidates only after they are fixed.

## Scaling analysis
- Plot exact memorization rate vs hidden width, grouped by input dimension.
- Plot numerical output error vs width and total parameter count.
- Plot runtime and serialized model size vs parameter count.
- Include uncertainty intervals and the number of trials per configuration.
- Test for a transition in behavior without assuming a power law.
- Distinguish effects of width from effects of input dimension.

For two hidden layers and one output, parameter count is:
P = (d+1)*h + (h+1)*h + (h+1).

## Interpretation
A single token contains only 12 bits. The experiment is intended to characterize learning and scaling, not practical secrecy. Avoid interpreting failure of a particular estimator as proof of protection.

## Deliverables
Implement a standalone Phase 4A runner, tests, configuration, CSV/JSON reports, plots, and RESULTS.md under experiments/knc_0/. Preserve the original ReMemory pipeline and prior experiments. Run on CPU first; do not use RunPod. Commit to experiment/knc-0 and stop for review before extending the output to multiple tokens.
