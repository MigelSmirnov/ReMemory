# KNC-0 Phase 2.5 results

## Setup

Ten independently trained `256 -> 64 -> 64 -> 64` ReLU models were evaluated
on CPU from commit `62ee2222e566e24e3ea97ea4ef2c5b61bed2f6ca` (also recorded in
`config.json`). Secret inputs and targets came from unrecorded OS CSPRNG entropy.
Training seeds and attacker probe seeds were independent of those secrets.

Each trained model reached exact 64-token recovery under its correct input.
Attack candidates were frozen using public weights and configuration before the
evaluator compared them with hidden targets.

## Findings

The random-output mean contained measurable target information, but did not
recover a target or key. At 100,000 probes its mean token MAE was 460.85
(95% normal CI 404.44–517.27), compared with 1,029.02 (989.03–1,069.01) for the
constant midpoint predictor and 1,962.57 (1,893.68–2,031.46) for an untrained
network's random-output mean. Full recovery was 0/10 and mean exact-token fraction
was 0.00156 (approximately 0.1 of 64 tokens per trial).

Convergence largely plateaued early: mean MAE was 441.85 at 100 probes, 454.26
at 1,000, 461.56 at 10,000, and 460.85 at 100,000. The confidence intervals
overlap, so this experiment does not support a claim that more probes beyond the
small sample materially improve the estimator. Mean runtime grew from 0.0025 s
at 100 probes to 2.223 s at 100,000 probes.

The parameter-only Gaussian moment estimator reached MAE 461.88
(406.97–516.78), essentially the same range as the sampled output mean without
performing probe search. Last-layer bias alone was much worse at MAE 1,793.51.
This indicates that the leakage measured here is a public-weight-derived estimate
of the learned output distribution, not evidence of secret-input recovery.

The independently trained shuffled-target control reached random-mean MAE 430.02
(358.76–501.29), overlapping the primary result. The effect therefore appears
for arbitrary token associations and should not be interpreted as semantic
reconstruction.

For hidden activations, the correct input activated 10.47% of first-layer and
23.28% of second-layer units on average. Random inputs activated 49.24% and
33.70%, respectively. Mean binary pattern agreement was 50.92% in layer 1 and
76.75% in layer 2. The high layer-2 agreement is partly driven by shared inactive
units and is evaluator-only; it is not an attacker-available objective in this
phase.

## Interpretation

Phase 2.5 confirms output-distribution leakage from trained public weights. It
does not demonstrate full data extraction, key recovery, cryptographic security,
or resistance to adaptive optimization. No candidate used hidden truth for
selection. Phase 3 should begin only after review of this methodology and result.

Machine-readable details are in `aggregate.json`, `per_trial.csv`, and each
`trial_*/report.json`. Public attacker bundles are the corresponding
`trial_*/public/` directories. The two SVG charts show MAE and cumulative runtime
against probe count.
