# Phase 4A.1 results

The CPU fallback allowed by the task was used: 30 fresh CSPRNG examples for each
of nine MLP configurations and three linear controls, 360 runs total, from commit
`b100156`. All models converged on the memorized input. Approximate constancy was
predefined as rounded-token standard deviation <=1 and a dominant-token fraction
>=99% over 512 public random inputs.

Exact off-key recovery is real but concentrated in narrow ReLU MLPs. For `h=1`,
zero input recovered the exact token in 22/30 runs at `d=1`, 19/30 at `d=4`, and
24/30 at `d=256`. The corresponding approximately-constant counts were 14/30,
9/30 and 12/30. Thus constant behavior explains a substantial subset, but not all,
of the exact recovery: some nonconstant piecewise-linear models also map zero or
large input regions to the memorized rounded token.

Recovery weakened with width. Zero-input exact counts for `h=2` were 16/30,
17/30 and 8/30; for `h=4` they were 6/30, 3/30 and 9/30. Random-mean exact recovery
tracked strict constancy more closely: at `h=1` it was 14/30, 9/30 and 13/30,
then mostly disappeared as width increased. Median and a single random input often
recovered more tokens than the mean because averaging raw outputs can move a
piecewise distribution away from its dominant rounded value.

All three linear controls were nonconstant and produced 0/30 exact recoveries for
zero, mean, median and single-random candidates. Their mean rounded-output standard
deviations were 1,082–1,314 tokens, versus 83–248 for `h=1` MLPs. This isolates the
effect to trained ReLU gating rather than single-token memorization alone.

These results concern recovery of one 12-bit output token, not recovery of the
secret input. `aggregate.json` reports numerators, denominators and Wilson 95%
intervals separately for exact, <=1, <=8 and <=64 errors, plus constant/nonconstant
strata, activation rates, bias/last-layer candidates, ranges and perturbation
sensitivity. `per_run.csv` contains all 360 non-secret trial records. Review is
required before Phase 4B.
