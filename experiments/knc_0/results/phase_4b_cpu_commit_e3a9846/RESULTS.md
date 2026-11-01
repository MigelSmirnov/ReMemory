# Phase 4B results

The full 24-configuration matrix was run from commit
`e3a984683f76d9ba3c0338ad1b46e08b2271d16b`: 30 fresh CSPRNG trials per
configuration, 720 CPU runs. All models exactly memorized every target token with
the correct input; there were no convergence failures.

## Off-key recovery

Exact off-key recovery persisted for multiple tokens, but was strongly dependent
on width. With `h=1`, zero-input full-array recovery was 22/30, 26/30, 22/30,
17/30, 25/30 and 24/30 for output lengths 1, 2, 4, 8, 16 and 64. At `m=64`, mean
exact-token fraction and empirical full-array recovery rate were both 0.80. This
cannot be interpreted as independent per-token errors: failures and successes are
strongly correlated across the output array.

For `h=1`, random-output-mean full recovery across the same lengths was 13/30,
7/30, 14/30, 6/30, 8/30 and 9/30. These counts closely tracked the predefined
near-constant-array counts: 13, 7, 14, 6, 8 and 9. Thus strict constant behavior
explains the random-mean full recovery in this narrow regime. Zero input recovers
more arrays because nonconstant ReLU regions can still map that fixed point to the
memorized rounded array.

The effect weakened with width. At `h=2, m=64`, zero input recovered 17/30 full
arrays while random mean recovered 0/30. At `h=4, m=64`, zero recovered 7/30 and
random mean 0/30. At `h=8`, zero full-array recovery ranged from 0 to 2/30 and
random mean recovered 0/30 for every output length.

Zero-input MAE at `m=64` was 101.6 tokens for `h=1`, 123.0 for `h=2`, 261.3 for
`h=4`, and 620.8 for `h=8`. Increasing width therefore reduced exact off-key
leakage in this experiment even though correct-key memorization remained 30/30.
The trend is empirical and non-monotonic in some shorter-output configurations;
it is not a universal scaling law.

## Interpretation

The single-token Phase 4A.1 phenomenon does persist for multi-token outputs. For
very narrow networks it is an array-level regime: training often creates a
constant or large fixed ReLU region whose output rounds to the entire memorized
target. Output length alone did not cause exponential decay because token errors
were not independent.

This is recovery of output tokens under off-key inputs, not recovery of the
secret 256-dimensional input. It is also not a cryptographic-security result.
`aggregate.json` reports Wilson intervals, token fractions, empirical full-array
rates, MAE, within-1/8/64 accuracy, capacity, activation and constancy diagnostics.
`per_run.csv` retains all 720 non-secret records. Four SVG plots show recovery,
capacity and parameter scaling. Review is required before architecture changes.
