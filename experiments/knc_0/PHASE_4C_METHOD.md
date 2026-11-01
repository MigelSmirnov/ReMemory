# Phase 4C methodology

The runner trains 30 fresh OS-CSPRNG trials for each width 1, 2, 4 and 8 at
`d=256, m=64`, plus 30 linear controls. Public analysis samples 4,096 standard
normal inputs and records activation-pattern frequencies and within-pattern output
variation. Target and secret input enter only evaluator code after public analysis.

For width one, all four formal binary patterns are checked analytically for
feasibility. Their public affine maps are derived from weights and classified as
constant when the linear coefficient is exactly zero. Zero-input recovery is then
classified as constant-region, shared-pattern nonconstant, rounding coincidence,
or no recovery using predefined rules. Larger-width pattern counts are empirical,
not exhaustive.
