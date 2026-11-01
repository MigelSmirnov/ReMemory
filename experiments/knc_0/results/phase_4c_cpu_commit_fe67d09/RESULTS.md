# Phase 4C results

From commit `fe67d098a2e615ac012a6c928c400c490f865e34`, 30 fresh CSPRNG
trials were run at each width 1, 2, 4 and 8 for `d=256, m=64`, plus 30 linear
controls. All 120 MLPs exactly memorized their targets with the correct input.

## Exact width-one mechanism

Width-one networks exhibited 2.33 observed activation patterns on average out of
four formal combinations. Every trained model had at least one analytically
feasible constant-output ReLU region. Zero input recovered the full 64-token array
in 19/30 models (Wilson 95% interval 45.5%–78.1%). All 19 recoveries were explained
by the zero input lying in a public, exactly constant affine region whose rounded
intercept equaled the memorized target. There were zero shared-pattern
nonconstant explanations and zero rounding coincidences.

The secret pattern itself was not recovered: zero and secret patterns agreed in
15/30 models, while independent random inputs agreed with the secret pattern
44.3% of the time. Random inputs recovered the full target on 59.5% of probes on
average because broad constant regions—not secret-input proximity—dominated input
space.

## Width comparison

Observed pattern complexity rose sharply: 6.8 distinct patterns at `h=2`, 51.5 at
`h=4`, and 1,240 at `h=8` among 4,096 probes. Zero full recovery declined from
19/30 to 10/30, 7/30 and 1/30 respectively.

At `h=2`, the ten zero recoveries split into five shared-secret-pattern
nonconstant cases and five rounding coincidences. At `h=4`, three of seven shared
the secret pattern and four were coincidences. At `h=8`, the only recovery was a
rounding coincidence; zero never matched the secret pattern. Random/secret pattern
agreement fell from 44.3% at `h=1` to 1.08% at `h=8`.

Mean normalized output standard deviation increased from 0.057 at `h=1` to about
0.10 for `h=4–8`. The linear control had much greater variation (0.578) and zero
full recoveries in 30 trials. Untrained controls and per-pattern stability are
recorded in `per_trial.csv`.

## Interpretation

The high width-one leakage has a direct piecewise-linear explanation: ReLU gating
creates broad exactly constant regions whose output bias/intercept has been trained
to the memorized array. It is output recovery without key recovery. Wider networks
create many more regions; occasional recovery then comes from shared patterns or
rounding coincidences rather than the single dominant constant-region mechanism.

These findings characterize this architecture and training protocol, not
cryptographic security. `aggregate.json`, `per_trial.csv`, `linear_control.json`
and `recovery_by_width.svg` contain the machine-readable evidence. Review is
required before Phase 4D.
