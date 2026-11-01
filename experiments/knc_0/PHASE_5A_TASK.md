# Phase 5A — Target Scaling

## Research question
Does the scale of the training target decide whether a single-association ReLU
MLP trained with the ReMemory recipe leaks its memorized output to a blind
PCA/lattice attack? Three models differ only in target scale:

| Model | Training target |
|---|---|
| A | raw tokens 0–4095, as in ReMemory |
| B | tokens / 4095 |
| C | tokens / 10000 |

For each model, check two things:

1. Does the correct key reconstruct the data 100%?
2. Can the PCA attack extract the data without the key?

If A is broken and B and C are not, that confirms that target scale affects this
vulnerability. It does **not** prove that B and C are secure.

## Scope
- ReMemory architecture only: `d=384 -> h -> h -> m`, ReLU, `h=max(128,(d+m)//2)`,
  Adam lr=0.01, MSE. Output length `m=256` random 12-bit tokens.
- No new architectures, no width 1024, no SAT solvers, no GPU.
- Identical stopping rule for A, B and C in the primary experiment; the effect of
  stopping rules is studied separately.
- Keep existing ReMemory code and previous phases unchanged.

## Primary experiment
Stopping rule R1 for all models: check every 10 steps and stop at the first check
where every rounded output token equals the target; cap 5,000 steps. The rule is
defined in token units, so it means the same thing for every scale.

## Stopping-rule study (separate)
- R2: token-space MSE `mean((out*scale - tokens)^2) <= 1e-5`, ReMemory's precision
  expressed in token units for every model; cap 5,000 steps.
- R3: MSE `<= 1e-5` in each model's own training units, the literal ReMemory rule.
  For A it coincides with R2; for B and C it is far looser in token units.
Failures to reach a rule within the cap are reported, not excluded.

## Attack (unchanged from the draft PCA attack)
Attacker receives public weights, architecture and the public output scale. It
samples 1,024 random unit inputs, takes the mean and the first two principal
components of the outputs in token units, enumerates integer values of two
well-conditioned coordinates (4096^2), solves for the plane coordinates and keeps
the candidate whose 24 check coordinates are closest to integers inside 0–4095.

Acceptance rule, pre-registered: accept iff every coordinate of the candidate is
within 0.25 of an integer and inside 0–4095. The threshold comes from the earlier
draft (`PCA_ATTACK.md`, chosen after viewing its results); this phase is its first
held-out test.

## Safeguards
- Fresh OS-CSPRNG secret input and target per trial; never persisted.
- Paired design: within a trial, A, B and C share the secret and the initial weights.
- Attacker API accepts no target or secret; candidate frozen before evaluation.
- 3 pilot trials disjoint from 30 held-out trials.
- Wilson 95% intervals for every rate.
