# Phase 5A results — Target Scaling

## Setup

Run on CPU from commit `44e342a4fabfca0856e29c7d344dad127342602c` (recorded in
`config.json`): 3 pilot and 30 held-out trials. ReMemory architecture
`384 -> 320 -> 320 -> 256`, Adam lr=0.01, MSE. In every trial A (raw tokens),
B (tokens/4095) and C (tokens/10000) shared one OS-CSPRNG secret and one
initialization. Python 3.13.18, PyTorch 2.15.0, NumPy 2.5.4, 4 CPU threads.

The attack and its acceptance rule (`max frac < 0.25`, all coordinates in 0–4095)
were frozen in the specification commit `54de5b2` before any run. Pilot results
did not change them. Intervals are Wilson 95%.

## Primary answer: same stopping rule

| Rule | Model | Correct key 100% | Blind full recovery | Accepted | False accepts |
|---|---|---|---|---|---|
| R1 exact rounding | A | 30/30 [0.89, 1.00] | **0/30** [0.00, 0.11] | 30/30 | **30/30** |
| R1 exact rounding | B | 30/30 [0.89, 1.00] | 0/30 [0.00, 0.11] | 0/30 | 0/30 |
| R1 exact rounding | C | 30/30 [0.89, 1.00] | 0/30 [0.00, 0.11] | 0/30 | 0/30 |
| R2 token MSE ≤ 1e-5 | A | 30/30 [0.89, 1.00] | **30/30** [0.89, 1.00] | 30/30 | 0/30 |
| R2 token MSE ≤ 1e-5 | B | 30/30 [0.89, 1.00] | 0/30 [0.00, 0.11] | 0/30 | 0/30 |
| R2 token MSE ≤ 1e-5 | C | 30/30 [0.89, 1.00] | 0/30 [0.00, 0.11] | 0/30 | 0/30 |

Under R2 (ReMemory precision expressed in tokens), A was recovered in every
trial and B and C in none, with matched secrets and initializations. This
confirms that target scale affects this vulnerability. It does not show that
B or C is secure: only this one attack was tested.

Under R1 the scale contrast disappears for this attack: A is not recovered either.

## Output geometry

Medians (min–max) over 30 held-out models, in token units:

| Model | RMS PC1 | RMS PC2 | RMS PC3 | Off-plane RMS | Target distance from plane, max |
|---|---|---|---|---|---|
| A | 1,958 (1,803–2,121) | 0.39 (0.28–0.64) | 0.013 (0.012–0.018) | 0.058 (0.051–0.063) | R2: 0.024 (0.015–0.052); R1: 0.32 (0.16–0.51) |
| B | 20,280 (14,680–29,460) | 77 (43–139) | 48 (39–87) | 186 (155–212) | 110 (76–167) |
| C | 27,510 (11,750–52,820) | 972 (564–4,840) | 263 (114–639) | 495 (385–770) | 84 (47–183) |

For A the output image is a two-dimensional plane to within ~0.06 tokens. For B
and C the off-plane spread is hundreds of tokens and the target lies 50–180
tokens away from the attacker's plane, so a two-component lattice search cannot
reach it. The principal-component statistics of each model are identical under
R1 and R2 (to the reported precision): the plane is set early in training, and
the extra R2 steps only move the target point onto the integer lattice.

## Stopping-rule study

**R1 versus R2 for A.** R1 stops at 190 steps (180–200), as soon as every token
rounds correctly; the remaining error is up to ±0.5 per token (token MSE 0.037).
The target is still near the plane (max distance 0.16–0.51), but it is not near an
integer point. The lattice search then prefers an all-zero array near the
plane's low-output end (max fractional error 0.13–0.18). The pre-registered rule
accepted it in all 30 trials. R2 adds ~76 steps (266, 263–268) and reduces the
token MSE to ~1e-5; the target becomes the best lattice point (max fractional
error 0.022–0.051) and is recovered every time.

**R3, the literal ReMemory rule in own units.** For B and C, MSE ≤ 1e-5 in
normalized units is far too coarse: token MSE 136 (B) and 826 (C), and the correct
key reconstructed 0/30 models in both. R3 therefore does not produce a working
memory for B or C; their attack results under R3 carry no information. For A,
R3 equals R2 and was not run separately.

## Acceptance rule failure

The pre-registered acceptance rule produced 30/30 false accepts for A under R1
and accepted nothing wrong elsewhere. The candidate it accepted was the trivial
all-zero array (zero fraction 1.0 in every case). A target-free attacker can see
that a candidate is trivial, but excluding it or searching beyond the best
lattice point was not pre-registered and is untested here. Since the target lies
within 0.5 tokens of the plane under R1, a better attack may still recover A/R1.
This result is a failure of the tested attack, not evidence that R1 protects A.

## Interpretation

- Target scale matters: with matched secrets, initializations and a
  token-equivalent stopping rule (R2), raw-scale targets leak to the blind
  attack (30/30) and normalized targets do not (0/30 for B and C).
- The stopping rule matters too: whether the leaking geometry becomes an exact
  recovery depends on training past the first exact rounding.
- Not shown: security of B or C against other attacks; resistance of A/R1 to a
  better candidate rule; behaviour at other widths, input dimensions or
  output lengths.

Machine-readable outputs: `config.json`, `aggregate.json` (rates with Wilson
intervals, continuous metrics as mean/min/median/max) and `per_trial.csv`
(264 rows: 24 pilot and 240 held-out). No secret inputs or targets are stored.
No plots were produced in this phase. Review is required before the next phase.
