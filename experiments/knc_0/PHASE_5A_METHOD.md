# Phase 5A methodology

Settings are frozen before held-out evaluation: `d=384`, `m=256`, `h=320`,
Adam lr=0.01, scales A=1 (raw tokens), B=4095, C=10000 (the model learns
`tokens/scale`; outputs are multiplied back by `scale` before rounding). Each trial
draws one secret `(x, tokens)` from OS entropy and one initialization seed; A, B
and C of the trial reuse both, so differences within a trial come only from the
target scale (and, in the stopping-rule study, from the rule).

Stopping rules: R1 exact rounded tokens (checked every 10 steps), R2 token-space
MSE <= 1e-5, R3 own-unit MSE <= 1e-5; cap 5,000 steps for all. The primary result
uses R1. R2 and R3 form the separate stopping-rule study on the same secrets.

Attacker (`attack`): 1,024 standard-normal inputs normalized to the unit sphere
(probe seed independent of secrets), SVD of centred outputs in token units,
two-coordinate integer enumeration, pre-registered acceptance `max frac < 0.25`.
Attacker-side diagnostics: per-component RMS, off-plane RMS, best and runner-up
check scores. Evaluator (`evaluate`): correct-key reconstruction, distance of the
target from the attacker's plane, exact token fraction and full recovery of the
frozen candidate.

Reported per configuration: memorization rate, steps, full recovery, acceptance,
false acceptances (accepted but wrong) and missed recoveries (correct but
rejected), all with Wilson 95% intervals.

Run:

```bash
python experiments/knc_0/phase_5a.py --out /tmp/knc5a
pytest -q -p no:cacheprovider tests/test_knc_0_phase_5a.py
```
