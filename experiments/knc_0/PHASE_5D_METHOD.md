# Phase 5D methodology — Text readability across training variants

Written and committed before any Phase 5D model was trained. The phase was
requested by the owner in conversation after exploratory runs (see "Prior
evidence"); there is no separate task file.

## Question
Phases 5A–5C scored random 12-bit tokens. ReMemory stores text: UTF-8 → base64 →
pairs of base64 characters → 12-bit tokens. A reconstruction that misses most tokens
by one or two units fails as a token array but may still be readable as text. For
four training variants and two languages, how much of a stored text can an analyst
read from public weights alone, and from public weights plus k known tokens?

## Prior evidence (exploratory, not saved, not part of this phase)
- S=10000, 15 paired secrets: Adam lr=0.01 put the key's first layer fully inactive
  in 9/15 models; Adam lr=0.001 and SGD never did; AdamW (wd 0.01) memorized only 3/15.
- English text, 3 models each, k=64: S=4095 / Adam 0.01 gave 62–76% of bytes correct
  with naive decoding; S=10000 / Adam 0.001 gave 35–39%; the known tokens alone give
  about 19%.
- Russian text, 3 models, S=10000 / Adam 0.001: unreadable, but measured with a
  position-wise character metric that undercounts. Bytes were not measured.

## Variants (output scale, optimizer)
| Id | S | Optimizer |
|---|---|---|
| V1 | 4095 | Adam lr=0.01 (Phases 5A–5C) |
| V2 | 10000 | Adam lr=0.01 |
| V3 | 10000 | Adam lr=0.001 |
| V4 | 10000 | SGD lr=0.1, momentum 0.9 |

Architecture `384 -> 320 -> 320 -> 256`, ReLU, MSE on `tokens/S`. Stopping rule:
token-space MSE ≤ 1e-5 on the saved model, cap 5,000 steps. Failures are recorded and
the models are excluded from attack tables, with their count stated. AdamW is not run
(it did not memorize in the exploratory run).

## Texts
Two languages, English and Russian. For each model a text is built from a fixed pool
of 12 neutral sentences per language written for this phase (no personal data): the
sentences are concatenated in a CSPRNG-chosen order (repeating the pool if needed)
and the result is encoded with the ReMemory codec (`memory/codec/base64_codec.py`,
re-implemented identically) and cut to exactly 256 tokens = 384 bytes. Each model gets
its own secret input (unit vector, OS CSPRNG) and its own text order. The four
variants of a trial share the same secret input, text and initialization seed (paired).

20 trials per language → 20 models per variant × language, 160 models in total.

## Analyst (public weights + S, and known positions/values where stated)
- k=0, `output_mean`: rounded mean output over 1,024 random unit inputs.
- k=0, `dead_l1`: `S·(W3·relu(b2) + b3)` (Phase 5C).
- k ∈ {16, 64}, `input_fit`: Phase 5B frozen setting (8 restarts, Adam lr=0.05,
  unit-sphere projection, 2,000 steps, objective on known positions only, restart
  chosen by known-position loss). Known tokens are placed into the candidate.
- Positions: uniform without replacement from a public RNG seeded by
  (phase seed, language, trial, k); one draw per k.
- Control: `input_fit` at k=64 with random values at the true positions.

No text-aware decoder is used: bytes come from naive base64 decoding of the rounded
tokens. The exploratory format-aware beam search added only 2–5 points and is too slow.

## Metrics (evaluator only, after the candidate is frozen)
- Token level, unknown positions: MAE, exact fraction.
- **Byte accuracy:** fraction of the 384 bytes equal to the original.
- **Known-only baseline:** byte accuracy when the known tokens are correct and every
  other token is uniformly random (mean of 20 draws).
- **Word recovery:** fraction of space-separated words of the original whose bytes are
  all correct at their original positions (readability proxy).
- **Mostly readable** (fixed now): word recovery ≥ 0.5. Reported as a rate with a
  Wilson 95% interval.
Means over models are given with percentile bootstrap 95% intervals (2,000 resamples).
The model is the unit; draws are not repeated.

## Limitations stated in advance
One architecture, one attack configuration, naive decoding only. Results are lower
bounds on what an analyst with a language model could read. Unsuccessful attacks are
not evidence of security.

Run:

```bash
python experiments/knc_0/phase_5d.py --out /tmp/knc5d
pytest -q -p no:cacheprovider tests/test_knc_0_phase_5d.py
```
