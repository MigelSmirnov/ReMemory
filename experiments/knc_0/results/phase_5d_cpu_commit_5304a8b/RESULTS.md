# Phase 5D results — Text readability across training variants

## Setup

Run on CPU (4 threads) from commit `5304a8be5559c0404f23ff797cc7c6cfd82952ed`;
methodology frozen in `0e217cc` (`PHASE_5D_METHOD.md`). Python 3.13.18, PyTorch
2.15.0; runtime 742 s; peak RSS 677 MiB.

- Variants: V1 S=4095 / Adam lr=0.01; V2 S=10000 / Adam lr=0.01; V3 S=10000 / Adam
  lr=0.001; V4 S=10000 / SGD lr=0.1, momentum 0.9. Architecture `384 -> 320 -> 320 ->
  256`, token-space MSE ≤ 1e-5, cap 5,000 steps.
- English and Russian texts: per model, 12 neutral sentences in a CSPRNG order, encoded
  with the ReMemory codec and cut to 256 tokens = 384 bytes. 20 trials per language;
  the four variants of a trial share the secret input, text and initialization.
- Attacks frozen from Phases 5A–5C; decoding is naive (no language or format model).

## Training

All 160 models met the stopping rule and reproduced their text exactly from the key
(20/20 in every variant × language). Key first layer fully inactive:

| | V1 S4095 Adam 0.01 | V2 S10000 Adam 0.01 | V3 S10000 Adam 0.001 | V4 S10000 SGD |
|---|---|---|---|---|
| English | 1/20 | 16/20 | 0/20 | 0/20 |
| Russian | 0/20 | 14/20 | 0/20 | 0/20 |

Median training steps: 212–251 for Adam, 300–435 for SGD.

## Results

Means over 20 models, percentile bootstrap 95% intervals in brackets. "Bytes" is the
fraction of the 384 bytes correct; "known only" is the byte accuracy given by the
known tokens alone (the rest random); "words" is the fraction of words recovered
without a single wrong byte; "readable" is the number of models with word recovery ≥ 0.5
(Wilson 95%).

### English

| Variant | Attack | Bytes | Known only | Words | Readable |
|---|---|---|---|---|---|
| V1 S4095 Adam 0.01 | blind `dead_l1` | 0.06 [0.01, 0.16] | 0 | 0.05 | 1/20 [0.01, 0.24] |
| | k=16 | 0.41 [0.37, 0.45] | 0.05 | 0.05 | 0/20 [0.00, 0.16] |
| | k=64 | **0.75** [0.69, 0.81] | 0.20 | **0.38** [0.27, 0.50] | **6/20** [0.15, 0.52] |
| V2 S10000 Adam 0.01 | blind `dead_l1` | **0.82** [0.63, 0.95] | 0 | **0.80** | **16/20** [0.58, 0.92] |
| | k=16 | 0.76 [0.62, 0.88] | 0.05 | 0.62 | 12/20 [0.39, 0.78] |
| | k=64 | **0.99** [0.97, 1.00] | 0.20 | **0.97** | **19/20** [0.76, 0.99] |
| V3 S10000 Adam 0.001 | blind | ≤ 0.01 | 0 | 0.00 | 0/20 |
| | k=16 | 0.14 [0.13, 0.15] | 0.05 | 0.01 | 0/20 |
| | k=64 | 0.37 [0.35, 0.38] | 0.20 | 0.06 | 0/20 [0.00, 0.16] |
| V4 S10000 SGD | blind | ≤ 0.01 | 0 | 0.00 | 0/20 |
| | k=16 | 0.11 [0.10, 0.12] | 0.05 | 0.01 | 0/20 |
| | k=64 | 0.32 [0.31, 0.33] | 0.20 | 0.05 | 0/20 [0.00, 0.16] |

### Russian

| Variant | Attack | Bytes | Known only | Words | Readable |
|---|---|---|---|---|---|
| V1 S4095 Adam 0.01 | blind | ≤ 0.02 | 0 | 0.00 | 0/20 |
| | k=16 | 0.29 [0.27, 0.31] | 0.05 | 0.01 | 0/20 |
| | k=64 | 0.52 [0.50, 0.53] | 0.20 | 0.04 | 0/20 [0.00, 0.16] |
| V2 S10000 Adam 0.01 | blind `dead_l1` | **0.71** [0.52, 0.90] | 0 | **0.70** | **14/20** [0.48, 0.85] |
| | k=16 | 0.73 [0.59, 0.88] | 0.05 | 0.61 | 12/20 [0.39, 0.78] |
| | k=64 | **0.95** [0.89, 1.00] | 0.20 | **0.86** | **17/20** [0.64, 0.95] |
| V3 S10000 Adam 0.001 | blind | ≤ 0.01 | 0 | 0.00 | 0/20 |
| | k=16 | 0.14 [0.13, 0.14] | 0.05 | 0.00 | 0/20 |
| | k=64 | 0.36 [0.35, 0.37] | 0.20 | 0.02 | 0/20 [0.00, 0.16] |
| V4 S10000 SGD | blind | ≤ 0.02 | 0 | 0.00 | 0/20 |
| | k=16 | 0.11 [0.10, 0.12] | 0.05 | 0.00 | 0/20 |
| | k=64 | 0.33 [0.32, 0.34] | 0.20 | 0.01 | 0/20 [0.00, 0.16] |

"Blind" rows for V1 Russian, V3 and V4 give the better of `output_mean` and `dead_l1`;
every value is in `aggregate.json`.

**Wrong-values control** (k=64, true positions, random values): byte accuracy
0.007–0.020 and 0/20 readable in every cell. It is below the known-only baseline
because the inserted known tokens are wrong as well. Recovery therefore depends on the
true known values.

Median unknown-token error at k=64: V1 0.7 (English) / 2.7 (Russian); V2 0.0 / 0.0;
V3 12.8 / 14.5; V4 22.8 / 21.3.

## Interpretation

- **V2 (S=10000, Adam 0.01) is unsafe without any knowledge.** The weight-only
  `dead_l1` readout makes 16/20 English and 14/20 Russian texts readable — exactly
  the models whose key has a fully inactive first layer. With 64 known tokens, 19/20
  and 17/20.
- **V1 (S=4095, Adam 0.01) depends on the language.** With 64 known tokens
  English is 75% byte-correct and readable by the fixed criterion in 6/20 models
  (word recovery 0.38); Russian reaches 52% of bytes but only 4% of words, 0/20
  readable. A Cyrillic letter takes two bytes and the small token errors fall into the
  second byte, so almost every word gets at least one wrong byte. The word criterion
  is strict: English V1 text with 75% correct bytes is easy for a person to read with
  typos (see the exploratory examples before this phase), even when word recovery is below 0.5.
- **V1 can also have an inactive first layer.** One English V1 model (1/20) had a
  fully inactive first layer at the key, and its text was read blind in full. Phase 5C
  saw 0/30 for S=4095; the effect is rare but not absent.
- **V3 and V4 (S=10000, Adam 0.001 or SGD) resisted every tested attack.** Blind
  attacks give ≤ 2% of bytes; with 64 known tokens the attack adds 12–17 points of bytes
  over the known-only 20%, but word recovery stays ≤ 6% and no model is readable. The
  first layer was never inactive at the key.
- These are naive-decoding lower bounds. A decoder with a language model may read more
  from V1 Russian, V3 and V4, where most bytes are close but not exact; this was not
  tested.

## Changes relative to Phases 5A–5C

- Phase 5B's "0 full recoveries" for S=4095 understated text leakage: with 64 known
  tokens English texts are mostly readable with typos.
- Phase 5C's dead-first-layer readout applies to text: it reads 70–80% of V2 texts
  blind, and occasionally a V1 text.
- Training choice matters more than target scale alone: at the same S=10000, Adam with
  lr=0.01 leaks fully, while Adam with lr=0.001 and SGD did not leak readable text in
  these tests.

## Limitations

- 20 models per cell, one position draw per k, one attack configuration, naive decoding.
- Fixed sentence pool: texts differ in order, not vocabulary.
- The readability criterion counts only error-free words; it understates what a reader
  with typo tolerance can recover.
- Unsuccessful attacks are not evidence of security; V3 and V4 have not been tested
  against stronger searches or language-model decoding.

## Files

`config.json`, `aggregate.json` (training and every language × variant × attack
condition, with bootstrap and Wilson intervals) and `per_trial.csv` (800 rows). The
texts come from the fixed public pool; secret inputs and orders are not stored. Review
is required before any further phase.
