# Exploratory scripts (not pre-registered)

Quick experiments run in conversation between and after the pre-registered phases.
None had a methodology committed beforehand, a pilot/held-out split, or a separate
evaluator process. Several use fixed public seeds instead of CSPRNG secrets. Their
numbers are indications only; the pre-registered results are under `../results/`.

Run from the repository root, e.g. `python experiments/knc_0/exploratory/v3_stress.py 10`.
All scripts are CPU-only.

| Script | What it does | Outcome (as observed in the run) |
|---|---|---|
| `optimizers.py N` | S=10000, four optimizers on the same N secrets: Adam 0.01, Adam 0.001, SGD (lr 0.1, momentum 0.9), AdamW (lr 0.01, wd 0.01). Attacks: `dead_l1` readout and `input_fit` with 64 known tokens. | N=15 (`logs/optimizers_15pairs.log`): Adam 0.01 — key first layer inactive 9/15, read from weights 9/15, known-64 full 13/15. Adam 0.001 and SGD — first layer never inactive, 0/15 for both attacks, known-64 MAE 15 and 23. AdamW memorized only 3/15. |
| `decode_test.py` | Russian text, S=10000, Adam 0.001, 3 models; `input_fit` with 64 known tokens, naive and format-aware beam decoding. | Unreadable. Its character-position metric undercounts (replacement characters shift the string); bytes were later measured in Phase 5D. |
| `decode_test_en.py` | Same for English text. Prints byte accuracy and the known-only baseline. | Bytes correct: 35–39% naive, 36–42% beam; known tokens alone ≈ 19%. |
| `decode_test_en_4095.py` | English text, S=4095, Adam 0.01, plus the blind mean-output candidate. | Bytes correct with 64 known tokens: 62–76% naive, 67–78% beam (readable with typos); blind ≈ 1–2%. |
| `examples.py` | For every Phase 5D variant and language, 5 models; prints the median model's original, weight-only and known-64 decodings (example texts for the post). | S=10000 / Adam 0.01 read in full from weights in both languages; S=4095 English readable with typos at k=64; the rest only fragments. |
| `v3_stress.py N` | Stress test of V3 (S=10000, Adam 0.001): output geometry, `dead_l1`, `input_fit` at 16× budget with k ∈ {64, 128, 256}. | See `../results/phase_5e_exploratory_v3_stress/RESULTS.md`. |
| `attack_lr.py N` | Cells trained with the unchanged code of [`latent_recall`](https://github.com/MigelSmirnov/latent_recall) (jitter/contrast training, 6-bit subtokens; set `LATENT_RECALL_DIR` to a clone). Attacks: `dead_l1`, mean output, and a line attack along PC1 that keeps the most integral non-trivial point whose tokens fit a 32,000-token vocabulary. | N=20, random unit keys and 80 random tokens: line attack recovered 20/20 fully; `dead_l1` and mean output 0/20. Outputs lie on one line (PC1 ≈ 20, PC2 ≈ 0.035); the target sits far along it. |

Also noted during this work: in `latent_recall`'s `train_with_jitter_and_contrast` the
term described as contrastive *repulsion* is `-cos(pred(key + noise), Y)`, which, when
minimized, *attracts* noisy-key outputs to the target.
