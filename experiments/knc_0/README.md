# KNC-0 — ReLU neural memory leakage baseline

This is an isolated experiment. It does not modify the ReMemory training or recall pipeline.

## Run

```bash
pip install torch numpy
python experiments/knc_0/run.py --epochs 5000 --probes 10000
```

CPU is sufficient; do not rent a GPU yet.

The script trains a 256 -> 64 -> 64 -> 64 ReLU MLP to memorize one mapping from a random input to 64 random 12-bit tokens (scaled to [0,1] for stable training). It saves `knc0_results/public/model.pt` and `config.json` without the input or target, then reloads the model to run cheap white-box probes.

The report includes exact token recovery and MAE for the correct key, zero input, one random input, average of random outputs, and last-layer bias. The "best_random_probe_oracle" metric is **not a realistic attacker method**: it uses the hidden target to pick the best of the random probes, solely as an evaluation upper bound.

SVD singular values are diagnostic only; they do not constitute key recovery.

**Limitations:** This does not establish cryptographic security or GPU-days-to-break. A random-output failure does not imply resistance to optimized attacks, parameter extraction, or side information. The benchmark uses random 12-bit tokens rather than meaningful text to prevent language-based shortcuts. It measures whole-token accuracy, not actual Base64 text decoding.

To compare seeds:
```bash
python experiments/knc_0/run.py --seed 1 --out knc0_seed1
python experiments/knc_0/run.py --seed 2 --out knc0_seed2
python experiments/knc_0/run.py --seed 3 --out knc0_seed3
```

Public artifacts are under each run's `public/` folder; `report.json` contains aggregate evaluation metrics only. Do not treat these experiments as a secure container for real secrets.
