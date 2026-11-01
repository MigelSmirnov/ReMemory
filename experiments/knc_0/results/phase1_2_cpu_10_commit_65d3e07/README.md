# KNC-0 Phase 1–2 CPU results

Produced from commit `65d3e0703ed6c7f9313ba78caf5267eb44933850` with:

```bash
python experiments/knc_0/run.py \
  --trials 10 --epochs 5000 --probes 10000 --threads 2 \
  --device cpu --run-id phase1_2_cpu_10_commit_65d3e07
```

## Environment

- Python 3.12.3; PyTorch 2.15.0+cpu; NumPy 2.5.4
- Linux 7.0.0-38-generic x86_64, glibc 2.39
- 8 logical CPUs; runner limited PyTorch to 2 threads; no GPU
- peak process RSS across the sequential run: 347,795,456 bytes

## Outcome

- Correct-key exact recovery: 10/10 trials.
- Training stopped after 200–400 steps.
- Mean training time: 0.2826 seconds per trial.
- Serialized model size: 101,565 bytes per trial.
- Full recovery by every tested cheap baseline: 0/10.
- The evaluation-only best-of-10,000 random-probe oracle matched 2/64 tokens
  on average and at most 3/64 tokens; it never recovered a full vector.

`summary.csv` contains per-seed training and oracle results.
`attack_summary.csv` aggregates every Phase 2 baseline. `aggregate.json` is the
machine-readable run-level result.

No secret input, target tokens, training seed in attacker configuration, or
model weights are included. Weights are deterministically reproducible from the
recorded revision and seeds, while exposing a synthetic seed as an attacker
artifact recreates both secret input and target. These results establish only
the cheap-baseline outcome, not cryptographic security or adaptive-attack resistance.
