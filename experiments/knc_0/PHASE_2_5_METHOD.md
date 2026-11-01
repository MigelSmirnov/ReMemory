# Phase 2.5 methodology

`phase_2_5.py` measures non-adaptive output leakage. It does not search for the
secret key and does not establish security.

The attacker API receives only a trained state dict, architecture, public probe
counts, and independent probe randomness. It freezes random-output-mean,
last-layer-bias, and parameter-only moment-propagation candidates without access
to the secret input or target. The evaluator subsequently scores those candidates.
Correct-input activation patterns are computed only in a separately named
evaluator function after attacker observations are frozen.

Secret inputs and targets use fresh OS CSPRNG entropy. That entropy is neither
returned nor persisted. Recorded training/probe seeds control only public model
initialization and attacker sampling and cannot regenerate evaluator secrets.

Run the full CPU experiment with:

```bash
python experiments/knc_0/phase_2_5.py --out knc0_phase_2_5_results
pytest -q tests/test_knc_0.py tests/test_knc_0_phase_2_5.py
```

Large probe counts are streamed in batches. Per-coordinate distributions use
fixed 4096-bin token histograms, so 100,000 probes do not require retaining the
probe matrix. The generated SVG files chart mean MAE and cumulative runtime.
