# Phase 4A methodology

Phase 4A is a single-token scaling study, not a secrecy experiment. It runs the
full `d -> h -> h -> 1` matrix from `PHASE_4A_TASK.md` plus a `d -> 1` linear
baseline. Each secret input and 12-bit target uses fresh OS entropy independent
of recorded training and probe seeds.

Zero-input, random-input-mean and independently initialized ensemble candidates
are produced without target access and scored only afterward. Failed training
runs remain in CSV and aggregate counts. Exact-rate uncertainty uses Wilson 95%
intervals; continuous metrics use normal mean intervals. The reported transition
is only the first tested width reaching a descriptive 95% exact rate—no power law
is fitted or assumed.

Run a smoke test:

```bash
python experiments/knc_0/phase_4a.py --out /tmp/knc4a_smoke \
  --input-dims 1 4 --widths 1 2 --trials 2 --epochs 500
```

Run the full CPU matrix:

```bash
python experiments/knc_0/phase_4a.py --out knc0_phase_4a_results
pytest -q -p no:cacheprovider tests/test_knc_0_phase_4a.py
```
