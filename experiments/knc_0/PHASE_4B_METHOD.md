# Phase 4B methodology

The standalone runner executes the specified 24 configurations with 30 fresh
OS-CSPRNG examples each. Public training/probe seeds cannot regenerate inputs or
targets. Attacker diagnostics accept only the public trained model and sampling
configuration; evaluator metrics are computed after candidates are frozen.

An output array is predefined as approximately constant only when every coordinate
has rounded-token standard deviation <=1 and at least 99% of 512 random inputs
produce its dominant token. Reports retain failures and separate correct-key
capacity from off-key token and full-array recovery. Wilson intervals are used for
recovery proportions; continuous metrics use normal mean intervals.

```bash
python experiments/knc_0/phase_4b.py --out knc0_phase_4b_results
pytest -q -p no:cacheprovider tests/test_knc_0_phase_4b.py
```
