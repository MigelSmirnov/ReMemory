# Phase 3 methodology

Phase 3 evaluates target-free white-box candidates. It does not claim security.

## Frozen settings

- 2 CSPRNG pilot models and 10 new CSPRNG held-out models.
- 16 starts; sparse-region search for 200 steps.
- Gradient and matched-random budgets: 10, 50 and 200 steps, corresponding to
  160, 800 and 3,200 forward-pass equivalents.
- The adaptive objective combines token-lattice proximity, valid output range,
  first-layer soft sparsity and a standard-normal input prior. Its weights and
  stopping rules are fixed before held-out evaluation.

Attacker functions accept public models, configuration and attacker randomness
only. They return frozen input or parameter candidates. Separately named evaluator
functions subsequently receive hidden targets and inputs to compute recovery and
input-similarity metrics. Secrets use OS CSPRNG entropy that is never persisted.

Phase 3.1 finds sparse first-layer regions and compares them with norm-matched
random inputs. Finding such a region is not key recovery. Phase 3.2 compares
gradient and random exploration under matched forward budgets. Phase 3.3 compares
public-weight moment propagation, bias and midpoint estimates; an explicitly
stronger known-initialization scenario receives initial weights and uses a fixed
bias-update-direction estimator without assuming Adam updates are rank one.

Run:

```bash
python experiments/knc_0/phase_3.py --out knc0_phase_3_results
pytest -q -p no:cacheprovider \
  tests/test_knc_0.py tests/test_knc_0_phase_2_5.py tests/test_knc_0_phase_3.py
```
