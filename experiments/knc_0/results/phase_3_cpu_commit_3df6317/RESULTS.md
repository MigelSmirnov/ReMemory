# KNC-0 Phase 3 results

## Setup

The runner and settings were frozen in commit
`3df631704b8b5aae10d63cf1660305f89e0b0ef2` before this run. Two pilot models
and ten new held-out models used independently generated OS-CSPRNG inputs and
targets. Every held-out model reached exact 64-token recovery with its correct
input. Hidden targets and inputs were used only by evaluator functions after
attacker candidates were frozen.

The CPU run used 16 starts and budgets of 160, 800 and 3,200 forward-pass
equivalents. Peak process RSS was 330,334,208 bytes. The settings were not tuned
against held-out target metrics.

## Phase 3.1 — ReLU geometry

Sparse optimization found a mean of 15.8 distinct first-layer patterns among 16
starts and reduced mean first-layer active fraction to 6.91%. This successfully
found many sparse regions, but did not recover the secret region or target.

The target-free selected sparse candidate had mean token MAE 1,284.37
(95% normal CI 1,143.94–1,424.80) and zero full recoveries. Across all sparse
candidates, mean MAE was 1,082.74, versus 901.07 for norm-matched random inputs.
Thus low activation fraction alone predicted worse, not better, reconstruction
in this experiment. Mean local output standard deviation within pattern-preserving
perturbations was 0.00044 in normalized output units, showing locally stable but
non-secret regions.

Evaluator-only mean cosine similarity to the secret input was 0.157 for the
selected sparse candidate (mean L2 distance 20.66). These values were unavailable
to the attacker and were not used for candidate selection. Discovering sparse
regions is therefore explicitly distinct from recovering the key.

## Phase 3.2 — Adaptive inputs

At the largest 3,200-forward budget, the gradient candidate reached mean MAE
892.17 (763.53–1,020.82), versus 1,112.12 (962.64–1,261.59) for matched random
search. Their confidence intervals overlap substantially. Within-64 accuracy was
6.25% for gradient and 5.47% for random. Both produced zero full recoveries.

Gradient mean MAE changed from 1,051.21 at 10 steps to 1,055.49 at 50 and 892.17
at 200; performance was not monotonic at small budgets. Mean cumulative runtime
at 200 steps was 0.262 seconds for gradient and 0.013 seconds for random search.
Each gradient budget used the same number of forward equivalents as its random
control plus matched backward passes.

Evaluator-only cosine similarity at the largest budget was 0.108 for gradient
and 0.033 for random, with L2 distances 21.65 and 22.23. Neither is evidence of
usable key recovery. The shuffled-target adaptive control had MAE 895.60 and also
zero full recoveries, showing no special advantage tied to coordinate ordering.

## Phase 3.3 — Parameter-only estimates

The Phase 2.5 moment-propagation estimator remained the strongest tested method:
mean MAE 480.50 (398.28–562.72), within-64 accuracy 10.63%, and zero full
recoveries. It substantially outperformed the adaptive input candidates without
attempting input recovery.

Bias-only MAE was 1,648.28, midpoint MAE was 1,052.56, and the untrained-network
moment control was 1,778.77. The stronger known-initialization bias-delta estimator
had MAE 1,254.97 and zero full recoveries, worse than midpoint. This negative
result is consistent with the warning that Adam parameter changes need not expose
a simple rank-one association.

## Conclusion and limitations

Phase 3 found no exact target extraction and no usable secret input. It found
many sparse ReLU regions, but sparsity alone was not a recovery signal. The tested
target-free gradient objective modestly improved over random search at the largest
budget but was weaker than the public-weight output-distribution estimate from
Phase 2.5.

These are negative results for the predefined methods and budgets, not evidence
of cryptographic security. Different adaptive objectives, priors or larger budgets
remain untested. Phase 4 must not begin before review.

Machine-readable outputs are `aggregate.json`, `per_trial.csv`,
`attack_summary.csv`, `evaluator_input_similarity.json`, and individual reports.
`accuracy_vs_compute.svg` plots within-64 accuracy. Final weights are under each
`public/` directory; initial weights are separately exposed only under each
`known_initialization/` stronger-scenario directory.
