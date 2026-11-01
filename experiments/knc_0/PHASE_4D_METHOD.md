# Phase 4D methodology

Settings are pre-registered before held-out evaluation: 4,096 standard-normal
inputs per model, Recall@10, and six target-free rankings (mode, coordinate-median
medoid, local stability among 64 frequent arrays, analytic moment propagation,
first random output and zero input). Three pilot models per width are disjoint from
30 held-out models. Candidate rankings are frozen before target occurrence and
identification are evaluated.

Occurrence is an evaluator-only oracle upper bound, not an attack. Reports split
failures into target never sampled, sampled but missed by mode ranking, and mode
identified. Untrained networks provide the clustering negative control. Fixed
sampling budgets prevent claims about asymptotic or exponential complexity.
