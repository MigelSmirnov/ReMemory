# KNC-0 — Phase 3: Adaptive White-Box Evaluation

## Goal
Assess whether adaptive, target-free analysis of a public 256→64→64→64 ReLU MLP reveals memorized 12-bit tokens or a usable input. This is a research experiment, not a cryptographic-security claim.

## Starting evidence
Phase 2.5: correct-key recovery 10/10; full recovery by cheap baselines 0/10; parameter-only MAE 461.88 versus midpoint MAE 1029.02. Evaluator-only first-layer active fraction: 10.47% on the correct input versus 49.24% on random inputs. These are hypotheses for testing, not known attack strategies.

## Threat model
The attacker has final weights, architecture, public code and nonsecret configuration. The attacker must not access the secret input, target tokens or evaluator-only correct-input activations. A separate stronger scenario may grant the original initialization weights. Generate secrets using OS entropy independently of public seeds. Freeze attack candidates before evaluation; never use the hidden target to select or optimize candidates.

## Phase 3.1 — ReLU geometry (first priority)
- Characterize publicly computable sparse-activation regions in the first layer.
- Measure how many distinct sparse regions or solutions exist, their input norms, and variation of model outputs within them.
- Test whether low activation fraction alone predicts better reconstruction than matched random-input controls.
- Evaluate similarity to the secret input and target only after candidate generation is finished.
- Explicitly distinguish discovering a sparse region from recovering the key.

## Phase 3.2 — Adaptive input analysis
- Compare random and gradient-based exploration under matched compute budgets.
- Use only attacker-observable objectives, such as output range, token lattice proximity, activation sparsity and input prior.
- Include multiple starts, degeneracy controls and predefined stopping rules.
- Measure target recovery only with an independent evaluator.

## Phase 3.3 — Parameter-only estimates
- Investigate output statistics derived from public weights and approximate hidden-activation distributions.
- Compare with Phase 2.5 moment propagation and random-output mean.
- Analyze known-initialization weight differences separately; Adam updates need not be rank-one.
- Do not conflate partial numerical leakage with exact token or key recovery.

## Metrics and reporting
For each attack, report token MAE/RMSE, exact-token fraction, whole-array recovery, within-1/8/64 token accuracy, elapsed CPU time, forward/backward passes, and memory usage. Report evaluator-only input similarity separately. Plot accuracy against compute budget.

Use separate pilot models for tuning and at least 10 new held-out models for evaluation (prefer 30 for final estimates). Include controls: untrained model, shuffled-target model, midpoint predictor, and Phase 2.5 parameter-only baseline. Report uncertainty and negative results.

## Deliverables
- Independent Phase 3 runner, tests, and documented CLI.
- RESULTS.md, aggregate.json, per_trial.csv, attack_summary.csv, plots and public attacker bundles.
- Explicit attacker/evaluator separation tests.
- Clear methodology, limitations and comparison with Phase 2.5.

## Execution
1. Verify Phase 2.5 reproducibility.
2. Implement Phase 3.1 and run a small CPU smoke test.
3. Proceed to 3.2 and 3.3 after validating the methods.
4. Freeze settings on pilot trials, then run held-out CPU evaluation.
5. Commit and push to `experiment/knc-0`; stop for review before Phase 4.

Do not modify `main`. Do not rent RunPod or incur GPU costs without explicit approval.
