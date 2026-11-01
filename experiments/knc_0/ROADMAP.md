# ReMemory / KNC — Research Roadmap v1.0

## Objective
Evaluate how much information a single-association ReLU MLP leaks when an attacker has the full architecture and trained weights but not the secret input. This is an experimental neural-memory study, **not a claim of cryptographic security**.

Work only in `experiment/knc-0`; preserve the existing ReMemory behavior. The starting implementation is `experiments/knc_0/run.py`.

## Rules
- Keep training, public attacker artifacts, and secret evaluation data isolated.
- Attacker sees final weights, architecture, public code, and configuration; never the key or target.
- No use of ground truth to choose candidate attacks or tune an attack in the same trial.
- Record seeds, code revision, library versions, device, runtime, RAM/VRAM, model size, and results as JSON/CSV.
- Start on CPU; use paid GPU only after profiling.
- Do not add AES, decoys, or LLM reconstruction until the baseline is understood.

## Phase 0 — Reproducible harness
Configurable CPU/CUDA, seeds (Python/NumPy/PyTorch), independent run IDs, clean artifacts, and automated smoke tests. Check existing `run.py` and fix any correctness/reproducibility issues. The attacker-facing directory must contain only model weights and public configuration. Keep evaluation-only ground truth in memory or a separately access-controlled location. Document that seed leakage can recreate synthetic secrets; do not expose training seeds as attacker inputs if they deterministically generate the key and target.

## Phase 1 — Memorization baseline
- MLP: 256 → 64 → 64 → 64; ReLU, Adam, MSE.
- One random 256-dimensional input mapped to 64 random 12-bit tokens, normalized for training.
- Minimum 10 independent trials.
- Metrics: training loss, rounded-token exact accuracy, whole-vector exact recovery, epochs, time, model bytes.
- Gate: reliable exact recovery under the correct input before attacking. If not achieved, debug training first.
- Keep random-token tests separate from text/byte-codec tests.

## Phase 2 — Cheap white-box baselines
Attacker loads only public weights/config and tries: zero input, random inputs, mean of random outputs, last-layer bias, weight statistics, first-layer SVD, and comparisons with an untrained model. Report token exact-match rate, full recovery, MAE, elapsed time. Any oracle-selected best random probe is **evaluation-only**, not a realistic attacker result. Do not equate SVD singular values with key recovery.

## Phase 3 — Adaptive attacks
Explore gradient-based input search and parameter-based extraction. Separate threat models:
A. final weights only;
B. final weights + known initialization;
C. final weights + partial knowledge of data format.
Define attacker-available objectives, compute/time budgets, and success-vs-cost curves. Measure both key recovery and direct data extraction. Never optimize against hidden target values.

## Phase 4 — Architecture ablation
Change one factor at a time:
- hidden width 32/64/128/256;
- hidden depth 1/2/3;
- ReLU/GELU/SiLU;
- SGD/Adam/AdamW;
- input dimension 128/256/512;
- output length 64/256/1024 tokens.
Compare only models reaching comparable correct-key reconstruction and under equal attack budgets. Use repeated seeds and report variability.

## Phase 5 — Original ReMemory benchmark (separate track)
Assess associative retrieval and reconstruction:
- exact byte and token accuracy;
- Recall@1/3, MRR;
- exact, paraphrased, partial, noisy and conflicting queries;
- 100/1000/10000 episodes;
- training/recall latency, CPU RAM/GPU VRAM, disk size;
- baseline: embedding retrieval with plaintext storage.
Crucially distinguish reconstructing from the saved original context vector vs reconstructing from the actual query embedding. Do not conflate retrieval accuracy with byte reconstruction accuracy.

## Deliverables
1. Reproducible runner and attack suite.
2. Experiment configs and tests.
3. JSON/CSV outputs and accuracy/cost plots.
4. README with execution steps and limitations.
5. Short research report: observed leakage, attack costs, uncertainty, and next experiments.

## Run order and budget
First complete Phase 0–2 CPU smoke tests and share results. Then decide whether Phase 3 merits RunPod. Budget is approximately $10; do not launch long GPU jobs without a cost estimate and explicit confirmation.

## First Codex task
Audit `experiments/knc_0/run.py`, run a short CPU smoke test, validate exact recovery and attack metrics, fix issues in this experimental branch only, and summarize Phase 1–2 results. Do not modify `main`.
