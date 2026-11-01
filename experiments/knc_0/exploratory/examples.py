import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from experiments.knc_0.phase_5d import VARIANTS, secret_text, train, tokens_to_bytes, text_scores, M, D
from experiments.knc_0.phase_5a import load_public, public_bundle, output_matrix
from experiments.knc_0.phase_5b import known_positions
from experiments.knc_0.phase_5c import adam_fit, dead_l1_candidate
torch.set_num_threads(4)
def show(ids, n):
    s = tokens_to_bytes(ids).decode("utf-8", "replace")
    return "".join(ch if ch.isprintable() else "·" for ch in s)[:n]
N = 5
for lang, width in (("en", 100), ("ru", 60)):
    for name, (scale, mk) in VARIANTS.items():
        trials = []
        for t in range(N):
            rng = np.random.default_rng(900 + t)          # public example seeds, not secrets of the study
            x = rng.standard_normal(D); x = torch.tensor((x / np.linalg.norm(x))[None], dtype=torch.float32)
            truth = secret_text(lang, rng)
            model, _, ok = train(x, truth, scale, mk, 950 + t, 5000)
            pub = load_public(public_bundle(model, scale))
            blind = dead_l1_candidate(pub, scale)
            mean = np.clip(np.round(output_matrix(pub, D, scale, 1024, 970 + t).mean(0)), 0, 4095).astype(int)
            sb, sm = text_scores(blind, truth, np.array([], int)), text_scores(mean, truth, np.array([], int))
            best_blind = blind if sb["byte_accuracy"] >= sm["byte_accuracy"] else mean
            idx = known_positions(M, 64, 990 + t)
            fit = adam_fit(pub, scale, idx, truth[idx].astype(float), seed=995 + t)
            cand = np.clip(np.round(fit["all_out"][fit["best"]]), 0, 4095).astype(int); cand[idx] = truth[idx]
            sc = text_scores(cand, truth, idx)
            trials.append((sc["byte_accuracy"], truth, best_blind, cand, max(sb["byte_accuracy"], sm["byte_accuracy"]), sc["word_recovery"], ok))
        trials.sort(key=lambda q: q[0])
        acc, truth, blind, cand, bacc, wr, ok = trials[N // 2]
        print(f"\n### {lang} {name}  (median of {N} by k=64 byte accuracy; memorized={ok})")
        print(f"truth   : {show(truth, width)}")
        print(f"weights : {show(blind, width)}   [bytes {bacc:.0%}]")
        print(f"k=64    : {show(cand, width)}   [bytes {acc:.0%}, words {wr:.0%}]")
