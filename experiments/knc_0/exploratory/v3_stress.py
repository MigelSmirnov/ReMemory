import sys, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from experiments.knc_0.phase_5d import train
from experiments.knc_0.phase_5a import load_public, public_bundle, output_matrix, to_tokens
from experiments.knc_0.phase_5b import known_positions
from experiments.knc_0.phase_5c import adam_fit, dead_l1_candidate
torch.set_num_threads(4)
S, M, D, N = 10000.0, 256, 384, int(sys.argv[1])
mk = lambda p: torch.optim.Adam(p, lr=1e-3)
rows = []; t0 = time.time()
for t in range(N):
    rng = np.random.default_rng(3000 + t)
    x = rng.standard_normal(D); x = torch.tensor((x / np.linalg.norm(x))[None], dtype=torch.float32)
    truth = rng.integers(0, 4096, M)
    model, steps, ok = train(x, truth, S, mk, 3100 + t, 5000)
    with torch.no_grad(): mem = bool(np.array_equal(to_tokens(model(x), S)[0].numpy(), truth))
    pub = load_public(public_bundle(model, S))
    # 1. geometry (attacker computes plane; evaluator measures distance of truth from it)
    O = output_matrix(pub, D, S, 2048, 3200 + t); mu = O.mean(0)
    _, sv, Vt = np.linalg.svd(O - mu, full_matrices=False); rms = sv / np.sqrt(len(O))
    dist = {k: float(np.abs((truth - mu) - Vt[:k].T @ (Vt[:k] @ (truth - mu))).max()) for k in (1, 2, 8, 32)}
    row = dict(t=t, mem=mem, steps=steps, pc=np.round(rms[:4], 1).tolist(), dist=dist,
               dead=float((dead_l1_candidate(pub, S) == truth).mean()))
    # 2-3. big-budget input fit
    for k in (64, 128, 256):
        idx = known_positions(M, k, 3300 + t * 10 + k)
        s0 = time.time()
        fit = adam_fit(pub, S, idx, truth[idx].astype(float), seed=3400 + t, steps=8000, restarts=32)
        cand = np.clip(np.round(fit["all_out"][fit["best"]]), 0, 4095).astype(int)
        unk = np.setdiff1d(np.arange(M), idx)
        row[f"k{k}"] = dict(known_mse=round(fit["known_mse"], 3), known_ok=bool((cand[idx] == truth[idx]).all()),
                            unk_exact=float((cand[unk] == truth[unk]).mean()) if len(unk) else None,
                            unk_mae=float(np.abs(cand[unk] - truth[unk]).mean()) if len(unk) else None,
                            sec=round(time.time() - s0, 1))
    rows.append(row); print(row, f"({time.time()-t0:.0f}s)", flush=True)
print("\nSUMMARY V3 (S=10000, Adam lr=1e-3), models:", len(rows), "memorized:", sum(r["mem"] for r in rows))
print("PC rms medians:", np.median([r["pc"] for r in rows], 0).tolist())
for k in (1, 2, 8, 32): print(f"truth max distance from {k}-D plane: median {np.median([r['dist'][k] for r in rows]):.1f} tokens")
print("dead_l1 exact median:", np.median([r["dead"] for r in rows]))
for k in (64, 128, 256):
    rr = [r[f"k{k}"] for r in rows]
    print(f"k={k}: known reproduced {sum(q['known_ok'] for q in rr)}/{len(rr)}, known MSE median {np.median([q['known_mse'] for q in rr]):.3f}"
          + (f", unknown exact median {np.median([q['unk_exact'] for q in rr]):.3f}, full {sum(q['unk_exact']==1.0 for q in rr)}/{len(rr)}, MAE median {np.median([q['unk_mae'] for q in rr]):.1f}" if k < 256 else "")
          + f", {np.median([q['sec'] for q in rr]):.0f}s/run")
