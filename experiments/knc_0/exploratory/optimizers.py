import sys, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from experiments.knc_0.phase_5a import secret_example, make_model, public_bundle, load_public, to_tokens, wilson
from experiments.knc_0.phase_5b import known_positions
from experiments.knc_0.phase_5c import dead_l1_candidate, layer_activity, adam_fit
torch.set_num_threads(4)
S = 10000.0
OPTS = {"adam_1e-2": lambda p: torch.optim.Adam(p, lr=1e-2),
        "adam_1e-3": lambda p: torch.optim.Adam(p, lr=1e-3),
        "sgd_m0.9_0.1": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
        "adamw_1e-2_wd1e-2": lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=1e-2)}

def train(x, tok, make_opt, seed, cap=5000):
    torch.manual_seed(seed); m = make_model(384, 256); opt = make_opt(m.parameters()); y = tok.float() / S
    for step in range(1, cap + 1):
        opt.zero_grad(); nn.functional.mse_loss(m(x), y).backward(); opt.step()
        with torch.no_grad(): tmse = float(((m(x) * S - tok.float()) ** 2).mean())
        if tmse <= 1e-5: return m, step, True, tmse
    return m, cap, False, tmse

res = {k: [] for k in OPTS}
t0 = time.time()
for t in range(int(sys.argv[1])):
    x, tok = secret_example(384, 256); truth = tok[0].numpy()
    idx = known_positions(256, 64, 60000 + t); vals = truth[idx].astype(float)
    for name, mk in OPTS.items():
        m, steps, ok, tmse = train(x, tok, mk, 61000 + t)
        with torch.no_grad(): mem = bool(torch.equal(to_tokens(m(x), S), tok))
        pub = load_public(public_bundle(m, S))
        c0 = dead_l1_candidate(pub, S)                                  # attacker: weights only
        fit = adam_fit(pub, S, idx, vals, seed=62000 + t)                # attacker: 64 known tokens
        c1 = np.clip(np.round(fit["all_out"][fit["best"]]), 0, 4095).astype(int)
        unk = np.setdiff1d(np.arange(256), idx)
        res[name].append(dict(mem=mem, ok=ok, steps=steps, l1=layer_activity(pub, x[0])["l1_active"],   # evaluator
                              w_full=bool((c0 == truth).all()), w_mae=float(np.abs(c0 - truth).mean()),
                              k_full=bool((c1[unk] == truth[unk]).all()), k_mae=float(np.abs(c1[unk] - truth[unk]).mean())))
    print(f"t={t} done {time.time()-t0:.0f}s", flush=True)
print(f"\nS=10000, {len(res['adam_1e-2'])} paired secrets per optimizer")
for name, r in res.items():
    n = len(r); f = lambda k: sum(q[k] for q in r)
    print(f"{name:18} rule {f('ok')}/{n} memorized {f('mem')}/{n} steps med {np.median([q['steps'] for q in r]):.0f} | "
          f"key L1 dead {sum(q['l1']==0 for q in r)}/{n} (L1 active med {np.median([q['l1'] for q in r]):.0f}) | "
          f"weights-only full {f('w_full')}/{n} {np.round(wilson(f('w_full'),n),2).tolist()} MAE med {np.median([q['w_mae'] for q in r]):.1f} | "
          f"64-known full {f('k_full')}/{n} {np.round(wilson(f('k_full'),n),2).tolist()} MAE med {np.median([q['k_mae'] for q in r]):.1f}")
