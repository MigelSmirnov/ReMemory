# Attacks on latent_recall cells trained with the package's own code (unchanged).
import sys, time, numpy as np, torch
import os
sys.path.insert(0, os.environ.get("LATENT_RECALL_DIR", "../latent_recall"))  # clone of MigelSmirnov/latent_recall
from latent_recall.mlp_cell import MLP, train_with_jitter_and_contrast
from latent_recall.codec import split_tokens
from latent_recall.trainer import infer_architecture
torch.set_num_threads(4)
N, NTOK = int(sys.argv[1]), 80                       # 80 LLM tokens -> 240 subtokens (0..63)
def rnd(rng, d): v = rng.standard_normal(d); return v / np.linalg.norm(v)
res = []
for t in range(N):
    rng = np.random.default_rng(100 + t)
    X = torch.tensor(rnd(rng, 384), dtype=torch.float32)            # unit key, like MiniLM embeddings
    toks = rng.integers(0, 32000, NTOK)                              # Llama-sized vocab
    Y = torch.tensor(split_tokens(toks), dtype=torch.float32)
    d, hid, m = infer_architecture(X[None], Y[None])
    torch.manual_seed(t); model = MLP(d, hid, m)
    t0 = time.time()
    model = train_with_jitter_and_contrast(model, X, Y, epochs=4000, lr=0.002, contrast_scale=0.35, verbose=False)
    model.eval(); tt = time.time() - t0
    for p in model.parameters(): p.requires_grad_(False)
    rd = lambda o: torch.clamp(o, 0, 63).round()
    truth = Y.numpy()
    with torch.no_grad(): mem = bool((rd(model(X))[0].numpy() == truth).all())
    # ---------- attacker: weights only ----------
    net = model.net
    with torch.no_grad():
        Z = torch.tensor(np.stack([rnd(np.random.default_rng(5000 + t * 10000 + i), 384) for i in range(1024)]), dtype=torch.float32)
        O = net(Z).double().numpy()
        dead = rd(net[4](torch.relu(net[2].bias)))[None].numpy()[0]   # W3 relu(b2) + b3
    rand_hit = float(np.mean([(np.clip(np.round(o), 0, 63) == truth).all() for o in O]))
    mean_c = np.clip(np.round(O.mean(0)), 0, 63)
    mu = O.mean(0); _, S, Vt = np.linalg.svd(O - mu, full_matrices=False); U = Vt[:2].T
    # 1-D line attack along PC1 (target-free): scan the whole interval where outputs stay in 0..63,
    # keep points whose rounded subtokens form valid Llama-vocab tokens (< 32000, public) and are
    # not trivial (no single value in > 50% of positions); pick the most integral one.
    u = Vt[0]
    lo_s = max(((-0.5 - mu[k]) / u[k] if u[k] > 0 else (63.5 - mu[k]) / u[k]) for k in range(len(u)) if abs(u[k]) > 1e-9)
    hi_s = min(((63.5 - mu[k]) / u[k] if u[k] > 0 else (-0.5 - mu[k]) / u[k]) for k in range(len(u)) if abs(u[k]) > 1e-9)
    best = (9, None, None)
    for sv in np.linspace(lo_s, hi_s, 200001):
        P = mu + sv * u
        R = np.clip(np.round(P), 0, 63)
        if (R[0::3] * 4096 + R[1::3] * 64 + R[2::3] >= 32000).any(): continue
        if np.bincount(R.astype(int), minlength=64).max() > 0.5 * len(R): continue
        e = np.abs(P - R).max()
        if e < best[0]: best = (e, R, sv)
    if best[1] is None: best = (9, np.zeros_like(mu), None)
    # ---------- evaluator ----------
    ok = lambda c: bool((c == truth).all()); fr = lambda c: float((c == truth).mean())
    with torch.no_grad(): l1 = int((torch.relu(net[0](X)) > 0).sum())
    res.append(dict(mem=mem, train_s=tt, l1=l1, rand=rand_hit, dead=ok(dead), dead_f=fr(dead), mean=ok(mean_c), mean_f=fr(mean_c),
                    plane=ok(best[1]), plane_f=fr(best[1]), plane_err=best[0], pcs=np.round(S[:4] / np.sqrt(len(O)), 3).tolist()))
    print(t, res[-1], flush=True)
n = len(res); f = lambda k: sum(r[k] for r in res)
print(f"\nlatent_recall cells: {n}, memorized {f('mem')}/{n}, train {np.median([r['train_s'] for r in res]):.1f}s median")
print(f"key L1 fully inactive: {sum(r['l1']==0 for r in res)}/{n} (median active {np.median([r['l1'] for r in res]):.0f})")
print(f"random unit input already gives the whole memory: median share {np.median([r['rand'] for r in res]):.3f}, models with any hit {sum(r['rand']>0 for r in res)}/{n}")
print(f"full recovery from weights only: dead-L1 {f('dead')}/{n} | mean output {f('mean')}/{n} | plane+lattice {f('plane')}/{n}")
print(f"subtoken accuracy medians: dead-L1 {np.median([r['dead_f'] for r in res]):.2f} mean {np.median([r['mean_f'] for r in res]):.2f} plane {np.median([r['plane_f'] for r in res]):.2f}")
