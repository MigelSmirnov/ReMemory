import sys, base64, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from experiments.knc_0.phase_5a import make_model, public_bundle, load_public, to_tokens
from experiments.knc_0.phase_5b import known_positions
from experiments.knc_0.phase_5c import adam_fit
from experiments.knc_0.phase_5a import output_matrix
torch.set_num_threads(4)
A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
def enc(t):
    b = base64.b64encode(t.encode()).decode().rstrip("=")
    return [(A.index(b[i]) << 6) + A.index(b[i + 1]) for i in range(0, len(b) - 1, 2)]
def to_bytes(ids):
    s = "".join(A[t >> 6] + A[t & 63] for t in ids); s += "=" * ((-len(s)) % 4)
    return base64.b64decode(s)
TEXT = ("In the morning a thick fog lay over the river, and the trees on the far bank were barely visible. "
        "By noon the wind grew stronger, the clouds broke apart, and the sun lit up the wet roofs of the houses. "
        "In the evening it became cool again, the street lamps came on, and warm light appeared in the windows. ") * 2
tok = enc(TEXT)[:256]; truth = np.array(tok); m = 256
GOOD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.-")
def char_score(bs):
    try: s = bs.decode("utf-8")
    except UnicodeDecodeError as e:
        return None
    return sum(1 if ch in GOOD else -3 for ch in s)

def beam_decode(est, known, R=40, width=200):
    """Left to right over tokens; every 2 tokens = 3 bytes. Candidates within +-R of estimate; known tokens fixed."""
    beams = [(0.0, [])]
    for i in range(m):
        cands = [known[i]] if i in known else range(max(0, int(round(est[i])) - R), min(4095, int(round(est[i])) + R) + 1)
        new = []
        for sc, seq in beams:
            for c in cands:
                s2 = seq + [c]
                if len(s2) % 2 == 0:      # 2 tokens = 4 base64 chars = 3 bytes: score the new bytes
                    bs = to_bytes(s2)
                    # score only complete UTF-8 prefix: drop up to 1 trailing incomplete char
                    val = None
                    for cut in (0, 1, 2, 3):
                        val = char_score(bs[:len(bs) - cut] if cut else bs)
                        if val is not None: break
                    if val is None:
                        val = char_score(bs.decode("utf-8", "replace").encode("utf-8")) - 50  # invalid: penalize, keep
                    new.append((val - 0.001 * abs(c - est[i]), s2))
                else:
                    new.append((sc - 0.001 * abs(c - est[i]), s2))
        new.sort(key=lambda q: -q[0]); beams = new[:width]
    return beams[0][1]

S = 4095.0
for seed in range(3):
    x = torch.tensor(np.random.default_rng(1000 + seed).standard_normal((1, 384)), dtype=torch.float32); x /= x.norm()
    torch.manual_seed(seed); net = make_model(384, m); opt = torch.optim.Adam(net.parameters(), lr=1e-2)   # as in Phases 5A-5C
    y = torch.tensor([tok], dtype=torch.float32) / S
    for step in range(5000):
        opt.zero_grad(); nn.functional.mse_loss(net(x), y).backward(); opt.step()
        with torch.no_grad():
            if float(((net(x) * S - y * S) ** 2).mean()) <= 1e-5: break
    with torch.no_grad(): mem = bool(torch.equal(to_tokens(net(x), S), torch.tensor([tok])))
    pub = load_public(public_bundle(net, S))
    idx = known_positions(m, 64, 7000 + seed)
    fit = adam_fit(pub, S, idx, truth[idx].astype(float), seed=8000 + seed)          # attacker
    est = fit["all_out"][fit["best"]]
    naive = np.clip(np.round(est), 0, 4095).astype(int); naive[idx] = truth[idx]
    known = {int(i): int(truth[i]) for i in idx}
    e = np.abs(est[np.setdiff1d(np.arange(m), idx)] - truth[np.setdiff1d(np.arange(m), idx)])
    print(f"errors: share within 40 = {np.mean(e<=40):.1%}, within 80 = {np.mean(e<=80):.1%}")
    beam = beam_decode(est, known)
    unk = np.setdiff1d(np.arange(m), idx)
    ref = to_bytes(tok).decode("utf-8", "replace")
    def show(ids):
        s = to_bytes(ids).decode("utf-8", "replace")
        same = sum(a == b for a, b in zip(s, ref)) / len(ref)
        return s, same
    s_n, a_n = show(naive); s_b, a_b = show(beam)
    tb = to_bytes(tok)
    bacc = lambda ids: np.mean(np.frombuffer(to_bytes(list(ids)), np.uint8) == np.frombuffer(tb, np.uint8))
    base = np.mean([bacc(np.where(np.isin(np.arange(m), idx), truth, np.random.default_rng(r).integers(0, 4096, m))) for r in range(20)])
    blind = np.clip(np.round(output_matrix(pub, 384, S, 1024, 9000 + seed).mean(0)), 0, 4095).astype(int)   # attacker: weights only
    print(f"BLIND mean-output: token MAE {np.abs(blind-truth).mean():.1f}, bytes {bacc(blind):.1%} | {to_bytes(list(blind)).decode('utf-8','replace')[:90]!r}")
    print(f"BYTES correct: known-64 only (rest random) {base:.1%} | naive {bacc(naive):.1%} | beam {bacc(beam):.1%}")
    print(f"\n=== model {seed}: memorized={mem}, steps={step+1}, unknown-token MAE={np.abs(est[unk]-truth[unk]).mean():.1f}, "
          f"max err={np.abs(est[unk]-truth[unk]).max():.0f}")
    print(f"naive decode : {a_n:5.1%} chars correct | {s_n[:110]!r}")
    print(f"beam decode  : {a_b:5.1%} chars correct, exact tokens {np.mean(np.array(beam)[unk]==truth[unk]):.1%} | {s_b[:110]!r}")
print(f"\ntruth        : {ref[:110]!r}")
