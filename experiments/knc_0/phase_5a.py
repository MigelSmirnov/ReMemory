#!/usr/bin/env python3
"""Phase 5A: target scaling versus the blind PCA/lattice attack.

A, B and C differ only in the training target scale (1, 4095, 10000). Within a
trial they share the secret and the initial weights. The attacker sees public
weights, architecture and output scale only.
"""
import argparse, csv, json, math, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.knc_0.run import environment_metadata  # noqa: E402

SCALES = {"A": 1.0, "B": 4095.0, "C": 10000.0}
RULES = ("R1", "R2", "R3")
ACCEPT_MAX_FRAC = 0.25  # pre-registered


# ------------------------------------------------------------------ model (ReMemory shape)
def make_model(d, m):
    h = max(128, (d + m) // 2)
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, m))


def secret_example(d, m):
    """Secret input and target from OS entropy; never persisted."""
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(os.urandom(32), "little")))
    x = rng.standard_normal(d)
    x /= np.linalg.norm(x)  # unit vector, like a normalized sentence embedding
    return torch.tensor(x[None], dtype=torch.float32), torch.tensor(rng.integers(0, 4096, (1, m)))


def to_tokens(raw, scale):
    return (raw * scale).round().clamp(0, 4095).long()


def train(x, tokens, scale, rule, init_seed, args):
    torch.manual_seed(init_seed)
    model = make_model(x.shape[1], tokens.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    target = tokens.float() / scale
    start, reached = time.perf_counter(), False
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        opt.step()
        if rule == "R1":
            if step % args.check_every == 0:
                with torch.no_grad():
                    reached = torch.equal(to_tokens(model(x), scale), tokens)
        elif rule == "R2":
            reached = float(loss) * scale * scale <= 1e-5
        else:  # R3
            reached = float(loss) <= 1e-5
        if reached:
            break
    model.eval()
    with torch.no_grad():
        token_mse = float(((model(x) * scale - tokens.float()) ** 2).mean())
    return model, {"steps": step, "rule_reached": bool(reached), "token_mse": token_mse,
                   "train_seconds": time.perf_counter() - start}


def public_bundle(model, scale):
    d, m = model[0].in_features, model[4].out_features
    return {"state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "arch": [d, model[0].out_features, m], "output_scale": scale}


def load_public(pub):
    d, h, m = pub["arch"]
    model = make_model(d, m)
    assert model[0].out_features == h
    model.load_state_dict(pub["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ------------------------------------------------------------------ ATTACKER (public only)
def output_matrix(model, d, scale, n, probe_seed):
    rng = np.random.default_rng(probe_seed)
    z = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    z = z / z.norm(dim=1, keepdim=True)
    with torch.no_grad():
        return model(z).double().numpy() * scale


def principal_plane(outputs):
    mu = outputs.mean(0)
    _, s, vt = np.linalg.svd(outputs - mu, full_matrices=False)
    return mu, vt[:2].T, s / math.sqrt(len(outputs))


def lattice_search(mu, basis, k_check=24, lo=0, hi=4095):
    by1 = np.argsort(-np.abs(basis[:, 1]))[:40]
    by0 = np.argsort(-np.abs(basis[:, 0]))[:40]
    _, i, j = max((abs(np.linalg.det(basis[[a, b]])), a, b) for a in by1 for b in by0 if a != b)
    inv = np.linalg.inv(basis[[i, j]])
    rest = [r for r in np.argsort(-np.linalg.norm(basis, axis=1)) if r not in (i, j)][:k_check]
    ur, mr = basis[rest], mu[rest]
    tj = np.arange(lo, hi + 1, dtype=float)
    top = []
    for ti in range(lo, hi + 1):
        ab = inv @ np.stack([np.full_like(tj, ti - mu[i]), tj - mu[j]])
        pred = mr[:, None] + ur @ ab
        score = np.abs(pred - np.round(pred)).max(0) + 10 * ((pred < lo - .5) | (pred > hi + .5)).any(0)
        for q in np.argsort(score)[:2]:
            top.append((float(score[q]), ab[:, q]))
        top = sorted(top, key=lambda t: t[0])[:2]
    return top


def attack(public, n_probes, probe_seed):
    """Blind PCA/lattice attack. Accepts no target, key or secret."""
    start = time.perf_counter()
    model = load_public(public)
    d, _, m = public["arch"]
    outputs = output_matrix(model, d, public["output_scale"], n_probes, probe_seed)
    mu, basis, rms = principal_plane(outputs)
    (best, ab), (runner_up, _) = lattice_search(mu, basis)
    full = mu + basis @ ab
    frac = np.abs(full - np.round(full))
    in_range = bool(full.min() >= -0.5 and full.max() <= 4095.5)
    return {"tokens": np.clip(np.round(full), 0, 4095).astype(int),
            "accept": bool(frac.max() < ACCEPT_MAX_FRAC and in_range),
            "rms_pc": rms[:5].tolist(), "offplane_rms": float(math.sqrt((rms[2:] ** 2).sum())),
            "max_frac": float(frac.max()), "best_score": best, "runner_up_score": runner_up,
            "plane": (mu, basis), "attack_seconds": time.perf_counter() - start}


# ------------------------------------------------------------------ EVALUATOR (secret side)
def evaluate(model, scale, x, tokens, candidate):
    truth = tokens[0].numpy()
    with torch.no_grad():
        key_tokens = to_tokens(model(x), scale)[0].numpy()
    mu, basis = candidate["plane"]
    resid = (truth - mu) - basis @ (basis.T @ (truth - mu))
    full = bool(np.array_equal(candidate["tokens"], truth))
    return {"memorized": bool(np.array_equal(key_tokens, truth)),
            "key_exact_fraction": float((key_tokens == truth).mean()),
            "target_plane_resid_max": float(np.abs(resid).max()),
            "exact_fraction": float((candidate["tokens"] == truth).mean()),
            "full_recovery": full,
            "false_accept": bool(candidate["accept"] and not full),
            "missed_recovery": bool(full and not candidate["accept"])}


# ------------------------------------------------------------------ driver
def run_trial(trial, rules, args, pilot):
    x, tokens = secret_example(args.input_dim, args.output_dim)
    init_seed = args.init_seed + trial + (100000 if pilot else 0)
    rows = []
    for rule in rules:
        for name, scale in SCALES.items():
            if rule == "R3" and name == "A":
                continue  # identical to R2 for A
            model, tr = train(x, tokens, scale, rule, init_seed, args)
            cand = attack(public_bundle(model, scale), args.probes,
                          args.probe_seed + trial + (100000 if pilot else 0))
            ev = evaluate(model, scale, x, tokens, cand)
            row = {"pilot": pilot, "trial": trial, "rule": rule, "model": name, "scale": scale, **tr, **ev,
                   "accept": cand["accept"], "max_frac": cand["max_frac"],
                   "best_score": cand["best_score"], "runner_up_score": cand["runner_up_score"],
                   "offplane_rms": cand["offplane_rms"], "attack_seconds": cand["attack_seconds"],
                   **{f"rms_pc{i + 1}": v for i, v in enumerate(cand["rms_pc"])}}
            rows.append(row)
            if not args.quiet:
                print(f"{'pilot' if pilot else 'held'} t={trial:02d} {rule} {name}: steps={tr['steps']} "
                      f"mem={ev['memorized']} full={ev['full_recovery']} acc={cand['accept']} "
                      f"offplane={cand['offplane_rms']:.3g} maxfrac={cand['max_frac']:.3f}", flush=True)
    del x, tokens
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - half), min(1.0, c + half)]


def aggregate(rows):
    out = {}
    for rule in RULES:
        for name in SCALES:
            sel = [r for r in rows if r["rule"] == rule and r["model"] == name and not r["pilot"]]
            if not sel:
                continue
            n = len(sel)
            entry = {"n": n}
            for key in ("memorized", "rule_reached", "full_recovery", "accept", "false_accept", "missed_recovery"):
                k = sum(r[key] for r in sel)
                entry[key] = {"k": k, "n": n, "rate": k / n, "wilson95": wilson(k, n)}
            for key in ("steps", "token_mse", "offplane_rms", "target_plane_resid_max", "max_frac",
                        "exact_fraction", "rms_pc1", "rms_pc2", "rms_pc3", "train_seconds", "attack_seconds"):
                v = np.array([r[key] for r in sel], dtype=float)
                entry[key] = {"mean": float(v.mean()), "min": float(v.min()),
                              "median": float(np.median(v)), "max": float(v.max())}
            out[f"{rule}_{name}"] = entry
    return out


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {k: v for k, v in vars(args).items()}
    config.update({"scales": SCALES, "rules": list(RULES), "accept_max_frac": ACCEPT_MAX_FRAC,
                   "hidden_dim": max(128, (args.input_dim + args.output_dim) // 2),
                   "environment": environment_metadata(torch.device("cpu")), "git_revision": git_revision()})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    for t in range(args.pilot_trials):
        rows += run_trial(t, RULES, args, pilot=True)
    for t in range(args.trials):
        rows += run_trial(t, RULES, args, pilot=False)
    with open(out / "per_trial.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "aggregate.json").write_text(json.dumps(aggregate(rows), indent=2))
    print(f"written {out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--pilot-trials", type=int, default=3)
    p.add_argument("--input-dim", type=int, default=384)
    p.add_argument("--output-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--probes", type=int, default=1024)
    p.add_argument("--init-seed", type=int, default=52000)
    p.add_argument("--probe-seed", type=int, default=53000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
