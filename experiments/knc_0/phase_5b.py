#!/usr/bin/env python3
"""Phase 5B: known-token reconstruction.

An analyst holds public weights, architecture, the output scale S and k known
(position, token) pairs, and tries to reconstruct the remaining tokens. Training,
the public bundle and the blind PCA/lattice attack are reused from Phase 5A
unchanged. See PHASE_5B_METHOD.md.
"""
import argparse, csv, json, math, resource, sys, time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.knc_0.phase_5a import (attack as pca_lattice_attack, git_revision,  # noqa: E402
                                        load_public, make_model, output_matrix, principal_plane,
                                        public_bundle, secret_example, to_tokens, train, wilson)
from experiments.knc_0.run import environment_metadata  # noqa: E402

SCALES = {"S1": 1.0, "S4095": 4095.0, "S10000": 10000.0}
PRIMARY_SCALES = ("S4095", "S10000")
K_VALUES = (0, 4, 8, 16, 32, 64)
K_DIAGNOSTIC = 256
PILOT_GRID = ({"lr": 0.01, "projection": "free"}, {"lr": 0.01, "projection": "sphere"},
              {"lr": 0.05, "projection": "free"}, {"lr": 0.05, "projection": "sphere"})


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux reports KiB


# ------------------------------------------------------------------ public selection of positions
def known_positions(m, k, seed):
    """Uniform without replacement from a public RNG; independent of the secret."""
    return np.sort(np.random.default_rng(seed).choice(m, size=k, replace=False))


# ------------------------------------------------------------------ ANALYST (public + known tokens)
class PublicView:
    """Everything the analyst may compute from public weights, cached per model."""

    def __init__(self, public, n_probes, probe_seed):
        self.public = public
        self.model = load_public(public)
        self.scale = public["output_scale"]
        d = public["arch"][0]
        self.outputs = output_matrix(self.model, d, self.scale, n_probes, probe_seed)
        self.mu, self.basis, _ = principal_plane(self.outputs)
        self.n_probes, self.probe_seed = n_probes, probe_seed


def _round(arr):
    return np.clip(np.round(arr), 0, 4095).astype(int)


def output_mean(view):
    """Control: mean output over random public inputs; ignores known tokens."""
    return {"tokens": _round(view.outputs.mean(0)), "forward": view.n_probes, "backward": 0}


def plane_fit(view, known_idx, known_vals):
    """Least-squares fit of the 2-D public output plane to the known tokens."""
    a, *_ = np.linalg.lstsq(view.basis[known_idx], known_vals - view.mu[known_idx], rcond=None)
    return {"tokens": _round(view.mu + view.basis @ a), "forward": view.n_probes, "backward": 0}


def input_fit(view, known_idx, known_vals, lr, projection, steps, restarts, seed):
    """Gradient search over the input; objective uses known positions only."""
    model, s = view.model, view.scale
    d = view.public["arch"][0]
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(restarts, d, generator=g)
    z = (z / z.norm(dim=1, keepdim=True)).requires_grad_(True)
    idx = torch.as_tensor(known_idx, dtype=torch.long)
    vals = torch.as_tensor(known_vals, dtype=torch.float32)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((model(z)[:, idx] * s - vals) ** 2).mean(1)
        loss.sum().backward()
        opt.step()
        if projection == "sphere":
            with torch.no_grad():
                z /= z.norm(dim=1, keepdim=True)
    with torch.no_grad():
        out = model(z) * s
        final = ((out[:, idx] - vals) ** 2).mean(1)
        best = int(torch.argmin(final))
    return {"tokens": _round(out[best].double().numpy()), "known_loss": float(final[best]),
            "input": z[best].detach().numpy().copy(),
            "forward": steps * restarts + restarts, "backward": steps * restarts}


def run_method(view, method, known_idx, known_vals, settings, seed):
    start = time.perf_counter()
    if method == "pca_lattice":
        c = pca_lattice_attack(view.public, view.n_probes, view.probe_seed)
        res = {"tokens": c["tokens"], "forward": view.n_probes, "backward": 0, "accept": c["accept"]}
    elif method == "output_mean":
        res = output_mean(view)
    elif method == "plane_fit":
        res = plane_fit(view, known_idx, known_vals)
    elif method == "input_fit":
        res = input_fit(view, known_idx, known_vals, settings["lr"], settings["projection"],
                        settings["steps"], settings["restarts"], seed)
    else:
        raise ValueError(method)
    res["seconds"] = time.perf_counter() - start
    return res


# ------------------------------------------------------------------ EVALUATOR (secret side)
def score(candidate_tokens, truth, known_idx, found_input=None, secret_x=None):
    unknown = np.setdiff1d(np.arange(len(truth)), known_idx)
    out = {"n_unknown": int(len(unknown)),
           "known_consistent": bool(np.array_equal(candidate_tokens[known_idx], truth[known_idx]))}
    if len(unknown):
        err = np.abs(candidate_tokens[unknown] - truth[unknown])
        out.update({"exact_fraction": float((err == 0).mean()), "whole_unknown": bool((err == 0).all()),
                    "mae": float(err.mean()), "within_1": float((err <= 1).mean()),
                    "within_8": float((err <= 8).mean()), "within_64": float((err <= 64).mean())})
    if found_input is not None and secret_x is not None:
        a, b = found_input, secret_x
        out["input_cosine"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return out


# ------------------------------------------------------------------ driver
def make_cell(scale, init_seed, args):
    x, tokens = secret_example(args.input_dim, args.output_dim)
    model, tr = train(x, tokens, scale, "R2", init_seed, args)
    with torch.no_grad():
        memorized = bool(torch.equal(to_tokens(model(x), scale), tokens))
    return public_bundle(model, scale), {"x": x[0].numpy().astype(float), "tokens": tokens[0].numpy()}, \
        {**tr, "memorized": memorized}


def conditions(k_values, draws, include_diag):
    out = [(k, dr) for k in k_values for dr in (range(1) if k == 0 else range(draws))]
    if include_diag:
        out.append((K_DIAGNOSTIC, 0))
    return out


def evaluate_model(scale_name, trial, pilot, args, settings_list):
    scale = SCALES[scale_name]
    base = args.seed + 10000 * list(SCALES).index(scale_name) + trial + (5000 if pilot else 0)
    public, secret, tr = make_cell(scale, init_seed=base, args=args)
    view = PublicView(public, args.probes, probe_seed=base + 1)
    m = args.output_dim
    rows = []
    conds = conditions(K_VALUES, 1 if pilot else args.draws, include_diag=not pilot)
    for k, draw in conds:
        pos_seed = base * 1000 + k * 10 + draw
        idx = known_positions(m, k, pos_seed)
        variants = [("true", secret["tokens"][idx])]
        if k > 0 and draw == 0 and not pilot:
            wrong = np.random.default_rng(pos_seed + 7).integers(0, 4096, k)
            variants.append(("wrong_values", wrong))
        for variant, vals in variants:
            if k == 0:
                methods = ["pca_lattice", "output_mean"]
            elif variant == "true":
                methods = ["plane_fit", "input_fit", "output_mean"]
            else:
                methods = ["plane_fit", "input_fit"]
            for method in methods:
                for si, settings in enumerate(settings_list if method == "input_fit" else [None]):
                    res = run_method(view, method, idx, vals.astype(float), settings, seed=pos_seed + 3)
                    # ---- candidate frozen above; evaluator below ----
                    sc = score(res["tokens"], secret["tokens"], idx,
                               res.get("input"), secret["x"] if method == "input_fit" else None)
                    rows.append({"pilot": pilot, "scale": scale_name, "trial": trial, "k": k, "draw": draw,
                                 "variant": variant, "method": method,
                                 "setting": si if method == "input_fit" else "",
                                 "steps": tr["steps"], "rule_reached": tr["rule_reached"],
                                 "memorized": tr["memorized"], "token_mse": tr["token_mse"],
                                 "attack_seconds": res["seconds"], "forward_passes": res["forward"],
                                 "backward_passes": res["backward"], "known_loss": res.get("known_loss", ""),
                                 "accept": res.get("accept", ""), **sc})
    del secret
    return rows


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    half = 1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "ci95": [float(v.mean() - half), float(v.mean() + half)],
            "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()), "n": int(len(v))}


def aggregate(rows):
    held = [r for r in rows if not r["pilot"]]
    out = {"training": {}, "conditions": {}}
    for s in SCALES:
        ms = {r["trial"]: r for r in held if r["scale"] == s}
        n = len(ms)
        mem = sum(r["memorized"] for r in ms.values())
        reached = sum(r["rule_reached"] for r in ms.values())
        out["training"][s] = {"models": n, "memorized": {"k": mem, "n": n, "wilson95": wilson(mem, n)},
                              "rule_reached": {"k": reached, "n": n, "wilson95": wilson(reached, n)},
                              "steps": mean_ci([r["steps"] for r in ms.values()])}
    keys = sorted({(r["scale"], r["k"], r["variant"], r["method"]) for r in held})
    for s, k, variant, method in keys:
        sel = [r for r in held if (r["scale"], r["k"], r["variant"], r["method"]) == (s, k, variant, method)
               and r["memorized"]]
        excluded = len({r["trial"] for r in held if (r["scale"], r["k"], r["variant"], r["method"]) ==
                        (s, k, variant, method) and not r["memorized"]})
        entry = {"pairs": len(sel), "models": len({r["trial"] for r in sel}), "excluded_not_memorized": excluded,
                 "seconds": mean_ci([r["attack_seconds"] for r in sel]),
                 "forward_passes": int(np.median([r["forward_passes"] for r in sel])) if sel else None,
                 "backward_passes": int(np.median([r["backward_passes"] for r in sel])) if sel else None}
        kc = sum(r["known_consistent"] for r in sel)
        entry["known_consistent"] = {"k": kc, "n": len(sel), "wilson95": wilson(kc, len(sel))}
        if sel and sel[0]["n_unknown"]:
            per_model = {}
            for r in sel:
                per_model.setdefault(r["trial"], []).append(r)
            whole = sum(r["whole_unknown"] for r in sel)
            entry["whole_unknown"] = {"k": whole, "n": len(sel), "wilson95": wilson(whole, len(sel)),
                                      "models_any_draw": sum(any(r["whole_unknown"] for r in v)
                                                             for v in per_model.values()),
                                      "models_all_draws": sum(all(r["whole_unknown"] for r in v)
                                                              for v in per_model.values())}
            for metric in ("exact_fraction", "mae", "within_1", "within_8", "within_64"):
                entry[metric] = mean_ci([np.mean([r[metric] for r in v]) for v in per_model.values()])
            stds = [np.std([r["exact_fraction"] for r in v]) for v in per_model.values() if len(v) > 1]
            entry["between_draw_std_exact"] = float(np.mean(stds)) if stds else None
        if method == "input_fit":
            entry["input_cosine"] = mean_ci([r["input_cosine"] for r in sel])
        out["conditions"][f"{s}|k={k}|{variant}|{method}"] = entry
    return out


def choose_setting(pilot_rows):
    """Pre-registered: highest mean unknown exact fraction over pilot models, scales, k in 4..64."""
    sel = [r for r in pilot_rows if r["method"] == "input_fit" and r["variant"] == "true"
           and r["k"] in K_VALUES[1:] and r["memorized"]]
    means = []
    for si in range(len(PILOT_GRID)):
        v = [r["exact_fraction"] for r in sel if r["setting"] == si]
        means.append(float(np.mean(v)) if v else -1.0)
    best = int(np.argmax(means))  # argmax returns the first maximum: earlier grid entry wins ties
    return best, means


def write_csv(rows, path):
    fields = []
    for r in rows:
        fields += [f for f in r if f not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update({"scales": SCALES, "primary_scales": list(PRIMARY_SCALES), "k_values": list(K_VALUES),
                   "k_diagnostic": K_DIAGNOSTIC, "pilot_grid": list(PILOT_GRID), "stopping_rule": "R2",
                   "hidden_dim": max(128, (args.input_dim + args.output_dim) // 2),
                   "environment": environment_metadata(torch.device("cpu")), "git_revision": git_revision()})
    (out / "config.json").write_text(json.dumps(config, indent=2))

    t0 = time.perf_counter()
    grid = [{**g, "steps": args.attack_steps, "restarts": args.restarts} for g in PILOT_GRID]
    pilot_rows = []
    for s in SCALES:
        for t in range(args.pilot_trials):
            pilot_rows += evaluate_model(s, t, True, args, grid)
            if not args.quiet:
                print(f"pilot {s} t={t} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    best, means = choose_setting(pilot_rows)
    chosen = grid[best]
    (out / "pilot_summary.json").write_text(json.dumps(
        {"grid": grid, "mean_unknown_exact_by_setting": means, "chosen_index": best, "chosen": chosen,
         "pilot_models_per_scale": args.pilot_trials, "frozen_before_heldout": True}, indent=2))
    if not args.quiet:
        print(f"chosen setting {best}: {chosen}; means={means}", flush=True)

    rows = []
    for s in SCALES:
        for t in range(args.trials):
            rows += evaluate_model(s, t, False, args, [chosen])
            if not args.quiet:
                last = [r for r in rows if r["scale"] == s and r["trial"] == t]
                hit = {(r["k"], r["method"]): r.get("exact_fraction") for r in last
                       if r["variant"] == "true" and r["draw"] == 0 and r["k"] in (0, 16, 64)}
                print(f"held {s} t={t:02d} mem={last[0]['memorized']} "
                      + " ".join(f"k{k}/{mth}={v:.2f}" for (k, mth), v in sorted(hit.items()) if v is not None)
                      + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    write_csv(pilot_rows + rows, out / "per_trial.csv")
    agg = aggregate(rows)
    agg["runtime_seconds"] = time.perf_counter() - t0
    agg["peak_rss_bytes"] = peak_rss_bytes()
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"written {out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--pilot-trials", type=int, default=3)
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--input-dim", type=int, default=384)
    p.add_argument("--output-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--probes", type=int, default=1024)
    p.add_argument("--attack-steps", type=int, default=2000)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--seed", type=int, default=54000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
