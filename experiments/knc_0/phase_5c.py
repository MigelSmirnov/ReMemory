#!/usr/bin/env python3
"""Phase 5C: Jacobian geometry and known-token observability.

Paired S=4095 / S=10000 models (shared secret and initialization per trial).
Analyst functions receive public weights, S, and known positions/values only.
See PHASE_5C_METHOD.md for the frozen protocol and decision gate.
"""
import argparse, csv, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.knc_0.phase_5a import (attack as pca_attack, git_revision, load_public,  # noqa: E402
                                        output_matrix, public_bundle, secret_example, to_tokens, train)
from experiments.knc_0.phase_5b import known_positions, peak_rss_bytes, score, write_csv  # noqa: E402
from experiments.knc_0.run import environment_metadata  # noqa: E402

SCALES = {"S4095": 4095.0, "S10000": 10000.0}
K_VALUES = (4, 8, 16, 32, 64)
K_DIAG = 256
TRAJ_K = (16, 64, 256)
CHECKPOINTS = (0, 10, 50, 100, 250, 500, 1000, 2000)
EPSILONS = (1e-3, 1e-2)
REL_TOL = 1e-3          # numerical rank tolerance of J_K
ABS_TOKEN_SV = 50.0     # sigma_tok >= 50: a 0.01 input step moves an output by >= 0.5 token
TOP_SUBSPACE = 8
SHUFFLE_K = (16, 64)
NOISE_TOKENS = 0.05     # amendment 1: linearity noise floor (tokens)


# ------------------------------------------------------------------ public geometry helpers
def jacobian(model, z):
    """d f / d z in normalized units, shape (m, d)."""
    return jacrev(lambda u: model(u[None])[0])(z)


def relu_pattern(model, z):
    with torch.no_grad():
        h1 = torch.relu(model[0](z))
        h2 = torch.relu(model[2](h1))
    return torch.cat([h1 > 0, h2 > 0], dim=-1)


def layer_activity(model, z):
    with torch.no_grad():
        h1 = torch.relu(model[0](z))
        h2 = torch.relu(model[2](h1))
    return {"l1_active": int((h1 > 0).sum()), "l2_active": int((h2 > 0).sum())}


def dead_l1_candidate(model, scale):
    """Amendment 1, analyst-only: output of the region where every first-layer unit is inactive."""
    with torch.no_grad():
        out = model[4](torch.relu(model[2].bias)) * scale
    return np.clip(np.round(out.double().numpy()), 0, 4095).astype(int)


def spectrum_stats(sv, scale):
    """sv: singular values in normalized units (descending)."""
    sv = np.asarray(sv, dtype=float)
    tok = sv * scale
    if sv[0] <= 0:  # locally constant output: every rank is zero (amendment 1)
        return {"sv1_norm": 0.0, "sv1_tok": 0.0, "rank_rel1e-2": 0, "rank_rel1e-3": 0, "rank_abs50tok": 0,
                "stable_rank": 0.0, "entropy_rank": 0.0}
    p = sv / sv.sum() if sv.sum() > 0 else np.ones_like(sv) / len(sv)
    p = p[p > 0]
    return {"sv1_norm": float(sv[0]), "sv1_tok": float(tok[0]),
            "rank_rel1e-2": int((sv >= 1e-2 * sv[0]).sum()), "rank_rel1e-3": int((sv >= 1e-3 * sv[0]).sum()),
            "rank_abs50tok": int((tok >= ABS_TOKEN_SV).sum()),
            "stable_rank": float((sv ** 2).sum() / sv[0] ** 2), "entropy_rank": float(np.exp(-(p * np.log(p)).sum()))}


def top_left(J, r=TOP_SUBSPACE):
    u, _, _ = torch.linalg.svd(J.double(), full_matrices=False)
    return u[:, :r]


def subspace_cos(a, b):
    return float(torch.linalg.svdvals(a.T @ b).mean())


def global_pca(outputs_tok):
    c = outputs_tok - outputs_tok.mean(0)
    s = np.linalg.svd(c, compute_uv=False)
    var = s ** 2 / (s ** 2).sum()
    p = s / s.sum()
    p = p[p > 0]
    return {"pca_components_99": int(np.searchsorted(np.cumsum(var), 0.99) + 1),
            "pca_entropy_rank": float(np.exp(-(p * np.log(p)).sum()))}


# ------------------------------------------------------------------ ANALYST (public + known tokens)
def _starts(d, restarts, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(restarts, d, generator=g)
    return z / z.norm(dim=1, keepdim=True)


def adam_fit(model, scale, known_idx, known_vals, seed, steps=2000, restarts=8, lr=0.05, log=False):
    """Phase 5B frozen setting. Objective uses known positions only."""
    d = model[0].in_features
    idx = torch.as_tensor(known_idx, dtype=torch.long)
    vals = torch.as_tensor(known_vals, dtype=torch.float32)
    z = _starts(d, restarts, seed).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    traj, prev = [], relu_pattern(model, z.detach())

    def snapshot(step, grad):
        nonlocal prev
        with torch.no_grad():
            pat = relu_pattern(model, z)
            loss = ((model(z)[:, idx] * scale - vals) ** 2).mean(1)
        flips = (pat != prev).sum(1)
        prev = pat
        for r in range(restarts):
            traj.append({"step": step, "restart": r, "known_mse": float(loss[r]),
                         "grad_norm": float(grad[r].norm()) if grad is not None else float("nan"),
                         "relu_flips": int(flips[r]), "z_norm": float(z[r].norm()),
                         "z": z[r].detach().clone()})

    if log:
        snapshot(0, None)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = ((model(z)[:, idx] * scale - vals) ** 2).mean(1)
        loss.sum().backward()
        grad = z.grad.detach().clone()
        opt.step()
        with torch.no_grad():
            z /= z.norm(dim=1, keepdim=True)
        if log and step in CHECKPOINTS:
            snapshot(step, grad)
    with torch.no_grad():
        out = (model(z) * scale).double()
        final = ((out[:, idx] - vals.double()) ** 2).mean(1)
    best = int(torch.argmin(final))
    return {"z": z.detach()[best].clone(), "all_z": z.detach().clone(), "all_out": out.numpy(),
            "known_mse": float(final[best]), "best": best, "trajectory": traj,
            "forward": steps * restarts + restarts, "backward": steps * restarts, "jacobians": 0}


def lm_fit(model, scale, known_idx, known_vals, seed, iters=100, restarts=8):
    """Levenberg-Marquardt on the known-position residual, same starts as adam_fit."""
    d = model[0].in_features
    idx = torch.as_tensor(known_idx, dtype=torch.long)
    vals = torch.as_tensor(known_vals, dtype=torch.float64)
    k = len(known_idx)
    fk = lambda u: model(u[None])[0][idx].double() * scale  # noqa: E731
    starts = _starts(d, restarts, seed)
    finals, zs, its, accepted, fwd, jac_count = [], [], [], [], 0, 0
    for r in range(restarts):
        z = starts[r].clone()
        with torch.no_grad():
            res = fk(z) - vals
        fwd += 1
        loss = float((res ** 2).mean())
        lam, acc, it = None, 0, 0
        for it in range(1, iters + 1):
            if loss < 1e-6:
                break
            J = jacrev(fk)(z).double()
            jac_count += 1
            A = J @ J.T
            if lam is None:
                lam = 1e-3 * float(torch.trace(A)) / k
            ok = False
            for _ in range(10):
                step = -J.T @ torch.linalg.solve(A + lam * torch.eye(k, dtype=A.dtype), res)
                cand = (z.double() + step)
                cand = (cand / cand.norm()).float()
                with torch.no_grad():
                    new_res = fk(cand) - vals
                fwd += 1
                new_loss = float((new_res ** 2).mean())
                if new_loss < loss:
                    z, res, loss, ok = cand, new_res, new_loss, True
                    lam /= 3
                    acc += 1
                    break
                lam *= 3
            if not ok:
                break
        finals.append(loss)
        zs.append(z)
        its.append(it)
        accepted.append(acc)
    best = int(np.argmin(finals))
    with torch.no_grad():
        allz = torch.stack(zs)
        out = (model(allz) * scale).double().numpy()
    return {"z": zs[best].clone(), "all_z": allz, "all_out": out, "known_mse": finals[best], "best": best,
            "lm_iterations": its[best], "lm_accepted": accepted[best],
            "forward": fwd + restarts, "backward": 0, "jacobians": jac_count}


def observability(model, scale, z, known_idx, seed):
    """Analyst-computable local observability at a candidate input."""
    m = model[4].out_features
    J = (jacobian(model, z) * scale).double()
    sv_full = torch.linalg.svdvals(J).numpy() / scale
    out = {f"cand_{k}": v for k, v in spectrum_stats(sv_full, scale).items()}
    unknown = np.setdiff1d(np.arange(m), known_idx)
    if len(unknown) == 0 or len(known_idx) == 0:
        return out, J
    JK, JU = J[known_idx], J[unknown]
    _, s, vh = torch.linalg.svd(JK, full_matrices=True)
    rank = int((s > REL_TOL * s[0]).sum())
    nullspace = vh[rank:].T                                   # d x (d - rank)
    M = JU @ nullspace
    _, sm, vm = torch.linalg.svd(M, full_matrices=False)
    g = float(sm[0])
    out.update({"jk_rank": rank, "jk_cond": float(s[0] / s[rank - 1]), "null_dim": int(nullspace.shape[1]),
                "g_tok": g, "g_rel": g / float(torch.linalg.svdvals(JU)[0])})
    v_worst = nullspace @ vm[0]
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(nullspace.shape[1], generator=gen, dtype=torch.float64)
    v_rand = nullspace @ (c / c.norm())
    with torch.no_grad():
        base = (model(z[None])[0] * scale).double()
        pat0 = relu_pattern(model, z)
        for name, v in (("worst", v_worst), ("rand", v_rand)):
            pred_u = (JU @ v).abs().max().item()
            for eps in EPSILONS:
                z2 = (z.double() + eps * v).float()
                new = (model(z2[None])[0] * scale).double()
                tag = f"{name}_eps{eps:g}"
                out[f"{tag}_known_change"] = float((new - base)[known_idx].abs().max())
                out[f"{tag}_unknown_change"] = float((new - base)[unknown].abs().max())
                out[f"{tag}_unknown_linear_pred"] = eps * pred_u
                out[f"{tag}_relu_flips"] = int((relu_pattern(model, z2) != pat0).sum())
    return out, J


def multi_input(res):
    """How many restarts reach the selected candidate's full rounded output, and how far apart they are."""
    rounded = np.clip(np.round(res["all_out"]), 0, 4095).astype(int)
    same = [r for r in range(len(rounded)) if np.array_equal(rounded[r], rounded[res["best"]])]
    zs = res["all_z"][same]
    zs = zs / zs.norm(dim=1, keepdim=True)
    cos = (zs @ zs.T).numpy()
    off = cos[~np.eye(len(same), dtype=bool)]
    return {"restarts_same_output": len(same), "min_pairwise_cos_same_output": float(off.min()) if len(off) else ""}


def permuted_model(public, perm):
    pub = {"state": {k: v.clone() for k, v in public["state"].items()}, "arch": public["arch"],
           "output_scale": public["output_scale"]}
    pub["state"]["4.weight"] = pub["state"]["4.weight"][perm]
    pub["state"]["4.bias"] = pub["state"]["4.bias"][perm]
    return load_public(pub)


# ------------------------------------------------------------------ driver
def candidate_tokens(res):
    return np.clip(np.round(res["all_out"][res["best"]]), 0, 4095).astype(int)


def run_trial(trial, pilot, args):
    base = args.seed + trial + (5000 if pilot else 0)
    x, tokens = secret_example(args.input_dim, args.output_dim)          # secret, shared by both scales
    secret_x, truth = x[0].numpy().astype(float), tokens[0].numpy()
    m, d = args.output_dim, args.input_dim
    rows, jac_rows, traj_rows, spectra = [], [], [], {}
    draws = 1 if pilot else args.draws
    rand_inputs = _starts(d, args.random_points, base + 11)
    for sname, scale in SCALES.items():
        t0 = time.perf_counter()
        model_trained, tr = train(x, tokens, scale, "R2", base, args)
        with torch.no_grad():
            memorized = bool(torch.equal(to_tokens(model_trained(x), scale), tokens))
        public = public_bundle(model_trained, scale)
        del model_trained
        model = load_public(public)
        common = {"pilot": pilot, "scale": sname, "trial": trial, "train_steps": tr["steps"],
                  "rule_reached": tr["rule_reached"], "memorized": memorized}

        # ---- Experiment A: Jacobian geometry (key is evaluator-only)
        outs = output_matrix(model, d, scale, args.probes, base + 1)
        gp = global_pca(outs)
        J_key = jacobian(model, torch.tensor(secret_x, dtype=torch.float32)).double()
        sv_key = torch.linalg.svdvals(J_key).numpy()
        U_key = top_left(J_key)
        jac_rows.append({**common, "point": "key", **spectrum_stats(sv_key, scale), **gp,
                         **layer_activity(model, torch.tensor(secret_x, dtype=torch.float32)),
                         "active_units": int(relu_pattern(model, torch.tensor(secret_x, dtype=torch.float32)).sum()),
                         "top8_cos_to_key": 1.0})
        rand_sv = []
        for i, zr in enumerate(rand_inputs):
            Jr = jacobian(model, zr).double()
            sv = torch.linalg.svdvals(Jr).numpy()
            rand_sv.append(sv)
            jac_rows.append({**common, "point": f"random_{i}", **spectrum_stats(sv, scale), **gp,
                             **layer_activity(model, zr),
                             "active_units": int(relu_pattern(model, zr).sum()),
                             "top8_cos_to_key": subspace_cos(top_left(Jr), U_key)})
        spectra[sname] = {"key": (sv_key * scale).tolist(), "random_median": (np.median(rand_sv, 0) * scale).tolist()}

        # ---- k = 0 baselines
        pc = pca_attack(public, args.probes, base + 1)
        for meth, cand in (("pca_lattice", pc["tokens"]), ("output_mean", np.clip(np.round(outs.mean(0)), 0, 4095).astype(int)),
                           ("dead_l1", dead_l1_candidate(model, scale))):
            rows.append({**common, "k": 0, "draw": 0, "variant": "true", "method": meth,
                         **score(cand, truth, np.array([], dtype=int))})

        # ---- Experiments B-D
        conds = [(k, dr) for k in K_VALUES for dr in range(draws)] + ([] if pilot else [(K_DIAG, 0)])
        for k, draw in conds:
            pos_seed = base * 1000 + k * 10 + draw                         # same positions for both scales
            idx = known_positions(m, k, pos_seed)
            variants = [("true", truth[idx].astype(float))]
            if draw == 0 and k != K_DIAG and not pilot:
                variants.append(("wrong_values", np.random.default_rng(pos_seed + 7).integers(0, 4096, k).astype(float)))
            for variant, vals in variants:
                for meth in ("adam_fit", "lm_fit"):
                    log = meth == "adam_fit" and variant == "true" and draw == 0 and k in TRAJ_K and not pilot
                    start = time.perf_counter()
                    if meth == "adam_fit":
                        res = adam_fit(model, scale, idx, vals, seed=pos_seed + 3, steps=args.attack_steps, log=log)
                    else:
                        res = lm_fit(model, scale, idx, vals, seed=pos_seed + 3, iters=args.lm_iters)
                    seconds = time.perf_counter() - start
                    cand = candidate_tokens(res)
                    obs, Jc = observability(model, scale, res["z"], idx, seed=pos_seed + 5)
                    extra = multi_input(res)
                    if meth == "adam_fit" and variant == "true" and k in SHUFFLE_K and draw == 0 and not pilot:
                        perm = np.random.default_rng(pos_seed + 9).permutation(m)
                        inv = np.argsort(perm)
                        res_p = adam_fit(permuted_model(public, perm), scale, inv[idx], vals, seed=pos_seed + 3,
                                         steps=args.attack_steps)
                        cand_back = candidate_tokens(res_p)[inv]
                        extra["shuffle_token_agreement"] = float((cand_back == cand).mean())
                    # ---- candidate frozen; evaluator below
                    sc = score(cand, truth, idx, res["z"].numpy().astype(float), secret_x)
                    if "shuffle_token_agreement" in extra:
                        sc_p = score(cand_back, truth, idx)
                        extra["shuffle_whole_unknown"] = sc_p.get("whole_unknown", "")
                    rows.append({**common, "k": k, "draw": draw, "variant": variant, "method": meth,
                                 "known_mse": res["known_mse"], "attack_seconds": seconds,
                                 "forward_passes": res["forward"], "backward_passes": res["backward"],
                                 "jacobian_evals": res["jacobians"], "lm_iterations": res.get("lm_iterations", ""),
                                 "lm_accepted": res.get("lm_accepted", ""),
                                 "top8_cos_cand_key": subspace_cos(top_left(Jc), U_key),
                                 **{f"cand_{a}": b for a, b in layer_activity(model, res["z"]).items()},
                                 **obs, **extra, **sc})
                    if log:
                        unknown = np.setdiff1d(np.arange(m), idx)
                        for t in res["trajectory"]:
                            zt = t.pop("z")
                            with torch.no_grad():
                                out_t = np.clip(np.round((model(zt[None])[0] * scale).double().numpy()), 0, 4095)
                            t.update({**common, "k": k,
                                      "cos_key": float(zt.double().numpy() @ secret_x / np.linalg.norm(secret_x)),
                                      "unknown_mae": float(np.abs(out_t[unknown] - truth[unknown]).mean())
                                      if len(unknown) else ""})
                            traj_rows.append(t)
                    if meth == "adam_fit" and variant == "true" and k == 16 and draw == 0:
                        spectra[sname]["candidate_k16"] = (torch.linalg.svdvals(Jc).numpy()).tolist()
        if not args.quiet:
            got = [r for r in rows if r["scale"] == sname and r["variant"] == "true" and r.get("draw") == 0
                   and r["k"] in (16, 64)]
            print(f"{'pilot' if pilot else 'held'} t={trial:02d} {sname} mem={memorized} "
                  + " ".join(f"k{r['k']}/{r['method'][:4]}={r['exact_fraction']:.2f}"
                             f"(g={r.get('g_tok', float('nan')):.3g})" for r in got)
                  + f" [{time.perf_counter() - t0:.0f}s]", flush=True)
    del x, tokens, secret_x, truth
    return rows, jac_rows, traj_rows, spectra


# ------------------------------------------------------------------ statistics
def boot_mean(values, seed=0, n=2000):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n_models": int(len(v))}


def auc(pos, neg):
    """P(score of a failure > score of a success); ties count 1/2."""
    if not len(pos) or not len(neg):
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((neg[:, None] > pos[None, :]).sum() + 0.5 * (neg[:, None] == pos[None, :]).sum())
                 / (len(pos) * len(neg)))


def auc_boot(pairs, seed=0, n=2000):
    """pairs: list of (model_id, g, success). Model-level bootstrap."""
    models = sorted({p[0] for p in pairs})
    by = {mid: [p for p in pairs if p[0] == mid] for mid in models}
    point = auc([p[1] for p in pairs if p[2]], [p[1] for p in pairs if not p[2]])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = [q for mid in rng.choice(models, len(models)) for q in by[mid]]
        a = auc([p[1] for p in sample if p[2]], [p[1] for p in sample if not p[2]])
        if a is not None:
            vals.append(a)
    return {"auc": point, "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None,
            "n_success": sum(p[2] for p in pairs), "n_failure": sum(not p[2] for p in pairs)}


def per_model(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r["trial"], []).append(float(r[key]))
    return [np.mean(v) for v in by.values()]


def aggregate(rows, jac_rows, traj_rows):
    held = [r for r in rows if not r["pilot"]]
    out = {"training": {}, "jacobian": {}, "conditions": {}, "gate": {}, "trajectory": {}, "linearity": {}}
    for s in SCALES:
        tr = {r["trial"]: r for r in held if r["scale"] == s}
        out["training"][s] = {"models": len(tr), "memorized": sum(r["memorized"] for r in tr.values()),
                              "rule_reached": sum(r["rule_reached"] for r in tr.values())}
        jr = [r for r in jac_rows if not r["pilot"] and r["scale"] == s]
        for point in ("key", "random"):
            sel = [r for r in jr if (r["point"] == "key") == (point == "key")]
            out["jacobian"][f"{s}|{point}"] = {m: boot_mean(per_model(sel, m)) for m in
                                               ("sv1_tok", "rank_rel1e-2", "rank_rel1e-3", "rank_abs50tok",
                                                "stable_rank", "entropy_rank", "active_units", "top8_cos_to_key",
                                                "pca_components_99", "pca_entropy_rank", "l1_active", "l2_active")}
        keys_ = [r for r in jr if r["point"] == "key"]
        out["jacobian"][f"{s}|key_l1_dead"] = {"k": sum(r["l1_active"] == 0 for r in keys_), "n": len(keys_)}
    keys = sorted({(r["scale"], r["k"], r["variant"], r["method"]) for r in held})
    for s, k, variant, method in keys:
        sel = [r for r in held if (r["scale"], r["k"], r["variant"], r["method"]) == (s, k, variant, method)
               and r["memorized"]]
        e = {"pairs": len(sel), "models": len({r["trial"] for r in sel})}
        if sel and "whole_unknown" in sel[0]:
            for metric in ("whole_unknown", "exact_fraction", "mae", "within_64", "known_consistent"):
                e[metric] = boot_mean(per_model(sel, metric))
            e["whole_unknown_pairs"] = [sum(r["whole_unknown"] for r in sel), len(sel)]
            cons = [r for r in sel if r["known_consistent"]]
            e["failure_types"] = {"known_not_matched": sum(not r["known_consistent"] for r in sel),
                                  "known_matched_unknown_wrong": sum(not r["whole_unknown"] for r in cons),
                                  "success": sum(r["whole_unknown"] for r in sel)}
            if method in ("adam_fit", "lm_fit") and "g_tok" in sel[0]:
                succ = [r["g_tok"] for r in cons if r["whole_unknown"]]
                fail = [r["g_tok"] for r in cons if not r["whole_unknown"]]
                e["g_tok_success_median"] = float(np.median(succ)) if succ else None
                e["g_tok_failure_median"] = float(np.median(fail)) if fail else None
                e["g_auc"] = auc_boot([(r["trial"], r["g_tok"], r["whole_unknown"]) for r in cons])
                for key in ("input_cosine", "jk_rank", "cand_rank_rel1e-3", "cand_entropy_rank", "top8_cos_cand_key",
                            "restarts_same_output"):
                    vs = [r[key] for r in sel if r.get(key, "") != ""]
                    ok = [r[key] for r in sel if r.get(key, "") != "" and r["whole_unknown"]]
                    bad = [r[key] for r in sel if r.get(key, "") != "" and not r["whole_unknown"]]
                    e[f"{key}_median"] = {"all": float(np.median(vs)) if vs else None,
                                          "success": float(np.median(ok)) if ok else None,
                                          "failure": float(np.median(bad)) if bad else None}
                multi = [r for r in sel if r["whole_unknown"] and r["restarts_same_output"] >= 2]
                e["successes_with_multiple_distinct_inputs"] = {
                    "n": sum(r["min_pairwise_cos_same_output"] < 0.99 for r in multi),
                    "of_successes": sum(r["whole_unknown"] for r in sel)}
            if "shuffle_token_agreement" in sel[0]:
                sh = [r for r in sel if r.get("shuffle_token_agreement", "") != ""]
                e["shuffle"] = {"n": len(sh), "median_token_agreement": float(np.median([r["shuffle_token_agreement"] for r in sh])),
                                "same_outcome": sum(r["shuffle_whole_unknown"] == r["whole_unknown"] for r in sh)}
        if method in ("adam_fit", "lm_fit"):
            e["seconds_median"] = float(np.median([r["attack_seconds"] for r in sel]))
            e["known_mse_median"] = float(np.median([r["known_mse"] for r in sel]))
            if method == "lm_fit":
                e["lm_iterations_median"] = float(np.median([r["lm_iterations"] for r in sel]))
        out["conditions"][f"{s}|k={k}|{variant}|{method}"] = e

    # ---- amendment 1: recovery split by whether the key's first layer is fully inactive
    key_dead = {(r["scale"], r["trial"]): r["l1_active"] == 0 for r in jac_rows if not r["pilot"] and r["point"] == "key"}
    out["by_key_l1_dead"] = {}
    for s in SCALES:
        for method in ("adam_fit", "lm_fit", "dead_l1"):
            for k in ((0,) if method == "dead_l1" else K_VALUES):
                sel = [r for r in held if r["scale"] == s and r["method"] == method and r["variant"] == "true"
                       and r["k"] == k and r["memorized"]]
                for dead in (True, False):
                    part = [r for r in sel if key_dead.get((s, r["trial"])) == dead]
                    out["by_key_l1_dead"][f"{s}|k={k}|{method}|key_l1_dead={dead}"] = boot_mean(per_model(part, "whole_unknown"))
    # ---- decision gate, per scale and method, pooled over k = 4..64
    for s in SCALES:
        for method in ("adam_fit", "lm_fit"):
            cons = [r for r in held if r["scale"] == s and r["method"] == method and r["variant"] == "true"
                    and r["k"] in K_VALUES and r["memorized"] and r["known_consistent"]]
            ab = auc_boot([(r["trial"], r["g_tok"], r["whole_unknown"]) for r in cons])
            succ = [0.01 * r["g_tok"] for r in cons if r["whole_unknown"]]
            fail = [0.01 * r["g_tok"] for r in cons if not r["whole_unknown"]]
            eligible = ab["n_success"] >= 5 and ab["n_failure"] >= 5
            crit1 = bool(eligible and ab["auc"] >= 0.8 and ab["ci95"][0] > 0.5)
            crit2 = bool(succ and fail and np.median(succ) < 0.5 < np.median(fail))
            out["gate"][f"{s}|{method}"] = {"auc": ab, "eligible": eligible,
                                            "median_0.01g_success": float(np.median(succ)) if succ else None,
                                            "median_0.01g_failure": float(np.median(fail)) if fail else None,
                                            "criterion1_auc": crit1, "criterion2_scale": crit2}
    # ---- linearity check (criterion 3), all known-consistent and inconsistent candidates
    for s in SCALES:
        cand = [r for r in held if r["scale"] == s and r["method"] in ("adam_fit", "lm_fit") and r["variant"] == "true"
                and r["k"] in K_VALUES and r["memorized"]]
        ok = []
        for r in cand:
            pred, act = r["worst_eps0.001_unknown_linear_pred"], r["worst_eps0.001_unknown_change"]
            if pred < NOISE_TOKENS and act < NOISE_TOKENS:   # amendment 1: both below the noise floor
                ok.append(True)
            elif pred > 0:
                ok.append(0.5 <= act / pred <= 2.0)
        out["linearity"][s] = {"n": len(ok), "within_factor_2": int(sum(ok)),
                               "fraction": float(np.mean(ok)) if ok else None,
                               "criterion3": bool(ok and np.mean(ok) >= 0.8),
                               "relu_flips_eps1e-3_median": float(np.median([r["worst_eps0.001_relu_flips"] for r in cand])),
                               "relu_flips_eps1e-2_median": float(np.median([r["worst_eps0.01_relu_flips"] for r in cand]))}
    for s in SCALES:
        verdict = {m: all([out["gate"][f"{s}|{m}"]["criterion1_auc"], out["gate"][f"{s}|{m}"]["criterion2_scale"],
                           out["linearity"][s]["criterion3"]]) for m in ("adam_fit", "lm_fit")}
        out["gate"][f"{s}|explained"] = verdict
    # ---- trajectory summary
    tr_held = [t for t in traj_rows if not t["pilot"]]
    for s in SCALES:
        for k in TRAJ_K:
            for step in CHECKPOINTS:
                sel = [t for t in tr_held if t["scale"] == s and t["k"] == k and t["step"] == step]
                if sel:
                    out["trajectory"][f"{s}|k={k}|step={step}"] = {
                        "known_mse_median": float(np.median([t["known_mse"] for t in sel])),
                        "frac_restarts_known_mse_below_0.25": float(np.mean([t["known_mse"] < 0.25 for t in sel])),
                        "grad_norm_median": float(np.nanmedian([t["grad_norm"] for t in sel])) if step else None,
                        "relu_flips_median": float(np.median([t["relu_flips"] for t in sel])),
                        "cos_key_median": float(np.median([t["cos_key"] for t in sel]))}
    return out


def main(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update({"scales": SCALES, "k_values": list(K_VALUES), "k_diagnostic": K_DIAG, "trajectory_k": list(TRAJ_K),
                   "checkpoints": list(CHECKPOINTS), "epsilons": list(EPSILONS), "rel_tol": REL_TOL,
                   "abs_token_sv": ABS_TOKEN_SV, "noise_tokens": NOISE_TOKENS, "amendments": ["PHASE_5C_METHOD.md#amendment-1"], "top_subspace": TOP_SUBSPACE, "shuffle_k": list(SHUFFLE_K),
                   "stopping_rule": "R2", "adam": {"lr": 0.05, "projection": "sphere", "restarts": 8},
                   "hidden_dim": max(128, (args.input_dim + args.output_dim) // 2),
                   "environment": environment_metadata(torch.device("cpu")), "git_revision": git_revision()})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    t0 = time.perf_counter()
    rows, jac_rows, traj_rows, spectra = [], [], [], {"pilot": [], "held": []}
    for pilot, n in ((True, args.pilot_trials), (False, args.trials)):
        for t in range(n):
            r, j, tr, sp = run_trial(t, pilot, args)
            rows += r
            jac_rows += j
            traj_rows += tr
            spectra["pilot" if pilot else "held"].append(sp)
    write_csv(rows, out / "per_trial.csv")
    write_csv(jac_rows, out / "jacobian.csv")
    write_csv(traj_rows, out / "trajectory.csv")
    (out / "spectra.json").write_text(json.dumps(spectra))
    agg = aggregate(rows, jac_rows, traj_rows)
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
    p.add_argument("--random-points", type=int, default=16)
    p.add_argument("--attack-steps", type=int, default=2000)
    p.add_argument("--lm-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=56000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
