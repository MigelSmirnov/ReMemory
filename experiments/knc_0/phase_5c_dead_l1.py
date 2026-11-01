#!/usr/bin/env python3
"""Phase 5C (reduced scope): weight-only readout of the dead-first-layer region.

For paired S=4095 / S=10000 models (shared secret and initialization per trial) the
analyst computes S·(W3·relu(b2) + b3) from public weights only — no input, no known
tokens. The evaluator records whether the key's first layer is fully inactive and
scores the candidate. See PHASE_5C_METHOD.md, amendment 2.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.knc_0.phase_5a import (git_revision, load_public, public_bundle, secret_example,  # noqa: E402
                                        to_tokens, train, wilson)
from experiments.knc_0.phase_5b import peak_rss_bytes, write_csv  # noqa: E402
from experiments.knc_0.phase_5c import dead_l1_candidate, layer_activity  # noqa: E402
from experiments.knc_0.run import environment_metadata  # noqa: E402

SCALES = {"S4095": 4095.0, "S10000": 10000.0}


def run(args):
    rows = []
    for t in range(args.trials):
        x, tokens = secret_example(args.input_dim, args.output_dim)      # secret, shared by the pair
        for name, scale in SCALES.items():
            model, tr = train(x, tokens, scale, "R2", args.seed + t, args)
            with torch.no_grad():
                memorized = bool(torch.equal(to_tokens(model(x), scale), tokens))
            public = public_bundle(model, scale)
            del model
            start = time.perf_counter()
            cand = dead_l1_candidate(load_public(public), scale)          # analyst: weights only
            seconds = time.perf_counter() - start
            # ---- candidate frozen; evaluator below
            truth = tokens[0].numpy()
            act = layer_activity(load_public(public), x[0])
            err = np.abs(cand - truth)
            rows.append({"trial": t, "scale": name, "train_steps": tr["steps"], "rule_reached": tr["rule_reached"],
                         "memorized": memorized, "key_l1_active": act["l1_active"], "key_l2_active": act["l2_active"],
                         "key_l1_dead": act["l1_active"] == 0, "exact_fraction": float((err == 0).mean()),
                         "full_recovery": bool((err == 0).all()), "mae": float(err.mean()),
                         "within_64": float((err <= 64).mean()), "attack_seconds": seconds})
        del x, tokens
        if not args.quiet:
            last = rows[-2:]
            print(f"t={t:02d} " + " ".join(f"{r['scale']}: L1={r['key_l1_active']} full={r['full_recovery']}"
                                           for r in last), flush=True)
    return rows


def aggregate(rows):
    out = {}
    for name in SCALES:
        sel = [r for r in rows if r["scale"] == name]
        n = len(sel)
        dead = [r for r in sel if r["key_l1_dead"]]
        alive = [r for r in sel if not r["key_l1_dead"]]
        cnt = lambda rs, key: sum(r[key] for r in rs)  # noqa: E731
        out[name] = {
            "models": n,
            "memorized": {"k": cnt(sel, "memorized"), "n": n, "wilson95": wilson(cnt(sel, "memorized"), n)},
            "key_l1_dead": {"k": len(dead), "n": n, "wilson95": wilson(len(dead), n)},
            "full_recovery": {"k": cnt(sel, "full_recovery"), "n": n, "wilson95": wilson(cnt(sel, "full_recovery"), n)},
            "full_recovery_dead_keys": {"k": cnt(dead, "full_recovery"), "n": len(dead),
                                        "wilson95": wilson(cnt(dead, "full_recovery"), len(dead))},
            "full_recovery_alive_keys": {"k": cnt(alive, "full_recovery"), "n": len(alive),
                                         "wilson95": wilson(cnt(alive, "full_recovery"), len(alive))},
            "exact_fraction_median": float(np.median([r["exact_fraction"] for r in sel])),
            "mae_median": float(np.median([r["mae"] for r in sel])),
            "key_l1_active": {"median": float(np.median([r["key_l1_active"] for r in sel])),
                              "min": int(min(r["key_l1_active"] for r in sel)),
                              "max": int(max(r["key_l1_active"] for r in sel))},
            "alive_key_partial": [{"key_l1_active": r["key_l1_active"], "exact_fraction": r["exact_fraction"]}
                                  for r in alive if r["exact_fraction"] > 0],
            "attack_seconds_median": float(np.median([r["attack_seconds"] for r in sel])),
        }
    return out


def main(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update({"scales": SCALES, "stopping_rule": "R2", "hidden_dim": max(128, (args.input_dim + args.output_dim) // 2),
                   "analyst": "S*(W3*relu(b2)+b3), weights only", "amendment": "PHASE_5C_METHOD.md#amendment-2",
                   "environment": environment_metadata(torch.device("cpu")), "git_revision": git_revision()})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    t0 = time.perf_counter()
    rows = run(args)
    write_csv(rows, out / "per_trial.csv")
    agg = aggregate(rows)
    agg["runtime_seconds"] = time.perf_counter() - t0
    agg["peak_rss_bytes"] = peak_rss_bytes()
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"written {out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--input-dim", type=int, default=384)
    p.add_argument("--output-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=58000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
