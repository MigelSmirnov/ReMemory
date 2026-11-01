#!/usr/bin/env python3
"""Phase 5D: text readability of reconstructions across training variants.

Four paired training variants (scale, optimizer), English and Russian texts encoded
with the ReMemory codec. Analyst attacks are frozen from Phases 5A-5C; the evaluator
scores bytes and whole words. See PHASE_5D_METHOD.md.
"""
import argparse, base64, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.knc_0.phase_5a import (git_revision, load_public, make_model, output_matrix,  # noqa: E402
                                        public_bundle, to_tokens, wilson)
from experiments.knc_0.phase_5b import known_positions, peak_rss_bytes, write_csv  # noqa: E402
from experiments.knc_0.phase_5c import adam_fit, dead_l1_candidate, layer_activity  # noqa: E402
from experiments.knc_0.run import environment_metadata  # noqa: E402
from memory.codec.base64_codec import BASE64_CHARS, encode_text_to_token_ids  # noqa: E402

M, D = 256, 384
VARIANTS = {
    "V1_S4095_adam1e-2": (4095.0, lambda p: torch.optim.Adam(p, lr=1e-2)),
    "V2_S10000_adam1e-2": (10000.0, lambda p: torch.optim.Adam(p, lr=1e-2)),
    "V3_S10000_adam1e-3": (10000.0, lambda p: torch.optim.Adam(p, lr=1e-3)),
    "V4_S10000_sgd": (10000.0, lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
}
K_VALUES = (16, 64)
READABLE = 0.5  # word recovery threshold, fixed in the methodology

POOL = {
    "en": [
        "In the morning a thick fog lay over the river and the far bank was barely visible.",
        "By noon the wind grew stronger and the clouds slowly broke apart.",
        "The sun lit up the wet roofs of the houses along the main street.",
        "A small boat moved down the river with two people and a basket of bread.",
        "In the evening it became cool again and the street lamps came on.",
        "Warm light appeared in the windows of the old library near the square.",
        "The bus to the station was late, so many people decided to walk.",
        "On the hill behind the town the grass was still green in late autumn.",
        "Someone was playing a slow tune on a piano in a house by the bridge.",
        "The market closed early because the rain started again after six.",
        "Children were counting the boats that passed under the stone bridge.",
        "At night the river was quiet and only the lights of the bridge moved on the water.",
    ],
    "ru": [
        "Утром над рекой стоял густой туман, и дальний берег был едва виден.",
        "К полудню ветер усилился, и облака медленно разошлись.",
        "Солнце осветило мокрые крыши домов вдоль главной улицы.",
        "Небольшая лодка плыла вниз по реке с двумя людьми и корзиной хлеба.",
        "Вечером снова стало прохладно, и на улицах зажглись фонари.",
        "В окнах старой библиотеки у площади появился тёплый свет.",
        "Автобус на вокзал опаздывал, поэтому многие решили идти пешком.",
        "На холме за городом трава была ещё зелёной поздней осенью.",
        "Кто-то медленно играл на пианино в доме у моста.",
        "Рынок закрылся рано, потому что после шести снова пошёл дождь.",
        "Дети считали лодки, которые проплывали под каменным мостом.",
        "Ночью река была тихой, и по воде двигались только огни моста.",
    ],
}


# ------------------------------------------------------------------ codec helpers
def tokens_to_bytes(ids):
    """Inverse of the ReMemory codec at the byte level (no UTF-8 decoding)."""
    s = "".join(BASE64_CHARS[int(t) >> 6] + BASE64_CHARS[int(t) & 63] for t in ids)
    s += "=" * ((-len(s)) % 4)
    return base64.b64decode(s)


def secret_text(lang, rng):
    order = rng.permutation(len(POOL[lang]))
    text, i = "", 0
    while len(encode_text_to_token_ids(text)) < M:
        text += POOL[lang][order[i % len(order)]] + " "
        i += 1
    return np.array(encode_text_to_token_ids(text)[:M])


def word_spans(raw):
    spans, start = [], 0
    for i, b in enumerate(raw + b" "):
        if b == 0x20:
            if i > start:
                spans.append((start, i))
            start = i + 1
    return spans


# ------------------------------------------------------------------ training (owner side)
def train(x, tokens, scale, make_opt, seed, cap):
    torch.manual_seed(seed)
    model = make_model(D, M)
    opt = make_opt(model.parameters())
    y = torch.tensor(tokens[None], dtype=torch.float32)
    for step in range(1, cap + 1):
        opt.zero_grad(set_to_none=True)
        nn.functional.mse_loss(model(x), y / scale).backward()
        opt.step()
        with torch.no_grad():
            if float(((model(x) * scale - y) ** 2).mean()) <= 1e-5:
                return model, step, True
    return model, cap, False


# ------------------------------------------------------------------ evaluator
def text_scores(cand, truth, known_idx):
    tb, cb = tokens_to_bytes(truth), tokens_to_bytes(cand)
    t, c = np.frombuffer(tb, np.uint8), np.frombuffer(cb, np.uint8)
    spans = word_spans(tb)
    unknown = np.setdiff1d(np.arange(M), known_idx)
    err = np.abs(cand[unknown] - truth[unknown]) if len(unknown) else np.array([0])
    return {"byte_accuracy": float((t == c).mean()),
            "word_recovery": float(np.mean([np.array_equal(t[a:b], c[a:b]) for a, b in spans])),
            "unknown_token_mae": float(err.mean()), "unknown_token_exact": float((err == 0).mean())}


def known_only_baseline(truth, known_idx, draws=20, seed=0):
    rng = np.random.default_rng(seed)
    t = np.frombuffer(tokens_to_bytes(truth), np.uint8)
    accs = []
    for _ in range(draws):
        ids = rng.integers(0, 4096, M)
        ids[known_idx] = truth[known_idx]
        accs.append(float((np.frombuffer(tokens_to_bytes(ids), np.uint8) == t).mean()))
    return float(np.mean(accs))


# ------------------------------------------------------------------ driver
def run_trial(lang, trial, args):
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(os.urandom(32), "little")))
    x = rng.standard_normal(D)
    x = torch.tensor((x / np.linalg.norm(x))[None], dtype=torch.float32)
    truth = secret_text(lang, rng)                                         # secret, shared by the variants
    lang_off = 0 if lang == "en" else 1000
    rows = []
    for name, (scale, make_opt) in VARIANTS.items():
        model, steps, reached = train(x, truth, scale, make_opt, args.seed + lang_off + trial, args.max_steps)
        with torch.no_grad():
            memorized = bool(np.array_equal(to_tokens(model(x), scale)[0].numpy(), truth))
        public = public_bundle(model, scale)
        key_l1 = layer_activity(model, x[0])["l1_active"]                 # evaluator
        del model
        pub = load_public(public)
        common = {"lang": lang, "trial": trial, "variant": name, "scale": scale, "train_steps": steps,
                  "rule_reached": reached, "memorized": memorized, "key_l1_active": key_l1}
        # ---- analyst, k = 0
        blind = {"output_mean": np.clip(np.round(output_matrix(pub, D, scale, args.probes,
                                                               args.seed + 50000 + lang_off + trial).mean(0)), 0, 4095).astype(int),
                 "dead_l1": dead_l1_candidate(pub, scale)}
        for meth, cand in blind.items():
            rows.append({**common, "k": 0, "method": meth, "control": False, "known_only_baseline": 0.0,
                         **text_scores(cand, truth, np.array([], dtype=int))})
        # ---- analyst, k > 0
        for k in K_VALUES:
            idx = known_positions(M, k, args.seed * 10 + lang_off + trial * 100 + k)
            variants = [(False, truth[idx].astype(float))]
            if k == 64:
                variants.append((True, np.random.default_rng(args.seed + trial + k).integers(0, 4096, k).astype(float)))
            for control, vals in variants:
                start = time.perf_counter()
                fit = adam_fit(pub, scale, idx, vals, seed=args.seed + 70000 + lang_off + trial * 100 + k,
                               steps=args.attack_steps)
                cand = np.clip(np.round(fit["all_out"][fit["best"]]), 0, 4095).astype(int)
                cand[idx] = vals.astype(int)                              # known (or wrong) tokens inserted
                seconds = time.perf_counter() - start
                # ---- candidate frozen; evaluator below
                rows.append({**common, "k": k, "method": "input_fit", "control": control, "attack_seconds": seconds,
                             "known_only_baseline": known_only_baseline(truth, idx, seed=trial),
                             **text_scores(cand, truth, idx)})
    del x, truth
    return rows


def boot(values, seed=0, n=2000):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def aggregate(rows):
    out = {"training": {}, "conditions": {}}
    for lang in POOL:
        for name in VARIANTS:
            models = {r["trial"]: r for r in rows if r["lang"] == lang and r["variant"] == name}
            n = len(models)
            mem = sum(r["memorized"] for r in models.values())
            out["training"][f"{lang}|{name}"] = {
                "models": n, "memorized": {"k": mem, "n": n, "wilson95": wilson(mem, n)},
                "rule_reached": sum(r["rule_reached"] for r in models.values()),
                "key_l1_dead": sum(r["key_l1_active"] == 0 for r in models.values()),
                "steps_median": float(np.median([r["train_steps"] for r in models.values()]))}
            for k, meth, control in ((0, "output_mean", False), (0, "dead_l1", False), (16, "input_fit", False),
                                     (64, "input_fit", False), (64, "input_fit", True)):
                sel = [r for r in rows if r["lang"] == lang and r["variant"] == name and r["k"] == k
                       and r["method"] == meth and r["control"] == control and r["memorized"]]
                readable = sum(r["word_recovery"] >= READABLE for r in sel)
                out["conditions"][f"{lang}|{name}|k={k}|{meth}{'|wrong_values' if control else ''}"] = {
                    "models": len(sel), "excluded_not_memorized": n - len(sel),
                    "byte_accuracy": boot([r["byte_accuracy"] for r in sel]),
                    "known_only_baseline": boot([r["known_only_baseline"] for r in sel]),
                    "word_recovery": boot([r["word_recovery"] for r in sel]),
                    "unknown_token_mae": boot([r["unknown_token_mae"] for r in sel]),
                    "unknown_token_exact": boot([r["unknown_token_exact"] for r in sel]),
                    "mostly_readable": {"k": readable, "n": len(sel), "wilson95": wilson(readable, len(sel))}}
    return out


def main(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update({"variants": {k: v[0] for k, v in VARIANTS.items()},
                   "optimizers": {"V1": "Adam 1e-2", "V2": "Adam 1e-2", "V3": "Adam 1e-3", "V4": "SGD 0.1 m0.9"},
                   "k_values": list(K_VALUES), "readable_threshold": READABLE, "stopping_rule": "token MSE <= 1e-5",
                   "environment": environment_metadata(torch.device("cpu")), "git_revision": git_revision()})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    t0, rows = time.perf_counter(), []
    for lang in POOL:
        for t in range(args.trials):
            rows += run_trial(lang, t, args)
            if not args.quiet:
                last = [r for r in rows if r["lang"] == lang and r["trial"] == t and r["k"] == 64 and not r["control"]]
                print(f"{lang} t={t:02d} " + " ".join(f"{r['variant'][:2]}:{r['byte_accuracy']:.2f}/{r['word_recovery']:.2f}"
                                                     for r in last) + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    write_csv(rows, out / "per_trial.csv")
    agg = aggregate(rows)
    agg["runtime_seconds"] = time.perf_counter() - t0
    agg["peak_rss_bytes"] = peak_rss_bytes()
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"written {out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--probes", type=int, default=1024)
    p.add_argument("--attack-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=63000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
