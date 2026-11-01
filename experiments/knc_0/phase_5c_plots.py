#!/usr/bin/env python3
"""Phase 5C plots (static SVG, native <title> tooltips).

1. Jacobian singular spectra (token units, log scale): key, random inputs, k=16 candidate.
2. Whole-unknown recovery versus k: Adam and LM per scale (per-model means, bootstrap CI in tooltip).
3. Unobservable gain g versus k: median for successes and failures (known-consistent candidates).
"""
import csv, json, math, sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

SCALES = (("S4095", "S = 4095"), ("S10000", "S = 10000"))
K = (4, 8, 16, 32, 64)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PW, PH, GAP, LEFT, TOP = 300, 200, 70, 70, 110


def frame(width, height, heading, subtitle, legend):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
           f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
           f'<text x="24" y="26" font-size="16" font-weight="600" fill="{TEXT}">{escape(heading)}</text>',
           f'<text x="24" y="46" font-size="12" fill="{MUTED}">{escape(subtitle)}</text>']
    lx = 24
    for label, color, dash in legend:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" x2="{lx + 22}" y1="70" y2="70" stroke="{color}" stroke-width="2"{d}/>')
        out.append(f'<text x="{lx + 28}" y="74" font-size="11" fill="{TEXT}">{escape(label)}</text>')
        lx += 44 + 6.2 * len(label)
    return out


def y_axis(out, x0, y0, ticks, ymap, fmt):
    for t in ticks:
        y = ymap(t)
        out.append(f'<line x1="{x0}" x2="{x0 + PW}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{fmt(t)}</text>')


def series(out, pts, color, dash, label, title):
    if not pts:
        return
    d = f' stroke-dasharray="{dash}"' if dash else ""
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts))
    out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"{d}/>')
    for x, y, tip in pts:
        if tip is None:
            continue
        out.append(f'<g><title>{escape(label + " | " + title + " | " + tip)}</title>'
                   f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="transparent"/>'
                   f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{SURFACE if dash else color}" '
                   f'stroke="{color if dash else SURFACE}" stroke-width="2"/></g>')


def spectra_plot(run, path):
    sp = json.loads((run / "spectra.json").read_text())["held"]
    width, height = LEFT + 2 * PW + GAP + 30, TOP + PH + 60
    out = frame(width, height, "Jacobian singular values (token units)",
                "Median over held-out models; first 64 of 256; log scale. Zero values are not drawn.",
                [("true key (evaluator-only)", BLUE, ""), ("random inputs", ORANGE, ""),
                 ("Adam candidate, k=16", AQUA, "")])
    lo, hi = -6, 6
    for i, (s, title) in enumerate(SCALES):
        x0, y0 = LEFT + i * (PW + GAP), TOP
        out.append(f'<text x="{x0}" y="{y0 - 10}" font-size="13" font-weight="600" fill="{TEXT}">{title}</text>')
        ymap = lambda v: y0 + PH - (v - lo) / (hi - lo) * PH  # noqa: E731
        y_axis(out, x0, y0, range(lo, hi + 1, 2), ymap, lambda t: f"1e{t}")
        for j in (1, 16, 32, 48, 64):
            out.append(f'<text x="{x0 + (j - 1) / 63 * PW:.1f}" y="{y0 + PH + 16}" font-size="11" '
                       f'text-anchor="middle" fill="{MUTED}">{j}</text>')
        n_key_zero = sum(1 for t in sp if t[s]["key"][0] == 0)
        for key, color, label in (("key", BLUE, "true key"), ("random_median", ORANGE, "random inputs"),
                                  ("candidate_k16", AQUA, "Adam candidate k=16")):
            arr = np.array([t[s][key] for t in sp if key in t[s]])
            if not len(arr):
                continue
            med = np.median(arr, 0)[:64]
            pts = []
            for j, v in enumerate(med):
                if v <= 0:
                    continue
                y = ymap(min(max(math.log10(v), lo), hi))
                tip = f"index {j + 1}: {v:.3g} tokens per unit input" if j in (0, 7, 15, 31, 63) else None
                pts.append((x0 + j / 63 * PW, y, tip))
            series(out, pts, color, "", label, title)
        if n_key_zero:
            out.append(f'<text x="{x0 + PW}" y="{y0 + PH + 34}" font-size="11" text-anchor="end" fill="{MUTED}">'
                       f'key Jacobian is exactly zero in {n_key_zero}/{len(sp)} models</text>')
    out.append(f'<text x="{width / 2:.0f}" y="{height - 6}" font-size="11" text-anchor="middle" fill="{MUTED}">'
               f'singular value index</text></svg>')
    path.write_text("\n".join(out))


def recovery_plot(agg, path):
    width, height = LEFT + 2 * PW + GAP + 30, TOP + PH + 50
    out = frame(width, height, "Whole unknown array recovered",
                "Per-model mean over 3 position draws, averaged over 30 models; 95% bootstrap CI in tooltips",
                [("Adam", BLUE, ""), ("Levenberg–Marquardt", ORANGE, ""), ("Adam, wrong known values", BLUE, "6 4")])
    for i, (s, title) in enumerate(SCALES):
        x0, y0 = LEFT + i * (PW + GAP), TOP
        out.append(f'<text x="{x0}" y="{y0 - 10}" font-size="13" font-weight="600" fill="{TEXT}">{title}</text>')
        ymap = lambda v: y0 + PH - v * PH  # noqa: E731
        y_axis(out, x0, y0, (0, 0.25, 0.5, 0.75, 1.0), ymap, lambda t: f"{int(t * 100)}%")
        xs = {k: x0 + 12 + j * (PW - 24) / (len(K) - 1) for j, k in enumerate(K)}
        for k, x in xs.items():
            out.append(f'<text x="{x:.1f}" y="{y0 + PH + 16}" font-size="11" text-anchor="middle" fill="{MUTED}">{k}</text>')
        for meth, variant, color, dash, label in (("adam_fit", "true", BLUE, "", "Adam"),
                                                  ("lm_fit", "true", ORANGE, "", "LM"),
                                                  ("adam_fit", "wrong_values", BLUE, "6 4", "Adam, wrong values")):
            pts = []
            for k in K:
                e = agg["conditions"].get(f"{s}|k={k}|{variant}|{meth}", {}).get("whole_unknown")
                if e:
                    tip = f"k={k}: {e['mean'] * 100:.1f}% (CI {e['ci95'][0] * 100:.1f}–{e['ci95'][1] * 100:.1f}%), {e['n_models']} models"
                    pts.append((xs[k], ymap(e["mean"]), tip))
            series(out, pts, color, dash, label, title)
    out.append(f'<text x="{width / 2:.0f}" y="{height - 8}" font-size="11" text-anchor="middle" fill="{MUTED}">'
               f'known tokens k (of 256)</text></svg>')
    path.write_text("\n".join(out))


def gain_plot(run, path):
    rows = list(csv.DictReader(open(run / "per_trial.csv")))
    held = [r for r in rows if r["pilot"] == "False" and r["method"] == "adam_fit" and r["variant"] == "true"
            and r["k"] in {str(k) for k in K} and r["known_consistent"] == "True" and r["g_tok"] != ""]
    width, height = LEFT + 2 * PW + GAP + 30, TOP + PH + 50
    out = frame(width, height, "Unobservable gain g at Adam candidates",
                "Median g (tokens of unknown change per unit input step invisible to known tokens); known-consistent candidates; log scale",
                [("whole unknown recovered", BLUE, ""), ("not recovered", ORANGE, ""), ("0.01·g = 0.5 token", MUTED, "3 3")])
    lo, hi = -6, 4
    for i, (s, title) in enumerate(SCALES):
        x0, y0 = LEFT + i * (PW + GAP), TOP
        out.append(f'<text x="{x0}" y="{y0 - 10}" font-size="13" font-weight="600" fill="{TEXT}">{title}</text>')
        ymap = lambda v: y0 + PH - (v - lo) / (hi - lo) * PH  # noqa: E731
        y_axis(out, x0, y0, range(lo, hi + 1, 2), ymap, lambda t: f"1e{t}")
        yref = ymap(math.log10(50))
        out.append(f'<line x1="{x0}" x2="{x0 + PW}" y1="{yref:.1f}" y2="{yref:.1f}" stroke="{MUTED}" stroke-dasharray="3 3"/>')
        xs = {k: x0 + 12 + j * (PW - 24) / (len(K) - 1) for j, k in enumerate(K)}
        for k, x in xs.items():
            out.append(f'<text x="{x:.1f}" y="{y0 + PH + 16}" font-size="11" text-anchor="middle" fill="{MUTED}">{k}</text>')
        for ok, color, label in ((True, BLUE, "recovered"), (False, ORANGE, "not recovered")):
            pts = []
            for k in K:
                vals = [float(r["g_tok"]) for r in held if r["scale"] == s and r["k"] == str(k)
                        and (r["whole_unknown"] == "True") == ok]
                if vals:
                    med = float(np.median(vals))
                    y = ymap(min(max(math.log10(max(med, 1e-12)), lo), hi))
                    pts.append((xs[k], y, f"k={k}: median g = {med:.3g} tokens/unit, {len(vals)} candidates"))
            series(out, pts, color, "", label, title)
    out.append(f'<text x="{width / 2:.0f}" y="{height - 8}" font-size="11" text-anchor="middle" fill="{MUTED}">'
               f'known tokens k (of 256)</text></svg>')
    path.write_text("\n".join(out))


def main(run_dir):
    run = Path(run_dir)
    agg = json.loads((run / "aggregate.json").read_text())
    spectra_plot(run, run / "jacobian_spectra.svg")
    recovery_plot(agg, run / "recovery_vs_k.svg")
    gain_plot(run, run / "unobservable_gain_vs_k.svg")


if __name__ == "__main__":
    main(sys.argv[1])
