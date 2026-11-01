#!/usr/bin/env python3
"""Phase 5B plots: accuracy on unknown positions versus number of known tokens.

Reads aggregate.json, writes static SVG small multiples (one panel per scale).
Native SVG <title> elements give per-point hover tooltips.
"""
import json, sys
from pathlib import Path
from xml.sax.saxutils import escape

SCALES = (("S1", "S = 1 (raw, control)"), ("S4095", "S = 4095"), ("S10000", "S = 10000"))
K = (0, 4, 8, 16, 32, 64)
# Reference categorical palette, first three slots (validated all-pairs, light surface).
SERIES = (
    ("input_fit", "true", "input fit", "#2a78d6", ""),
    ("plane_fit", "true", "plane fit", "#eb6834", ""),
    ("output_mean", "true", "output mean (control)", "#1baf7a", ""),
    ("input_fit", "wrong_values", "input fit, wrong known values (control)", "#2a78d6", "6 4"),
)
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def value(agg, scale, k, method, variant, metric):
    e = agg["conditions"].get(f"{scale}|k={k}|{variant}|{method}")
    if not e or metric not in e:
        return None
    m = e[metric]
    if isinstance(m, dict) and "mean" in m:
        return m["mean"], m["ci95"], e["models"]
    if isinstance(m, dict) and "k" in m:
        return m["k"] / m["n"], m["wilson95"], e["models"]
    return None


def panel(agg, scale, title, metric, x0, y0, w, h):
    out = [f'<text x="{x0}" y="{y0 - 10}" font-size="13" font-weight="600" fill="{TEXT}">{escape(title)}</text>']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + h - t * h
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{int(t * 100)}%</text>')
    xs = {k: x0 + 12 + i * (w - 24) / (len(K) - 1) for i, k in enumerate(K)}
    for k, x in xs.items():
        out.append(f'<text x="{x:.1f}" y="{y0 + h + 16}" font-size="11" text-anchor="middle" fill="{MUTED}">{k}</text>')
    for method, variant, label, color, dash in SERIES:
        pts = []
        for k in K:
            if k == 0 and variant == "wrong_values":
                continue
            m = "pca_lattice" if (k == 0 and method in ("input_fit", "plane_fit")) else method
            v = value(agg, scale, k, m, "true" if k == 0 else variant, metric)
            if v is not None:
                pts.append((k, v))
        if not pts:
            continue
        path = " ".join(f"{'M' if i == 0 else 'L'}{xs[k]:.1f},{y0 + h - v[0] * h:.1f}" for i, (k, v) in enumerate(pts))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>')
        for k, (v, ci, n) in pts:
            cx, cy = xs[k], y0 + h - v * h
            tip = (f"{label} | {title} | k={k}" + (" (blind PCA/lattice)" if k == 0 and method != "output_mean" else "")
                   + f" | {v * 100:.1f}% (95% CI {ci[0] * 100:.1f}–{ci[1] * 100:.1f}%), {n} models")
            out.append(f'<g><title>{escape(tip)}</title><circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="transparent"/>'
                       f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color if not dash else SURFACE}" '
                       f'stroke="{color if dash else SURFACE}" stroke-width="2"/></g>')
    return out


def chart(agg, metric, heading, subtitle, path):
    pw, ph, gap, left, top = 240, 180, 56, 56, 112
    width = left + 3 * pw + 2 * gap + 24
    height = top + ph + 64
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
           f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
           f'<text x="{left - 40}" y="26" font-size="16" font-weight="600" fill="{TEXT}">{escape(heading)}</text>',
           f'<text x="{left - 40}" y="46" font-size="12" fill="{MUTED}">{escape(subtitle)}</text>']
    lx = left - 40
    for method, variant, label, color, dash in SERIES:  # legend, one row
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" x2="{lx + 22}" y1="70" y2="70" stroke="{color}" stroke-width="2"{dash_attr}/>')
        out.append(f'<text x="{lx + 28}" y="74" font-size="11" fill="{TEXT}">{escape(label)}</text>')
        lx += 40 + 6.2 * len(label)
    for i, (scale, title) in enumerate(SCALES):
        out += panel(agg, scale, title, metric, left + i * (pw + gap), top, pw, ph)
    out.append(f'<text x="{width / 2:.0f}" y="{height - 12}" font-size="11" text-anchor="middle" fill="{MUTED}">'
               f'known tokens k (of 256); k = 0 is the blind baseline (Phase 5A PCA/lattice). Coinciding lines hide each other: see tooltips or RESULTS.md</text>')
    out.append("</svg>")
    Path(path).write_text("\n".join(out))


def main(run_dir):
    agg = json.loads((Path(run_dir) / "aggregate.json").read_text())
    chart(agg, "exact_fraction", "Exact recovery of unknown tokens",
          "Mean over held-out models of the per-model mean across position draws",
          Path(run_dir) / "exact_fraction_vs_k.svg")
    chart(agg, "whole_unknown", "Whole unknown array recovered",
          "Share of model × draw pairs; Wilson 95% interval in each point's tooltip",
          Path(run_dir) / "whole_unknown_vs_k.svg")


if __name__ == "__main__":
    main(sys.argv[1])
