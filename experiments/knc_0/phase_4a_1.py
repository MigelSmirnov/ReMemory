#!/usr/bin/env python3
"""Phase 4A.1: exact off-key recovery and constant-function diagnostics."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.knc_0.phase_4a import (  # noqa: E402
    SingleTokenLinear, SingleTokenMLP, csprng_example, parameter_count_formula,
    token_from_output,
)
from experiments.knc_0.run import environment_metadata  # noqa: E402


CANDIDATES = (
    "zero", "random_mean", "random_median", "random_single",
    "bias_only", "last_layer_mean_hidden", "midpoint", "uniform_random",
)


def attacker_diagnostics(model, input_dim, seed, samples, perturbation, linear=False):
    """Public-only diagnostics. No target or memorized input is accepted."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, input_dim, generator=generator)
    with torch.no_grad():
        raw = model(x).reshape(-1)
        rounded = token_from_output(raw).reshape(-1)
        zero = token_from_output(model(torch.zeros(1, input_dim)))
        candidates = {
            "zero": zero,
            "random_mean": token_from_output(raw.mean().reshape(1, 1)),
            "random_median": token_from_output(raw.median().reshape(1, 1)),
            "random_single": rounded[:1].reshape(1, 1),
            "midpoint": torch.tensor([[2048]]),
            "uniform_random": torch.randint(0, 4096, (1, 1), generator=generator),
        }
        if linear:
            candidates["bias_only"] = token_from_output(model.net.bias.reshape(1, 1))
            candidates["last_layer_mean_hidden"] = candidates["random_mean"].clone()
            activation = {"layer1": None, "layer2": None}
        else:
            h1 = torch.relu(model.net[0](x))
            h2 = torch.relu(model.net[2](h1))
            last = model.net[4]
            candidates["bias_only"] = token_from_output(last.bias.reshape(1, 1))
            candidates["last_layer_mean_hidden"] = token_from_output(
                (h2.mean(dim=0) @ last.weight.squeeze(0) + last.bias).reshape(1, 1)
            )
            activation = {
                "layer1": float((h1 > 0).float().mean()),
                "layer2": float((h2 > 0).float().mean()),
            }
        perturb = x + perturbation * torch.randn(x.shape, generator=generator)
        perturbed_tokens = token_from_output(model(perturb)).reshape(-1)
        sensitivity = float((perturbed_tokens - rounded).abs().float().mean())
        counts = torch.bincount(rounded, minlength=4096)
        same_fraction = float(counts.max() / samples)
        token_std = float(rounded.float().std())
        diagnostics = {
            "token_std": token_std,
            "token_range": int(rounded.max() - rounded.min()),
            "dominant_token_fraction": same_fraction,
            "sensitivity_mean_abs_tokens": sensitivity,
            "raw_output_std": float(raw.std()),
            "approximately_constant": token_std <= 1.0 and same_fraction >= 0.99,
            "activation": activation,
        }
    return candidates, diagnostics


def score_candidate(candidate, target):
    error = int((candidate - target).abs().item())
    return {
        "absolute_error": error,
        "exact": error == 0,
        "within_1": error <= 1,
        "within_8": error <= 8,
        "within_64": error <= 64,
    }


def run_one(args, architecture, input_dim, width, trial, offset):
    secret_x, target = csprng_example(input_dim)
    seed = args.seed + offset
    torch.manual_seed(seed)
    linear = architecture == "linear"
    model = SingleTokenLinear(input_dim) if linear else SingleTokenMLP(input_dim, width)
    parameters = sum(x.numel() for x in model.parameters())
    if not linear:
        assert parameters == parameter_count_formula(input_dim, width)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    normalized = target.float() / 4095.0
    started = time.perf_counter()
    exact = False
    for step in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(secret_x), normalized)
        loss.backward()
        optimizer.step()
        if (step + 1) % args.check_every == 0 or step + 1 == args.epochs:
            exact = bool(torch.equal(token_from_output(model(secret_x)), target))
            if exact:
                break
    model.eval()
    candidates, diagnostics = attacker_diagnostics(
        model, input_dim, args.probe_seed + offset, args.samples,
        args.perturbation, linear,
    )
    return {
        "architecture": architecture,
        "input_dim": input_dim,
        "hidden_width": 0 if linear else width,
        "trial": trial,
        "parameter_count": parameters,
        "training_steps": step + 1,
        "training_loss": float(loss.item()),
        "training_seconds": time.perf_counter() - started,
        "correct_key_exact": exact,
        "failed_to_converge": not exact,
        **diagnostics,
        "candidates": {name: score_candidate(candidates[name], target) for name in CANDIDATES},
    }


def wilson(successes, total):
    z = 1.96
    p = successes / total if total else 0.0
    if not total:
        return {"numerator": 0, "denominator": 0, "rate": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"numerator": successes, "denominator": total, "rate": p, "ci95_low": center - half, "ci95_high": center + half}


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def candidate_summary(items, name):
    return {
        threshold: wilson(sum(x["candidates"][name][threshold] for x in items), len(items))
        for threshold in ("exact", "within_1", "within_8", "within_64")
    } | {"absolute_error": mean_ci([x["candidates"][name]["absolute_error"] for x in items])}


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["architecture"], row["input_dim"], row["hidden_width"]), []).append(row)
    configurations = []
    for (architecture, input_dim, width), items in sorted(groups.items()):
        constant = [x for x in items if x["approximately_constant"]]
        nonconstant = [x for x in items if not x["approximately_constant"]]
        configurations.append({
            "architecture": architecture, "input_dim": input_dim, "hidden_width": width,
            "parameter_count": items[0]["parameter_count"], "trials": len(items),
            "training_failures": sum(x["failed_to_converge"] for x in items),
            "approximately_constant": wilson(len(constant), len(items)),
            "token_std": mean_ci([x["token_std"] for x in items]),
            "token_range": mean_ci([x["token_range"] for x in items]),
            "dominant_token_fraction": mean_ci([x["dominant_token_fraction"] for x in items]),
            "sensitivity": mean_ci([x["sensitivity_mean_abs_tokens"] for x in items]),
            "layer1_active": mean_ci([x["activation"]["layer1"] for x in items]) if architecture == "mlp" else None,
            "layer2_active": mean_ci([x["activation"]["layer2"] for x in items]) if architecture == "mlp" else None,
            "candidates": {name: candidate_summary(items, name) for name in CANDIDATES},
            "zero_exact_by_constant_class": {
                "constant": wilson(sum(x["candidates"]["zero"]["exact"] for x in constant), len(constant)),
                "nonconstant": wilson(sum(x["candidates"]["zero"]["exact"] for x in nonconstant), len(nonconstant)),
            },
        })
    return {"configurations": configurations, "total_runs": len(rows)}


def flatten(row):
    result = {k: v for k, v in row.items() if k not in ("candidates", "activation")}
    result["layer1_active"] = row["activation"]["layer1"]
    result["layer2_active"] = row["activation"]["layer2"]
    for name, metric in row["candidates"].items():
        for key, value in metric.items():
            result[f"{name}_{key}"] = value
    return result


def write_csv(rows, path):
    values = [flatten(x) for x in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def write_svg(configurations, value_getter, title, path):
    mlp = [x for x in configurations if x["architecture"] == "mlp"]
    dimensions = sorted({x["input_dim"] for x in mlp})
    colors = ["#2457c5", "#218739", "#b22323"]
    maximum = max(value_getter(x) for x in mlp) or 1
    content = []
    for dimension, color in zip(dimensions, colors):
        items = sorted((x for x in mlp if x["input_dim"] == dimension), key=lambda x: x["hidden_width"])
        points = [(80 + index * 200, 330 - 260 * value_getter(item) / maximum) for index, item in enumerate(items)]
        content.append(f'<polyline points="{" ".join(f"{x},{y:.1f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        content.extend(f'<circle cx="{x}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in points)
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="390"><rect width="100%" height="100%" fill="white"/><text x="320" y="25" text-anchor="middle" font-size="17">{title}</text><line x1="60" y1="330" x2="600" y2="330" stroke="black"/><line x1="60" y1="45" x2="60" y2="330" stroke="black"/>{"".join(content)}<text x="320" y="380" text-anchor="middle">hidden width 1 / 2 / 4</text></svg>''')


def main(args):
    torch.set_num_threads(args.threads)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    config = vars(args).copy()
    config["environment"] = environment_metadata(torch.device("cpu"))
    config["constant_threshold"] = {"token_std_max": 1.0, "dominant_token_fraction_min": 0.99}
    config["secret_generation"] = "OS CSPRNG; entropy not persisted"
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows, offset = [], 0
    for input_dim in args.input_dims:
        for width in args.widths:
            for trial in range(args.trials):
                rows.append(run_one(args, "mlp", input_dim, width, trial, offset)); offset += 1
        for trial in range(args.trials):
            rows.append(run_one(args, "linear", input_dim, 0, trial, offset)); offset += 1
    result = aggregate(rows)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2))
    write_csv(rows, output / "per_run.csv")
    write_svg(result["configurations"], lambda x: x["candidates"]["zero"]["exact"]["rate"], "Zero-input exact recovery vs width", output / "exact_recovery_vs_width.svg")
    write_svg(result["configurations"], lambda x: x["token_std"]["mean"], "Output token standard deviation vs width", output / "output_variance_vs_width.svg")
    print(json.dumps({"total_runs": len(rows)}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knc0_phase_4a_1_results")
    parser.add_argument("--input-dims", type=int, nargs="+", default=[1, 4, 256])
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--perturbation", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41100)
    parser.add_argument("--probe-seed", type=int, default=42100)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
