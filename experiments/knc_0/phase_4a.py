#!/usr/bin/env python3
"""Phase 4A: single-token ReLU scaling study."""

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.knc_0.run import environment_metadata  # noqa: E402


class SingleTokenMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class SingleTokenLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.net(x)


def parameter_count_formula(input_dim, hidden_dim):
    return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1)


def csprng_example(input_dim):
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(os.urandom(32))))
    x = torch.tensor(rng.standard_normal((1, input_dim)), dtype=torch.float32)
    token = torch.tensor([[int(rng.integers(0, 4096))]], dtype=torch.long)
    return x, token


def token_from_output(output):
    return (output * 4095).round().clamp(0, 4095).long()


def candidate_metrics(candidate, target):
    error = int((candidate - target).abs().item())
    return {
        "absolute_error": error,
        "exact": error == 0,
        "within_1": error <= 1,
        "within_8": error <= 8,
        "within_64": error <= 64,
    }


def public_candidates(model, input_dim, probe_seed, probes, ensemble_size, linear=False):
    """Target-free candidates fixed before evaluator scoring."""
    generator = torch.Generator().manual_seed(probe_seed)
    with torch.no_grad():
        zero = token_from_output(model(torch.zeros(1, input_dim)))
        random_raw = model(torch.randn(probes, input_dim, generator=generator)).mean(dim=0, keepdim=True)
        random_mean = token_from_output(random_raw)
        initialized_outputs = []
        for index in range(ensemble_size):
            torch.manual_seed(probe_seed + 100000 + index)
            fresh = SingleTokenLinear(input_dim) if linear else SingleTokenMLP(input_dim, model.net[0].out_features)
            initialized_outputs.append(float(fresh(torch.zeros(1, input_dim)).item()))
        initialized = torch.tensor(initialized_outputs)
        initialized_mean = token_from_output(initialized.mean().reshape(1, 1))
    return {
        "zero": zero,
        "random_mean": random_mean,
        "untrained_ensemble_mean": initialized_mean,
        "untrained_output_mean": float(initialized.mean()),
        "untrained_output_std": float(initialized.std()),
    }


def serialized_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def run_one(args, architecture, input_dim, hidden_dim, trial, seed_offset):
    secret_x, target_token = csprng_example(input_dim)
    training_seed = args.seed + seed_offset
    torch.manual_seed(training_seed)
    if architecture == "linear":
        model = SingleTokenLinear(input_dim)
    else:
        model = SingleTokenMLP(input_dim, hidden_dim)
    parameters = sum(p.numel() for p in model.parameters())
    if architecture == "mlp":
        assert parameters == parameter_count_formula(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    target = target_token.float() / 4095.0
    started = time.perf_counter()
    exact = False
    for step in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(secret_x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if (step + 1) % args.check_every == 0 or step + 1 == args.epochs:
            with torch.no_grad():
                exact = bool(torch.equal(token_from_output(model(secret_x)), target_token))
            if exact:
                break
    elapsed = time.perf_counter() - started
    model.eval()
    candidates = public_candidates(
        model, input_dim, args.probe_seed + seed_offset,
        args.probes, args.ensemble_size, architecture == "linear",
    )
    with torch.no_grad():
        correct_candidate = token_from_output(model(secret_x))
    report = {
        "architecture": architecture,
        "input_dim": input_dim,
        "hidden_width": hidden_dim if architecture == "mlp" else 0,
        "trial": trial,
        "training_seed": training_seed,
        "parameter_count": parameters,
        "exact_memorization": bool(torch.equal(correct_candidate, target_token)),
        "correct_absolute_error": int((correct_candidate - target_token).abs().item()),
        "training_loss": float(loss.item()),
        "training_steps": step + 1,
        "elapsed_seconds": elapsed,
        "model_bytes": serialized_size(model),
        "failed_to_converge": not exact,
        "zero": candidate_metrics(candidates["zero"], target_token),
        "random_mean": candidate_metrics(candidates["random_mean"], target_token),
        "untrained_ensemble_mean": candidate_metrics(candidates["untrained_ensemble_mean"], target_token),
        "untrained_output_mean": candidates["untrained_output_mean"],
        "untrained_output_std": candidates["untrained_output_std"],
    }
    return report


def wilson(successes, total):
    if total == 0:
        return {"rate": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"rate": p, "ci95_low": center - half, "ci95_high": center + half}


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["architecture"], row["input_dim"], row["hidden_width"])
        groups.setdefault(key, []).append(row)
    configurations = []
    for (architecture, input_dim, width), items in sorted(groups.items()):
        configurations.append({
            "architecture": architecture,
            "input_dim": input_dim,
            "hidden_width": width,
            "trials": len(items),
            "parameter_count": items[0]["parameter_count"],
            "exact_memorization": wilson(sum(x["exact_memorization"] for x in items), len(items)),
            "failures": sum(x["failed_to_converge"] for x in items),
            "training_steps": mean_ci([x["training_steps"] for x in items]),
            "elapsed_seconds": mean_ci([x["elapsed_seconds"] for x in items]),
            "model_bytes": mean_ci([x["model_bytes"] for x in items]),
            "zero_absolute_error": mean_ci([x["zero"]["absolute_error"] for x in items]),
            "random_mean_absolute_error": mean_ci([x["random_mean"]["absolute_error"] for x in items]),
            "untrained_absolute_error": mean_ci([x["untrained_ensemble_mean"]["absolute_error"] for x in items]),
            "random_mean_within": {
                threshold: wilson(sum(x["random_mean"][threshold] for x in items), len(items))
                for threshold in ("within_1", "within_8", "within_64")
            },
        })
    # A transition is descriptive: first tested width reaching >=95% exact rate.
    transitions = {}
    for input_dim in sorted({x["input_dim"] for x in rows if x["architecture"] == "mlp"}):
        candidates = [x for x in configurations if x["architecture"] == "mlp" and x["input_dim"] == input_dim]
        reached = [x["hidden_width"] for x in candidates if x["exact_memorization"]["rate"] >= 0.95]
        transitions[str(input_dim)] = min(reached) if reached else None
    return {"configurations": configurations, "descriptive_transition_width_95pct": transitions}


def flatten_row(row):
    flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
    for candidate in ("zero", "random_mean", "untrained_ensemble_mean"):
        for key, value in row[candidate].items():
            flat[f"{candidate}_{key}"] = value
    return flat


def write_csv(rows, path):
    flattened = [flatten_row(row) for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def write_svg(configurations, metric_path, title, path):
    mlp = [x for x in configurations if x["architecture"] == "mlp"]
    dimensions = sorted({x["input_dim"] for x in mlp})
    colors = ["#2457c5", "#b22323", "#218739", "#8b3fa0"]
    all_values = []
    for item in mlp:
        value = item
        for key in metric_path:
            value = value[key]
        all_values.append(float(value))
    max_value = max(all_values) or 1
    content = []
    for dimension, color in zip(dimensions, colors):
        items = sorted((x for x in mlp if x["input_dim"] == dimension), key=lambda x: x["hidden_width"])
        points = []
        for index, item in enumerate(items):
            value = item
            for key in metric_path:
                value = value[key]
            points.append((70 + index * 95, 330 - 260 * float(value) / max_value))
        content.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        content.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in points)
        content.append(f'<text x="500" y="{50 + dimensions.index(dimension)*18}" fill="{color}">d={dimension}</text>')
    labels = "".join(f'<text x="{70+i*95}" y="350" text-anchor="middle">{width}</text>' for i, width in enumerate(sorted({x["hidden_width"] for x in mlp})))
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="390"><rect width="100%" height="100%" fill="white"/><text x="320" y="25" text-anchor="middle" font-size="17">{title}</text><line x1="60" y1="330" x2="600" y2="330" stroke="black"/><line x1="60" y1="45" x2="60" y2="330" stroke="black"/>{"".join(content)}{labels}<text x="320" y="380" text-anchor="middle">hidden width</text></svg>''')


def main(args):
    torch.set_num_threads(args.threads)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    config = vars(args).copy()
    config["environment"] = environment_metadata(torch.device("cpu"))
    config["secret_generation"] = "OS CSPRNG; entropy not persisted"
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    offset = 0
    for input_dim in args.input_dims:
        for width in args.widths:
            for trial in range(args.trials):
                rows.append(run_one(args, "mlp", input_dim, width, trial, offset))
                offset += 1
        for trial in range(args.trials):
            rows.append(run_one(args, "linear", input_dim, 0, trial, offset))
            offset += 1
    result = aggregate(rows)
    result["total_runs"] = len(rows)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2))
    write_csv(rows, output / "per_run.csv")
    configurations = result["configurations"]
    write_svg(configurations, ["exact_memorization", "rate"], "Exact memorization vs width", output / "exact_vs_width.svg")
    write_svg(configurations, ["random_mean_absolute_error", "mean"], "Random-mean error vs width", output / "error_vs_width.svg")
    write_svg(configurations, ["elapsed_seconds", "mean"], "Runtime vs width", output / "runtime_vs_width.svg")
    write_svg(configurations, ["model_bytes", "mean"], "Serialized size vs width", output / "size_vs_width.svg")
    print(json.dumps({"total_runs": len(rows), "transitions": result["descriptive_transition_width_95pct"]}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knc0_phase_4a_results")
    parser.add_argument("--input-dims", type=int, nargs="+", default=[1, 4, 16, 256])
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--probes", type=int, default=256)
    parser.add_argument("--ensemble-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--probe-seed", type=int, default=42000)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
