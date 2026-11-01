#!/usr/bin/env python3
"""KNC-0 Phase 2.5: non-adaptive output-leakage diagnostics."""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.knc_0.run import Memory, environment_metadata, metrics, tokens_from_output


def csprng_secret(input_dim, output_dim):
    """Generate evaluator-only data; entropy is deliberately never returned."""
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(os.urandom(32))))
    secret_x = torch.tensor(rng.standard_normal((1, input_dim)), dtype=torch.float32)
    target_tokens = torch.tensor(rng.integers(0, 4096, (1, output_dim)), dtype=torch.long)
    permutation = torch.tensor(rng.permutation(output_dim), dtype=torch.long)
    return secret_x, target_tokens, target_tokens[:, permutation]


def train_mapping(secret_x, target_tokens, config, training_seed):
    """Evaluator-side training. The CSPRNG secret never enters public config."""
    torch.manual_seed(training_seed)
    model = Memory(config["input_dim"], config["hidden_dim"], config["output_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    target = target_tokens.float() / 4095.0
    started = time.perf_counter()
    for step in range(config["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(secret_x), target)
        loss.backward()
        optimizer.step()
        if (step + 1) % config["check_every"] == 0 or step + 1 == config["epochs"]:
            exact = metrics(tokens_from_output(model(secret_x)), target_tokens)["all_exact"]
            if exact:
                break
    model.eval()
    return model, {
        "steps": step + 1,
        "seconds": time.perf_counter() - started,
        "final_loss": float(loss.item()),
        "correct_key": metrics(tokens_from_output(model(secret_x)), target_tokens),
    }


def parameter_only_candidate(model):
    """Gaussian moment propagation using public parameters only."""
    linear1, linear2, linear3 = model.net[0], model.net[2], model.net[4]

    def relu_moments(mean, variance):
        std = variance.clamp_min(1e-12).sqrt()
        alpha = mean / std
        phi = torch.exp(-0.5 * alpha.square()) / math.sqrt(2 * math.pi)
        cdf = 0.5 * (1 + torch.erf(alpha / math.sqrt(2)))
        first = std * phi + mean * cdf
        second = (variance + mean.square()) * cdf + mean * std * phi
        return first, (second - first.square()).clamp_min(0)

    mean1 = linear1.bias
    var1 = linear1.weight.square().sum(dim=1)
    relu1_mean, relu1_var = relu_moments(mean1, var1)
    mean2 = linear2.weight @ relu1_mean + linear2.bias
    var2 = linear2.weight.square() @ relu1_var
    relu2_mean, _ = relu_moments(mean2, var2)
    return tokens_from_output((linear3.weight @ relu2_mean + linear3.bias).unsqueeze(0))


def _quantile_from_hist(hist, q):
    threshold = max(1, math.ceil(hist.sum(dim=1, keepdim=True)[0, 0].item() * q))
    return (hist.cumsum(dim=1) >= threshold).to(torch.int8).argmax(dim=1).float()


def attacker_observe(model, public_config, probe_counts, batch_size, probe_seed):
    """Attacker API: consumes public model/config only and returns frozen candidates."""
    del public_config  # architecture is already embodied by the loaded model
    max_probes = max(probe_counts)
    generator = torch.Generator().manual_seed(probe_seed)
    output_dim = model.net[-1].out_features
    cumulative = torch.zeros(output_dim, dtype=torch.float64)
    cumulative_sq = torch.zeros(output_dim, dtype=torch.float64)
    histogram = torch.zeros((output_dim, 4096), dtype=torch.int64)
    active1 = torch.zeros(model.net[0].out_features, dtype=torch.float64)
    active2 = torch.zeros(model.net[2].out_features, dtype=torch.float64)
    candidates, runtimes = {}, {}
    seen = 0
    started = time.perf_counter()

    with torch.no_grad():
        while seen < max_probes:
            size = min(batch_size, max_probes - seen)
            x = torch.randn(size, model.net[0].in_features, generator=generator)
            h1 = torch.relu(model.net[0](x))
            h2 = torch.relu(model.net[2](h1))
            raw = model.net[4](h2)
            token_outputs = tokens_from_output(raw)
            cumulative += raw.double().sum(dim=0)
            cumulative_sq += raw.double().square().sum(dim=0)
            active1 += (h1 > 0).double().sum(dim=0)
            active2 += (h2 > 0).double().sum(dim=0)
            for coordinate in range(output_dim):
                histogram[coordinate] += torch.bincount(
                    token_outputs[:, coordinate], minlength=4096
                )
            seen += size
            if seen in probe_counts:
                candidates[str(seen)] = tokens_from_output((cumulative / seen).float().unsqueeze(0))
                runtimes[str(seen)] = time.perf_counter() - started

    raw_mean = cumulative / seen
    raw_std = (cumulative_sq / seen - raw_mean.square()).clamp_min(0).sqrt()
    quantiles = {str(q): _quantile_from_hist(histogram, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return {
        "mean_candidates": candidates,
        "runtimes": runtimes,
        "parameter_only": parameter_only_candidate(model),
        "bias_only": tokens_from_output(model.net[-1].bias.unsqueeze(0)),
        "coordinate_stats": {"raw_mean": raw_mean, "raw_std": raw_std, "quantiles": quantiles},
        "random_active_probability": {"layer1": active1 / seen, "layer2": active2 / seen},
    }


def correct_activation_diagnostics(model, secret_x, random_active_probability):
    """Evaluator-only comparison; this function is never called by attack code."""
    with torch.no_grad():
        h1 = torch.relu(model.net[0](secret_x))
        h2 = torch.relu(model.net[2](h1))
    result = {}
    for name, hidden in (("layer1", h1), ("layer2", h2)):
        correct_active = (hidden.squeeze(0) > 0).double()
        probability = random_active_probability[name]
        agreement = correct_active * probability + (1 - correct_active) * (1 - probability)
        result[name] = {
            "correct_active_fraction": float(correct_active.mean()),
            "random_active_fraction": float(probability.mean()),
            "mean_pattern_agreement": float(agreement.mean()),
        }
    return result


def evaluate_observation(observation, target_tokens):
    """Evaluator scores already-frozen attacker candidates against hidden truth."""
    target_float = target_tokens.float()
    convergence = {}
    for count, candidate in observation["mean_candidates"].items():
        basic = metrics(candidate, target_tokens)
        error = candidate.float() - target_float
        convergence[count] = {
            **basic,
            "rmse_tokens": float(error.square().mean().sqrt()),
            "runtime_seconds": observation["runtimes"][count],
        }
    candidates = {
        "parameter_only": observation["parameter_only"],
        "bias_only": observation["bias_only"],
        "midpoint": torch.full_like(target_tokens, 2048),
    }
    candidate_metrics = {name: metrics(value, target_tokens) for name, value in candidates.items()}
    stats = observation["coordinate_stats"]
    mean_tokens = tokens_from_output(stats["raw_mean"].float().unsqueeze(0)).squeeze(0).float()
    deviations = mean_tokens - target_tokens.squeeze(0).float()
    coordinate_summary = {
        "mean_output_std": float(stats["raw_std"].mean()),
        "mean_absolute_deviation_from_target": float(deviations.abs().mean()),
        "rmse_deviation_from_target": float(deviations.square().mean().sqrt()),
        "mean_interquartile_range": float((stats["quantiles"]["0.75"] - stats["quantiles"]["0.25"]).mean()),
        "mean_90pct_range": float((stats["quantiles"]["0.95"] - stats["quantiles"]["0.05"]).mean()),
    }
    return convergence, candidate_metrics, coordinate_summary


def save_public_bundle(model, public_config, directory):
    directory.mkdir(parents=True)
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "config.json").write_text(json.dumps(public_config, indent=2))


def run_trial(args, trial_index, output_dir):
    secret_x, target_tokens, shuffled_tokens = csprng_secret(args.input_dim, args.output_dim)
    config = {
        "input_dim": args.input_dim, "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim, "lr": args.lr, "epochs": args.epochs,
        "check_every": args.check_every,
    }
    training_seed = args.training_seed + trial_index
    model, training = train_mapping(secret_x, target_tokens, config, training_seed)
    if not training["correct_key"]["all_exact"]:
        raise RuntimeError(f"trial {trial_index} failed correct-key recovery")
    trial_dir = output_dir / f"trial_{trial_index:02d}"
    public_config = {
        "input_dim": args.input_dim, "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim, "activation": "relu",
    }
    save_public_bundle(model, public_config, trial_dir / "public")
    observation = attacker_observe(
        model, public_config, args.probe_counts, args.batch_size,
        args.probe_seed + trial_index,
    )
    convergence, baselines, coordinate_summary = evaluate_observation(observation, target_tokens)
    activation = correct_activation_diagnostics(
        model, secret_x, observation["random_active_probability"]
    )

    # Calibration control: same secret input and protocol, independently
    # initialized model, evaluator-only shuffled target.
    control, control_training = train_mapping(
        secret_x, shuffled_tokens, config, args.control_seed + trial_index
    )
    control_observation = attacker_observe(
        control, public_config, [max(args.probe_counts)], args.batch_size,
        args.control_probe_seed + trial_index,
    )
    control_convergence, control_baselines, _ = evaluate_observation(
        control_observation, shuffled_tokens
    )

    # Untrained calibration is public and does not consume evaluator secrets.
    torch.manual_seed(args.untrained_seed + trial_index)
    untrained = Memory(args.input_dim, args.hidden_dim, args.output_dim).eval()
    untrained_observation = attacker_observe(
        untrained, public_config, [max(args.probe_counts)], args.batch_size,
        args.untrained_probe_seed + trial_index,
    )
    _, untrained_baselines, _ = evaluate_observation(untrained_observation, target_tokens)
    untrained_mean = metrics(
        untrained_observation["mean_candidates"][str(max(args.probe_counts))], target_tokens
    )

    report = {
        "trial": trial_index,
        "training_seed": training_seed,
        "training": training,
        "mean_convergence": convergence,
        "baselines": baselines,
        "coordinate_summary": coordinate_summary,
        "activation_diagnostics_evaluator_only": activation,
        "shuffled_target_control": {
            "training": control_training,
            "random_mean": control_convergence[str(max(args.probe_counts))],
            "parameter_only": control_baselines["parameter_only"],
        },
        "untrained_control": {
            "random_mean": untrained_mean,
            "parameter_only": untrained_baselines["parameter_only"],
        },
    }
    (trial_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def aggregate_reports(reports, args):
    convergence = {}
    for count in args.probe_counts:
        key = str(count)
        convergence[key] = {
            "mae_tokens": mean_ci95([r["mean_convergence"][key]["mae_tokens"] for r in reports]),
            "rmse_tokens": mean_ci95([r["mean_convergence"][key]["rmse_tokens"] for r in reports]),
            "exact_fraction": mean_ci95([r["mean_convergence"][key]["exact_fraction"] for r in reports]),
            "full_recoveries": sum(r["mean_convergence"][key]["all_exact"] for r in reports),
            "runtime_seconds": mean_ci95([r["mean_convergence"][key]["runtime_seconds"] for r in reports]),
        }
    baseline_names = reports[0]["baselines"]
    baselines = {
        name: {
            "mae_tokens": mean_ci95([r["baselines"][name]["mae_tokens"] for r in reports]),
            "exact_fraction": mean_ci95([r["baselines"][name]["exact_fraction"] for r in reports]),
            "full_recoveries": sum(r["baselines"][name]["all_exact"] for r in reports),
        }
        for name in baseline_names
    }
    return {
        "trials": len(reports),
        "correct_key_full_recoveries": sum(r["training"]["correct_key"]["all_exact"] for r in reports),
        "probe_convergence": convergence,
        "baselines": baselines,
        "coordinate_statistics": {
            key: mean_ci95([r["coordinate_summary"][key] for r in reports])
            for key in reports[0]["coordinate_summary"]
        },
        "activation_diagnostics_evaluator_only": {
            layer: {
                key: mean_ci95([r["activation_diagnostics_evaluator_only"][layer][key] for r in reports])
                for key in reports[0]["activation_diagnostics_evaluator_only"][layer]
            }
            for layer in ("layer1", "layer2")
        },
        "shuffled_target_control_mae": mean_ci95([
            r["shuffled_target_control"]["random_mean"]["mae_tokens"] for r in reports
        ]),
        "untrained_random_mean_mae": mean_ci95([
            r["untrained_control"]["random_mean"]["mae_tokens"] for r in reports
        ]),
    }


def write_csv(reports, args, path):
    fields = ["trial", "probe_count", "mae_tokens", "rmse_tokens", "exact_fraction", "all_exact", "runtime_seconds"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for count in args.probe_counts:
                writer.writerow({"trial": report["trial"], "probe_count": count, **report["mean_convergence"][str(count)]})


def write_svg(aggregate, metric, title, path):
    points = [(int(k), v[metric]["mean"]) for k, v in aggregate["probe_convergence"].items()]
    max_y = max(y for _, y in points) or 1
    coords = []
    for index, (_, value) in enumerate(points):
        x = 60 + index * (520 / max(1, len(points) - 1))
        y = 330 - 260 * value / max_y
        coords.append((x, y, value))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
    labels = "".join(f'<text x="{x:.1f}" y="355" text-anchor="middle">{count:g}</text>' for (count, _), (x, _, _) in zip(points, coords))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4"/><text x="{x:.1f}" y="{y-9:.1f}" text-anchor="middle">{value:.3g}</text>' for x, y, value in coords)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="390" viewBox="0 0 640 390">
<rect width="100%" height="100%" fill="white"/><text x="320" y="28" text-anchor="middle" font-size="18">{title}</text>
<line x1="60" y1="330" x2="590" y2="330" stroke="black"/><line x1="60" y1="60" x2="60" y2="330" stroke="black"/>
<polyline points="{polyline}" fill="none" stroke="#2457c5" stroke-width="2"/>{dots}{labels}
<text x="320" y="382" text-anchor="middle">probe count</text></svg>'''
    path.write_text(svg)


def main(args):
    torch.set_num_threads(args.threads)
    output_dir = Path(args.out)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    public_experiment_config = {
        "input_dim": args.input_dim, "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim, "trials": args.trials,
        "probe_counts": args.probe_counts, "batch_size": args.batch_size,
        "epochs": args.epochs, "lr": args.lr, "threads": args.threads,
        "environment": environment_metadata(torch.device("cpu")),
        "secret_generation": "OS CSPRNG; entropy not persisted",
    }
    (output_dir / "config.json").write_text(json.dumps(public_experiment_config, indent=2))
    reports = [run_trial(args, index, output_dir) for index in range(args.trials)]
    aggregate = aggregate_reports(reports, args)
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    write_csv(reports, args, output_dir / "per_trial.csv")
    write_svg(aggregate, "mae_tokens", "Random-output mean: MAE vs probe count", output_dir / "mae_vs_probes.svg")
    write_svg(aggregate, "runtime_seconds", "Random-output mean: runtime vs probe count", output_dir / "time_vs_probes.svg")
    print(json.dumps(aggregate, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knc0_phase_2_5_results")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--probe-counts", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--training-seed", type=int, default=25000)
    parser.add_argument("--probe-seed", type=int, default=26000)
    parser.add_argument("--control-seed", type=int, default=27000)
    parser.add_argument("--control-probe-seed", type=int, default=28000)
    parser.add_argument("--untrained-seed", type=int, default=29000)
    parser.add_argument("--untrained-probe-seed", type=int, default=30000)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
