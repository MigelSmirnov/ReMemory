#!/usr/bin/env python3
"""KNC-0 Phase 3: adaptive, target-free white-box evaluation on CPU."""

import argparse
import csv
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.knc_0.phase_2_5 import (  # noqa: E402
    csprng_secret, parameter_only_candidate, train_mapping,
)
from experiments.knc_0.run import (  # noqa: E402
    Memory, environment_metadata, metrics, tokens_from_output,
)


def clone_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def train_with_initial(secret_x, target_tokens, config, training_seed):
    torch.manual_seed(training_seed)
    initial = Memory(config["input_dim"], config["hidden_dim"], config["output_dim"])
    initial_state = clone_state(initial)
    model, training = train_mapping(secret_x, target_tokens, config, training_seed)
    return model, initial_state, training


def public_objective(model, x, sparsity_weight=0.1):
    """Attacker-observable objective; never consumes evaluator truth."""
    pre1 = model.net[0](x)
    h1 = torch.relu(pre1)
    h2 = torch.relu(model.net[2](h1))
    raw = model.net[4](h2)
    scaled = raw * 4095.0
    lattice = (scaled - scaled.round()).abs().mean(dim=1)
    output_range = (torch.relu(-raw) + torch.relu(raw - 1)).mean(dim=1)
    soft_active = torch.sigmoid(pre1 / 0.1).mean(dim=1)
    prior = (x.square().mean(dim=1) - 1).square() + x.mean(dim=1).square()
    total = lattice + 2.0 * output_range + sparsity_weight * soft_active + 0.1 * prior
    return total, {
        "lattice": lattice, "output_range": output_range,
        "soft_active": soft_active, "prior": prior,
    }


def activation_pattern(model, x):
    with torch.no_grad():
        return model.net[0](x) > 0


def sparse_region_attack(model, starts, steps, lr, seed):
    """Phase 3.1 attacker: discover sparse first-layer regions, target-free."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(starts, model.net[0].in_features, generator=generator, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pre = model.net[0](x)
        soft_active = torch.sigmoid(pre / 0.1).mean(dim=1)
        prior = (x.square().mean(dim=1) - 1).square() + x.mean(dim=1).square()
        (soft_active + 0.1 * prior).mean().backward()
        optimizer.step()
    frozen = x.detach()
    with torch.no_grad():
        patterns = activation_pattern(model, frozen)
        active = patterns.float().mean(dim=1)
        norms = frozen.norm(dim=1)
        scores = active + 0.1 * (frozen.square().mean(dim=1) - 1).abs()
        selected = int(scores.argmin())
        random_control = torch.randn(frozen.shape, generator=generator)
        random_control *= (norms / random_control.norm(dim=1).clamp_min(1e-12)).unsqueeze(1)

        # Public geometry diagnostics around each solution. Perturbations crossing
        # a ReLU boundary are excluded before output variation is measured.
        local_std = []
        for index in range(starts):
            perturb = frozen[index:index + 1] + 0.01 * torch.randn(
                32, frozen.shape[1], generator=generator
            )
            keep = (activation_pattern(model, perturb) == patterns[index]).all(dim=1)
            if keep.any():
                local_std.append(float(model(perturb[keep]).std(dim=0).mean()))
        packed_patterns = [bytes(row.to(torch.uint8).tolist()) for row in patterns]
    return {
        "candidates": frozen,
        "matched_random": random_control,
        "selected": frozen[selected:selected + 1],
        "selected_index": selected,
        "active_fractions": active,
        "input_norms": norms,
        "distinct_patterns": len(set(packed_patterns)),
        "mean_local_output_std": float(np.mean(local_std)) if local_std else None,
        "elapsed_seconds": time.perf_counter() - started,
        "forward_passes": starts * steps,
        "backward_passes": starts * steps,
    }


def adaptive_attack(model, starts, max_steps, checkpoints, lr, seed):
    """Phase 3.2 gradient candidates frozen at predefined budgets."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(starts, model.net[0].in_features, generator=generator, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    candidates, observables, runtimes = {}, {}, {}
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        objective, _ = public_objective(model, x)
        objective.mean().backward()
        optimizer.step()
        if step in checkpoints:
            with torch.no_grad():
                score, components = public_objective(model, x)
                selected = int(score.argmin())
                candidates[str(step)] = x[selected:selected + 1].detach().clone()
                observables[str(step)] = {
                    "objective": float(score[selected]),
                    **{name: float(value[selected]) for name, value in components.items()},
                }
                runtimes[str(step)] = time.perf_counter() - started
    return {"candidates": candidates, "observables": observables, "runtimes": runtimes}


def random_budget_attack(model, starts, checkpoints, seed):
    """Matched forward-pass control using the identical public objective."""
    generator = torch.Generator().manual_seed(seed)
    candidates, observables, runtimes = {}, {}, {}
    best_score, best_x, best_components = float("inf"), None, None
    seen = 0
    started = time.perf_counter()
    for checkpoint in checkpoints:
        needed = starts * checkpoint - seen
        for offset in range(0, needed, 1024):
            x = torch.randn(min(1024, needed - offset), model.net[0].in_features, generator=generator)
            with torch.no_grad():
                score, components = public_objective(model, x)
                index = int(score.argmin())
                if float(score[index]) < best_score:
                    best_score = float(score[index])
                    best_x = x[index:index + 1].clone()
                    best_components = {name: float(value[index]) for name, value in components.items()}
        seen = starts * checkpoint
        candidates[str(checkpoint)] = best_x.clone()
        observables[str(checkpoint)] = {"objective": best_score, **best_components}
        runtimes[str(checkpoint)] = time.perf_counter() - started
    return {"candidates": candidates, "observables": observables, "runtimes": runtimes}


def known_initialization_candidate(model, initial_state):
    """Exploratory stronger-scenario estimator using public final+initial weights."""
    delta = model.net[-1].bias.detach() - initial_state["net.4.bias"]
    # Adam bias updates mostly expose direction, not target magnitude. This fixed
    # mapping is declared before evaluation and makes no rank-one assumption.
    normalized = torch.where(delta >= 0, torch.tensor(0.75), torch.tensor(0.25))
    return tokens_from_output(normalized.unsqueeze(0))


def extended_metrics(model, candidate_x, target_tokens, secret_x):
    with torch.no_grad():
        guess = tokens_from_output(model(candidate_x))
        error = (guess - target_tokens).abs()
        cosine = torch.nn.functional.cosine_similarity(candidate_x, secret_x).item()
        l2 = torch.linalg.vector_norm(candidate_x - secret_x).item()
    return {
        **metrics(guess, target_tokens),
        "rmse_tokens": float(error.float().square().mean().sqrt()),
        "within_1_fraction": float((error <= 1).float().mean()),
        "within_8_fraction": float((error <= 8).float().mean()),
        "within_64_fraction": float((error <= 64).float().mean()),
        "input_cosine_evaluator_only": cosine,
        "input_l2_evaluator_only": l2,
    }


def evaluate_candidate_set(model, candidates, target_tokens, secret_x):
    """Evaluator-only: called only after attacker candidate generation returns."""
    return {
        key: extended_metrics(model, candidate, target_tokens, secret_x)
        for key, candidate in candidates.items()
    }


def evaluate_population(model, population, target_tokens, secret_x):
    values = [extended_metrics(model, x[None, :], target_tokens, secret_x) for x in population]
    return {
        "mean_mae_tokens": float(np.mean([x["mae_tokens"] for x in values])),
        "mean_exact_fraction": float(np.mean([x["exact_fraction"] for x in values])),
        "mean_within_64_fraction": float(np.mean([x["within_64_fraction"] for x in values])),
        "mean_input_cosine_evaluator_only": float(np.mean([x["input_cosine_evaluator_only"] for x in values])),
    }


def parameter_metrics(candidate, target_tokens):
    error = (candidate - target_tokens).abs()
    return {
        **metrics(candidate, target_tokens),
        "rmse_tokens": float(error.float().square().mean().sqrt()),
        "within_1_fraction": float((error <= 1).float().mean()),
        "within_8_fraction": float((error <= 8).float().mean()),
        "within_64_fraction": float((error <= 64).float().mean()),
    }


def run_trial(args, trial, output_dir, pilot=False):
    secret_x, target_tokens, shuffled_tokens = csprng_secret(args.input_dim, args.output_dim)
    config = {
        "input_dim": args.input_dim, "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim, "lr": args.train_lr,
        "epochs": args.epochs, "check_every": args.check_every,
    }
    seed_base = args.seed + trial + (0 if pilot else 10000)
    model, initial_state, training = train_with_initial(secret_x, target_tokens, config, seed_base)
    if not training["correct_key"]["all_exact"]:
        raise RuntimeError(f"trial {trial} did not reach exact correct-key recovery")
    trial_dir = output_dir / ("pilot" if pilot else "heldout") / f"trial_{trial:02d}"
    public_dir = trial_dir / "public"
    public_dir.mkdir(parents=True)
    torch.save(model.state_dict(), public_dir / "model.pt")
    (public_dir / "config.json").write_text(json.dumps({
        "input_dim": args.input_dim, "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim, "activation": "relu",
    }, indent=2))
    known_init_dir = trial_dir / "known_initialization"
    known_init_dir.mkdir()
    torch.save(initial_state, known_init_dir / "initial_model.pt")

    sparse = sparse_region_attack(model, args.starts, args.sparse_steps, args.attack_lr, seed_base + 1000)
    adaptive = adaptive_attack(
        model, args.starts, max(args.budgets), args.budgets,
        args.attack_lr, seed_base + 2000,
    )
    random_control = random_budget_attack(model, args.starts, args.budgets, seed_base + 3000)
    parameter_started = time.perf_counter()
    parameter_candidates = {
        "phase2_5_moment": parameter_only_candidate(model),
        "bias_only": tokens_from_output(model.net[-1].bias.detach().unsqueeze(0)),
        "midpoint": torch.full_like(target_tokens, 2048),
        "known_initialization_delta": known_initialization_candidate(model, initial_state),
    }
    parameter_elapsed = time.perf_counter() - parameter_started

    # Controls are trained/generated independently and evaluated only after their
    # target-free candidates are frozen.
    shuffled_model, shuffled_training = train_mapping(
        secret_x, shuffled_tokens, config, seed_base + 4000
    )
    shuffled_adaptive = adaptive_attack(
        shuffled_model, args.starts, max(args.budgets), [max(args.budgets)],
        args.attack_lr, seed_base + 5000,
    )
    torch.manual_seed(seed_base + 6000)
    untrained = Memory(args.input_dim, args.hidden_dim, args.output_dim).eval()

    sparse_selected = {"selected": sparse["selected"]}
    report = {
        "trial": trial, "pilot": pilot, "training_seed": seed_base,
        "training": training,
        "phase3_1_geometry": {
            "distinct_patterns": sparse["distinct_patterns"],
            "solution_count": args.starts,
            "mean_input_norm": float(sparse["input_norms"].mean()),
            "std_input_norm": float(sparse["input_norms"].std()),
            "mean_active_fraction": float(sparse["active_fractions"].mean()),
            "min_active_fraction": float(sparse["active_fractions"].min()),
            "mean_local_output_std": sparse["mean_local_output_std"],
            "elapsed_seconds": sparse["elapsed_seconds"],
            "forward_passes": sparse["forward_passes"],
            "backward_passes": sparse["backward_passes"],
            "selected_evaluation": evaluate_candidate_set(model, sparse_selected, target_tokens, secret_x)["selected"],
            "population_evaluation": evaluate_population(model, sparse["candidates"], target_tokens, secret_x),
            "matched_random_evaluation": evaluate_population(model, sparse["matched_random"], target_tokens, secret_x),
        },
        "phase3_2_adaptive": evaluate_candidate_set(model, adaptive["candidates"], target_tokens, secret_x),
        "phase3_2_random_control": evaluate_candidate_set(model, random_control["candidates"], target_tokens, secret_x),
        "phase3_2_observables": adaptive["observables"],
        "phase3_2_random_observables": random_control["observables"],
        "phase3_2_runtime_seconds": adaptive["runtimes"],
        "phase3_2_random_runtime_seconds": random_control["runtimes"],
        "phase3_3_parameter": {name: parameter_metrics(value, target_tokens) for name, value in parameter_candidates.items()},
        "phase3_3_resources": {
            "elapsed_seconds": parameter_elapsed,
            "forward_passes": 0,
            "backward_passes": 0,
        },
        "controls": {
            "shuffled_training": shuffled_training,
            "shuffled_adaptive": evaluate_candidate_set(
                shuffled_model, shuffled_adaptive["candidates"], shuffled_tokens, secret_x
            )[str(max(args.budgets))],
            "untrained_moment_vs_primary_target": parameter_metrics(
                parameter_only_candidate(untrained), target_tokens
            ),
        },
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    (trial_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def summarize_metrics(items):
    keys = [
        "mae_tokens", "rmse_tokens", "exact_fraction", "within_1_fraction",
        "within_8_fraction", "within_64_fraction",
    ]
    return {
        **{key: mean_ci([item[key] for item in items]) for key in keys},
        "full_recoveries": sum(item["all_exact"] for item in items),
    }


def aggregate(reports, args):
    budgets = [str(x) for x in args.budgets]
    return {
        "heldout_trials": len(reports),
        "correct_key_full_recoveries": sum(r["training"]["correct_key"]["all_exact"] for r in reports),
        "phase3_1": {
            "distinct_patterns": mean_ci([r["phase3_1_geometry"]["distinct_patterns"] for r in reports]),
            "mean_active_fraction": mean_ci([r["phase3_1_geometry"]["mean_active_fraction"] for r in reports]),
            "selected": summarize_metrics([r["phase3_1_geometry"]["selected_evaluation"] for r in reports]),
            "population_mae": mean_ci([r["phase3_1_geometry"]["population_evaluation"]["mean_mae_tokens"] for r in reports]),
            "matched_random_mae": mean_ci([r["phase3_1_geometry"]["matched_random_evaluation"]["mean_mae_tokens"] for r in reports]),
        },
        "phase3_2_gradient": {budget: summarize_metrics([r["phase3_2_adaptive"][budget] for r in reports]) for budget in budgets},
        "phase3_2_random": {budget: summarize_metrics([r["phase3_2_random_control"][budget] for r in reports]) for budget in budgets},
        "phase3_3": {
            name: summarize_metrics([r["phase3_3_parameter"][name] for r in reports])
            for name in reports[0]["phase3_3_parameter"]
        },
        "controls": {
            "shuffled_adaptive": summarize_metrics([r["controls"]["shuffled_adaptive"] for r in reports]),
            "untrained_moment": summarize_metrics([r["controls"]["untrained_moment_vs_primary_target"] for r in reports]),
        },
    }


def write_csv(reports, args, path):
    fields = ["trial", "attack", "budget", "mae_tokens", "rmse_tokens", "exact_fraction", "all_exact", "within_1_fraction", "within_8_fraction", "within_64_fraction", "elapsed_seconds", "forward_passes", "backward_passes", "peak_process_rss_bytes"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for attack_key, attack_name in (("phase3_2_adaptive", "gradient"), ("phase3_2_random_control", "random")):
                for budget in args.budgets:
                    key = str(budget)
                    metric = report[attack_key][key]
                    runtime_key = "phase3_2_runtime_seconds" if attack_name == "gradient" else "phase3_2_random_runtime_seconds"
                    writer.writerow({
                        "trial": report["trial"], "attack": attack_name, "budget": budget,
                        **{k: metric[k] for k in fields if k in metric},
                        "elapsed_seconds": report[runtime_key][key],
                        "forward_passes": args.starts * budget,
                        "backward_passes": args.starts * budget if attack_name == "gradient" else 0,
                        "peak_process_rss_bytes": report["peak_process_rss_bytes"],
                    })
            sparse = report["phase3_1_geometry"]
            writer.writerow({
                "trial": report["trial"], "attack": "sparse_region", "budget": args.sparse_steps,
                **{k: sparse["selected_evaluation"][k] for k in fields if k in sparse["selected_evaluation"]},
                "elapsed_seconds": sparse["elapsed_seconds"],
                "forward_passes": sparse["forward_passes"], "backward_passes": sparse["backward_passes"],
                "peak_process_rss_bytes": report["peak_process_rss_bytes"],
            })
            for name, metric in report["phase3_3_parameter"].items():
                writer.writerow({
                    "trial": report["trial"], "attack": name, "budget": "parameter",
                    **{k: metric[k] for k in fields if k in metric},
                    **report["phase3_3_resources"],
                    "peak_process_rss_bytes": report["peak_process_rss_bytes"],
                })


def write_attack_summary(aggregate_result, path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attack", "budget", "mean_mae_tokens", "mean_exact_fraction", "full_recoveries"])
        sparse = aggregate_result["phase3_1"]["selected"]
        writer.writerow(["sparse_region", "fixed", sparse["mae_tokens"]["mean"], sparse["exact_fraction"]["mean"], sparse["full_recoveries"]])
        for group in ("phase3_2_gradient", "phase3_2_random"):
            for budget, result in aggregate_result[group].items():
                writer.writerow([group, budget, result["mae_tokens"]["mean"], result["exact_fraction"]["mean"], result["full_recoveries"]])
        for name, result in aggregate_result["phase3_3"].items():
            writer.writerow([name, "parameter", result["mae_tokens"]["mean"], result["exact_fraction"]["mean"], result["full_recoveries"]])


def write_svg(aggregate_result, args, path):
    series = [("gradient", aggregate_result["phase3_2_gradient"], "#b22323"), ("random", aggregate_result["phase3_2_random"], "#2457c5")]
    max_passes = args.starts * max(args.budgets)
    content = []
    for name, data, color in series:
        points = []
        for budget in args.budgets:
            x = 60 + 520 * (args.starts * budget) / max_passes
            y = 330 - 260 * data[str(budget)]["within_64_fraction"]["mean"]
            points.append((x, y))
        content.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        content.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in points)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="390"><rect width="100%" height="100%" fill="white"/>
<text x="320" y="28" text-anchor="middle" font-size="18">Within-64 accuracy vs compute budget</text>
<line x1="60" y1="330" x2="590" y2="330" stroke="black"/><line x1="60" y1="60" x2="60" y2="330" stroke="black"/>
{"".join(content)}<text x="320" y="375" text-anchor="middle">forward-pass equivalents</text></svg>'''
    path.write_text(svg)


def main(args):
    torch.set_num_threads(args.threads)
    output_dir = Path(args.out)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    config = vars(args).copy()
    config["environment"] = environment_metadata(torch.device("cpu"))
    config["secret_generation"] = "OS CSPRNG; entropy not persisted"
    config["frozen_before_heldout"] = True
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    pilots = [run_trial(args, index, output_dir, pilot=True) for index in range(args.pilot_trials)]
    heldout = [run_trial(args, index, output_dir, pilot=False) for index in range(args.trials)]
    aggregate_result = aggregate(heldout, args)
    aggregate_result["pilot_trials"] = len(pilots)
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate_result, indent=2))
    write_csv(heldout, args, output_dir / "per_trial.csv")
    write_attack_summary(aggregate_result, output_dir / "attack_summary.csv")
    write_svg(aggregate_result, args, output_dir / "accuracy_vs_compute.svg")
    print(json.dumps(aggregate_result, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knc0_phase_3_results")
    parser.add_argument("--pilot-trials", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--sparse-steps", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 50, 200])
    parser.add_argument("--attack-lr", type=float, default=0.03)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--train-lr", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31000)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
