#!/usr/bin/env python3
"""KNC-0 reproducible single-association leakage baseline."""

import argparse
import csv
import json
import os
import platform
import random
import resource
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Memory(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def tokens_from_output(output):
    return (output * 4095).round().clamp(0, 4095).long()


def metrics(guess, truth):
    guess = guess.reshape(-1).cpu()
    truth = truth.reshape(-1).cpu()
    err = (guess - truth).abs()
    return {
        "exact_fraction": float((guess == truth).float().mean()),
        "all_exact": bool(torch.equal(guess, truth)),
        "mae_tokens": float(err.float().mean()),
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(requested)


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(device):
    data = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cpu_count": os.cpu_count(),
        "git_revision": git_revision(),
    }
    if device.type == "cuda":
        data["gpu"] = torch.cuda.get_device_name(device)
        data["cuda"] = torch.version.cuda
    return data


def fresh_output_dir(base, run_id, seed):
    path = Path(base) / run_id / f"seed_{seed}"
    if path.exists():
        raise FileExistsError(f"refusing to reuse trial directory: {path}")
    (path / "public").mkdir(parents=True)
    return path


def run_trial(args, run_id, seed, device):
    seed_everything(seed)
    output_dir = fresh_output_dir(args.out, run_id, seed)
    public_dir = output_dir / "public"
    rng = np.random.default_rng(seed)
    secret_x = torch.tensor(
        rng.standard_normal((1, args.input_dim)), dtype=torch.float32, device=device
    )
    target_tokens = torch.tensor(
        rng.integers(0, 4096, size=(1, args.output_dim)), dtype=torch.long, device=device
    )
    target = target_tokens.float() / 4095.0

    model = Memory(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for step in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(secret_x)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        if (step + 1) % args.check_every == 0 or step + 1 == args.epochs:
            with torch.no_grad():
                exact = metrics(tokens_from_output(model(secret_x)), target_tokens)["all_exact"]
            if not args.quiet:
                print(f"seed={seed} step={step + 1} mse={loss.item():.8g} all_exact={exact}")
            if exact:
                break
    training_seconds = time.perf_counter() - start
    model.eval()

    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, public_dir / "model.pt")
    public_config = {
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "activation": "relu",
    }
    (public_dir / "config.json").write_text(json.dumps(public_config, indent=2))

    # This construction consumes only public files. Hidden truth is used by the
    # evaluator below solely to score guesses.
    attacker = Memory(**{k: public_config[k] for k in ("input_dim", "hidden_dim", "output_dim")})
    attacker.load_state_dict(
        torch.load(public_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    attacker.to(device).eval()
    attack_start = time.perf_counter()
    with torch.no_grad():
        random_x = torch.randn(args.probes, args.input_dim, device=device)
        raw = attacker(random_x)
        guesses = {
            "zero_input": tokens_from_output(attacker(torch.zeros(1, args.input_dim, device=device))),
            "random_first": tokens_from_output(raw[:1]),
            "random_mean": tokens_from_output(raw.mean(dim=0, keepdim=True)),
            "output_bias_only": tokens_from_output(attacker.net[-1].bias.unsqueeze(0)),
        }
        probes = tokens_from_output(raw)
        exact_per_probe = (probes == target_tokens).float().mean(dim=1)
        best_probe_idx = int(exact_per_probe.argmax())
        guesses["best_random_probe_oracle"] = probes[best_probe_idx:best_probe_idx + 1]
        untrained = Memory(args.input_dim, args.hidden_dim, args.output_dim).to(device).eval()
        guesses["untrained_random_first"] = tokens_from_output(untrained(random_x[:1]))

    first_weight = attacker.net[0].weight.detach()
    singular = torch.linalg.svdvals(first_weight)
    all_weights = torch.cat([p.detach().reshape(-1).float() for p in attacker.parameters()])
    attack_seconds = time.perf_counter() - attack_start
    model_path = public_dir / "model.pt"
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": seed,
        "config": {
            "epochs": args.epochs, "lr": args.lr, "probes": args.probes,
            "threads": args.threads, **public_config,
        },
        "environment": environment_metadata(device),
        "steps": step + 1,
        "final_training_loss": float(loss.item()),
        "train_seconds": training_seconds,
        "attack_seconds": attack_seconds,
        "model_bytes": model_path.stat().st_size,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "correct_key": metrics(tokens_from_output(model(secret_x)), target_tokens),
        "attacks": {name: metrics(guess, target_tokens) for name, guess in guesses.items()},
        "weight_statistics": {
            "mean": float(all_weights.mean()), "std": float(all_weights.std()),
            "min": float(all_weights.min()), "max": float(all_weights.max()),
        },
        "first_layer_top_singular_values": [float(x) for x in singular[:8].cpu()],
        "note": "best_random_probe_oracle uses hidden truth only for evaluation; it is not attacker-available.",
    }
    (output_dir / "report.json").write_text(json.dumps(result, indent=2))
    return result


def write_summary(results, path):
    fields = [
        "run_id", "seed", "steps", "final_training_loss", "train_seconds",
        "attack_seconds", "model_bytes", "correct_exact_fraction", "correct_all_exact",
        "best_random_exact_fraction", "best_random_all_exact",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "run_id": result["run_id"], "seed": result["seed"],
                "steps": result["steps"], "final_training_loss": result["final_training_loss"],
                "train_seconds": result["train_seconds"], "attack_seconds": result["attack_seconds"],
                "model_bytes": result["model_bytes"],
                "correct_exact_fraction": result["correct_key"]["exact_fraction"],
                "correct_all_exact": result["correct_key"]["all_exact"],
                "best_random_exact_fraction": result["attacks"]["best_random_probe_oracle"]["exact_fraction"],
                "best_random_all_exact": result["attacks"]["best_random_probe_oracle"]["all_exact"],
            })


def main(args):
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    results = [run_trial(args, run_id, args.seed + index, device) for index in range(args.trials)]
    run_dir = Path(args.out) / run_id
    write_summary(results, run_dir / "summary.csv")
    aggregate = {
        "run_id": run_id,
        "trials": len(results),
        "successful_exact_recovery": sum(r["correct_key"]["all_exact"] for r in results),
        "attacker_full_recoveries": {
            name: sum(r["attacks"][name]["all_exact"] for r in results)
            for name in results[0]["attacks"]
        },
        "mean_train_seconds": float(np.mean([r["train_seconds"] for r in results])),
    }
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="first trial seed")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--probes", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--run-id")
    parser.add_argument("--out", default="knc0_results")
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
