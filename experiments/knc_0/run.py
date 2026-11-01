#!/usr/bin/env python3
"""KNC-0: single-association ReLU memory and white-box leakage baselines.

Run: python experiments/knc_0/run.py --epochs 5000
Requires: torch, numpy. CPU by default.
Public artifacts contain ONLY model weights and architecture.
"""
import argparse
import json
import time
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
    guess = guess.reshape(-1)
    truth = truth.reshape(-1)
    err = (guess - truth).abs()
    return {
        "exact_fraction": float((guess == truth).float().mean()),
        "all_exact": bool(torch.equal(guess, truth)),
        "mae_tokens": float(err.float().mean()),
    }


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.out)
    public_dir = output_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)

    # The attacker never receives these ground-truth values.
    secret_x = torch.tensor(rng.standard_normal((1, 256)), dtype=torch.float32)
    target_tokens = torch.tensor(rng.integers(0, 4096, size=(1, 64)), dtype=torch.long)
    target = target_tokens.float() / 4095.0

    model = Memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    for step in range(args.epochs):
        optimizer.zero_grad()
        pred = model(secret_x)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                exact = metrics(tokens_from_output(model(secret_x)), target_tokens)["all_exact"]
            print(f"step={step+1} mse={loss.item():.8g} all_exact={exact}")
            if exact:
                break
    training_seconds = time.perf_counter() - start
    model.eval()

    torch.save(model.state_dict(), public_dir / "model.pt")
    (public_dir / "config.json").write_text(
        json.dumps({"input_dim": 256, "hidden_dim": 64, "output_dim": 64}, indent=2)
    )

    # Simulate a fresh attacker process: load ONLY public files.
    attacker = Memory(**json.loads((public_dir / "config.json").read_text()))
    attacker.load_state_dict(torch.load(public_dir / "model.pt", map_location="cpu", weights_only=True))
    attacker.eval()
    attack_start = time.perf_counter()
    with torch.no_grad():
        random_x = torch.randn(args.probes, 256)
        raw = attacker(random_x)
        guesses = {
            "zero_input": tokens_from_output(attacker(torch.zeros(1, 256))),
            "random_first": tokens_from_output(raw[:1]),
            "random_mean": tokens_from_output(raw.mean(dim=0, keepdim=True)),
            "output_bias_only": tokens_from_output(attacker.net[-1].bias.unsqueeze(0)),
        }
        # Evaluate every random probe; this is a diagnostic, not an attack oracle.
        probes = tokens_from_output(raw)
        exact_per_probe = (probes == target_tokens).float().mean(dim=1)
        best_probe_idx = int(exact_per_probe.argmax())
        guesses["best_random_probe_oracle"] = probes[best_probe_idx:best_probe_idx+1]

    # SVD describes weights but cannot directly identify the secret without
    # known initialization or a validated recovery algorithm.
    singular = torch.linalg.svdvals(attacker.net[0].weight.detach())
    attack_seconds = time.perf_counter() - attack_start
    result = {
        "seed": args.seed,
        "steps": step + 1,
        "train_seconds": training_seconds,
        "attack_seconds": attack_seconds,
        "probes": args.probes,
        "correct_key": metrics(tokens_from_output(model(secret_x)), target_tokens),
        "attacks": {name: metrics(guess, target_tokens) for name, guess in guesses.items()},
        "first_layer_top_singular_values": [float(x) for x in singular[:8]],
        "note": "best_random_probe_oracle uses ground truth to select the best probe: evaluation upper bound, NOT an attacker-available strategy.",
    }
    # No secret key or target tokens are written to disk.
    (output_dir / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--probes", type=int, default=10000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="knc0_results")
    run(p.parse_args())
