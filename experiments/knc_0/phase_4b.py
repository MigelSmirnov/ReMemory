#!/usr/bin/env python3
"""Phase 4B: multi-token capacity and target-free output leakage diagnostics."""

import argparse, csv, io, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.knc_0.run import Memory, environment_metadata, tokens_from_output  # noqa: E402

CANDIDATES = ("zero", "random_single", "random_mean", "random_median", "bias_only", "midpoint")


def secret_example(input_dim, output_dim):
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(os.urandom(32))))
    return (
        torch.tensor(rng.standard_normal((1, input_dim)), dtype=torch.float32),
        torch.tensor(rng.integers(0, 4096, (1, output_dim)), dtype=torch.long),
    )


def serialized_size(model):
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); return buffer.tell()


def attacker_diagnostics(model, seed, samples):
    """Public-only candidate generation; deliberately accepts no target or key."""
    generator = torch.Generator().manual_seed(seed)
    input_dim = model.net[0].in_features
    x = torch.randn(samples, input_dim, generator=generator)
    with torch.no_grad():
        h1 = torch.relu(model.net[0](x)); h2 = torch.relu(model.net[2](h1)); raw = model.net[4](h2)
        rounded = tokens_from_output(raw)
        candidates = {
            "zero": tokens_from_output(model(torch.zeros(1, input_dim))),
            "random_single": rounded[:1],
            "random_mean": tokens_from_output(raw.mean(dim=0, keepdim=True)),
            "random_median": tokens_from_output(raw.median(dim=0).values.unsqueeze(0)),
            "bias_only": tokens_from_output(model.net[-1].bias.unsqueeze(0)),
            "midpoint": torch.full((1, raw.shape[1]), 2048, dtype=torch.long),
        }
        std = rounded.float().std(dim=0)
        ranges = rounded.max(dim=0).values - rounded.min(dim=0).values
        dominant = []
        for coordinate in range(rounded.shape[1]):
            dominant.append(float(torch.bincount(rounded[:, coordinate], minlength=4096).max() / samples))
        dominant = torch.tensor(dominant)
        per_coordinate_constant = (std <= 1.0) & (dominant >= 0.99)
        diagnostics = {
            "mean_token_std": float(std.mean()),
            "mean_token_range": float(ranges.float().mean()),
            "mean_dominant_token_fraction": float(dominant.mean()),
            "constant_coordinate_fraction": float(per_coordinate_constant.float().mean()),
            "approximately_constant_array": bool(per_coordinate_constant.all()),
            "layer1_active_fraction": float((h1 > 0).float().mean()),
            "layer2_active_fraction": float((h2 > 0).float().mean()),
        }
    return candidates, diagnostics


def evaluate(candidate, target):
    error = (candidate - target).abs()
    return {
        "exact_token_count": int((error == 0).sum()),
        "exact_token_fraction": float((error == 0).float().mean()),
        "full_array_exact": bool((error == 0).all()),
        "mae_tokens": float(error.float().mean()),
        "within_1_fraction": float((error <= 1).float().mean()),
        "within_8_fraction": float((error <= 8).float().mean()),
        "within_64_fraction": float((error <= 64).float().mean()),
    }


def run_one(args, width, output_dim, trial, offset):
    secret_x, target = secret_example(256, output_dim)
    torch.manual_seed(args.seed + offset)
    model = Memory(256, width, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    normalized = target.float() / 4095.0
    started = time.perf_counter(); exact = False
    for step in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(secret_x), normalized)
        loss.backward(); optimizer.step()
        if (step + 1) % args.check_every == 0 or step + 1 == args.epochs:
            exact = bool(torch.equal(tokens_from_output(model(secret_x)), target))
            if exact: break
    elapsed = time.perf_counter() - started; model.eval()
    candidates, diagnostics = attacker_diagnostics(model, args.probe_seed + offset, args.samples)
    correct = evaluate(tokens_from_output(model(secret_x)), target)
    return {
        "hidden_width": width, "output_tokens": output_dim, "trial": trial,
        "parameter_count": sum(x.numel() for x in model.parameters()),
        "model_bytes": serialized_size(model), "training_steps": step + 1,
        "training_loss": float(loss.item()), "elapsed_seconds": elapsed,
        "correct_key": correct, "failed_to_converge": not exact,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        **diagnostics,
        "candidates": {name: evaluate(candidates[name], target) for name in CANDIDATES},
    }


def wilson(successes, total):
    z=1.96; p=successes/total if total else 0.0
    if not total: return {"numerator":0,"denominator":0,"rate":0.0,"ci95_low":0.0,"ci95_high":0.0}
    d=1+z*z/total; c=(p+z*z/(2*total))/d; h=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return {"numerator":successes,"denominator":total,"rate":p,"ci95_low":c-h,"ci95_high":c+h}


def mean_ci(values):
    a=np.asarray(values,float); mean=float(a.mean()); half=float(1.96*a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else 0.0
    return {"mean":mean,"ci95_low":mean-half,"ci95_high":mean+half}


def metric_summary(items, name):
    ms=[x["candidates"][name] for x in items]
    return {
        "exact_token_fraction": mean_ci([x["exact_token_fraction"] for x in ms]),
        "full_array_exact": wilson(sum(x["full_array_exact"] for x in ms),len(ms)),
        "mae_tokens": mean_ci([x["mae_tokens"] for x in ms]),
        **{key:wilson(sum(x[key] for x in ms),len(ms)) for key in ("within_1_fraction","within_8_fraction","within_64_fraction")},
    }


def aggregate(rows):
    groups={}
    for row in rows: groups.setdefault((row["hidden_width"],row["output_tokens"]),[]).append(row)
    configs=[]
    for (width,length),items in sorted(groups.items()):
        successful=[x for x in items if not x["failed_to_converge"]]
        configs.append({
            "hidden_width":width,"output_tokens":length,"trials":len(items),
            "parameter_count":items[0]["parameter_count"],"model_bytes":mean_ci([x["model_bytes"] for x in items]),
            "memorization_success":wilson(len(successful),len(items)),"training_steps":mean_ci([x["training_steps"] for x in items]),
            "elapsed_seconds":mean_ci([x["elapsed_seconds"] for x in items]),"failures":len(items)-len(successful),
            "approximately_constant_array":wilson(sum(x["approximately_constant_array"] for x in successful),len(successful)),
            "constant_coordinate_fraction":mean_ci([x["constant_coordinate_fraction"] for x in successful]),
            "mean_token_std":mean_ci([x["mean_token_std"] for x in successful]),
            "mean_dominant_token_fraction":mean_ci([x["mean_dominant_token_fraction"] for x in successful]),
            "layer1_active_fraction":mean_ci([x["layer1_active_fraction"] for x in successful]),
            "layer2_active_fraction":mean_ci([x["layer2_active_fraction"] for x in successful]),
            "candidates":{name:metric_summary(successful,name) for name in CANDIDATES},
        })
    return {"total_runs":len(rows),"configurations":configs}


def flatten(row):
    out={k:v for k,v in row.items() if k not in ("correct_key","candidates")}
    for k,v in row["correct_key"].items(): out[f"correct_key_{k}"]=v
    for name,metric in row["candidates"].items():
        for k,v in metric.items(): out[f"{name}_{k}"]=v
    return out


def write_csv(rows,path):
    vals=[flatten(x) for x in rows]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(vals[0]));w.writeheader();w.writerows(vals)


def write_svg(configs,getter,title,path):
    colors=["#2457c5","#218739","#b22323","#8b3fa0"]; widths=sorted({x["hidden_width"] for x in configs}); lengths=sorted({x["output_tokens"] for x in configs}); mx=max(getter(x) for x in configs) or 1; content=[]
    for width,color in zip(widths,colors):
        items=sorted((x for x in configs if x["hidden_width"]==width),key=lambda x:x["output_tokens"]); pts=[]
        for i,item in enumerate(items): pts.append((65+i*100,330-260*getter(item)/mx))
        content.append(f'<polyline points="{" ".join(f"{x},{y:.1f}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="2"/>'); content += [f'<circle cx="{x}" cy="{y:.1f}" r="4" fill="{color}"/>' for x,y in pts]
    labels="".join(f'<text x="{65+i*100}" y="350" text-anchor="middle">{v}</text>' for i,v in enumerate(lengths))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="390"><rect width="100%" height="100%" fill="white"/><text x="340" y="25" text-anchor="middle">{title}</text><line x1="55" y1="330" x2="650" y2="330" stroke="black"/><line x1="55" y1="45" x2="55" y2="330" stroke="black"/>{"".join(content)}{labels}<text x="340" y="380" text-anchor="middle">output tokens</text></svg>')


def main(args):
    torch.set_num_threads(args.threads); out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); config=vars(args).copy();config["environment"]=environment_metadata(torch.device("cpu"));config["constant_threshold"]={"per_coordinate_token_std_max":1.0,"dominant_fraction_min":0.99,"array_rule":"all coordinates"};config["secret_generation"]="OS CSPRNG; entropy not persisted";(out/"config.json").write_text(json.dumps(config,indent=2))
    rows=[];offset=0
    for width in args.widths:
        for length in args.output_lengths:
            for trial in range(args.trials): rows.append(run_one(args,width,length,trial,offset));offset+=1
    result=aggregate(rows);(out/"aggregate.json").write_text(json.dumps(result,indent=2));write_csv(rows,out/"per_run.csv");cs=result["configurations"]
    write_svg(cs,lambda x:x["candidates"]["zero"]["exact_token_fraction"]["mean"],"Zero-input exact-token fraction",out/"token_exact_vs_length.svg")
    write_svg(cs,lambda x:x["candidates"]["zero"]["full_array_exact"]["rate"],"Zero-input full-array recovery",out/"array_recovery_vs_length.svg")
    write_svg(cs,lambda x:x["memorization_success"]["rate"],"Correct-key memorization success",out/"memorization_vs_length.svg")
    write_svg(cs,lambda x:x["parameter_count"],"Parameter count",out/"parameters_vs_length.svg")
    print(json.dumps({"total_runs":len(rows),"failures":sum(x["failed_to_converge"] for x in rows)},indent=2))


def build_parser():
    p=argparse.ArgumentParser();p.add_argument("--out",default="knc0_phase_4b_results");p.add_argument("--widths",type=int,nargs="+",default=[1,2,4,8]);p.add_argument("--output-lengths",type=int,nargs="+",default=[1,2,4,8,16,64]);p.add_argument("--trials",type=int,default=30);p.add_argument("--samples",type=int,default=512);p.add_argument("--epochs",type=int,default=5000);p.add_argument("--check-every",type=int,default=100);p.add_argument("--lr",type=float,default=.01);p.add_argument("--threads",type=int,default=2);p.add_argument("--seed",type=int,default=43000);p.add_argument("--probe-seed",type=int,default=44000);return p


if __name__=="__main__": main(build_parser().parse_args())
