#!/usr/bin/env python3
"""Phase 4D: candidate occurrence versus target-free identification."""
import argparse,csv,json,math,os,sys,time
from pathlib import Path
import numpy as np,torch
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from experiments.knc_0.phase_2_5 import parameter_only_candidate
from experiments.knc_0.run import Memory,environment_metadata,tokens_from_output

METHODS=("mode","median_medoid","local_stability","analytic_moment","random_first","zero")

def secret_example():
 rng=np.random.Generator(np.random.PCG64DXSM(int.from_bytes(os.urandom(32))))
 return torch.tensor(rng.standard_normal((1,256)),dtype=torch.float32),torch.tensor(rng.integers(0,4096,(1,64)),dtype=torch.long)

def patterns(model,x):
 h1=torch.relu(model.net[0](x));h2=torch.relu(model.net[2](h1));bits=torch.cat((h1>0,h2>0),dim=1).to(torch.int64)
 # Bytes avoid integer overflow for widths 32/64.
 return [bytes(row.to(torch.uint8).tolist()) for row in bits]

def train_model(model,x,target,args):
 opt=torch.optim.Adam(model.parameters(),lr=args.lr);normalized=target.float()/4095;start=time.perf_counter();exact=False
 for step in range(args.epochs):
  opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(x),normalized);loss.backward();opt.step()
  if (step+1)%args.check_every==0 or step+1==args.epochs:
   exact=bool(torch.equal(tokens_from_output(model(x)),target))
   if exact:break
 model.eval();return {"exact":exact,"steps":step+1,"loss":float(loss.item()),"seconds":time.perf_counter()-start}

def attacker_rank(model,seed,samples,top_k,local_candidates,local_probes,perturbation):
 """Public-only generation and ranking. No truth/key argument is accepted."""
 g=torch.Generator().manual_seed(seed);x=torch.randn(samples,256,generator=g);start=time.perf_counter()
 with torch.no_grad():
  raw=model(x);rounded=tokens_from_output(raw);unique,inverse,counts=torch.unique(rounded,dim=0,return_inverse=True,return_counts=True)
  order=torch.argsort(counts,descending=True);mode=unique[order[:top_k]]
  median=rounded.float().median(dim=0).values;dist=(unique.float()-median).abs().mean(dim=1);medoid=unique[torch.argsort(dist)[:top_k]]
  stable_pool=order[:min(local_candidates,len(order))];stable=[]
  for uid in stable_pool.tolist():
   representative=int((inverse==uid).nonzero()[0]);base=x[representative:representative+1];near=base+perturbation*torch.randn(local_probes,256,generator=g);variance=float(model(near).var(dim=0,unbiased=False).mean());stable.append((variance,unique[uid]))
  stable.sort(key=lambda z:z[0]);local=torch.stack([z[1] for z in stable[:top_k]])
  analytic=parameter_only_candidate(model)
  zero=tokens_from_output(model(torch.zeros(1,256)))
  ranking={"mode":mode,"median_medoid":medoid,"local_stability":local,"analytic_moment":analytic,"random_first":rounded[:1],"zero":zero}
  pattern_values=patterns(model,x)
  diagnostics={"distinct_arrays":int(len(unique)),"max_array_frequency":int(counts.max()),"max_array_fraction":float(counts.max()/samples),"distinct_patterns":len(set(pattern_values)),"elapsed_seconds":time.perf_counter()-start,"sampled_arrays":rounded}
 return ranking,diagnostics

def evaluate_ranking(ranking,target,top_k):
 result={}
 for name,candidates in ranking.items():
  matches=(candidates==target).all(dim=1);indices=matches.nonzero().reshape(-1);rank=int(indices[0])+1 if len(indices) else None
  result[name]={"recall_at_1":bool(len(matches) and matches[0]),"recall_at_k":bool(matches[:top_k].any()),"reciprocal_rank":0.0 if rank is None else 1.0/rank,"rank":rank}
 return result

def run_trial(args,width,trial,offset,pilot=False):
 secret_x,target=secret_example();seed=args.seed+offset;torch.manual_seed(seed);model=Memory(256,width,64);training=train_model(model,secret_x,target,args)
 ranking,diag=attacker_rank(model,args.probe_seed+offset,args.samples,args.top_k,args.local_candidates,args.local_probes,args.perturbation)
 # Freeze complete here; evaluator begins below.
 evaluator=evaluate_ranking(ranking,target,args.top_k);sample_matches=(diag.pop("sampled_arrays")==target).all(dim=1);occurred=bool(sample_matches.any());occurrences=int(sample_matches.sum())
 torch.manual_seed(args.negative_seed+offset);negative=Memory(256,width,64).eval();_,negative_diag=attacker_rank(negative,args.negative_probe_seed+offset,args.samples,args.top_k,args.local_candidates,args.local_probes,args.perturbation);negative_diag.pop("sampled_arrays")
 identified=any(v["recall_at_k"] for v in evaluator.values())
 if not occurred:failure="target_never_observed"
 elif evaluator["mode"]["recall_at_k"]:failure="mode_identified"
 else:failure="target_observed_not_mode_identified"
 return {"width":width,"trial":trial,"pilot":pilot,"parameters":sum(p.numel() for p in model.parameters()),"training":training,"occurrence":{"occurred":occurred,"count":occurrences,"fraction":occurrences/args.samples},"candidate_diagnostics":diag,"rankings":evaluator,"failure_decomposition":failure,"any_method_recall_at_k":identified,"negative_control":negative_diag}

def wilson(k,n):
 if not n:return {"numerator":0,"denominator":0,"rate":0.0,"ci95_low":0.0,"ci95_high":0.0}
 z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return {"numerator":k,"denominator":n,"rate":p,"ci95_low":c-h,"ci95_high":c+h}
def mean_ci(v):
 a=np.asarray(v,float);m=float(a.mean());h=float(1.96*a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else 0;return {"mean":m,"ci95_low":m-h,"ci95_high":m+h}

def aggregate(rows):
 configs=[]
 for width in sorted({r["width"] for r in rows}):
  x=[r for r in rows if r["width"]==width];methods={}
  for name in METHODS:methods[name]={"recall_at_1":wilson(sum(r["rankings"][name]["recall_at_1"] for r in x),len(x)),"recall_at_k":wilson(sum(r["rankings"][name]["recall_at_k"] for r in x),len(x)),"mrr":mean_ci([r["rankings"][name]["reciprocal_rank"] for r in x])}
  configs.append({"width":width,"parameters":x[0]["parameters"],"trials":len(x),"memorization_success":wilson(sum(r["training"]["exact"] for r in x),len(x)),"occurrence_models":wilson(sum(r["occurrence"]["occurred"] for r in x),len(x)),"occurrence_fraction":mean_ci([r["occurrence"]["fraction"] for r in x]),"distinct_arrays":mean_ci([r["candidate_diagnostics"]["distinct_arrays"] for r in x]),"distinct_patterns":mean_ci([r["candidate_diagnostics"]["distinct_patterns"] for r in x]),"runtime_seconds":mean_ci([r["candidate_diagnostics"]["elapsed_seconds"] for r in x]),"methods":methods,"failure_decomposition":{name:sum(r["failure_decomposition"]==name for r in x) for name in ("target_never_observed","target_observed_not_mode_identified","mode_identified")},"negative_control":{"distinct_arrays":mean_ci([r["negative_control"]["distinct_arrays"] for r in x]),"max_array_fraction":mean_ci([r["negative_control"]["max_array_fraction"] for r in x])}})
 return {"configurations":configs}

def flatten(r):
 out={"width":r["width"],"trial":r["trial"],"parameters":r["parameters"],**{f"training_{k}":v for k,v in r["training"].items()},**{f"occurrence_{k}":v for k,v in r["occurrence"].items()},**r["candidate_diagnostics"],"failure_decomposition":r["failure_decomposition"]}
 for name,v in r["rankings"].items():
  for k,z in v.items():out[f"{name}_{k}"]=z
 return out
def write_csv(rows,path):
 vals=[flatten(r) for r in rows]
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(vals[0]));w.writeheader();w.writerows(vals)
def write_rankings(agg,path):
 with path.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(["width","method","recall_at_1","recall_at_k","mrr"])
  for c in agg['configurations']:
   for name,v in c['methods'].items():w.writerow([c['width'],name,v['recall_at_1']['rate'],v['recall_at_k']['rate'],v['mrr']['mean']])
def svg(agg,getter,title,path):
 pts=[]
 for i,c in enumerate(agg['configurations']):pts.append((80+i*120,330-260*getter(c)))
 path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="390"><rect width="100%" height="100%" fill="white"/><text x="340" y="25" text-anchor="middle">{title}</text><line x1="60" y1="330" x2="650" y2="330" stroke="black"/><polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="#2457c5" stroke-width="2"/>{"".join(f"<circle cx=\"{x}\" cy=\"{y}\" r=\"4\"/>" for x,y in pts)}</svg>')

def main(args):
 torch.set_num_threads(args.threads);out=Path(args.out)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);cfg=vars(args).copy();cfg["environment"]=environment_metadata(torch.device('cpu'));cfg["secret_generation"]="OS CSPRNG; entropy not persisted";cfg["pilot_disjoint_from_heldout"]=True;(out/'config.json').write_text(json.dumps(cfg,indent=2));offset=0;pilots=[]
 for width in args.widths:
  for trial in range(args.pilot_trials):pilots.append(run_trial(args,width,trial,offset,True));offset+=1
 rows=[]
 for width in args.widths:
  for trial in range(args.trials):rows.append(run_trial(args,width,trial,offset));offset+=1
 agg=aggregate(rows);agg['pilot_trials']=len(pilots);(out/'aggregate.json').write_text(json.dumps(agg,indent=2));write_csv(rows,out/'per_trial.csv');write_rankings(agg,out/'ranking_summary.csv');(out/'pilot_summary.json').write_text(json.dumps(aggregate(pilots),indent=2));svg(agg,lambda c:c['occurrence_models']['rate'],'Models where target occurred',out/'occurrence_vs_width.svg');svg(agg,lambda c:c['methods']['mode']['recall_at_k']['rate'],'Mode Recall@10',out/'identification_vs_width.svg');svg(agg,lambda c:c['distinct_arrays']['mean']/args.samples,'Candidate diversity fraction',out/'diversity_vs_width.svg');svg(agg,lambda c:c['runtime_seconds']['mean']/max(z['runtime_seconds']['mean'] for z in agg['configurations']),'Relative sampling runtime',out/'runtime_vs_width.svg');print(json.dumps({"heldout":len(rows),"failures":sum(not r['training']['exact'] for r in rows)},indent=2))
def parser():
 p=argparse.ArgumentParser();p.add_argument('--out',default='knc0_phase_4d_results');p.add_argument('--widths',type=int,nargs='+',default=[4,8,16,32,64]);p.add_argument('--pilot-trials',type=int,default=3);p.add_argument('--trials',type=int,default=30);p.add_argument('--samples',type=int,default=4096);p.add_argument('--top-k',type=int,default=10);p.add_argument('--local-candidates',type=int,default=64);p.add_argument('--local-probes',type=int,default=32);p.add_argument('--perturbation',type=float,default=.01);p.add_argument('--epochs',type=int,default=5000);p.add_argument('--check-every',type=int,default=100);p.add_argument('--lr',type=float,default=.01);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=48000);p.add_argument('--probe-seed',type=int,default=49000);p.add_argument('--negative-seed',type=int,default=50000);p.add_argument('--negative-probe-seed',type=int,default=51000);return p
if __name__=='__main__':main(parser().parse_args())
