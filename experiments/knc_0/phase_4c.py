#!/usr/bin/env python3
"""Phase 4C: ReLU activation-pattern mechanism study."""
import argparse,csv,json,math,os,sys,time
from pathlib import Path
import numpy as np, torch
from torch import nn
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from experiments.knc_0.run import Memory,environment_metadata,tokens_from_output

class LinearControl(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Linear(256,64)
 def forward(self,x):return self.net(x)

def secret_example():
 rng=np.random.Generator(np.random.PCG64DXSM(int.from_bytes(os.urandom(32))))
 return torch.tensor(rng.standard_normal((1,256)),dtype=torch.float32),torch.tensor(rng.integers(0,4096,(1,64)),dtype=torch.long)

def encode_patterns(a1,a2):
 bits=torch.cat((a1,a2),dim=1).to(torch.int64);powers=(2**torch.arange(bits.shape[1],dtype=torch.int64)).unsqueeze(0)
 return (bits*powers).sum(dim=1)

def forward_with_patterns(model,x):
 p1=model.net[0](x);h1=torch.relu(p1);p2=model.net[2](h1);h2=torch.relu(p2);raw=model.net[4](h2)
 return raw,encode_patterns(p1>0,p2>0)

def h1_formal_regions(model):
 """Public exact feasibility/affine characterization for width one."""
 w1=model.net[0].weight[0].detach();b1=float(model.net[0].bias[0].detach());w2=float(model.net[2].weight[0,0].detach());b2=float(model.net[2].bias[0].detach());w3=model.net[4].weight[:,0].detach();b3=model.net[4].bias.detach()
 regions=[]
 for first in (0,1):
  for second in (0,1):
   if first==0:
    feasible=(b2>0)==bool(second);slope=torch.zeros_like(w1).repeat(64,1);intercept=b3+(w3*b2 if second else 0)
   else:
    if second:feasible=(w2>0) or (w2==0 and b2>0) or (w2<0 and b2>0)
    else:feasible=(w2<0) or (w2==0 and b2<=0) or (w2>0 and b2<0)
    if second:slope=w3[:,None]*w2*w1[None,:];intercept=b3+w3*(w2*b1+b2)
    else:slope=torch.zeros((64,256));intercept=b3
   regions.append({"pattern":first+2*second,"feasible":bool(feasible),"constant":bool(torch.all(slope==0)),"slope_norm":float(torch.linalg.vector_norm(slope)),"rounded_intercept":tokens_from_output(intercept.unsqueeze(0)).squeeze(0)})
 return regions

def public_analysis(model,seed,samples):
 g=torch.Generator().manual_seed(seed);x=torch.randn(samples,256,generator=g)
 with torch.no_grad():
  raw,patterns=forward_with_patterns(model,x);zero_raw,zero_pattern=forward_with_patterns(model,torch.zeros(1,256));rounded=tokens_from_output(raw)
  unique,counts=torch.unique(patterns,return_counts=True);entries=[]
  for pattern,count in zip(unique.tolist(),counts.tolist()):
   keep=patterns==pattern;entries.append({"pattern":pattern,"count":count,"frequency":count/samples,"mean_output_std":float(raw[keep].std(dim=0,unbiased=False).mean()),"dominant_array_fraction":float((rounded[keep]==rounded[keep][0]).all(dim=1).float().mean())})
  result={"zero_candidate":tokens_from_output(zero_raw),"zero_pattern":int(zero_pattern[0]),"patterns":entries,"distinct_patterns":len(entries),"mean_output_std":float(raw.std(dim=0).mean()),"random_tokens":rounded,"random_patterns":patterns,"layer1_active":float(sum(e["frequency"]*bin(e["pattern"]&((1<<model.net[0].out_features)-1)).count('1')/model.net[0].out_features for e in entries)),"layer2_active":float(sum(e["frequency"]*bin(e["pattern"]>>model.net[0].out_features).count('1')/model.net[0].out_features for e in entries))}
  if model.net[0].out_features==1:result["formal_regions"]=h1_formal_regions(model)
 return result

def score(candidate,target):
 err=(candidate-target).abs();return {"exact_fraction":float((err==0).float().mean()),"full_exact":bool((err==0).all()),"mae":float(err.float().mean())}

def train(model,x,target,args):
 opt=torch.optim.Adam(model.parameters(),lr=args.lr);norm=target.float()/4095;start=time.perf_counter();exact=False
 for step in range(args.epochs):
  opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(x),norm);loss.backward();opt.step()
  if (step+1)%args.check_every==0 or step+1==args.epochs:
   exact=bool(torch.equal(tokens_from_output(model(x)),target))
   if exact:break
 model.eval();return {"steps":step+1,"loss":float(loss.item()),"seconds":time.perf_counter()-start,"exact":exact}

def run_trial(args,width,trial,offset):
 x,target=secret_example();torch.manual_seed(args.seed+offset);model=Memory(256,width,64);training=train(model,x,target,args);analysis=public_analysis(model,args.probe_seed+offset,args.samples)
 with torch.no_grad():_,secret_pattern=forward_with_patterns(model,x);secret_pattern=int(secret_pattern[0]);random_full=(analysis["random_tokens"]==target).all(dim=1);same=analysis["random_patterns"]==secret_pattern
 zero=score(analysis["zero_candidate"],target);pattern_entry=next((p for p in analysis["patterns"] if p["pattern"]==secret_pattern),None);formal=None
 if width==1:formal=next(r for r in analysis["formal_regions"] if r["pattern"]==analysis["zero_pattern"])
 if zero["full_exact"] and formal and formal["constant"]:explanation="constant_region"
 elif zero["full_exact"] and analysis["zero_pattern"]==secret_pattern:explanation="shared_pattern_nonconstant"
 elif zero["full_exact"]:explanation="rounding_coincidence"
 else:explanation="no_zero_recovery"
 torch.manual_seed(args.untrained_seed+offset);untrained=Memory(256,width,64).eval();ua=public_analysis(untrained,args.probe_seed+50000+offset,args.samples)
 return {"width":width,"trial":trial,"parameters":sum(p.numel() for p in model.parameters()),"training":training,"distinct_patterns":analysis["distinct_patterns"],"zero_pattern":analysis["zero_pattern"],"secret_pattern_evaluator_only":secret_pattern,"zero_secret_pattern_agreement_evaluator_only":analysis["zero_pattern"]==secret_pattern,"random_secret_pattern_agreement_evaluator_only":float(same.float().mean()),"random_full_recovery":float(random_full.float().mean()),"random_full_recovery_same_secret_pattern":float(random_full[same].float().mean()) if same.any() else None,"mean_output_std":analysis["mean_output_std"],"layer1_active":analysis["layer1_active"],"layer2_active":analysis["layer2_active"],"zero":zero,"explanation":explanation,"pattern_frequencies":analysis["patterns"],"formal_regions":analysis.get("formal_regions"),"secret_region_observed_frequency":pattern_entry["frequency"] if pattern_entry else 0.0,"untrained_distinct_patterns":ua["distinct_patterns"],"untrained_output_std":ua["mean_output_std"]}

def run_linear(args,trial,offset):
 x,target=secret_example();torch.manual_seed(args.seed+offset);model=LinearControl();training=train(model,x,target,args);g=torch.Generator().manual_seed(args.probe_seed+offset)
 with torch.no_grad():raw=model(torch.randn(args.samples,256,generator=g));zero=tokens_from_output(model(torch.zeros(1,256)));rounded=tokens_from_output(raw)
 return {"trial":trial,"training":training,"zero":score(zero,target),"mean_output_std":float(raw.std(dim=0).mean()),"random_full_recovery":float((rounded==target).all(dim=1).float().mean())}

def wilson(k,n):
 if not n:return {"numerator":0,"denominator":0,"rate":0.0,"ci95_low":0.0,"ci95_high":0.0}
 z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return {"numerator":k,"denominator":n,"rate":p,"ci95_low":c-h,"ci95_high":c+h}
def mean_ci(v):
 a=np.asarray(v,float);m=float(a.mean());h=float(1.96*a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else 0;return {"mean":m,"ci95_low":m-h,"ci95_high":m+h}

def aggregate(rows,linear):
 out=[]
 for width in sorted({r["width"] for r in rows}):
  x=[r for r in rows if r["width"]==width];out.append({"width":width,"trials":len(x),"training_success":wilson(sum(r["training"]["exact"] for r in x),len(x)),"zero_full_recovery":wilson(sum(r["zero"]["full_exact"] for r in x),len(x)),"zero_exact_fraction":mean_ci([r["zero"]["exact_fraction"] for r in x]),"distinct_patterns":mean_ci([r["distinct_patterns"] for r in x]),"random_full_recovery":mean_ci([r["random_full_recovery"] for r in x]),"zero_secret_pattern_agreement_evaluator_only":wilson(sum(r["zero_secret_pattern_agreement_evaluator_only"] for r in x),len(x)),"random_secret_pattern_agreement_evaluator_only":mean_ci([r["random_secret_pattern_agreement_evaluator_only"] for r in x]),"mean_output_std":mean_ci([r["mean_output_std"] for r in x]),"explanations":{name:sum(r["explanation"]==name for r in x) for name in ("constant_region","shared_pattern_nonconstant","rounding_coincidence","no_zero_recovery")},"models_with_constant_formal_region":sum(any(z["feasible"] and z["constant"] for z in r["formal_regions"]) for r in x) if width==1 else None})
 return {"configurations":out,"linear_control":{"trials":len(linear),"zero_full_recovery":wilson(sum(r["zero"]["full_exact"] for r in linear),len(linear)),"mean_output_std":mean_ci([r["mean_output_std"] for r in linear])}}

def flatten(r):
 out={k:v for k,v in r.items() if k not in ("training","zero","pattern_frequencies","formal_regions")};out.update({f"training_{k}":v for k,v in r["training"].items()});out.update({f"zero_{k}":v for k,v in r["zero"].items()});out["pattern_frequencies_json"]=json.dumps(r["pattern_frequencies"],separators=(',',':'));out["formal_regions_json"]=json.dumps(r["formal_regions"],default=lambda x:x.tolist() if hasattr(x,'tolist') else x,separators=(',',':'));return out
def write_csv(rows,path):
 vals=[flatten(r) for r in rows]
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(vals[0]));w.writeheader();w.writerows(vals)
def write_svg(configs,path):
 pts=[]
 for i,c in enumerate(configs):pts.append((80+i*150,330-260*c["zero_full_recovery"]["rate"]))
 path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="390"><rect width="100%" height="100%" fill="white"/><text x="320" y="25" text-anchor="middle">Zero-input full recovery by width</text><line x1="60" y1="330" x2="600" y2="330" stroke="black"/><polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="#2457c5" stroke-width="2"/>{"".join(f"<circle cx=\"{x}\" cy=\"{y}\" r=\"4\"/>" for x,y in pts)}</svg>')

def main(args):
 torch.set_num_threads(args.threads);out=Path(args.out)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);cfg=vars(args).copy();cfg["environment"]=environment_metadata(torch.device('cpu'));cfg["secret_generation"]="OS CSPRNG; entropy not persisted";(out/'config.json').write_text(json.dumps(cfg,indent=2));rows=[];offset=0
 for width in args.widths:
  for trial in range(args.trials):rows.append(run_trial(args,width,trial,offset));offset+=1
 linear=[]
 for trial in range(args.trials):linear.append(run_linear(args,trial,offset));offset+=1
 agg=aggregate(rows,linear);(out/'aggregate.json').write_text(json.dumps(agg,indent=2));write_csv(rows,out/'per_trial.csv');(out/'linear_control.json').write_text(json.dumps(linear,indent=2));write_svg(agg['configurations'],out/'recovery_by_width.svg');print(json.dumps({"trials":len(rows),"linear":len(linear),"failures":sum(not r['training']['exact'] for r in rows)},indent=2))
def parser():
 p=argparse.ArgumentParser();p.add_argument('--out',default='knc0_phase_4c_results');p.add_argument('--widths',type=int,nargs='+',default=[1,2,4,8]);p.add_argument('--trials',type=int,default=30);p.add_argument('--samples',type=int,default=4096);p.add_argument('--epochs',type=int,default=5000);p.add_argument('--check-every',type=int,default=100);p.add_argument('--lr',type=float,default=.01);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=45000);p.add_argument('--probe-seed',type=int,default=46000);p.add_argument('--untrained-seed',type=int,default=47000);return p
if __name__=='__main__':main(parser().parse_args())
