import inspect,torch
from experiments.knc_0.phase_4d import Memory,attacker_rank,evaluate_ranking
def test_attacker_api_has_no_truth():
 p=set(inspect.signature(attacker_rank).parameters);assert 'target' not in p and 'secret' not in p
def test_rank_and_evaluate_boundary():
 m=Memory(256,4,64).eval();ranking,d=attacker_rank(m,1,64,3,8,4,.01);assert set(ranking)=={'mode','median_medoid','local_stability','analytic_moment','random_first','zero'};assert d['distinct_arrays']>=1;target=torch.zeros((1,64),dtype=torch.long);result=evaluate_ranking(ranking,target,3);assert all('recall_at_k' in x for x in result.values())
def test_target_observation_does_not_change_rankings():
 m=Memory(256,4,64).eval();a,_=attacker_rank(m,2,32,2,4,2,.01);b,_=attacker_rank(m,2,32,2,4,2,.01);assert all(torch.equal(a[k],b[k]) for k in a)
