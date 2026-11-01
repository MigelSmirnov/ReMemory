import inspect, torch
from experiments.knc_0.phase_4b import Memory, attacker_diagnostics, evaluate, wilson

def test_attacker_boundary():
    params=set(inspect.signature(attacker_diagnostics).parameters);assert "target" not in params and "secret_x" not in params

def test_multi_token_diagnostics():
    model=Memory(256,2,4).eval();candidates,diagnostics=attacker_diagnostics(model,1,32)
    assert candidates["zero"].shape==(1,4);assert 0<=diagnostics["constant_coordinate_fraction"]<=1

def test_evaluator_metrics_and_wilson():
    target=torch.tensor([[1,2]]);result=evaluate(torch.tensor([[1,3]]),target)
    assert result["exact_token_count"]==1 and not result["full_array_exact"]
    assert wilson(3,10)["numerator"]==3
