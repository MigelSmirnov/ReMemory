import inspect,torch
from experiments.knc_0.phase_4c import Memory,public_analysis,h1_formal_regions,forward_with_patterns
def test_public_boundary():assert 'target' not in inspect.signature(public_analysis).parameters and 'secret' not in inspect.signature(public_analysis).parameters
def test_width_one_enumeration():
 m=Memory(256,1,64).eval();regions=h1_formal_regions(m);assert len(regions)==4;assert {r['pattern'] for r in regions}=={0,1,2,3};assert all(r['rounded_intercept'].shape==(64,) for r in regions)
def test_pattern_analysis_smoke():
 m=Memory(256,2,64).eval();a=public_analysis(m,1,32);assert 1<=a['distinct_patterns']<=16;raw,p=forward_with_patterns(m,torch.zeros(1,256));assert raw.shape==(1,64) and p.shape==(1,)
