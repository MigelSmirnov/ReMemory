import inspect

import torch

from experiments.knc_0.phase_4a import (
    SingleTokenMLP, candidate_metrics, parameter_count_formula, public_candidates,
)


def test_parameter_formula_matches_model():
    for input_dim, width in ((1, 1), (4, 8), (256, 64)):
        model = SingleTokenMLP(input_dim, width)
        assert sum(p.numel() for p in model.parameters()) == parameter_count_formula(input_dim, width)


def test_candidate_generation_has_no_target_argument():
    parameters = set(inspect.signature(public_candidates).parameters)
    assert "target" not in parameters
    model = SingleTokenMLP(4, 2).eval()
    candidates = public_candidates(model, 4, 1, 8, 3)
    assert candidates["zero"].shape == (1, 1)


def test_single_token_metrics():
    target = torch.tensor([[100]])
    assert candidate_metrics(torch.tensor([[100]]), target)["exact"]
    assert candidate_metrics(torch.tensor([[164]]), target)["within_64"]
    assert not candidate_metrics(torch.tensor([[165]]), target)["within_64"]
