import inspect

import torch

from experiments.knc_0.phase_4a import SingleTokenMLP
from experiments.knc_0.phase_4a_1 import attacker_diagnostics, score_candidate, wilson


def test_attacker_diagnostics_has_no_secret_or_target_parameters():
    parameters = set(inspect.signature(attacker_diagnostics).parameters)
    assert "target" not in parameters
    assert "secret_x" not in parameters


def test_constant_diagnostics_and_candidates_smoke():
    model = SingleTokenMLP(4, 2).eval()
    candidates, diagnostics = attacker_diagnostics(model, 4, 1, 32, 0.01)
    assert "random_median" in candidates
    assert 0 <= diagnostics["dominant_token_fraction"] <= 1
    assert isinstance(diagnostics["approximately_constant"], bool)


def test_exact_thresholds_are_distinct():
    target = torch.tensor([[100]])
    result = score_candidate(torch.tensor([[101]]), target)
    assert not result["exact"] and result["within_1"]
    interval = wilson(5, 10)
    assert interval["numerator"] == 5 and interval["denominator"] == 10
