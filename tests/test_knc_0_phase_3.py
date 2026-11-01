import inspect

import torch

from experiments.knc_0.phase_3 import (
    Memory, adaptive_attack, evaluate_candidate_set, known_initialization_candidate,
    public_objective, random_budget_attack, sparse_region_attack,
)


def test_attacker_apis_have_no_evaluator_secrets():
    for function in (sparse_region_attack, adaptive_attack, random_budget_attack, public_objective):
        parameters = set(inspect.signature(function).parameters)
        assert "target_tokens" not in parameters
        assert "secret_x" not in parameters


def test_sparse_and_adaptive_smoke():
    torch.manual_seed(1)
    model = Memory(8, 6, 4).eval()
    sparse = sparse_region_attack(model, starts=3, steps=2, lr=0.01, seed=2)
    assert sparse["candidates"].shape == (3, 8)
    assert 1 <= sparse["distinct_patterns"] <= 3
    adaptive = adaptive_attack(model, starts=3, max_steps=2, checkpoints=[1, 2], lr=0.01, seed=3)
    random = random_budget_attack(model, starts=3, checkpoints=[1, 2], seed=4)
    assert set(adaptive["candidates"]) == {"1", "2"}
    assert set(random["candidates"]) == {"1", "2"}


def test_evaluator_runs_only_after_candidate_is_frozen():
    model = Memory(8, 6, 4).eval()
    frozen = {"candidate": torch.randn(1, 8)}
    target = torch.tensor([[1, 2, 3, 4]])
    result = evaluate_candidate_set(model, frozen, target, torch.randn(1, 8))
    assert "mae_tokens" in result["candidate"]
    assert "input_cosine_evaluator_only" in result["candidate"]


def test_known_initialization_candidate_is_public_shape():
    model = Memory(8, 6, 4)
    initial = {"net.4.bias": model.net[-1].bias.detach().clone()}
    candidate = known_initialization_candidate(model, initial)
    assert candidate.shape == (1, 4)
