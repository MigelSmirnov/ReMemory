import torch

from experiments.knc_0.phase_2_5 import (
    Memory, attacker_observe, correct_activation_diagnostics,
    evaluate_observation, parameter_only_candidate,
)


def test_parameter_candidate_has_expected_shape_and_range():
    model = Memory(8, 6, 4).eval()
    candidate = parameter_only_candidate(model)
    assert candidate.shape == (1, 4)
    assert bool(((candidate >= 0) & (candidate <= 4095)).all())


def test_attacker_boundary_and_streamed_counts():
    torch.manual_seed(1)
    model = Memory(8, 6, 4).eval()
    public = {"input_dim": 8, "hidden_dim": 6, "output_dim": 4}
    observation = attacker_observe(model, public, [10, 20], 10, 2)
    assert set(observation["mean_candidates"]) == {"10", "20"}
    assert observation["coordinate_stats"]["raw_mean"].shape == (4,)
    target = torch.tensor([[1, 2, 3, 4]])
    convergence, baselines, summary = evaluate_observation(observation, target)
    assert set(convergence) == {"10", "20"}
    assert set(baselines) == {"parameter_only", "bias_only", "midpoint"}
    assert "mean_90pct_range" in summary


def test_correct_activation_is_evaluator_only_postprocessing():
    model = Memory(8, 6, 4).eval()
    public = {"input_dim": 8, "hidden_dim": 6, "output_dim": 4}
    observation = attacker_observe(model, public, [10], 10, 3)
    diagnostics = correct_activation_diagnostics(
        model, torch.randn(1, 8), observation["random_active_probability"]
    )
    assert set(diagnostics) == {"layer1", "layer2"}
    assert all(0 <= diagnostics[layer]["mean_pattern_agreement"] <= 1 for layer in diagnostics)
