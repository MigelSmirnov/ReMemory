import inspect

import numpy as np
import torch

from experiments.knc_0.phase_5a import load_public, make_model, public_bundle
from experiments.knc_0.phase_5c import (adam_fit, auc, jacobian, lm_fit, multi_input, observability,
                                        permuted_model, spectrum_stats)

FORBIDDEN = {"target", "tokens", "truth", "secret", "secret_x", "x", "key", "unknown"}


def _model(d=12, m=8, seed=0):
    torch.manual_seed(seed)
    return load_public(public_bundle(make_model(d, m).eval(), 4095.0))


def test_analyst_apis_take_no_secret():
    for fn in (adam_fit, lm_fit, observability, multi_input):
        assert not set(inspect.signature(fn).parameters) & FORBIDDEN, fn.__name__


def test_jacobian_matches_finite_difference():
    model = _model()
    z = torch.randn(12, dtype=torch.float64).float()
    J = jacobian(model, z).double()
    e = torch.zeros(12)
    e[3] = 1e-3
    with torch.no_grad():
        fd = (model(z + e) - model(z - e)).double() / 2e-3
    assert torch.allclose(J[:, 3], fd, atol=1e-3)


def test_observability_nullspace_preserves_known_outputs_linearly():
    model = _model(d=24, m=10)
    z = torch.randn(24)
    idx = np.array([0, 3, 5])
    obs, J = observability(model, 4095.0, z, idx, seed=1)
    assert obs["jk_rank"] <= 3 and obs["null_dim"] == 24 - obs["jk_rank"]
    assert obs["g_tok"] >= 0 and 0 <= obs["g_rel"] <= 1 + 1e-9
    assert obs["worst_eps0.001_known_change"] < 0.05 * max(obs["worst_eps0.001_unknown_change"], 1e-6) + 1e-3


def test_fits_depend_only_on_known_values():
    model = _model()
    idx, vals = np.array([1, 2, 6]), np.array([100.0, 2000.0, 3000.0])
    for fn in (lambda: adam_fit(model, 4095.0, idx, vals, seed=3, steps=20),
               lambda: lm_fit(model, 4095.0, idx, vals, seed=3, iters=5)):
        a, b = fn(), fn()
        assert torch.equal(a["z"], b["z"])


def test_permuted_model_permutes_outputs():
    torch.manual_seed(0)
    pub = public_bundle(make_model(12, 8).eval(), 1.0)
    perm = np.random.default_rng(0).permutation(8)
    z = torch.randn(1, 12)
    with torch.no_grad():
        assert torch.allclose(permuted_model(pub, perm)(z)[0], load_public(pub)(z)[0][perm])


def test_spectrum_stats_and_auc():
    s = spectrum_stats(np.array([1.0, 0.1, 0.0005]), 4095.0)
    assert s["rank_rel1e-2"] == 2 and s["rank_rel1e-3"] == 2 and s["rank_abs50tok"] == 2
    assert auc([1, 2], [3, 4]) == 1.0 and auc([3, 4], [1, 2]) == 0.0


def test_dead_l1_candidate_equals_output_when_first_layer_is_silent():
    from experiments.knc_0.phase_5c import dead_l1_candidate, layer_activity
    model = _model(d=12, m=8)
    with torch.no_grad():
        model[0].bias.fill_(-100.0)  # every first-layer unit inactive for small inputs
        z = torch.randn(12) * 0.01
        out = np.clip(np.round((model(z[None])[0] * 4095.0).double().numpy()), 0, 4095).astype(int)
    assert layer_activity(model, z)["l1_active"] == 0
    assert np.array_equal(dead_l1_candidate(model, 4095.0), out)


def test_zero_spectrum_has_zero_rank():
    s = spectrum_stats(np.zeros(5), 4095.0)
    assert s["rank_rel1e-3"] == 0 and s["entropy_rank"] == 0.0


def test_observability_at_locally_constant_point():
    model = _model(d=12, m=8)
    with torch.no_grad():
        model[0].bias.fill_(-100.0)
    obs, J = observability(model, 4095.0, torch.randn(12) * 0.01, np.array([0, 2]), seed=1)
    assert obs["jk_rank"] == 0 and obs["g_tok"] == 0.0 and obs["locally_constant"]


def test_lm_survives_zero_jacobian():
    model = _model(d=12, m=8)
    with torch.no_grad():
        model[0].bias.fill_(-1e6)
    res = lm_fit(model, 4095.0, np.array([0, 1]), np.array([5.0, 9.0]), seed=1, iters=3)
    assert np.isfinite(res["known_mse"])
