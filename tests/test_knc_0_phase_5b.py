import inspect

import numpy as np
import torch

from experiments.knc_0.phase_5a import make_model, public_bundle
from experiments.knc_0.phase_5b import (PublicView, conditions, input_fit, known_positions, output_mean,
                                        plane_fit, score)

FORBIDDEN = {"target", "tokens", "truth", "secret", "secret_x", "x", "key", "unknown"}


def _view(scale=4095.0):
    torch.manual_seed(0)
    return PublicView(public_bundle(make_model(16, 8).eval(), scale), n_probes=32, probe_seed=1)


def test_analyst_apis_take_no_secret():
    for fn in (input_fit, plane_fit, output_mean, PublicView):
        assert not set(inspect.signature(fn).parameters) & FORBIDDEN, fn.__name__


def test_public_bundle_contents():
    assert set(public_bundle(make_model(16, 8), 1.0)) == {"state", "arch", "output_scale"}


def test_positions_public_and_without_replacement():
    a, b = known_positions(256, 16, 5), known_positions(256, 16, 5)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 16 and a.max() < 256


def test_candidate_depends_only_on_known_tokens():
    view = _view()
    idx = known_positions(8, 4, 3)
    known = np.array([10.0, 20.0, 30.0, 40.0])
    t1, t2 = np.zeros(8, int), np.full(8, 4000)
    t1[idx] = t2[idx] = known.astype(int)  # two targets that agree on the known positions
    for fn in (lambda: plane_fit(view, idx, known),
               lambda: input_fit(view, idx, known, 0.05, "free", 20, 2, seed=9)):
        a, b = fn()["tokens"], fn()["tokens"]
        assert np.array_equal(a, b)
        assert score(a, t1, idx)["n_unknown"] == 4 and score(b, t2, idx)["n_unknown"] == 4


def test_score_counts_unknown_only_and_k_all_has_none():
    truth = np.arange(8)
    idx = np.array([0, 1])
    cand = truth.copy()
    cand[0] = 99  # wrong known token must not affect unknown metrics
    s = score(cand, truth, idx)
    assert s["exact_fraction"] == 1.0 and s["whole_unknown"] and not s["known_consistent"]
    assert "exact_fraction" not in score(truth, truth, np.arange(8))


def test_conditions_layout():
    c = conditions((0, 4), draws=3, include_diag=True)
    assert c == [(0, 0), (4, 0), (4, 1), (4, 2), (256, 0)]
