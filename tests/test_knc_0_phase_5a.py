import inspect

import numpy as np
import torch

from experiments.knc_0.phase_5a import (SCALES, attack, evaluate, make_model, public_bundle,
                                        to_tokens, wilson)


def test_attacker_api_has_no_truth():
    params = set(inspect.signature(attack).parameters)
    assert not params & {"target", "tokens", "secret", "x", "key"}


def test_public_bundle_holds_weights_and_scale_only():
    pub = public_bundle(make_model(16, 8), 4095.0)
    assert set(pub) == {"state", "arch", "output_scale"}


def test_scales_and_token_rounding():
    assert SCALES == {"A": 1.0, "B": 4095.0, "C": 10000.0}
    t = torch.tensor([[0, 1, 4095]])
    for s in SCALES.values():
        assert torch.equal(to_tokens(t.float() / s, s), t)


def test_attack_and_evaluate_smoke():
    torch.manual_seed(0)
    model = make_model(16, 8).eval()
    cand = attack(public_bundle(model, 1.0), n_probes=32, probe_seed=1)
    assert cand["tokens"].shape == (8,) and isinstance(cand["accept"], bool)
    ev = evaluate(model, 1.0, torch.zeros(1, 16), torch.zeros(1, 8, dtype=torch.long), cand)
    assert {"memorized", "full_recovery", "false_accept", "missed_recovery"} <= set(ev)


def test_attack_is_deterministic():
    torch.manual_seed(0)
    pub = public_bundle(make_model(16, 8), 1.0)
    a, b = attack(pub, 32, 1), attack(pub, 32, 1)
    assert np.array_equal(a["tokens"], b["tokens"])


def test_wilson_bounds():
    lo, hi = wilson(30, 30)
    assert 0.88 < lo < 0.9 and hi == 1.0
