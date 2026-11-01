import numpy as np

from experiments.knc_0.phase_5d import (POOL, known_only_baseline, secret_text, text_scores, tokens_to_bytes,
                                        word_spans)
from memory.codec.base64_codec import decode_token_ids_to_text, encode_text_to_token_ids


def test_tokens_to_bytes_matches_rememory_codec():
    text = "Утро над рекой. Fog over the river."
    ids = encode_text_to_token_ids(text)
    assert tokens_to_bytes(ids).decode("utf-8", "replace").startswith(decode_token_ids_to_text(ids)[:20])


def test_secret_text_is_256_tokens_384_bytes():
    for lang in POOL:
        t = secret_text(lang, np.random.default_rng(0))
        assert len(t) == 256 and len(tokens_to_bytes(t)) == 384 and t.max() < 4096


def test_perfect_candidate_scores_one_and_wrong_word_counts():
    t = secret_text("en", np.random.default_rng(1))
    s = text_scores(t.copy(), t, np.array([], dtype=int))
    assert s["byte_accuracy"] == 1.0 and s["word_recovery"] == 1.0
    bad = t.copy()
    bad[0] ^= 1
    assert text_scores(bad, t, np.array([], dtype=int))["word_recovery"] < 1.0


def test_word_spans():
    assert word_spans(b"ab  cd e") == [(0, 2), (4, 6), (7, 8)]


def test_known_only_baseline_bounds():
    t = secret_text("ru", np.random.default_rng(2))
    lo = known_only_baseline(t, np.array([], dtype=int))
    hi = known_only_baseline(t, np.arange(256))
    assert lo < 0.05 and hi == 1.0
