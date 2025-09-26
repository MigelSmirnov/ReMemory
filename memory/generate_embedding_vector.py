# -*- coding: utf-8 -*-
"""
memory/generate_embedding_vector.py

Lightweight semantic embedding utility with a fallback mechanism.
- Primary: SentenceTransformer (multilingual model)
- Fallback: Stable hash-based vector (works even without model)

Compatible API:
- get_embedding_vector(text_or_tokens)  -> List[float]
"""

from typing import List
import re

# --- Lazy model loader to avoid heavy init at import time ---
_model = None
_model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # default multilingual model


def _ensure_model():
    """
    Load SentenceTransformer model lazily (only on first use).
    Returns None if the model cannot be loaded (fallback will be used).
    """
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _model = SentenceTransformer(_model_name)
        return _model
    except Exception:
        _model = None
        return None


def _fallback_hash(text: str, dim: int = 256) -> List[float]:
    """
    Create a deterministic hash-based vector representation as a fallback.
    Works without external models (lower quality but deterministic).
    """
    import hashlib
    import struct
    v = [0.0] * dim
    for tok in re.findall(r"[\w\-\u0400-\u04FF]+", (text or "").lower()):
        h = hashlib.blake2s(tok.encode("utf-8"), digest_size=8).digest()
        i = struct.unpack("<Q", h)[0] % dim
        v[i] += 1.0
    return v


def get_embedding_vector(text_or_tokens) -> List[float]:
    """
    Compute a semantic embedding vector for a string or list of tokens.

    Args:
        text_or_tokens: A string or list of keywords.

    Returns:
        A semantic embedding vector as a list of floats.
    """
    if isinstance(text_or_tokens, (list, tuple)):
        text = " ".join(str(x) for x in text_or_tokens if x)
    else:
        text = str(text_or_tokens or "")

    model = _ensure_model()
    if model is not None:
        try:
            arr = model.encode([text], show_progress_bar=False, normalize_embeddings=True)
            return arr[0].tolist()
        except Exception:
            return _fallback_hash(text)
    else:
        return _fallback_hash(text)
