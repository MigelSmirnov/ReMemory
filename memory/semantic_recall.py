# -*- coding: utf-8 -*-
"""
ReMemory: Semantic Recall
Finds the most relevant memory cells based on a semantic query and reconstructs the stored text.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from memory.generate_embedding_vector import get_embedding_vector
from memory.mlp_core.mlp_decoder import reconstruct_token_ids
from memory.codec.base64_codec import decode_token_ids_to_text

# ✅ Use shared project paths (no hardcoded directory)
from memory.common_paths import CELLS_DIR


# -------------------- Utilities --------------------

def _cosine(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return -1.0
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


# -------------------- Main Recall API --------------------

def semantic_recall_plain(
    query: str,
    top_k: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the most semantically similar memory cell(s) and reconstruct the stored text.

    Args:
        query: Natural language query or semantic signal.
        top_k: Number of top matching memory cells to return (for similarity distribution).

    Returns:
        A dictionary containing similarity scores and reconstructed text from the best match.
    """
    query_vec = get_embedding_vector(query)
    if query_vec is None:
        print("❌ Failed to compute embedding for the query.")
        return None

    if not CELLS_DIR.exists():
        print(f"❌ Memory directory not found: {CELLS_DIR}")
        return None

    scored: List[Tuple[str, float]] = []

    # Compute cosine similarity between the query and each memory cell context vector
    for d in os.listdir(CELLS_DIR):
        if not d.startswith("vec_"):
            continue
        cell_dir = CELLS_DIR / d
        context_file = cell_dir / "context_vector.json"
        if not context_file.exists():
            continue
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                vec = json.load(f)
            score = _cosine(query_vec, vec)
            scored.append((d, score))
        except Exception:
            continue

    if not scored:
        print("⚠️ Memory is empty or contains no valid cells.")
        return None

    # Sort memory cells by similarity score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top_k results for similarity distribution
    top_cells = scored[:top_k]
    distribution = [{"cell_id": cid, "score": float(score)} for cid, score in top_cells]

    # ✅ Reconstruct the text from the most similar memory cell
    top_cell_id, top_score = scored[0]
    top_cell_dir = CELLS_DIR / top_cell_id
    context_path = top_cell_dir / "context_vector.json"
    model_path = top_cell_dir / "model.pt"
    config_path = top_cell_dir / "model_config.json"

    text = "[Reconstruction error]"
    try:
        with open(context_path, "r", encoding="utf-8") as f:
            stored_vector = json.load(f)
        token_ids = reconstruct_token_ids(
            context_vector=stored_vector,
            model_path=str(model_path),
            config_path=str(config_path),
            token_range=(0, 4095),
        )
        text = decode_token_ids_to_text(token_ids)
    except Exception as e:
        print(f"⚠️ Reconstruction failed: {e}")

    return {
        "distribution": distribution,
        "top_cell": {
            "cell_id": top_cell_id,
            "score": float(top_score),
            "text": text
        },
    }
