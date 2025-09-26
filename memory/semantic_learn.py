# -*- coding: utf-8 -*-
"""
ReMemory: Training memory cells from a semantic signal.
Creates a "memory" that is not stored explicitly but reconstructed based on meaning.

Usage:
    from memory.semantic_learn import semantic_learn

    result = semantic_learn("summer 2024 in Paris", "I went to Paris with a friend and we...")
"""

import os
import json
from pathlib import Path
from typing import List, Any, Union

# ✅ Import global paths (no hardcoded directories)
from memory.common_paths import CELLS_DIR

# === Core imports ===
from memory.generate_embedding_vector import get_embedding_vector
from memory.codec.base64_codec import encode_text_to_token_ids
from memory.mlp_core.mlp_trainer import train_cell


# ===== Utility functions =====
def _ensure_dir(p: Path) -> None:
    """Create a directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def _to_list(x: Any) -> List[float]:
    """Convert a tensor/array to a regular Python list."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, list):
        return x
    return list(x)


def _save_json(path: Path, data: Any) -> None:
    """Save data as a nicely formatted UTF-8 JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _next_cell_id() -> str:
    """Generate the next cell ID like vec_0001, vec_0002, etc."""
    _ensure_dir(CELLS_DIR)
    existing = [d for d in os.listdir(CELLS_DIR) if d.startswith("vec_")]
    try:
        num = 1 + max([int(d.split("_")[1]) for d in existing], default=0)
    except Exception:
        num = 1
    return f"vec_{num:04d}"


# ===== Main API =====
def semantic_learn(keywords: Union[str, List[str]], text: str):
    """
    Train a new memory cell.

    Args:
        keywords: A semantic signal (string or list of keywords).
        text: The full memory text to encode.
    """

    if isinstance(keywords, str):
        keywords = [keywords]

    # 1. Compute semantic embedding
    context_vector = get_embedding_vector(keywords)
    if context_vector is None:
        raise ValueError("❌ Failed to obtain semantic embedding.")
    context_vector = _to_list(context_vector)

    # 2. Encode text into token IDs
    token_ids = encode_text_to_token_ids(text)

    # 3. Create a new memory cell directory
    cell_id = _next_cell_id()
    cell_dir = CELLS_DIR / cell_id
    _ensure_dir(cell_dir)

    # 4. Save context vector
    _save_json(cell_dir / "context_vector.json", context_vector)

    # 5. Train MLP to reconstruct text
    train_result = train_cell(context_vector, token_ids, cell_id)

    return {
        "cell_id": cell_id,
        "tokens_len": len(token_ids),
        "epochs": train_result["actual_epochs"],
        "final_loss": train_result["final_loss"],
        "saved": {
            "context_vector.json": True,
            "model.pt": True
        }
    }


__all__ = ["semantic_learn", "CELLS_DIR"]
