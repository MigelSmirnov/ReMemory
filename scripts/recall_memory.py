#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ReMemory Recall CLI
A simple command-line interface to query and retrieve reconstructed memory from semantic signals.
"""

import sys
from pathlib import Path
import argparse

# 💡 Add project root to sys.path for module imports
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from memory.semantic_recall import semantic_recall_plain


def main():
    parser = argparse.ArgumentParser(description="ReMemory: Semantic Recall CLI")
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top memory cells to retrieve (default: 3)",
    )
    args = parser.parse_args()

    # 🧠 Ask the user for a query phrase
    query = input("🔎 Enter a phrase to search in memory: ").strip()
    if not query:
        print("❌ Empty query — exiting.")
        return

    # 🔍 Perform semantic recall
    result = semantic_recall_plain(
        query=query,
        top_k=args.top_k,
    )

    if not result or not result.get("top_cell"):
        print("❌ No memory match found.")
        return

    # 📊 Show best match
    top = result["top_cell"]
    print("\n🔎 Best match:")
    print(f"📁 ID: {top['cell_id']}")
    print(f"📈 Similarity: {top['score']:.4f}\n")
    print("🧠 Reconstructed text:\n")
    print(top["text"])

    # 📊 Show similarity distribution
    print("\n📊 Top closest memories:")
    for i, c in enumerate(result["distribution"], 1):
        print(f"{i}. {c['cell_id']} — score={c['score']:.4f}")


if __name__ == "__main__":
    main()
