#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ReMemory – Demo training script with detailed logs.
Automatically finds a JSON dataset in ./data/ and trains memory cells from it.
For each cell, shows loss and epochs info.
"""

import json
from pathlib import Path
import sys

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.semantic_learn import semantic_learn

DATA_DIR = PROJECT_ROOT / "data"


def find_dataset() -> Path:
    """Return the first JSON file found in ./data/."""
    if not DATA_DIR.exists():
        raise FileNotFoundError("❌ Folder './data' not found. Create it and add a JSON dataset.")

    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("❌ No JSON dataset found in './data/'. Please add one.")

    dataset = json_files[0]
    print(f"📁 Using dataset: {dataset}")
    return dataset


def train_from_dataset(dataset_path: Path) -> None:
    """Train memory cells from the dataset with detailed logging."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("❌ Dataset must be a list of objects with 'keywords' and 'text' fields.")

    total = len(data)
    print(f"🔍 Found {total} stories in dataset.")
    trained = 0

    for i, item in enumerate(data, start=1):
        keywords = item.get("keywords")
        text = item.get("text")

        if not keywords or not text:
            print(f"⚠️ Skipping #{i}: missing 'keywords' or 'text'")
            continue

        print(f"\n🧠 Training memory cell {i}/{total}...")
        try:
            result = semantic_learn(keywords=keywords, text=text)

            print("📊 Training result:")
            print(f"   - Cell ID:        {result['cell_id']}")
            print(f"   - Tokens length:  {result['tokens_len']}")
            print(f"   - Epochs used:    {result['epochs']}")
            print(f"   - Final loss:     {result['final_loss']:.8f}")

            # ✅ Правильная проверка
            if float(result['final_loss']) <= 1e-5:
                print("   - ✅ Target reached (≤ 1e-5)")
                trained += 1
            else:
                print("   - ⚠️ Target not reached")

        except Exception as e:
            print(f"❌ Error training cell #{i}: {e}")    

def main():
    dataset = find_dataset()
    train_from_dataset(dataset)


if __name__ == "__main__":
    main()
