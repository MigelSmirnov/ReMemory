"""
Common project paths used across modules.

This file defines base directories such as the location for memory cells.
All other modules should import paths from here instead of hardcoding them.
"""

from pathlib import Path
import os

# 🚀 Project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 📦 Directory where memory cells are stored.
# Can be overridden by the REM_CELLS_DIR environment variable.
CELLS_DIR = Path(os.getenv("REM_CELLS_DIR", PROJECT_ROOT / "memory_cells"))

# ✅ Ensure the directory exists (optional but handy)
CELLS_DIR.mkdir(parents=True, exist_ok=True)
