from __future__ import annotations

# PURPOSE: Central path definitions for the project.
# Resolves the project root relative to this file's location so imports work
# regardless of where the user runs commands from (the root or a subdirectory).

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
