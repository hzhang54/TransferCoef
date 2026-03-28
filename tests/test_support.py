from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

def add_src_to_path() -> Path:
    """Ensure the local src directory is importable for tests."""

    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    src_path_str = str(src_path)

    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

    return project_root

def has_packages(*package_names: str) -> bool:
    """Return True when all requested packages are importable."""
    return all(importlib.util.find_spec(package_name) is not None for package_name in package_names)
    