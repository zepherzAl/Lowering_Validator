"""Small shared helpers for the EMS desktop tools.

The module deliberately has no GIS dependency, so it can also be used by a
future command-line runner and unit tests.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def run_folder(parent: Path, tool_name: str) -> Path:
    """Create a unique, timestamped output folder without replacing prior runs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = parent / f"{tool_name}_{stamp}"
    suffix = 2
    while target.exists():
        target = parent / f"{tool_name}_{stamp}_{suffix}"
        suffix += 1
    target.mkdir(parents=True, exist_ok=False)
    return target


def safe_output_path(path: Path) -> Path:
    """Return a non-existing sibling path; never silently replace a file."""
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{suffix}{path.suffix}")
        suffix += 1
    return candidate


class SettingsStore:
    """Best-effort local settings persistence, stored beside the application."""
    def __init__(self, filename: str) -> None:
        self.path = Path(__file__).with_name(filename)

    def load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, values: dict[str, Any]) -> None:
        try:
            self.path.write_text(json.dumps(values, indent=2), encoding="utf-8")
        except OSError:
            pass
