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


def open_path(path: Path) -> bool:
    """Open a file or folder in the system's file browser; False if it is missing."""
    import os
    import subprocess
    import sys

    path = Path(path)
    if not path.exists():
        return False
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    return True


class Tooltip:
    """Show a short help note when the mouse rests on a widget."""

    def __init__(self, widget, text: str, delay_ms: int = 400, wrap_px: int = 420) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wrap_px = wrap_px
        self._job = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._job = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None

    def _show(self) -> None:
        import tkinter as tk

        self._job = None
        if self._window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        self._window.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._window, text=self.text, justify="left", wraplength=self.wrap_px,
            background="#ffffe0", relief="solid", borderwidth=1, padx=6, pady=4,
        ).pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None
