#!/usr/bin/env python3
"""Select, verify, create, or safely copy the shared EMS SQLite database."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ems_database import (
    EMSDatabase,
    configured_database_path,
    copy_database_verified,
    set_configured_database_path,
    validate_database,
)


class EMSDatabaseManager(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EMS Database Manager")
        self.geometry("780x560")
        self.minsize(680, 480)
        self.path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select or verify an EMS database.")
        self._build()
        try:
            configured = configured_database_path()
        except ValueError as exc:
            self.status_var.set(str(exc))
        else:
            if configured is not None:
                self.path_var.set(str(configured))
                if not configured.is_file():
                    self.status_var.set(
                        "Configured database is unavailable. No replacement was created."
                    )
                else:
                    self.verify_current(show_errors=False)

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(
            frame,
            text=(
                "The selected SQLite file is shared by the EMS applications. "
                "Changing its location does not change the EMS source root."
            ),
            wraplength=720,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        ttk.Label(frame, text="Active EMS database").grid(
            row=1, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Entry(frame, textvariable=self.path_var, state="readonly").grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Button(frame, text="Open database folder", command=self.open_folder).grid(
            row=1, column=2, padx=(8, 0)
        )

        actions = ttk.LabelFrame(frame, text="Database location")
        actions.grid(row=2, column=0, columnspan=3, sticky="ew", pady=12)
        for column in range(3):
            actions.columnconfigure(column, weight=1)
        ttk.Button(
            actions, text="Locate existing database...", command=self.locate_database
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=8)
        ttk.Button(
            actions, text="Create new database...", command=self.create_database
        ).grid(row=0, column=1, sticky="ew", padx=6, pady=8)
        ttk.Button(
            actions, text="Copy active database...", command=self.copy_database
        ).grid(row=0, column=2, sticky="ew", padx=6, pady=8)
        ttk.Button(
            actions, text="Verify active database", command=self.verify_current
        ).grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 8))

        ttk.Label(frame, text="Status").grid(row=3, column=0, sticky="nw")
        ttk.Label(
            frame,
            textvariable=self.status_var,
            wraplength=720,
            justify="left",
        ).grid(row=3, column=1, columnspan=2, sticky="w")

        self.details = tk.Text(frame, height=16, wrap="word", state="disabled")
        self.details.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        frame.rowconfigure(4, weight=1)

    def _show_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def _display_info(self, info) -> None:
        self.path_var.set(str(info.path))
        roots = "\n".join(f"  {root}" for root in info.source_roots) or "  (none yet)"
        profiles = "\n".join(f"  {profile}" for profile in info.profiles) or "  (none yet)"
        self._show_details(
            f"Database: {info.path}\n"
            f"Size: {info.size_bytes / 1_000_000_000:.2f} GB\n"
            f"Indexed source files: {info.source_file_count:,}\n"
            f"SQLite user_version: {info.user_version}\n"
            f"SQLite schema_version: {info.schema_version}\n\n"
            f"Indexed source roots:\n{roots}\n\n"
            f"Indexed profiles:\n{profiles}"
        )

    def verify_current(self, *, show_errors: bool = True) -> bool:
        raw = self.path_var.get().strip()
        if not raw:
            self.status_var.set("No database path is selected.")
            return False
        try:
            info = validate_database(Path(raw))
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            self.status_var.set(f"Database verification failed: {exc}")
            if show_errors:
                messagebox.showerror("EMS Database Manager", str(exc), parent=self)
            return False
        self._display_info(info)
        self.status_var.set("EMS database schema and metadata verified.")
        return True

    def locate_database(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title="Locate existing EMS database",
            filetypes=[("SQLite databases", "*.sqlite3 *.sqlite *.db"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            info = validate_database(Path(selected))
            set_configured_database_path(info.path)
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            messagebox.showerror(
                "EMS Database Manager",
                f"The selected file was not activated.\n\n{exc}",
                parent=self,
            )
            return
        self._display_info(info)
        self.status_var.set("Verified database selected for both EMS applications.")

    def create_database(self) -> None:
        source_root = filedialog.askdirectory(
            parent=self,
            title="Select the EMS source root this database will index",
        )
        if not source_root:
            return
        selected = filedialog.asksaveasfilename(
            parent=self,
            title="Create EMS database",
            defaultextension=".sqlite3",
            initialfile="EMS_Master.sqlite3",
            filetypes=[("SQLite databases", "*.sqlite3")],
        )
        if not selected:
            return
        path = Path(selected)
        if path.exists():
            messagebox.showerror(
                "EMS Database Manager",
                "The selected file already exists. Choose a new filename; "
                "existing databases are never overwritten.",
                parent=self,
            )
            return
        try:
            database = EMSDatabase(Path(source_root), path)
            del database
            info = validate_database(path)
            set_configured_database_path(info.path)
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            self.status_var.set("Database was not activated.")
            messagebox.showerror("EMS Database Manager", str(exc), parent=self)
            return
        self._display_info(info)
        self.status_var.set(
            "New empty EMS database created and selected. "
            "The source files will be indexed when the Validator next runs."
        )

    def copy_database(self) -> None:
        source = configured_database_path()
        if source is None:
            messagebox.showerror(
                "EMS Database Manager",
                "Select or configure the active EMS database before copying it.",
                parent=self,
            )
            return
        if not source.is_file():
            messagebox.showerror(
                "EMS Database Manager",
                f"The configured EMS database is unavailable:\n{source}",
                parent=self,
            )
            return
        selected = filedialog.asksaveasfilename(
            parent=self,
            title="Copy EMS database to new location",
            defaultextension=".sqlite3",
            initialfile=source.name,
            filetypes=[("SQLite databases", "*.sqlite3")],
        )
        if not selected:
            return
        destination = Path(selected)
        if destination.resolve() == source.resolve():
            messagebox.showerror(
                "EMS Database Manager",
                "Choose a different destination from the current database.",
                parent=self,
            )
            return
        if not messagebox.askokcancel(
            "Close EMS applications before copying",
            "For the safest move, close the Validator and any other tool using "
            "this database before continuing. The original will be retained. "
            "A SQLite backup is used so WAL data is handled by SQLite rather "
            "than copied as sidecar files.\n\nContinue?",
            parent=self,
        ):
            return

        def report_progress(done: int, total: int) -> None:
            percent = 100.0 * done / total if total else 100.0
            self.status_var.set(
                f"Copying database with SQLite backup: {percent:.1f}% "
                f"({done:,} of {total:,} pages)"
            )
            self.update_idletasks()

        try:
            info = copy_database_verified(
                source, destination, progress=report_progress
            )
            set_configured_database_path(info.path)
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            self.status_var.set("Copy failed; the original remains configured.")
            messagebox.showerror("EMS Database Manager", str(exc), parent=self)
            return
        self._display_info(info)
        self.status_var.set(
            "Copy verified and activated. The original database was retained."
        )

    def open_folder(self) -> None:
        raw = self.path_var.get().strip()
        if not raw:
            messagebox.showinfo(
                "EMS Database Manager", "No database path is selected.", parent=self
            )
            return
        folder = Path(raw).expanduser().parent
        if not folder.is_dir():
            messagebox.showerror(
                "EMS Database Manager",
                f"Database folder is unavailable:\n{folder}",
                parent=self,
            )
            return
        if os.name == "nt":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])


def main() -> None:
    EMSDatabaseManager().mainloop()


if __name__ == "__main__":
    main()
