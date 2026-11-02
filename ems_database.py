"""Shared, rebuildable SQLite index for EMS point records."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event
from time import perf_counter
from typing import Callable, Iterable, Iterator, Mapping

INSERT_BATCH_SIZE = 50_000
SYNC_PROGRESS_INTERVAL = 250_000
DATABASE_CONFIG_FILENAME = "ems_toolkit_config.json"
EXPECTED_INDEXES = {
    "idx_ems_datetime": ("source_root", "profile", "timestamp"),
    "idx_ems_source_datetime": (
        "source_root", "profile", "source_path", "timestamp"
    ),
    "idx_ems_source_vessel": ("source_root", "profile", "vessel"),
    "idx_ems_dredge_datetime": (
        "source_root", "profile", "dredge_numeric", "timestamp"
    ),
}


class EMSDatabaseUnavailableError(FileNotFoundError):
    """Configured EMS database cannot be reached without user intervention."""


@dataclass(frozen=True)
class EMSRecord:
    """One normalized EMS row in the source file's native coordinate system."""

    timestamp: str | None
    x: float | None
    y: float | None
    dredge: str | None
    vessel: str | None
    crs: str
    dredge_numeric: float | None = None
    dredge_numeric_parsed: bool = False


@dataclass(frozen=True)
class SyncResult:
    discovered: int
    imported: int
    reused: int
    failed: int
    removed: int


@dataclass(frozen=True)
class EMSImportBatch:
    """A database-ready batch and timings measured while preparing it."""

    rows: list[tuple[object, ...]]
    file_read_seconds: float = 0.0
    vector_conversion_seconds: float = 0.0
    parameter_build_seconds: float = 0.0
    vector_timings: Mapping[str, float] | None = None


def fallback_vessel_name(source_path: str | os.PathLike[str]) -> str:
    """Vessel name for EMS files without a vessel column.

    Uses the file name, unless it has no letters (for example
    "19-01-2022.txt" inside a "232003506 - (Welsh Piper)" folder), in which
    case the parent folder name identifies the vessel.
    """
    text = os.fspath(source_path)
    path = PureWindowsPath(text) if "\\" in text else PurePosixPath(text)
    if any(character.isalpha() for character in path.stem):
        return path.stem
    return path.parent.name or path.stem


def discover_files(
    source: Path,
    extensions: Iterable[str],
    *,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    """Return supported source files without opening or parsing their contents."""
    source = Path(source)
    allowed = {extension.lower() if extension.startswith(".") else f".{extension.lower()}" for extension in extensions}
    if source.is_file():
        return [source] if source.suffix.lower() in allowed else []
    if not source.is_dir():
        raise FileNotFoundError(f"EMS source does not exist: {source}")

    discovered = []
    visited_directories = 0

    def report_walk_error(error: OSError) -> None:
        raise error

    for directory, subdirectories, filenames in os.walk(
        source, onerror=report_walk_error
    ):
        subdirectories.sort()
        visited_directories += 1
        if log is not None and visited_directories % 100 == 0:
            log(
                f"EMS discovery scanned {visited_directories:,} folders; "
                f"found {len(discovered):,} supported files so far..."
            )
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix.lower() in allowed:
                discovered.append(path)
    return sorted(discovered)


def _root_path(source: Path) -> Path:
    source = Path(source).expanduser().resolve()
    return source if source.is_dir() else source.parent


def database_path_for_source(source: Path) -> Path:
    """Choose one per-source cache path shared by both applications."""
    root = str(_root_path(source))
    return _database_path_for_root(root)


def _database_path_for_root(root: str) -> Path:
    if os.name == "nt":
        root = root.casefold()
    digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:24]
    if os.name == "nt":
        cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        cache_root = cache_root / "EMS_Toolkit" / "indexes"
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_root = cache_root / "ems_toolkit" / "indexes"
    return cache_root / f"ems_{digest}.sqlite3"


def shared_config_path() -> Path:
    roaming = Path(
        os.environ.get(
            "APPDATA", Path.home() / "AppData" / "Roaming"
        )
    )
    return roaming / "EMS_Toolkit" / DATABASE_CONFIG_FILENAME


def configured_database_path() -> Path | None:
    config_path = shared_config_path()
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        raw_path = config["ems_database_path"]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("ems_database_path must be a non-empty string.")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not read the shared EMS database configuration "
            f"'{config_path}': {exc}"
        ) from exc
    return Path(raw_path).expanduser()


def set_configured_database_path(path: Path) -> None:
    """Persist an already verified EMS database path for both applications."""
    config_path = shared_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"ems_database_path": str(Path(path).expanduser().resolve())}
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{config_path.name}.", suffix=".tmp", dir=config_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, config_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def resolve_database_path(source: Path, path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()

    configured = configured_database_path()
    if configured is not None:
        if not configured.is_file():
            raise EMSDatabaseUnavailableError(
                "EMS database not found.\n\n"
                f"Previously configured:\n{configured}\n\n"
                "Locate the existing database or open the EMS Database Manager. "
                "No replacement database was created."
            )
        return configured

    legacy_path = database_path_for_source(source)
    if legacy_path.is_file():
        return legacy_path
    raise EMSDatabaseUnavailableError(
        "No existing EMS database is configured or available for this source.\n\n"
        f"Legacy location checked:\n{legacy_path}\n\n"
        "Use the EMS Database Manager to locate an existing database or "
        "explicitly create a new one."
    )


@dataclass(frozen=True)
class EMSDatabaseInfo:
    path: Path
    size_bytes: int
    source_roots: tuple[str, ...]
    profiles: tuple[str, ...]
    source_file_count: int
    user_version: int
    schema_version: int


def validate_database(path: Path) -> EMSDatabaseInfo:
    """Validate EMS schema and metadata without scanning record contents."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"EMS database does not exist: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        try:
            objects = {
                (row[0], row[1]): row[2]
                for row in connection.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE type IN ('table', 'index')"
                )
            }
            required_tables = {"source_files", "ems_records", "ems_record_bounds"}
            missing_tables = required_tables - {
                name for kind, name in objects if kind == "table"
            }
            if missing_tables:
                raise ValueError(
                    "Not an EMS database; missing tables: "
                    + ", ".join(sorted(missing_tables))
                )
            actual_indexes = {
                name for kind, name in objects if kind == "index"
            }
            missing_indexes = set(EXPECTED_INDEXES) - actual_indexes
            if missing_indexes:
                raise ValueError(
                    "Not a complete EMS database; missing indexes: "
                    + ", ".join(sorted(missing_indexes))
                )
            for index_name, expected_columns in EXPECTED_INDEXES.items():
                actual_columns = tuple(
                    row[2]
                    for row in connection.execute(
                        f'PRAGMA index_info("{index_name}")'
                    )
                )
                if actual_columns != expected_columns:
                    raise ValueError(
                        f"Not a compatible EMS database; index {index_name} "
                        f"has columns {actual_columns}, expected {expected_columns}."
                    )
            rtree_sql = objects[("table", "ems_record_bounds")] or ""
            if "using rtree" not in rtree_sql.casefold():
                raise ValueError(
                    "Not a compatible EMS database; ems_record_bounds is not an R-tree."
                )

            expected_columns = {
                "source_files": {
                    "source_root", "source_path", "profile", "size_bytes",
                    "modified_ns", "record_count", "status", "imported_at",
                },
                "ems_records": {
                    "id", "source_root", "source_path", "profile", "row_number",
                    "timestamp", "x", "y", "dredge", "dredge_numeric",
                    "vessel", "crs",
                },
            }
            for table, expected in expected_columns.items():
                columns = {
                    row[1]
                    for row in connection.execute(
                        f'PRAGMA table_info("{table}")'
                    )
                }
                missing = expected - columns
                if missing:
                    raise ValueError(
                        f"Not a compatible EMS database; {table} is missing "
                        f"columns: {', '.join(sorted(missing))}"
                    )
            rtree_columns = {
                row[1]
                for row in connection.execute(
                    'PRAGMA table_info("ems_record_bounds")'
                )
            }
            if rtree_columns != {"id", "min_x", "max_x", "min_y", "max_y"}:
                raise ValueError(
                    "Not a compatible EMS database; ems_record_bounds has "
                    "unexpected columns."
                )

            source_roots = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT source_root FROM source_files "
                    "ORDER BY source_root LIMIT 100"
                )
            )
            profiles = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT profile FROM source_files "
                    "ORDER BY profile LIMIT 100"
                )
            )
            source_file_count = connection.execute(
                "SELECT COUNT(*) FROM source_files"
            ).fetchone()[0]
            user_version = connection.execute("PRAGMA user_version").fetchone()[0]
            schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
            if user_version != 0:
                raise ValueError(
                    f"Unsupported EMS database user_version {user_version}; "
                    "this application expects version 0."
                )
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Could not open a valid EMS SQLite database '{path}': {exc}") from exc

    return EMSDatabaseInfo(
        path=path.resolve(),
        size_bytes=path.stat().st_size,
        source_roots=source_roots,
        profiles=profiles,
        source_file_count=int(source_file_count),
        user_version=int(user_version),
        schema_version=int(schema_version),
    )


def copy_database_verified(
    source: Path,
    destination: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> EMSDatabaseInfo:
    """Make a consistent SQLite backup, verify it, and retain the original."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    validate_database(source)
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; choose a new location: {destination}"
        )
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Destination folder does not exist: {destination.parent}"
        )

    temporary = destination.with_name(destination.name + ".copying")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary copy path already exists; inspect it before retrying: {temporary}"
        )
    try:
        source_uri = source.as_uri() + "?mode=ro"
        source_connection = sqlite3.connect(source_uri, uri=True, timeout=30)
        destination_connection = sqlite3.connect(temporary, timeout=30)
        try:
            def on_backup(_status, remaining, total):
                if progress is not None:
                    progress(total - remaining, total)

            source_connection.backup(
                destination_connection,
                pages=16_384,
                progress=on_backup,
                sleep=0.25,
            )
        finally:
            destination_connection.close()
            source_connection.close()

        validate_database(temporary)
        os.replace(temporary, destination)
        return validate_database(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(temporary) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        raise


def _numeric(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value.strip())
    except (AttributeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class EMSDatabase:
    """Persistent EMS cache whose records are keyed by source root and profile."""

    @staticmethod
    def exists(source: Path, path: Path | None = None) -> bool:
        """Check for a cache without creating it."""
        if path is not None:
            cache_path = Path(path)
        else:
            configured = configured_database_path()
            cache_path = (
                configured
                if configured is not None
                else database_path_for_source(source)
            )
        return cache_path.is_file()

    def __init__(
        self,
        source: Path,
        path: Path | None = None,
        *,
        log: Callable[[str], None] | None = None,
    ) -> None:
        startup_started = perf_counter()
        timings: dict[str, float] = {}
        index_existence: dict[str, bool] = {}

        started = perf_counter()
        self.source_root = _root_path(source)
        timings["Source-root normalisation"] = perf_counter() - started

        started = perf_counter()
        if path is None:
            self.path = resolve_database_path(source)
        else:
            self.path = Path(path).expanduser()
        timings["Database path/hash calculation"] = perf_counter() - started

        if self.path.exists():
            started = perf_counter()
            validate_database(self.path)
            timings["Database identity/schema validation"] = (
                perf_counter() - started
            )

        started = perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        timings["Directory creation/check"] = perf_counter() - started

        self._enable_wal(timings)
        pragmas = self._initialise(timings, index_existence)
        started = perf_counter()
        size_bytes = self.path.stat().st_size if self.path.exists() else 0
        timings["Database size query"] = perf_counter() - started
        timings["TOTAL EMSDatabase startup"] = perf_counter() - startup_started
        if log is not None:
            log(
                self._format_startup_profile(
                    timings, index_existence, pragmas, size_bytes
                )
            )

    def _enable_wal(self, timings: dict[str, float] | None = None) -> None:
        opening_started = perf_counter()
        connection = sqlite3.connect(self.path, timeout=30)
        if timings is not None:
            timings["WAL SQLite connect"] = perf_counter() - opening_started
        try:
            started = perf_counter()
            connection.execute("PRAGMA busy_timeout = 30000")
            if timings is not None:
                timings["WAL busy_timeout PRAGMA"] = perf_counter() - started
            started = perf_counter()
            connection.execute("PRAGMA journal_mode = WAL")
            if timings is not None:
                timings["WAL journal_mode PRAGMA"] = perf_counter() - started
        finally:
            started = perf_counter()
            connection.close()
            if timings is not None:
                timings["WAL connection close"] = perf_counter() - started

    def _connect(
        self,
        *,
        cache_size_kib: int | None = None,
        timing: dict[str, float] | None = None,
    ) -> sqlite3.Connection:
        started = perf_counter()
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        if cache_size_kib is not None:
            connection.execute(f"PRAGMA cache_size = -{int(cache_size_kib)}")
        if timing is not None:
            timing["sqlite_connection_setup"] = (
                timing.get("sqlite_connection_setup", 0.0)
                + perf_counter()
                - started
            )
        return connection

    @contextmanager
    def _connection(
        self,
        *,
        cache_size_kib: int | None = None,
        timing: dict[str, float] | None = None,
    ):
        connection = self._connect(
            cache_size_kib=cache_size_kib,
            timing=timing,
        )
        try:
            yield connection
            commit_started = perf_counter()
            connection.commit()
            if timing is not None:
                timing["sqlite_commit"] = (
                    timing.get("sqlite_commit", 0.0)
                    + perf_counter()
                    - commit_started
                )
        except Exception:
            rollback_started = perf_counter()
            connection.rollback()
            if timing is not None:
                timing["sqlite_rollback"] = (
                    timing.get("sqlite_rollback", 0.0)
                    + perf_counter()
                    - rollback_started
                )
            raise
        finally:
            close_started = perf_counter()
            connection.close()
            if timing is not None:
                timing["sqlite_connection_close"] = (
                    timing.get("sqlite_connection_close", 0.0)
                    + perf_counter()
                    - close_started
                )

    def _initialise(
        self,
        timings: dict[str, float] | None = None,
        index_existence: dict[str, bool] | None = None,
    ) -> dict[str, object]:
        opening_started = perf_counter()
        connection = sqlite3.connect(self.path, timeout=30)
        if timings is not None:
            timings["Schema SQLite connect"] = perf_counter() - opening_started
        connection.row_factory = sqlite3.Row

        pragma_values: dict[str, object] = {}
        try:
            for name, sql in (
                ("foreign_keys", "PRAGMA foreign_keys = ON"),
                ("busy_timeout", "PRAGMA busy_timeout = 30000"),
                ("synchronous", "PRAGMA synchronous = NORMAL"),
                ("temp_store", "PRAGMA temp_store = MEMORY"),
            ):
                started = perf_counter()
                connection.execute(sql)
                if timings is not None:
                    timings[f"Initialization {name} PRAGMA"] = (
                        perf_counter() - started
                    )

            objects = (
                (
                    "source_files",
                    "table",
                    """CREATE TABLE IF NOT EXISTS source_files (
                    source_root TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    record_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    crs TEXT,
                    PRIMARY KEY (source_root, source_path, profile)
                )""",
                ),
                (
                    "ems_records",
                    "table",
                    """CREATE TABLE IF NOT EXISTS ems_records (
                    id INTEGER PRIMARY KEY,
                    source_root TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    row_number INTEGER NOT NULL,
                    timestamp TEXT,
                    x REAL,
                    y REAL,
                    dredge TEXT,
                    dredge_numeric REAL,
                    vessel TEXT,
                    crs TEXT NOT NULL
                )""",
                ),
                (
                    "ems_record_bounds",
                    "table",
                    """CREATE VIRTUAL TABLE IF NOT EXISTS ems_record_bounds USING rtree(
                    id, min_x, max_x, min_y, max_y
                )""",
                ),
            )
            for object_name, object_type, create_sql in objects:
                started = perf_counter()
                existed = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
                    (object_type, object_name),
                ).fetchone() is not None
                if timings is not None:
                    timings[f"Schema check: {object_name}"] = (
                        perf_counter() - started
                    )
                started = perf_counter()
                connection.execute(create_sql)
                if timings is not None:
                    timings[f"Schema create/check: {object_name}"] = (
                        perf_counter() - started
                    )

            started = perf_counter()
            source_file_columns = {
                row[1]
                for row in connection.execute(
                    'PRAGMA table_info("source_files")'
                )
            }
            if "crs" not in source_file_columns:
                connection.execute(
                    "ALTER TABLE source_files ADD COLUMN crs TEXT"
                )
                if timings is not None:
                    timings["Migration: source_files.crs column added"] = (
                        perf_counter() - started
                    )
            elif timings is not None:
                timings["Migration check: source_files.crs column"] = (
                    perf_counter() - started
                )

            indexes = (
                (
                    "idx_ems_datetime",
                    """CREATE INDEX IF NOT EXISTS idx_ems_datetime
                       ON ems_records (source_root, profile, timestamp)""",
                ),
                (
                    "idx_ems_source_datetime",
                    """CREATE INDEX IF NOT EXISTS idx_ems_source_datetime
                       ON ems_records (source_root, profile, source_path, timestamp)""",
                ),
                (
                    "idx_ems_source_vessel",
                    """CREATE INDEX IF NOT EXISTS idx_ems_source_vessel
                       ON ems_records (source_root, profile, vessel)""",
                ),
                (
                    "idx_ems_dredge_datetime",
                    """CREATE INDEX IF NOT EXISTS idx_ems_dredge_datetime
                       ON ems_records (source_root, profile, dredge_numeric, timestamp)""",
                ),
            )
            for index_name, create_sql in indexes:
                started = perf_counter()
                existed = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                    (index_name,),
                ).fetchone() is not None
                if timings is not None:
                    timings[f"Index existence check: {index_name}"] = (
                        perf_counter() - started
                    )
                if index_existence is not None:
                    index_existence[index_name] = existed
                started = perf_counter()
                connection.execute(create_sql)
                if timings is not None:
                    timings[f"Index CREATE IF NOT EXISTS: {index_name}"] = (
                        perf_counter() - started
                    )

            for pragma in ("user_version", "schema_version"):
                started = perf_counter()
                pragma_values[pragma] = connection.execute(
                    f"PRAGMA {pragma}"
                ).fetchone()[0]
                if timings is not None:
                    timings[f"Metadata query: {pragma}"] = perf_counter() - started

            for pragma in (
                "journal_mode",
                "synchronous",
                "temp_store",
                "cache_size",
                "mmap_size",
                "page_size",
                "wal_autocheckpoint",
            ):
                started = perf_counter()
                pragma_values[pragma] = connection.execute(
                    f"PRAGMA {pragma}"
                ).fetchone()[0]
                if timings is not None:
                    timings[f"Effective PRAGMA query: {pragma}"] = (
                        perf_counter() - started
                    )

            started = perf_counter()
            connection.commit()
            if timings is not None:
                timings["Schema transaction commit"] = perf_counter() - started
        except Exception:
            started = perf_counter()
            connection.rollback()
            if timings is not None:
                timings["Schema transaction rollback"] = perf_counter() - started
            raise
        finally:
            started = perf_counter()
            connection.close()
            if timings is not None:
                timings["Schema connection close"] = perf_counter() - started

        return pragma_values

    def _format_startup_profile(
        self,
        timings: dict[str, float],
        index_existence: dict[str, bool],
        pragma_values: dict[str, object],
        size_bytes: int,
    ) -> str:
        total = timings["TOTAL EMSDatabase startup"]
        measured = sum(
            value
            for key, value in timings.items()
            if key != "TOTAL EMSDatabase startup"
        )
        other = max(0.0, total - measured)
        lines = [
            "EMS database startup profile",
            "",
            f"Database: {self.path}",
            f"Database size: {size_bytes / 1_000_000_000:.1f} GB",
            "",
        ]
        for label in (
            "Source-root normalisation",
            "Database path/hash calculation",
            "Directory creation/check",
            "Database size query",
            "Database identity/schema validation",
            "WAL SQLite connect",
            "WAL busy_timeout PRAGMA",
            "WAL journal_mode PRAGMA",
            "WAL connection close",
            "Schema SQLite connect",
            "Initialization foreign_keys PRAGMA",
            "Initialization busy_timeout PRAGMA",
            "Initialization synchronous PRAGMA",
            "Initialization temp_store PRAGMA",
            "Schema check: source_files",
            "Schema create/check: source_files",
            "Schema check: ems_records",
            "Schema create/check: ems_records",
            "Schema check: ems_record_bounds",
            "Schema create/check: ems_record_bounds",
        ):
            lines.append(f"{label + ':':42} {timings.get(label, 0.0):8.3f} sec")
        lines.append("")
        lines.append("Index checks:")
        for index_name, existed in index_existence.items():
            lines.append(
                f"    {index_name}: existed before startup: "
                f"{'YES' if existed else 'NO'}"
            )
            check_label = f"Index existence check: {index_name}"
            lines.append(
                f"        Existing-index lookup time: "
                f"{timings.get(check_label, 0.0):.3f} sec"
            )
            label = f"Index CREATE IF NOT EXISTS: {index_name}"
            lines.append(
                f"        {index_name} CREATE INDEX IF NOT EXISTS time: "
                f"{timings.get(label, 0.0):.3f} sec"
            )
        lines.extend(("", "Schema/version and effective PRAGMA metadata:"))
        for name, value in pragma_values.items():
            prefix = "PRAGMA " if name in {
                "journal_mode",
                "synchronous",
                "temp_store",
                "cache_size",
                "mmap_size",
                "page_size",
                "wal_autocheckpoint",
            } else ""
            timing_name = (
                f"Effective PRAGMA query: {name}"
                if prefix
                else f"Metadata query: {name}"
            )
            lines.append(
                f"    {prefix}{name}: {value} "
                f"({timings.get(timing_name, 0.0):.3f} sec)"
            )
        lines.extend(
            (
                "",
                f"Source-file CRS migration/check: "
                f"{timings.get('Migration: source_files.crs column added', timings.get('Migration check: source_files.crs column', 0.0)):.3f} sec.",
                "No ems_records migration or scan is performed at startup.",
                f"Schema transaction commit: {timings.get('Schema transaction commit', 0.0):.3f} sec",
                f"Schema connection close: {timings.get('Schema connection close', 0.0):.3f} sec",
                "ANALYZE: not performed by EMSDatabase startup.",
                "WAL checkpoint: not performed by EMSDatabase startup.",
                f"Other startup work: {other:.3f} sec",
                "",
                f"TOTAL EMSDatabase startup: {total:.3f} sec",
            )
        )
        return "\n".join(lines)

    @staticmethod
    def _normalise_timestamp(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if (
            len(value) == 26
            and value[4] == "-"
            and value[7] == "-"
            and value[10] == " "
            and value[13] == ":"
            and value[16] == ":"
            and value[19] == "."
            and value[:4].isdigit()
            and value[5:7].isdigit()
            and value[8:10].isdigit()
            and value[11:13].isdigit()
            and value[14:16].isdigit()
            and value[17:19].isdigit()
            and value[20:].isdigit()
        ):
            return value
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value
        return parsed.isoformat(sep=" ", timespec="microseconds")

    def sync_files(
        self,
        files: Iterable[Path],
        profile: str,
        parser: Callable[[Path], Iterable[EMSRecord]],
        *,
        force: bool = False,
        log: Callable[[str], None] | None = None,
        cancel_event: Event | None = None,
        progress_callback: Callable[[int], None] | None = None,
        retry_failed: bool = True,
        profile_crs: str | None = None,
        bulk_parser: Callable[
            [Path, str, str, str], Iterable[EMSImportBatch]
        ] | None = None,
    ) -> SyncResult:
        """Import new/changed files; retry unchanged failures unless disabled."""
        sync_started = perf_counter()
        root = str(self.source_root)
        candidates: dict[str, tuple[Path, os.stat_result]] = {}
        for file_path in sorted({Path(path).expanduser().resolve() for path in files}):
            try:
                candidates[str(file_path)] = (file_path, file_path.stat())
            except OSError as exc:
                if log:
                    log(f"Could not inspect EMS file {file_path}: {exc}")

        imported = reused = failed = removed = 0
        if log:
            log(f"Checking {len(candidates):,} discovered EMS files against the index...")
        with self._connection() as connection:
            if profile_crs is not None:
                connection.execute(
                    "UPDATE source_files SET crs = ? "
                    "WHERE source_root = ? AND profile = ? "
                    "AND status = 'ready' AND crs IS NULL",
                    (profile_crs, root, profile),
                )
            previous = {
                row["source_path"]: row
                for row in connection.execute(
                    "SELECT source_path, size_bytes, modified_ns, status, crs "
                    "FROM source_files WHERE source_root = ? AND profile = ?",
                    (root, profile),
                )
            }

        for file_number, (source_path, (file_path, stat)) in enumerate(candidates.items(), start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Processing cancelled by user.")
            if log and (file_number == 1 or file_number % 25 == 0):
                log(
                    f"Indexing EMS file {file_number:,} of {len(candidates):,}: "
                    f"{file_path.name}"
                )
            old = previous.get(source_path)
            if (
                not force
                and old is not None
                and (old["status"] == "ready" or (
                    old["status"] == "failed" and not retry_failed
                ))
                and old["size_bytes"] == stat.st_size
                and old["modified_ns"] == stat.st_mtime_ns
            ):
                if old["status"] == "ready":
                    reused += 1
                else:
                    failed += 1
                if progress_callback:
                    progress_callback(int(file_number / max(1, len(candidates)) * 40))
                continue

            try:
                file_started = perf_counter()
                row_count = 0
                stage_times = {
                    "file_read": 0.0,
                    "vector_conversion": 0.0,
                    "record_build": 0.0,
                    "parameter_build": 0.0,
                    "sqlite_insert": 0.0,
                    "rtree": 0.0,
                    "file_cleanup": 0.0,
                    "source_file_metadata": 0.0,
                }
                vector_timings: dict[str, float] = {}
                observed_crs = profile_crs
                crs_ambiguous = False
                with self._connection(
                    cache_size_kib=64 * 1024,
                    timing=stage_times,
                ) as connection:
                    cleanup_started = perf_counter()
                    connection.execute(
                        "DELETE FROM ems_record_bounds WHERE id IN "
                        "(SELECT id FROM ems_records WHERE source_root = ? AND source_path = ? AND profile = ?)",
                        (root, source_path, profile),
                    )
                    connection.execute(
                        "DELETE FROM ems_records WHERE source_root = ? AND source_path = ? AND profile = ?",
                        (root, source_path, profile),
                    )
                    stage_times["file_cleanup"] += (
                        perf_counter() - cleanup_started
                    )
                    batch = []
                    insert_sql = """
                        INSERT INTO ems_records
                            (source_root, source_path, profile, row_number, timestamp,
                             x, y, dredge, dredge_numeric, vessel, crs)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    def insert_rows(rows: list[tuple[object, ...]]) -> None:
                        nonlocal row_count
                        if not rows:
                            return
                        insert_started = perf_counter()
                        connection.executemany(insert_sql, rows)
                        stage_times["sqlite_insert"] += (
                            perf_counter() - insert_started
                        )
                        row_count += len(rows)
                        if cancel_event is not None and cancel_event.is_set():
                            raise RuntimeError("Processing cancelled by user.")
                        while row_count >= next_progress[0]:
                            if log:
                                log(
                                    f"Indexing {file_path.name}: "
                                    f"{next_progress[0]:,} rows processed..."
                                )
                            next_progress[0] += SYNC_PROGRESS_INTERVAL

                    next_progress = [SYNC_PROGRESS_INTERVAL]
                    if bulk_parser is not None:
                        for parsed_batch in bulk_parser(
                            file_path, root, source_path, profile
                        ):
                            stage_times["file_read"] += (
                                parsed_batch.file_read_seconds
                            )
                            stage_times["vector_conversion"] += (
                                parsed_batch.vector_conversion_seconds
                            )
                            if parsed_batch.vector_timings:
                                for name, seconds in parsed_batch.vector_timings.items():
                                    vector_timings[name] = (
                                        vector_timings.get(name, 0.0) + seconds
                                    )
                            stage_times["parameter_build"] += (
                                parsed_batch.parameter_build_seconds
                            )
                            if (
                                profile_crs is None
                                and not crs_ambiguous
                                and parsed_batch.rows
                            ):
                                batch_crs = parsed_batch.rows[0][10]
                                if observed_crs is None:
                                    observed_crs = (
                                        str(batch_crs) if batch_crs is not None else None
                                    )
                                elif batch_crs != observed_crs:
                                    observed_crs = None
                                    crs_ambiguous = True
                            insert_rows(parsed_batch.rows)
                    else:
                        records = iter(parser(file_path))
                        while True:
                            record_started = perf_counter()
                            try:
                                record = next(records)
                            except StopIteration:
                                break
                            if profile_crs is None and not crs_ambiguous:
                                if observed_crs is None:
                                    observed_crs = record.crs
                                elif record.crs != observed_crs:
                                    observed_crs = None
                                    crs_ambiguous = True
                            stage_times["record_build"] += (
                                perf_counter() - record_started
                            )
                            parameter_started = perf_counter()
                            row_number = row_count + len(batch) + 1
                            x = (
                                record.x
                                if record.x is not None and math.isfinite(record.x)
                                else None
                            )
                            y = (
                                record.y
                                if record.y is not None and math.isfinite(record.y)
                                else None
                            )
                            timestamp = self._normalise_timestamp(record.timestamp)
                            if record.dredge_numeric_parsed:
                                dredge_numeric = record.dredge_numeric
                                if dredge_numeric is not None and not math.isfinite(
                                    dredge_numeric
                                ):
                                    dredge_numeric = None
                            else:
                                dredge_numeric = _numeric(record.dredge)
                            batch.append(
                                (
                                    root,
                                    source_path,
                                    profile,
                                    row_number,
                                    timestamp,
                                    x,
                                    y,
                                    record.dredge,
                                    dredge_numeric,
                                    record.vessel,
                                    record.crs,
                                )
                            )
                            stage_times["parameter_build"] += (
                                perf_counter() - parameter_started
                            )
                            if len(batch) >= INSERT_BATCH_SIZE:
                                insert_rows(batch)
                                batch.clear()
                    if batch:
                        insert_rows(batch)
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Processing cancelled by user.")
                    rtree_started = perf_counter()
                    connection.execute(
                        """
                        INSERT INTO ems_record_bounds
                        SELECT id, x, x, y, y
                        FROM ems_records
                        WHERE source_root = ? AND source_path = ? AND profile = ?
                          AND x IS NOT NULL AND y IS NOT NULL
                        """,
                        (root, source_path, profile),
                    )
                    stage_times["rtree"] = perf_counter() - rtree_started
                    metadata_started = perf_counter()
                    connection.execute(
                        """
                        INSERT INTO source_files
                            (source_root, source_path, profile, size_bytes, modified_ns,
                             record_count, status, imported_at, crs)
                        VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                        ON CONFLICT(source_root, source_path, profile) DO UPDATE SET
                            size_bytes = excluded.size_bytes,
                            modified_ns = excluded.modified_ns,
                            record_count = excluded.record_count,
                            status = excluded.status,
                            imported_at = excluded.imported_at,
                            crs = excluded.crs
                        """,
                        (
                            root,
                            source_path,
                            profile,
                            stat.st_size,
                            stat.st_mtime_ns,
                            row_count,
                            datetime.now().isoformat(timespec="seconds"),
                            observed_crs,
                        ),
                    )
                    stage_times["source_file_metadata"] += (
                        perf_counter() - metadata_started
                    )
                imported += 1
                elapsed = perf_counter() - file_started
                rate = row_count / elapsed if elapsed > 0 else 0.0
                measured = sum(stage_times.values())
                other = max(0.0, elapsed - measured)
                vector_total = stage_times["vector_conversion"]
                detailed_vector_time = sum(vector_timings.values())
                vector_timings["other_vector_work"] = max(
                    0.0, vector_total - detailed_vector_time
                )
                if log:
                    log(
                        f"EMS indexing performance - {file_path.name}\n\n"
                        f"Rows:                         {row_count:,}\n\n"
                        f"File read:                    {stage_times['file_read']:8.1f} sec\n"
                        f"Vector conversion:            {stage_times['vector_conversion']:8.1f} sec\n"
                        f"    X conversion:              {vector_timings.get('x_conversion', 0.0):8.1f} sec\n"
                        f"    Y conversion:              {vector_timings.get('y_conversion', 0.0):8.1f} sec\n"
                        f"    Datetime source assembly:  {vector_timings.get('datetime_source_assembly', 0.0):8.1f} sec\n"
                        f"    Datetime parsing:          {vector_timings.get('datetime_parsing', 0.0):8.1f} sec\n"
                        f"    Datetime formatting:       {vector_timings.get('datetime_formatting', 0.0):8.1f} sec\n"
                        f"    Dredge text preparation:   {vector_timings.get('dredge_text', 0.0):8.1f} sec\n"
                        f"    Dredge numeric conversion: {vector_timings.get('dredge_numeric', 0.0):8.1f} sec\n"
                        f"    Vessel preparation:        {vector_timings.get('vessel', 0.0):8.1f} sec\n"
                        f"    Validity/mask processing:  {vector_timings.get('validity_mask', 0.0):8.1f} sec\n"
                        f"    NumPy conversion:          {vector_timings.get('numpy_conversion', 0.0):8.1f} sec\n"
                        f"    Other vector work:         {vector_timings['other_vector_work']:8.1f} sec\n"
                        f"Record/object build:          {stage_times['record_build']:8.1f} sec\n"
                        f"SQLite parameter/batch build: {stage_times['parameter_build']:8.1f} sec\n"
                        f"SQLite ems_records insert:    {stage_times['sqlite_insert']:8.1f} sec\n"
                        f"R-tree creation:               {stage_times['rtree']:8.1f} sec\n"
                        f"SQLite file cleanup:          {stage_times['file_cleanup']:8.1f} sec\n"
                        f"SQLite source metadata:        {stage_times['source_file_metadata']:8.1f} sec\n"
                        f"SQLite connection setup:      {stage_times['sqlite_connection_setup']:8.1f} sec\n"
                        f"SQLite commit:                {stage_times['sqlite_commit']:8.1f} sec\n"
                        f"SQLite connection close:      {stage_times['sqlite_connection_close']:8.1f} sec\n"
                        f"Other:                        {other:8.1f} sec\n\n"
                        f"Total:                        {elapsed:8.1f} sec\n"
                        f"Overall rate:                 {rate:,.0f} rows/sec"
                    )
            except Exception as exc:
                failed += 1
                with self._connection() as connection:
                    connection.execute(
                        "DELETE FROM ems_record_bounds WHERE id IN "
                        "(SELECT id FROM ems_records WHERE source_root = ? AND source_path = ? AND profile = ?)",
                        (root, source_path, profile),
                    )
                    connection.execute(
                        "DELETE FROM ems_records WHERE source_root = ? AND source_path = ? AND profile = ?",
                        (root, source_path, profile),
                    )
                    connection.execute(
                        """
                        INSERT INTO source_files
                            (source_root, source_path, profile, size_bytes, modified_ns,
                             record_count, status, imported_at, crs)
                        VALUES (?, ?, ?, ?, ?, 0, 'failed', ?, NULL)
                        ON CONFLICT(source_root, source_path, profile) DO UPDATE SET
                            size_bytes = excluded.size_bytes,
                            modified_ns = excluded.modified_ns,
                            record_count = 0,
                            status = excluded.status,
                            imported_at = excluded.imported_at,
                            crs = NULL
                        """,
                        (
                            root,
                            source_path,
                            profile,
                            stat.st_size,
                            stat.st_mtime_ns,
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                if log:
                    log(f"Skipped {file_path.name}: {exc}")
                if (
                    isinstance(exc, RuntimeError)
                    and str(exc) == "Processing cancelled by user."
                ):
                    if log:
                        log(
                            f"EMS sync cancelled after "
                            f"{perf_counter() - sync_started:.1f} seconds."
                        )
                    raise
            if progress_callback:
                progress_callback(int(file_number / max(1, len(candidates)) * 40))

        disappeared = set(previous) - set(candidates)
        if disappeared:
            with self._connection() as connection:
                for source_path in disappeared:
                    connection.execute(
                        "DELETE FROM ems_record_bounds WHERE id IN "
                        "(SELECT id FROM ems_records WHERE source_root = ? AND source_path = ? AND profile = ?)",
                        (root, source_path, profile),
                    )
                    connection.execute(
                        "DELETE FROM ems_records WHERE source_root = ? AND source_path = ? AND profile = ?",
                        (root, source_path, profile),
                    )
                    connection.execute(
                        "DELETE FROM source_files WHERE source_root = ? AND source_path = ? AND profile = ?",
                        (root, source_path, profile),
                    )
                    removed += 1

        result = SyncResult(len(candidates), imported, reused, failed, removed)
        if log:
            log(
                f"EMS index: {imported:,} imported/refreshed, {reused:,} unchanged reused, "
                f"{failed:,} failed, {removed:,} removed; {len(candidates):,} source files."
            )
            log(f"EMS sync time: {perf_counter() - sync_started:.1f} seconds.")
            log(f"Shared EMS database: {self.path}")
        return result

    def rebuild(
        self,
        files: Iterable[Path],
        profile: str,
        parser: Callable[[Path], Iterable[EMSRecord]],
        *,
        log: Callable[[str], None] | None = None,
        cancel_event: Event | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> SyncResult:
        """Force a complete refresh from the authoritative EMS source files."""
        return self.sync_files(
            files,
            profile,
            parser,
            force=True,
            log=log,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )

    def repair_fallback_vessels(
        self,
        profile: str,
        log: Callable[[str], None] | None = None,
    ) -> int:
        """Rename rows that took a date-only file name as their vessel.

        Older imports used the file name for every file without a vessel
        column, so same-named files from different vessel folders were merged
        into one vessel. Each file is fixed in its own transaction and files
        already fixed are skipped, so this is safe to interrupt and re-run.
        """
        root = str(self.source_root)
        with self._connection() as connection:
            paths = [
                row["source_path"]
                for row in connection.execute(
                    "SELECT source_path FROM source_files "
                    "WHERE source_root = ? AND profile = ? AND status = 'ready'",
                    (root, profile),
                )
            ]
        updated = 0
        for source_path in paths:
            text = str(source_path)
            path = PureWindowsPath(text) if "\\" in text else PurePosixPath(text)
            old_name = path.stem
            new_name = fallback_vessel_name(text)
            if new_name == old_name:
                continue
            with self._connection() as connection:
                sample = connection.execute(
                    "SELECT vessel FROM ems_records WHERE source_root = ? "
                    "AND profile = ? AND source_path = ? LIMIT 1",
                    (root, profile, source_path),
                ).fetchone()
                if sample is None or sample["vessel"] != old_name:
                    continue
                if log is not None:
                    log(
                        f"Repairing EMS vessel name for {text}: "
                        f"'{old_name}' -> '{new_name}' (one-time update)..."
                    )
                started = perf_counter()
                changed = connection.execute(
                    "UPDATE ems_records SET vessel = ? WHERE source_root = ? "
                    "AND profile = ? AND source_path = ? AND vessel = ?",
                    (new_name, root, profile, source_path, old_name),
                ).rowcount
            updated += changed
            if log is not None:
                log(
                    f"Renamed {changed:,} EMS records in "
                    f"{perf_counter() - started:.1f} sec."
                )
        return updated

    def ready_profiles(self, prefix: str = "") -> list[tuple[str, int]]:
        """Import profiles for this source root with ready files, most files first."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT profile, COUNT(*) AS files FROM source_files "
                "WHERE source_root = ? AND status = 'ready' AND profile LIKE ? "
                "GROUP BY profile ORDER BY files DESC, profile",
                (str(self.source_root), prefix.replace("%", "") + "%"),
            ).fetchall()
        return [(str(row["profile"]), int(row["files"])) for row in rows]

    def months_without_records(
        self, profile: str, start: datetime, end: datetime
    ) -> list[str]:
        """Calendar months between start and end with no EMS rows at all."""
        root = str(self.source_root)
        missing: list[str] = []
        year, month = start.year, start.month
        with self._connection() as connection:
            while (year, month) <= (end.year, end.month):
                next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
                row = connection.execute(
                    "SELECT 1 FROM ems_records WHERE source_root = ? AND profile = ? "
                    "AND timestamp >= ? AND timestamp < ? LIMIT 1",
                    (
                        root,
                        profile,
                        f"{year:04d}-{month:02d}-01 00:00:00.000000",
                        f"{next_year:04d}-{next_month:02d}-01 00:00:00.000000",
                    ),
                ).fetchone()
                if row is None:
                    missing.append(f"{year:04d}-{month:02d}")
                year, month = next_year, next_month
        return missing

    def record_count(self, profile: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(record_count), 0) AS total FROM source_files "
                "WHERE source_root = ? AND profile = ? AND status = 'ready'",
                (str(self.source_root), profile),
            ).fetchone()
        return int(row["total"])

    def source_file_count(self, profile: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM source_files "
                "WHERE source_root = ? AND profile = ? AND status = 'ready'",
                (str(self.source_root), profile),
            ).fetchone()
        return int(row["total"])

    def count_dated_records(
        self, profile: str, start: str | None = None, end: str | None = None
    ) -> int:
        clauses = ["source_root = ?", "profile = ?", "timestamp IS NOT NULL"]
        values: list[object] = [str(self.source_root), profile]
        if start is not None:
            clauses.append("timestamp >= ?")
            values.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            values.append(end)
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM ems_records WHERE {' AND '.join(clauses)}",
                values,
            ).fetchone()
        return int(row["total"])

    def _build_query_records_sql(
        self,
        profile: str,
        *,
        start: str | None = None,
        end: str | None = None,
        bounds_by_crs: Mapping[str, tuple[float, float, float, float]] | None = None,
        dredge_value: float | None = None,
        dredge_text_value: str | None = None,
        dredge_mode: str = "equals",
        coordinates_only: bool = False,
        order_by_vessel_time: bool = False,
        simplify_vessel_order: bool = False,
        spatial_first: bool = False,
    ) -> tuple[str, list[object], str]:
        clauses = ["r.source_root = ?", "r.profile = ?", "f.status = 'ready'"]
        values: list[object] = [str(self.source_root), profile]
        if start is not None:
            clauses.append("r.timestamp >= ?")
            values.append(start)
        if end is not None:
            clauses.append("r.timestamp <= ?")
            values.append(end)
        if dredge_value is not None:
            if dredge_mode == "equals":
                numeric_clause = "r.dredge_numeric = ?"
            elif dredge_mode == "greater_or_equal":
                numeric_clause = "r.dredge_numeric >= ?"
            else:
                raise ValueError(f"Unsupported dredge comparison mode: {dredge_mode}")
            values.append(dredge_value)
            if dredge_text_value is not None:
                clauses.append(
                    f"({numeric_clause} OR (r.dredge_numeric IS NULL "
                    "AND casefold(r.dredge) = casefold(?)))"
                )
                values.append(dredge_text_value.strip())
            else:
                clauses.append(numeric_clause)
        elif dredge_text_value is not None:
            clauses.append(
                "(r.dredge_numeric IS NULL AND casefold(r.dredge) = casefold(?))"
            )
            values.append(dredge_text_value.strip())
        if coordinates_only:
            clauses.extend(("r.x IS NOT NULL", "r.y IS NOT NULL"))

        use_spatial_index = bounds_by_crs is not None
        if use_spatial_index:
            spatial_parts = []
            for crs, (min_x, min_y, max_x, max_y) in bounds_by_crs.items():
                spatial_parts.append(
                    "(r.crs = ? AND b.max_x >= ? AND b.min_x <= ? "
                    "AND b.max_y >= ? AND b.min_y <= ?)"
                )
                values.extend((crs, min_x, max_x, min_y, max_y))
            clauses.append("(" + " OR ".join(spatial_parts) + ")" if spatial_parts else "0")

        joins = (
            "JOIN ems_record_bounds AS b ON b.id = r.id "
            if use_spatial_index
            else ""
        )
        if spatial_first:
            if not use_spatial_index:
                raise ValueError("Spatial-first querying requires CRS-specific bounds.")
            from_where = """
                FROM ems_record_bounds AS b
                CROSS JOIN ems_records AS r ON b.id = r.id
                JOIN source_files AS f
                  ON f.source_root = r.source_root
                 AND f.source_path = r.source_path
                 AND f.profile = r.profile
                WHERE
            """ + " AND ".join(clauses)
        else:
            from_where = f"""
                FROM ems_records AS r
                JOIN source_files AS f
                  ON f.source_root = r.source_root
                 AND f.source_path = r.source_path
                 AND f.profile = r.profile
                {joins}
                WHERE {' AND '.join(clauses)}
            """
        query = f"""
            SELECT r.source_path, r.row_number, r.timestamp, r.x, r.y,
                   r.dredge, r.vessel, r.crs
            {from_where}
        """
        if order_by_vessel_time:
            ordering = (
                "r.vessel, r.timestamp, r.source_path, r.row_number"
                if simplify_vessel_order
                else "r.vessel IS NOT NULL, r.vessel, r.timestamp, "
                     "r.source_path, r.row_number"
            )
        else:
            ordering = "r.source_path, r.row_number"
        ordered_query = f"{query.strip()}\nORDER BY {ordering}"
        count_query = f"SELECT COUNT(*) {from_where}"
        return ordered_query, values, count_query

    def count_query_records(
        self,
        profile: str,
        *,
        start: str | None = None,
        end: str | None = None,
        bounds_by_crs: Mapping[str, tuple[float, float, float, float]] | None = None,
        dredge_value: float | None = None,
        dredge_text_value: str | None = None,
        dredge_mode: str = "equals",
        coordinates_only: bool = False,
        log: Callable[[str], None] | None = None,
    ) -> tuple[int, float]:
        """Explicit diagnostic COUNT using the iterator's filters, without ordering."""
        _, values, count_query = self._build_query_records_sql(
            profile,
            start=start,
            end=end,
            bounds_by_crs=bounds_by_crs,
            dredge_value=dredge_value,
            dredge_text_value=dredge_text_value,
            dredge_mode=dredge_mode,
            coordinates_only=coordinates_only,
        )
        connection = self._connect()
        connection.create_function(
            "casefold",
            1,
            lambda value: "" if value is None else str(value).strip().casefold(),
        )
        try:
            if log is not None:
                log(f"EMS query COUNT diagnostic SQL:\n{count_query.strip()}")
                log(f"EMS query COUNT parameters: {tuple(values)!r}")
                log("Starting unordered matching COUNT diagnostic...")
            started = perf_counter()
            row = connection.execute(count_query, values).fetchone()
            elapsed = perf_counter() - started
            count = int(row[0])
            if log is not None:
                log(
                    f"EMS query COUNT diagnostic: {count:,} matching rows "
                    f"in {elapsed:.3f} sec."
                )
            return count, elapsed
        finally:
            connection.close()

    def source_crs_values(
        self,
        profile: str,
        *,
        log: Callable[[str], None] | None = None,
        diagnostic_label: str = "source_crs_values",
        diagnostic_fallback: bool = False,
    ) -> list[str]:
        query = (
            "SELECT DISTINCT crs FROM source_files "
            "WHERE source_root = ? AND profile = ? "
            "AND status = 'ready' AND crs IS NOT NULL "
            "ORDER BY crs"
        )
        values = (str(self.source_root), profile)
        with self._connection() as connection:
            try:
                self._log_query_diagnostics(
                    connection, query, values, log, diagnostic_label
                )
                started = perf_counter()
                rows = connection.execute(query, values).fetchall()
                if log is not None:
                    log(
                        f"{diagnostic_label} SQL execution completed in "
                        f"{perf_counter() - started:.3f} sec."
                    )
            except sqlite3.OperationalError as exc:
                if "no such column: crs" not in str(exc).casefold():
                    raise
                rows = []
                if log is not None:
                    log(
                        f"{diagnostic_label}: source_files.crs is unavailable "
                        "in this legacy schema."
                    )
            if rows or not diagnostic_fallback:
                return [str(row["crs"]) for row in rows]
            if log is not None:
                log(
                    f"{diagnostic_label}: running explicit record-level "
                    "CRS diagnostic fallback."
                )
            fallback_query = (
                "SELECT DISTINCT r.crs FROM ems_records AS r "
                "JOIN source_files AS f ON f.source_root = r.source_root "
                "AND f.source_path = r.source_path AND f.profile = r.profile "
                "WHERE r.source_root = ? AND r.profile = ? "
                "AND f.status = 'ready' ORDER BY r.crs"
            )
            self._log_query_diagnostics(
                connection,
                fallback_query,
                values,
                log,
                f"{diagnostic_label} explicit record fallback",
            )
            rows = connection.execute(fallback_query, values).fetchall()
            return [str(row["crs"]) for row in rows]

    def resolve_crs_values(
        self,
        profile: str,
        *,
        configured_crs: str | None = None,
        log: Callable[[str], None] | None = None,
        diagnostic_fallback: bool = False,
    ) -> list[str]:
        started = perf_counter()
        if log is not None:
            log("Resolving EMS CRS values...")
        if configured_crs is not None:
            values = [configured_crs]
            source = "profile configuration"
        else:
            values = self.source_crs_values(
                profile,
                log=log,
                diagnostic_fallback=diagnostic_fallback,
            )
            source = "source_files metadata"
        if log is not None:
            log(f"EMS CRS source: {source}")
            log(f"EMS CRS values: {', '.join(values) if values else '(none)'}")
            log(
                "EMS CRS resolution completed in "
                f"{perf_counter() - started:.3f} sec."
            )
        return values

    @staticmethod
    def _log_query_diagnostics(
        connection: sqlite3.Connection,
        query: str,
        values: Iterable[object],
        log: Callable[[str], None] | None,
        label: str,
    ) -> None:
        if log is None:
            return
        parameters = tuple(values)
        log(f"{label} SQL:\n{query.strip()}")
        log(f"{label} parameters: {parameters!r}")
        explain_started = perf_counter()
        plan_rows = connection.execute(
            "EXPLAIN QUERY PLAN " + query, parameters
        ).fetchall()
        log(
            f"{label} EXPLAIN QUERY PLAN completed in "
            f"{perf_counter() - explain_started:.3f} sec."
        )
        plan = [str(row[3]) for row in plan_rows]
        for detail in plan:
            log(f"{label} query plan: {detail}")
        uses_index = any(
            "USING INDEX" in detail.upper()
            or "USING COVERING INDEX" in detail.upper()
            or "USING AUTOMATIC INDEX" in detail.upper()
            or "USING INTEGER PRIMARY KEY" in detail.upper()
            for detail in plan
        )
        scans_records = any(
            "SCAN R" in detail.upper()
            or "SCAN EMS_RECORDS" in detail.upper()
            for detail in plan
        )
        temp_distinct = any(
            "USE TEMP B-TREE FOR DISTINCT" in detail.upper()
            for detail in plan
        )
        temp_order = any(
            "USE TEMP B-TREE FOR ORDER BY" in detail.upper()
            for detail in plan
        )
        rtree_search = any(
            "VIRTUAL TABLE INDEX" in detail.upper()
            for detail in plan
        )
        log(
            f"{label} plan summary: using index={'YES' if uses_index else 'NO'}; "
            f"scanning ems_records={'YES' if scans_records else 'NO'}; "
            f"using ems_record_bounds={'YES' if rtree_search else 'NO'}; "
            f"R-tree virtual-table search={'YES' if rtree_search else 'NO'}; "
            f"temporary B-tree for DISTINCT={'YES' if temp_distinct else 'NO'}; "
            f"temporary B-tree for ORDER BY={'YES' if temp_order else 'NO'}."
        )

    def iter_records(
        self,
        profile: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> Iterator[dict[str, object]]:
        """Stream dated EMS rows in vessel/time order without loading the archive."""
        clauses = ["r.source_root = ?", "r.profile = ?", "f.status = 'ready'"]
        values: list[object] = [str(self.source_root), profile]
        if start is not None:
            clauses.append("r.timestamp >= ?")
            values.append(start)
        if end is not None:
            clauses.append("r.timestamp <= ?")
            values.append(end)

        query = f"""
            SELECT r.source_path, r.row_number, r.timestamp, r.x, r.y,
                   r.dredge, r.vessel, r.crs
            FROM ems_records AS r
            JOIN source_files AS f
              ON f.source_root = r.source_root
             AND f.source_path = r.source_path
             AND f.profile = r.profile
            WHERE {' AND '.join(clauses)}
            ORDER BY r.vessel IS NOT NULL, r.vessel, r.timestamp,
                     r.source_path, r.row_number
        """
        connection = self._connect()
        try:
            cursor = connection.execute(query, values)
            for row in cursor:
                yield dict(row)
        finally:
            connection.close()

    def query_records(
        self,
        profile: str,
        *,
        start: str | None = None,
        end: str | None = None,
        bounds_by_crs: Mapping[str, tuple[float, float, float, float]] | None = None,
        dredge_value: float | None = None,
        dredge_mode: str = "equals",
        coordinates_only: bool = False,
    ) -> list[dict[str, object]]:
        """Fetch cached records using indexed date and optional CRS-specific bounds."""
        return list(
            self.iter_query_records(
                profile,
                start=start,
                end=end,
                bounds_by_crs=bounds_by_crs,
                dredge_value=dredge_value,
                dredge_mode=dredge_mode,
                coordinates_only=coordinates_only,
            )
        )

    def iter_query_records(
        self,
        profile: str,
        *,
        start: str | None = None,
        end: str | None = None,
        bounds_by_crs: Mapping[str, tuple[float, float, float, float]] | None = None,
        dredge_value: float | None = None,
        dredge_text_value: str | None = None,
        dredge_mode: str = "equals",
        coordinates_only: bool = False,
        order_by_vessel_time: bool = False,
        log: Callable[[str], None] | None = None,
        cancel_event: Event | None = None,
        simplify_vessel_order: bool = False,
        spatial_first: bool = False,
        diagnostic_label: str = "iter_query_records",
    ) -> Iterator[dict[str, object]]:
        """Stream records selected by indexed dates, dredge state, and optional R-tree bounds."""
        query, values, _ = self._build_query_records_sql(
            profile,
            start=start,
            end=end,
            bounds_by_crs=bounds_by_crs,
            dredge_value=dredge_value,
            dredge_text_value=dredge_text_value,
            dredge_mode=dredge_mode,
            coordinates_only=coordinates_only,
            order_by_vessel_time=order_by_vessel_time,
            simplify_vessel_order=simplify_vessel_order,
            spatial_first=spatial_first,
        )
        connection = self._connect()
        connection.create_function(
            "casefold",
            1,
            lambda value: "" if value is None else str(value).strip().casefold(),
        )
        if cancel_event is not None:
            connection.set_progress_handler(
                lambda: 1 if cancel_event.is_set() else 0,
                1000,
            )
        rows_streamed = 0
        stream_started = perf_counter()
        try:
            self._log_query_diagnostics(
                connection,
                query,
                values,
                log,
                diagnostic_label,
            )
            execute_started = perf_counter()
            cursor = connection.execute(query, values)
            if log is not None:
                log(
                    f"{diagnostic_label} SQL execution returned cursor in "
                    f"{perf_counter() - execute_started:.3f} sec."
                )
            for row in cursor:
                rows_streamed += 1
                elapsed = perf_counter() - stream_started
                if rows_streamed == 1:
                    if log is not None:
                        log(
                            f"{diagnostic_label} time to first row: "
                            f"{elapsed:.3f} sec."
                        )
                if rows_streamed in (100_000, 1_000_000) and log is not None:
                    log(
                        f"{diagnostic_label} time to {rows_streamed:,} rows: "
                        f"{elapsed:.3f} sec."
                    )
                yield dict(row)
        except sqlite3.OperationalError as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("EMS record query cancelled by user.") from exc
            raise
        finally:
            connection.set_progress_handler(None, 0)
            if log is not None:
                log(
                    f"{diagnostic_label} total rows streamed: {rows_streamed:,} in "
                    f"{perf_counter() - stream_started:.3f} sec."
                )
            connection.close()
