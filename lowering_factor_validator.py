#!/usr/bin/env python3
"""
Lowering Factor Calibration and Validation Tool
Version 1.0

Calibrates dredging production rate and grid lowering factor from:
  * an existing bathymetric difference raster;
  * a calibration grid;
  * licence polygons; and
  * EMS point records or delimited EMS files.

Positive standardised change means seabed lowering.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyogrio
from rasterio.windows import Window

try:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import Point, box
    from pyproj import CRS, Transformer
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError, cKDTree
except ImportError as exc:
    raise SystemExit(
        "Missing spatial dependency. Install packages listed in requirements.txt.\n"
        f"Details: {exc}"
    )

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ems_database import (
    INSERT_BATCH_SIZE,
    EMSDatabase,
    EMSDatabaseUnavailableError,
    EMSImportBatch,
    EMSRecord,
    discover_files,
    fallback_vessel_name,
    set_configured_database_path,
    validate_database,
)
from resolution_grid import (
    create_aligned_child_grid,
    full_cell_mask,
    inspect_master_grid,
)

APP_NAME = "Lowering Factor Calibration and Validation Tool"
VERSION = "1.0.0"
RESOLUTION_250 = "250 m Regulatory"
RESOLUTION_50 = "50 m Experimental"
RESOLUTION_BOTH = "Both"
LEGACY_RESOLUTION_50 = "50 m Internal Detail"
EMS_READ_CHUNK_SIZE = 250_000


@dataclass
class Settings:
    difference_raster: str
    calibration_grid: str
    licence_layer: str
    ems_source: str
    output_folder: str
    difference_source: str = "RASTER"
    before_xyz: str = ""
    after_xyz: str = ""
    before_surface_raster: str = ""
    after_surface_raster: str = ""
    before_xyz_crs: str = "EPSG:27700"
    after_xyz_crs: str = "EPSG:27700"
    calibration_grid_layer: str = ""
    licence_layer_name: str = ""
    grid_id_field: str = "GridID"
    licence_field: str = "LicenceNo"
    ems_x_field: str = "GPSLON"
    ems_y_field: str = "GPSLAT"
    ems_datetime_field: str = "DATETIME"
    ems_time_field: str = "TIME"
    ems_dredge_field: str = "DREDGE"
    ems_dredge_value: str = "3"
    ems_dredge_mode: str = "equals"
    ems_vessel_field: str = "VESSEL"
    survey_start: str = ""
    survey_end: str = ""
    source_sign: str = "positive_is_lowering"
    raster_value_convention: str = "ELEVATION"
    difference_units: str = "metres"
    change_tolerance_m: float = 0.05
    validation_mode: str = "DEEPENING_ONLY"
    analysis_extent: str = "FULL_OVERLAP"
    bulk_density: float = 1.7
    sample_interval_minutes: float = 1.0
    minimum_ems_hours: float = 0.5
    minimum_coverage_percent: float = 80.0
    minimum_detectable_change_m: float = 0.20
    maximum_rate_tph: float = 1500.0
    green_tolerance_percent: float = 15.0
    amber_tolerance_percent: float = 30.0
    configured_rate_tph: float = 500.0
    projected_crs: str = "EPSG:27700"
    resolution_mode: str = RESOLUTION_250
    write_model_factors: bool = False
    background_correction: str = "UNDREDGED_CELLS"
    background_min_cells: int = 5
    rate_method: str = "POOLED"
    reuse_ems_cache: bool = True
    licence_filter: str = ""
    model_path: str = ""
    model_layer: str = "areas"
    model_id_field: str = "area_id"

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        clean = {field: data[field] for field in cls.__annotations__ if field in data}
        if clean.get("resolution_mode") == LEGACY_RESOLUTION_50:
            clean["resolution_mode"] = RESOLUTION_50
        return cls(**clean)

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.resolution_mode == LEGACY_RESOLUTION_50:
            self.resolution_mode = RESOLUTION_50
        if self.resolution_mode not in {
            RESOLUTION_250,
            RESOLUTION_50,
            RESOLUTION_BOTH,
        }:
            raise ValueError("Resolution mode must be 250 m Regulatory, 50 m Experimental or Both.")
        required = [
            "calibration_grid",
            "licence_layer",
            "ems_source",
            "output_folder",
        ]
        missing = [field for field in required if not getattr(self, field, "").strip()]
        if missing:
            raise ValueError(f"Select all required input and output paths: {', '.join(missing)}")
        if self.difference_source not in {"RASTER", "XYZ_SURVEYS", "PAIRED_RASTERS"}:
            raise ValueError("Difference source must be RASTER, XYZ_SURVEYS or PAIRED_RASTERS.")
        if self.difference_source == "RASTER" and not self.difference_raster.strip():
            raise ValueError("Select a bathymetric difference raster.")
        if self.difference_source == "XYZ_SURVEYS":
            if not self.before_xyz.strip() or not self.after_xyz.strip():
                raise ValueError("Select both the earlier and later XYZ survey files.")
            for label, value in (
                ("Earlier survey", self.before_xyz_crs),
                ("Later survey", self.after_xyz_crs),
            ):
                try:
                    survey_crs = CRS.from_user_input(value)
                except Exception as exc:
                    raise ValueError(f"{label} CRS '{value}' is invalid.") from exc
                if not survey_crs.is_projected:
                    raise ValueError(f"{label} CRS must be projected and use metre coordinates.")
                if not survey_crs.axis_info or not all(
                    math.isclose(
                        axis.unit_conversion_factor,
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for axis in survey_crs.axis_info[:2]
                ):
                    raise ValueError(f"{label} CRS must use metres on both horizontal axes.")
        if self.difference_source == "PAIRED_RASTERS":
            if not self.before_surface_raster.strip() or not self.after_surface_raster.strip():
                raise ValueError("Select both the earlier and later interpolated surface rasters.")

        for field in ["grid_id_field", "licence_field", "ems_x_field", "ems_y_field", "ems_datetime_field", "ems_dredge_field", "ems_vessel_field"]:
            if not getattr(self, field, "").strip():
                raise ValueError(f"Field setting '{field}' cannot be empty.")

        if self.bulk_density <= 0:
            raise ValueError("Bulk density must be greater than zero.")
        if self.sample_interval_minutes <= 0:
            raise ValueError("EMS sample interval must be greater than zero.")
        if self.minimum_ems_hours < 0:
            raise ValueError("Minimum EMS time per cell cannot be negative.")
        if not 0 <= self.minimum_coverage_percent <= 100:
            raise ValueError("Minimum raster coverage must be between 0 and 100 percent.")
        if self.minimum_detectable_change_m < 0:
            raise ValueError("Minimum detectable lowering cannot be negative.")
        if self.maximum_rate_tph <= 0:
            raise ValueError("Maximum plausible rate must be greater than zero.")
        if self.configured_rate_tph <= 0:
            raise ValueError("Configured EMS rate must be greater than zero.")
        if not 0 <= self.green_tolerance_percent <= 100:
            raise ValueError("GREEN tolerance must be between 0 and 100 percent.")
        if not 0 <= self.amber_tolerance_percent <= 100:
            raise ValueError("AMBER tolerance must be between 0 and 100 percent.")
        if self.green_tolerance_percent > self.amber_tolerance_percent:
            raise ValueError("GREEN tolerance cannot exceed AMBER tolerance.")
        if self.source_sign not in {"positive_is_lowering", "negative_is_lowering"}:
            raise ValueError("Difference raster sign must be 'positive_is_lowering' or 'negative_is_lowering'.")
        if self.raster_value_convention not in {"ELEVATION", "POSITIVE_DEPTH"}:
            raise ValueError("Raster value convention must be 'ELEVATION' or 'POSITIVE_DEPTH'.")
        if self.difference_units != "metres":
            raise ValueError(
                "Bathymetric difference units must be configured as metres. "
                "Convert non-metre data before validation."
            )
        if self.validation_mode not in {"DEEPENING_ONLY", "NET_CHANGE", "TRACK_FOOTPRINT", "FOOTPRINT_AND_CONTROL_AREA"}:
            raise ValueError("Validation mode must be one of DEEPENING_ONLY, NET_CHANGE, TRACK_FOOTPRINT, FOOTPRINT_AND_CONTROL_AREA.")
        if self.analysis_extent not in {"TRACK_FOOTPRINT", "VALIDATION_POLYGON", "FULL_OVERLAP"}:
            raise ValueError("Analysis extent must be TRACK_FOOTPRINT, VALIDATION_POLYGON or FULL_OVERLAP.")
        if self.change_tolerance_m < 0:
            raise ValueError("Change detection tolerance cannot be negative.")
        if self.ems_dredge_mode not in {"equals", "greater_or_equal"}:
            raise ValueError("Dredge comparison mode must be 'equals' or 'greater_or_equal'.")
        if not isinstance(self.write_model_factors, bool):
            raise ValueError("write_model_factors must be true or false.")
        if self.background_correction not in {"NONE", "UNDREDGED_CELLS"}:
            raise ValueError("Background correction must be NONE or UNDREDGED_CELLS.")
        if int(self.background_min_cells) < 1:
            raise ValueError("Background minimum cells must be at least 1.")
        self.background_min_cells = int(self.background_min_cells)
        if self.rate_method not in {"POOLED", "CELL_FILTERS"}:
            raise ValueError("Rate method must be POOLED or CELL_FILTERS.")
        if not isinstance(self.reuse_ems_cache, bool):
            raise ValueError("reuse_ems_cache must be true or false.")
        if not isinstance(self.licence_filter, str):
            raise ValueError("licence_filter must be text, e.g. '137, 522'.")
        if self.write_model_factors and self.model_path.strip():
            if not Path(self.model_path).is_file():
                raise ValueError(f"Model to update was not found: {self.model_path}")
            if not self.model_id_field.strip():
                raise ValueError("Enter the model's area ID field (e.g. area_id).")

        try:
            CRS.from_user_input(self.projected_crs)
        except Exception as exc:
            raise ValueError(f"Invalid projected CRS '{self.projected_crs}'.") from exc


def notify(cb: Optional[Callable[[str], None]], text: str) -> None:
    if cb:
        cb(text)


def settings_from_file(path: str | os.PathLike[str]) -> Settings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Settings.from_dict(raw)


def parse_cli_settings(argv: Optional[list[str]] = None) -> Settings:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--config", type=str, help="JSON settings file")
    parser.add_argument("--difference-raster", type=str, default=None)
    parser.add_argument("--difference-source", choices=["RASTER", "XYZ_SURVEYS", "PAIRED_RASTERS"], default=None)
    parser.add_argument("--before-xyz", type=str, default=None, help="Earlier survey XYZ file")
    parser.add_argument("--after-xyz", type=str, default=None, help="Later survey XYZ file")
    parser.add_argument("--before-surface-raster", type=str, default=None)
    parser.add_argument("--after-surface-raster", type=str, default=None)
    parser.add_argument("--before-xyz-crs", type=str, default=None)
    parser.add_argument("--after-xyz-crs", type=str, default=None)
    parser.add_argument("--calibration-grid", type=str, default=None)
    parser.add_argument("--calibration-grid-layer", type=str, default=None)
    parser.add_argument("--licence-layer", type=str, default=None)
    parser.add_argument("--licence-layer-name", type=str, default=None)
    parser.add_argument("--ems-source", type=str, default=None)
    parser.add_argument("--output-folder", type=str, default=None)
    parser.add_argument("--grid-id-field", type=str, default=None)
    parser.add_argument("--licence-field", type=str, default=None)
    parser.add_argument("--ems-x-field", type=str, default=None)
    parser.add_argument("--ems-y-field", type=str, default=None)
    parser.add_argument("--ems-datetime-field", type=str, default=None)
    parser.add_argument("--ems-time-field", type=str, default=None)
    parser.add_argument("--ems-dredge-field", type=str, default=None)
    parser.add_argument("--ems-dredge-value", type=str, default=None)
    parser.add_argument("--ems-dredge-mode", choices=["equals", "greater_or_equal"], default=None)
    parser.add_argument("--ems-vessel-field", type=str, default=None)
    parser.add_argument("--survey-start", type=str, default=None)
    parser.add_argument("--survey-end", type=str, default=None)
    parser.add_argument(
        "--source-sign",
        choices=["positive_is_lowering", "negative_is_lowering"],
        default=None,
        help=(
            "Which sign means seabed lowering AFTER the raster value convention "
            "is applied (POSITIVE_DEPTH values are negated first)."
        ),
    )
    parser.add_argument("--raster-value-convention", choices=["ELEVATION", "POSITIVE_DEPTH"], default=None)
    parser.add_argument("--difference-units", choices=["metres"], default=None)
    parser.add_argument("--change-tolerance-m", type=float, default=None)
    parser.add_argument("--validation-mode", choices=["DEEPENING_ONLY", "NET_CHANGE", "TRACK_FOOTPRINT", "FOOTPRINT_AND_CONTROL_AREA"], default=None)
    parser.add_argument("--analysis-extent", choices=["TRACK_FOOTPRINT", "VALIDATION_POLYGON", "FULL_OVERLAP"], default=None)
    parser.add_argument("--bulk-density", type=float, default=None)
    parser.add_argument("--sample-interval-minutes", type=float, default=None)
    parser.add_argument("--minimum-ems-hours", type=float, default=None)
    parser.add_argument("--minimum-coverage-percent", type=float, default=None)
    parser.add_argument("--minimum-detectable-change-m", type=float, default=None)
    parser.add_argument("--maximum-rate-tph", type=float, default=None)
    parser.add_argument("--green-tolerance-percent", type=float, default=None)
    parser.add_argument("--amber-tolerance-percent", type=float, default=None)
    parser.add_argument("--configured-rate-tph", type=float, default=None)
    parser.add_argument("--projected-crs", type=str, default=None)
    parser.add_argument(
        "--resolution-mode",
        choices=[RESOLUTION_250, RESOLUTION_50, RESOLUTION_BOTH],
        default=None,
    )
    parser.add_argument(
        "--write-model-factors",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Fill blank lowering_factor values in the licence GeoPackage from "
            "the 250 m validated rates. Off unless this flag is given."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="GeoPackage whose lowering_factor values write-back fills "
        "(default: the licence layer itself).",
    )
    parser.add_argument("--model-layer", default=None, help="Model layer (default: areas).")
    parser.add_argument(
        "--model-id-field", default=None, help="Model area ID field (default: area_id)."
    )
    parser.add_argument(
        "--background-correction",
        choices=["UNDREDGED_CELLS", "NONE"],
        default=None,
        help="Subtract the median lowering of undredged cells (natural change) before rates.",
    )
    parser.add_argument("--background-min-cells", type=int, default=None)
    parser.add_argument(
        "--no-ems-cache",
        dest="reuse_ems_cache",
        action="store_const",
        const=False,
        default=None,
        help="Query the EMS database even when saved EMS records match.",
    )
    parser.add_argument(
        "--licences",
        dest="licence_filter",
        default=None,
        help="Comma-separated licence IDs to run (default: all).",
    )
    parser.add_argument(
        "--rate-method",
        choices=["POOLED", "CELL_FILTERS"],
        default=None,
        help=(
            "POOLED: licence rate from every cell passing the EMS-hours and "
            "coverage filters. CELL_FILTERS: also reject cells on detection "
            "threshold and per-cell rate (previous behaviour)."
        ),
    )
    args = parser.parse_args(argv)

    if args.config:
        loaded = settings_from_file(args.config)
    else:
        loaded = Settings(
            difference_raster="",
            calibration_grid="",
            licence_layer="",
            ems_source="",
            output_folder="",
            grid_id_field="GridID",
            licence_field="LicenceNo",
            ems_x_field="GPSLON",
            ems_y_field="GPSLAT",
            ems_datetime_field="DATETIME",
            ems_time_field="TIME",
            ems_dredge_field="DREDGE",
            ems_dredge_value="3",
            ems_dredge_mode="equals",
            ems_vessel_field="VESSEL",
            survey_start="",
            survey_end="",
            source_sign="positive_is_lowering",
            raster_value_convention="ELEVATION",
            difference_units="metres",
            change_tolerance_m=0.05,
            validation_mode="DEEPENING_ONLY",
            analysis_extent="FULL_OVERLAP",
            bulk_density=1.7,
            sample_interval_minutes=1.0,
            minimum_ems_hours=0.5,
            minimum_coverage_percent=80.0,
            minimum_detectable_change_m=0.20,
            maximum_rate_tph=1500.0,
            green_tolerance_percent=15.0,
            amber_tolerance_percent=30.0,
            configured_rate_tph=500.0,
            projected_crs="EPSG:27700",
        )

    cli_overrides = {
        "difference_raster": args.difference_raster,
        "difference_source": args.difference_source,
        "before_xyz": args.before_xyz,
        "after_xyz": args.after_xyz,
        "before_surface_raster": args.before_surface_raster,
        "after_surface_raster": args.after_surface_raster,
        "before_xyz_crs": args.before_xyz_crs,
        "after_xyz_crs": args.after_xyz_crs,
        "calibration_grid": args.calibration_grid,
        "calibration_grid_layer": args.calibration_grid_layer,
        "licence_layer": args.licence_layer,
        "licence_layer_name": args.licence_layer_name,
        "ems_source": args.ems_source,
        "output_folder": args.output_folder,
        "grid_id_field": args.grid_id_field,
        "licence_field": args.licence_field,
        "ems_x_field": args.ems_x_field,
        "ems_y_field": args.ems_y_field,
        "ems_datetime_field": args.ems_datetime_field,
        "ems_time_field": args.ems_time_field,
        "ems_dredge_field": args.ems_dredge_field,
        "ems_dredge_value": args.ems_dredge_value,
        "ems_dredge_mode": args.ems_dredge_mode,
        "ems_vessel_field": args.ems_vessel_field,
        "survey_start": args.survey_start,
        "survey_end": args.survey_end,
        "source_sign": args.source_sign,
        "raster_value_convention": args.raster_value_convention,
        "difference_units": args.difference_units,
        "change_tolerance_m": args.change_tolerance_m,
        "validation_mode": args.validation_mode,
        "analysis_extent": args.analysis_extent,
        "bulk_density": args.bulk_density,
        "sample_interval_minutes": args.sample_interval_minutes,
        "minimum_ems_hours": args.minimum_ems_hours,
        "minimum_coverage_percent": args.minimum_coverage_percent,
        "minimum_detectable_change_m": args.minimum_detectable_change_m,
        "maximum_rate_tph": args.maximum_rate_tph,
        "green_tolerance_percent": args.green_tolerance_percent,
        "amber_tolerance_percent": args.amber_tolerance_percent,
        "configured_rate_tph": args.configured_rate_tph,
        "projected_crs": args.projected_crs,
        "resolution_mode": args.resolution_mode,
        "write_model_factors": args.write_model_factors,
        "background_correction": args.background_correction,
        "background_min_cells": args.background_min_cells,
        "rate_method": args.rate_method,
        "reuse_ems_cache": args.reuse_ems_cache,
        "licence_filter": args.licence_filter,
        "model_path": args.model_path,
        "model_layer": args.model_layer,
        "model_id_field": args.model_id_field,
    }
    for field_name, value in cli_overrides.items():
        if value is not None:
            setattr(loaded, field_name, value)
    return loaded


def validate_date(value: str, label: str) -> Optional[pd.Timestamp]:
    if not value.strip():
        return None
    dt = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"{label} is not a valid date. Use dd/mm/yyyy.")
    return pd.Timestamp(dt).normalize()


def field_lookup(columns, wanted: str) -> str:
    match = {str(c).strip().lower(): str(c) for c in columns}.get(wanted.strip().lower())
    if not match:
        raise ValueError(f"Required field '{wanted}' was not found.")
    return match


def _field_lookup_any(columns, wanted: str, aliases: tuple[str, ...]) -> str:
    available = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in (wanted, *aliases):
        match = available.get(candidate.strip().lower())
        if match is not None:
            return match
    expected = "', '".join((wanted, *aliases))
    raise ValueError(f"Required field not found; expected one of '{expected}'.")


def _ems_datetime_values(frame: pd.DataFrame, settings: Settings) -> pd.Series:
    try:
        datetime_col = field_lookup(frame.columns, settings.ems_datetime_field)
    except ValueError:
        date_col = _field_lookup_any(frame.columns, "DATE", ("GPSDATE",))
        try:
            time_col = field_lookup(frame.columns, settings.ems_time_field)
        except ValueError:
            time_col = _field_lookup_any(frame.columns, "TIME", ("GPSTIME",))
        return frame[date_col].astype("string").str.strip() + " " + frame[
            time_col
        ].astype("string").str.strip()
    return frame[datetime_col]


def read_vector(
    path: str,
    layer: str = "",
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer or None, bbox=bbox, engine="pyogrio")
    if gdf.empty:
        raise ValueError(f"Vector layer contains no features: {path}")
    if gdf.crs is None:
        raise ValueError(f"Vector layer has no CRS: {path}")
    return gdf


def drop_partial_cells(
    grid: gpd.GeoDataFrame,
    grid_id_column: str,
    projected_crs: str,
    log: Callable[[str], None],
    *,
    cell_size_m: float = 250.0,
    label: str = "calibration grid",
) -> gpd.GeoDataFrame:
    """Remove features that are not complete cell_size_m squares, and log them."""
    check = grid
    if not CRS.from_user_input(grid.crs).is_projected:
        check = grid.to_crs(projected_crs)
    full = np.asarray(full_cell_mask(check, cell_size_m), dtype=bool)
    if full.all():
        return grid
    dropped_ids = grid.loc[~full, grid_id_column].astype(str).tolist()
    kept = grid.loc[full].copy()
    if kept.empty:
        raise ValueError(
            f"The {label} has no complete {cell_size_m:g} x {cell_size_m:g} m "
            f"cells: all {len(grid):,} features are clipped or not "
            f"{cell_size_m:g} m squares. Select the unclipped master grid."
        )
    shown = ", ".join(dropped_ids[:30])
    more = f" and {len(dropped_ids) - 30:,} more" if len(dropped_ids) > 30 else ""
    log(
        f"Removed {len(dropped_ids):,} of {len(grid):,} {label} cells that are "
        f"not complete {cell_size_m:g} x {cell_size_m:g} m squares "
        f"(GridIDs: {shown}{more})."
    )
    return kept


def inspect_authoritative_master_grid(
    grid: gpd.GeoDataFrame,
    grid_id_column: str,
    *,
    cell_size_m: float = 250.0,
    log: Callable[[str], None] | None = None,
):
    """Inspect authoritative grid alignment, tolerating minor polygon-edge defects.

    The normal resolution_grid inspection remains authoritative.  The fallback is
    deliberately conservative: it only substitutes a feature's axis-aligned
    envelope when that envelope is already within 1 metre of the expected cell
    dimensions.  It does not infer alignment from the survey raster and it does
    not accept genuinely clipped or undersized cells.
    """
    if grid.empty:
        raise ValueError("The authoritative calibration grid contains no features.")
    if grid.crs is None:
        raise ValueError("The authoritative calibration grid has no CRS.")

    projected_grid = grid
    grid_crs = CRS.from_user_input(grid.crs)
    if not grid_crs.is_projected:
        raise ValueError(
            "The authoritative calibration grid must use a projected CRS with "
            "metre coordinates before its 250 m alignment can be inspected."
        )

    try:
        return inspect_master_grid(
            projected_grid, grid_id_column, cell_size_m=cell_size_m
        )
    except ValueError as original_error:
        # Some GIS exports contain tiny edge offsets or non-square internal
        # vertices even though the feature envelope is the authoritative cell.
        bounds = projected_grid.geometry.bounds
        widths = bounds["maxx"] - bounds["minx"]
        heights = bounds["maxy"] - bounds["miny"]
        tolerance_m = 1.0
        near_full = (
            widths.sub(cell_size_m).abs().le(tolerance_m)
            & heights.sub(cell_size_m).abs().le(tolerance_m)
            & projected_grid.geometry.notna()
            & ~projected_grid.geometry.is_empty
        )
        candidate_count = int(near_full.sum())
        if candidate_count == 0:
            raise ValueError(
                f"{original_error} The complete authoritative layer was read, "
                f"but none of its {len(projected_grid):,} feature envelopes is "
                f"within {tolerance_m:g} m of {cell_size_m:g} x {cell_size_m:g} m. "
                "Confirm that the selected calibration-grid layer is the "
                "unclipped 250 m regulatory master grid, rather than a survey- "
                "or licence-clipped derivative."
            ) from original_error

        fallback = projected_grid.loc[near_full].copy()
        fallback.geometry = fallback.geometry.envelope
        if log is not None:
            log(
                "Standard master-grid inspection found no exact full cell; "
                f"retrying with {candidate_count:,} near-250 m authoritative "
                "feature envelope(s)."
            )
        try:
            layout = inspect_master_grid(
                fallback, grid_id_column, cell_size_m=cell_size_m
            )
        except ValueError as fallback_error:
            raise ValueError(
                f"{original_error} A conservative retry using {candidate_count:,} "
                "near-250 m feature envelopes also failed. Confirm that the "
                "selected layer contains the original, unrotated authoritative "
                "250 m cells and valid unique grid IDs."
            ) from fallback_error
        if log is not None:
            log(
                "Master-grid alignment established from near-250 m feature "
                "envelopes; the survey raster was not used to define the origin."
            )
        return layout


def raster_bounds_in_vector_crs(
    raster_path: str,
    vector_path: str,
    vector_layer: str = "",
) -> tuple[float, float, float, float]:
    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            raise ValueError("Bathymetric difference raster has no CRS.")
        raster_crs = CRS.from_user_input(raster.crs)
        raster_bounds = raster.bounds
    vector_info = pyogrio.read_info(vector_path, layer=vector_layer or None)
    if not vector_info.get("crs"):
        raise ValueError(f"Vector layer has no CRS: {vector_path}")
    vector_crs = CRS.from_user_input(vector_info["crs"])
    bounds = Transformer.from_crs(
        raster_crs, vector_crs, always_xy=True
    ).transform_bounds(*raster_bounds, densify_pts=21)
    return tuple(float(value) for value in bounds)


def vector_layer_choices(path: str, identifying_field: str) -> tuple[list[str], str]:
    """List vector layers and recommend the layer containing the expected ID field."""
    try:
        layer_rows = pyogrio.list_layers(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Could not list vector layers in '{path}': {exc}") from exc
    if len(layer_rows) == 0:
        raise ValueError(f"No vector layers were found in '{path}'.")

    layers = [str(row[0]) for row in layer_rows]
    matching_layers = []
    for layer_name in layers:
        try:
            fields = pyogrio.read_info(path, layer=layer_name)["fields"]
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Could not inspect fields in layer '{layer_name}' from '{path}': {exc}"
            ) from exc
        if any(str(field).casefold() == identifying_field.casefold() for field in fields):
            matching_layers.append(layer_name)
    return layers, matching_layers[0] if matching_layers else layers[0]


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def update_model_lowering_factors(
    model_path: str,
    layer_name: str,
    area_id_field: str,
    summary: pd.DataFrame,
    log: Callable[[str], None],
) -> int:
    """Fill blank model area factors from validated 250 m rates, without overwriting."""
    path = Path(model_path)
    if path.suffix.lower() != ".gpkg":
        log("Source-model factor update skipped: the selected licence source is not a GeoPackage.")
        return 0

    selected_layer = layer_name or None
    info = pyogrio.read_info(path, layer=selected_layer)
    driver = str(info.get("driver", "")).upper()
    if driver != "GPKG":
        log(
            "Source-model factor update skipped: "
            f"'{path}' is not a GeoPackage data source."
        )
        return 0

    actual_layer = str(info["layer_name"])
    fields = [str(field) for field in info["fields"]]
    factor_field = next(
        (field for field in fields if field.casefold() == "lowering_factor"),
        None,
    )
    if factor_field is None:
        log(
            f"Source-model factor update skipped: layer '{actual_layer}' "
            "does not contain a lowering_factor field."
        )
        return 0
    actual_id_field = field_lookup(fields, area_id_field)

    quoted_layer = _quote_sqlite_identifier(actual_layer)
    quoted_id = _quote_sqlite_identifier(actual_id_field)
    quoted_factor = _quote_sqlite_identifier(factor_field)
    rate_by_area: dict[str, float] = {}
    for row in summary.itertuples(index=False):
        rate = pd.to_numeric(
            pd.Series([getattr(row, "ValidatedRate_m3ph")]), errors="coerce"
        ).iloc[0]
        if pd.isna(rate) or not np.isfinite(rate) or rate <= 0:
            continue
        area_id = str(getattr(row, "LicenceNo")).strip()
        if area_id in rate_by_area:
            raise ValueError(
                f"More than one validation rate was produced for model area {area_id!r}."
            )
        rate_by_area[area_id] = float(rate) / (250.0 * 250.0)

    if not rate_by_area:
        log("Source-model factor update skipped: no valid positive 250 m validation rates were produced.")
        return 0

    updated = 0
    unmatched = set(rate_by_area)
    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("BEGIN IMMEDIATE")
        triggers = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'trigger' AND tbl_name = ? AND lower(name) LIKE ?",
            (actual_layer, f"rtree_{actual_layer.casefold()}_%"),
        ).fetchall()
        for trigger_name, _ in triggers:
            connection.execute(
                f"DROP TRIGGER {_quote_sqlite_identifier(trigger_name)}"
            )

        rows = connection.execute(
            f"SELECT CAST({quoted_id} AS TEXT), {quoted_factor} FROM {quoted_layer}"
        ).fetchall()
        seen_ids = set()
        duplicate_ids = set()
        for row in rows:
            area_id = row[0]
            if area_id is not None and area_id in seen_ids:
                duplicate_ids.add(area_id)
            if area_id is not None:
                seen_ids.add(area_id)
        duplicate_ids = sorted(duplicate_ids)
        if duplicate_ids:
            raise ValueError(
                f"Cannot update model factors because layer '{actual_layer}' "
                f"contains duplicate area IDs: {', '.join(duplicate_ids[:10])}."
            )

        for area_id, factor in rate_by_area.items():
            matched = connection.execute(
                f"SELECT {quoted_factor} FROM {quoted_layer} "
                f"WHERE CAST({quoted_id} AS TEXT) = ?",
                (area_id,),
            ).fetchall()
            if not matched:
                continue
            unmatched.discard(area_id)
            if len(matched) != 1:
                raise ValueError(
                    f"Expected one row for area ID {area_id!r} in layer '{actual_layer}', "
                    f"found {len(matched)}."
                )
            if matched[0][0] is not None and str(matched[0][0]).strip():
                continue
            connection.execute(
                f"UPDATE {quoted_layer} SET {quoted_factor} = ? "
                f"WHERE CAST({quoted_id} AS TEXT) = ? "
                f"AND ({quoted_factor} IS NULL OR TRIM(CAST({quoted_factor} AS TEXT)) = '')",
                (factor, area_id),
            )
            updated += connection.execute("SELECT changes()").fetchone()[0]
        for _, trigger_sql in triggers:
            connection.execute(trigger_sql)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    if unmatched:
        log(
            "No source-model area matched these validation IDs: "
            + ", ".join(sorted(unmatched)[:20])
        )
    log(
        f"Updated {updated:,} blank lowering_factor value(s) in "
        f"{path.name} | {actual_layer}; existing nonblank values were preserved."
    )
    return updated


def _iter_delimited_chunks(path: Path, *, include_read_time: bool = False):
    attempts = [
        dict(sep=","),
        dict(sep="\t"),
        dict(sep=";"),
        dict(sep=None, engine="python"),
    ]
    last = None
    selected_kwargs = None
    for kwargs in attempts:
        try:
            header = pd.read_csv(
                path, nrows=0, encoding_errors="replace", **kwargs
            )
            if len(header.columns) > 1:
                selected_kwargs = kwargs
                break
        except Exception as exc:
            last = exc
    if selected_kwargs is None:
        raise ValueError(f"Could not read {path.name}: {last}")
    read_options = {
        "encoding_errors": "replace",
        "chunksize": EMS_READ_CHUNK_SIZE,
        **selected_kwargs,
    }
    if selected_kwargs.get("engine") != "python":
        read_options["low_memory"] = False
    with pd.read_csv(path, **read_options) as chunks:
        while True:
            read_started = time.perf_counter()
            try:
                frame = next(chunks)
            except StopIteration:
                break
            read_seconds = time.perf_counter() - read_started
            yield (frame, read_seconds) if include_read_time else frame


def _validator_record(
    x,
    y,
    timestamp,
    dredge,
    vessel,
    crs: str,
) -> EMSRecord:
    if isinstance(timestamp, (datetime, pd.Timestamp)):
        parsed_datetime = timestamp
    else:
        parsed_datetime = pd.to_datetime(timestamp, dayfirst=True, errors="coerce")
    try:
        dredge_numeric = float(dredge)
        if not math.isfinite(dredge_numeric):
            dredge_numeric = None
    except (TypeError, ValueError):
        dredge_numeric = None
    return EMSRecord(
        timestamp=None if pd.isna(parsed_datetime) else parsed_datetime.isoformat(
            sep=" ", timespec="microseconds"
        ),
        x=None if pd.isna(x) else float(x),
        y=None if pd.isna(y) else float(y),
        dredge=None if pd.isna(dredge) else str(dredge),
        vessel=None if vessel is None else str(vessel),
        crs=crs,
        dredge_numeric=dredge_numeric,
        dredge_numeric_parsed=True,
    )


def _iter_validator_delimited_records(file: Path, settings: Settings):
    columns = None
    for df in _iter_delimited_chunks(file):
        if columns is None:
            columns = df.columns
            x_col = _field_lookup_any(
                columns, settings.ems_x_field, ("GPSLON", "LONGITUDE")
            )
            y_col = _field_lookup_any(
                columns, settings.ems_y_field, ("GPSLAT", "LATITUDE")
            )
            dredge_col = field_lookup(columns, settings.ems_dredge_field)
            vessel_col = next(
                (
                    column
                    for column in columns
                    if str(column).strip().lower()
                    == settings.ems_vessel_field.strip().lower()
                ),
                None,
            )
        xs = pd.to_numeric(df[x_col], errors="coerce")
        ys = pd.to_numeric(df[y_col], errors="coerce")
        timestamps = pd.to_datetime(
            _ems_datetime_values(df, settings),
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        dredges = df[dredge_col]
        dredge_numerics = pd.to_numeric(dredges, errors="coerce")
        vessels = df[vessel_col] if vessel_col else [fallback_vessel_name(file)] * len(df)
        valid_coordinates = xs.notna() & ys.notna()
        for x, y, timestamp, dredge, dredge_numeric, vessel, valid in zip(
            xs, ys, timestamps, dredges, dredge_numerics, vessels, valid_coordinates
        ):
            if not valid:
                continue
            if pd.isna(dredge_numeric) or not math.isfinite(float(dredge_numeric)):
                dredge_numeric = None
            else:
                dredge_numeric = float(dredge_numeric)
            yield EMSRecord(
                timestamp=(
                    None
                    if pd.isna(timestamp)
                    else timestamp.isoformat(sep=" ", timespec="microseconds")
                ),
                x=float(x),
                y=float(y),
                dredge=None if pd.isna(dredge) else str(dredge),
                vessel=None if vessel is None else str(vessel),
                crs="EPSG:4326",
                dredge_numeric=dredge_numeric,
                dredge_numeric_parsed=True,
            )


def _split_ems_datetime_components(values: pd.Series):
    """Return ISO date/time components and a mask of valid fixed-format values."""
    text = values.astype("string")
    day = text.str.slice(0, 2)
    month_name = text.str.slice(3, 6).str.lower()
    year = text.str.slice(7, 11)
    hour = text.str.slice(12, 14)
    minute = text.str.slice(15, 17)
    second = text.str.slice(18, 20)

    month_lookup = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    month = month_name.map(month_lookup)
    numeric_parts = [
        pd.to_numeric(part, errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        for part in (day, year, hour, minute, second)
    ]
    days, years, hours, minutes, seconds = numeric_parts
    months = pd.to_numeric(month, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    leap_year = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
    month_lengths = np.select(
        [
            np.isin(months, (1, 3, 5, 7, 8, 10, 12)),
            months == 2,
        ],
        [31, np.where(leap_year, 29, 28)],
        default=30,
    )
    within_pandas_datetime_range = (
        (years > 1677)
        | (
            (years == 1677)
            & (
                (months > 9)
                | (
                    (months == 9)
                    & (
                        (days > 21)
                        | (
                            (days == 21)
                            & (
                                (hours > 0)
                                | (
                                    (hours == 0)
                                    & (
                                        (minutes > 12)
                                        | ((minutes == 12) & (seconds >= 44))
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    ) & (
        (years < 2262)
        | (
            (years == 2262)
            & (
                (months < 4)
                | (
                    (months == 4)
                    & (
                        (days < 11)
                        | (
                            (days == 11)
                            & (
                                (hours < 23)
                                | (
                                    (hours == 23)
                                    & (
                                        (minutes < 47)
                                        | ((minutes == 47) & (seconds <= 16))
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    valid = (
        text.str.len().eq(20).to_numpy()
        & text.str.slice(2, 3).eq(" ").to_numpy()
        & text.str.slice(6, 7).eq(" ").to_numpy()
        & text.str.slice(11, 12).eq(" ").to_numpy()
        & text.str.slice(14, 15).eq(":").to_numpy()
        & text.str.slice(17, 18).eq(":").to_numpy()
        & np.isfinite(months)
        & np.isfinite(years)
        & (years >= 1)
        & (years <= 9999)
        & within_pandas_datetime_range
        & np.isfinite(days)
        & (days >= 1)
        & (days <= month_lengths)
        & np.isfinite(hours)
        & (hours >= 0)
        & (hours <= 23)
        & np.isfinite(minutes)
        & (minutes >= 0)
        & (minutes <= 59)
        & np.isfinite(seconds)
        & (seconds >= 0)
        & (seconds <= 59)
    )
    return year, month, day, hour, minute, second, valid


def _iter_validator_delimited_batches(
    file: Path,
    settings: Settings,
    source_root: str,
    source_path: str,
    profile: str,
):
    columns = None
    next_row_number = 1
    pending_read = 0.0
    pending_conversion = 0.0
    pending_parameter_build = 0.0
    pending_vector_timings: dict[str, float] = {}

    def record_vector_time(name: str, elapsed: float) -> None:
        pending_vector_timings[name] = (
            pending_vector_timings.get(name, 0.0) + elapsed
        )

    for df, read_seconds in _iter_delimited_chunks(
        file, include_read_time=True
    ):
        pending_read += read_seconds
        vector_started = time.perf_counter()
        if columns is None:
            columns = df.columns
            x_col = _field_lookup_any(
                columns, settings.ems_x_field, ("GPSLON", "LONGITUDE")
            )
            y_col = _field_lookup_any(
                columns, settings.ems_y_field, ("GPSLAT", "LATITUDE")
            )
            dredge_col = field_lookup(columns, settings.ems_dredge_field)
            vessel_col = next(
                (
                    column
                    for column in columns
                    if str(column).strip().lower()
                    == settings.ems_vessel_field.strip().lower()
                ),
                None,
            )
            datetime_col = next(
                (
                    column
                    for column in columns
                    if str(column).strip().lower()
                    == settings.ems_datetime_field.strip().lower()
                ),
                None,
            )
            uses_split_datetime = datetime_col is None

        operation_started = time.perf_counter()
        xs = pd.to_numeric(df[x_col], errors="coerce")
        record_vector_time(
            "x_conversion", time.perf_counter() - operation_started
        )

        operation_started = time.perf_counter()
        ys = pd.to_numeric(df[y_col], errors="coerce")
        record_vector_time(
            "y_conversion", time.perf_counter() - operation_started
        )

        operation_started = time.perf_counter()
        datetime_source = _ems_datetime_values(df, settings)
        record_vector_time(
            "datetime_source_assembly",
            time.perf_counter() - operation_started,
        )

        operation_started = time.perf_counter()
        if uses_split_datetime:
            (
                year,
                month,
                day,
                hour,
                minute,
                second,
                valid_datetime,
            ) = _split_ems_datetime_components(datetime_source)
        else:
            timestamps = pd.to_datetime(
                datetime_source,
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )
        record_vector_time(
            "datetime_parsing", time.perf_counter() - operation_started
        )

        operation_started = time.perf_counter()
        if uses_split_datetime:
            normalized = (
                year
                + "-"
                + month
                + "-"
                + day
                + " "
                + hour
                + ":"
                + minute
                + ":"
                + second
                + ".000000"
            )
            timestamp_text = normalized.astype(object).where(
                valid_datetime, None
            )
        else:
            try:
                timestamp_text = timestamps.dt.strftime(
                    "%Y-%m-%d %H:%M:%S.%f%z"
                ).str.replace(
                    r"([+-]\d{2})(\d{2})$",
                    r"\1:\2",
                    regex=True,
                )
                timestamp_text = timestamp_text.where(timestamps.notna(), None)
            except AttributeError:
                timestamp_text = timestamps.map(
                    lambda value: (
                        None
                        if pd.isna(value)
                        else value.isoformat(sep=" ", timespec="microseconds")
                    )
                )
        record_vector_time(
            "datetime_formatting", time.perf_counter() - operation_started
        )

        dredges = df[dredge_col]
        operation_started = time.perf_counter()
        dredge_text = (
            dredges.astype("string")
            .astype(object)
            .where(dredges.notna(), None)
        )
        record_vector_time("dredge_text", time.perf_counter() - operation_started)

        operation_started = time.perf_counter()
        dredge_numerics = pd.to_numeric(dredges, errors="coerce")
        record_vector_time(
            "dredge_numeric", time.perf_counter() - operation_started
        )

        operation_started = time.perf_counter()
        valid_coordinates = (xs.notna() & ys.notna()).to_numpy()
        valid_positions = np.flatnonzero(valid_coordinates)
        record_vector_time(
            "validity_mask", time.perf_counter() - operation_started
        )

        operation_started = time.perf_counter()
        x_values = xs.to_numpy(dtype=np.float64, na_value=np.nan)
        y_values = ys.to_numpy(dtype=np.float64, na_value=np.nan)
        dredge_values = dredge_text.to_numpy(dtype=object)
        numeric_values = dredge_numerics.to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        timestamp_values = timestamp_text.to_numpy(dtype=object)
        record_vector_time(
            "numpy_conversion", time.perf_counter() - operation_started
        )

        operation_started = time.perf_counter()
        if vessel_col:
            vessel_values = df[vessel_col].astype(str).to_numpy(dtype=object)
        else:
            vessel_values = None
        record_vector_time("vessel", time.perf_counter() - operation_started)
        pending_conversion += time.perf_counter() - vector_started

        for offset in range(0, len(valid_positions), INSERT_BATCH_SIZE):
            selected_positions = valid_positions[
                offset : offset + INSERT_BATCH_SIZE
            ]
            parameter_started = time.perf_counter()
            rows = [
                (
                    source_root,
                    source_path,
                    profile,
                    next_row_number + index,
                    timestamp_values[position],
                    (
                        float(x_values[position])
                        if math.isfinite(x_values[position])
                        else None
                    ),
                    (
                        float(y_values[position])
                        if math.isfinite(y_values[position])
                        else None
                    ),
                    dredge_values[position],
                    (
                        None
                        if not math.isfinite(numeric_values[position])
                        else float(numeric_values[position])
                    ),
                    (
                        fallback_vessel_name(file)
                        if vessel_values is None
                        else vessel_values[position]
                    ),
                    "EPSG:4326",
                )
                for index, position in enumerate(selected_positions)
            ]
            pending_parameter_build += time.perf_counter() - parameter_started
            next_row_number += len(rows)
            vector_detail = pending_vector_timings
            pending_vector_timings = {}
            yield EMSImportBatch(
                rows=rows,
                file_read_seconds=pending_read,
                vector_conversion_seconds=pending_conversion,
                parameter_build_seconds=pending_parameter_build,
                vector_timings=vector_detail,
            )
            pending_read = pending_conversion = pending_parameter_build = 0.0

    if (
        pending_read
        or pending_conversion
        or pending_parameter_build
        or pending_vector_timings
    ):
        yield EMSImportBatch(
            rows=[],
            file_read_seconds=pending_read,
            vector_conversion_seconds=pending_conversion,
            parameter_build_seconds=pending_parameter_build,
            vector_timings=pending_vector_timings,
        )


def _iter_validator_vector_records(file: Path, settings: Settings):
    gdf = read_vector(str(file))
    dredge_col = field_lookup(gdf.columns, settings.ems_dredge_field)
    timestamps = pd.to_datetime(
        _ems_datetime_values(gdf, settings),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )
    vessel_col = next(
        (
            column
            for column in gdf.columns
            if str(column).strip().lower() == settings.ems_vessel_field.strip().lower()
        ),
        None,
    )
    source_crs = str(gdf.crs)
    for geometry, timestamp, dredge, vessel in zip(
        gdf.geometry,
        timestamps,
        gdf[dredge_col],
        gdf[vessel_col] if vessel_col else [fallback_vessel_name(file)] * len(gdf),
    ):
        if geometry is None or geometry.is_empty or geometry.geom_type != "Point":
            continue
        yield _validator_record(
            geometry.x, geometry.y, timestamp, dredge, vessel, source_crs
        )


def _sampled_record_is_in_bounds(
    record: dict[str, object],
    timestamp: pd.Timestamp,
    bounds_by_crs: dict[str, tuple[float, float, float, float]] | None,
    seconds: int,
    sample_state: list[object],
) -> bool:
    if not _is_first_time_bin_sample(record, timestamp, seconds, sample_state):
        return False
    return _record_is_in_bounds(record, bounds_by_crs)


def _is_first_time_bin_sample(
    record: dict[str, object],
    timestamp: pd.Timestamp,
    seconds: int,
    sample_state: list[object],
) -> bool:
    vessel = record["vessel"]
    sample_bin = timestamp.floor(f"{seconds}s")
    if vessel != sample_state[0]:
        sample_state[0] = vessel
        sample_state[1] = None
    if sample_bin == sample_state[1]:
        return False
    sample_state[1] = sample_bin
    return True


def _record_is_in_bounds(
    record: dict[str, object],
    bounds_by_crs: dict[str, tuple[float, float, float, float]] | None,
) -> bool:
    if bounds_by_crs is None:
        return True
    bounds = bounds_by_crs.get(str(record["crs"]))
    x, y = record["x"], record["y"]
    return bounds is not None and (
        bounds[0] <= float(x) <= bounds[2]
        and bounds[1] <= float(y) <= bounds[3]
    )


def _transform_bounds_by_crs(
    area_grid: gpd.GeoDataFrame,
    crs_values: list[str],
    source_bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    bounds = source_bounds or tuple(
        float(value) for value in area_grid.total_bounds
    )
    return {
        crs: tuple(
            float(value)
            for value in Transformer.from_crs(
                area_grid.crs, crs, always_xy=True
            ).transform_bounds(*bounds, densify_pts=21)
        )
        for crs in crs_values
    }


def _counter_median(counts: "collections.Counter[int]") -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    running = 0
    for value in sorted(counts):
        running += counts[value]
        if running * 2 >= total:
            return float(value)
    return None


def check_ems_logging_interval(
    median_gap_seconds: float | None,
    sample_interval_seconds: int,
    log: Callable[[str], None],
) -> None:
    """Warn when EMS records are further apart than the configured interval."""
    if median_gap_seconds is None:
        log("EMS logging interval check: not enough consecutive records to measure it.")
        return
    log(
        f"EMS logging interval check: median gap between consecutive active "
        f"records {median_gap_seconds:g} sec; configured sample interval "
        f"{sample_interval_seconds:g} sec."
    )
    if median_gap_seconds > sample_interval_seconds * 1.5:
        log(
            "WARNING: EMS records are logged less often than the configured "
            "sample interval, so EMS hours are understated and rates overstated "
            f"by about {median_gap_seconds / sample_interval_seconds:.1f}x. "
            "Set the EMS sample interval to match the logging interval."
        )


def _query_sampled_ems(
    database: EMSDatabase,
    profile: str,
    *,
    start: str | None,
    end: str | None,
    dredge_value: float | None,
    dredge_text_value: str,
    dredge_mode: str,
    coordinates_only: bool,
    exact_bounds_by_crs: dict[str, tuple[float, float, float, float]] | None,
    query_bounds_by_crs: dict[str, tuple[float, float, float, float]] | None,
    seconds: int,
    log: Callable[[str], None],
    label: str,
    cancel_event: threading.Event | None = None,
) -> dict[str, object]:
    spatial_first = query_bounds_by_crs is not None
    state: list[object] = [object(), None]
    gap_counts: collections.Counter[int] = collections.Counter()
    previous_vessel: object = object()
    previous_timestamp: pd.Timestamp | None = None
    retained: list[dict[str, object]] = []
    active_count = 0
    rows_streamed = 0
    sampled_candidates = 0
    query_started = time.perf_counter()
    milestones: dict[int, float] = {}
    if label == "Current pathway":
        log("Starting indexed EMS query...")
    iterator_started = time.perf_counter()
    records = database.iter_query_records(
        profile,
        start=start,
        end=end,
        bounds_by_crs=query_bounds_by_crs,
        dredge_value=dredge_value,
        dredge_text_value=dredge_text_value,
        dredge_mode=dredge_mode,
        coordinates_only=coordinates_only,
        order_by_vessel_time=True,
        log=log,
        spatial_first=spatial_first,
        diagnostic_label=label,
        cancel_event=cancel_event,
    )
    if label == "Current pathway":
        log(
            "iter_query_records iterator created in "
            f"{time.perf_counter() - iterator_started:.3f} sec."
        )
    if label == "Current pathway":
        log("Waiting for first matching EMS record...")
    first_started = time.perf_counter()
    first_record = next(records, None)
    first_elapsed = time.perf_counter() - first_started
    if label == "Current pathway":
        log(
            f"First EMS record returned after {first_elapsed:.3f} sec."
            if first_record is not None
            else "No EMS record was returned by the indexed query."
        )
    log(
        f"{label} time to first row: {first_elapsed:.3f} sec."
        if first_record is not None
        else f"{label} returned no records."
    )
    try:
        records_with_first = (
            itertools.chain((first_record,), records)
            if first_record is not None
            else ()
        )
        for record in records_with_first:
            rows_streamed += 1
            if rows_streamed in (100_000, 1_000_000):
                milestones[rows_streamed] = time.perf_counter() - query_started
            if rows_streamed % 100_000 == 0:
                if label == "Current pathway":
                    log(
                        f"Scanned {rows_streamed:,} indexed EMS rows; "
                        f"{len(retained):,} sampled records overlap the survey/grid extent..."
                    )
                else:
                    log(
                        f"{label} streamed {rows_streamed:,} records; "
                        f"{len(retained):,} sampled records overlap the analysis envelope..."
                    )
            try:
                timestamp = pd.Timestamp(datetime.fromisoformat(record["timestamp"]))
            except (TypeError, ValueError):
                continue
            x, y = record["x"], record["y"]
            if x is None or y is None:
                continue

            active_count += 1
            if (
                previous_timestamp is not None
                and record["vessel"] == previous_vessel
            ):
                gap = int(round((timestamp - previous_timestamp).total_seconds()))
                if gap > 0:
                    gap_counts[gap] += 1
            previous_vessel = record["vessel"]
            previous_timestamp = timestamp
            if not _is_first_time_bin_sample(
                record, timestamp, seconds, state
            ):
                continue
            sampled_candidates += 1
            if not _record_is_in_bounds(record, exact_bounds_by_crs):
                continue
            dredge_text = (
                "" if record["dredge"] is None else str(record["dredge"]).strip()
            )
            retained.append({
                **record,
                "timestamp": timestamp,
                "x": float(x),
                "y": float(y),
                "dredge": dredge_text,
                "vessel": record["vessel"],
            })
    finally:
        records.close()
    elapsed = time.perf_counter() - query_started
    milestone_report = "; ".join(
        f"time to {target:,}: {milestones[target]:.3f} sec"
        if target in milestones
        else f"time to {target:,}: not reached"
        for target in (100_000, 1_000_000)
    )
    rtree_candidate_report = (
        f"R-tree candidate records: {rows_streamed:,}; "
        if spatial_first
        else ""
    )
    log(
        f"{label} summary: {rtree_candidate_report}"
        "Records entering/returned by ordered query: "
        f"{rows_streamed:,}; time to first row: {first_elapsed:.3f} sec; "
        f"rows streamed: {rows_streamed:,}; sampled spatial candidates: "
        f"{sampled_candidates:,}; retained after current bounds check: "
        f"{len(retained):,}; "
        f"{milestone_report}; total query/processing time: {elapsed:.3f} sec."
    )
    return {
        "retained": retained,
        "median_gap_seconds": _counter_median(gap_counts),
        "active_count": active_count,
        "rows_streamed": rows_streamed,
        "sampled_candidates": sampled_candidates,
        "first_row_seconds": first_elapsed,
        "total_seconds": elapsed,
    }


class RunCancelled(RuntimeError):
    """Raised when the user cancels a run or stops it after pre-run checks."""


EMS_CACHE_VERSION = "ems-cache-v1"
VECTOR_EMS_EXTENSIONS = {".gpkg", ".shp", ".geojson", ".tab"}


def selected_licences(settings: Settings) -> list[str]:
    """Licence IDs from the licence filter text; empty means all licences."""
    return [
        item.strip()
        for item in re.split(r"[,;\n]", settings.licence_filter or "")
        if item.strip()
    ]


def _ems_source_kind(settings: Settings) -> str:
    source = Path(settings.ems_source)
    if source.is_file() and source.suffix.lower() in VECTOR_EMS_EXTENSIONS:
        return "vector"
    return "delimited"


def _ems_profile(settings: Settings, source_kind: str) -> str:
    profile_settings = {
        "kind": source_kind,
        "x": settings.ems_x_field,
        "y": settings.ems_y_field,
        "datetime": settings.ems_datetime_field,
        "time": settings.ems_time_field,
        "dredge": settings.ems_dredge_field,
        "vessel": settings.ems_vessel_field,
    }
    return "validator:" + json.dumps(profile_settings, sort_keys=True)


def _ems_cache_paths(settings: Settings) -> tuple[Path, Path]:
    key = {
        "version": EMS_CACHE_VERSION,
        "source": os.path.abspath(settings.ems_source),
        "profile": _ems_profile(settings, _ems_source_kind(settings)),
        "dredge_value": settings.ems_dredge_value,
        "dredge_mode": settings.ems_dredge_mode,
        "survey_start": settings.survey_start,
        "survey_end": settings.survey_end,
        "sample_interval_minutes": settings.sample_interval_minutes,
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    folder = Path(settings.output_folder) / "ems_cache"
    return folder / f"ems_{digest}.pkl", folder / f"ems_{digest}.json"


def find_ems_cache(
    settings: Settings, area_grid: gpd.GeoDataFrame
) -> tuple[Path, dict] | None:
    """Saved EMS records matching these settings and covering the grid, if any."""
    if not settings.reuse_ems_cache:
        return None
    data_path, meta_path = _ems_cache_paths(settings)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bounds = tuple(float(v) for v in area_grid.total_bounds)
        if CRS.from_user_input(meta["crs"]) != CRS.from_user_input(area_grid.crs):
            bounds = Transformer.from_crs(
                area_grid.crs, meta["crs"], always_xy=True
            ).transform_bounds(*bounds, densify_pts=21)
        cached = meta["bounds"]
        tolerance = 1e-6
        covers = (
            cached[0] - tolerance <= bounds[0]
            and cached[1] - tolerance <= bounds[1]
            and cached[2] + tolerance >= bounds[2]
            and cached[3] + tolerance >= bounds[3]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (data_path, meta) if covers else None


def load_ems_cached(
    settings: Settings,
    log: Callable[[str], None],
    area_grid: gpd.GeoDataFrame,
    *,
    cancel_event: threading.Event | None = None,
) -> gpd.GeoDataFrame:
    """Load sampled EMS records, reusing the saved copy when settings match."""
    found = find_ems_cache(settings, area_grid)
    if found is not None:
        data_path, meta = found
        try:
            ems = pd.read_pickle(data_path)
            log(
                f"Using saved EMS records from {meta.get('created_at', 'an earlier run')} "
                f"({len(ems):,} records). Untick 'Reuse EMS data' to rebuild them "
                "from the EMS database (needed if EMS files have changed)."
            )
            return ems
        except Exception as exc:  # corrupt or incompatible pickle
            log(f"Could not read saved EMS records ({exc}); querying the EMS database.")
    elif settings.reuse_ems_cache:
        log("No saved EMS records match these settings and grid extent; querying the EMS database.")
    ems = load_ems(settings, log, area_grid, cancel_event=cancel_event)
    data_path, meta_path = _ems_cache_paths(settings)
    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = data_path.with_suffix(".partial")
        ems.to_pickle(temporary)
        temporary.replace(data_path)
        meta_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "crs": CRS.from_user_input(area_grid.crs).to_wkt(),
                    "bounds": [float(v) for v in area_grid.total_bounds],
                    "records": int(len(ems)),
                    "settings": {
                        "ems_source": settings.ems_source,
                        "survey_start": settings.survey_start,
                        "survey_end": settings.survey_end,
                        "ems_dredge_value": settings.ems_dredge_value,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log(f"Saved EMS records for reuse in later runs: {data_path}")
    except OSError as exc:
        log(f"WARNING: could not save EMS records for reuse: {exc}")
    return ems


def _grid_overlaps(
    grid: gpd.GeoDataFrame, grid_id_column: str, projected_crs: str
) -> list[tuple[str, str, float]]:
    """Pairs of grid cells overlapping by more than 1% of the smaller cell."""
    work = grid[[grid_id_column, "geometry"]].to_crs(projected_crs).reset_index(drop=True)
    geometries = work.geometry
    tree = geometries.sindex
    left, right = tree.query(geometries, predicate="intersects")
    pairs = []
    for i, j in zip(left, right):
        if i >= j:
            continue
        a, b = geometries.iloc[i], geometries.iloc[j]
        area = a.intersection(b).area
        if area > 0.01 * min(a.area, b.area):
            pairs.append(
                (str(work[grid_id_column].iloc[i]), str(work[grid_id_column].iloc[j]), float(area))
            )
    return pairs


def preflight_checks(
    settings: Settings,
    master_grid: gpd.GeoDataFrame,
    licences: gpd.GeoDataFrame,
    grid_id_column: str,
    raster_validation: dict | None,
    log: Callable[[str], None],
    *,
    check_ems_months: bool = True,
) -> list[str]:
    """Quick checks before the slow EMS stage. Returns plain-English warnings."""
    log("Running pre-run checks...")
    warnings: list[str] = []

    licence_field = field_lookup(licences.columns, settings.licence_field)
    work_licences = licences.to_crs(master_grid.crs)
    points = gpd.GeoDataFrame(
        geometry=master_grid.geometry.representative_point(), crs=master_grid.crs
    )
    joined = gpd.sjoin(points, work_licences[[licence_field, "geometry"]], predicate="within")
    with_cells = set(joined[licence_field].astype(str))

    # Overlaps only matter when both cells belong to the same licence (the
    # calculation assigns each cell by its representative point): those
    # cells count the same seabed twice in one licence's rate. Cells outside
    # every licence are ignored by the rate, and overlaps between
    # neighbouring licences' edge cells are a boundary effect.
    cell_licence = (
        joined.assign(_id=master_grid.loc[joined.index, grid_id_column].astype(str).values)
        .drop_duplicates("_id")
        .set_index("_id")[licence_field]
        .astype(str)
    )
    same, boundary, unlicensed = [], [], 0
    for a, b, area in _grid_overlaps(master_grid, grid_id_column, settings.projected_crs):
        licence_a, licence_b = cell_licence.get(a), cell_licence.get(b)
        if licence_a is None or licence_b is None:
            unlicensed += 1
        elif licence_a == licence_b:
            same.append((a, b, licence_a))
        else:
            boundary.append((a, b))
    if same:
        examples = ", ".join(f"{a}/{b}" for a, b, _ in same[:5])
        licences_hit = ", ".join(sorted({licence for _, _, licence in same}))
        warnings.append(
            f"{len(same):,} pairs of overlapping grid cells belong to the same licence "
            f"({licences_hit}; for example {examples}). Lowering and EMS hours there "
            "are counted twice in that licence's rate. Use a grid layer with one "
            "cell per location."
        )
    if boundary:
        log(
            f"Grid check: {len(boundary):,} overlapping cell pairs straddle a boundary "
            "between neighbouring licences (each cell counts for its own licence only)."
        )
    if unlicensed:
        log(
            f"Grid check: {unlicensed:,} overlapping cell pairs involve cells outside "
            "every licence; those cells are not used in any rate."
        )
    empty = [
        str(value) for value in work_licences[licence_field].astype(str)
        if str(value) not in with_cells
    ]
    if empty:
        warnings.append(
            f"{len(empty)} licence(s) have no grid cells and will get no result: "
            + ", ".join(empty[:10])
        )

    start = validate_date(settings.survey_start, "Survey start")
    end = validate_date(settings.survey_end, "Survey end")
    if start is None or end is None:
        warnings.append(
            "Survey start or end date is blank, so EMS records from all dates are "
            "used. Enter the two survey dates so only dredging between them counts."
        )
    elif start >= end:
        warnings.append(
            f"Survey start ({settings.survey_start}) is not before survey end "
            f"({settings.survey_end})."
        )
    elif check_ems_months:
        source = Path(settings.ems_source)
        try:
            if source.exists() and EMSDatabase.exists(source):
                database = EMSDatabase(source)
                missing = database.months_without_records(
                    _ems_profile(settings, _ems_source_kind(settings)),
                    start.to_pydatetime(),
                    end.to_pydatetime(),
                )
                if missing:
                    warnings.append(
                        f"The EMS database has no records at all for {len(missing)} "
                        f"month(s) in the survey period: {', '.join(missing[:12])}"
                        + (" ..." if len(missing) > 12 else "")
                        + ". Dredging in those months is missing, which overstates "
                        "the rate (newly added EMS files are imported during the run)."
                    )
        except Exception as exc:  # the check must never block a run
            log(f"EMS month coverage check skipped: {exc}")

    info = raster_validation or {}
    valid = int(info.get("valid", 0) or 0)
    if valid and "positive_values_are_lowering" in info:
        lowering = int(info.get("positive" if info["positive_values_are_lowering"] else "negative", 0))
        share = 100.0 * lowering / valid
        log(
            f"Raster check: {share:.0f}% of valid pixels will be read as seabed "
            "lowering with the current sign settings."
        )
        if share < 20.0:
            warnings.append(
                f"Only {share:.0f}% of raster pixels will be read as lowering. "
                "Check the raster value convention and lowering sign."
            )

    for warning in warnings:
        log("PRE-RUN CHECK: " + warning)
    if not warnings:
        log("Pre-run checks passed.")
    return warnings


def load_ems(
    settings: Settings,
    log: Callable[[str], None],
    area_grid: gpd.GeoDataFrame | None = None,
    *,
    spatial_first_experiment: bool | None = None,
    cancel_event: threading.Event | None = None,
) -> gpd.GeoDataFrame:
    source = Path(settings.ems_source)
    if not source.exists():
        raise FileNotFoundError(f"EMS source does not exist: {source}")
    log("Opening the shared EMS index...")
    database_started = time.perf_counter()
    database = EMSDatabase(source, log=log)
    log(
        "Shared EMS index opened in "
        f"{time.perf_counter() - database_started:.1f} seconds."
    )
    log(f"EMS index database: {database.path}")
    vector_exts = {".gpkg", ".shp", ".geojson", ".tab"}
    if source.is_file() and source.suffix.lower() in vector_exts:
        files = [source]
        parser = lambda file: _iter_validator_vector_records(file, settings)
        bulk_parser = None
        source_kind = "vector"
    else:
        log(f"Discovering EMS files under {source}...")
        files = discover_files(source, (".csv", ".txt", ".dat"), log=log)
        if not files:
            raise ValueError("No CSV, TXT or DAT EMS files were found.")
        log(f"EMS discovery found {len(files):,} candidate files.")
        parser = lambda file: _iter_validator_delimited_records(file, settings)
        bulk_parser = lambda file, root, source_path, profile: (
            _iter_validator_delimited_batches(
                file, settings, root, source_path, profile
            )
        )
        source_kind = "delimited"

    profile = _ems_profile(settings, source_kind)
    profile_crs = "EPSG:4326" if source_kind == "delimited" else None
    database.sync_files(
        files,
        profile,
        parser,
        log=log,
        bulk_parser=bulk_parser,
        profile_crs=profile_crs,
        cancel_event=cancel_event,
    )
    log("EMS index synchronization complete; preparing the survey record query...")
    if source_kind == "delimited":
        database.repair_fallback_vessels(profile, log=log)

    log("Preparing survey EMS query...")
    started = time.perf_counter()
    date_started = time.perf_counter()
    start = validate_date(settings.survey_start, "Survey start")
    log(
        "Survey start date prepared in "
        f"{time.perf_counter() - date_started:.3f} sec."
    )
    date_started = time.perf_counter()
    end = validate_date(settings.survey_end, "Survey end")
    log(
        "Survey end date prepared in "
        f"{time.perf_counter() - date_started:.3f} sec."
    )
    start_text = (
        start.isoformat(sep=" ", timespec="microseconds") if start is not None else None
    )
    end_text = (
        (end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).isoformat(
            sep=" ", timespec="microseconds"
        )
        if end is not None
        else None
    )
    log(f"Survey dates prepared in {time.perf_counter() - started:.3f} sec.")

    crs_values = database.resolve_crs_values(
        profile, configured_crs=profile_crs, log=log
    )
    if len(crs_values) != 1:
        raise ValueError("EMS source records contain more than one coordinate reference system.")

    bounds_by_crs = None
    if area_grid is not None:
        log("Transforming survey/grid bounds...")
        bounds_started = time.perf_counter()
        analysis_bounds = tuple(float(value) for value in area_grid.total_bounds)
        log(
            "Spatial EMS prefilter analysis envelope "
            f"(grid CRS {area_grid.crs}): "
            f"Min X {analysis_bounds[0]:.3f}, Max X {analysis_bounds[2]:.3f}, "
            f"Min Y {analysis_bounds[1]:.3f}, Max Y {analysis_bounds[3]:.3f}"
        )
        bounds_by_crs = _transform_bounds_by_crs(
            area_grid, crs_values, analysis_bounds
        )
        log(
            "Bounds transformation completed in "
            f"{time.perf_counter() - bounds_started:.3f} sec."
        )
        for crs, (min_x, min_y, max_x, max_y) in bounds_by_crs.items():
            log(
                "Spatial EMS prefilter: "
                f"CRS {crs}; Min X {min_x:.3f}; Max X {max_x:.3f}; "
                f"Min Y {min_y:.3f}; Max Y {max_y:.3f}"
            )

    log("Preparing dredge filter...")
    dredge_started = time.perf_counter()
    seconds = max(1, int(round(settings.sample_interval_minutes * 60)))
    try:
        wanted_numeric = float(settings.ems_dredge_value)
        if not math.isfinite(wanted_numeric):
            wanted_numeric = None
    except (TypeError, ValueError):
        wanted_numeric = None
    log(
        "Dredge filter prepared in "
        f"{time.perf_counter() - dredge_started:.3f} sec."
    )
    if os.environ.get("EMS_QUERY_COUNT_DIAGNOSTIC") == "1":
        database.count_query_records(
            profile,
            start=start_text,
            end=end_text,
            dredge_value=wanted_numeric,
            dredge_text_value=settings.ems_dredge_value,
            dredge_mode=settings.ems_dredge_mode,
            coordinates_only=True,
            log=log,
        )
    query_options = {
        "start": start_text,
        "end": end_text,
        "dredge_value": wanted_numeric,
        "dredge_text_value": settings.ems_dredge_value,
        "dredge_mode": settings.ems_dredge_mode,
        "coordinates_only": True,
        "exact_bounds_by_crs": bounds_by_crs,
        "seconds": seconds,
        "log": log,
        "cancel_event": cancel_event,
    }
    log("Current pathway: date/dredge filters, vessel/time ordering, then sampling.")
    current_result = _query_sampled_ems(
        database,
        profile,
        **query_options,
        query_bounds_by_crs=None,
        label="Current pathway",
    )
    retained = current_result["retained"]
    check_ems_logging_interval(
        current_result["median_gap_seconds"], seconds, log
    )
    if current_result["active_count"] == 0:
        raise ValueError("No active EMS records remain after dredge and date filtering.")

    if spatial_first_experiment is None:
        spatial_first_experiment = (
            os.environ.get("EMS_SPATIAL_FIRST_EXPERIMENT") == "1"
        )
    spatial_result = None
    if spatial_first_experiment:
        if bounds_by_crs is None:
            log(
                "Spatial-first experiment skipped: no analysis grid envelope "
                "was provided."
            )
        else:
            log(
                "Spatial-first experimental pathway enabled. Its result is "
                "compared only; validation continues to use the current pathway."
            )
            spatial_result = _query_sampled_ems(
                database,
                profile,
                **query_options,
                query_bounds_by_crs=bounds_by_crs,
                label="Spatial-first pathway",
            )
            current_keys = {
                (record["source_path"], record["row_number"])
                for record in retained
            }
            spatial_keys = {
                (record["source_path"], record["row_number"])
                for record in spatial_result["retained"]
            }
            log(
                "Spatial-first output comparison: "
                f"{len(current_keys & spatial_keys):,} shared records; "
                f"{len(current_keys - spatial_keys):,} current-only records; "
                f"{len(spatial_keys - current_keys):,} spatial-first-only records."
            )
            log(
                "Spatial-first benchmark: R-tree-qualified records entering "
                f"the ordered query: {spatial_result['rows_streamed']:,}; "
                "current ordered-query records: "
                f"{current_result['rows_streamed']:,}."
            )

    if not retained:
        empty = pd.DataFrame(columns=[
            "source_path", "row_number", "_datetime", "_x", "_y",
            "_dredge", "_vessel", "crs",
        ])
        return gpd.GeoDataFrame(
            empty,
            geometry=gpd.GeoSeries([], crs=crs_values[0]),
            crs=crs_values[0],
        )

    gdf = pd.DataFrame.from_records(retained).rename(
        columns={
            "timestamp": "_datetime",
            "x": "_x",
            "y": "_y",
            "dredge": "_dredge",
            "vessel": "_vessel",
        }
    )
    gdf = gpd.GeoDataFrame(
        gdf,
        geometry=gpd.points_from_xy(gdf["_x"], gdf["_y"]),
        crs=crs_values[0],
    )
    log(f"Retained {len(gdf):,} area-candidate active sampled EMS records.")
    return gdf


def raster_cell_area(ds) -> float:
    return abs(ds.transform.a * ds.transform.e - ds.transform.b * ds.transform.d)


def read_xyz(path: str, label: str, log: Callable[[str], None]) -> tuple[np.ndarray, np.ndarray]:
    try:
        frame = pd.read_csv(
            path,
            sep=r"[\s,;]+",
            engine="python",
            header=None,
            comment="#",
            usecols=[0, 1, 2],
            encoding_errors="replace",
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read {label} XYZ file '{path}': {exc}") from exc

    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    values = frame.to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] < 3:
        raise ValueError(f"{label} XYZ file must contain at least three valid X Y Z points.")

    points_frame = pd.DataFrame(values, columns=["x", "y", "z"])
    points_frame = points_frame.groupby(["x", "y"], as_index=False, sort=False)["z"].mean()
    points = points_frame[["x", "y"]].to_numpy(dtype=float)
    elevations = points_frame["z"].to_numpy(dtype=float)
    if points.shape[0] < 3:
        raise ValueError(f"{label} XYZ file must contain at least three unique XY locations.")
    if np.all(elevations >= 0):
        elevations *= -1
        log(f"{label} XYZ Z values were all non-negative; inverted their signs to use negative seabed elevations.")
    log(f"Loaded {len(points):,} unique points from {label} survey.")
    return points, elevations


def transform_xyz_points(
    points: np.ndarray,
    source_crs: str,
    target_crs: CRS,
    label: str,
    log: Callable[[str], None],
) -> np.ndarray:
    source = CRS.from_user_input(source_crs)
    if not source.is_projected or not source.axis_info or not all(
        math.isclose(axis.unit_conversion_factor, 1.0, rel_tol=0, abs_tol=1e-9)
        for axis in source.axis_info[:2]
    ):
        raise ValueError(f"{label} CRS must be projected and use metre coordinates.")
    transformer = Transformer.from_crs(source, target_crs, always_xy=True)
    x_values, y_values = transformer.transform(
        points[:, 0], points[:, 1], errcheck=True
    )
    transformed = np.column_stack((x_values, y_values))
    if not np.isfinite(transformed).all():
        raise ValueError(
            f"{label} survey coordinates could not be transformed from "
            f"{source.to_string()} to {target_crs.to_string()}."
        )
    log(f"Transformed {label.lower()} survey from {source.to_string()} to {target_crs.to_string()}.")
    return transformed


def estimate_point_spacing(points: np.ndarray) -> float:
    sample_limit = 20_000
    if len(points) > sample_limit:
        rng = np.random.default_rng(0)
        sample = points[rng.choice(len(points), sample_limit, replace=False)]
    else:
        sample = points
    distances, _ = cKDTree(sample).query(sample, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if not nearest.size:
        raise ValueError("Could not estimate XYZ point spacing; check for duplicate or invalid XY coordinates.")
    return float(np.median(nearest))


def create_difference_raster(
    before_path: str,
    after_path: str,
    output_path: Path,
    projected_crs: str,
    log: Callable[[str], None],
    before_crs: str = "EPSG:27700",
    after_crs: str = "EPSG:27700",
) -> Path:
    before_xy, before_z = read_xyz(before_path, "Earlier", log)
    after_xy, after_z = read_xyz(after_path, "Later", log)

    crs = CRS.from_user_input(projected_crs)
    if not crs.is_projected:
        raise ValueError("Calculation CRS must be projected to interpolate XYZ surveys.")
    if not crs.axis_info or not math.isclose(crs.axis_info[0].unit_conversion_factor, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("XYZ coordinates and the calculation CRS must use metres.")

    before_xy = transform_xyz_points(before_xy, before_crs, crs, "Earlier", log)
    after_xy = transform_xyz_points(after_xy, after_crs, crs, "Later", log)

    cell_size = max(estimate_point_spacing(before_xy), estimate_point_spacing(after_xy))
    left = max(float(before_xy[:, 0].min()), float(after_xy[:, 0].min()))
    right = min(float(before_xy[:, 0].max()), float(after_xy[:, 0].max()))
    bottom = max(float(before_xy[:, 1].min()), float(after_xy[:, 1].min()))
    top = min(float(before_xy[:, 1].max()), float(after_xy[:, 1].max()))
    width = int(math.ceil((right - left) / cell_size))
    height = int(math.ceil((top - bottom) / cell_size))
    if width < 1 or height < 1:
        raise ValueError("The two XYZ surveys have no overlapping area.")
    if width * height > 20_000_000:
        raise ValueError(
            "The automatically estimated difference raster would exceed 20 million cells. "
            "Reduce the XYZ input extent or point density and try again."
        )

    try:
        before_surface = LinearNDInterpolator(before_xy, before_z, fill_value=np.nan)
        after_surface = LinearNDInterpolator(after_xy, after_z, fill_value=np.nan)
    except QhullError as exc:
        raise ValueError("Could not triangulate an XYZ survey; each survey needs non-collinear XY points.") from exc

    nodata = -9999.0
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": rasterio.transform.from_origin(left, top, cell_size, cell_size),
        "nodata": nodata,
        "compress": "deflate",
    }
    columns = left + (np.arange(width) + 0.5) * cell_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(output_path, "w", **profile) as destination:
            for row_start in range(0, height, 256):
                row_end = min(row_start + 256, height)
                rows = top - (np.arange(row_start, row_end) + 0.5) * cell_size
                xx, yy = np.meshgrid(columns, rows)
                query = np.column_stack((xx.ravel(), yy.ravel()))
                before_values = before_surface(query)
                after_values = after_surface(query)
                valid = np.isfinite(before_values) & np.isfinite(after_values)
                # Elevation lowering is positive in the validator's internal convention.
                differences = np.full(before_values.shape, nodata, dtype=np.float32)
                differences[valid] = (before_values[valid] - after_values[valid]).astype(np.float32)
                destination.write(differences.reshape(row_end - row_start, width), 1, window=((row_start, row_end), (0, width)))
    except QhullError as exc:
        raise ValueError("Could not interpolate an XYZ survey; check that its points are not collinear.") from exc

    with rasterio.open(output_path) as result:
        valid_cells = int(np.count_nonzero(result.read(1, masked=True).mask == 0))
    if valid_cells == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("The XYZ surveys do not have any shared area that can be interpolated.")
    log(
        f"Created difference raster {output_path.name}: {width:,} x {height:,} cells at "
        f"{cell_size:g} m; {valid_cells:,} cells have data from both surveys."
    )
    return output_path


def create_difference_from_surface_rasters(
    before_path: str,
    after_path: str,
    output_path: Path,
    log: Callable[[str], None],
) -> Path:
    nodata = -9999.0
    valid_cells = 0
    temporary_path = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )
    try:
        with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
            if before.count != 1 or after.count != 1:
                raise ValueError("Each interpolated surface raster must contain exactly one band.")
            if before.crs is None or after.crs is None:
                raise ValueError("Both interpolated surface rasters must have a defined CRS.")
            if before.crs != after.crs:
                raise ValueError(
                    f"Interpolated surface rasters have mismatched CRS values: "
                    f"{before.crs} vs {after.crs}."
                )
            if not (
                math.isclose(before.transform.b, 0.0, abs_tol=1e-12)
                and math.isclose(before.transform.d, 0.0, abs_tol=1e-12)
                and math.isclose(after.transform.b, 0.0, abs_tol=1e-12)
                and math.isclose(after.transform.d, 0.0, abs_tol=1e-12)
            ):
                raise ValueError("Paired surface rasters must use north-up, unrotated grids.")
            if not (
                math.isclose(before.transform.a, after.transform.a, rel_tol=0, abs_tol=1e-9)
                and math.isclose(before.transform.e, after.transform.e, rel_tol=0, abs_tol=1e-9)
            ):
                raise ValueError(
                    "Earlier and later surface rasters must have the same cell size."
                )

            after_column, after_row = (~before.transform) * (
                after.transform.c,
                after.transform.f,
            )
            column_offset, row_offset = round(after_column), round(after_row)
            if not (
                math.isclose(after_column, column_offset, rel_tol=0, abs_tol=1e-7)
                and math.isclose(after_row, row_offset, rel_tol=0, abs_tol=1e-7)
            ):
                raise ValueError(
                    "Earlier and later surface rasters are not aligned to the same "
                    "pixel grid. Align their cell origins in MapInfo Pro before running "
                    "the Validator."
                )
            overlap_column_start = max(0, column_offset)
            overlap_row_start = max(0, row_offset)
            overlap_column_end = min(
                before.width, column_offset + after.width
            )
            overlap_row_end = min(before.height, row_offset + after.height)
            overlap_width = overlap_column_end - overlap_column_start
            overlap_height = overlap_row_end - overlap_row_start
            if overlap_width <= 0 or overlap_height <= 0:
                raise ValueError("The interpolated surface rasters have no overlapping extent.")

            profile = before.profile.copy()
            profile.update(
                driver="GTiff",
                width=overlap_width,
                height=overlap_height,
                transform=before.window_transform(
                    rasterio.windows.Window(
                        overlap_column_start,
                        overlap_row_start,
                        overlap_width,
                        overlap_height,
                    )
                ),
                count=1,
                dtype="float32",
                nodata=nodata,
                compress="deflate",
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(temporary_path, "w", **profile) as destination:
                next_progress_row = 10_000
                window_rows = max(1, min(512, 2_000_000 // overlap_width))
                for row_start in range(0, overlap_height, window_rows):
                    row_end = min(row_start + window_rows, overlap_height)
                    window = rasterio.windows.Window(
                        overlap_column_start,
                        overlap_row_start + row_start,
                        overlap_width,
                        row_end - row_start,
                    )
                    after_window = rasterio.windows.Window(
                        overlap_column_start - column_offset,
                        overlap_row_start - row_offset + row_start,
                        overlap_width,
                        row_end - row_start,
                    )
                    earlier = before.read(1, window=window, masked=True)
                    later = after.read(1, window=after_window, masked=True)
                    valid = (
                        ~np.ma.getmaskarray(earlier)
                        & ~np.ma.getmaskarray(later)
                        & np.isfinite(earlier.data)
                        & np.isfinite(later.data)
                    )
                    differences = np.full(earlier.shape, nodata, dtype=np.float32)
                    differences[valid] = (
                        earlier.data[valid] - later.data[valid]
                    ).astype(np.float32)
                    valid_cells += int(np.count_nonzero(valid))
                    output_window = rasterio.windows.Window(
                        0, row_start, overlap_width, row_end - row_start
                    )
                    destination.write(differences, 1, window=output_window)
                    if row_end >= next_progress_row or row_end == before.height:
                        log(f"Calculated raster difference through row {row_end:,} of {before.height:,}...")
                        next_progress_row = row_end + 10_000
            raster_width, raster_height = overlap_width, overlap_height
    except (OSError, rasterio.errors.RasterioError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Could not create a difference raster from the surfaces: {exc}") from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    if valid_cells == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("The interpolated surface rasters have no shared valid-data cells.")
    temporary_path.replace(output_path)
    log(
        f"Created {output_path.name} from the shared surface extent: "
        f"{raster_width:,} x {raster_height:,} cells; {valid_cells:,} shared valid cells."
    )
    return output_path


def prepare_difference_raster(
    settings: Settings,
    output_folder: Path,
    log: Callable[[str], None],
) -> str:
    if settings.difference_source == "XYZ_SURVEYS":
        return str(create_difference_raster(
            settings.before_xyz,
            settings.after_xyz,
            output_folder / "calculated_bathymetric_difference.tif",
            settings.projected_crs,
            log,
            before_crs=settings.before_xyz_crs,
            after_crs=settings.after_xyz_crs,
        ))
    if settings.difference_source == "PAIRED_RASTERS":
        log("Calculating earlier-minus-later difference from the aligned MapInfo surface rasters...")
        return str(create_difference_from_surface_rasters(
            settings.before_surface_raster,
            settings.after_surface_raster,
            output_folder / "calculated_bathymetric_difference.tif",
            log,
        ))
    return settings.difference_raster


def _inspect_raster_values(source, sample_limit: int = 100_000) -> dict[str, int | float]:
    pixel_count = source.width * source.height
    sample_stride = max(1, math.ceil(pixel_count / sample_limit))
    result: dict[str, int | float] = {
        "masked": 0,
        "nan": 0,
        "infinite": 0,
        "valid": 0,
        "negative": 0,
        "zero": 0,
        "positive": 0,
        "integer_valued": 0,
        "minimum": math.inf,
        "maximum": -math.inf,
        "mean": 0.0,
    }
    samples = []

    for row_off in range(0, source.height, 1024):
        height = min(1024, source.height - row_off)
        for col_off in range(0, source.width, 1024):
            width = min(1024, source.width - col_off)
            window = Window(col_off, row_off, width, height)
            block = source.read(1, window=window, masked=True)
            values = np.asarray(block.data)
            masked = np.ma.getmaskarray(block)
            unmasked = ~masked
            finite = unmasked & np.isfinite(values)
            finite_values = values[finite].astype(np.float64, copy=False)

            result["masked"] += int(masked.sum())
            result["nan"] += int((unmasked & np.isnan(values)).sum())
            result["infinite"] += int((unmasked & np.isinf(values)).sum())
            if finite_values.size == 0:
                continue

            result["valid"] += int(finite_values.size)
            result["negative"] += int((finite_values < 0).sum())
            result["zero"] += int((finite_values == 0).sum())
            result["positive"] += int((finite_values > 0).sum())
            result["integer_valued"] += int(
                np.equal(finite_values, np.trunc(finite_values)).sum()
            )
            result["minimum"] = min(
                float(result["minimum"]), float(finite_values.min())
            )
            result["maximum"] = max(
                float(result["maximum"]), float(finite_values.max())
            )
            previous_count = int(result["valid"]) - int(finite_values.size)
            scale = float(np.max(np.abs(finite_values)))
            block_mean = (
                float(np.mean(finite_values / scale) * scale)
                if scale
                else 0.0
            )
            result["mean"] = (
                float(result["mean"])
                * (previous_count / int(result["valid"]))
                + block_mean
                * (finite_values.size / int(result["valid"]))
            )

            if len(samples) < sample_limit:
                rows = np.arange(row_off, row_off + height, dtype=np.int64)[:, None]
                columns = np.arange(col_off, col_off + width, dtype=np.int64)[None, :]
                systematic = ((rows * source.width + columns) % sample_stride) == 0
                block_samples = values[finite & systematic]
                if block_samples.size:
                    remaining = sample_limit - len(samples)
                    samples.extend(
                        block_samples[:remaining].astype(np.float64).tolist()
                    )

    if samples:
        result["median_sample"] = float(np.median(np.asarray(samples)))
        result["median_sample_count"] = len(samples)
    else:
        result["median_sample"] = math.nan
        result["median_sample_count"] = 0
    if not result["valid"]:
        result["mean"] = math.nan
        result["minimum"] = math.nan
        result["maximum"] = math.nan
    return result


def _raster_summary_lines(
    raster_path: str,
    info: dict[str, object],
) -> list[str]:
    lowering_sign = (
        "Positive"
        if info["positive_values_are_lowering"]
        else "Negative"
    )
    nodata = "not defined" if info["nodata"] is None else str(info["nodata"])
    if info["valid"]:
        valid_range = (
            f"{float(info['minimum']):.6g} m to "
            f"{float(info['maximum']):.6g} m"
        )
    else:
        valid_range = "none"
    return [
        "BATHYMETRIC DIFFERENCE RASTER",
        f"File: {raster_path}",
        f"Format: {info['format']}",
        f"GDAL/rasterio driver: {info['driver']}",
        f"Bands: {info['count']}",
        f"Data type: {info['dtype']}",
        f"CRS: {info['crs']}",
        f"Dimensions: {info['width']:,} x {info['height']:,}",
        f"Pixel size: {info['pixel_width']:.6g} x {info['pixel_height']:.6g} "
        f"{info['pixel_units']}",
        f"NoData: {nodata}",
        f"Valid pixels: {int(info['valid']):,}",
        f"Valid range: {valid_range}",
        f"Configured lowering sign: {lowering_sign} values",
        "Bathymetric difference convention: "
        f"{'positive' if info['positive_values_are_lowering'] else 'negative'} "
        "values represent seabed lowering",
        f"Raster value convention: {info['raster_value_convention']}",
        f"Bathymetric difference units: {info['units']}",
        "Status: VALID",
    ]


def validate_bathymetric_difference_raster(
    raster_path: str,
    difference_units: str = "metres",
    source_sign: str = "positive_is_lowering",
    raster_value_convention: str = "ELEVATION",
    log: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Validate a numeric bathymetric-difference raster and report bounded-memory statistics."""
    if difference_units != "metres":
        raise ValueError(
            "Bathymetric difference units must be configured as metres. "
            "Convert non-metre data before validation."
        )
    if source_sign not in {"positive_is_lowering", "negative_is_lowering"}:
        raise ValueError(f"Unsupported seabed-lowering sign convention: {source_sign}")
    if raster_value_convention not in {"ELEVATION", "POSITIVE_DEPTH"}:
        raise ValueError(
            f"Unsupported raster value convention: {raster_value_convention}"
        )

    path = Path(raster_path)
    if not path.is_file():
        raise ValueError(f"Bathymetric difference raster does not exist: {path}")

    is_flt = path.suffix.casefold() == ".flt"
    expected_header = path.with_suffix(".hdr")
    expected_projection = path.with_suffix(".prj")
    if is_flt and not expected_header.is_file():
        raise ValueError(
            f"Missing ESRI Float Grid header:\n{expected_header}"
        )
    if is_flt and not expected_projection.is_file():
        raise ValueError(
            "The ESRI Float Grid has no usable coordinate reference system.\n"
            "Supply the matching .prj file or explicitly assign the correct CRS "
            "before validation.\n"
            f"Expected projection file: {expected_projection}"
        )

    try:
        source = rasterio.open(path)
    except (OSError, rasterio.errors.RasterioError) as exc:
        if is_flt:
            raise ValueError(
                f"Could not open ESRI Float Grid '{path}' using rasterio/GDAL; "
                f"check that its .flt and .hdr are valid: {exc}"
            ) from exc
        raise ValueError(
            f"Could not open bathymetric-difference raster '{path}': {exc}"
        ) from exc

    with source:
        if source.width <= 0 or source.height <= 0:
            raise ValueError(
                f"Bathymetric-difference raster has invalid dimensions: "
                f"{source.width} x {source.height}."
            )
        if source.count != 1:
            types = ", ".join(source.dtypes)
            color_names = {getattr(value, "name", str(value)).casefold() for value in source.colorinterp}
            looks_rendered = (
                source.count in {3, 4}
                and all(dtype == "uint8" for dtype in source.dtypes)
            ) or bool(color_names & {"red", "green", "blue", "alpha", "palette"})
            rendered_description = (
                " and appears to be an RGB/RGBA rendered image"
                if looks_rendered
                else ""
            )
            raise ValueError(
                "UNSUITABLE BATHYMETRIC-DIFFERENCE RASTER\n\n"
                f"The selected file contains {source.count} bands "
                f"({types}){rendered_description}.\n\n"
                "The Validator requires a single-band numeric "
                "bathymetric-difference surface containing seabed-change "
                "values in metres. Please export the underlying difference "
                "grid as an ESRI Binary Float Grid (.flt with matching .hdr "
                "and .prj files), or as a valid single-band floating-point "
                "GeoTIFF.\n\n"
                "No EMS data has been queried and no validation has been performed."
            )
        dtype = source.dtypes[0]
        if dtype not in {"float32", "float64"}:
            raise ValueError(
                "UNSUITABLE BATHYMETRIC-DIFFERENCE RASTER\n\n"
                f"The selected file has one {dtype} band, not a supported "
                "float32 or float64 numeric surface. Integer or colour-channel "
                "values are not assumed to represent metres.\n\n"
                "Export a single-band floating-point difference grid. No EMS "
                "data has been queried and no validation has been performed."
            )
        color_names = {
            getattr(value, "name", str(value)).casefold()
            for value in source.colorinterp
        }
        if color_names & {"red", "green", "blue", "alpha", "palette"}:
            raise ValueError(
                "UNSUITABLE BATHYMETRIC-DIFFERENCE RASTER\n\n"
                "The selected band is marked as colour or palette data, not "
                "numeric seabed change. Export the underlying difference "
                "surface as a single-band floating-point raster."
            )
        transform_values = tuple(source.transform)
        determinant = (
            source.transform.a * source.transform.e
            - source.transform.b * source.transform.d
        )
        if not all(math.isfinite(value) for value in transform_values) or not math.isfinite(determinant) or determinant == 0:
            raise ValueError(
                "Bathymetric-difference raster has an invalid geotransform or "
                "zero pixel dimensions."
            )
        if source.crs is None:
            if is_flt:
                raise ValueError(
                    "The ESRI Float Grid has no usable coordinate reference system.\n"
                    "Supply the matching .prj file or explicitly assign the correct CRS "
                    "before validation."
                )
            raise ValueError(
                f"Bathymetric difference raster has no CRS: {path}. "
                "Assign its correct CRS before validation; the Validator will not guess."
            )
        coordinate_crs = CRS.from_user_input(source.crs)

        if is_flt:
            expected_size = source.width * source.height * np.dtype(dtype).itemsize
            try:
                header_lines = expected_header.read_text(
                    encoding="ascii", errors="replace"
                ).splitlines()
            except OSError as exc:
                raise ValueError(
                    f"Could not read ESRI Float Grid header '{expected_header}': {exc}"
                ) from exc
            skip_bytes = 0
            header_values = {}
            for line in header_lines:
                parts = line.split()
                if len(parts) >= 2:
                    header_values[parts[0].casefold()] = parts[1]
            byte_order = header_values.get("byteorder", "").upper()
            normalized_byte_order = byte_order.replace("_", "").replace("-", "")
            if normalized_byte_order and normalized_byte_order not in {
                "I",
                "M",
                "LSBFIRST",
                "MSBFIRST",
                "LITTLEENDIAN",
                "BIGENDIAN",
            }:
                raise ValueError(
                    f"Invalid BYTEORDER value in '{expected_header}': {byte_order}. "
                    "Use I/LSBFirst (little-endian) or M/MSBFirst (big-endian)."
                )
            for key, actual in (
                ("ncols", source.width),
                ("nrows", source.height),
                ("nbits", np.dtype(dtype).itemsize * 8),
            ):
                if key in header_values:
                    try:
                        declared = int(header_values[key])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid {key.upper()} value in '{expected_header}': "
                            f"{header_values[key]}"
                        ) from exc
                    if declared != actual:
                        raise ValueError(
                            f"The ESRI Float Grid header {key.upper()} value "
                            f"({declared}) does not match the raster ({actual})."
                        )
            pixel_type = header_values.get("pixeltype", "").casefold()
            if pixel_type and pixel_type != "float":
                raise ValueError(
                    f"The ESRI Float Grid PIXELTYPE must be FLOAT, not "
                    f"'{header_values['pixeltype']}'."
                )
            if "skipbytes" in header_values:
                try:
                    skip_bytes = int(header_values["skipbytes"])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid SKIPBYTES value in '{expected_header}': "
                        f"{header_values['skipbytes']}"
                    ) from exc
                if skip_bytes < 0:
                    raise ValueError(
                        f"Invalid negative SKIPBYTES value in '{expected_header}'."
                    )
            actual_size = path.stat().st_size
            expected_size += skip_bytes
            if actual_size != expected_size:
                raise ValueError(
                    "ESRI Float Grid data size does not match its header: "
                    f"expected {expected_size:,} bytes for {source.width:,} x "
                    f"{source.height:,} {dtype} cells, found {actual_size:,} bytes."
                )

        metadata_units = (source.units[0] or "").strip() if source.units else ""
        normalized_units = metadata_units.casefold().replace(" ", "")
        metre_units = {"m", "meter", "meters", "metre", "metres"}
        if metadata_units and normalized_units not in metre_units:
            raise ValueError(
                "Bathymetric difference raster units are not metres: "
                f"'{metadata_units}'. Convert the raster to metres before validation."
            )
        stats = _inspect_raster_values(source)
        if not stats["valid"]:
            raise ValueError(
                "Bathymetric-difference raster contains no finite, non-NoData "
                "pixels. Check its NoData values and source data. No EMS data "
                "has been queried and no validation has been performed."
            )
        positive_values_are_lowering = (
            raster_value_convention == "ELEVATION"
        ) == (source_sign == "positive_is_lowering")
        info: dict[str, object] = {
            **stats,
            "format": (
                "ESRI Binary Float Grid"
                if source.driver == "EHdr"
                else "GeoTIFF"
                if source.driver == "GTiff"
                else source.driver
            ),
            "driver": source.driver,
            "count": source.count,
            "dtype": dtype,
            "crs": source.crs.to_string(),
            "width": source.width,
            "height": source.height,
            "pixel_width": math.hypot(source.transform.a, source.transform.d),
            "pixel_height": math.hypot(source.transform.b, source.transform.e),
            "pixel_units": (
                "m"
                if coordinate_crs.is_projected
                and coordinate_crs.axis_info
                and math.isclose(
                    coordinate_crs.axis_info[0].unit_conversion_factor,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                else (
                    coordinate_crs.axis_info[0].unit_name
                    if coordinate_crs.axis_info
                    else "map units"
                )
            ),
            "nodata": source.nodata,
            "units": metadata_units or difference_units,
            "metadata_units": metadata_units or None,
            "positive_values_are_lowering": positive_values_are_lowering,
            "raster_value_convention": raster_value_convention,
            "source_sign": source_sign,
            "valid_percent": 100.0 * int(stats["valid"]) / (source.width * source.height),
        }

    if log is not None:
        for line in _raster_summary_lines(
            str(path), info
        ):
            log(line)
        log(
            "Raster diagnostics: "
            f"valid {int(info['valid']):,} ({float(info['valid_percent']):.4g}%); "
            f"masked/NoData {int(info['masked']):,}; NaN {int(info['nan']):,}; "
            f"infinite {int(info['infinite']):,}; "
            f"negative/zero/positive {int(info['negative']):,}/"
            f"{int(info['zero']):,}/{int(info['positive']):,}; "
            f"mean {float(info['mean']):.6g} m; "
            f"median {float(info['median_sample']):.6g} m "
            f"(systematic sample of {int(info['median_sample_count']):,})."
        )
        valid_count = int(info["valid"])
        log(
            "Valid-value percentages (negative/zero/positive): "
            f"{100 * int(info['negative']) / valid_count:.3f}%/"
            f"{100 * int(info['zero']) / valid_count:.3f}%/"
            f"{100 * int(info['positive']) / valid_count:.3f}%."
        )
        if (
            int(info["integer_valued"]) == valid_count
            and float(info["minimum"]) >= 0
            and float(info["maximum"]) <= 255
        ):
            log(
                "WARNING: Every valid value is an integer in the 0-255 range. "
                "This is suspicious for continuous bathymetric-change data; "
                "confirm that the input is not a rendered or quantized image."
            )
        if not info["metadata_units"]:
            log(
                "Raster units are not recorded in metadata; using the explicit "
                "project setting: metres."
            )
    return info


def normalise_change(raw_difference: float, raster_value_convention: str) -> float:
    if raster_value_convention == "ELEVATION":
        return float(raw_difference)
    if raster_value_convention == "POSITIVE_DEPTH":
        return float(-raw_difference)
    raise ValueError(f"Unsupported raster value convention: {raster_value_convention}")


def validate_raster_compatibility(reference, candidate, label: str) -> None:
    if reference.crs != candidate.crs:
        raise ValueError(f"{label} rasters have mismatched CRS values: {reference.crs} vs {candidate.crs}.")
    if reference.transform != candidate.transform:
        raise ValueError(f"{label} rasters are not aligned on the same grid. Check cell size, origin and affine transform.")
    if reference.dtypes[0] != candidate.dtypes[0]:
        raise ValueError(f"{label} rasters have incompatible data types: {reference.dtypes[0]} vs {candidate.dtypes[0]}.")


def classify_change(value: float, tolerance: float) -> str:
    if np.isnan(value):
        return "NODATA_OR_EXCLUDED"
    if value > tolerance:
        return "DEEPENING"
    if value < -tolerance:
        return "ACCRETION"
    return "NO_SIGNIFICANT_CHANGE"


def summarise_change_distribution(values: pd.Series, areas: pd.Series, tolerance_m: float) -> dict:
    clean_values = np.asarray(values, dtype=float)
    clean_areas = np.asarray(areas, dtype=float)
    valid = np.isfinite(clean_values)
    if not np.any(valid):
        return {
            "valid_analysis_area_m2": 0.0,
            "excluded_area_m2": 0.0,
            "deepening_area_m2": 0.0,
            "accretion_area_m2": 0.0,
            "stable_area_m2": 0.0,
            "gross_deepening_volume_m3": 0.0,
            "gross_accretion_volume_m3": 0.0,
            "net_lowering_volume_m3": 0.0,
            "observed_mean_deepening_m": np.nan,
            "observed_median_deepening_m": np.nan,
            "observed_mean_accretion_m": np.nan,
            "observed_median_accretion_m": np.nan,
        }

    deep_mask = valid & (clean_values > tolerance_m)
    accretion_mask = valid & (clean_values < -tolerance_m)
    stable_mask = valid & ~(deep_mask | accretion_mask)

    deep_values = clean_values[deep_mask]
    accretion_values = np.abs(clean_values[accretion_mask])
    deep_area = float(clean_areas[deep_mask].sum())
    accretion_area = float(clean_areas[accretion_mask].sum())
    stable_area = float(clean_areas[stable_mask].sum())
    valid_area = float(clean_areas[valid].sum())

    deep_volume = float(np.sum(deep_values * clean_areas[deep_mask]))
    acc_volume = float(np.sum(accretion_values * clean_areas[accretion_mask]))
    net_volume = deep_volume - acc_volume

    return {
        "valid_analysis_area_m2": valid_area,
        "excluded_area_m2": max(0.0, float(clean_areas.sum()) - valid_area),
        "deepening_area_m2": deep_area,
        "accretion_area_m2": accretion_area,
        "stable_area_m2": stable_area,
        "gross_deepening_volume_m3": deep_volume,
        "gross_accretion_volume_m3": acc_volume,
        "net_lowering_volume_m3": net_volume,
        "observed_mean_deepening_m": float(np.mean(deep_values)) if deep_values.size else np.nan,
        "observed_median_deepening_m": float(np.median(deep_values)) if deep_values.size else np.nan,
        "observed_mean_accretion_m": float(np.mean(accretion_values)) if accretion_values.size else np.nan,
        "observed_median_accretion_m": float(np.median(accretion_values)) if accretion_values.size else np.nan,
    }


def build_change_warnings(settings: Settings, grid_calc: gpd.GeoDataFrame) -> list[str]:
    warnings = []
    if settings.change_tolerance_m == 0:
        warnings.append("Change tolerance is zero; cells on the threshold are treated as significant change.")
    if settings.analysis_extent == "FULL_OVERLAP":
        warnings.append("Cells without valid raster coverage are excluded from the analysis extent.")
    if settings.validation_mode == "DEEPENING_ONLY":
        warnings.append("Deepening-only mode excludes accretion from the primary comparison and may bias results when accretion is substantial.")
    if "ChangeClass" in grid_calc.columns and not grid_calc["ChangeClass"].eq("DEEPENING").any():
        warnings.append("No significant deepening cells were present in the analysis area.")
    if "CoveragePct" in grid_calc.columns and (grid_calc["CoveragePct"] < settings.minimum_coverage_percent).any():
        warnings.append("Some cells had coverage below the configured minimum; review the excluded or low-coverage areas before interpreting results.")
    if "NODATA_OR_EXCLUDED" in grid_calc.get("ChangeClass", pd.Series(dtype=str)).astype(str).values:
        warnings.append("NoData or excluded cells were present in the analysis extent and were omitted from the primary change metrics.")
    return warnings


def zonal_change(grid: gpd.GeoDataFrame, raster_path: str, sign: str, log: Callable[[str], None], raster_value_convention: str = "ELEVATION", change_tolerance_m: float = 0.05) -> pd.DataFrame:
    rows = []
    lowering_pixels_found = False
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Bathymetric difference raster has no CRS.")
        work = grid.to_crs(src.crs)
        pixel_area = raster_cell_area(src)
        nodata = src.nodata
        for pos, geom in enumerate(work.geometry, 1):
            vals = np.array([], dtype=float)
            if geom is not None and not geom.is_empty:
                try:
                    arr, _ = mask(src, [geom], crop=True, filled=False, all_touched=False)
                    band = arr[0]
                    vals = band.compressed() if np.ma.isMaskedArray(band) else band.ravel()
                    vals = vals[np.isfinite(vals)]
                    if nodata is not None:
                        vals = vals[vals != nodata]
                    convention_sign = (
                        1.0 if raster_value_convention == "ELEVATION" else -1.0
                    )
                    lowering_sign = (
                        1.0 if sign == "positive_is_lowering" else -1.0
                    )
                    lowering_pixels_found |= bool(
                        np.any(vals * convention_sign * lowering_sign > 0)
                    )
                except ValueError:
                    pass
            mean = float(np.mean(vals)) if vals.size else np.nan
            if np.isfinite(mean):
                mean = normalise_change(mean, raster_value_convention)
                if sign == "negative_is_lowering":
                    mean *= -1.0
            valid_area = min(float(geom.area), vals.size * pixel_area) if vals.size else 0.0
            polygon_area = float(geom.area) if geom is not None else 0.0
            change_class = classify_change(mean, change_tolerance_m) if np.isfinite(mean) else "NODATA_OR_EXCLUDED"
            rows.append((mean, valid_area, 100.0 * valid_area / polygon_area if polygon_area else 0.0, int(vals.size), change_class))
            if pos % 100 == 0:
                log(f"Calculated bathymetric statistics for {pos:,} grid cells...")
    if not lowering_pixels_found:
        log(
            "WARNING: No bathymetric-difference pixels matching the configured "
            "seabed-lowering sign were found in the analysis area."
        )
    return pd.DataFrame(rows, columns=["ObsLower_m", "ValidArea_m2", "CoveragePct", "RasterPixels", "ChangeClass"])


def estimate_background_lowering(
    grid: pd.DataFrame,
    settings: Settings,
    log: Callable[[str], None],
) -> dict[object, dict[str, object]]:
    """Median lowering of undredged cells, per licence where possible.

    Undredged cells have no EMS hours, valid bathymetry and adequate raster
    coverage. Their lowering is natural change (or a survey offset), which also
    happens between the narrow dredge furrows inside dredged cells, so it is
    subtracted from every cell.
    """
    licences = list(pd.unique(grid["LicenceNo"]))
    if settings.background_correction == "NONE":
        log("Background correction: off (all observed lowering is treated as dredging).")
        return {
            licence: {"value": 0.0, "cells": 0, "iqr": np.nan, "source": "none"}
            for licence in licences
        }
    lowering = pd.to_numeric(grid["ObsLower_m"], errors="coerce")
    controls = (
        pd.to_numeric(grid["EMS_Hours"], errors="coerce").fillna(0).eq(0)
        & np.isfinite(lowering)
        & pd.to_numeric(grid["CoveragePct"], errors="coerce").ge(
            settings.minimum_coverage_percent
        )
    )
    pooled = lowering[controls]
    minimum = settings.background_min_cells
    result: dict[object, dict[str, object]] = {}
    for licence in licences:
        own = lowering[controls & grid["LicenceNo"].eq(licence)]
        if len(own) >= minimum:
            values, source = own, "licence"
        elif len(pooled) >= minimum:
            values, source = pooled, "all licences"
        else:
            log(
                f"WARNING: licence {licence}: only {len(own)} undredged cells "
                f"({len(pooled)} across all licences); at least {minimum} are "
                "needed, so no background correction was applied."
            )
            result[licence] = {
                "value": 0.0, "cells": int(len(own)), "iqr": np.nan,
                "source": "insufficient",
            }
            continue
        median = float(values.median())
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        result[licence] = {
            "value": median, "cells": int(len(values)), "iqr": iqr, "source": source,
        }
        log(
            f"Background lowering for licence {licence}: {median:.3f} m "
            f"(median of {len(values)} undredged cells, {source}; IQR {iqr:.3f} m). "
            "Subtracted from every cell before volumes and rates."
        )
        if iqr > abs(median):
            log(
                f"WARNING: licence {licence}: undredged-cell lowering varies by "
                "more than its median, so the background correction is "
                "uncertain. Compare with the uncorrected and regression rates."
            )
    return result


def _regression_rate_m3ph(volumes: pd.Series, hours: pd.Series) -> float:
    """Slope of cell volume against EMS hours (m3/h), ignoring a constant offset."""
    x = pd.to_numeric(hours, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(volumes, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan")
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def status_for_difference(diff_pct: float, settings: Settings, mandatory_ok: bool) -> str:
    if not mandatory_ok or not np.isfinite(diff_pct):
        return "INSUFFICIENT_DATA"
    absolute = abs(diff_pct)
    if absolute <= settings.green_tolerance_percent:
        return "GREEN"
    if absolute <= settings.amber_tolerance_percent:
        return "AMBER"
    return "RED"


def _numeric_cell_value(value: object) -> tuple[str, float | None]:
    if value is None or value is pd.NA or value is pd.NaT:
        return "missing", None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return "missing", None
    if math.isnan(numeric):
        return "nan", numeric
    if math.isinf(numeric):
        return "infinity", numeric
    if numeric > 0:
        return "positive", numeric
    if numeric < 0:
        return "negative", numeric
    return "zero", numeric


def _numeric_distribution(values: pd.Series) -> dict[str, object]:
    categories = {
        "positive": 0,
        "zero": 0,
        "negative": 0,
        "nan": 0,
        "infinity": 0,
        "missing": 0,
    }
    finite_values = []
    for value in values:
        category, numeric = _numeric_cell_value(value)
        categories[category] += 1
        if category in {"positive", "zero", "negative"} and numeric is not None:
            finite_values.append(numeric)
    return {
        **categories,
        "finite": len(finite_values),
        "min": min(finite_values) if finite_values else None,
        "median": float(np.median(finite_values)) if finite_values else None,
        "mean": float(np.mean(finite_values)) if finite_values else None,
        "max": max(finite_values) if finite_values else None,
        "positive_min": min(
            (value for value in finite_values if value > 0), default=None
        ),
        "positive_median": (
            float(np.median([value for value in finite_values if value > 0]))
            if any(value > 0 for value in finite_values)
            else None
        ),
        "positive_mean": (
            float(np.mean([value for value in finite_values if value > 0]))
            if any(value > 0 for value in finite_values)
            else None
        ),
        "positive_max": max(
            (value for value in finite_values if value > 0), default=None
        ),
    }


def _diagnostic_number(value: object, precision: int = 6) -> str:
    category, numeric = _numeric_cell_value(value)
    if category == "missing":
        return "missing"
    if category == "nan":
        return "NaN"
    if category == "infinity":
        return "infinity" if numeric > 0 else "-infinity"
    return f"{numeric:.{precision}g}"


def _cell_rejection_reasons(
    row: Mapping[str, object],
    settings: Settings,
) -> list[str]:
    reasons = []
    if not bool(row.get("raster_valid", False)):
        reasons.append("NO_VALID_BATHYMETRY")
        return reasons

    _, ems_count = _numeric_cell_value(row.get("EMS_PointCount"))
    if ems_count is None or ems_count == 0:
        reasons.append("NO_EMS_RECORDS")

    duration_state, duration = _numeric_cell_value(row.get("EMS_Hours"))
    if duration_state in {"missing", "nan", "infinity"}:
        reasons.append("INVALID_DREDGING_DURATION")
    elif duration == 0:
        reasons.append("ZERO_DREDGING_DURATION")
    elif duration is not None and duration < 0:
        reasons.append("NEGATIVE_DREDGING_DURATION")
    if duration is not None and math.isfinite(duration) and (
        duration < settings.minimum_ems_hours
    ):
        reasons.append("DURATION_BELOW_MINIMUM")

    coverage_state, coverage = _numeric_cell_value(row.get("CoveragePct"))
    if coverage_state in {"missing", "nan", "infinity"}:
        reasons.append("INVALID_RASTER_COVERAGE")
    elif coverage is not None and coverage < settings.minimum_coverage_percent:
        reasons.append("COVERAGE_BELOW_MINIMUM")

    bathy_state, bathy = _numeric_cell_value(row.get("ObsLower_m"))
    if bathy_state in {"missing", "nan", "infinity"}:
        reasons.append("INVALID_BATHYMETRY")
    elif bathy == 0:
        reasons.append("ZERO_BATHY_CHANGE")
    elif bathy is not None and bathy < 0:
        reasons.append("NEGATIVE_BATHY_CHANGE")
    if bathy is not None and math.isfinite(bathy) and (
        bathy < settings.minimum_detectable_change_m
    ):
        reasons.append("BELOW_MINIMUM_DETECTABLE_CHANGE")

    volume_state, volume = _numeric_cell_value(row.get("NetVolume_m3"))
    if volume_state in {"missing", "nan", "infinity"}:
        reasons.append("INVALID_VOLUME")
    elif volume == 0:
        reasons.append("ZERO_VOLUME")
    elif volume is not None and volume < 0:
        reasons.append("NEGATIVE_VOLUME")

    rate_state, rate = _numeric_cell_value(row.get("ObsRate_tph"))
    if rate_state in {"missing", "nan", "infinity"}:
        reasons.append("INVALID_RATE")
    elif rate == 0:
        reasons.append("ZERO_RATE")
    elif rate is not None and rate < 0:
        reasons.append("NEGATIVE_RATE")
    if rate is not None and math.isfinite(rate) and rate > settings.maximum_rate_tph:
        reasons.append("RATE_ABOVE_MAXIMUM")

    if not reasons:
        reasons.append("VALID_POSITIVE_RATE")
    return reasons


def _build_validation_diagnostic_table(
    grid_with_raster_stats: gpd.GeoDataFrame,
    calculated_grid: gpd.GeoDataFrame,
    settings: Settings,
) -> pd.DataFrame:
    diagnostics = grid_with_raster_stats.drop(
        columns=["geometry"], errors="ignore"
    ).copy()
    diagnostics["raster_valid"] = (
        pd.to_numeric(diagnostics["RasterPixels"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    calculated_columns = (
        "RawObsLower_m",
        "Background_m",
        "ObsLower_m",
        "ChangeClass",
        "QA_Flags",
        "EMS_PointCount",
        "EMS_Min",
        "EMS_Hours",
        "NetVolume_m3",
        "CutVolume_m3",
        "FillVolume_m3",
        "ObsRate_m3ph",
        "BulkDensity",
        "ObsRate_tph",
        "ObsLowerFac",
        "IncludeFlag",
        "CalibrationClass",
        "QA_Reason",
    )
    for column in calculated_columns:
        if column not in diagnostics:
            diagnostics[column] = pd.Series(
                index=diagnostics.index,
                dtype=(
                    "object"
                    if column
                    in {"IncludeFlag", "CalibrationClass", "QA_Reason", "ChangeClass", "QA_Flags"}
                    else "float64"
                ),
            )
    for column in calculated_columns:
        if column in calculated_grid:
            diagnostics.loc[calculated_grid.index, column] = (
                calculated_grid[column]
            )

    reason_lists = [
        _cell_rejection_reasons(row, settings)
        for row in diagnostics.to_dict(orient="records")
    ]
    diagnostics["rejection_reasons"] = [
        ";".join(reasons) for reasons in reason_lists
    ]
    diagnostics["primary_rejection_reason"] = [
        reasons[0] for reasons in reason_lists
    ]
    rates = diagnostics["ObsRate_tph"].map(_numeric_cell_value)
    diagnostics["valid_positive_rate"] = [
        bool(include) and category == "positive"
        for include, (category, _) in zip(
            diagnostics["IncludeFlag"].astype("boolean").fillna(False), rates
        )
    ]
    return diagnostics


def _diagnose_ems_spatial_assignment(
    ems: gpd.GeoDataFrame,
    raster_candidate_grid: gpd.GeoDataFrame,
    unlicensed_grid: gpd.GeoDataFrame,
    deterministic_boundaries: bool = False,
) -> tuple[dict[str, int], pd.Series, pd.Series]:
    points = ems.copy().reset_index(drop=True)
    points["_ems_diagnostic_row"] = np.arange(len(points), dtype=np.int64)
    geometry = points.geometry
    valid_geometry = (
        geometry.notna()
        & ~geometry.is_empty
        & geometry.is_valid
        & geometry.geom_type.eq("Point")
    )
    try:
        valid_geometry &= np.isfinite(geometry.x) & np.isfinite(geometry.y)
    except (AttributeError, TypeError, ValueError):
        valid_geometry &= False
    valid_points = points.loc[valid_geometry].copy()
    invalid_geometry_count = len(points) - len(valid_points)

    joined = gpd.sjoin(
        valid_points,
        raster_candidate_grid[["GridID", "RasterPixels", "geometry"]],
        how="left",
        predicate="intersects" if deterministic_boundaries else "within",
    )
    if deterministic_boundaries:
        sort_columns = ["_ems_diagnostic_row"]
        for column in ("ChildRow", "ChildColumn", "GridID"):
            if column in joined:
                sort_columns.append(column)
        joined = (
            joined.sort_values(sort_columns, kind="stable")
            .drop_duplicates("_ems_diagnostic_row", keep="first")
        )
    has_grid_match = joined["index_right"].notna()
    valid_joined = joined.loc[
        has_grid_match
        & pd.to_numeric(joined["RasterPixels"], errors="coerce").fillna(0).gt(0)
    ]
    no_bathy_joined = joined.loc[
        has_grid_match
        & pd.to_numeric(joined["RasterPixels"], errors="coerce").fillna(0).le(0)
    ]
    valid_record_ids = set(valid_joined["_ems_diagnostic_row"].astype(int))
    no_bathy_record_ids = (
        set(no_bathy_joined["_ems_diagnostic_row"].astype(int))
        - valid_record_ids
    )
    all_valid_record_ids = set(valid_points["_ems_diagnostic_row"].astype(int))
    remaining_ids = all_valid_record_ids - valid_record_ids - no_bathy_record_ids

    no_licence_record_ids: set[int] = set()
    if remaining_ids and not unlicensed_grid.empty:
        remaining_points = valid_points.loc[
            valid_points["_ems_diagnostic_row"].isin(remaining_ids)
        ]
        unlicensed_join = gpd.sjoin(
            remaining_points,
            unlicensed_grid[["GridID", "geometry"]],
            how="left",
            predicate="intersects" if deterministic_boundaries else "within",
        )
        if deterministic_boundaries:
            unlicensed_join = (
                unlicensed_join.sort_values(
                    ["_ems_diagnostic_row", "GridID"], kind="stable"
                )
                .drop_duplicates("_ems_diagnostic_row", keep="first")
            )
        no_licence_record_ids = set(
            unlicensed_join.loc[
                unlicensed_join["index_right"].notna(),
                "_ems_diagnostic_row",
            ].astype(int)
        )
    outside_record_ids = (
        all_valid_record_ids
        - valid_record_ids
        - no_bathy_record_ids
        - no_licence_record_ids
    )

    cell_record_counts = valid_joined.groupby("GridID").size()
    all_grid_record_counts = joined.loc[has_grid_match].groupby("GridID").size()
    counts = {
        "total": len(ems),
        "valid_geometry": len(valid_points),
        "invalid_geometry": invalid_geometry_count,
        "valid_grid": len(valid_record_ids),
        "no_bathymetry": len(no_bathy_record_ids),
        "no_licence": len(no_licence_record_ids),
        "outside_grid": len(outside_record_ids),
    }
    return counts, cell_record_counts, all_grid_record_counts


def _ems_grid_counts(
    ems: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    deterministic_boundaries: bool,
) -> pd.Series:
    if not deterministic_boundaries:
        joined = gpd.sjoin(
            ems, grid[["GridID", "geometry"]], how="inner", predicate="within"
        )
        return joined.groupby("GridID").size()

    points = ems.copy().reset_index(drop=True)
    points["_ems_assignment_row"] = np.arange(len(points), dtype=np.int64)
    grid_columns = ["GridID", "geometry"]
    for column in ("ChildRow", "ChildColumn"):
        if column in grid:
            grid_columns.append(column)
    joined = gpd.sjoin(
        points,
        grid[grid_columns],
        how="inner",
        predicate="intersects",
    )
    sort_columns = ["_ems_assignment_row"]
    sort_columns.extend(
        column for column in ("ChildRow", "ChildColumn", "GridID") if column in joined
    )
    selected = (
        joined.sort_values(sort_columns, kind="stable")
        .drop_duplicates("_ems_assignment_row", keep="first")
    )
    return selected.groupby("GridID").size()


def _format_diagnostic_stat(value: object, units: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:,.6g}{(' ' + units) if units else ''}"


def _log_validation_diagnostics(
    grid_with_raster_stats: gpd.GeoDataFrame,
    calculated_grid: gpd.GeoDataFrame,
    diagnostic_table: pd.DataFrame,
    ems_record_count: int,
    assignment_counts: Mapping[str, int],
    ems_per_cell: pd.Series,
    settings: Settings,
    difference_source: str,
    raster_value_convention: str,
    difference_sign: str,
    log: Callable[[str], None],
) -> None:
    all_cells = len(diagnostic_table)
    valid_cells = int(diagnostic_table["raster_valid"].sum())
    no_raster_cells = all_cells - valid_cells
    log("250 m VALIDATION DIAGNOSTICS")
    log("GRID / RASTER")
    log(f"Grid cells in raster extent: {all_cells:,}")
    log(f"Grid cells with valid bathymetry: {valid_cells:,}")
    log(f"Grid cells without valid pixels: {no_raster_cells:,}")
    log("EMS")
    log(f"Area-candidate sampled EMS records: {ems_record_count:,}")
    log(
        "EMS records successfully assigned to a 250 m grid cell: "
        f"{assignment_counts.get('valid_grid', 0):,}"
    )
    log(
        "EMS records assigned to cells without valid bathymetry: "
        f"{assignment_counts.get('no_bathymetry', 0):,}"
    )
    log(
        "EMS records associated with grid cells lacking a licence assignment: "
        f"{assignment_counts.get('no_licence', 0):,}"
    )
    log(
        "EMS records outside retained/licensed grid cells: "
        f"{assignment_counts.get('outside_grid', 0):,}"
    )
    log(
        "EMS records with invalid point geometry/coordinates: "
        f"{assignment_counts.get('invalid_geometry', 0):,}"
    )
    assigned_cells = int((ems_per_cell > 0).sum())
    log(f"Grid cells receiving EMS records: {assigned_cells:,}")
    log(f"Grid cells receiving no EMS records: {valid_cells - assigned_cells:,}")
    positive_cell_counts = ems_per_cell[ems_per_cell > 0]
    if positive_cell_counts.empty:
        log("EMS records per receiving cell (min/median/mean/max): n/a")
    else:
        log(
            "EMS records per receiving cell (min/median/mean/max): "
            f"{positive_cell_counts.min():,.0f} / "
            f"{positive_cell_counts.median():,.1f} / "
            f"{positive_cell_counts.mean():,.2f} / "
            f"{positive_cell_counts.max():,.0f}"
        )

    valid_table = diagnostic_table[diagnostic_table["raster_valid"]]
    duration = _numeric_distribution(valid_table["EMS_Hours"])
    log("DREDGING ACTIVITY")
    log(
        f"Cells with valid calculated dredging duration: {duration['finite']:,}"
    )
    log(f"Cells with zero dredging duration: {duration['zero']:,}")
    log(
        "Cells with negative dredging duration: "
        f"{duration['negative']:,}"
    )
    log(
        "Cells with invalid/missing dredging duration: "
        f"{duration['nan'] + duration['infinity'] + duration['missing']:,}"
    )
    log(
        "Dredging duration formula/units: EMS record count x "
        f"{settings.sample_interval_minutes:g} minutes, converted to hours "
        "by dividing by 60; duration units are hours."
    )
    positive_duration = [
        float(value)
        for value in valid_table["EMS_Hours"]
        if _numeric_cell_value(value)[0] == "positive"
    ]
    if positive_duration:
        log(
            "Positive duration min/median/max: "
            f"{min(positive_duration):.6g} / "
            f"{float(np.median(positive_duration)):.6g} / "
            f"{max(positive_duration):.6g} hours"
        )
    else:
        log("Positive duration min/median/max: n/a")

    bathy = _numeric_distribution(valid_table["ObsLower_m"])
    log("BATHYMETRIC CHANGE")
    log(
        "Cells with positive bathymetric change: "
        f"{bathy['positive']:,}"
    )
    log(f"Cells with zero bathymetric change: {bathy['zero']:,}")
    log(
        "Cells with negative bathymetric change: "
        f"{bathy['negative']:,}"
    )
    log(
        "Cells with invalid/NaN bathymetric change: "
        f"{bathy['nan'] + bathy['infinity'] + bathy['missing']:,}"
    )
    log(
        "Finite ObsLower_m min/median/mean/max: "
        f"{_format_diagnostic_stat(bathy['min'], 'm')} / "
        f"{_format_diagnostic_stat(bathy['median'], 'm')} / "
        f"{_format_diagnostic_stat(bathy['mean'], 'm')} / "
        f"{_format_diagnostic_stat(bathy['max'], 'm')}"
    )
    if difference_source == "XYZ_SURVEYS":
        raster_formula = (
            "Earlier XYZ Z minus later XYZ Z (the XYZ loader normalizes "
            "all-nonnegative elevation values to negative elevations)."
        )
    elif difference_source == "PAIRED_RASTERS":
        raster_formula = "Earlier surface raster minus later surface raster."
    else:
        raster_formula = (
            "User-supplied difference raster; raw sign is interpreted using "
            "the configured raster convention and lowering sign."
        )
    log(f"Raster difference convention: {raster_formula}")
    log(
        "Configured raster value convention: "
        f"{raster_value_convention}; configured lowering sign: {difference_sign}."
    )
    log(
        "After normalise_change and the configured sign adjustment, "
        "positive ObsLower_m is expected to mean seabed lowering/extracted "
        "material; negative means accretion or the opposite signed change."
    )
    if (
        difference_source == "RASTER"
        and raster_value_convention == "POSITIVE_DEPTH"
        and difference_sign == "negative_is_lowering"
    ):
        log(
            "SIGN-CONVENTION WARNING: POSITIVE_DEPTH normalization negates "
            "the earlier-minus-later raster value, then negative_is_lowering "
            "negates it again. This configured combination makes a normalized "
            "negative result positive only after the second inversion; verify "
            "the source raster's documented convention. No sign was changed "
            "by diagnostics."
        )

    volume = _numeric_distribution(valid_table["NetVolume_m3"])
    log("CALCULATED DREDGING VOLUME")
    log(
        "Formula: NetVolume_m3 = ObsLower_m x ValidArea_m2; units are cubic metres."
    )
    log(f"Cells with calculable volume: {volume['finite']:,}")
    log(f"Positive volume: {volume['positive']:,}")
    log(f"Zero volume: {volume['zero']:,}")
    log(f"Negative volume: {volume['negative']:,}")
    log(
        "Invalid volume: "
        f"{volume['nan'] + volume['infinity'] + volume['missing']:,}"
    )
    log(
        "Positive volume min/median/mean/max: "
        f"{_format_diagnostic_stat(volume['positive_min'], 'm3')} / "
        f"{_format_diagnostic_stat(volume['positive_median'], 'm3')} / "
        f"{_format_diagnostic_stat(volume['positive_mean'], 'm3')} / "
        f"{_format_diagnostic_stat(volume['positive_max'], 'm3')}"
    )
    rate = _numeric_distribution(valid_table["ObsRate_tph"])
    log("CALCULATED VALIDATION RATE")
    log(
        "Formula: ObsRate_tph = "
        "(NetVolume_m3 / EMS_Hours) x BulkDensity; units are tonnes/hour."
    )
    log(
        f"Cells where rate was calculated: {len(valid_table):,}; "
        f"finite rates: {rate['finite']:,}; positive: {rate['positive']:,}; "
        f"zero: {rate['zero']:,}; negative: {rate['negative']:,}; "
        f"NaN: {rate['nan']:,}; infinity: {rate['infinity']:,}; "
        f"None/missing: {rate['missing']:,}."
    )
    log(
        "Cells excluded from the licence rate: "
        f"{int((~valid_table['IncludeFlag'].astype('boolean').fillna(False)).sum()):,}"
    )
    if getattr(settings, "rate_method", "CELL_FILTERS") == "POOLED":
        log(
            "Rate method POOLED: cells are excluded only for EMS_Hours < "
            f"{settings.minimum_ems_hours:g} h, CoveragePct < "
            f"{settings.minimum_coverage_percent:g}% or no valid bathymetry. "
            "The summary below lists per-cell QA flags; detection-threshold "
            "and per-cell rate flags do not remove cells from the licence rate."
        )
    log(
        "Per-cell criteria: EMS_Hours < "
        f"{settings.minimum_ems_hours:g} h; CoveragePct < "
        f"{settings.minimum_coverage_percent:g}%; ObsLower_m non-finite or < "
        f"{settings.minimum_detectable_change_m:g} m; ObsRate_tph non-finite "
        "or <= 0; ObsRate_tph > "
        f"{settings.maximum_rate_tph:g} tonnes/hour."
    )

    primary_counts = diagnostic_table["primary_rejection_reason"].value_counts()
    log("VALIDATION REJECTION SUMMARY (primary reason; totals reconcile to grid cells)")
    reason_labels = (
        ("NO_EMS_RECORDS", "No EMS records"),
        ("NO_VALID_BATHYMETRY", "No valid bathymetry"),
        ("ZERO_DREDGING_DURATION", "Zero duration"),
        ("INVALID_DREDGING_DURATION", "Invalid duration"),
        ("ZERO_BATHY_CHANGE", "Zero bathymetric change"),
        ("NEGATIVE_BATHY_CHANGE", "Negative bathymetric change"),
        ("INVALID_BATHYMETRY", "Invalid bathymetric change"),
        ("ZERO_VOLUME", "Zero volume"),
        ("NEGATIVE_VOLUME", "Negative volume"),
        ("INVALID_VOLUME", "Invalid volume"),
        ("ZERO_RATE", "Zero validation rate"),
        ("NEGATIVE_RATE", "Negative validation rate"),
        ("INVALID_RATE", "Invalid validation rate"),
        ("COVERAGE_BELOW_MINIMUM", "Below raster coverage threshold"),
        ("DURATION_BELOW_MINIMUM", "Below minimum EMS duration"),
        ("BELOW_MINIMUM_DETECTABLE_CHANGE", "Below detectable-change threshold"),
        ("RATE_ABOVE_MAXIMUM", "Above maximum rate"),
        ("VALID_POSITIVE_RATE", "Valid positive rate"),
    )
    for code, label in reason_labels:
        log(f"{label + ':':38s}{int(primary_counts.get(code, 0)):>8,}")
    log(
        "All applicable reasons (multi-reason cells counted in each applicable "
        "category) are in the diagnostic CSV rejection_reasons column."
    )
    if not int(valid_table["IncludeFlag"].astype("boolean").fillna(False).sum()):
        reasons = primary_counts.drop(
            labels=["VALID_POSITIVE_RATE"], errors="ignore"
        )
        if len(reasons):
            primary = str(reasons.idxmax())
            log(
                "Evidence-based result: no valid positive cell rates remain; "
                f"the most frequent primary rejection reason is {primary} "
                f"({int(reasons.max()):,} cells)."
            )
        else:
            log("Evidence-based result: no valid positive cell rates remain.")


def _resolution_diagnostic_metrics(
    diagnostics: pd.DataFrame,
    settings: Settings,
) -> dict[str, float | int]:
    raster_valid = diagnostics["raster_valid"].fillna(False).astype(bool)
    ems_count = pd.to_numeric(
        diagnostics["EMS_PointCount"], errors="coerce"
    ).fillna(0)
    duration = pd.to_numeric(
        diagnostics["EMS_Hours"], errors="coerce"
    )
    lowering = pd.to_numeric(
        diagnostics["ObsLower_m"], errors="coerce"
    )
    volume = pd.to_numeric(
        diagnostics["NetVolume_m3"], errors="coerce"
    )
    rate = pd.to_numeric(
        diagnostics["ObsRate_tph"], errors="coerce"
    )
    valid_positive_rate = diagnostics["valid_positive_rate"].fillna(False).astype(bool)
    valid_lowering = lowering[raster_valid & np.isfinite(lowering)]
    valid_volume = volume[raster_valid & np.isfinite(volume)]
    positive_rates = rate[raster_valid & np.isfinite(rate) & rate.gt(0)]
    ems_cells = raster_valid & ems_count.gt(0) & np.isfinite(duration)
    ems_durations = duration[ems_cells]

    metrics: dict[str, float | int] = {
        "grid_cells": len(diagnostics),
        "valid_raster_cells": int(raster_valid.sum()),
        "cells_receiving_ems": int((raster_valid & ems_count.gt(0)).sum()),
        "no_ems": int((raster_valid & ems_count.le(0)).sum()),
        "median_ems_hours": (
            float(ems_durations.median()) if not ems_durations.empty else math.nan
        ),
        "mean_ems_hours": (
            float(ems_durations.mean()) if not ems_durations.empty else math.nan
        ),
        "median_lowering_m": (
            float(valid_lowering.median()) if not valid_lowering.empty else math.nan
        ),
        "median_volume_m3": (
            float(valid_volume.median()) if not valid_volume.empty else math.nan
        ),
        "median_positive_rate_tph": (
            float(positive_rates.median())
            if not positive_rates.empty
            else math.nan
        ),
        "coverage_below_minimum": int(
            (raster_valid & pd.to_numeric(
                diagnostics["CoveragePct"], errors="coerce"
            ).lt(settings.minimum_coverage_percent)).sum()
        ),
        "duration_below_minimum": int(
            (raster_valid & duration.lt(settings.minimum_ems_hours)).sum()
        ),
        "lowering_below_minimum": int(
            (raster_valid & lowering.lt(settings.minimum_detectable_change_m)).sum()
        ),
        "rate_nonpositive": int(
            (raster_valid & np.isfinite(rate) & rate.le(0)).sum()
        ),
        "rate_above_maximum": int(
            (raster_valid & np.isfinite(rate) & rate.gt(settings.maximum_rate_tph)).sum()
        ),
        "valid_positive_rates": int(valid_positive_rate.sum()),
        "positive_rate_min_tph": (
            float(positive_rates.min()) if not positive_rates.empty else math.nan
        ),
        "positive_rate_p25_tph": (
            float(positive_rates.quantile(0.25))
            if not positive_rates.empty
            else math.nan
        ),
        "positive_rate_median_tph": (
            float(positive_rates.median()) if not positive_rates.empty else math.nan
        ),
        "positive_rate_p75_tph": (
            float(positive_rates.quantile(0.75))
            if not positive_rates.empty
            else math.nan
        ),
        "positive_rate_mean_tph": (
            float(positive_rates.mean()) if not positive_rates.empty else math.nan
        ),
        "positive_rate_p95_tph": (
            float(positive_rates.quantile(0.95))
            if not positive_rates.empty
            else math.nan
        ),
        "positive_rate_max_tph": (
            float(positive_rates.max()) if not positive_rates.empty else math.nan
        ),
    }

    duration_bins = (
        ("duration_lt_10_minutes", ems_durations.lt(10.0 / 60.0)),
        (
            "duration_10_to_30_minutes",
            ems_durations.ge(10.0 / 60.0) & ems_durations.lt(0.5),
        ),
        (
            "duration_30_to_60_minutes",
            ems_durations.ge(0.5) & ems_durations.lt(1.0),
        ),
        (
            "duration_1_to_2_hours",
            ems_durations.ge(1.0) & ems_durations.le(2.0),
        ),
        ("duration_gt_2_hours", ems_durations.gt(2.0)),
    )
    metrics.update(
        {name: int(mask.sum()) for name, mask in duration_bins}
    )

    rate_bins = (
        ("rate_0_to_500_tph", positive_rates.lt(500.0)),
        (
            "rate_500_to_1000_tph",
            positive_rates.ge(500.0) & positive_rates.lt(1000.0),
        ),
        (
            "rate_1000_to_1500_tph",
            positive_rates.ge(1000.0) & positive_rates.le(1500.0),
        ),
        (
            "rate_1500_to_2500_tph",
            positive_rates.gt(1500.0) & positive_rates.le(2500.0),
        ),
        (
            "rate_2500_to_5000_tph",
            positive_rates.gt(2500.0) & positive_rates.le(5000.0),
        ),
        ("rate_gt_5000_tph", positive_rates.gt(5000.0)),
    )
    metrics.update({name: int(mask.sum()) for name, mask in rate_bins})
    reason_codes = (
        "NO_EMS_RECORDS",
        "NO_VALID_BATHYMETRY",
        "INVALID_DREDGING_DURATION",
        "ZERO_DREDGING_DURATION",
        "DURATION_BELOW_MINIMUM",
        "INVALID_RASTER_COVERAGE",
        "COVERAGE_BELOW_MINIMUM",
        "INVALID_BATHYMETRY",
        "ZERO_BATHY_CHANGE",
        "NEGATIVE_BATHY_CHANGE",
        "BELOW_MINIMUM_DETECTABLE_CHANGE",
        "INVALID_VOLUME",
        "ZERO_VOLUME",
        "NEGATIVE_VOLUME",
        "INVALID_RATE",
        "ZERO_RATE",
        "NEGATIVE_RATE",
        "RATE_ABOVE_MAXIMUM",
    )
    reason_rows = diagnostics["rejection_reasons"].fillna("").astype(str).str.split(";")
    metrics.update(
        {
            f"rejection_{code}": int(
                reason_rows.map(lambda reasons: code in reasons).sum()
            )
            for code in reason_codes
        }
    )
    return metrics


def _aggregate_50m_parent_cells(
    diagnostics: pd.DataFrame,
    bulk_density: float,
) -> pd.DataFrame:
    if "Parent250_ID" not in diagnostics:
        raise ValueError("50 m diagnostics do not contain Parent250_ID values.")
    rows = []
    for parent_id, children in diagnostics.groupby("Parent250_ID", dropna=False):
        if pd.isna(parent_id):
            continue
        raster_valid = children["raster_valid"].fillna(False).astype(bool)
        ems_count = pd.to_numeric(
            children["EMS_PointCount"], errors="coerce"
        ).fillna(0)
        hours = pd.to_numeric(children["EMS_Hours"], errors="coerce").fillna(0)
        volume = pd.to_numeric(children["NetVolume_m3"], errors="coerce")
        rate_children = (
            raster_valid
            & ems_count.gt(0)
            & hours.gt(0)
            & np.isfinite(hours)
            & np.isfinite(volume)
        )
        total_hours = float(hours[raster_valid & np.isfinite(hours)].sum())
        corresponding_hours = float(hours[rate_children].sum())
        corresponding_volume = float(volume[rate_children].sum())
        combined_m3ph = (
            corresponding_volume / corresponding_hours
            if corresponding_hours > 0
            else math.nan
        )
        valid_area = pd.to_numeric(
            children["ValidArea_m2"], errors="coerce"
        ).fillna(0)
        rows.append(
            {
                "Parent250_ID": parent_id,
                "Total50mChildren": len(children),
                "ValidRasterChildren": int(raster_valid.sum()),
                "EMSReceivingChildren": int((raster_valid & ems_count.gt(0)).sum()),
                "ValidChildArea_m2": float(valid_area[raster_valid].sum()),
                "TotalEMS_Hours": total_hours,
                "TotalValidChildVolume_m3": float(
                    volume[raster_valid & np.isfinite(volume)].sum()
                ),
                "RateContributing50mChildren": int(rate_children.sum()),
                "RateContributingEMS_Hours": corresponding_hours,
                "RateContributingVolume_m3": corresponding_volume,
                "ExperimentalCombinedRate_m3ph": combined_m3ph,
                "ExperimentalCombinedRate_tph": combined_m3ph * bulk_density,
                "AggregationMethod": (
                    "Sum net volume across valid raster-covered EMS-bearing "
                    "children divided by their summed EMS hours; not an "
                    "average of child rates; diagnostic only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_resolution_comparison(
    output_folder: Path,
    settings: Settings,
    diagnostics_250m: pd.DataFrame,
    diagnostics_50m: pd.DataFrame,
    log: Callable[[str], None],
) -> None:
    metrics_250 = _resolution_diagnostic_metrics(diagnostics_250m, settings)
    metrics_50 = _resolution_diagnostic_metrics(diagnostics_50m, settings)
    comparison = pd.DataFrame(
        [
            {"Metric": metric, "250m": metrics_250[metric], "50m": metrics_50[metric]}
            for metric in metrics_250
        ]
    )
    comparison_path = output_folder / "validation_50m_vs_250m_summary.csv"
    comparison.to_csv(comparison_path, index=False)

    parent_aggregation = _aggregate_50m_parent_cells(
        diagnostics_50m, settings.bulk_density
    )
    aggregation_path = output_folder / "validation_50m_parent_250m_aggregation.csv"
    parent_aggregation.to_csv(aggregation_path, index=False)

    metric_labels = (
        ("grid_cells", "Grid cells considered"),
        ("valid_raster_cells", "Valid raster cells"),
        ("cells_receiving_ems", "Cells receiving EMS"),
        ("median_ems_hours", "Median EMS hours"),
        ("mean_ems_hours", "Mean EMS hours"),
        ("median_lowering_m", "Median lowering (m)"),
        ("median_volume_m3", "Median volume (m3)"),
        ("median_positive_rate_tph", "Median positive rate (t/h)"),
        ("rate_above_maximum", "Rate above maximum"),
        ("duration_below_minimum", "Duration below minimum"),
        ("lowering_below_minimum", "Lowering below minimum"),
        ("coverage_below_minimum", "Coverage below minimum"),
        ("valid_positive_rates", "Valid positive cells"),
        ("rejection_NO_EMS_RECORDS", "NO_EMS_RECORDS"),
        ("rejection_COVERAGE_BELOW_MINIMUM", "COVERAGE_BELOW_MINIMUM"),
        ("rejection_DURATION_BELOW_MINIMUM", "DURATION_BELOW_MINIMUM"),
        (
            "rejection_BELOW_MINIMUM_DETECTABLE_CHANGE",
            "BELOW_MINIMUM_DETECTABLE_CHANGE",
        ),
        ("rejection_RATE_ABOVE_MAXIMUM", "RATE_ABOVE_MAXIMUM"),
    )
    log("VALIDATION RESOLUTION COMPARISON")
    log(f"{'':34s}{'250 m':>14s}{'50 m':>14s}")
    for key, label in metric_labels:
        left, right = metrics_250[key], metrics_50[key]
        if isinstance(left, float) and not math.isfinite(left):
            left = "n/a"
        elif isinstance(left, float):
            left = f"{left:,.4g}"
        if isinstance(right, float) and not math.isfinite(right):
            right = "n/a"
        elif isinstance(right, float):
            right = f"{right:,.4g}"
        log(f"{label + ':':34s}{str(left):>14s}{str(right):>14s}")
    contributing_children = int(
        parent_aggregation["RateContributing50mChildren"].sum()
    ) if not parent_aggregation.empty else 0
    log(
        "50 m child cells contributing to valid parent 250 m rates: "
        f"{contributing_children:,}."
    )
    log(
        "50 m duration bands (cells with EMS; <10 min, 10-30 min, "
        "30-60 min, 1-2 h, >2 h): "
        f"{metrics_50['duration_lt_10_minutes']:,}, "
        f"{metrics_50['duration_10_to_30_minutes']:,}, "
        f"{metrics_50['duration_30_to_60_minutes']:,}, "
        f"{metrics_50['duration_1_to_2_hours']:,}, "
        f"{metrics_50['duration_gt_2_hours']:,}."
    )
    log(
        "Positive rate bands before maximum-rate rejection (0-500, "
        "500-1,000, 1,000-1,500, 1,500-2,500, 2,500-5,000, >5,000 t/h): "
        f"{metrics_50['rate_0_to_500_tph']:,}, "
        f"{metrics_50['rate_500_to_1000_tph']:,}, "
        f"{metrics_50['rate_1000_to_1500_tph']:,}, "
        f"{metrics_50['rate_1500_to_2500_tph']:,}, "
        f"{metrics_50['rate_2500_to_5000_tph']:,}, "
        f"{metrics_50['rate_gt_5000_tph']:,}."
    )
    log(
        "50 m rates remain experimental; if they remain implausibly high, "
        "investigate EMS track-footprint volume attribution rather than "
        "reducing grid size further."
    )
    log(f"Resolution comparison written to: {comparison_path}")
    log(f"Experimental parent aggregation written to: {aggregation_path}")


def _run_validation_single(
    settings: Settings,
    callback: Optional[Callable[[str], None]] = None,
    *,
    grid_override: gpd.GeoDataFrame | None = None,
    licences_override: gpd.GeoDataFrame | None = None,
    ems_override: gpd.GeoDataFrame | None = None,
    difference_raster_override: str | None = None,
    raster_validation: dict[str, object] | None = None,
    grid_generation_seconds: float = 0.0,
    output_suffix: str = "",
    resolution_label: str = RESOLUTION_250,
):
    settings.validate()
    messages = []
    validation_started = time.perf_counter()

    def log(text):
        messages.append(text)
        notify(callback, text)

    out = Path(settings.output_folder)
    log(f"{APP_NAME} v{VERSION}")
    log(f"Validation run started: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    difference_raster = difference_raster_override or settings.difference_raster
    difference_sign = settings.source_sign
    raster_value_convention = settings.raster_value_convention
    if difference_raster_override is not None and settings.difference_source in {"XYZ_SURVEYS", "PAIRED_RASTERS"}:
        difference_sign = "positive_is_lowering"
        if settings.difference_source == "XYZ_SURVEYS":
            raster_value_convention = "ELEVATION"
            log("Using the shared generated XYZ difference raster (positive means lowering).")
        else:
            raster_value_convention = settings.raster_value_convention
            log("Using the shared difference raster generated from the paired surface rasters.")
    elif difference_raster_override is None and settings.difference_source == "XYZ_SURVEYS":
        difference_raster = prepare_difference_raster(settings, out, log)
        difference_sign = "positive_is_lowering"
        raster_value_convention = "ELEVATION"
        log("Using generated XYZ difference values (positive means lowering).")
    elif difference_raster_override is None and settings.difference_source == "PAIRED_RASTERS":
        difference_raster = prepare_difference_raster(settings, out, log)
        difference_sign = "positive_is_lowering"
        raster_value_convention = settings.raster_value_convention
        log("Using the difference generated from the paired surface rasters.")
    if raster_validation is None:
        log("Validating bathymetric-difference raster before EMS processing...")
        raster_validation = validate_bathymetric_difference_raster(
            difference_raster,
            settings.difference_units,
            difference_sign,
            raster_value_convention,
            log,
        )
    out.mkdir(parents=True, exist_ok=True)
    log("Loading calibration grid and licence polygons...")

    grid = (
        grid_override.copy()
        if grid_override is not None
        else read_vector(settings.calibration_grid, settings.calibration_grid_layer)
    )
    licences = (
        licences_override.copy()
        if licences_override is not None
        else read_vector(settings.licence_layer, settings.licence_layer_name)
    )
    grid_id = field_lookup(grid.columns, settings.grid_id_field)
    licence_field = field_lookup(licences.columns, settings.licence_field)
    grid = grid.rename(columns={grid_id: "GridID"})
    licences = licences.rename(columns={licence_field: "LicenceNo"})

    projected = CRS.from_user_input(settings.projected_crs)
    if not projected.is_projected:
        raise ValueError("Calculation CRS must be projected so areas are measured in square metres.")
    grid_calc = grid.to_crs(projected)
    licences_calc = licences.to_crs(projected)[["LicenceNo", "geometry"]]
    if grid_calc["GridID"].isna().any() or grid_calc["GridID"].astype(str).duplicated().any():
        raise ValueError("Grid ID values must be populated and unique.")

    reps = grid_calc[["GridID", "geometry"]].copy()
    reps.geometry = reps.geometry.representative_point()
    assigned = gpd.sjoin(reps, licences_calc, how="left", predicate="within")
    licence_map = assigned.drop_duplicates("GridID").set_index("GridID")["LicenceNo"]
    grid_calc["LicenceNo"] = grid_calc["GridID"].map(licence_map)
    unlicensed_grid = grid_calc.loc[
        grid_calc["LicenceNo"].isna(), ["GridID", "geometry"]
    ].copy()
    grid_calc = grid_calc[grid_calc["LicenceNo"].notna()].copy()
    if grid_calc.empty:
        raise ValueError("No calibration grid cells were assigned to a licence polygon.")

    if settings.analysis_extent == "FULL_OVERLAP":
        with rasterio.open(difference_raster) as source:
            if source.crs is None:
                raise ValueError("Bathymetric difference raster has no CRS.")
            raster_bounds = box(*source.bounds)
            grid_in_raster_crs = grid_calc.to_crs(source.crs)
            bounds_candidates = grid_in_raster_crs.geometry.intersects(raster_bounds)
        grid_calc = grid_calc.loc[bounds_candidates].copy()
        if grid_calc.empty:
            raise ValueError(
                "No licence-assigned calibration cells overlap the bathymetric raster extent."
            )
        log(
            f"Restricted full-overlap analysis to {len(grid_calc):,} grid cells "
            "within the raster extent; checking valid-data coverage..."
        )

    log(f"Prepared {len(grid_calc):,} grid cells across {grid_calc['LicenceNo'].nunique():,} licences.")
    log("Calculating bathymetric change statistics...")
    raster_stats_started = time.perf_counter()
    stats = zonal_change(
        grid_calc,
        difference_raster,
        difference_sign,
        log,
        raster_value_convention=raster_value_convention,
        change_tolerance_m=settings.change_tolerance_m,
    )
    raster_stats_seconds = time.perf_counter() - raster_stats_started
    for col in stats.columns:
        grid_calc[col] = stats[col].to_numpy()
    grid_with_raster_stats = grid_calc.copy()
    if settings.analysis_extent == "FULL_OVERLAP":
        covered = pd.to_numeric(grid_calc["RasterPixels"], errors="coerce").fillna(0) > 0
        excluded_count = int((~covered).sum())
        grid_calc = grid_calc.loc[covered].copy()
        if grid_calc.empty:
            raise ValueError(
                "No calibration grid cells contain valid bathymetric raster data. "
                "Check the survey overlap, CRS selections and raster NoData coverage."
            )
        log(
            f"Retained {len(grid_calc):,} grid cells with valid raster coverage; "
            f"excluded {excluded_count:,} cells without valid pixels."
        )
    grid_calc["NetVolume_m3"] = grid_calc["ObsLower_m"] * grid_calc["ValidArea_m2"]
    grid_calc["CutVolume_m3"] = np.where(grid_calc["NetVolume_m3"] > 0, grid_calc["NetVolume_m3"], 0.0)
    grid_calc["FillVolume_m3"] = np.where(grid_calc["NetVolume_m3"] < 0, -grid_calc["NetVolume_m3"], 0.0)

    log("Loading and spatially assigning EMS records...")
    ems_assignment_started = time.perf_counter()
    ems = (
        ems_override.copy()
        if ems_override is not None
        else load_ems(settings, log, grid_calc)
    ).to_crs(projected)
    assignment_counts, assigned_cell_counts, all_candidate_cell_counts = (
        _diagnose_ems_spatial_assignment(
            ems,
            grid_with_raster_stats,
            unlicensed_grid,
            deterministic_boundaries=resolution_label == RESOLUTION_50,
        )
    )
    counts = _ems_grid_counts(
        ems,
        grid_calc,
        deterministic_boundaries=resolution_label == RESOLUTION_50,
    )
    ems_assignment_seconds = time.perf_counter() - ems_assignment_started
    grid_calc["EMS_PointCount"] = grid_calc["GridID"].map(counts).fillna(0).astype(int)
    grid_calc["EMS_Min"] = grid_calc["EMS_PointCount"] * settings.sample_interval_minutes
    grid_calc["EMS_Hours"] = grid_calc["EMS_Min"] / 60.0

    background = estimate_background_lowering(grid_calc, settings, log)
    grid_calc["RawObsLower_m"] = grid_calc["ObsLower_m"]
    grid_calc["Background_m"] = grid_calc["LicenceNo"].map(
        lambda licence: background[licence]["value"]
    ).astype(float)
    grid_calc["ObsLower_m"] = grid_calc["RawObsLower_m"] - grid_calc["Background_m"]
    grid_calc["ChangeClass"] = [
        classify_change(value, settings.change_tolerance_m)
        if np.isfinite(value)
        else "NODATA_OR_EXCLUDED"
        for value in grid_calc["ObsLower_m"]
    ]
    grid_calc["NetVolume_m3"] = grid_calc["ObsLower_m"] * grid_calc["ValidArea_m2"]
    grid_calc["CutVolume_m3"] = np.where(grid_calc["NetVolume_m3"] > 0, grid_calc["NetVolume_m3"], 0.0)
    grid_calc["FillVolume_m3"] = np.where(grid_calc["NetVolume_m3"] < 0, -grid_calc["NetVolume_m3"], 0.0)

    validation_calculation_started = time.perf_counter()
    with np.errstate(divide="ignore", invalid="ignore"):
        grid_calc["ObsRate_m3ph"] = grid_calc["NetVolume_m3"] / grid_calc["EMS_Hours"]
        grid_calc["BulkDensity"] = settings.bulk_density
        grid_calc["ObsRate_tph"] = grid_calc["ObsRate_m3ph"] * settings.bulk_density
        grid_calc["ObsLowerFac"] = grid_calc["ObsLower_m"] / grid_calc["EMS_Hours"]

    reasons = []
    classes = []
    include = []
    flags = []
    pooled = settings.rate_method == "POOLED"
    for row in grid_calc.itertuples():
        cell_reasons = []
        cell_flags = []
        if row.EMS_Hours < settings.minimum_ems_hours:
            cell_reasons.append("EMS time below minimum")
        if row.CoveragePct < settings.minimum_coverage_percent:
            cell_reasons.append("bathymetric coverage below minimum")
        # In POOLED mode these are per-cell QA flags only: the licence rate is
        # total volume / total hours, so low-change and high-rate cells stay in.
        cell_checks = cell_flags if pooled else cell_reasons
        if not np.isfinite(row.ObsLower_m):
            cell_reasons.append("no valid bathymetric change")
        elif row.ObsLower_m < settings.minimum_detectable_change_m:
            cell_checks.append("lowering below detection threshold or accretion")
        if not np.isfinite(row.ObsRate_tph) or row.ObsRate_tph <= 0:
            cell_checks.append("non-positive or invalid production rate")
        elif row.ObsRate_tph > settings.maximum_rate_tph:
            cell_checks.append("production rate exceeds maximum")
        flags.append("; ".join(cell_flags))
        accepted = not cell_reasons
        include.append(accepted)
        if accepted:
            classes.append("A_ACCEPTED")
            reasons.append("")
        elif row.EMS_Hours > 0 and row.CoveragePct > 0 and np.isfinite(row.ObsLower_m):
            classes.append("B_REVIEW")
            reasons.append("; ".join(cell_reasons))
        else:
            classes.append("C_EXCLUDED")
            reasons.append("; ".join(cell_reasons))
    grid_calc["IncludeFlag"] = include
    grid_calc["CalibrationClass"] = classes
    grid_calc["QA_Reason"] = reasons
    grid_calc["QA_Flags"] = flags
    diagnostic_table = _build_validation_diagnostic_table(
        grid_with_raster_stats,
        grid_calc,
        settings,
    )
    diagnostic_table["EMS_PointCount"] = (
        diagnostic_table["GridID"].map(all_candidate_cell_counts).fillna(0).astype(int)
    )
    # Preserve the production counts for cells that enter the normal rate calculation.
    diagnostic_table.loc[grid_calc.index, "EMS_PointCount"] = grid_calc[
        "EMS_PointCount"
    ]
    diagnostic_table["EMS_Min"] = (
        diagnostic_table["EMS_PointCount"] * settings.sample_interval_minutes
    )
    diagnostic_table["EMS_Hours"] = diagnostic_table["EMS_Min"] / 60.0
    diagnostic_table["rejection_reasons"] = [
        ";".join(_cell_rejection_reasons(row, settings))
        for row in diagnostic_table.to_dict(orient="records")
    ]
    diagnostic_table["primary_rejection_reason"] = [
        reasons.split(";", 1)[0]
        for reasons in diagnostic_table["rejection_reasons"]
    ]
    diagnostic_table["valid_positive_rate"] = (
        diagnostic_table["IncludeFlag"].astype("boolean").fillna(False)
        & pd.to_numeric(diagnostic_table["ObsRate_tph"], errors="coerce").gt(0)
        & np.isfinite(
            pd.to_numeric(diagnostic_table["ObsRate_tph"], errors="coerce")
        )
    )
    validation_calculation_seconds = (
        time.perf_counter() - validation_calculation_started
    )
    valid_cell_counts = grid_calc.set_index("GridID")["EMS_PointCount"]
    ems_per_cell = valid_cell_counts[valid_cell_counts > 0]
    if resolution_label == RESOLUTION_250:
        diagnostic_name = "validation_250m_diagnostics.csv" if not output_suffix else "validation_250m_regulatory_diagnostics.csv"
    else:
        diagnostic_name = "validation_50m_diagnostics.csv"
    diagnostic_table_path = out / diagnostic_name
    output_started = time.perf_counter()
    diagnostic_table.to_csv(diagnostic_table_path, index=False)
    log(
        "250 m diagnostic units: grid coordinates/metres; cell and valid "
        "areas/square metres; ObsLower_m/metres; NetVolume_m3/cubic metres; "
        "dredging duration/hours; ObsRate_tph/tonnes per hour; "
        "ObsRate_m3ph/cubic metres per hour."
    )
    log(
        "EMS spatial assignment: "
        f"{assignment_counts['valid_grid']:,} assigned to valid raster cells; "
        f"{assignment_counts['no_bathymetry']:,} assigned to cells without "
        f"valid bathymetry; {assignment_counts['no_licence']:,} assigned to "
        f"unlicensed cells; {assignment_counts['outside_grid']:,} outside "
        f"retained/licensed cells; {assignment_counts['invalid_geometry']:,} "
        "invalid point geometries."
    )
    _log_validation_diagnostics(
        grid_with_raster_stats,
        grid_calc,
        diagnostic_table,
        len(ems),
        assignment_counts,
        ems_per_cell,
        settings,
        settings.difference_source,
        raster_value_convention,
        difference_sign,
        log,
    )
    failed_sample = diagnostic_table.loc[
        ~diagnostic_table["valid_positive_rate"]
    ].head(10)
    if not failed_sample.empty:
        log("Representative failed 250 m cells (first 10):")
        for row in failed_sample.itertuples(index=False):
            log(
                f"Cell: {getattr(row, 'GridID', 'missing')}; "
                f"Licence: {getattr(row, 'LicenceNo', 'missing')}; "
                f"EMS samples: {_diagnostic_number(row.EMS_PointCount, 0)}; "
                f"Duration: {_diagnostic_number(row.EMS_Hours)} hours; "
                f"Bathy change: {_diagnostic_number(row.ObsLower_m)} m; "
                f"Volume: {_diagnostic_number(row.NetVolume_m3)} m3; "
                f"Rate: {_diagnostic_number(row.ObsRate_tph)} t/h; "
                f"Rejected because: {row.primary_rejection_reason}"
            )
    log(
        f"Diagnostic table written to: {diagnostic_table_path}"
    )
    summaries = []
    for licence, grp in grid_calc.groupby("LicenceNo", dropna=False):
        accepted = grp[grp["IncludeFlag"]].copy()
        total_hours = accepted["EMS_Hours"].sum()
        total_volume = accepted["NetVolume_m3"].sum()
        rate_m3ph = total_volume / total_hours if total_hours > 0 else np.nan
        rate_tph = rate_m3ph * settings.bulk_density if np.isfinite(rate_m3ph) else np.nan
        diff_pct = ((rate_tph - settings.configured_rate_tph) / settings.configured_rate_tph * 100.0
                    if np.isfinite(rate_tph) and settings.configured_rate_tph > 0 else np.nan)
        mandatory_ok = len(accepted) > 0 and total_hours > 0
        area_status = status_for_difference(diff_pct, settings, mandatory_ok)
        ems_cells = grp[
            (grp["EMS_Hours"] > 0) & np.isfinite(grp["ObsLower_m"])
        ]
        all_ems_hours = float(ems_cells["EMS_Hours"].sum())
        all_ems_volume = float(ems_cells["NetVolume_m3"].sum())
        all_ems_rate_tph = (
            all_ems_volume / all_ems_hours * settings.bulk_density
            if all_ems_hours > 0
            else np.nan
        )
        excluded_hours_pct = (
            (all_ems_hours - float(total_hours)) / all_ems_hours * 100.0
            if all_ems_hours > 0
            else np.nan
        )
        if np.isfinite(excluded_hours_pct) and excluded_hours_pct > 25.0:
            excluded = ems_cells.loc[~ems_cells["IncludeFlag"].astype(bool)]
            low_coverage = excluded["CoveragePct"] < settings.minimum_coverage_percent
            coverage_hours = float(excluded.loc[low_coverage, "EMS_Hours"].sum())
            other_hours = float(excluded.loc[~low_coverage, "EMS_Hours"].sum())
            causes = []
            if coverage_hours > 0:
                causes.append(
                    f"{coverage_hours:,.1f} h in cells the difference raster covers "
                    f"less than {settings.minimum_coverage_percent:g}%"
                )
            if other_hours > 0:
                causes.append(
                    f"{other_hours:,.1f} h in cells failing other checks "
                    "(see QA_Reason)"
                )
            if total_hours <= 0 and coverage_hours >= 0.5 * all_ems_hours:
                log(
                    f"WARNING: licence {licence}: no rate - the difference raster "
                    f"covers too little of the dredged cells "
                    f"({int(low_coverage.sum())} cell(s) below "
                    f"{settings.minimum_coverage_percent:g}% coverage, "
                    f"{coverage_hours:,.1f} EMS hours). This survey does not "
                    "measure the dredged area of this licence."
                )
            else:
                log(
                    f"WARNING: licence {licence}: {excluded_hours_pct:.1f}% of EMS "
                    f"hours were excluded: {'; '.join(causes)}. Validated rate "
                    f"{rate_tph:.1f} t/h; rate using every cell with EMS time and "
                    f"valid bathymetry {all_ems_rate_tph:.1f} t/h."
                    + (
                        " Dropping low-change cells with their hours can raise "
                        "the validated rate."
                        if settings.rate_method == "CELL_FILTERS"
                        else ""
                    )
                )
        weighted_median = np.nan
        if len(accepted):
            ordered = accepted.sort_values("ObsRate_tph")
            cum = ordered["EMS_Hours"].cumsum()
            if total_hours > 0:
                weighted_median = float(ordered.loc[cum >= total_hours / 2, "ObsRate_tph"].iloc[0])
        summaries.append({
            "LicenceNo": str(licence),
            "ConfiguredRate_tph": settings.configured_rate_tph,
            "ValidatedRate_tph": rate_tph,
            "ValidatedRate_m3ph": rate_m3ph,
            "DifferencePct": diff_pct,
            "MedianCellRate_tph": float(accepted["ObsRate_tph"].median()) if len(accepted) else np.nan,
            "WeightedMedianRate_tph": weighted_median,
            "LowerQuartile_tph": float(accepted["ObsRate_tph"].quantile(.25)) if len(accepted) else np.nan,
            "UpperQuartile_tph": float(accepted["ObsRate_tph"].quantile(.75)) if len(accepted) else np.nan,
            "AcceptedCells": int(len(accepted)),
            "ReviewCells": int((grp["CalibrationClass"] == "B_REVIEW").sum()),
            "ExcludedCells": int((grp["CalibrationClass"] == "C_EXCLUDED").sum()),
            "AcceptedEMS_Hours": float(total_hours),
            "AcceptedVolume_m3": float(total_volume),
            "MeanCoveragePct": float(accepted["CoveragePct"].mean()) if len(accepted) else 0.0,
            "QA_Status": area_status,
            "RateMethod": settings.rate_method,
            "Background_m": background[licence]["value"],
            "BackgroundCells": background[licence]["cells"],
            "BackgroundIQR_m": background[licence]["iqr"],
            "BackgroundSource": background[licence]["source"],
            "UncorrectedRate_tph": (
                float((accepted["RawObsLower_m"] * accepted["ValidArea_m2"]).sum())
                / total_hours * settings.bulk_density
                if total_hours > 0 else np.nan
            ),
            "RegressionRate_tph": _regression_rate_m3ph(
                accepted["RawObsLower_m"] * accepted["ValidArea_m2"],
                accepted["EMS_Hours"],
            ) * settings.bulk_density,
            "AllEMSCellsRate_tph": all_ems_rate_tph,
            "AllEMSCellsHours": all_ems_hours,
            "ExcludedEMSHoursPct": excluded_hours_pct,
        })

        volume_rate = rate_m3ph
        pred = grp["EMS_Hours"] * volume_rate if np.isfinite(volume_rate) else np.nan
        grid_calc.loc[grp.index, "PredVolume_m3"] = pred
        grid_calc.loc[grp.index, "PredLower_m"] = pred / grp["ValidArea_m2"].replace(0, np.nan)
        residual = grp["ObsLower_m"] - grid_calc.loc[grp.index, "PredLower_m"]
        grid_calc.loc[grp.index, "Residual_m"] = residual
        grid_calc.loc[grp.index, "ResidualPct"] = residual / grp["ObsLower_m"].replace(0, np.nan) * 100.0
        grid_calc.loc[grp.index, "QA_Status"] = area_status

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        raise ValueError("No licence summaries could be generated from the calibration grid.")

    gpkg = out / f"lowering_factor_validation_grid{output_suffix}.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    output_crs = grid.crs
    all_cells = grid_calc.to_crs(output_crs)
    all_cells.to_file(gpkg, layer="calibration_grid_all", driver="GPKG")
    for class_name, layer in [("A_ACCEPTED", "calibration_grid_accepted"), ("B_REVIEW", "calibration_grid_review"), ("C_EXCLUDED", "calibration_grid_excluded")]:
        subset = all_cells[all_cells["CalibrationClass"] == class_name]
        if not subset.empty:
            subset.to_file(gpkg, layer=layer, driver="GPKG")
    licence_summary = licences.copy()
    licence_summary["LicenceNo"] = licence_summary["LicenceNo"].astype(str)
    licence_summary = licence_summary.merge(summary_df, on="LicenceNo", how="left")
    licence_summary.to_file(gpkg, layer="licence_summary", driver="GPKG")

    summary_path = out / f"lowering_factor_validation_summary{output_suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    profiles = []
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    for item in summaries:
        profiles.append({
            "schema_version": "1.0",
            "licence_no": item["LicenceNo"],
            "survey_start": settings.survey_start,
            "survey_end": settings.survey_end,
            "bulk_density_t_m3": settings.bulk_density,
            "validated_rate_t_h": None if not np.isfinite(item["ValidatedRate_tph"]) else round(item["ValidatedRate_tph"], 6),
            "validated_rate_m3_h": None if not np.isfinite(item["ValidatedRate_m3ph"]) else round(item["ValidatedRate_m3ph"], 6),
            "calculation_crs": settings.projected_crs,
            "accepted_cell_count": item["AcceptedCells"],
            "accepted_ems_hours": round(item["AcceptedEMS_Hours"], 6),
            "mean_coverage_percent": round(item["MeanCoveragePct"], 3),
            "qa_status": item["QA_Status"],
            "rate_method": item["RateMethod"],
            "background_lowering_m": None if not np.isfinite(item["Background_m"]) else round(item["Background_m"], 6),
            "background_source": item["BackgroundSource"],
            "configured_rate_t_h": settings.configured_rate_tph,
            "created_by": f"{APP_NAME} v{VERSION}",
            "created_at": created,
            "calibration_id": f"LFV-{item['LicenceNo']}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        })
    profile_path = out / f"lowering_rate_profile{output_suffix}.json"
    for profile in profiles:
        profile["resolution"] = resolution_label
    with open(profile_path, "w", encoding="utf-8") as handle:
        json.dump({"profiles": profiles}, handle, indent=2, allow_nan=False)

    metadata = {
        "app_name": APP_NAME,
        "version": VERSION,
        "resolution": resolution_label,
        "created_at": created,
        "settings": settings.as_dict(),
        "summary": summary_df.to_dict(orient="records"),
        "warnings": [
            "Measured bathymetric change may include dredging, sediment movement, accretion and survey effects.",
            "Review residual maps and all AMBER/RED results before operational use.",
        ],
    }
    (out / f"lowering_factor_validation_metadata{output_suffix}.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False), encoding="utf-8"
    )

    report_lines = [
        APP_NAME, f"Version: {VERSION}", f"Created: {created}", "",
        "RESOLUTION", resolution_label, "", "CALCULATION SETTINGS",
        json.dumps(settings.as_dict(), indent=2), "", "LICENCE RESULTS"
    ]
    report_lines.append(summary_df.to_string(index=False))
    report_lines.extend(["", "IMPORTANT QA NOTE", "Measured bathymetric change may include dredging, sediment movement, accretion and survey effects. Review residual maps and all AMBER/RED results before operational use."])
    (out / f"lowering_factor_validation_report{output_suffix}.txt").write_text("\n".join(report_lines), encoding="utf-8")
    log(f"Complete. Outputs written to {out}")
    if resolution_label == RESOLUTION_250:
        rate_distribution = _numeric_distribution(
            diagnostic_table.loc[
                diagnostic_table["raster_valid"], "ObsRate_tph"
            ]
        )
        log("250 m VALIDATION DIAGNOSTIC SUMMARY")
        log(
            f"Grid cells considered: {len(diagnostic_table):,}; "
            f"valid raster cells: {int(diagnostic_table['raster_valid'].sum()):,}"
        )
        log(
            f"EMS area-candidate records: {assignment_counts['total']:,}; "
            f"EMS records assigned to valid cells: {assignment_counts['valid_grid']:,}; "
            f"cells receiving EMS: {int((ems_per_cell > 0).sum()):,}"
        )
        log(
            f"Cells with calculable duration: "
            f"{_numeric_distribution(diagnostic_table.loc[diagnostic_table['raster_valid'], 'EMS_Hours'])['finite']:,}; "
            f"bathy change: {_numeric_distribution(diagnostic_table.loc[diagnostic_table['raster_valid'], 'ObsLower_m'])['finite']:,}; "
            f"volume: {_numeric_distribution(diagnostic_table.loc[diagnostic_table['raster_valid'], 'NetVolume_m3'])['finite']:,}; "
            f"calculated rate: {len(grid_calc):,}."
        )
        log(
            f"Positive rates: {rate_distribution['positive']:,}; "
            f"zero: {rate_distribution['zero']:,}; "
            f"negative: {rate_distribution['negative']:,}; "
            "invalid: "
            f"{rate_distribution['nan'] + rate_distribution['infinity'] + rate_distribution['missing']:,}; "
            f"valid positive rates: {int(diagnostic_table['valid_positive_rate'].sum()):,}."
        )
        primary_counts = diagnostic_table[
            "primary_rejection_reason"
        ].value_counts()
        primary_rejection = (
            str(primary_counts.drop(labels=["VALID_POSITIVE_RATE"], errors="ignore").idxmax())
            if (primary_counts.index != "VALID_POSITIVE_RATE").any()
            else "none"
        )
        log(f"Primary rejection reason: {primary_rejection}")
        log(f"Diagnostic table: {diagnostic_table_path}")
    if resolution_label == RESOLUTION_50:
        output_seconds = time.perf_counter() - output_started
        log(f"50 m grid generation: {grid_generation_seconds:.3f} sec")
        log(f"50 m raster statistics: {raster_stats_seconds:.3f} sec")
        log(f"50 m EMS assignment: {ems_assignment_seconds:.3f} sec")
        log(
            "50 m validation calculation: "
            f"{validation_calculation_seconds:.3f} sec"
        )
        log(f"50 m output: {output_seconds:.3f} sec")
        log(
            "Total 50 m validation: "
            f"{time.perf_counter() - validation_started:.3f} sec"
        )
    (out / f"processing_log{output_suffix}.txt").write_text(
        "\n".join(messages), encoding="utf-8"
    )
    return summary_df


RUN_LOG_NAME = "run_log.txt"


def run_validation(
    settings: Settings,
    callback: Optional[Callable[[str], None]] = None,
    *,
    cancel_event: threading.Event | None = None,
    confirm: Optional[Callable[[list[str]], bool]] = None,
) -> pd.DataFrame:
    """Run a validation and save every message to run_log.txt in the output folder.

    The log is written however the run ends: completed, cancelled or failed.
    """
    lines: list[str] = []
    started = datetime.now().astimezone()

    def recording_callback(text: str) -> None:
        lines.append(f"{datetime.now():%H:%M:%S}  {text}")
        notify(callback, text)

    outcome = "COMPLETED"
    try:
        return _run_validation(
            settings, recording_callback, cancel_event=cancel_event, confirm=confirm
        )
    except RunCancelled as exc:
        outcome = f"CANCELLED: {exc}"
        raise
    except BaseException as exc:
        outcome = f"FAILED: {exc}"
        if cancel_event is not None and cancel_event.is_set():
            outcome = "CANCELLED: Run cancelled by user."
        lines.append(traceback.format_exc())
        raise
    finally:
        finished = datetime.now().astimezone()
        header = [
            f"{APP_NAME} v{VERSION} - run log",
            f"Started:  {started.isoformat(timespec='seconds')}",
            f"Finished: {finished.isoformat(timespec='seconds')} "
            f"({(finished - started).total_seconds() / 60:.1f} min)",
            f"Outcome:  {outcome}",
            "",
            "SETTINGS",
            json.dumps(settings.as_dict(), indent=2),
            "",
            "MESSAGES",
        ]
        try:
            # Only write where the run created (or found) its output folder, so
            # a run stopped by an invalid raster leaves no folder behind.
            folder = Path(settings.output_folder)
            if folder.is_dir():
                (folder / RUN_LOG_NAME).write_text(
                    "\n".join(header + lines) + "\n", encoding="utf-8"
                )
        except OSError:
            pass


def _run_validation(
    settings: Settings,
    callback: Optional[Callable[[str], None]] = None,
    *,
    cancel_event: threading.Event | None = None,
    confirm: Optional[Callable[[list[str]], bool]] = None,
) -> pd.DataFrame:
    """Validate at the selected resolution(s), loading shared inputs once.

    cancel_event stops the run at the next log message or EMS database step.
    confirm receives pre-run check warnings and returns False to stop.
    """
    settings.validate()

    def checked_callback(text: str) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("Run cancelled by user.")
        notify(callback, text)

    def log(text: str) -> None:
        checked_callback(text)

    output_folder = Path(settings.output_folder)
    log(f"{APP_NAME} v{VERSION}")
    log(f"Requested validation resolutions: {settings.resolution_mode}")
    difference_raster = prepare_difference_raster(settings, output_folder, log)
    difference_sign = (
        settings.source_sign
        if settings.difference_source == "RASTER"
        else "positive_is_lowering"
    )
    raster_value_convention = (
        "ELEVATION"
        if settings.difference_source == "XYZ_SURVEYS"
        else settings.raster_value_convention
    )
    log("Validating bathymetric-difference raster before EMS processing...")
    raster_validation = validate_bathymetric_difference_raster(
        difference_raster,
        settings.difference_units,
        difference_sign,
        raster_value_convention,
        log,
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    grid_bbox = None
    licence_bbox = None
    if settings.analysis_extent == "FULL_OVERLAP":
        grid_bbox = raster_bounds_in_vector_crs(
            difference_raster,
            settings.calibration_grid,
            settings.calibration_grid_layer,
        )
        licence_bbox = raster_bounds_in_vector_crs(
            difference_raster,
            settings.licence_layer,
            settings.licence_layer_name,
        )
        log("Survey raster bounds prepared for working-grid filtering.")

    detailed_grid = None
    detailed_grid_generation_seconds = 0.0
    layout = None
    if settings.resolution_mode in {RESOLUTION_50, RESOLUTION_BOTH}:
        log("Inspecting authoritative 250 m master-grid alignment...")
        alignment_grid = read_vector(
            settings.calibration_grid,
            settings.calibration_grid_layer,
        )
        log(
            f"Loaded {len(alignment_grid):,} feature(s) from the complete "
            "authoritative grid layer for alignment inspection."
        )
        grid_id_column = field_lookup(
            alignment_grid.columns, settings.grid_id_field
        )
        alignment_grid = drop_partial_cells(
            alignment_grid,
            grid_id_column,
            settings.projected_crs,
            log,
            label="master grid layer",
        )
        layout = inspect_authoritative_master_grid(
            alignment_grid,
            grid_id_column,
            cell_size_m=250.0,
            log=log,
        )
        origin_x = (
            math.cos(layout.angle_radians) * layout.origin_u
            - math.sin(layout.angle_radians) * layout.origin_v
        )
        origin_y = (
            math.sin(layout.angle_radians) * layout.origin_u
            + math.cos(layout.angle_radians) * layout.origin_v
        )
        log(f"Master grid CRS: {CRS.from_user_input(alignment_grid.crs).to_string()}")
        log(
            "Master cell size: "
            f"{layout.cell_size_m:g} x {layout.cell_size_m:g} m"
        )
        if getattr(layout, "parent_frames", ()):
            log(
                "Master grid geometry: independently aligned nominal 250 m "
                "parent frames (locally transformed national grid)."
            )
        else:
            log(f"Master grid origin: X={origin_x:.3f}, Y={origin_y:.3f}")
        log("Master-grid alignment validated.")

        if grid_bbox is not None:
            log("Restricting working grid to survey raster coverage...")
            master_grid = read_vector(
                settings.calibration_grid,
                settings.calibration_grid_layer,
                bbox=grid_bbox,
            )
            master_grid = drop_partial_cells(
                master_grid,
                grid_id_column,
                settings.projected_crs,
                log,
                label="working grid",
            )
        else:
            master_grid = alignment_grid
        log(
            f"Loaded {len(master_grid):,} relevant 250 m parent cells."
        )
    else:
        log("Loading the authoritative 250 m master grid...")
        if grid_bbox is not None:
            log("Restricting working grid to survey raster coverage...")
        master_grid = read_vector(
            settings.calibration_grid,
            settings.calibration_grid_layer,
            bbox=grid_bbox,
        )
        grid_id_column = field_lookup(
            master_grid.columns, settings.grid_id_field
        )
        master_grid = drop_partial_cells(
            master_grid,
            grid_id_column,
            settings.projected_crs,
            log,
            label="calibration grid",
        )

    licences = read_vector(
        settings.licence_layer,
        settings.licence_layer_name,
        bbox=licence_bbox,
    )
    wanted = selected_licences(settings)
    if wanted:
        licence_column = field_lookup(licences.columns, settings.licence_field)
        licence_ids = licences[licence_column].astype(str).str.strip()
        unknown = sorted(set(wanted) - set(licence_ids))
        licences = licences.loc[licence_ids.isin(wanted)].copy()
        if licences.empty:
            raise ValueError(
                "None of the selected licences were found in the licence layer"
                + (" within the survey raster extent" if licence_bbox else "")
                + f": {', '.join(wanted)}."
            )
        if unknown:
            log(f"WARNING: selected licences not found and skipped: {', '.join(unknown)}")
        selection = licences.to_crs(master_grid.crs).geometry
        selection_area = (
            selection.union_all() if hasattr(selection, "union_all") else selection.unary_union
        )
        in_selection = master_grid.geometry.representative_point().within(selection_area)
        master_grid = master_grid.loc[in_selection].copy()
        if master_grid.empty:
            raise ValueError("No grid cells fall inside the selected licences.")
        log(
            f"Running {len(licences):,} selected licence(s) "
            f"({', '.join(sorted(set(licences[licence_column].astype(str))))}) "
            f"with {len(master_grid):,} grid cells."
        )

    warnings = preflight_checks(
        settings,
        master_grid,
        licences,
        grid_id_column,
        raster_validation,
        log,
        check_ems_months=find_ems_cache(settings, master_grid) is None,
    )
    if warnings and confirm is not None and not confirm(warnings):
        raise RunCancelled("Run stopped after pre-run checks.")

    if settings.resolution_mode in {RESOLUTION_50, RESOLUTION_BOTH}:
        log(
            "Survey/raster candidate parent cells: "
            f"{len(master_grid):,}"
        )
        log("Generating aligned 50 m experimental grid...")
        grid_generation_started = time.perf_counter()
        detailed_grid = create_aligned_child_grid(master_grid, layout, child_size_m=50.0)
        detailed_grid_generation_seconds = time.perf_counter() - grid_generation_started
        projected = CRS.from_user_input(settings.projected_crs)
        candidate_children = detailed_grid.to_crs(projected)
        candidate_licences = licences.to_crs(projected)[["geometry"]]
        child_representatives = candidate_children[
            ["Parent250_ID", "geometry"]
        ].copy()
        child_representatives.geometry = (
            child_representatives.geometry.representative_point()
        )
        licence_matches = gpd.sjoin(
            child_representatives,
            candidate_licences,
            how="inner",
            predicate="within",
        )
        licence_assigned_parents = (
            licence_matches["Parent250_ID"].nunique()
            if not licence_matches.empty
            else 0
        )
        log(
            "Licence-assigned parent cells: "
            f"{licence_assigned_parents:,}"
        )
        if not licence_matches.empty:
            assigned_parent_ids = set(licence_matches["Parent250_ID"])
            detailed_grid = detailed_grid.loc[
                detailed_grid["Parent250_ID"].isin(assigned_parent_ids)
            ].copy()
        log(
            f"Generated {len(detailed_grid):,} aligned 50 m experimental cells "
            f"from {detailed_grid['Parent250_ID'].nunique():,} licence-relevant "
            "authoritative 250 m parent cells."
        )
        log(
            "50 m grid generation: "
            f"{detailed_grid_generation_seconds:.3f} sec."
        )

    log("Loading and sampling EMS records once for the survey/grid overlap...")
    ems = load_ems_cached(settings, log, master_grid, cancel_event=cancel_event)

    if settings.resolution_mode == RESOLUTION_250:
        resolutions = [(RESOLUTION_250, master_grid, "", 0.0)]
    elif settings.resolution_mode == RESOLUTION_50:
        resolutions = [
            (
                RESOLUTION_50,
                detailed_grid,
                "_50m_experimental",
                detailed_grid_generation_seconds,
            )
        ]
    else:
        resolutions = [
            (RESOLUTION_250, master_grid, "_250m_regulatory", 0.0),
            (
                RESOLUTION_50,
                detailed_grid,
                "_50m_experimental",
                detailed_grid_generation_seconds,
            ),
        ]

    summaries = []
    for resolution_label, grid, suffix, generation_seconds in resolutions:
        log(f"Running like-for-like validation at {resolution_label}...")
        summary = _run_validation_single(
            settings,
            checked_callback,
            grid_override=grid,
            licences_override=licences,
            ems_override=ems,
            difference_raster_override=difference_raster,
            raster_validation=raster_validation,
            grid_generation_seconds=generation_seconds,
            output_suffix=suffix,
            resolution_label=resolution_label,
        )
        if settings.resolution_mode != RESOLUTION_250:
            summary.insert(0, "Resolution", resolution_label)
        summaries.append(summary)

    result = pd.concat(summaries, ignore_index=True) if len(summaries) > 1 else summaries[0]
    if settings.resolution_mode == RESOLUTION_BOTH:
        combined_path = output_folder / "lowering_factor_validation_summary_both_resolutions.csv"
        result.to_csv(combined_path, index=False)
        log(f"Combined resolution summary written to {combined_path}")
        diagnostics_250m = pd.read_csv(
            output_folder / "validation_250m_regulatory_diagnostics.csv"
        )
        diagnostics_50m = pd.read_csv(
            output_folder / "validation_50m_diagnostics.csv"
        )
        _write_resolution_comparison(
            output_folder,
            settings,
            diagnostics_250m,
            diagnostics_50m,
            log,
        )

    if not settings.write_model_factors:
        log(
            "Source-model factor update skipped: writing lowering_factor values "
            "to the licence GeoPackage is switched off. Review the results, then "
            "re-run with model write-back switched on to fill blank factors."
        )
    elif settings.resolution_mode == RESOLUTION_50:
        log(
            "Source-model factor update skipped: it requires the 250 m regulatory "
            "validation result."
        )
    else:
        factor_summary = (
            result.loc[result["Resolution"] == RESOLUTION_250]
            if settings.resolution_mode == RESOLUTION_BOTH
            else result
        )
        writeback_rates = pd.to_numeric(
            factor_summary["ValidatedRate_m3ph"], errors="coerce"
        ).to_numpy(dtype=float)
        finite_rates = np.isfinite(writeback_rates)
        positive_rates = finite_rates & (writeback_rates > 0)
        log("LOWERING_FACTOR WRITE-BACK DIAGNOSTICS")
        log(
            f"Calculated 250 m licence rates: {len(writeback_rates):,}; "
            f"finite: {int(finite_rates.sum()):,}; "
            f"positive: {int(positive_rates.sum()):,}; "
            f"eligible for write-back: {int(positive_rates.sum()):,}."
        )
        log(
            "Write-back eligibility matches update_model_lowering_factors: "
            "ValidatedRate_m3ph must be finite and > 0; the stored factor is "
            "that rate divided by 250 x 250 m2."
        )
        for row in factor_summary.itertuples(index=False):
            rate = pd.to_numeric(
                pd.Series([getattr(row, "ValidatedRate_m3ph")]), errors="coerce"
            ).iloc[0]
            if pd.notna(rate) and np.isfinite(rate) and rate > 0:
                log(
                    f"Write-back candidate {getattr(row, 'LicenceNo')}: "
                    f"{getattr(row, 'ValidatedRate_tph'):.3f} t/h; "
                    f"QA status {getattr(row, 'QA_Status')}."
                )
        target, layer, id_field = writeback_target(settings)
        log(
            f"Write-back target: {target} | layer {layer or '(default)'} | "
            f"matching LicenceNo to {id_field}."
        )
        update_model_lowering_factors(target, layer, id_field, factor_summary, log)
    return result


def writeback_target(settings: Settings) -> tuple[str, str, str]:
    """(GeoPackage, layer, ID field) that model write-back fills.

    The separate model when one is set; otherwise the licence layer itself,
    as before.
    """
    if settings.model_path.strip():
        return (
            settings.model_path.strip(),
            settings.model_layer.strip(),
            settings.model_id_field.strip(),
        )
    return settings.licence_layer, settings.licence_layer_name, settings.licence_field


RUN_STAGES = (
    ("Validating bathymetric-difference raster", "Checking the difference raster..."),
    ("Inspecting authoritative 250 m master-grid", "Loading and checking the grid..."),
    ("Loading the authoritative 250 m master grid", "Loading the grid..."),
    ("Running pre-run checks", "Running pre-run checks..."),
    ("Generating aligned 50 m", "Building the 50 m grid..."),
    ("Using saved EMS records", "Loading saved EMS records..."),
    ("Opening the shared EMS index", "Opening the EMS database..."),
    ("Discovering EMS files", "Scanning EMS files..."),
    ("Repairing EMS vessel name", "Repairing vessel names (one-time)..."),
    ("Starting indexed EMS query", "Querying EMS records (slowest step)..."),
    ("Scanned ", "Querying EMS records (slowest step)..."),
    ("Saved EMS records for reuse", "EMS records saved for next time..."),
    ("Running like-for-like validation at", None),
    ("Calculating bathymetric change statistics", "Calculating raster statistics..."),
    ("Loading and spatially assigning EMS", "Assigning EMS records to cells..."),
    ("Diagnostic table written", "Writing outputs..."),
)


def run_stage(text: str) -> str | None:
    """Short progress label for a log message, or None to keep the current one."""
    for prefix, stage in RUN_STAGES:
        if text.startswith(prefix):
            if stage is None:
                return "Validating at " + text.split(" at ", 1)[1].rstrip(".") + "..."
            return stage
    return None


def licence_choices(path: str, layer: str, field: str) -> list[str]:
    """Sorted licence IDs from the licence layer, for the licence picker."""
    info = pyogrio.read_info(path, layer=layer or None)
    actual = field_lookup([str(name) for name in info["fields"]], field)
    frame = pyogrio.read_dataframe(
        path, layer=layer or None, columns=[actual], read_geometry=False
    )
    values = sorted({str(value).strip() for value in frame[actual].dropna()})
    return [value for value in values if value]


def _format_result_value(key: str, value: object) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if key in {"ValidatedRate_tph", "UncorrectedRate_tph", "RegressionRate_tph", "AcceptedEMS_Hours"}:
        return f"{float(value):,.0f}"
    if key == "Background_m":
        return f"{float(value):.3f}"
    return str(value)


def _open_path(path: Path) -> None:
    if not path.exists():
        messagebox.showerror(APP_NAME, f"Not found: {path}")
        return
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def _open_newest(folder: Path, pattern: str) -> None:
    matches = sorted(folder.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        messagebox.showerror(APP_NAME, f"No {pattern} file found in {folder}")
        return
    _open_path(matches[-1])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("980x820")
        self.minsize(880, 700)
        self.vars = {}
        self._run_messages = None
        self._run_thread = None
        self._cancel_event = None
        self._run_started = None
        self._warning_count = 0
        self._last_summary = None
        self._last_output = None
        self.run_button = None
        self._build()

    def add_path(self, parent, row, label, key, directory=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        def browse():
            value = filedialog.askdirectory() if directory else filedialog.askopenfilename()
            if value:
                var.set(value)
        button_text = "Select folder..." if directory else "Select file..."
        ttk.Button(parent, text=button_text, command=browse).grid(row=row, column=2, padx=6, pady=4)

    def add_difference_raster_path(self, parent, row):
        ttk.Label(parent, text="Bathymetric difference raster").grid(
            row=row, column=0, sticky="w", padx=6, pady=4
        )
        variable = tk.StringVar()
        self.vars["difference_raster"] = variable
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=6, pady=4
        )

        def browse():
            value = filedialog.askopenfilename(
                title="Select bathymetric-difference raster",
                filetypes=[
                    ("ESRI Binary Float Grid", "*.flt"),
                    ("Numeric raster datasets", "*.flt *.tif *.tiff"),
                    ("All files", "*.*"),
                ],
            )
            if not value:
                return
            variable.set(value)
            messages = []
            try:
                self.log("Validating selected bathymetric-difference raster...")
                validate_bathymetric_difference_raster(
                    value,
                    self.vars["difference_units"].get(),
                    self.vars["source_sign"].get(),
                    self.vars["raster_value_convention"].get(),
                    messages.append,
                )
                for message in messages:
                    self.log(message)
            except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
                self.log(f"ERROR: {exc}")
                messagebox.showerror(APP_NAME, str(exc), parent=self)

        ttk.Button(parent, text="Select file...", command=browse).grid(
            row=row, column=2, padx=6, pady=4
        )

    def add_entry(self, parent, row, label, key, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        var = tk.StringVar(value=str(default))
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=3)

    def load_vector_layers(self, path_key, id_field_key, layer_key, combo):
        try:
            layers, recommended = vector_layer_choices(
                self.vars[path_key].get().strip(),
                self.vars[id_field_key].get().strip(),
            )
            combo.configure(values=layers)
            current = self.vars[layer_key].get().strip()
            self.vars[layer_key].set(current if current in layers else recommended)
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def _build(self):
        canvas = tk.Canvas(self, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        def on_wheel(event):
            # Leave scrolling to the log box and open dropdown lists.
            if isinstance(event.widget, (tk.Text, tk.Listbox)) or str(event.widget).endswith("popdown.f.l"):
                return
            if getattr(event, "num", None) == 4:
                step = -3
            elif getattr(event, "num", None) == 5:
                step = 3
            else:
                step = -1 * int(event.delta / 120) * 3 if abs(event.delta) >= 120 else -int(event.delta)
            canvas.yview_scroll(step, "units")

        self.bind_all("<MouseWheel>", on_wheel, add="+")
        self.bind_all("<Button-4>", on_wheel, add="+")
        self.bind_all("<Button-5>", on_wheel, add="+")
        body.columnconfigure(0, weight=1)

        files = ttk.LabelFrame(body, text="1. Input and output data")
        files.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        files.columnconfigure(1, weight=1)
        self.add_difference_raster_path(files, 0)
        self.add_path(files, 1, "Calibration grid", "calibration_grid")
        self.add_path(files, 2, "Licence polygons", "licence_layer")
        self.add_path(files, 3, "EMS file or root folder", "ems_source", directory=True)
        ttk.Button(
            files,
            text="Select file...",
            command=lambda: self.vars["ems_source"].set(
                filedialog.askopenfilename() or self.vars["ems_source"].get()
            ),
        ).grid(row=3, column=3, padx=3)
        self.add_path(files, 4, "Output folder", "output_folder", directory=True)
        ttk.Label(files, text="Validation resolution").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        self.vars["resolution_mode"] = tk.StringVar(value=RESOLUTION_250)
        ttk.Combobox(
            files,
            textvariable=self.vars["resolution_mode"],
            values=[RESOLUTION_250, RESOLUTION_50, RESOLUTION_BOTH],
            state="readonly",
        ).grid(row=5, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(
            files,
            text="The selected calibration grid must be the 250 m regulatory master.",
        ).grid(row=6, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(files, text="Calibration grid layer").grid(
            row=7, column=0, sticky="w", padx=6, pady=4
        )
        self.vars["calibration_grid_layer"] = tk.StringVar()
        grid_layer_combo = ttk.Combobox(
            files,
            textvariable=self.vars["calibration_grid_layer"],
            state="readonly",
        )
        grid_layer_combo.grid(row=7, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(
            files,
            text="Load layers",
            command=lambda: self.load_vector_layers(
                "calibration_grid",
                "grid_id_field",
                "calibration_grid_layer",
                grid_layer_combo,
            ),
        ).grid(row=7, column=2, padx=4)
        ttk.Label(files, text="Licence polygon layer").grid(
            row=8, column=0, sticky="w", padx=6, pady=4
        )
        self.vars["licence_layer_name"] = tk.StringVar()
        licence_layer_combo = ttk.Combobox(
            files,
            textvariable=self.vars["licence_layer_name"],
            state="readonly",
        )
        licence_layer_combo.grid(row=8, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(
            files,
            text="Load layers",
            command=lambda: self.load_vector_layers(
                "licence_layer",
                "licence_field",
                "licence_layer_name",
                licence_layer_combo,
            ),
        ).grid(row=8, column=2, padx=4)
        ttk.Label(
            files,
            text=(
                "For model write-back, select its areas layer and set Licence number field "
                "to area_id; blank lowering_factor values are filled."
            ),
        ).grid(row=9, column=1, columnspan=4, sticky="w", padx=6, pady=2)

        fields = ttk.LabelFrame(body, text="2. Fields and survey period")
        fields.grid(row=1, column=0, sticky="ew", padx=10, pady=8)
        fields.columnconfigure(1, weight=1)
        entries = [
            ("Grid ID field", "grid_id_field", "GridID"), ("Licence number field", "licence_field", "LicenceNo"),
            ("EMS longitude/X field", "ems_x_field", "GPSLON"), ("EMS latitude/Y field", "ems_y_field", "GPSLAT"),
            ("EMS date-time field", "ems_datetime_field", "DATETIME"), ("EMS dredge field", "ems_dredge_field", "DREDGE"),
            ("EMS vessel field (optional)", "ems_vessel_field", "VESSEL"), ("Active dredge value", "ems_dredge_value", "3"),
            ("Survey start, dd/mm/yyyy", "survey_start", ""), ("Survey end, dd/mm/yyyy", "survey_end", ""),
            ("Calculation CRS", "projected_crs", "EPSG:27700"),
        ]
        for r, item in enumerate(entries): self.add_entry(fields, r, *item)
        self.vars["ems_time_field"] = tk.StringVar(value="TIME")
        ttk.Label(fields, text="EMS time field (when date and time are separate)").grid(row=11, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(fields, textvariable=self.vars["ems_time_field"]).grid(row=11, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Dredge comparison").grid(row=12, column=0, sticky="w", padx=6, pady=3)
        self.vars["ems_dredge_mode"] = tk.StringVar(value="equals")
        ttk.Combobox(fields, textvariable=self.vars["ems_dredge_mode"], values=["equals", "greater_or_equal"], state="readonly").grid(row=12, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Lowering sign after value convention (single raster input)").grid(row=13, column=0, sticky="w", padx=6, pady=3)
        self.vars["source_sign"] = tk.StringVar(value="positive_is_lowering")
        ttk.Combobox(fields, textvariable=self.vars["source_sign"], values=["positive_is_lowering", "negative_is_lowering"], state="readonly").grid(row=13, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Raster value convention").grid(row=14, column=0, sticky="w", padx=6, pady=3)
        self.vars["raster_value_convention"] = tk.StringVar(value="ELEVATION")
        ttk.Combobox(fields, textvariable=self.vars["raster_value_convention"], values=["ELEVATION", "POSITIVE_DEPTH"], state="readonly").grid(row=14, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Bathymetric difference units").grid(row=15, column=0, sticky="w", padx=6, pady=3)
        self.vars["difference_units"] = tk.StringVar(value="metres")
        ttk.Combobox(fields, textvariable=self.vars["difference_units"], values=["metres"], state="readonly").grid(row=15, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Change tolerance (m)").grid(row=16, column=0, sticky="w", padx=6, pady=3)
        self.vars["change_tolerance_m"] = tk.StringVar(value="0.05")
        ttk.Entry(fields, textvariable=self.vars["change_tolerance_m"]).grid(row=16, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Validation mode").grid(row=17, column=0, sticky="w", padx=6, pady=3)
        self.vars["validation_mode"] = tk.StringVar(value="DEEPENING_ONLY")
        ttk.Combobox(fields, textvariable=self.vars["validation_mode"], values=["DEEPENING_ONLY", "NET_CHANGE", "TRACK_FOOTPRINT", "FOOTPRINT_AND_CONTROL_AREA"], state="readonly").grid(row=17, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Analysis extent").grid(row=18, column=0, sticky="w", padx=6, pady=3)
        self.vars["analysis_extent"] = tk.StringVar(value="FULL_OVERLAP")
        ttk.Combobox(fields, textvariable=self.vars["analysis_extent"], values=["TRACK_FOOTPRINT", "VALIDATION_POLYGON", "FULL_OVERLAP"], state="readonly").grid(row=18, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Background (natural) lowering correction").grid(row=20, column=0, sticky="w", padx=6, pady=3)
        self.vars["background_correction"] = tk.StringVar(value="UNDREDGED_CELLS")
        ttk.Combobox(fields, textvariable=self.vars["background_correction"], values=["UNDREDGED_CELLS", "NONE"], state="readonly").grid(row=20, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Minimum undredged cells for background").grid(row=21, column=0, sticky="w", padx=6, pady=3)
        self.vars["background_min_cells"] = tk.StringVar(value="5")
        ttk.Entry(fields, textvariable=self.vars["background_min_cells"]).grid(row=21, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Licence rate method").grid(row=22, column=0, sticky="w", padx=6, pady=3)
        self.vars["rate_method"] = tk.StringVar(value="POOLED")
        ttk.Combobox(fields, textvariable=self.vars["rate_method"], values=["POOLED", "CELL_FILTERS"], state="readonly").grid(row=22, column=1, sticky="ew", padx=6)
        ttk.Label(fields, text="Licences to run (blank = all)").grid(row=23, column=0, sticky="w", padx=6, pady=3)
        self.vars["licence_filter"] = tk.StringVar(value="")
        ttk.Entry(fields, textvariable=self.vars["licence_filter"]).grid(row=23, column=1, sticky="ew", padx=6)
        ttk.Button(fields, text="Choose...", command=self.choose_licences).grid(row=23, column=2, padx=4)
        self.vars["reuse_ems_cache"] = tk.StringVar(value="True")
        ttk.Checkbutton(
            fields,
            text="Reuse EMS data from an earlier run when EMS settings and dates match (much faster)",
            variable=self.vars["reuse_ems_cache"],
            onvalue="True",
            offvalue="False",
        ).grid(row=24, column=0, columnspan=3, sticky="w", padx=6, pady=3)

        qa = ttk.LabelFrame(body, text="3. Calculation and QA settings")
        qa.grid(row=2, column=0, sticky="ew", padx=10, pady=8)
        qa.columnconfigure(1, weight=1)
        numeric = [
            ("Bulk density (t/m³)", "bulk_density", 1.7), ("EMS sample interval (minutes)", "sample_interval_minutes", 1.0),
            ("Minimum EMS time per cell (hours)", "minimum_ems_hours", 0.5), ("Minimum raster coverage (%)", "minimum_coverage_percent", 80),
            ("Minimum detectable lowering (m)", "minimum_detectable_change_m", 0.2), ("Maximum plausible rate (t/hour)", "maximum_rate_tph", 1500),
            ("Configured EMS rate (t/hour)", "configured_rate_tph", 500), ("GREEN tolerance (%)", "green_tolerance_percent", 15),
            ("AMBER tolerance (%)", "amber_tolerance_percent", 30),
        ]
        for r, item in enumerate(numeric): self.add_entry(qa, r, *item)

        writeback = ttk.LabelFrame(body, text="4. Model write-back (optional)")
        writeback.grid(row=3, column=0, sticky="ew", padx=10, pady=8)
        writeback.columnconfigure(1, weight=1)
        self.vars["write_model_factors"] = tk.StringVar(value="False")
        ttk.Checkbutton(
            writeback,
            text="Write the validated 250 m lowering factor to the model "
            "(fills blank lowering_factor values only)",
            variable=self.vars["write_model_factors"],
            onvalue="True",
            offvalue="False",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=3)
        ttk.Label(writeback, text="Model to update (write-back target)").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.vars["model_path"] = tk.StringVar(value="")
        ttk.Entry(writeback, textvariable=self.vars["model_path"]).grid(row=1, column=1, sticky="ew", padx=6)

        def browse_model():
            value = filedialog.askopenfilename(
                title="Select the Lowering Model GeoPackage",
                filetypes=[("GeoPackage", "*.gpkg"), ("All files", "*.*")],
            )
            if value:
                self.vars["model_path"].set(value)

        ttk.Button(writeback, text="Select file...", command=browse_model).grid(row=1, column=2, padx=4)
        ttk.Label(writeback, text="Model layer").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        self.vars["model_layer"] = tk.StringVar(value="areas")
        model_layer_combo = ttk.Combobox(writeback, textvariable=self.vars["model_layer"], state="readonly", values=["areas"])
        model_layer_combo.grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Button(
            writeback,
            text="Load layers",
            command=lambda: self.load_vector_layers("model_path", "model_id_field", "model_layer", model_layer_combo),
        ).grid(row=2, column=2, padx=4)
        ttk.Label(writeback, text="Model area ID field (matched to licence number)").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        self.vars["model_id_field"] = tk.StringVar(value="area_id")
        ttk.Entry(writeback, textvariable=self.vars["model_id_field"]).grid(row=3, column=1, sticky="ew", padx=6)
        ttk.Label(
            writeback,
            text="Leave 'Model to update' blank to write to the licence layer instead.",
        ).grid(row=4, column=1, columnspan=2, sticky="w", padx=6, pady=(0, 3))


        action = ttk.Frame(body)
        action.grid(row=4, column=0, sticky="ew", padx=10, pady=8)
        self.run_button = ttk.Button(action, text="Run validation", command=self.run)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(action, text="Cancel", command=self.cancel_run, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        ttk.Button(action, text="Save settings", command=self.save_settings).pack(side="left", padx=8)
        ttk.Button(action, text="Load settings", command=self.load_settings).pack(side="left")
        ttk.Button(action, text="Show last results", command=self.show_last_results).pack(side="left", padx=8)

        status = ttk.Frame(body)
        status.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 4))
        status.columnconfigure(1, weight=1)
        self.stage_label = ttk.Label(status, text="Ready.")
        self.stage_label.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(status, mode="indeterminate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=10)
        self.elapsed_label = ttk.Label(status, text="")
        self.elapsed_label.grid(row=0, column=2, sticky="e")

        log_frame = ttk.LabelFrame(body, text="Processing log")
        log_frame.grid(row=6, column=0, sticky="nsew", padx=10, pady=8)
        self.logbox = tk.Text(log_frame, height=12, wrap="word")
        self.logbox.pack(fill="both", expand=True, padx=6, pady=6)

    def log(self, text):
        self.logbox.insert("end", text + "\n")
        self.logbox.see("end")
        self.update_idletasks()

    def settings(self):
        data = {k: v.get().strip() for k, v in self.vars.items()}
        numeric = ["bulk_density", "sample_interval_minutes", "minimum_ems_hours", "minimum_coverage_percent", "minimum_detectable_change_m", "maximum_rate_tph", "configured_rate_tph", "green_tolerance_percent", "amber_tolerance_percent", "change_tolerance_m"]
        for key in numeric:
            if key in data and data[key] != "":
                data[key] = float(data[key])
        data["write_model_factors"] = data.get("write_model_factors") == "True"
        data["reuse_ems_cache"] = data.get("reuse_ems_cache", "True") == "True"
        required = ["calibration_grid", "licence_layer", "ems_source", "output_folder"]
        missing = [k for k in required if not data.get(k)]
        if missing:
            raise ValueError(
                "Select all required input and output paths: " + ", ".join(missing)
            )
        settings = Settings(**data)
        settings.validate()
        return settings

    def run(self):
        if self._run_thread is not None and self._run_thread.is_alive():
            return
        try:
            self.logbox.delete("1.0", "end")
            settings = self.settings()
        except Exception as exc:
            self.log("ERROR: " + str(exc))
            self.log(traceback.format_exc())
            messagebox.showerror(APP_NAME, str(exc))
            return

        self._run_messages = queue.Queue()
        self._cancel_event = threading.Event()
        self._warning_count = 0
        self._last_output = settings.output_folder
        self._run_started = time.monotonic()

        self.run_button.configure(state="disabled", text="Running...")
        self.cancel_button.configure(state="normal", text="Cancel")
        self.stage_label.configure(text="Starting...")
        self.progress.start(15)
        self._tick_elapsed()
        self._run_thread = threading.Thread(
            target=_run_validation_worker,
            args=(settings, self._run_messages, self._cancel_event),
            name="validator-run",
            daemon=True,
        )
        self._run_thread.start()
        self.after(100, self._poll_run_messages)

    def cancel_run(self):
        if self._cancel_event is not None and not self._cancel_event.is_set():
            self._cancel_event.set()
            self.cancel_button.configure(state="disabled", text="Cancelling...")
            self.stage_label.configure(text="Cancelling at the next safe point...")

    def _tick_elapsed(self):
        if self._run_started is None:
            return
        seconds = int(time.monotonic() - self._run_started)
        self.elapsed_label.configure(
            text=f"Elapsed {seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
        )
        self.after(1000, self._tick_elapsed)

    def _finish_run(self, stage_text):
        self.progress.stop()
        self._run_started = None
        self.run_button.configure(state="normal", text="Run validation")
        self.cancel_button.configure(state="disabled", text="Cancel")
        self.stage_label.configure(text=stage_text)

    def choose_licences(self):
        path = self.vars["licence_layer"].get().strip()
        if not path:
            messagebox.showerror(APP_NAME, "Select the licence polygons first.", parent=self)
            return
        try:
            values = licence_choices(
                path,
                self.vars["licence_layer_name"].get().strip(),
                self.vars["licence_field"].get().strip(),
            )
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self)
            return
        dialog = tk.Toplevel(self)
        dialog.title("Choose licences to run")
        dialog.transient(self)
        ttk.Label(dialog, text="Select one or more licences (Ctrl/Shift-click). None selected = all.").pack(padx=12, pady=(12, 6), anchor="w")
        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=12)
        listbox = tk.Listbox(frame, selectmode="extended", height=min(20, max(5, len(values))), exportselection=False)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        current = set(selected_licences(Settings(difference_raster="", calibration_grid="", licence_layer="", ems_source="", output_folder="", licence_filter=self.vars["licence_filter"].get())))
        for position, value in enumerate(values):
            listbox.insert("end", value)
            if value in current:
                listbox.selection_set(position)
        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", padx=12, pady=12)

        def apply():
            chosen = [values[i] for i in listbox.curselection()]
            self.vars["licence_filter"].set("" if len(chosen) in (0, len(values)) else ", ".join(chosen))
            dialog.destroy()

        ttk.Button(buttons, text="Select all", command=lambda: listbox.selection_set(0, "end")).pack(side="left")
        ttk.Button(buttons, text="Clear", command=lambda: listbox.selection_clear(0, "end")).pack(side="left", padx=6)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="OK", command=apply).pack(side="right", padx=6)
        dialog.grab_set()

    def show_last_results(self):
        if self._last_summary is None:
            messagebox.showinfo(APP_NAME, "No results yet in this session.", parent=self)
            return
        self._show_results(self._last_summary)

    def _show_results(self, summary):
        window = tk.Toplevel(self)
        window.title("Validation results")
        window.geometry("1100x380")
        columns = [
            ("Resolution", "Resolution", 120),
            ("LicenceNo", "Licence", 80),
            ("ValidatedRate_tph", "Validated t/h", 100),
            ("UncorrectedRate_tph", "No background t/h", 120),
            ("RegressionRate_tph", "Regression t/h", 100),
            ("Background_m", "Background m", 95),
            ("BackgroundCells", "Bkg cells", 70),
            ("AcceptedCells", "Cells used", 75),
            ("AcceptedEMS_Hours", "EMS hours", 80),
            ("QA_Status", "vs configured", 105),
        ]
        columns = [item for item in columns if item[0] in summary.columns]
        if self._warning_count:
            ttk.Label(
                window,
                text=f"{self._warning_count} warning(s) in the processing log - review them before using these rates.",
                foreground="#9a5b00",
            ).pack(anchor="w", padx=10, pady=(10, 0))
        tree = ttk.Treeview(window, columns=[c[0] for c in columns], show="headings", height=10)
        for key, heading, width in columns:
            tree.heading(key, text=heading)
            tree.column(key, width=width, anchor="e" if key not in {"Resolution", "LicenceNo", "QA_Status"} else "w")
        colours = {"GREEN": "#d9f2d9", "AMBER": "#fff0c2", "RED": "#f8d0d0", "INSUFFICIENT_DATA": "#e6e6e6"}
        for status, colour in colours.items():
            tree.tag_configure(status, background=colour)
        for row in summary.to_dict(orient="records"):
            tree.insert("", "end", values=[_format_result_value(key, row.get(key)) for key, _, _ in columns], tags=(str(row.get("QA_Status", "")),))
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(
            window,
            text="'vs configured' compares the validated rate with the configured EMS rate. "
            "Validated and regression rates close together suggest a reliable background correction.",
            wraplength=1060,
        ).pack(anchor="w", padx=10)
        buttons = ttk.Frame(window)
        buttons.pack(fill="x", padx=10, pady=10)
        output = Path(self._last_output or ".")
        ttk.Button(buttons, text="Open output folder", command=lambda: _open_path(output)).pack(side="left")
        ttk.Button(buttons, text="Open summary CSV", command=lambda: _open_newest(output, "lowering_factor_validation_summary*.csv")).pack(side="left", padx=6)
        ttk.Button(buttons, text="Open run log", command=lambda: _open_path(output / RUN_LOG_NAME)).pack(side="left")
        ttk.Button(buttons, text="Close", command=window.destroy).pack(side="right")

    def _show_database_unavailable_dialog(
        self, error: EMSDatabaseUnavailableError
    ) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("EMS database not found")
        dialog.transient(self)
        dialog.resizable(False, False)
        ttk.Label(
            dialog,
            text=str(error),
            justify="left",
            wraplength=540,
            padding=16,
        ).pack(fill="x")
        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill="x")

        def locate_existing() -> None:
            selected = filedialog.askopenfilename(
                parent=dialog,
                title="Locate existing EMS database",
                filetypes=[
                    ("SQLite databases", "*.sqlite3 *.sqlite *.db"),
                    ("All files", "*.*"),
                ],
            )
            if not selected:
                return
            try:
                info = validate_database(Path(selected))
                set_configured_database_path(info.path)
            except (OSError, sqlite3.DatabaseError, ValueError) as exc:
                messagebox.showerror(APP_NAME, str(exc), parent=dialog)
                return
            messagebox.showinfo(
                APP_NAME,
                "The verified EMS database location has been updated. "
                "Run validation again to continue.",
                parent=dialog,
            )
            dialog.destroy()

        def open_manager() -> None:
            manager = Path(__file__).with_name("ems_database_manager.py")
            try:
                subprocess.Popen([sys.executable, str(manager)])
            except OSError as exc:
                messagebox.showerror(APP_NAME, str(exc), parent=dialog)
                return
            dialog.destroy()

        ttk.Button(
            buttons,
            text="Locate Existing Database",
            command=locate_existing,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            buttons,
            text="Open EMS Database Manager",
            command=open_manager,
        ).pack(side="left", padx=6)
        ttk.Button(
            buttons, text="Cancel", command=dialog.destroy
        ).pack(side="right", padx=(6, 0))
        dialog.grab_set()
        dialog.wait_window()

    def _poll_run_messages(self):
        if self._run_messages is None:
            return
        finished = False
        while True:
            try:
                message = self._run_messages.get_nowait()
            except queue.Empty:
                break

            kind = message[0]
            if kind == "log":
                text = message[1]
                self.log(text)
                if "WARNING" in text or text.startswith("PRE-RUN CHECK"):
                    self._warning_count += 1
                stage = run_stage(text)
                if stage:
                    self.stage_label.configure(text=stage)
            elif kind == "confirm":
                _, warnings, reply, done = message
                reply["ok"] = messagebox.askyesno(
                    "Pre-run checks",
                    "The pre-run checks found:\n\n"
                    + "\n\n".join(f"- {warning}" for warning in warnings)
                    + "\n\nContinue with the run anyway?",
                    parent=self,
                )
                done.set()
            elif kind == "cancelled":
                self.log(message[1])
                self._finish_run(message[1])
                finished = True
            elif kind == "error":
                _, exc, traceback_text = message
                self.log("ERROR: " + str(exc))
                self.log(traceback_text)
                self._finish_run("Stopped with an error - see the log.")
                if isinstance(exc, EMSDatabaseUnavailableError):
                    self._show_database_unavailable_dialog(exc)
                else:
                    messagebox.showerror(APP_NAME, str(exc))
                finished = True
            elif kind == "complete":
                summary = message[1]
                display_columns = ["LicenceNo", "ValidatedRate_tph", "QA_Status"]
                if "Resolution" in summary.columns:
                    display_columns.insert(0, "Resolution")
                self._finish_run("Complete.")
                self._last_summary = summary
                self._show_results(summary)
                finished = True

        if finished:
            self._run_messages = None
            self._run_thread = None
        elif self._run_thread is not None and self._run_thread.is_alive():
            self.after(100, self._poll_run_messages)
        else:
            self._run_messages = None
            self._run_thread = None
            self._finish_run("Ready.")

    def save_settings(self):
        try:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path:
                settings = self.settings()
                settings.validate()
                Path(path).write_text(json.dumps(settings.as_dict(), indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def load_settings(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = settings_from_file(path)
            for key, value in data.as_dict().items():
                if key in self.vars:
                    self.vars[key].set(str(value))
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))


def _run_validation_worker(settings: Settings, messages, cancel_event=None) -> None:
    def confirm(warnings: list[str]) -> bool:
        reply = {"ok": False}
        done = threading.Event()
        messages.put(("confirm", warnings, reply, done))
        while not done.wait(0.2):
            if cancel_event is not None and cancel_event.is_set():
                return False
        return bool(reply["ok"])

    try:
        summary = run_validation(
            settings,
            lambda text: messages.put(("log", text)),
            cancel_event=cancel_event,
            confirm=confirm,
        )
    except RunCancelled as exc:
        messages.put(("cancelled", str(exc)))
    except Exception as exc:
        if cancel_event is not None and cancel_event.is_set():
            messages.put(("cancelled", "Run cancelled by user."))
        else:
            messages.put(("error", exc, traceback.format_exc()))
    else:
        messages.put(("complete", summary))


def main(argv=None):
    settings = parse_cli_settings(argv)
    settings.validate()
    summary = run_validation(settings)
    print(summary[["LicenceNo", "ValidatedRate_tph", "QA_Status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    App().mainloop()
