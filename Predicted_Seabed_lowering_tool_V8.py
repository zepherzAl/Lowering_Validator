import sys
import threading
from pathlib import Path
import math
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time

import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from ems_toolkit_common import SettingsStore, run_folder

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ems_database import EMSDatabase, EMSRecord, discover_files
from resolution_grid import create_aligned_child_grid, inspect_master_grid
from lowering_model import (
    assign_points_to_areas,
    load_lowering_model,
    lowering_model_summary,
    validate_lowering_model,
)

# Version 8 introduces multi-area lowering with a master Lowering Model
# GeoPackage, automatic area detection, area-specific factors, combined 250 m /
# 50 m grids, pre-run model validation, and combined outputs/processing summary.
# The v7 single-grid workflow remains available as a legacy entry point.

APP_TITLE = "Predicted Seabed Lowering Tool v8\n Tool by Alex Peacock"
APP_FOOTER = "Predicted Seabed Lowering Tool v8 | A programme by Alex Peacock"
ICON_ICO = "lowering.ico"
ICON_PNG = "lowering_icon.png"
CEMEX_LOGO = "cemex_logo.png"
CEMEX_BLUE = "#0000BB"
CEMEX_LIGHT_BLUE = "#EAF2FF"

DATE_COL = "DATE"
TIME_COL = "TIME"
LAT_COL = "GPSLAT"
LON_COL = "GPSLON"
DREDGE_COL = "DREDGE"
INPUT_CRS = "EPSG:4326"
RESOLUTION_250 = "250 m Regulatory"
RESOLUTION_50 = "50 m Internal Detail"
RESOLUTION_BOTH = "Both"
VOLUME_QC_TOLERANCE_M3 = 1e-6


def resource_path(relative_path: str) -> str:
    """
    Returns the correct path for resources when running as a .py file
    and when bundled by PyInstaller.
    """
    try:
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    except Exception:
        base_path = Path(__file__).resolve().parent

    return str(base_path / relative_path)


# ==================================================
# CORE EMS PROCESSING FUNCTIONS
# ==================================================

def _iter_lowerer_ems_records(file_path: Path):
    """Parse the Lowerer's fixed EMS format into shared normalized records."""
    df = pd.read_csv(file_path, dtype=str)
    df.columns = [str(column).strip() for column in df.columns]
    required = [DATE_COL, TIME_COL, LAT_COL, LON_COL, DREDGE_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    timestamps = pd.to_datetime(
        df[DATE_COL].str.strip() + " " + df[TIME_COL].str.strip(),
        errors="coerce",
        dayfirst=True,
    )
    latitudes = pd.to_numeric(df[LAT_COL], errors="coerce")
    longitudes = pd.to_numeric(df[LON_COL], errors="coerce")
    for timestamp, latitude, longitude, dredge in zip(
        timestamps, latitudes, longitudes, df[DREDGE_COL]
    ):
        yield EMSRecord(
            timestamp=None if pd.isna(timestamp) else timestamp.isoformat(sep=" ", timespec="microseconds"),
            x=None if pd.isna(longitude) else float(longitude),
            y=None if pd.isna(latitude) else float(latitude),
            dredge=None if pd.isna(dredge) else str(dredge),
            vessel=file_path.name,
            crs=INPUT_CRS,
        )


def sample_to_interval(df: pd.DataFrame, sample_minutes: int) -> pd.DataFrame:
    """Reduce EMS records to one point per selected interval."""
    df = df.sort_values(["source_file", "datetime"]).copy()
    df["sample_bin"] = df["datetime"].dt.floor(f"{sample_minutes}min")
    sampled = df.groupby(["source_file", "sample_bin"], as_index=False).first()
    sampled = sampled.drop(columns=["sample_bin"])
    return sampled


def process_all_tracks(
    input_folder: Path,
    dredge_value: float,
    dredge_mode: str,
    sample_minutes: int,
    start_date=None,
    end_date=None,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
    spatial_bounds=None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int, int, int]:
    """Synchronize the shared index and retain full and area-candidate samples."""
    profile = "lowerer:DATE,TIME,GPSLAT,GPSLON,DREDGE:v1"
    files = discover_files(input_folder, (".txt", ".csv"))
    database = EMSDatabase(input_folder)
    database.sync_files(
        files,
        profile,
        _iter_lowerer_ems_records,
        log=log_callback,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )

    total_files = database.source_file_count(profile)
    if total_files == 0:
        raise FileNotFoundError(f"No valid EMS CSV or TXT files found in: {input_folder}")

    def boundary(value):
        if value is None:
            return None
        return pd.Timestamp(value).isoformat(sep=" ", timespec="microseconds")

    start_text = boundary(start_date)
    end_text = boundary(end_date)
    database_dredge_mode = "equals" if dredge_mode == "equals" else "greater_or_equal"
    total_raw_records = database.record_count(profile)
    total_date_filtered_records = database.count_dated_records(profile, start_text, end_text)
    records = database.query_records(
        profile,
        start=start_text,
        end=end_text,
        dredge_value=dredge_value,
        dredge_mode=database_dredge_mode,
        coordinates_only=True,
    )
    candidate_records = (
        database.query_records(
            profile,
            start=start_text,
            end=end_text,
            bounds_by_crs={INPUT_CRS: spatial_bounds},
            dredge_value=dredge_value,
            dredge_mode=database_dredge_mode,
            coordinates_only=True,
        )
        if spatial_bounds is not None
        else records
    )
    candidate_keys = {
        (record["source_path"], record["row_number"])
        for record in candidate_records
    }

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Processing cancelled by user.")

    df = pd.DataFrame.from_records(
        records,
        columns=[
            "source_path",
            "row_number",
            "timestamp",
            "x",
            "y",
            "dredge",
            "vessel",
            "crs",
        ],
    )
    df["source_file"] = df["source_path"].map(lambda value: Path(str(value)).name)
    df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df[LAT_COL] = pd.to_numeric(df["y"], errors="coerce")
    df[LON_COL] = pd.to_numeric(df["x"], errors="coerce")
    df[DREDGE_COL] = pd.to_numeric(df["dredge"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    if start_date is not None:
        df = df[df["datetime"] >= start_date]
    if end_date is not None:
        df = df[df["datetime"] <= end_date]
    if dredge_mode == "equals":
        df = df[df[DREDGE_COL] == dredge_value]
    else:
        df = df[df[DREDGE_COL] >= dredge_value]
    cleaned = df.dropna(subset=[LAT_COL, LON_COL]).copy()
    total_dredge_records = len(cleaned)

    sampled = sample_to_interval(cleaned, sample_minutes=sample_minutes)
    sampled_count = len(sampled)
    sample_keys = list(zip(sampled["source_path"], sampled["row_number"]))
    candidate_mask = [key in candidate_keys for key in sample_keys]
    candidates = sampled.loc[candidate_mask].copy()
    output_columns = ["source_file", "datetime", LAT_COL, LON_COL, DREDGE_COL]
    combined = sampled[output_columns].copy()
    candidates = candidates[output_columns].copy()

    if log_callback:
        log_callback(
            f"Indexed records: {total_raw_records:,} raw | "
            f"{total_date_filtered_records:,} within date range | "
            f"{total_dredge_records:,} dredge records | "
            f"{sampled_count:,} sampled records"
        )
    return (
        combined,
        candidates,
        total_raw_records,
        total_date_filtered_records,
        total_dredge_records,
        sampled_count,
        total_files
    )


def create_points_layer(points_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert cleaned EMS records to WGS84 point geometry."""
    return gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df[LON_COL], points_df[LAT_COL]),
        crs=INPUT_CRS
    )


def calculate_lowering_factor(
    tonnes_per_hour: float,
    bulk_density_t_per_m3: float,
    grid_size_m: float
) -> tuple[float, float, float]:
    """
    Calculate lowering factor automatically.

    m3/hour = tonnes/hour / bulk density
    grid area = grid size x grid size
    lowering factor = m3/hour / grid area
    """
    if tonnes_per_hour < 0:
        raise ValueError("Average dredging rate cannot be negative.")
    if bulk_density_t_per_m3 <= 0:
        raise ValueError("Bulk density must be greater than zero.")
    if grid_size_m <= 0:
        raise ValueError("Grid size must be greater than zero.")

    volume_rate_m3_hr = tonnes_per_hour / bulk_density_t_per_m3
    grid_area_m2 = grid_size_m * grid_size_m
    lowering_factor_m_hr = volume_rate_m3_hr / grid_area_m2
    return lowering_factor_m_hr, volume_rate_m3_hr, grid_area_m2


def spatial_join_points_to_grid(
    points_gdf: gpd.GeoDataFrame,
    grid_path: Path,
    grid_id_col: str,
    sample_minutes: int,
    spatial_predicate: str,
    create_grid_id_if_missing: bool,
    lowering_factor: float,
    volume_rate_m3_hr: float,
    grid_area_m2: float,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
    grid_gdf: gpd.GeoDataFrame | None = None,
    candidate_points_gdf: gpd.GeoDataFrame | None = None,
):
    """
    Assign sampled dredging points to grid cells, then calculate dredging time,
    volume removed and predicted lowering.
    """
    if log_callback:
        log_callback("Reading grid layer...")

    grid = grid_gdf if grid_gdf is not None else gpd.read_file(grid_path)

    if grid.empty:
        raise RuntimeError("The grid layer is empty.")
    if not grid.crs:
        raise RuntimeError("The grid layer does not have a recognised coordinate reference system.")

    if grid_id_col not in grid.columns:
        if create_grid_id_if_missing:
            if log_callback:
                log_callback(f"Grid ID field '{grid_id_col}' not found. Creating grid_id from row number.")
            grid_id_col = "grid_id"
            grid[grid_id_col] = range(1, len(grid) + 1)
        else:
            raise ValueError(f"Grid ID field '{grid_id_col}' not found. Available fields are: {list(grid.columns)}")

    if progress_callback:
        progress_callback(50)

    if log_callback:
        log_callback("Reprojecting area-filtered points to grid CRS...")
    points_to_join = candidate_points_gdf if candidate_points_gdf is not None else points_gdf
    points_projected = points_to_join.to_crs(grid.crs)

    if progress_callback:
        progress_callback(60)

    if log_callback:
        log_callback(f"Running spatial join using predicate: {spatial_predicate}")

    joined = gpd.sjoin(
        points_projected,
        grid[[grid_id_col, "geometry"]],
        how="left",
        predicate=spatial_predicate
    )

    omitted_points_count = len(points_gdf) - len(points_projected)
    outside_points_count = int(joined[grid_id_col].isna().sum()) + omitted_points_count
    inside_points_count = int(joined[grid_id_col].notna().sum())
    total_points_count = len(joined) + omitted_points_count

    if log_callback:
        log_callback(f"Sampled points inside grid: {inside_points_count:,}")
        log_callback(f"Sampled points outside grid: {outside_points_count:,}")
        if total_points_count > 0:
            coverage = (inside_points_count / total_points_count) * 100
            log_callback(f"Grid coverage: {coverage:.1f}%")

    if progress_callback:
        progress_callback(75)

    if log_callback:
        log_callback("Counting points per grid cell...")

    # This summary initially contains only cells that received at least one point.
    active_cell_summary = (
        joined
        .dropna(subset=[grid_id_col])
        .groupby(grid_id_col)
        .size()
        .reset_index(name="point_count")
    )

    active_cell_summary["time_minutes"] = active_cell_summary["point_count"] * sample_minutes
    active_cell_summary["time_hours"] = active_cell_summary["time_minutes"] / 60
    active_cell_summary["volume_m3"] = (active_cell_summary["time_hours"] * volume_rate_m3_hr).round(2)
    active_cell_summary["lowering_m"] = (active_cell_summary["time_hours"] * lowering_factor).round(2)

    # IMPORTANT:
    # Always merge the active-cell summary back onto the full input grid using a left
    # join from the grid. This keeps grid cells with no dredging points and fills
    # their values with zero. This is essential for cumulative quarterly updates.
    grid_out = grid.merge(active_cell_summary, on=grid_id_col, how="left")

    grid_out["point_count"] = grid_out["point_count"].fillna(0).astype(int)
    grid_out["time_minutes"] = grid_out["time_minutes"].fillna(0).round(2)
    grid_out["time_hours"] = grid_out["time_hours"].fillna(0).round(2)
    grid_out["volume_m3"] = grid_out["volume_m3"].fillna(0).round(2)
    grid_out["lowering_m"] = grid_out["lowering_m"].fillna(0).round(2)
    grid_out["grid_area_m2"] = round(grid_area_m2, 2)
    grid_out["lower_fac"] = round(lowering_factor, 6)

    # The CSV summary should also contain every grid cell, not just active cells.
    geometry_col = grid_out.geometry.name
    summary = pd.DataFrame(grid_out.drop(columns=[geometry_col]))

    if log_callback:
        total_grid_cells = len(grid_out)
        active_grid_cells = int((grid_out["point_count"] > 0).sum())
        inactive_grid_cells = int((grid_out["point_count"] == 0).sum())
        log_callback(f"Grid cells retained in output: {total_grid_cells:,}")
        log_callback(f"Grid cells with dredging points: {active_grid_cells:,}")
        log_callback(f"Grid cells without dredging points: {inactive_grid_cells:,}")

    if progress_callback:
        progress_callback(85)

    return grid_out, summary, joined, grid_id_col


def volume_conservation_qc(
    parent_grid: gpd.GeoDataFrame,
    child_grid: gpd.GeoDataFrame,
    grid_id_col: str,
    sample_minutes: int,
    volume_rate_m3_hr: float,
) -> pd.DataFrame:
    """Compare unrounded volume implied by lowering depth and nominal cell area."""
    if "ParentGridID" not in child_grid.columns:
        raise ValueError("The detailed grid is missing its parent-cell identifiers.")
    if "grid_area_m2" not in parent_grid or "grid_area_m2" not in child_grid:
        raise ValueError("Grid outputs are missing the cell areas required for volume QC.")

    parent_area = float(parent_grid["grid_area_m2"].iloc[0])
    child_area = float(child_grid["grid_area_m2"].iloc[0])
    interval_hours = sample_minutes / 60.0
    parent = pd.DataFrame({
        "parent_id": parent_grid[grid_id_col].astype(str),
        "parent_count": parent_grid["point_count"].astype(float),
    })
    children = pd.DataFrame({
        "parent_id": child_grid["ParentGridID"].astype(str),
        "child_count": child_grid["point_count"].astype(float),
    })
    child_counts = children.groupby("parent_id", as_index=False)["child_count"].sum()
    comparison = parent.merge(child_counts, on="parent_id", how="left")
    comparison["child_count"] = comparison["child_count"].fillna(0.0)

    parent_lowering = (
        comparison["parent_count"] * interval_hours * volume_rate_m3_hr / parent_area
    )
    child_lowering = (
        comparison["child_count"] * interval_hours * volume_rate_m3_hr / child_area
    )
    comparison["volume_250m_m3"] = parent_lowering * parent_area
    comparison["volume_50m_m3"] = child_lowering * child_area
    comparison["absolute_difference_m3"] = (
        comparison["volume_50m_m3"] - comparison["volume_250m_m3"]
    ).abs()
    comparison["difference_percent"] = comparison.apply(
        lambda row: (
            row["absolute_difference_m3"] / abs(row["volume_250m_m3"]) * 100.0
            if row["volume_250m_m3"] != 0
            else (0.0 if row["absolute_difference_m3"] == 0 else float("inf"))
        ),
        axis=1,
    )
    comparison["outside_tolerance"] = (
        comparison["absolute_difference_m3"] > VOLUME_QC_TOLERANCE_M3
    )

    total_250 = float(comparison["volume_250m_m3"].sum())
    total_50 = float(comparison["volume_50m_m3"].sum())
    total_difference = abs(total_50 - total_250)
    total_percent = total_difference / abs(total_250) * 100.0 if total_250 else (
        0.0 if total_difference == 0 else float("inf")
    )
    overall = pd.DataFrame([{
        "scope": "area_total",
        "parent_id": "",
        "volume_250m_m3": total_250,
        "volume_50m_m3": total_50,
        "absolute_difference_m3": total_difference,
        "difference_percent": total_percent,
        "outside_tolerance": total_difference > VOLUME_QC_TOLERANCE_M3,
        "parent_cells_checked": int(len(comparison)),
        "parent_cells_outside_tolerance": int(comparison["outside_tolerance"].sum()),
        "maximum_parent_difference_m3": float(comparison["absolute_difference_m3"].max())
        if not comparison.empty else 0.0,
        "tolerance_m3": VOLUME_QC_TOLERANCE_M3,
    }])
    per_parent = comparison.rename(columns={"parent_id": "parent_id"}).assign(
        scope="parent_cell",
        parent_cells_checked=1,
        parent_cells_outside_tolerance=comparison["outside_tolerance"].astype(int),
        maximum_parent_difference_m3=comparison["absolute_difference_m3"],
        tolerance_m3=VOLUME_QC_TOLERANCE_M3,
    )
    return pd.concat([overall, per_parent], ignore_index=True, sort=False)



def update_cumulative_grid(
    existing_grid_path: Path,
    quarter_grid: gpd.GeoDataFrame,
    grid_id_col: str,
    lowering_factor: float,
    grid_area_m2: float,
    log_callback=None
) -> gpd.GeoDataFrame:
    """
    Adds the new quarter's results onto an existing cumulative lowering grid.

    The selected existing grid must contain the same grid ID field used by the
    current run. If the existing grid does not already contain the cumulative
    fields, they are created and initialised to zero.
    """
    if log_callback:
        log_callback("Reading existing cumulative grid...")

    existing_grid = gpd.read_file(existing_grid_path)

    if existing_grid.empty:
        raise RuntimeError("The existing cumulative grid is empty.")

    if grid_id_col not in existing_grid.columns:
        raise ValueError(
            f"Grid ID field '{grid_id_col}' was not found in the existing cumulative grid. "
            f"Available fields are: {list(existing_grid.columns)}"
        )

    if grid_id_col not in quarter_grid.columns:
        raise ValueError(
            f"Grid ID field '{grid_id_col}' was not found in the current quarter grid."
        )

    update_cols = [
        "point_count",
        "time_minutes",
        "time_hours",
        "volume_m3",
        "lowering_m"
    ]

    for col in update_cols:
        if col not in existing_grid.columns:
            existing_grid[col] = 0
        if col not in quarter_grid.columns:
            quarter_grid[col] = 0

    existing_cell_count = len(existing_grid)
    quarter_cell_count = len(quarter_grid)

    quarter_values = quarter_grid[[grid_id_col] + update_cols].copy()
    quarter_values = quarter_values.rename(
        columns={col: f"{col}_quarter" for col in update_cols}
    )

    # Keep a copy of quarter geometry so an outer merge can preserve any cells that
    # are present in the current full quarter grid but missing from the existing
    # cumulative grid.
    quarter_geometry_lookup = pd.DataFrame({
        grid_id_col: quarter_grid[grid_id_col],
        "quarter_geometry": quarter_grid.geometry
    })

    cumulative_grid = existing_grid.merge(
        quarter_values,
        on=grid_id_col,
        how="outer"
    )

    cumulative_grid = cumulative_grid.merge(
        quarter_geometry_lookup,
        on=grid_id_col,
        how="left"
    )

    # If the existing cumulative grid was missing a cell, fill its geometry from
    # the current quarter grid.
    cumulative_grid["geometry"] = cumulative_grid["geometry"].where(
        cumulative_grid["geometry"].notna(),
        cumulative_grid["quarter_geometry"]
    )
    cumulative_grid = cumulative_grid.drop(columns=["quarter_geometry"])
    cumulative_grid = gpd.GeoDataFrame(cumulative_grid, geometry="geometry", crs=existing_grid.crs or quarter_grid.crs)

    for col in update_cols:
        quarter_col = f"{col}_quarter"
        cumulative_grid[col] = cumulative_grid[col].fillna(0) + cumulative_grid[quarter_col].fillna(0)
        cumulative_grid = cumulative_grid.drop(columns=[quarter_col])

    cumulative_grid["point_count"] = cumulative_grid["point_count"].round(0).astype(int)
    cumulative_grid["time_minutes"] = cumulative_grid["time_minutes"].round(2)
    cumulative_grid["time_hours"] = cumulative_grid["time_hours"].round(2)
    cumulative_grid["volume_m3"] = cumulative_grid["volume_m3"].round(2)
    cumulative_grid["lowering_m"] = cumulative_grid["lowering_m"].round(2)
    cumulative_grid["grid_area_m2"] = round(grid_area_m2, 2)
    cumulative_grid["lower_fac"] = round(lowering_factor, 6)

    if log_callback:
        log_callback(f"Existing cumulative grid cells: {existing_cell_count:,}")
        log_callback(f"Current quarter grid cells: {quarter_cell_count:,}")
        log_callback(f"Updated cumulative grid cells: {len(cumulative_grid):,}")
        log_callback("Existing cumulative grid updated with current quarter results.")

    return cumulative_grid


def update_multi_area_cumulative_grid(
    existing_grid_path: Path,
    quarter_grid: gpd.GeoDataFrame,
    log_callback=None,
) -> gpd.GeoDataFrame:
    """Accumulate quarter results by the composite area_id/grid_id key."""
    existing = gpd.read_file(existing_grid_path)
    required = {"area_id", "grid_id", "geometry"}
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(
            "The existing cumulative grid must contain area_id and grid_id fields; "
            f"missing: {', '.join(sorted(missing))}."
        )
    if existing.crs is None or quarter_grid.crs is None:
        raise ValueError("Both cumulative and quarter grids must have a defined CRS.")
    if existing[["area_id", "grid_id"]].isna().any().any():
        raise ValueError("Existing cumulative area_id/grid_id keys must not be null.")
    if quarter_grid[["area_id", "grid_id"]].isna().any().any():
        raise ValueError("Quarter area_id/grid_id keys must not be null.")

    existing_keys = existing["area_id"].astype(str) + "\x1f" + existing["grid_id"].astype(str)
    quarter = quarter_grid.to_crs(existing.crs).copy()
    quarter_keys = quarter["area_id"].astype(str) + "\x1f" + quarter["grid_id"].astype(str)
    if existing_keys.duplicated().any():
        raise ValueError("Existing cumulative area_id/grid_id keys must be unique.")
    if quarter_keys.duplicated().any():
        raise ValueError("Quarter area_id/grid_id keys must be unique.")

    update_columns = [
        "point_count", "time_minutes", "time_hours", "volume_m3", "lowering_m"
    ]
    for column in update_columns:
        if column not in existing.columns:
            existing[column] = 0
        if column not in quarter.columns:
            quarter[column] = 0

    existing_data = pd.DataFrame(existing.drop(columns=[existing.geometry.name]))
    existing_data["_area_grid_key"] = existing_keys.to_numpy()
    quarter_data = pd.DataFrame(quarter.drop(columns=[quarter.geometry.name]))
    quarter_data["_area_grid_key"] = quarter_keys.to_numpy()
    metadata_columns = [
        column for column in (
            "area_id", "grid_id", "area_name", "lowering_factor",
            "calibrated_lowering_factor", "grid_area_m2", "lower_fac",
        )
        if column in quarter_data.columns
    ]
    quarter_values = quarter_data[
        ["_area_grid_key"] + update_columns + metadata_columns
    ].copy()
    rename_columns = {
        column: f"{column}_quarter"
        for column in update_columns + metadata_columns
    }
    quarter_values = quarter_values.rename(columns=rename_columns)
    cumulative = existing_data.merge(
        quarter_values, on="_area_grid_key", how="outer"
    )

    existing_geometry = pd.DataFrame({
        "_area_grid_key": existing_keys.to_numpy(),
        "_existing_geometry": existing.geometry.to_numpy(),
    })
    quarter_geometry = pd.DataFrame({
        "_area_grid_key": quarter_keys.to_numpy(),
        "_quarter_geometry": quarter.geometry.to_numpy(),
    })
    cumulative = cumulative.merge(existing_geometry, on="_area_grid_key", how="left")
    cumulative = cumulative.merge(quarter_geometry, on="_area_grid_key", how="left")
    cumulative["geometry"] = cumulative["_existing_geometry"].where(
        cumulative["_existing_geometry"].notna(), cumulative["_quarter_geometry"]
    )
    for column in update_columns:
        quarter_column = f"{column}_quarter"
        cumulative[column] = (
            pd.to_numeric(cumulative[column], errors="coerce").fillna(0)
            + pd.to_numeric(cumulative[quarter_column], errors="coerce").fillna(0)
        )
        cumulative = cumulative.drop(columns=quarter_column)
    for column in metadata_columns:
        quarter_column = f"{column}_quarter"
        if quarter_column in cumulative.columns:
            if column not in cumulative.columns:
                cumulative[column] = cumulative[quarter_column]
            else:
                cumulative[column] = cumulative[quarter_column].where(
                    cumulative[quarter_column].notna(), cumulative[column]
                )
            cumulative = cumulative.drop(columns=quarter_column)

    cumulative = cumulative.drop(
        columns=["_area_grid_key", "_existing_geometry", "_quarter_geometry"]
    )
    cumulative["point_count"] = cumulative["point_count"].round(0).astype(int)
    for column in ("time_minutes", "time_hours", "volume_m3", "lowering_m"):
        cumulative[column] = cumulative[column].round(2)
    cumulative["calculated_lowering"] = cumulative["lowering_m"]
    result = gpd.GeoDataFrame(cumulative, geometry="geometry", crs=existing.crs)
    if log_callback:
        log_callback(f"Existing cumulative grid cells: {len(existing):,}")
        log_callback(f"Quarter grid cells: {len(quarter_grid):,}")
        log_callback(f"Updated cumulative grid cells: {len(result):,}")
    return result


def run_single_grid_lowering_workflow_legacy(
    input_folder: Path,
    grid_path: Path,
    output_folder: Path,
    grid_id_col: str,
    dredge_value: float,
    dredge_mode: str,
    sample_minutes: int,
    use_date_filter: bool,
    start_date,
    end_date,
    tonnes_per_hour: float,
    bulk_density: float,
    grid_size_m: float,
    spatial_predicate: str,
    create_grid_id_if_missing: bool,
    export_cleaned_csv: bool,
    export_points_gpkg: bool,
    export_grid_gpkg: bool,
    export_summary_csv: bool,
    update_existing_grid: bool = False,
    existing_grid_path: Path | None = None,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
    resolution_mode: str = RESOLUTION_250,
):
    """Full EMS lowering workflow."""
    start_time = time.time()
    output_folder = run_folder(output_folder, "EMS_Seabed_Lowering")

    if resolution_mode not in {RESOLUTION_250, RESOLUTION_50, RESOLUTION_BOTH}:
        raise ValueError(f"Unsupported output resolution: {resolution_mode}")
    if update_existing_grid and resolution_mode == RESOLUTION_50:
        raise ValueError("Cumulative-grid updates are available only for the 250 m regulatory grid.")
    if not math.isclose(grid_size_m, 250.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("The regulatory master grid size must be 250 m.")
    _, volume_rate_m3_hr, _ = calculate_lowering_factor(
        tonnes_per_hour,
        bulk_density,
        grid_size_m,
    )

    if log_callback:
        log_callback("Starting EMS lowering workflow...")
        log_callback(f"Requested outputs: {resolution_mode}")
        log_callback("The 250 m regulatory layer is the authoritative master grid.")

    if log_callback:
        log_callback("")
        log_callback("========== DATE FILTER ==========")
        if use_date_filter:
            log_callback(f"Start date: {start_date.strftime('%d/%m/%Y')}")
            log_callback(f"End date: {end_date.strftime('%d/%m/%Y')}")
            log_callback("Records outside this date range will be ignored.")
        else:
            log_callback("Date filter disabled. All valid EMS dates will be processed.")
        log_callback("=================================")
        log_callback("")

    grid_gdf = gpd.read_file(grid_path)
    if grid_gdf.empty:
        raise RuntimeError("The grid layer is empty.")
    if not grid_gdf.crs:
        raise RuntimeError("The grid layer does not have a recognised coordinate reference system.")
    if grid_id_col not in grid_gdf.columns:
        if not create_grid_id_if_missing:
            raise ValueError(
                f"Grid ID field '{grid_id_col}' not found. "
                f"Available fields are: {list(grid_gdf.columns)}"
            )
        if log_callback:
            log_callback(f"Grid ID field '{grid_id_col}' not found. Creating grid_id from row number.")
        grid_id_col = "grid_id"
        grid_gdf[grid_id_col] = range(1, len(grid_gdf) + 1)

    master_layout = inspect_master_grid(grid_gdf, grid_id_col, cell_size_m=250.0)
    resolution_grids = []
    if resolution_mode in {RESOLUTION_250, RESOLUTION_BOTH}:
        resolution_grids.append((RESOLUTION_250, 250.0, grid_gdf))
    if resolution_mode in {RESOLUTION_50, RESOLUTION_BOTH}:
        detailed_grid = create_aligned_child_grid(
            grid_gdf,
            master_layout,
            child_size_m=50.0,
        )
        resolution_grids.append((RESOLUTION_50, 50.0, detailed_grid))

    volume_rate_m3_hr = tonnes_per_hour / bulk_density
    if log_callback:
        log_callback(f"Validated {len(grid_gdf):,} 250 m master cells.")
        if resolution_mode in {RESOLUTION_50, RESOLUTION_BOTH}:
            log_callback(
                f"Generated {len(detailed_grid):,} aligned 50 m cells "
                "(25 per master cell before footprint masking)."
            )
    spatial_bounds = tuple(
        float(value)
        for value in Transformer.from_crs(
            grid_gdf.crs, INPUT_CRS, always_xy=True
        ).transform_bounds(*grid_gdf.total_bounds, densify_pts=21)
    )

    (
        cleaned_points_df,
        spatial_points_df,
        total_raw_records,
        total_date_filtered_records,
        total_dredge_records,
        total_sampled_records,
        total_files
    ) = process_all_tracks(
        input_folder=input_folder,
        dredge_value=dredge_value,
        dredge_mode=dredge_mode,
        sample_minutes=sample_minutes,
        start_date=start_date if use_date_filter else None,
        end_date=end_date if use_date_filter else None,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        spatial_bounds=spatial_bounds,
    )

    if log_callback:
        log_callback(f"Combined sampled records: {len(cleaned_points_df):,}")
        log_callback("Creating spatial point layer...")

    points_gdf = create_points_layer(cleaned_points_df)
    spatial_points_gdf = create_points_layer(spatial_points_df)

    if progress_callback:
        progress_callback(45)

    calculated_grids = {}
    for resolution_name, cell_size, target_grid in resolution_grids:
        lowering_factor, rate, cell_area = calculate_lowering_factor(
            tonnes_per_hour=tonnes_per_hour,
            bulk_density_t_per_m3=bulk_density,
            grid_size_m=cell_size,
        )
        if log_callback:
            log_callback("")
            log_callback(f"========== {resolution_name.upper()} CALCULATION ==========")
            log_callback(f"Average dredging rate: {tonnes_per_hour:.2f} tonnes/hour")
            log_callback(f"Bulk density: {bulk_density:.2f} tonnes/m³")
            log_callback(f"Volume rate: {rate:.2f} m³/hour")
            log_callback(f"Cell size: {cell_size:.2f} m; cell area: {cell_area:.2f} m²")
            log_callback(f"Lowering factor: {lowering_factor:.6f} m/hour")
            log_callback("Predicted lowering = dredging hours × lowering factor")

        grid_out, grid_summary, _joined, final_grid_id_col = spatial_join_points_to_grid(
            points_gdf=points_gdf,
            grid_path=grid_path,
            grid_id_col=grid_id_col,
            sample_minutes=sample_minutes,
            spatial_predicate=spatial_predicate,
            create_grid_id_if_missing=create_grid_id_if_missing,
            lowering_factor=lowering_factor,
            volume_rate_m3_hr=rate,
            grid_area_m2=cell_area,
            log_callback=log_callback,
            progress_callback=progress_callback,
            grid_gdf=target_grid,
            candidate_points_gdf=spatial_points_gdf,
        )
        calculated_grids[resolution_name] = {
            "grid": grid_out,
            "summary": grid_summary,
            "grid_id_col": final_grid_id_col,
            "lowering_factor": lowering_factor,
            "volume_rate_m3_hr": rate,
            "grid_area_m2": cell_area,
        }

    elapsed_seconds = time.time() - start_time
    elapsed_minutes = int(elapsed_seconds // 60)
    elapsed_remainder = elapsed_seconds % 60


    if log_callback:
        for resolution_name, result in calculated_grids.items():
            grid_out = result["grid"]
            hit_cells = int((grid_out["point_count"] > 0).sum())
            total_cells = len(grid_out)
            log_callback(
                f"{resolution_name}: {hit_cells:,} of {total_cells:,} cells "
                f"received dredging points; total volume {grid_out['volume_m3'].sum():.2f} m³."
            )
        log_callback("Writing outputs...")
        log_callback(
            f"Processing time: "
            f"{elapsed_minutes} min "
            f"{elapsed_remainder:.1f} sec"
        )
        log_callback("")
        log_callback("========== RESULTS SUMMARY ==========")

        log_callback(
            f"EMS files processed: "
            f"{total_files:,}"
        )

        log_callback(
            f"Raw records analysed: "
            f"{total_raw_records:,}"
        )

        log_callback(
            f"Records within date range: "
            f"{total_date_filtered_records:,}"
        )

        log_callback(
            f"Dredging records: "
            f"{total_dredge_records:,}"
        )

        log_callback(
            f"Sampled records retained: "
            f"{total_sampled_records:,}"
        )
        log_callback("=====================================")
        

    outputs_written = []

    if export_cleaned_csv:
        cleaned_csv_path = output_folder / "cleaned_sampled_dredge_points.csv"
        cleaned_points_df.to_csv(cleaned_csv_path, index=False)
        outputs_written.append(cleaned_csv_path)

    if export_points_gpkg:
        points_gpkg_path = output_folder / "cleaned_sampled_dredge_points.gpkg"
        points_gdf.to_file(points_gpkg_path, layer="cleaned_sampled_dredge_points", driver="GPKG")
        outputs_written.append(points_gpkg_path)

    multiple_resolutions = len(calculated_grids) > 1
    for resolution_name, result in calculated_grids.items():
        suffix = ""
        if multiple_resolutions:
            suffix = "_250m_regulatory" if resolution_name == RESOLUTION_250 else "_50m_internal"
        elif resolution_name == RESOLUTION_50:
            suffix = "_50m_internal"
        if export_grid_gpkg:
            grid_gpkg_path = output_folder / f"grid_dredge_time_lowering{suffix}.gpkg"
            result["grid"].to_file(
                grid_gpkg_path,
                layer="grid_dredge_time_lowering",
                driver="GPKG",
            )
            outputs_written.append(grid_gpkg_path)
        if export_summary_csv:
            summary_csv_path = output_folder / f"grid_dredge_time_lowering_summary{suffix}.csv"
            result["summary"].to_csv(summary_csv_path, index=False)
            outputs_written.append(summary_csv_path)

    if resolution_mode == RESOLUTION_BOTH:
        qc = volume_conservation_qc(
            calculated_grids[RESOLUTION_250]["grid"],
            calculated_grids[RESOLUTION_50]["grid"],
            calculated_grids[RESOLUTION_250]["grid_id_col"],
            sample_minutes,
            volume_rate_m3_hr,
        )
        qc_path = output_folder / "resolution_volume_conservation_qc.csv"
        qc.to_csv(qc_path, index=False)
        outputs_written.append(qc_path)
        overall = qc.loc[qc["scope"] == "area_total"].iloc[0]
        if log_callback:
            log_callback("")
            log_callback("========== CROSS-RESOLUTION VOLUME QC ==========")
            log_callback(f"250 m total removed volume: {overall['volume_250m_m3']:.6f} m³")
            log_callback(f"50 m total removed volume: {overall['volume_50m_m3']:.6f} m³")
            log_callback(f"Absolute difference: {overall['absolute_difference_m3']:.9f} m³")
            log_callback(f"Percentage difference: {overall['difference_percent']:.9f}%")
            log_callback(f"Parent cells checked: {int(overall['parent_cells_checked']):,}")
            log_callback(
                "Parent cells outside tolerance: "
                f"{int(overall['parent_cells_outside_tolerance']):,}"
            )
            log_callback(
                "Maximum parent-cell difference: "
                f"{overall['maximum_parent_difference_m3']:.9f} m³"
            )
            log_callback(f"Absolute tolerance: {VOLUME_QC_TOLERANCE_M3:g} m³")
            if bool(overall["outside_tolerance"]):
                log_callback(
                    "QC difference exceeds floating-point tolerance. Review boundary-assigned "
                    "EMS points, masked cells, date filters and coordinate transformations; "
                    "neither resolution has been adjusted."
                )

    if update_existing_grid:
        if existing_grid_path is None:
            raise ValueError("Update mode is selected, but no existing cumulative grid was supplied.")

        parent_result = calculated_grids[RESOLUTION_250]
        cumulative_grid = update_cumulative_grid(
            existing_grid_path=existing_grid_path,
            quarter_grid=parent_result["grid"],
            grid_id_col=parent_result["grid_id_col"],
            lowering_factor=parent_result["lowering_factor"],
            grid_area_m2=parent_result["grid_area_m2"],
            log_callback=log_callback
        )

        cumulative_gpkg_path = output_folder / "cumulative_grid_dredge_time_lowering_UPDATED.gpkg"
        cumulative_grid.to_file(
            cumulative_gpkg_path,
            layer="cumulative_grid_dredge_time_lowering_UPDATED",
            driver="GPKG"
        )
        outputs_written.append(cumulative_gpkg_path)

    if progress_callback:
        progress_callback(100)

    if log_callback:
        log_callback("Workflow complete.")
        log_callback("")
        log_callback("Outputs written:")
        for output in outputs_written:
            log_callback(f"  {output}")
        log_callback("")
        log_callback(f"Grid ID field used: {final_grid_id_col}")

    return outputs_written


def run_lowering_workflow(
    input_folder: Path,
    model_path: Path,
    output_folder: Path,
    dredge_value: float,
    dredge_mode: str,
    sample_minutes: int,
    use_date_filter: bool,
    start_date,
    end_date,
    spatial_predicate: str,
    export_cleaned_csv: bool,
    export_points_gpkg: bool,
    export_grid_gpkg: bool,
    export_summary_csv: bool,
    resolution_mode: str = RESOLUTION_250,
    processing_mode: str = "All affected areas",
    selected_area_id: str | None = None,
    update_existing_grid: bool = False,
    existing_grid_path: Path | None = None,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
):
    """Run the existing grid calculation for every uniquely matched model area."""
    start_time = time.time()
    output_folder = run_folder(output_folder, "EMS_Seabed_Lowering_Multi_Area")
    if resolution_mode not in {RESOLUTION_250, RESOLUTION_50, RESOLUTION_BOTH}:
        raise ValueError(f"Unsupported output resolution: {resolution_mode}")
    if processing_mode not in {"All affected areas", "Selected area"}:
        raise ValueError(f"Unsupported processing mode: {processing_mode}")
    if processing_mode == "Selected area" and not selected_area_id:
        raise ValueError("Select an area_id when using Selected area mode.")
    if update_existing_grid and resolution_mode == RESOLUTION_50:
        raise ValueError(
            "Cumulative-grid updates are available only when the 250 m regulatory output is selected."
        )
    if update_existing_grid and existing_grid_path is None:
        raise ValueError("Select an existing multi-area cumulative grid to update.")

    model = load_lowering_model(model_path)
    validation = validate_lowering_model(model)
    if log_callback:
        log_callback("Starting multi-area EMS lowering workflow...")
        log_callback(lowering_model_summary(model, validation))
        for warning in validation.warnings:
            log_callback(f"WARNING: {warning}")
        for error in validation.errors:
            log_callback(f"MODEL ERROR: {error}")
        for area_key, errors in sorted(validation.area_errors.items()):
            for error in errors:
                log_callback(f"AREA {area_key} ERROR: {error}")
        for area_key in sorted(validation.eligible_area_ids):
            area = model.areas.loc[model.areas["area_id"].astype(str).eq(area_key)].iloc[0]
            log_callback(
                f"Validated area {area_key} ({area['area_name']}), "
                f"lowering factor {float(area['lowering_factor']):.9g} m/hour."
            )

    if not validation.eligible_area_ids:
        raise ValueError("The Lowering Model has no active areas that pass validation.")
    if processing_mode == "Selected area" and selected_area_id not in validation.eligible_area_ids:
        raise ValueError(
            f"Selected area_id {selected_area_id!r} is not an active, valid area in the model."
        )

    generated_grids_50m = {}

    if log_callback:
        log_callback("")
        log_callback("========== DATE FILTER ==========")
        if use_date_filter:
            log_callback(f"Start date: {start_date.strftime('%d/%m/%Y')}")
            log_callback(f"End date: {end_date.strftime('%d/%m/%Y')}")
        else:
            log_callback("Date filter disabled. All valid EMS dates will be processed.")
        log_callback("=================================")

    (
        cleaned_points_df,
        _candidate_points_df,
        total_raw_records,
        total_date_filtered_records,
        total_dredge_records,
        total_sampled_records,
        total_files,
    ) = process_all_tracks(
        input_folder=input_folder,
        dredge_value=dredge_value,
        dredge_mode=dredge_mode,
        sample_minutes=sample_minutes,
        start_date=start_date if use_date_filter else None,
        end_date=end_date if use_date_filter else None,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Processing cancelled by user.")

    points_gdf = create_points_layer(cleaned_points_df)
    assigned_points = assign_points_to_areas(points_gdf, model, validation)
    assigned_points_out = assigned_points.to_crs(INPUT_CRS)
    ambiguous_count = int(assigned_points["area_assignment"].eq("ambiguous").sum())
    unmatched_count = int(assigned_points["area_assignment"].eq("unmatched").sum())
    if log_callback:
        log_callback(f"Sampled EMS records: {len(assigned_points):,}")
        log_callback(f"Unmatched records outside valid active areas: {unmatched_count:,}")
        log_callback(f"Ambiguous records intersecting multiple areas: {ambiguous_count:,}")

    area_rows = []
    grid_outputs = {RESOLUTION_250: [], RESOLUTION_50: []}
    grid_summaries = {RESOLUTION_250: [], RESOLUTION_50: []}
    areas_encountered = set(
        assigned_points.loc[
            assigned_points["area_assignment"].eq("matched"), "area_id"
        ].dropna().astype(str)
    )
    for candidates in assigned_points.loc[
        assigned_points["area_assignment"].eq("ambiguous"), "area_candidates"
    ]:
        areas_encountered.update(json.loads(candidates))
    area_filter = set(validation.eligible_area_ids)
    if processing_mode == "Selected area":
        area_filter &= {str(selected_area_id)}
    matched_area_ids = set(
        assigned_points.loc[
            assigned_points["area_assignment"].eq("matched"), "area_id"
        ].dropna().astype(str)
    )
    affected_area_ids = matched_area_ids & area_filter

    if resolution_mode in {RESOLUTION_50, RESOLUTION_BOTH}:
        for area_key in sorted(affected_area_ids):
            if model.grid_50m is None:
                area_grid = model.grid_250m.loc[
                    model.grid_250m["area_id"].astype(str).eq(area_key)
                ].copy()
                try:
                    layout = inspect_master_grid(
                        area_grid, "grid_id", cell_size_m=250.0
                    )
                    detailed = create_aligned_child_grid(
                        area_grid, layout, child_size_m=50.0
                    )
                    generated_grids_50m[area_key] = detailed.rename(
                        columns={"ParentGridID": "parent_250_id"}
                    )
                except (ValueError, RuntimeError) as exc:
                    validation.add_area_error(
                        area_key, f"Could not generate aligned 50 m grid: {exc}"
                    )
                    validation.eligible_area_ids.discard(area_key)
                    area_filter.discard(area_key)
                    if log_callback:
                        log_callback(
                            f"AREA {area_key} ERROR: could not generate 50 m grid: {exc}"
                        )
            else:
                subset = model.grid_50m.loc[
                    model.grid_50m["area_id"].astype(str).eq(area_key)
                ].copy()
                if subset.empty:
                    validation.add_area_error(
                        area_key, "Active area has no 50 m grid cells."
                    )
                    validation.eligible_area_ids.discard(area_key)
                    area_filter.discard(area_key)
                else:
                    subset["parent_250_id"] = subset["parent_250_id"].astype(str)
                    generated_grids_50m[area_key] = subset

    for area_key in sorted(area_filter & areas_encountered):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Processing cancelled by user.")
        area = model.areas.loc[model.areas["area_id"].astype(str).eq(area_key)].iloc[0]
        area_name = str(area["area_name"])
        calibrated_factor = float(area["lowering_factor"])
        area_points = assigned_points.loc[
            assigned_points["area_assignment"].eq("matched")
            & assigned_points["area_id"].astype(str).eq(area_key)
        ].copy()
        area_row = {
            "summary_scope": "area",
            "area_id": area["area_id"],
            "area_name": area_name,
            "lowering_factor": calibrated_factor,
            "areas_encountered": int(area_key in areas_encountered),
            "areas_successfully_processed": 0,
            "ems_records_tracks_processed": len(area_points),
            "ems_records_tracks_assigned": len(area_points),
            "unmatched_ems_records_tracks": 0,
            "ambiguous_boundary_records": sum(
                area_key in json.loads(candidates)
                for candidates in assigned_points.loc[
                    assigned_points["area_assignment"].eq("ambiguous"),
                    "area_candidates",
                ]
            ),
            "250m_grid_cells_affected": 0,
            "50m_grid_cells_affected": 0,
            "warnings_errors": "; ".join(validation.area_errors.get(area_key, [])),
        }
        if area_points.empty:
            area_row["warnings_errors"] = (
                area_row["warnings_errors"] + "; " if area_row["warnings_errors"] else ""
            ) + "No uniquely assigned sampled EMS records for this area."
            area_rows.append(area_row)
            continue

        area_grid = model.grid_250m.loc[
            model.grid_250m["area_id"].astype(str).eq(area_key)
        ].copy()
        resolutions = []
        if resolution_mode in {RESOLUTION_250, RESOLUTION_BOTH}:
            resolutions.append((RESOLUTION_250, 250.0, area_grid))
        if resolution_mode in {RESOLUTION_50, RESOLUTION_BOTH}:
            resolutions.append((RESOLUTION_50, 50.0, generated_grids_50m[area_key]))

        area_failed = False
        area_resolution_outputs = []
        for resolution_name, cell_size, target_grid in resolutions:
            factor = calibrated_factor * (250.0 / cell_size) ** 2
            cell_area = cell_size ** 2
            volume_rate = calibrated_factor * 250.0 ** 2
            area_points_in_grid_crs = area_points.to_crs(target_grid.crs)
            try:
                grid_out, grid_summary, _joined, _grid_id_col = spatial_join_points_to_grid(
                    points_gdf=area_points_in_grid_crs,
                    grid_path=model.path,
                    grid_id_col="grid_id",
                    sample_minutes=sample_minutes,
                    spatial_predicate=spatial_predicate,
                    create_grid_id_if_missing=False,
                    lowering_factor=factor,
                    volume_rate_m3_hr=volume_rate,
                    grid_area_m2=cell_area,
                    log_callback=log_callback,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                    grid_gdf=target_grid,
                    candidate_points_gdf=area_points_in_grid_crs,
                )
            except (ValueError, RuntimeError) as exc:
                validation.add_area_error(area_key, f"{resolution_name} calculation failed: {exc}")
                area_failed = True
                break

            grid_out["area_id"] = area["area_id"]
            grid_out["area_name"] = area_name
            grid_out["lowering_factor"] = factor
            grid_out["calibrated_lowering_factor"] = calibrated_factor
            grid_out["calculated_lowering"] = grid_out["lowering_m"]
            if resolution_name == RESOLUTION_50:
                if "parent_250_id" not in grid_out.columns:
                    if "ParentGridID" not in grid_out.columns:
                        raise ValueError(
                            f"Area {area_key} 50 m grid is missing parent_250_id."
                        )
                    grid_out = grid_out.rename(columns={"ParentGridID": "parent_250_id"})
                grid_out["parent_250_id"] = grid_out["parent_250_id"].astype(str)
                grid_summary = pd.DataFrame(grid_out.drop(columns=[grid_out.geometry.name]))
            else:
                grid_out["grid_id"] = grid_out["grid_id"]
                grid_summary = pd.DataFrame(grid_out.drop(columns=[grid_out.geometry.name]))
            area_resolution_outputs.append((resolution_name, grid_out, grid_summary))
            area_row[f"{'250m' if resolution_name == RESOLUTION_250 else '50m'}_grid_cells_affected"] = int(
                (grid_out["point_count"] > 0).sum()
            )

        if not area_failed:
            area_row["areas_successfully_processed"] = 1
            for resolution_name, grid_out, grid_summary in area_resolution_outputs:
                grid_outputs[resolution_name].append(grid_out)
                grid_summaries[resolution_name].append(grid_summary)
        else:
            area_row["250m_grid_cells_affected"] = 0
            area_row["50m_grid_cells_affected"] = 0
        area_row["warnings_errors"] = "; ".join(validation.area_errors.get(area_key, []))
        area_rows.append(area_row)

    for area_key in sorted(areas_encountered - area_filter):
        area_matches = model.areas.loc[model.areas["area_id"].astype(str).eq(area_key)]
        if area_matches.empty:
            continue
        area = area_matches.iloc[0]
        issues = validation.area_errors.get(area_key, [])
        if processing_mode == "Selected area" and area_key in validation.eligible_area_ids:
            issues = ["Not processed: Selected area mode targeted a different area_id."]
        area_rows.append({
            "summary_scope": "area",
            "area_id": area["area_id"],
            "area_name": str(area["area_name"]),
            "lowering_factor": area["lowering_factor"],
            "areas_encountered": 1,
            "areas_successfully_processed": 0,
            "ems_records_tracks_processed": 0,
            "ems_records_tracks_assigned": int(
                (
                    assigned_points["area_assignment"].eq("matched")
                    & assigned_points["area_id"].astype(str).eq(area_key)
                ).sum()
            ),
            "unmatched_ems_records_tracks": 0,
            "ambiguous_boundary_records": sum(
                area_key in json.loads(candidates)
                for candidates in assigned_points.loc[
                    assigned_points["area_assignment"].eq("ambiguous"),
                    "area_candidates",
                ]
            ),
            "250m_grid_cells_affected": 0,
            "50m_grid_cells_affected": 0,
            "warnings_errors": "; ".join(issues) or "Area is not available for processing.",
        })

    combined_grids = {}
    combined_summaries = {}
    for resolution_name in (RESOLUTION_250, RESOLUTION_50):
        if grid_outputs[resolution_name]:
            combined_grids[resolution_name] = gpd.GeoDataFrame(
                pd.concat(grid_outputs[resolution_name], ignore_index=True),
                geometry="geometry",
                crs=grid_outputs[resolution_name][0].crs,
            )
            combined_summaries[resolution_name] = pd.concat(
                grid_summaries[resolution_name], ignore_index=True
            )

    summary_rows = list(area_rows)
    summary_rows.append({
        "summary_scope": "run",
        "area_id": "",
        "area_name": "",
        "lowering_factor": "",
        "areas_encountered": len(areas_encountered),
        "areas_successfully_processed": sum(
            row["areas_successfully_processed"] for row in area_rows
        ),
        "ems_records_tracks_processed": sum(
            row["ems_records_tracks_processed"]
            for row in area_rows
            if row["areas_successfully_processed"]
        ),
        "ems_records_tracks_assigned": int(
            assigned_points["area_assignment"].eq("matched").sum()
        ),
        "unmatched_ems_records_tracks": unmatched_count,
        "ambiguous_boundary_records": ambiguous_count,
        "250m_grid_cells_affected": sum(
            row["250m_grid_cells_affected"] for row in area_rows
        ),
        "50m_grid_cells_affected": sum(
            row["50m_grid_cells_affected"] for row in area_rows
        ),
        "warnings_errors": "; ".join(
            validation.errors
            + validation.warnings
            + [
                f"Area {area_key}: {message}"
                for area_key, messages in sorted(validation.area_errors.items())
                for message in messages
            ]
        ),
    })
    processing_summary = pd.DataFrame(summary_rows)
    outputs_written = []
    if export_cleaned_csv:
        cleaned_csv_path = output_folder / "cleaned_sampled_dredge_points_multi_area.csv"
        assigned_points_out.drop(columns="geometry").to_csv(cleaned_csv_path, index=False)
        outputs_written.append(cleaned_csv_path)
    if export_points_gpkg:
        points_gpkg_path = output_folder / "cleaned_sampled_dredge_points_multi_area.gpkg"
        assigned_points_out.to_file(
            points_gpkg_path,
            layer="cleaned_sampled_dredge_points",
            driver="GPKG",
        )
        outputs_written.append(points_gpkg_path)

    multiple_resolutions = resolution_mode == RESOLUTION_BOTH
    for resolution_name, grid_out in combined_grids.items():
        if multiple_resolutions:
            suffix = "_250m_regulatory" if resolution_name == RESOLUTION_250 else "_50m_internal"
        else:
            suffix = "" if resolution_name == RESOLUTION_250 else "_50m_internal"
        if export_grid_gpkg:
            grid_gpkg_path = output_folder / f"grid_dredge_time_lowering{suffix}.gpkg"
            grid_out.to_file(
                grid_gpkg_path,
                layer="grid_dredge_time_lowering",
                driver="GPKG",
            )
            outputs_written.append(grid_gpkg_path)
        if export_summary_csv:
            summary_csv_path = output_folder / f"grid_dredge_time_lowering_summary{suffix}.csv"
            combined_summaries[resolution_name].to_csv(summary_csv_path, index=False)
            outputs_written.append(summary_csv_path)

    if resolution_mode == RESOLUTION_BOTH:
        qc_tables = []
        for area_key in sorted(area_filter):
            if area_key not in validation.eligible_area_ids:
                continue
            area = model.areas.loc[model.areas["area_id"].astype(str).eq(area_key)].iloc[0]
            parent = combined_grids[RESOLUTION_250].loc[
                combined_grids[RESOLUTION_250]["area_id"].astype(str).eq(area_key)
            ]
            children = combined_grids[RESOLUTION_50].loc[
                combined_grids[RESOLUTION_50]["area_id"].astype(str).eq(area_key)
            ]
            if parent.empty or children.empty:
                continue
            children_for_qc = children.rename(columns={"parent_250_id": "ParentGridID"})
            qc = volume_conservation_qc(
                parent,
                children_for_qc,
                "grid_id",
                sample_minutes,
                float(area["lowering_factor"]) * 250.0 ** 2,
            )
            qc["area_id"] = area["area_id"]
            qc["area_name"] = area["area_name"]
            qc_tables.append(qc)
        if qc_tables:
            qc_path = output_folder / "resolution_volume_conservation_qc.csv"
            pd.concat(qc_tables, ignore_index=True).to_csv(qc_path, index=False)
            outputs_written.append(qc_path)

    if update_existing_grid:
        if not grid_outputs[RESOLUTION_250]:
            raise ValueError("No 250 m area results were produced for the cumulative-grid update.")
        cumulative = update_multi_area_cumulative_grid(
            existing_grid_path,
            combined_grids[RESOLUTION_250],
            log_callback=log_callback,
        )
        cumulative_path = output_folder / "cumulative_grid_dredge_time_lowering_UPDATED.gpkg"
        cumulative.to_file(
            cumulative_path,
            layer="cumulative_grid_dredge_time_lowering_UPDATED",
            driver="GPKG",
        )
        outputs_written.append(cumulative_path)

    summary_path = output_folder / "multi_area_processing_summary.csv"
    processing_summary.to_csv(summary_path, index=False)
    outputs_written.append(summary_path)

    if progress_callback:
        progress_callback(100)
    elapsed = time.time() - start_time
    if log_callback:
        log_callback("")
        log_callback("========== MULTI-AREA RESULTS SUMMARY ==========")
        log_callback(f"EMS files processed: {total_files:,}")
        log_callback(f"Raw EMS records indexed: {total_raw_records:,}")
        log_callback(f"Records in date range: {total_date_filtered_records:,}")
        log_callback(f"Valid dredging records: {total_dredge_records:,}")
        log_callback(f"Sampled EMS records: {total_sampled_records:,}")
        log_callback(f"Areas encountered: {len(areas_encountered):,}")
        log_callback(f"Areas successfully processed: {sum(row['areas_successfully_processed'] for row in area_rows):,}")
        log_callback(f"Unmatched records: {unmatched_count:,}")
        log_callback(f"Ambiguous boundary records: {ambiguous_count:,}")
        log_callback(f"Processing time: {elapsed / 60:.1f} minutes")
        log_callback("Outputs written:")
        for output in outputs_written:
            log_callback(f"  {output}")

    return outputs_written


# ==================================================
# TKINTER GUI
# ==================================================

class EMSLoweringApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("940x815")
        self.minsize(900, 745)
        # No custom icon for this build.
        # self.set_app_icon()
        self.configure(bg=CEMEX_BLUE)
        self.settings_store = SettingsStore(".ems_seabed_lowering_v8_settings.json")
        self.saved_settings = self.settings_store.load()
        self.cancel_event = threading.Event()

        self.input_folder_var = tk.StringVar(value=self.saved_settings.get("input_folder", ""))
        self.model_path_var = tk.StringVar(value=self.saved_settings.get("model_path", ""))
        self.output_folder_var = tk.StringVar(value=self.saved_settings.get("output_folder", ""))
        self.existing_grid_path_var = tk.StringVar(
            value=self.saved_settings.get("existing_grid_path", "")
        )
        self.update_existing_grid_var = tk.BooleanVar(
            value=self.saved_settings.get("update_existing_grid", False)
        )
        self.dredge_value_var = tk.StringVar(value=self.saved_settings.get("dredge_value", "3"))
        self.sample_minutes_var = tk.StringVar(value=self.saved_settings.get("sample_minutes", "1"))
        self.use_date_filter_var = tk.BooleanVar(value=self.saved_settings.get("use_date_filter", False))
        self.start_date_var = tk.StringVar(value=self.saved_settings.get("start_date", ""))
        self.end_date_var = tk.StringVar(value=self.saved_settings.get("end_date", ""))
        self.resolution_mode_var = tk.StringVar(
            value=self.saved_settings.get("resolution_mode", RESOLUTION_250)
        )
        self.processing_mode_var = tk.StringVar(
            value=self.saved_settings.get("processing_mode", "All affected areas")
        )
        self.selected_area_id_var = tk.StringVar(value=self.saved_settings.get("selected_area_id", ""))
        self.model_summary_var = tk.StringVar(value="No Lowering Model loaded.")
        self.dredge_mode_var = tk.StringVar(value="equals")
        self.spatial_predicate_var = tk.StringVar(value="within")

        self.export_cleaned_csv_var = tk.BooleanVar(value=True)
        self.export_points_gpkg_var = tk.BooleanVar(value=True)
        self.export_grid_gpkg_var = tk.BooleanVar(value=True)
        self.export_summary_csv_var = tk.BooleanVar(value=True)

        self.create_widgets()
        self.update_existing_grid_var.trace_add("write", lambda *_: self._toggle_dependent_inputs())
        self.use_date_filter_var.trace_add("write", lambda *_: self._toggle_dependent_inputs())
        self._toggle_dependent_inputs()
        if self.model_path_var.get():
            self.refresh_model_summary(show_errors=False)

    def set_app_icon(self):
        """Set the Tkinter window icon when running as .py or PyInstaller .exe."""
        try:
            ico_path = resource_path(ICON_ICO)
            self.iconbitmap(ico_path)
        except Exception:
            try:
                png_path = resource_path(ICON_PNG)
                self._icon_image = tk.PhotoImage(file=png_path)
                self.iconphoto(True, self._icon_image)
            except Exception:
                pass

    def create_widgets(self):
        style = ttk.Style(self)
        style.configure("App.TFrame", background=CEMEX_LIGHT_BLUE)
        style.configure("App.TLabel", background=CEMEX_LIGHT_BLUE)

        main = ttk.Frame(self, padding=12, style="App.TFrame")
        main.pack(fill="both", expand=True, padx=6, pady=6)

        # Main layout:
        # 1) Scrollable settings area at the top
        # 2) Run button, progress bar, log and footer fixed at the bottom
        # This keeps the Run button visible on smaller laptop screens.
        scroll_container = ttk.Frame(main, style="App.TFrame")
        scroll_container.pack(fill="both", expand=True, pady=(0, 10))

        canvas = tk.Canvas(scroll_container, bg=CEMEX_LIGHT_BLUE, highlightthickness=0)
        form_scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="App.TFrame")

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=form_scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        form_scrollbar.pack(side="right", fill="y")

        def update_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_scrollable_frame(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        scrollable_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_scrollable_frame)
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)

        # Header banner without image/logo dependencies.
        header = tk.Frame(scrollable_frame, bg=CEMEX_BLUE, padx=14, pady=12)
        header.pack(fill="x", pady=(0, 12))

        title_block = tk.Frame(header, bg=CEMEX_BLUE)
        title_block.pack(side="left", fill="x", expand=True)

        title = tk.Label(
            title_block,
            text=APP_TITLE,
            bg=CEMEX_BLUE,
            fg="white",
            font=("Segoe UI", 17, "bold")
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            title_block,
            text="Automatically assign EMS samples to model areas and apply each area's calibrated lowering factor",
            bg=CEMEX_BLUE,
            fg="white",
            font=("Segoe UI", 9)
        )
        subtitle.pack(anchor="w", pady=(3, 0))

        settings_frame = ttk.LabelFrame(scrollable_frame, text="Inputs and lowering model", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))

        self.add_path_row(settings_frame, 0, "EMS track root folder", self.input_folder_var, self.browse_input_folder, "Browse folder")
        self.add_path_row(settings_frame, 1, "Lowering Model GeoPackage", self.model_path_var, self.browse_model_file, "Browse file")
        self.add_path_row(settings_frame, 2, "Output folder", self.output_folder_var, self.browse_output_folder, "Browse folder")
        ttk.Label(settings_frame, textvariable=self.model_summary_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(2, 8)
        )
        ttk.Label(settings_frame, text="Processing mode").grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Combobox(
            settings_frame,
            textvariable=self.processing_mode_var,
            values=["All affected areas", "Selected area"],
            state="readonly",
            width=24,
        ).grid(row=4, column=1, sticky="w", padx=(0, 8), pady=5)
        ttk.Label(settings_frame, text="Selected area_id").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=5
        )
        self.selected_area_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.selected_area_id_var,
            values=[],
            state="readonly",
            width=24,
        )
        self.selected_area_combo.grid(row=5, column=1, sticky="w", padx=(0, 8), pady=5)

        ttk.Label(settings_frame, text="Lowering output resolution").grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Combobox(
            settings_frame,
            textvariable=self.resolution_mode_var,
            values=[RESOLUTION_250, RESOLUTION_50, RESOLUTION_BOTH],
            state="readonly",
            width=28,
        ).grid(row=6, column=1, sticky="w", padx=(0, 8), pady=5)
        ttk.Checkbutton(
            settings_frame,
            text="Update existing multi-area cumulative grid",
            variable=self.update_existing_grid_var,
        ).grid(row=7, column=1, sticky="w", padx=(0, 8), pady=5)
        self.existing_grid_entry = self.add_path_row(
            settings_frame,
            8,
            "Existing cumulative grid",
            self.existing_grid_path_var,
            self.browse_existing_grid_file,
            "Browse file",
        )

        settings_frame.columnconfigure(1, weight=1)

        parameter_frame = ttk.LabelFrame(scrollable_frame, text="EMS processing settings", padding=10)
        parameter_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(parameter_frame, text="DREDGE value").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parameter_frame, textvariable=self.dredge_value_var, width=10).grid(row=0, column=1, sticky="w", padx=(0, 20), pady=4)

        dredge_mode_frame = ttk.Frame(parameter_frame)
        dredge_mode_frame.grid(row=0, column=2, sticky="w", pady=4)
        ttk.Radiobutton(dredge_mode_frame, text="DREDGE = value", variable=self.dredge_mode_var, value="equals").pack(side="left", padx=(0, 12))
        ttk.Radiobutton(dredge_mode_frame, text="DREDGE >= value", variable=self.dredge_mode_var, value="greater_equal").pack(side="left")

        ttk.Label(parameter_frame, text="Sample interval, minutes").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(parameter_frame, textvariable=self.sample_minutes_var, values=["1", "2", "5", "10", "15"], width=8, state="readonly").grid(row=1, column=1, sticky="w", padx=(0, 20), pady=4)

        ttk.Label(parameter_frame, text="Spatial join predicate").grid(row=1, column=2, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(parameter_frame, textvariable=self.spatial_predicate_var, values=["within", "intersects"], width=12, state="readonly").grid(row=1, column=3, sticky="w", pady=4)

        ttk.Label(
            parameter_frame,
            text="250 m factors are read from areas.lowering_factor; 50 m factors are scaled to preserve volume."
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 2))

        ttk.Checkbutton(
            parameter_frame,
            text="Enable date filter",
            variable=self.use_date_filter_var
        ).grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 4))

        ttk.Label(parameter_frame, text="Start date, dd/mm/yyyy").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        self.start_date_entry = ttk.Entry(parameter_frame, textvariable=self.start_date_var, width=14)
        self.start_date_entry.grid(row=4, column=1, sticky="w", padx=(0, 20), pady=4)

        ttk.Label(parameter_frame, text="End date, dd/mm/yyyy").grid(row=4, column=2, sticky="w", padx=(0, 8), pady=4)
        self.end_date_entry = ttk.Entry(parameter_frame, textvariable=self.end_date_var, width=14)
        self.end_date_entry.grid(row=4, column=3, sticky="w", pady=4)

        output_frame = ttk.LabelFrame(scrollable_frame, text="Outputs", padding=10)
        output_frame.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(output_frame, text="Cleaned sampled points CSV", variable=self.export_cleaned_csv_var).grid(row=0, column=0, sticky="w", padx=(0, 20), pady=4)
        ttk.Checkbutton(output_frame, text="Cleaned sampled points GeoPackage", variable=self.export_points_gpkg_var).grid(row=0, column=1, sticky="w", padx=(0, 20), pady=4)
        ttk.Checkbutton(output_frame, text="Grid with dredging time and lowering GeoPackage", variable=self.export_grid_gpkg_var).grid(row=1, column=0, sticky="w", padx=(0, 20), pady=4)
        ttk.Checkbutton(output_frame, text="Grid summary CSV", variable=self.export_summary_csv_var).grid(row=1, column=1, sticky="w", padx=(0, 20), pady=4)

        run_frame = ttk.Frame(main, style="App.TFrame")
        run_frame.pack(fill="x", pady=(0, 8))
        self.run_button = ttk.Button(run_frame, text="Run EMS Lowering Workflow", command=self.run_processing_thread)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(run_frame, text="Cancel", command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(side="left", padx=(8, 0))
        self.status_var = tk.StringVar(value="Ready — outputs are written to a new timestamped run folder.")
        ttk.Label(run_frame, textvariable=self.status_var).pack(side="left", padx=(8, 0))
        self.progress = ttk.Progressbar(run_frame, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=(12, 0))

        log_frame = ttk.LabelFrame(main, text="Status log", padding=10)
        log_frame.pack(fill="x", expand=False, pady=(0, 4))
        self.log_text = tk.Text(log_frame, wrap="word", height=8, font=("Consolas", 10))
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        footer = ttk.Label(main, text=APP_FOOTER, font=("Segoe UI", 8), style="App.TLabel")
        footer.pack(side="bottom", anchor="e", pady=(4, 0))

        self.log("Ready.")
        self.log("Expected EMS fields: DATE, TIME, GPSLAT, GPSLON, DREDGE")
        self.log("EMS source files are indexed in the shared SQLite cache; raw files are unchanged.")


    def add_path_row(self, parent, row, label, variable, command, browse_text):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=5)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=5)
        ttk.Button(parent, text=browse_text, command=command).grid(row=row, column=2, sticky="e", pady=5)
        return entry

    def _toggle_dependent_inputs(self):
        date_state = "normal" if self.use_date_filter_var.get() else "disabled"
        update_state = "normal" if self.update_existing_grid_var.get() else "disabled"
        self.start_date_entry.configure(state=date_state)
        self.end_date_entry.configure(state=date_state)
        self.existing_grid_entry.configure(state=update_state)

    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select EMS track root folder")
        if folder:
            self.input_folder_var.set(folder)

    def browse_model_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Lowering Model GeoPackage",
            filetypes=[("GeoPackage", "*.gpkg"), ("All files", "*.*")],
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.refresh_model_summary()

    def refresh_model_summary(self, show_errors=True):
        model_path = self.model_path_var.get().strip()
        if not model_path:
            self.model_summary_var.set("No Lowering Model loaded.")
            self.selected_area_combo.configure(values=[])
            return
        try:
            model = load_lowering_model(model_path)
            validation = validate_lowering_model(model)
            self.model_summary_var.set(lowering_model_summary(model, validation))
            if {"active", "area_id"}.issubset(model.areas.columns):
                active_ids = model.areas.loc[
                    model.areas["active"].astype(str).str.casefold().isin(
                        {"true", "1", "yes", "y"}
                    ),
                    "area_id",
                ].dropna().astype(str).tolist()
            else:
                active_ids = []
            self.selected_area_combo.configure(values=active_ids)
            if self.selected_area_id_var.get() not in active_ids:
                self.selected_area_id_var.set(active_ids[0] if active_ids else "")
        except Exception as exc:
            self.model_summary_var.set(f"Unable to load model: {exc}")
            self.selected_area_combo.configure(values=[])
            if show_errors:
                messagebox.showerror("Lowering Model error", str(exc))

    def browse_existing_grid_file(self):
        file_path = filedialog.askopenfilename(
            title="Select existing multi-area cumulative grid",
            filetypes=[("GeoPackage", "*.gpkg"), ("All files", "*.*")],
        )
        if file_path:
            self.existing_grid_path_var.set(file_path)

    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_folder_var.set(folder)

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.update_idletasks()

    def safe_log(self, message):
        self.after(0, lambda: self.log(message))

    def set_progress(self, value):
        self.after(0, lambda: (self.progress.configure(value=value), self.status_var.set(f"Processing: {value}%")))

    def validate_inputs(self):
        input_folder = Path(self.input_folder_var.get())
        model_path = Path(self.model_path_var.get())
        output_folder = Path(self.output_folder_var.get())
        existing_grid_path_text = self.existing_grid_path_var.get().strip()
        existing_grid_path = Path(existing_grid_path_text) if existing_grid_path_text else None
        update_existing_grid = self.update_existing_grid_var.get()

        if not input_folder.exists() or not input_folder.is_dir():
            raise ValueError("Please select a valid EMS track root folder.")
        if not model_path.exists() or not model_path.is_file():
            raise ValueError("Please select a valid Lowering Model GeoPackage.")
        if not self.output_folder_var.get().strip():
            raise ValueError("Please select a valid output folder.")

        try:
            dredge_value = float(self.dredge_value_var.get())
        except ValueError:
            raise ValueError("DREDGE value must be numeric.")

        try:
            sample_minutes = int(self.sample_minutes_var.get())
        except ValueError:
            raise ValueError("Sample interval must be a whole number.")
        if sample_minutes <= 0:
            raise ValueError("Sample interval must be greater than zero.")

        use_date_filter = self.use_date_filter_var.get()
        start_date = None
        end_date = None

        if use_date_filter:
            start_text = self.start_date_var.get().strip()
            end_text = self.end_date_var.get().strip()

            if not start_text or not end_text:
                raise ValueError("Date filter is enabled. Please enter both a start date and an end date in dd/mm/yyyy format.")

            try:
                start_date = pd.to_datetime(start_text, dayfirst=True, errors="raise").normalize()
                end_date = pd.to_datetime(end_text, dayfirst=True, errors="raise").normalize()
            except Exception:
                raise ValueError("Start date and end date must use a valid date format, for example 01/03/2024.")

            if end_date < start_date:
                raise ValueError("End date must be the same as or after the start date.")

            # Make the end date inclusive for the whole selected day.
            end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        resolution_mode = self.resolution_mode_var.get()
        if resolution_mode not in {RESOLUTION_250, RESOLUTION_50, RESOLUTION_BOTH}:
            raise ValueError("Select 250 m Regulatory, 50 m Internal Detail, or Both.")
        if update_existing_grid:
            if resolution_mode == RESOLUTION_50:
                raise ValueError(
                    "Cumulative-grid updates require the 250 m regulatory output."
                )
            if existing_grid_path is None or not existing_grid_path.is_file():
                raise ValueError(
                    "Select an existing multi-area cumulative grid GeoPackage."
                )
        processing_mode = self.processing_mode_var.get()
        if processing_mode not in {"All affected areas", "Selected area"}:
            raise ValueError("Select All affected areas or Selected area processing mode.")
        selected_area_id = self.selected_area_id_var.get().strip() or None
        if processing_mode == "Selected area":
            if not selected_area_id:
                raise ValueError("Select an area_id when using Selected area mode.")
            model = load_lowering_model(model_path)
            if not {"active", "area_id"}.issubset(model.areas.columns):
                raise ValueError("The Lowering Model areas layer lacks active or area_id fields.")
            active_ids = set(
                model.areas.loc[
                    model.areas["active"].astype(str).str.casefold().isin(
                        {"true", "1", "yes", "y"}
                    ),
                    "area_id",
                ].dropna().astype(str)
            )
            if selected_area_id not in active_ids:
                raise ValueError(f"Selected area_id {selected_area_id!r} is not active in the model.")

        export_options = [
            self.export_cleaned_csv_var.get(),
            self.export_points_gpkg_var.get(),
            self.export_grid_gpkg_var.get(),
            self.export_summary_csv_var.get()
        ]
        if not any(export_options):
            raise ValueError("Please select at least one output.")

        return {
            "input_folder": input_folder,
            "model_path": model_path,
            "output_folder": output_folder,
            "dredge_value": dredge_value,
            "dredge_mode": self.dredge_mode_var.get(),
            "sample_minutes": sample_minutes,
            "use_date_filter": use_date_filter,
            "start_date": start_date,
            "end_date": end_date,
            "resolution_mode": resolution_mode,
            "processing_mode": processing_mode,
            "selected_area_id": selected_area_id,
            "update_existing_grid": update_existing_grid,
            "existing_grid_path": existing_grid_path,
            "spatial_predicate": self.spatial_predicate_var.get(),
            "export_cleaned_csv": self.export_cleaned_csv_var.get(),
            "export_points_gpkg": self.export_points_gpkg_var.get(),
            "export_grid_gpkg": self.export_grid_gpkg_var.get(),
            "export_summary_csv": self.export_summary_csv_var.get(),
        }

    def save_settings(self):
        values = {
            "input_folder": self.input_folder_var.get(), "model_path": self.model_path_var.get(),
            "output_folder": self.output_folder_var.get(),
            "dredge_value": self.dredge_value_var.get(), "sample_minutes": self.sample_minutes_var.get(),
            "use_date_filter": self.use_date_filter_var.get(), "start_date": self.start_date_var.get(),
            "end_date": self.end_date_var.get(),
            "resolution_mode": self.resolution_mode_var.get(),
            "processing_mode": self.processing_mode_var.get(),
            "selected_area_id": self.selected_area_id_var.get(),
            "existing_grid_path": self.existing_grid_path_var.get(),
            "update_existing_grid": self.update_existing_grid_var.get(),
        }
        self.settings_store.save(values)

    def cancel_processing(self):
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status_var.set("Cancelling after the current file…")

    def run_processing_thread(self):
        try:
            settings = self.validate_inputs()
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.cancel_event.clear()
        self.cancel_button.configure(state="normal")
        self.status_var.set("Scanning and validating EMS files…")
        self.save_settings()
        self.progress.configure(value=0)
        self.log("")
        self.log("========================================")
        self.log("New run started")
        self.log("========================================")

        settings["cancel_event"] = self.cancel_event
        thread = threading.Thread(target=self.run_processing, args=(settings,), daemon=True)
        thread.start()

    def run_processing(self, settings):
        try:
            run_lowering_workflow(**settings, log_callback=self.safe_log, progress_callback=self.set_progress)
            self.after(0, lambda: self.status_var.set("Complete — see output paths in the log."))
            self.after(0, lambda: messagebox.showinfo("Complete", "EMS lowering workflow completed successfully."))
        except Exception as exc:
            self.safe_log("")
            self.safe_log(f"ERROR: {exc}")
            self.after(0, lambda: messagebox.showerror("Processing error", str(exc)))
        finally:
            self.after(0, lambda: self.run_button.configure(state="normal"))
            self.after(0, lambda: self.cancel_button.configure(state="disabled"))


if __name__ == "__main__":
    app = EMSLoweringApp()
    app.mainloop()
