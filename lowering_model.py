"""Loading, validating and spatially assigning a Lowering Model GeoPackage."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from pyproj import CRS


AREA_COLUMNS = {"area_id", "area_name", "lowering_factor", "active", "geometry"}
GRID_250_COLUMNS = {"grid_id", "area_id", "geometry"}
GRID_50_COLUMNS = {"grid_id", "parent_250_id", "area_id", "geometry"}
AREA_BOUNDARY_TOLERANCE_M = 0.01


@dataclass
class LoweringModel:
    path: Path
    areas: gpd.GeoDataFrame
    grid_250m: gpd.GeoDataFrame
    grid_50m: gpd.GeoDataFrame | None


@dataclass
class ModelValidation:
    eligible_area_ids: set[str] = field(default_factory=set)
    area_errors: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_area_error(self, area_id: str, message: str) -> None:
        self.area_errors.setdefault(area_id, []).append(message)


def _layer_names(path: Path) -> set[str]:
    try:
        return {str(row[0]) for row in pyogrio.list_layers(path)}
    except Exception as exc:
        raise ValueError(f"Could not inspect Lowering Model GeoPackage {path}: {exc}") from exc


def load_lowering_model(path: Path | str) -> LoweringModel:
    """Read model layers once, requiring the area and regulatory grid layers."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Lowering Model GeoPackage not found: {model_path}")

    layers = _layer_names(model_path)
    required_layers = {"areas", "grid_250m"}
    missing_layers = sorted(required_layers - layers)
    if missing_layers:
        raise ValueError(
            "Lowering Model GeoPackage is missing required layer(s): "
            + ", ".join(missing_layers)
        )

    areas = gpd.read_file(model_path, layer="areas")
    grid_250m = gpd.read_file(model_path, layer="grid_250m")
    grid_50m = (
        gpd.read_file(model_path, layer="grid_50m")
        if "grid_50m" in layers else None
    )
    return LoweringModel(model_path, areas, grid_250m, grid_50m)


def _has_columns(frame: gpd.GeoDataFrame, required: set[str], layer: str) -> list[str]:
    return sorted(required - set(frame.columns))


def _boolean_value(value) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def validate_lowering_model(model: LoweringModel) -> ModelValidation:
    """Validate model structure and isolate unusable areas from processable ones."""
    result = ModelValidation()
    missing = _has_columns(model.areas, AREA_COLUMNS, "areas")
    if missing:
        result.errors.append(f"areas layer is missing required field(s): {', '.join(missing)}")
    missing = _has_columns(model.grid_250m, GRID_250_COLUMNS, "grid_250m")
    if missing:
        result.errors.append(f"grid_250m layer is missing required field(s): {', '.join(missing)}")
    if model.grid_50m is not None:
        missing = _has_columns(model.grid_50m, GRID_50_COLUMNS, "grid_50m")
        if missing:
            result.errors.append(f"grid_50m layer is missing required field(s): {', '.join(missing)}")
    if result.errors:
        return result

    layers = [("areas", model.areas), ("grid_250m", model.grid_250m)]
    if model.grid_50m is not None:
        layers.append(("grid_50m", model.grid_50m))
    if any(frame.crs is None for _, frame in layers):
        result.errors.append("Every model layer must have a defined CRS.")
        return result

    areas_crs = CRS.from_user_input(model.areas.crs)
    if not areas_crs.is_projected or not areas_crs.axis_info or any(
        not math.isclose(axis.unit_conversion_factor, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for axis in areas_crs.axis_info[:2]
    ):
        result.errors.append("The areas layer CRS must be projected and use metres.")
    for name, frame in layers[1:]:
        if CRS.from_user_input(frame.crs) != areas_crs:
            result.errors.append(f"{name} CRS does not match the areas layer CRS.")
    if result.errors:
        return result

    area_ids = model.areas["area_id"]
    invalid_area_ids = area_ids.isna() | area_ids.astype(str).str.strip().eq("")
    if invalid_area_ids.any():
        result.errors.append(
            f"areas layer contains {int(invalid_area_ids.sum())} missing area_id value(s)."
        )
    area_keys = area_ids.astype(str)
    duplicate_area_keys = set(area_keys[~invalid_area_ids][area_keys[~invalid_area_ids].duplicated(keep=False)])
    for area_key in duplicate_area_keys:
        result.add_area_error(area_key, "area_id is duplicated.")

    parsed_active = model.areas["active"].map(_boolean_value)
    for position, row in model.areas.iterrows():
        key = str(row["area_id"]) if pd.notna(row["area_id"]) else ""
        if not key:
            continue
        if parsed_active.loc[position] is None:
            result.add_area_error(key, "active must be a non-null Boolean value.")
        elif parsed_active.loc[position]:
            factor = pd.to_numeric(pd.Series([row["lowering_factor"]]), errors="coerce").iloc[0]
            if pd.isna(factor) or not math.isfinite(float(factor)) or float(factor) < 0:
                result.add_area_error(key, "lowering_factor must be a finite, non-negative number.")
            if pd.isna(row["area_name"]) or not str(row["area_name"]).strip():
                result.add_area_error(key, "area_name must be populated for active areas.")

        geometry = row.geometry
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            result.add_area_error(key, "Area geometry is null, empty or invalid.")
        elif geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            result.add_area_error(key, "Area geometry must be a polygon.")

    known_area_keys = set(area_keys[~invalid_area_ids])
    grid_keys = model.grid_250m["area_id"].astype(str)
    unknown_grid_areas = set(grid_keys) - known_area_keys
    if unknown_grid_areas:
        result.errors.append(
            "grid_250m references unknown area_id value(s): "
            + ", ".join(sorted(unknown_grid_areas))
        )

    active_by_key = {
        str(row["area_id"]): _boolean_value(row["active"])
        for _, row in model.areas.iterrows()
        if pd.notna(row["area_id"])
    }
    for layer_name, frame in layers[1:]:
        id_column = "grid_id"
        ids = frame[id_column]
        missing_ids = ids.isna() | ids.astype(str).str.strip().eq("")
        if missing_ids.any():
            result.errors.append(
                f"{layer_name} contains {int(missing_ids.sum())} missing grid_id value(s)."
            )
            for area_key in set(frame.loc[missing_ids, "area_id"].dropna().astype(str)):
                result.add_area_error(area_key, f"{layer_name} contains missing grid_id values.")
        duplicate_ids = set(ids[~missing_ids].astype(str)[ids[~missing_ids].astype(str).duplicated(keep=False)])
        if duplicate_ids:
            result.errors.append(
                f"{layer_name} contains duplicate grid_id value(s): "
                + ", ".join(sorted(duplicate_ids)[:10])
            )
            for area_key in set(frame.loc[ids.astype(str).isin(duplicate_ids), "area_id"].astype(str)):
                result.add_area_error(area_key, f"{layer_name} contains duplicate grid_id values.")

        for index, row in frame.iterrows():
            area_key = str(row["area_id"]) if pd.notna(row["area_id"]) else ""
            if area_key not in known_area_keys:
                continue
            geometry = row.geometry
            if geometry is None or geometry.is_empty or not geometry.is_valid:
                result.add_area_error(area_key, f"{layer_name} has a null, empty or invalid geometry.")
            elif geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                result.add_area_error(area_key, f"{layer_name} geometry must be polygonal.")

    for area_key, is_active in active_by_key.items():
        if is_active is True and area_key not in unknown_grid_areas:
            if not (grid_keys == area_key).any():
                result.add_area_error(area_key, "Active area has no 250 m grid cells.")

    if model.grid_50m is not None:
        unknown_50_areas = (
            set(model.grid_50m["area_id"].dropna().astype(str)) - known_area_keys
        )
        if unknown_50_areas:
            result.errors.append(
                "grid_50m references unknown area_id value(s): "
                + ", ".join(sorted(unknown_50_areas))
            )
        parent_keys = set(model.grid_250m["grid_id"].dropna().astype(str))
        for _, row in model.grid_50m.iterrows():
            area_key = str(row["area_id"]) if pd.notna(row["area_id"]) else ""
            parent_key = str(row["parent_250_id"]) if pd.notna(row["parent_250_id"]) else ""
            parents = model.grid_250m[
                model.grid_250m["grid_id"].astype(str).eq(parent_key)
                & model.grid_250m["area_id"].astype(str).eq(area_key)
            ]
            if parent_key not in parent_keys or parents.empty:
                result.add_area_error(
                    area_key,
                    f"50 m cell references missing or mismatched parent_250_id {parent_key!r}.",
                )
            elif row.geometry is not None and not row.geometry.is_empty:
                parent_geometry = parents.iloc[0].geometry
                if parent_geometry is not None and not parent_geometry.is_empty:
                    if row.geometry.difference(parent_geometry).area > 1e-4:
                        result.add_area_error(
                            area_key,
                            f"50 m cell {row['grid_id']!r} extends outside its parent 250 m cell.",
                        )

    active = model.areas.loc[parsed_active.eq(True)].copy()
    if len(active) > 1:
        active = active.loc[active.geometry.notna() & ~active.geometry.is_empty]
        try:
            pairs = gpd.sjoin(
                active.reset_index(names="_area_index"),
                active[["area_id", "geometry"]].reset_index(drop=True),
                how="inner",
                predicate="intersects",
                lsuffix="left",
                rsuffix="right",
            )
            ambiguous_pairs = pairs.loc[
                pairs["_area_index"].astype(str) != pairs["index_right"].astype(str)
            ]
            if not ambiguous_pairs.empty:
                result.warnings.append(
                    "Active area polygons overlap or touch; EMS points intersecting multiple "
                    "areas will be marked ambiguous."
                )
        except (ImportError, ValueError) as exc:
            result.warnings.append(f"Could not pre-check active area overlaps: {exc}")

    for _, row in model.areas.iterrows():
        key = str(row["area_id"]) if pd.notna(row["area_id"]) else ""
        if not key or _boolean_value(row["active"]) is not True:
            continue
        if key in duplicate_area_keys:
            continue
        if result.area_errors.get(key):
            continue
        if key not in unknown_grid_areas:
            result.eligible_area_ids.add(key)

    return result


def assign_points_to_areas(
    points: gpd.GeoDataFrame,
    model: LoweringModel,
    validation: ModelValidation,
) -> gpd.GeoDataFrame:
    """Assign each sampled point only when it intersects exactly one usable area."""
    active = model.areas.loc[
        model.areas["active"].map(_boolean_value).eq(True),
        ["area_id", "area_name", "geometry"],
    ].copy()
    active = active.loc[
        active["area_id"].notna()
        & active.geometry.notna()
        & ~active.geometry.is_empty
        & active.geometry.is_valid
    ]
    assigned = points.to_crs(model.areas.crs).reset_index(drop=True).copy()
    assigned["_point_id"] = range(len(assigned))
    assigned["area_id"] = pd.NA
    assigned["area_name"] = pd.NA
    assigned["area_assignment"] = "unmatched"
    assigned["area_candidates"] = "[]"
    if active.empty or assigned.empty:
        return assigned.drop(columns="_point_id")

    exact_matches = gpd.sjoin(
        assigned[["_point_id", "geometry"]],
        active,
        how="inner",
        predicate="intersects",
    )
    near_points = assigned[["_point_id", "geometry"]].copy()
    near_points.geometry = near_points.geometry.buffer(AREA_BOUNDARY_TOLERANCE_M)
    near_matches = gpd.sjoin(
        near_points,
        active,
        how="inner",
        predicate="intersects",
    )
    exact_by_point = {
        int(point_id): group
        for point_id, group in exact_matches.groupby("_point_id")
    }
    near_by_point = {
        int(point_id): set(group["area_id"].astype(str))
        for point_id, group in near_matches.groupby("_point_id")
    }
    for point_id, group in exact_by_point.items():
        exact_ids = set(group["area_id"].astype(str))
        candidate_ids = near_by_point.get(point_id, exact_ids)
        assigned.loc[point_id, "area_candidates"] = json.dumps(
            sorted(candidate_ids)
        )
        if len(exact_ids) > 1 or (len(exact_ids) == 1 and len(candidate_ids) > 1):
            assigned.loc[point_id, "area_assignment"] = "ambiguous"
            continue
        match = group.iloc[0]
        assigned.loc[point_id, "area_id"] = match["area_id"]
        assigned.loc[point_id, "area_name"] = match["area_name"]
        assigned.loc[point_id, "area_assignment"] = "matched"
    return assigned.drop(columns="_point_id")


def lowering_model_summary(model: LoweringModel, validation: ModelValidation) -> str:
    active = (
        model.areas["active"].map(_boolean_value).eq(True)
        if "active" in model.areas
        else pd.Series(False, index=model.areas.index)
    )
    if "lowering_factor" in model.areas:
        factors = pd.to_numeric(
            model.areas.loc[active, "lowering_factor"], errors="coerce"
        )
        areas_with_factor = factors.map(
            lambda value: pd.notna(value)
            and math.isfinite(float(value))
            and float(value) >= 0
        )
    else:
        areas_with_factor = pd.Series(dtype=bool)
    grid_50_count = len(model.grid_50m) if model.grid_50m is not None else len(model.grid_250m) * 25
    return (
        f"Active areas: {int(active.sum())} | "
        f"250 m grid cells: {len(model.grid_250m)} | "
        f"50 m grid cells: {grid_50_count} | "
        f"Areas with lowering factors: {int(areas_with_factor.sum())} | "
        f"Processable areas: {len(validation.eligible_area_ids)}"
    )
