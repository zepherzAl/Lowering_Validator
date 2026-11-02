"""Alignment checks and exact vector-grid refinement for lowering workflows."""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections import Counter
from typing import Callable

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry


ALIGNMENT_TOLERANCE_M = 0.01
AREA_TOLERANCE_M2 = 5.0
PHASE_DOMINANCE_RATIO = 0.95
PHASE_CONFLICT_MIN_CELLS = 2
CHILDREN_PER_AXIS = 5


@dataclass(frozen=True)
class MasterGridLayout:
    """Inferred affine lattice and exact parent-cell indices."""

    grid_id_column: str
    cell_size_m: float
    angle_radians: float
    origin_u: float
    origin_v: float
    parent_ids: tuple[object, ...]
    parent_indices: tuple[tuple[int, int] | None, ...]
    local_geometries: tuple[BaseGeometry | None, ...]
    parent_frames: tuple[tuple[float, float, float, float, float, float] | None, ...] = ()

    @property
    def row_count(self) -> int:
        rows = [index[0] for index in self.parent_indices if index is not None]
        return max(rows) - min(rows) + 1 if rows else 0

    @property
    def column_count(self) -> int:
        columns = [index[1] for index in self.parent_indices if index is not None]
        return max(columns) - min(columns) + 1 if columns else 0


def _close(left: float, right: float, tolerance: float = ALIGNMENT_TOLERANCE_M) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _local_transform(geometry: BaseGeometry, angle: float) -> BaseGeometry:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return affine_transform(geometry, [cosine, sine, -sine, cosine, 0.0, 0.0])


def _world_transform(geometry: BaseGeometry, angle: float) -> BaseGeometry:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return affine_transform(geometry, [cosine, -sine, sine, cosine, 0.0, 0.0])


def _phase_distance(left: float, right: float, period: float) -> float:
    return min(abs(left - right), period - abs(left - right))


def _phase_center(values: list[float], period: float) -> float:
    angles = [value * 2.0 * math.pi / period for value in values]
    x = sum(math.cos(angle) for angle in angles)
    y = sum(math.sin(angle) for angle in angles)
    phase = (math.atan2(y, x) % (2.0 * math.pi)) * period / (2.0 * math.pi)
    if _phase_distance(phase, 0.0, period) <= ALIGNMENT_TOLERANCE_M:
        return 0.0
    return phase


def _axis_lattice_phase(
    cells: list[tuple[int, float, float]],
    axis: int,
    cell_size_m: float,
    parent_ids: tuple[object, ...],
) -> tuple[float, list[dict[str, object]]]:
    samples = [
        (coordinate % cell_size_m, parent_ids[position])
        for position, coordinate, _ in cells
    ] if axis == 0 else [
        (coordinate % cell_size_m, parent_ids[position])
        for position, _, coordinate in cells
    ]

    clusters: list[list[tuple[float, object]]] = []
    for phase, parent_id in sorted(samples, key=lambda item: item[0]):
        matching = next(
            (
                cluster
                for cluster in clusters
                if _phase_distance(
                    phase,
                    _phase_center([item[0] for item in cluster], cell_size_m),
                    cell_size_m,
                )
                <= ALIGNMENT_TOLERANCE_M
            ),
            None,
        )
        if matching is None:
            clusters.append([(phase, parent_id)])
        else:
            matching.append((phase, parent_id))

    merged = True
    while merged and len(clusters) > 1:
        merged = False
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                left_center = _phase_center(
                    [item[0] for item in left], cell_size_m
                )
                right_center = _phase_center(
                    [item[0] for item in right], cell_size_m
                )
                if (
                    _phase_distance(left_center, right_center, cell_size_m)
                    <= ALIGNMENT_TOLERANCE_M
                ):
                    left.extend(right)
                    del clusters[right_index]
                    merged = True
                    break
            if merged:
                break

    phase_clusters = [
        {
            "phase": _phase_center([item[0] for item in cluster], cell_size_m),
            "count": len(cluster),
            "ids": [str(item[1]) for item in cluster],
        }
        for cluster in clusters
    ]
    phase_clusters.sort(
        key=lambda cluster: int(cluster["count"]), reverse=True
    )
    if not phase_clusters:
        raise ValueError("No complete parent cells are available for alignment.")
    return float(phase_clusters[0]["phase"]), phase_clusters


def _distribution(values: list[float]) -> str:
    if not values:
        return "n/a"
    series = pd.Series(values, dtype="float64")
    return (
        f"min={series.min():.6f}, p25={series.quantile(0.25):.6f}, "
        f"median={series.median():.6f}, p75={series.quantile(0.75):.6f}, "
        f"max={series.max():.6f}"
    )


def _grid_diagnostics(
    grid: gpd.GeoDataFrame,
    grid_id_column: str,
    cell_size_m: float,
    geometries: list[BaseGeometry | None],
    local_geometries: list[BaseGeometry | None],
    complete_cells: list[tuple[int, float, float]],
    x_phases: list[dict[str, object]],
    y_phases: list[dict[str, object]],
) -> str:
    geometry_types = Counter(
        geometry.geom_type
        for geometry in geometries
        if geometry is not None
    )
    nonempty = [
        geometry
        for geometry in geometries
        if geometry is not None and not geometry.is_empty
    ]
    bounds = [geometry.bounds for geometry in local_geometries if geometry is not None]
    widths = [item[2] - item[0] for item in bounds]
    heights = [item[3] - item[1] for item in bounds]
    areas = [geometry.area for geometry in local_geometries if geometry is not None]
    approx_square = sum(
        abs(width - cell_size_m) <= ALIGNMENT_TOLERANCE_M
        and abs(height - cell_size_m) <= ALIGNMENT_TOLERANCE_M
        for width, height in zip(widths, heights)
    )
    approx_area = sum(
        abs(area - cell_size_m**2) <= AREA_TOLERANCE_M2 for area in areas
    )
    complete_ids = {
        position for position, _, _ in complete_cells
    }

    def format_phases(phases: list[dict[str, object]]) -> str:
        if not phases:
            return "none"
        return "; ".join(
            f"{float(item['phase']):.6f} m ({item['count']} cells; "
            f"IDs: {', '.join(item['ids'][:10])})"
            for item in phases[:5]
        )

    invalid_count = sum(
        geometry is not None and not geometry.is_empty and not geometry.is_valid
        for geometry in geometries
    )
    null_empty_count = len(geometries) - len(nonempty)
    return "\n".join(
        [
            "Master-grid diagnostics:",
            f"  Layer CRS: {grid.crs}",
            f"  Feature count: {len(grid)}",
            f"  Geometry types: {dict(geometry_types)}",
            f"  Null/empty geometries: {null_empty_count}",
            f"  Invalid geometries: {invalid_count}",
            (
                "  Bounds: "
                + (
                    f"X={min(item[0] for item in bounds):.6f}.."
                    f"{max(item[2] for item in bounds):.6f}, "
                    f"Y={min(item[1] for item in bounds):.6f}.."
                    f"{max(item[3] for item in bounds):.6f}"
                    if bounds
                    else "n/a"
                )
            ),
            f"  Polygon width distribution (m): {_distribution(widths)}",
            f"  Polygon height distribution (m): {_distribution(heights)}",
            f"  Polygon area distribution (m2): {_distribution(areas)}",
            f"  Approximately {cell_size_m:g} x {cell_size_m:g} m: {approx_square}",
            f"  Approximately {cell_size_m**2:g} m2: {approx_area}",
            f"  Verified complete parent cells: {len(complete_ids)}",
            f"  Candidate X phases: {format_phases(x_phases)}",
            f"  Candidate Y phases: {format_phases(y_phases)}",
            (
                "  Tolerances: coordinate/dimension "
                f"{ALIGNMENT_TOLERANCE_M:g} m; area {AREA_TOLERANCE_M2:g} m2; "
                f"phase dominance {PHASE_DOMINANCE_RATIO:.0%}"
            ),
            f"  ID field: {grid_id_column}",
        ]
    )


def _raise_grid_error(
    message: str,
    diagnostics: str,
    log: Callable[[str], None] | None,
) -> None:
    if log is not None:
        for line in diagnostics.splitlines():
            log(line)
    raise ValueError(f"{message}\n{diagnostics}")


def _grid_angle(geometry: BaseGeometry) -> float:
    rectangle = geometry.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon):
        raise ValueError("The master grid contains a geometry without a rectangular affine envelope.")
    corners = list(rectangle.exterior.coords)[:-1]
    if len(corners) != 4:
        raise ValueError("The master grid must use polygon cells with four affine envelope corners.")
    edges = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > ALIGNMENT_TOLERANCE_M:
            edges.append((length, math.atan2(dy, dx)))
    if len(edges) != 4:
        raise ValueError("Could not infer the master grid orientation.")
    angle = max(edges, key=lambda item: item[0])[1]
    return (angle + math.pi / 4) % (math.pi / 2) - math.pi / 4


def _parent_affine_frame(geometry: BaseGeometry, cell_size_m: float) -> tuple[float, float, float, float, float, float] | None:
    """Return an affine frame for one nominal parent after CRS transformation."""
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        return None
    if abs(float(geometry.area) - cell_size_m ** 2) > 200.0:
        return None
    rectangle = geometry.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon):
        return None
    corners = list(rectangle.exterior.coords)[:-1]
    if len(corners) != 4:
        return None
    start_index = min(range(4), key=lambda i: (corners[i][1], corners[i][0]))
    origin = corners[start_index]
    neighbours = [corners[(start_index - 1) % 4], corners[(start_index + 1) % 4]]
    vectors = [(x - origin[0], y - origin[1]) for x, y in neighbours]
    if any(abs(math.hypot(dx, dy) - cell_size_m) > 1.0 for dx, dy in vectors):
        return None
    if vectors[1][0] > vectors[0][0]:
        vectors.reverse()
    ux, uy = vectors[0]
    vx, vy = vectors[1]
    if ux * vy - uy * vx < 0:
        ux, uy, vx, vy = vx, vy, ux, uy
    return (float(origin[0]), float(origin[1]), float(ux), float(uy), float(vx), float(vy))


def _independent_parent_layout(grid: gpd.GeoDataFrame, grid_id_column: str, cell_size_m: float, log: Callable[[str], None] | None) -> MasterGridLayout | None:
    """Handle a transformed national grid whose local angle varies spatially."""
    geometries = tuple(grid.geometry)
    frames = tuple(_parent_affine_frame(g, cell_size_m) for g in geometries)
    if not frames or any(frame is None for frame in frames):
        return None
    if log is not None:
        log(f"Using independently verified nominal {cell_size_m:g} m parent frames for {len(frames):,} cells; no single national rigid lattice is assumed.")
    return MasterGridLayout(
        grid_id_column=grid_id_column, cell_size_m=float(cell_size_m),
        angle_radians=0.0, origin_u=0.0, origin_v=0.0,
        parent_ids=tuple(grid[grid_id_column].tolist()),
        parent_indices=tuple((i, 0) for i in range(len(grid))),
        local_geometries=geometries, parent_frames=frames,
    )


def inspect_master_grid(
    grid: gpd.GeoDataFrame,
    grid_id_column: str,
    cell_size_m: float = 250.0,
    log: Callable[[str], None] | None = None,
) -> MasterGridLayout:
    """Infer and validate a projected, square, affine-aligned master grid."""
    if grid.empty:
        raise ValueError("The 250 m master grid contains no cells.")
    if grid.crs is None:
        raise ValueError("The 250 m master grid has no CRS.")
    if grid_id_column not in grid.columns:
        raise ValueError(f"Master grid ID field '{grid_id_column}' was not found.")
    identifiers = grid[grid_id_column]
    if identifiers.isna().any() or identifiers.astype(str).duplicated().any():
        raise ValueError("Master grid ID values must be populated and unique.")
    if cell_size_m <= 0:
        raise ValueError("Master grid cell size must be positive.")

    crs = CRS.from_user_input(grid.crs)
    if not crs.is_projected:
        raise ValueError("The master grid CRS must be projected to define metre-based cells.")
    if not crs.axis_info or not all(
        math.isclose(axis.unit_conversion_factor, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for axis in crs.axis_info[:2]
    ):
        raise ValueError("The master grid CRS must use metres on both horizontal axes.")

    geometries = list(grid.geometry)
    valid_geometries = [
        geometry
        for geometry in geometries
        if geometry is not None and not geometry.is_empty
    ]
    invalid_ids = [
        str(identifiers.iloc[position])
        for position, geometry in enumerate(geometries)
        if geometry is not None and not geometry.is_empty and not geometry.is_valid
    ]
    unsupported_ids = [
        str(identifiers.iloc[position])
        for position, geometry in enumerate(geometries)
        if geometry is not None
        and not geometry.is_empty
        and geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ]
    local_geometries: list[BaseGeometry | None] = []
    complete_cells: list[tuple[int, float, float]] = []
    x_phases: list[dict[str, object]] = []
    y_phases: list[dict[str, object]] = []
    angle = 0.0
    if invalid_ids or unsupported_ids:
        details = _grid_diagnostics(
            grid,
            grid_id_column,
            cell_size_m,
            geometries,
            geometries,
            complete_cells,
            x_phases,
            y_phases,
        )
        reasons = []
        if invalid_ids:
            reasons.append(f"Invalid geometry IDs: {', '.join(invalid_ids[:20])}")
        if unsupported_ids:
            reasons.append(
                f"Non-polygon geometry IDs: {', '.join(unsupported_ids[:20])}"
            )
        _raise_grid_error(
            "\n".join(reasons)
            + "\nSelect a valid authoritative 250 m polygon master-grid layer.",
            details,
            log,
        )
    if valid_geometries:
        angle = _grid_angle(valid_geometries[0])
        local_geometries = [
            _local_transform(geometry, angle)
            if geometry is not None and not geometry.is_empty
            else None
            for geometry in geometries
        ]
        for position, geometry in enumerate(local_geometries):
            if geometry is None or not geometry.is_valid:
                continue
            min_u, min_v, max_u, max_v = geometry.bounds
            width, height = max_u - min_u, max_v - min_v
            if (
                abs(width - cell_size_m) <= ALIGNMENT_TOLERANCE_M
                and abs(height - cell_size_m) <= ALIGNMENT_TOLERANCE_M
                and abs(geometry.area - cell_size_m**2) <= AREA_TOLERANCE_M2
            ):
                expected = box(min_u, min_v, min_u + cell_size_m, min_v + cell_size_m)
                if geometry.symmetric_difference(expected).area <= AREA_TOLERANCE_M2:
                    complete_cells.append((position, min_u, min_v))

    def diagnostics() -> str:
        return _grid_diagnostics(
            grid,
            grid_id_column,
            cell_size_m,
            geometries,
            local_geometries,
            complete_cells,
            x_phases,
            y_phases,
        )

    if not complete_cells:
        independent_layout = _independent_parent_layout(grid, grid_id_column, cell_size_m, log)
        if independent_layout is not None:
            return independent_layout
        diagnostic_cells = [
            (position, geometry.bounds[0], geometry.bounds[1])
            for position, geometry in enumerate(local_geometries)
            if geometry is not None and not geometry.is_empty
        ]
        if diagnostic_cells:
            x_phases = _axis_lattice_phase(
                diagnostic_cells, 0, cell_size_m, tuple(identifiers.tolist())
            )[1]
            y_phases = _axis_lattice_phase(
                diagnostic_cells, 1, cell_size_m, tuple(identifiers.tolist())
            )[1]
        details = diagnostics()
        reasons = []
        reasons.append(
            "No complete 250 m x 250 m parent cell is available to establish "
            "the authoritative lattice."
        )
        _raise_grid_error(
            "\n".join(reasons) + (
                "\nSelect the original, unclipped authoritative 250 m master-grid layer."
            ),
            details,
            log,
        )

    origin_u_phase, x_phases = _axis_lattice_phase(
        complete_cells, 0, cell_size_m, tuple(identifiers.tolist())
    )
    origin_v_phase, y_phases = _axis_lattice_phase(
        complete_cells, 1, cell_size_m, tuple(identifiers.tolist())
    )
    phase_conflicts = []
    for axis, phases in (("X", x_phases), ("Y", y_phases)):
        if len(phases) < 2:
            continue
        total_support = len(complete_cells)
        dominant_support = int(phases[0]["count"])
        clear_majority = dominant_support / total_support >= PHASE_DOMINANCE_RATIO
        conflicts = [
            item
            for item in phases[1:]
            if int(item["count"])
            >= max(
                PHASE_CONFLICT_MIN_CELLS,
                math.ceil(total_support * (1.0 - PHASE_DOMINANCE_RATIO)),
            )
        ]
        if not clear_majority or conflicts:
            phase_conflicts.append(
                f"{axis} phases: "
                + "; ".join(
                    f"{float(item['phase']):.6f} m supported by "
                    f"{item['count']} complete cells "
                    f"(IDs: {', '.join(item['ids'][:10])})"
                    for item in phases
                )
            )
    if phase_conflicts:
        details = diagnostics()
        _raise_grid_error(
            "Cannot establish one 250 m grid alignment.\n"
            + "\n".join(phase_conflicts)
            + "\n"
            "The selected layer may contain cells from different grid origins. "
            "Select the original, unclipped authoritative 250 m master-grid layer.",
            details,
            log,
        )

    indices_by_position: dict[int, tuple[int, int]] = {}
    for position, min_u, min_v in complete_cells:
        col_float = (min_u - origin_u_phase) / cell_size_m
        row_float = (min_v - origin_v_phase) / cell_size_m
        column, row = round(col_float), round(row_float)
        if (
            abs(col_float - column) * cell_size_m > ALIGNMENT_TOLERANCE_M
            or abs(row_float - row) * cell_size_m > ALIGNMENT_TOLERANCE_M
        ):
            x_conflict = abs(col_float - column) * cell_size_m > ALIGNMENT_TOLERANCE_M
            axis = "X" if x_conflict else "Y"
            phase = min_u % cell_size_m if x_conflict else min_v % cell_size_m
            phases = x_phases if x_conflict else y_phases
            details = diagnostics()
            _raise_grid_error(
                "Cannot establish one 250 m grid alignment.\n"
                f"Conflicting {axis} phase: {phase:.6f} m for "
                f"GridID {identifiers.iloc[position]!r}.\n"
                + "; ".join(
                    f"{float(item['phase']):.6f} m supported by {item['count']} "
                    f"complete cells (IDs: {', '.join(item['ids'][:10])})"
                    for item in phases
                ),
                details,
                log,
            )
        indices_by_position[position] = (row, column)

    dominant_x = x_phases[0]
    dominant_y = y_phases[0]
    incompatible_complete = []
    for position, min_u, min_v in complete_cells:
        if (
            _phase_distance(min_u % cell_size_m, float(dominant_x["phase"]), cell_size_m)
            > ALIGNMENT_TOLERANCE_M
            or _phase_distance(min_v % cell_size_m, float(dominant_y["phase"]), cell_size_m)
            > ALIGNMENT_TOLERANCE_M
        ):
            incompatible_complete.append(str(identifiers.iloc[position]))
    if incompatible_complete:
        details = diagnostics()
        _raise_grid_error(
            "Complete parent cells do not share the dominant 250 m lattice. "
            "Conflicting GridIDs: " + ", ".join(incompatible_complete[:20]),
            details,
            log,
        )

    all_indices = list(indices_by_position.values())
    minimum_row = min(row for row, _ in all_indices)
    minimum_column = min(column for _, column in all_indices)
    origin_u = origin_u_phase + minimum_column * cell_size_m
    origin_v = origin_v_phase + minimum_row * cell_size_m
    raw_indices: list[tuple[int, int] | None] = [None] * len(geometries)
    parent_indices: list[tuple[int, int] | None] = [None] * len(geometries)
    for position, index in indices_by_position.items():
        raw_indices[position] = index
        parent_indices[position] = (
            index[0] - minimum_row,
            index[1] - minimum_column,
        )

    occupied = list(raw_indices[position] for position, _, _ in complete_cells)
    if len(set(occupied)) != len(occupied):
        details = diagnostics()
        _raise_grid_error(
            "The master grid contains overlapping or duplicate 250 m parent cells.",
            details,
            log,
        )

    partial_ids = [
        str(identifiers.iloc[position])
        for position, geometry in enumerate(local_geometries)
        if geometry is not None and position not in indices_by_position
    ]
    if partial_ids:
        details = diagnostics()
        _raise_grid_error(
            "The selected master-grid layer contains partial or non-250 m parent "
            "features that cannot be used as experimental 50 m parents. "
            "GridIDs: " + ", ".join(partial_ids[:30]) + (
                ". Select the original, unclipped authoritative 250 m master-grid layer."
            ),
            details,
            log,
        )

    canonical_local_geometries: list[BaseGeometry | None] = [None] * len(geometries)
    for position, index in enumerate(parent_indices):
        if index is not None:
            canonical_local_geometries[position] = box(
                origin_u + index[1] * cell_size_m,
                origin_v + index[0] * cell_size_m,
                origin_u + (index[1] + 1) * cell_size_m,
                origin_v + (index[0] + 1) * cell_size_m,
            )

    return MasterGridLayout(
        grid_id_column=grid_id_column,
        cell_size_m=float(cell_size_m),
        angle_radians=angle,
        origin_u=origin_u,
        origin_v=origin_v,
        parent_ids=tuple(identifiers.tolist()),
        parent_indices=tuple(parent_indices),
        local_geometries=tuple(canonical_local_geometries),
    )


def create_aligned_child_grid(
    master_grid: gpd.GeoDataFrame,
    layout: MasterGridLayout,
    child_size_m: float = 50.0,
) -> gpd.GeoDataFrame:
    """Refine each parent into exactly 25 children and mask to its geometry."""
    if not _close(layout.cell_size_m, child_size_m * CHILDREN_PER_AXIS):
        raise ValueError(
            f"Cannot refine {layout.cell_size_m:g} m cells into {child_size_m:g} m cells "
            f"at {CHILDREN_PER_AXIS} subdivisions per axis."
        )
    if child_size_m <= 0:
        raise ValueError("Child grid cell size must be positive.")
    if layout.grid_id_column not in master_grid.columns:
        raise ValueError(
            f"Master grid ID field '{layout.grid_id_column}' was not found."
        )
    identifiers = master_grid[layout.grid_id_column]
    if identifiers.isna().any() or identifiers.astype(str).duplicated().any():
        raise ValueError("Master grid ID values must be populated and unique.")

    layout_positions = {
        str(parent_id): position
        for position, parent_id in enumerate(layout.parent_ids)
    }
    try:
        parent_positions = [
            layout_positions[str(parent_id)] for parent_id in identifiers
        ]
    except KeyError as exc:
        raise ValueError(
            f"Working grid parent {exc.args[0]!r} is not part of the inspected "
            "authoritative master grid."
        ) from exc

    working_geometries = list(master_grid.geometry)
    incompatible_ids = []
    for row_position, layout_position in enumerate(parent_positions):
        geometry = working_geometries[row_position]
        expected = layout.local_geometries[layout_position]
        if geometry is None or geometry.is_empty or expected is None:
            incompatible_ids.append(str(identifiers.iloc[row_position]))
            continue
        if layout.parent_frames:
            if layout.parent_frames[layout_position] is None or not geometry.is_valid:
                incompatible_ids.append(str(identifiers.iloc[row_position]))
        else:
            local_geometry = _local_transform(geometry, layout.angle_radians)
            if (
                not local_geometry.is_valid
                or local_geometry.symmetric_difference(expected).area > AREA_TOLERANCE_M2
                or any(
                    not _close(actual, target)
                    for actual, target in zip(local_geometry.bounds, expected.bounds)
                )
            ):
                incompatible_ids.append(str(identifiers.iloc[row_position]))
    if incompatible_ids:
        raise ValueError(
            "Working parent geometries differ from the inspected authoritative "
            "250 m cells. Conflicting GridIDs: "
            + ", ".join(incompatible_ids[:20])
        )

    reserved_columns = {
        "ParentGridID", "ParentRow", "ParentColumn", "ChildRow", "ChildColumn",
        "GridRow50m", "GridColumn50m", "Parent250_ID", "Child50_ID",
    }
    conflicts = reserved_columns.intersection(master_grid.columns)
    if conflicts:
        raise ValueError(f"Master grid already contains reserved refinement fields: {sorted(conflicts)}")

    grid_id_column = layout.grid_id_column
    records = []
    minimum_row = min(
        index[0] for index in layout.parent_indices if index is not None
    )
    minimum_column = min(
        index[1] for index in layout.parent_indices if index is not None
    )
    cosine = math.cos(layout.angle_radians)
    sine = math.sin(layout.angle_radians)
    generated_count = 0

    for parent_position, (_, parent) in enumerate(master_grid.iterrows()):
        parent_id = parent[grid_id_column]
        layout_position = parent_positions[parent_position]
        parent_index = layout.parent_indices[layout_position]
        parent_geometry = layout.local_geometries[layout_position]
        parent_row, parent_column = (
            (parent_index[0] - minimum_row, parent_index[1] - minimum_column)
            if parent_index is not None
            else (-1, -1)
        )
        for child_row in range(CHILDREN_PER_AXIS):
            for child_column in range(CHILDREN_PER_AXIS):
                child_geometry = None
                if parent_geometry is not None and parent_index is not None:
                    if layout.parent_frames:
                        frame = layout.parent_frames[layout_position]
                        if frame is not None:
                            x0, y0, ux, uy, vx, vy = frame
                            a0, a1 = child_column / 5.0, (child_column + 1) / 5.0
                            b0, b1 = child_row / 5.0, (child_row + 1) / 5.0
                            child_rectangle = Polygon([
                                (x0 + a0*ux + b0*vx, y0 + a0*uy + b0*vy),
                                (x0 + a1*ux + b0*vx, y0 + a1*uy + b0*vy),
                                (x0 + a1*ux + b1*vx, y0 + a1*uy + b1*vy),
                                (x0 + a0*ux + b1*vx, y0 + a0*uy + b1*vy),
                            ])
                            clipped = child_rectangle.intersection(parent_geometry)
                            if not clipped.is_empty and clipped.area > AREA_TOLERANCE_M2:
                                child_geometry = clipped
                    else:
                        row_index, column_index = parent_index
                        left = layout.origin_u + column_index * layout.cell_size_m + child_column * child_size_m
                        bottom = layout.origin_v + row_index * layout.cell_size_m + child_row * child_size_m
                        child_rectangle = box(left, bottom, left + child_size_m, bottom + child_size_m)
                        clipped = child_rectangle.intersection(parent_geometry)
                        if not clipped.is_empty and clipped.area > AREA_TOLERANCE_M2:
                            child_geometry = affine_transform(clipped, [cosine, -sine, sine, cosine, 0.0, 0.0])
                generated_count += 1
                row_data = parent.drop(labels=[master_grid.geometry.name]).to_dict()
                row_data.update({
                    grid_id_column: f"{parent_id}_r{child_row + 1}_c{child_column + 1}",
                    "Child50_ID": f"{parent_id}_r{child_row + 1}_c{child_column + 1}",
                    "Parent250_ID": parent_id,
                    "ParentGridID": parent_id,
                    "ParentRow": parent_row,
                    "ParentColumn": parent_column,
                    "ChildRow": child_row,
                    "ChildColumn": child_column,
                    "GridRow50m": parent_row * CHILDREN_PER_AXIS + child_row,
                    "GridColumn50m": parent_column * CHILDREN_PER_AXIS + child_column,
                    master_grid.geometry.name: child_geometry,
                })
                records.append(row_data)

        if parent_geometry is not None:
            child_area = sum(
                record[master_grid.geometry.name].area
                for record in records[-CHILDREN_PER_AXIS ** 2:]
                if record[master_grid.geometry.name] is not None
            )
            if not math.isclose(
                child_area,
                parent_geometry.area,
                rel_tol=0.0,
                abs_tol=max(AREA_TOLERANCE_M2, parent_geometry.area * 1e-10),
            ):
                raise ValueError(
                    f"50 m children do not preserve the valid footprint of parent {parent_id!r}."
                )

    expected_count = len(master_grid) * CHILDREN_PER_AXIS ** 2
    if generated_count != expected_count:
        raise RuntimeError("Internal error: the child grid does not contain 25 records per parent.")

    children = gpd.GeoDataFrame(records, geometry=master_grid.geometry.name, crs=master_grid.crs)
    children["ParentRow"] = pd.array(children["ParentRow"], dtype="Int64")
    children["ParentColumn"] = pd.array(children["ParentColumn"], dtype="Int64")
    children["ChildRow"] = children["ChildRow"].astype("int64")
    children["ChildColumn"] = children["ChildColumn"].astype("int64")
    children["GridRow50m"] = children["GridRow50m"].astype("int64")
    children["GridColumn50m"] = children["GridColumn50m"].astype("int64")

    expected_parent_geometries = [
        (layout.local_geometries[position] if layout.parent_frames else _world_transform(layout.local_geometries[position], layout.angle_radians))
        for position in parent_positions
        if layout.local_geometries[position] is not None
    ]
    parent_bounds = gpd.GeoSeries(
        expected_parent_geometries, crs=master_grid.crs
    ).total_bounds
    valid_child_bounds = children.loc[children.geometry.notna(), children.geometry.name].total_bounds
    if len(valid_child_bounds) == 4 and any(
        not _close(float(parent_value), float(child_value), tolerance=ALIGNMENT_TOLERANCE_M)
        for parent_value, child_value in zip(parent_bounds, valid_child_bounds)
    ):
        raise ValueError("The refined grid extent does not match the valid master-grid extent.")

    valid_parents = children["ParentRow"].notna() & children["ParentColumn"].notna()
    if not (
        children.loc[valid_parents, "GridRow50m"]
        == children.loc[valid_parents, "ParentRow"] * CHILDREN_PER_AXIS
        + children.loc[valid_parents, "ChildRow"]
    ).all() or not (
        children.loc[valid_parents, "GridColumn50m"]
        == children.loc[valid_parents, "ParentColumn"] * CHILDREN_PER_AXIS
        + children.loc[valid_parents, "ChildColumn"]
    ).all():
        raise ValueError("The refined grid row or column alignment is inconsistent.")
    return children
