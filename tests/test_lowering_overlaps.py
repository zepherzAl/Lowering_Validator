import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, box
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lowering_model
lowerer = importlib.import_module("Predicted_Seabed_lowering_tool_V8")

# Area A covers x 0-1000, area B covers x 500-1500; they share x 500-1000.
AREAS = {"A": box(0, 0, 1000, 250), "B": box(500, 0, 1500, 250)}
FACTOR = 0.1  # m per hour of dredging in a 250 m cell


def _write_model(path: Path, factor: float = FACTOR) -> None:
    areas = gpd.GeoDataFrame(
        {
            "area_id": list(AREAS),
            "area_name": [f"Area {key}" for key in AREAS],
            "lowering_factor": [factor, factor],
            "active": [1, 1],
        },
        geometry=list(AREAS.values()),
        crs="EPSG:27700",
    )
    cells = []
    for key, (x0, x1) in {"A": (0, 1000), "B": (500, 1500)}.items():
        for index, x in enumerate(range(x0, x1, 250), start=1):
            cells.append((f"{key}_{index:02d}", key, box(x, 0, x + 250, 250)))
    grid = gpd.GeoDataFrame(
        {"grid_id": [c[0] for c in cells], "area_id": [c[1] for c in cells]},
        geometry=[c[2] for c in cells],
        crs="EPSG:27700",
    )
    areas.to_file(path, layer="areas", driver="GPKG")
    grid.to_file(path, layer="grid_250m", driver="GPKG")


def _points(*groups):
    """(x, y, count) in EPSG:27700 -> sampled EMS frame in WGS84, one per minute."""
    to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    rows, minute = [], 0
    for x, y, count in groups:
        lon, lat = to_wgs.transform(x, y)
        for _ in range(count):
            rows.append({
                "source_file": "ship.txt",
                "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute),
                "GPSLAT": lat, "GPSLON": lon, "DREDGE": 3.0,
            })
            minute += 1
    return pd.DataFrame(rows)


class AssignPointsTests(unittest.TestCase):
    def test_overlap_points_are_shared_not_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.gpkg"
            _write_model(path)
            model = lowering_model.load_lowering_model(path)
        validation = lowering_model.validate_lowering_model(model)
        points = gpd.GeoDataFrame(
            geometry=[Point(100, 100), Point(700, 100), Point(1400, 100), Point(3000, 100)],
            crs="EPSG:27700",
        )
        assigned = lowering_model.assign_points_to_areas(points, model, validation)
        self.assertEqual(
            assigned["area_assignment"].tolist(), ["matched", "shared", "matched", "unmatched"]
        )
        self.assertEqual(json.loads(assigned["area_candidates"].iloc[1]), ["A", "B"])
        self.assertEqual(len(lowering_model.points_for_area(assigned, "A")), 2)
        self.assertEqual(len(lowering_model.points_for_area(assigned, "B")), 2)
        self.assertIn("counted in every area", " ".join(validation.warnings))


class WorkflowOverlapTests(unittest.TestCase):
    def test_shared_dredging_lowers_both_areas_and_is_flagged(self):
        # 60 min only in A, 30 min in the shared strip, 12 min only in B.
        sampled = _points((125, 125, 60), (625, 125, 30), (1125, 125, 12))
        tracks = (sampled, sampled, len(sampled), len(sampled), len(sampled), len(sampled), 1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_path = root / "model.gpkg"
            _write_model(model_path)
            messages = []
            with patch.object(lowerer, "process_all_tracks", return_value=tracks):
                outputs = lowerer.run_lowering_workflow(
                    input_folder=root, model_path=model_path, output_folder=root,
                    dredge_value=3, dredge_mode="equals", sample_minutes=1,
                    use_date_filter=False, start_date=None, end_date=None,
                    spatial_predicate="within", export_cleaned_csv=False,
                    export_points_gpkg=False, export_grid_gpkg=False,
                    export_summary_csv=True, log_callback=messages.append,
                )
            cells = pd.read_csv(next(p for p in outputs if p.name == "grid_dredge_time_lowering_summary.csv"))
            summary = pd.read_csv(next(p for p in outputs if p.name == "multi_area_processing_summary.csv"))

        by_cell = cells.set_index("grid_id")
        self.assertAlmostEqual(by_cell.loc["A_01", "lowering_m"], 0.10)  # 1 h in A only
        self.assertAlmostEqual(by_cell.loc["A_03", "lowering_m"], 0.05)  # 0.5 h shared
        self.assertAlmostEqual(by_cell.loc["B_01", "lowering_m"], 0.05)  # same seabed, area B
        self.assertAlmostEqual(by_cell.loc["B_03", "lowering_m"], 0.02)  # 0.2 h in B only
        self.assertEqual(by_cell.loc["A_03", "shared_point_count"], 30)
        self.assertEqual(by_cell.loc["A_01", "shared_point_count"], 0)

        areas = summary.loc[summary["summary_scope"].eq("area")].set_index("area_id")
        self.assertEqual(areas.loc["A", "ems_records_tracks_assigned"], 90)
        self.assertEqual(areas.loc["B", "ems_records_tracks_assigned"], 42)
        self.assertEqual(areas.loc["A", "shared_overlap_records"], 30)
        run = summary.loc[summary["summary_scope"].eq("run")].iloc[0]
        self.assertEqual(run["shared_overlap_records"], 30)
        self.assertTrue(any("double-counts" in message for message in messages))



class SameNamedVesselFileTests(unittest.TestCase):
    """2022 archive: each vessel's file is 19-01-2022.txt inside its own folder."""

    def test_vessels_dredging_at_the_same_time_are_both_counted(self):
        from ems_database import EMSDatabase

        header = "DATE,TIME,GPSLAT,GPSLON,DREDGE\n"
        rows = "".join(
            f"01/06/2022,10:{minute:02d}:{second:02d},50.59,-1.55,3\n"
            for minute in range(5)
            for second in (0, 10, 20, 30, 40, 50)
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "Tracks"
            for vessel in ("232003506 - (Welsh Piper)", "232215000 - (Sand Falcon)"):
                path = root / "2022" / "1" / vessel / "19-01-2022.txt"
                path.parent.mkdir(parents=True)
                path.write_text(header + rows, encoding="utf-8")
            database_path = Path(folder) / "index.sqlite3"
            with patch.object(
                lowerer, "EMSDatabase", lambda source: EMSDatabase(source, database_path)
            ):
                combined, *_ = lowerer.process_all_tracks(
                    input_folder=root, dredge_value=3, dredge_mode="equals", sample_minutes=1
                )
        # 5 minutes x 2 vessels; merging by file name would give only 5.
        self.assertEqual(len(combined), 10)
        self.assertEqual(
            sorted(combined["vessel"].unique()),
            ["232003506 - (Welsh Piper)", "232215000 - (Sand Falcon)"],
        )



class SmallLoweringIsKeptTests(unittest.TestCase):
    """10 minutes a quarter in a 250 m cell at 0.0169 m/h used to round to 0.00 m."""

    def test_light_dredging_accumulates_over_quarters(self):
        factor = 0.0169
        sampled = _points((125, 125, 10))
        tracks = (sampled, sampled, 10, 10, 10, 10, 1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_path = root / "model.gpkg"
            _write_model(model_path, factor)
            with patch.object(lowerer, "process_all_tracks", return_value=tracks):
                outputs = lowerer.run_lowering_workflow(
                    input_folder=root, model_path=model_path, output_folder=root,
                    dredge_value=3, dredge_mode="equals", sample_minutes=1,
                    use_date_filter=False, start_date=None, end_date=None,
                    spatial_predicate="within", export_cleaned_csv=False,
                    export_points_gpkg=False, export_grid_gpkg=True,
                    export_summary_csv=False, processing_mode="Selected area",
                    selected_area_id="A",
                )
                quarter_path = next(p for p in outputs if p.suffix == ".gpkg")
                quarter = gpd.read_file(quarter_path)
                cumulative = lowerer.update_multi_area_cumulative_grid(quarter_path, quarter)

        cell = quarter.set_index("grid_id").loc["A_01"]
        expected = 10 / 60 * factor  # 0.0028167 m
        self.assertAlmostEqual(cell["lowering_m"], expected, places=6)
        self.assertAlmostEqual(cell["volume_m3"], 10 / 60 * factor * 62_500, places=4)
        two_quarters = cumulative.set_index("grid_id").loc["A_01", "lowering_m"]
        self.assertAlmostEqual(two_quarters, 2 * expected, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
