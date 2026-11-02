import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lowering_factor_validator as validator
from ems_database import EMSDatabase, EMSRecord


def _settings(root: Path, **overrides) -> validator.Settings:
    values = dict(
        difference_raster="d.tif",
        calibration_grid="g.gpkg",
        licence_layer="l.gpkg",
        ems_source=str(root / "ems"),
        output_folder=str(root / "out"),
        survey_start="01/05/2022",
        survey_end="13/10/2025",
    )
    values.update(overrides)
    return validator.Settings(**values)


def _grid(cells):
    return gpd.GeoDataFrame(
        {"GridID": [name for name, _ in cells]},
        geometry=[geometry for _, geometry in cells],
        crs="EPSG:27700",
    )


class LicenceSelectionTests(unittest.TestCase):
    def test_parses_commas_semicolons_and_blanks(self):
        settings = _settings(Path("."), licence_filter=" 137, 522;; 600 ")
        self.assertEqual(validator.selected_licences(settings), ["137", "522", "600"])
        self.assertEqual(validator.selected_licences(_settings(Path("."))), [])


class EMSCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.grid = _grid([("a", box(0, 0, 500, 500))])
        self.ems = gpd.GeoDataFrame(
            {"_datetime": [pd.Timestamp("2023-01-01")]},
            geometry=[Point(10, 10)],
            crs="EPSG:27700",
        )

    def test_second_run_reuses_saved_records(self):
        settings = _settings(self.root)
        messages = []
        with patch.object(validator, "load_ems", return_value=self.ems) as load:
            validator.load_ems_cached(settings, messages.append, self.grid)
            again = validator.load_ems_cached(settings, messages.append, self.grid)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(len(again), 1)
        self.assertTrue(any("Using saved EMS records" in m for m in messages))

    def test_changed_dates_or_wider_extent_or_untick_query_again(self):
        settings = _settings(self.root)
        with patch.object(validator, "load_ems", return_value=self.ems) as load:
            validator.load_ems_cached(settings, lambda _: None, self.grid)
            validator.load_ems_cached(
                _settings(self.root, survey_end="14/10/2025"), lambda _: None, self.grid
            )
            validator.load_ems_cached(
                settings, lambda _: None, _grid([("b", box(0, 0, 900, 900))])
            )
            validator.load_ems_cached(
                _settings(self.root, reuse_ems_cache=False), lambda _: None, self.grid
            )
        self.assertEqual(load.call_count, 4)

    def test_smaller_extent_reuses_records(self):
        settings = _settings(self.root)
        with patch.object(validator, "load_ems", return_value=self.ems) as load:
            validator.load_ems_cached(settings, lambda _: None, self.grid)
            validator.load_ems_cached(
                settings, lambda _: None, _grid([("c", box(100, 100, 200, 200))])
            )
        self.assertEqual(load.call_count, 1)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.licences = gpd.GeoDataFrame(
            {"LicenceNo": ["137", "999"]},
            geometry=[box(0, 0, 1000, 500), box(5000, 5000, 6000, 6000)],
            crs="EPSG:27700",
        )

    def test_reports_overlaps_empty_licences_and_raster_sign(self):
        grid = _grid([
            ("137_1", box(0, 0, 250, 250)),
            ("522_1", box(100, 0, 350, 250)),
            ("137_2", box(500, 0, 750, 250)),
        ])
        raster = {"valid": 100, "positive": 5, "negative": 95, "positive_values_are_lowering": True}
        warnings = validator.preflight_checks(
            _settings(self.root), grid, self.licences, "GridID", raster, lambda _: None
        )
        text = "\n".join(warnings)
        self.assertIn("1 pairs of overlapping grid cells belong to the same licence (137", text)
        self.assertIn("137_1/522_1", text)
        self.assertIn("have no grid cells", text)
        self.assertIn("999", text)
        self.assertIn("Only 5% of raster pixels", text)

    def test_overlaps_outside_licences_or_across_boundaries_do_not_warn(self):
        licences = gpd.GeoDataFrame(
            {"LicenceNo": ["137", "522"]},
            geometry=[box(0, 0, 500, 250), box(500, 0, 1000, 250)],
            crs="EPSG:27700",
        )
        grid = _grid([
            ("137_1", box(250, 0, 500, 250)),    # in 137
            ("522_1", box(400, 0, 650, 250)),    # in 522, overlaps 137_1 across the boundary
            ("out_a", box(2000, 0, 2250, 250)),  # outside every licence
            ("out_b", box(2100, 0, 2350, 250)),  # outside, overlaps out_a
        ])
        raster = {"valid": 10, "positive": 9, "negative": 1, "positive_values_are_lowering": True}
        messages = []
        warnings = validator.preflight_checks(
            _settings(self.root), grid, licences, "GridID", raster, messages.append
        )
        self.assertEqual(warnings, [])
        text = "\n".join(messages)
        self.assertIn("1 overlapping cell pairs straddle a boundary", text)
        self.assertIn("1 overlapping cell pairs involve cells outside every licence", text)

    def test_clean_inputs_pass(self):
        grid = _grid([("a", box(0, 0, 250, 250)), ("b", box(250, 0, 500, 250))])
        licences = self.licences.iloc[[0]]
        raster = {"valid": 10, "positive": 9, "negative": 1, "positive_values_are_lowering": True}
        messages = []
        warnings = validator.preflight_checks(
            _settings(self.root), grid, licences, "GridID", raster, messages.append
        )
        self.assertEqual(warnings, [])
        self.assertIn("Pre-run checks passed.", messages)

    def test_blank_or_reversed_dates_are_reported(self):
        grid = _grid([("a", box(0, 0, 250, 250))])
        licences = self.licences.iloc[[0]]
        blank = validator.preflight_checks(
            _settings(self.root, survey_start=""), grid, licences, "GridID", {}, lambda _: None
        )
        reversed_dates = validator.preflight_checks(
            _settings(self.root, survey_start="01/01/2026"), grid, licences, "GridID", {}, lambda _: None
        )
        self.assertIn("blank", blank[0])
        self.assertIn("is not before", reversed_dates[0])


class MonthsWithoutRecordsTests(unittest.TestCase):
    def test_lists_months_with_no_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "tracks"
            root.mkdir()
            source = root / "ship.txt"
            source.write_text("x", encoding="utf-8")
            database = EMSDatabase(root, Path(folder) / "index.sqlite3")
            records = [
                EMSRecord("2022-05-10 00:00:00.000000", 1.0, 51.0, "3", "ship", "EPSG:4326"),
                EMSRecord("2022-07-01 00:00:00.000000", 1.0, 51.0, "3", "ship", "EPSG:4326"),
            ]
            database.sync_files([source], "p", lambda _: iter(records))
            missing = database.months_without_records(
                "p", datetime(2022, 5, 1), datetime(2022, 8, 15)
            )
        self.assertEqual(missing, ["2022-06", "2022-08"])


class RunStageTests(unittest.TestCase):
    def test_maps_log_messages_to_stages(self):
        self.assertEqual(
            validator.run_stage("Starting indexed EMS query..."),
            "Querying EMS records (slowest step)...",
        )
        self.assertEqual(
            validator.run_stage("Running like-for-like validation at 250 m Regulatory..."),
            "Validating at 250 m Regulatory...",
        )
        self.assertIsNone(validator.run_stage("Some other message"))


class CancelAndConfirmTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.grid = _grid([("a", box(0, 0, 250, 250)), ("b", box(0, 0, 250, 250))])
        self.licences = gpd.GeoDataFrame(
            {"LicenceNo": ["L1", "L2"]},
            geometry=[box(0, 0, 250, 250), box(1000, 1000, 1250, 1250)],
            crs="EPSG:27700",
        )

    def _patches(self, reads):
        return (
            patch.object(validator, "prepare_difference_raster", return_value="d.tif"),
            patch.object(validator, "validate_bathymetric_difference_raster", return_value={}),
            patch.object(validator, "read_vector", side_effect=reads),
            patch.object(validator, "load_ems"),
        )

    def test_declining_preflight_stops_before_ems(self):
        settings = _settings(self.root, analysis_extent="TRACK_FOOTPRINT")
        p1, p2, p3, p4 = self._patches([self.grid, self.licences])
        with p1, p2, p3, p4 as load_ems:
            with self.assertRaises(validator.RunCancelled):
                validator.run_validation(settings, confirm=lambda warnings: False)
        load_ems.assert_not_called()

    def test_cancel_event_stops_run(self):
        settings = _settings(self.root, analysis_extent="TRACK_FOOTPRINT")
        cancel = threading.Event()
        cancel.set()
        p1, p2, p3, p4 = self._patches([self.grid, self.licences])
        with p1, p2, p3, p4:
            with self.assertRaises(validator.RunCancelled):
                validator.run_validation(settings, cancel_event=cancel)

    def test_licence_filter_drops_other_licences_and_their_cells(self):
        # A blank start date guarantees a pre-run warning, so declining stops the run there.
        settings = _settings(
            self.root, analysis_extent="TRACK_FOOTPRINT", licence_filter="L1, L9",
            survey_start="",
        )
        grid = _grid([("a", box(0, 0, 250, 250)), ("far", box(1000, 1000, 1250, 1250))])
        messages = []
        p1, p2, p3, p4 = self._patches([grid, self.licences])
        with p1, p2, p3, p4:
            with self.assertRaises(validator.RunCancelled):
                validator.run_validation(settings, messages.append, confirm=lambda w: False)
        text = "\n".join(messages)
        self.assertIn("Running 1 selected licence(s) (L1) with 1 grid cells.", text)
        self.assertIn("selected licences not found and skipped: L9", text)



class RunLogTests(unittest.TestCase):
    def test_cancelled_run_still_writes_run_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = _settings(root, analysis_extent="TRACK_FOOTPRINT", survey_start="")
            grid = _grid([("a", box(0, 0, 250, 250))])
            licences = gpd.GeoDataFrame(
                {"LicenceNo": ["L1"]}, geometry=[box(0, 0, 250, 250)], crs="EPSG:27700"
            )
            with (
                patch.object(validator, "prepare_difference_raster", return_value="d.tif"),
                patch.object(validator, "validate_bathymetric_difference_raster", return_value={}),
                patch.object(validator, "read_vector", side_effect=[grid, licences]),
                patch.object(validator, "load_ems"),
            ):
                with self.assertRaises(validator.RunCancelled):
                    validator.run_validation(settings, confirm=lambda w: False)
            text = (root / "out" / "run_log.txt").read_text(encoding="utf-8")
        self.assertIn("Outcome:  CANCELLED: Run stopped after pre-run checks.", text)
        self.assertIn("PRE-RUN CHECK: Survey start or end date is blank", text)
        self.assertIn('"licence_filter": ""', text)


class LowCoverageWarningTests(unittest.TestCase):
    """Two 250 m cells on a 125 m raster; NoData pixels lower each cell's coverage."""

    def _run(self, values):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        proj = str(Path(rasterio.__file__).parent / "proj_data")
        raster = root / "d.tif"
        with patch.dict("os.environ", {"PROJ_LIB": proj}):
            with rasterio.open(
                raster, "w", driver="GTiff", width=4, height=2, count=1, dtype="float32",
                crs="EPSG:27700", transform=from_origin(0, 250, 125, 125), nodata=-9999.0,
            ) as dataset:
                dataset.write(np.array(values, dtype=np.float32), 1)
            grid = _grid([("A", box(0, 0, 250, 250)), ("B", box(250, 0, 500, 250))])
            licences = gpd.GeoDataFrame(
                {"LicenceNo": ["L1"]}, geometry=[box(0, 0, 500, 250)], crs="EPSG:27700"
            )
            ems = gpd.GeoDataFrame(
                {"_datetime": [pd.Timestamp("2024-01-01")] * 90},
                geometry=[Point(125, 125)] * 60 + [Point(375, 125)] * 30,
                crs="EPSG:27700",
            )
            settings = validator.Settings(
                difference_raster=str(raster), calibration_grid="g", licence_layer="l",
                ems_source="e", output_folder=str(root / "out"),
                background_correction="NONE", maximum_rate_tph=1e9,
            )
            messages = []
            validator._run_validation_single(
                settings, messages.append, grid_override=grid, licences_override=licences,
                ems_override=ems, difference_raster_override=str(raster),
            )
        return "\n".join(messages)

    def test_partly_covered_licence_names_coverage_as_cause(self):
        n = -9999.0
        log = self._run([[n, n, 0.5, 0.5], [n, 0.5, 0.5, 0.5]])  # A 25%, B 100%
        self.assertIn("66.7% of EMS hours were excluded: 1.0 h in cells the difference raster covers less than 80%", log)

    def test_uncovered_licence_says_no_rate_from_this_survey(self):
        n = -9999.0
        log = self._run([[n, n, n, n], [n, 0.5, 0.5, n]])  # A 25%, B 25%
        self.assertIn("no rate - the difference raster covers too little of the dredged cells", log)



class ModelWriteBackTargetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.model = self.root / "Lowering_Model.gpkg"
        gpd.GeoDataFrame(
            {
                "area_id": ["137", "522"],
                "area_name": ["Area 137", "Area 522"],
                "lowering_factor": [None, 0.02],
                "active": [1, 1],
            },
            geometry=[box(0, 0, 250, 250), box(250, 0, 500, 250)],
            crs="EPSG:27700",
        ).to_file(self.model, layer="areas", driver="GPKG")

    def test_target_is_model_when_set_and_licence_layer_otherwise(self):
        with_model = _settings(self.root, model_path=str(self.model))
        self.assertEqual(
            validator.writeback_target(with_model), (str(self.model), "areas", "area_id")
        )
        without = _settings(self.root, licence_layer="ADZ.gpkg", licence_layer_name="Query2")
        self.assertEqual(
            validator.writeback_target(without), ("ADZ.gpkg", "Query2", "LicenceNo")
        )

    def test_missing_model_file_is_reported_before_running(self):
        settings = _settings(
            self.root, write_model_factors=True, model_path=str(self.root / "nope.gpkg")
        )
        with self.assertRaisesRegex(ValueError, "Model to update was not found"):
            settings.validate()

    def test_run_fills_blank_model_factor_and_keeps_existing_one(self):
        settings = _settings(
            self.root, analysis_extent="TRACK_FOOTPRINT", write_model_factors=True,
            model_path=str(self.model), licence_layer="ADZ.gpkg",
        )
        grid = _grid([("a", box(0, 0, 250, 250))])
        licences = gpd.GeoDataFrame(
            {"LicenceNo": ["137", "522"]},
            geometry=[box(0, 0, 250, 250), box(250, 0, 500, 250)],
            crs="EPSG:27700",
        )
        summary = pd.DataFrame({
            "LicenceNo": ["137", "522"],
            "ValidatedRate_m3ph": [1_057.0, 900.0],
            "ValidatedRate_tph": [1_797.0, 1_530.0],
            "QA_Status": ["RED", "RED"],
        })
        empty_ems = gpd.GeoDataFrame(
            {"_datetime": []}, geometry=gpd.GeoSeries([], crs="EPSG:27700"), crs="EPSG:27700"
        )
        messages = []
        with (
            patch.object(validator, "prepare_difference_raster", return_value="d.tif"),
            patch.object(validator, "validate_bathymetric_difference_raster", return_value={}),
            patch.object(validator, "read_vector", side_effect=[grid, licences]),
            patch.object(validator, "load_ems", return_value=empty_ems),
            patch.object(validator, "_run_validation_single", return_value=summary),
        ):
            validator.run_validation(settings, messages.append)

        areas = gpd.read_file(self.model, layer="areas").set_index("area_id")
        self.assertAlmostEqual(areas.loc["137", "lowering_factor"], 1_057.0 / 62_500)
        self.assertAlmostEqual(areas.loc["522", "lowering_factor"], 0.02)  # not overwritten
        text = "\n".join(messages)
        self.assertIn("Write-back target:", text)
        self.assertIn("matching LicenceNo to area_id", text)
        self.assertIn("Updated 1 blank lowering_factor value(s)", text)


if __name__ == "__main__":
    unittest.main()
