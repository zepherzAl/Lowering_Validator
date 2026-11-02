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
        self.assertIn("1 pairs of grid cells overlap", text)
        self.assertIn("137_1/522_1", text)
        self.assertIn("have no grid cells", text)
        self.assertIn("999", text)
        self.assertIn("Only 5% of raster pixels", text)

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


if __name__ == "__main__":
    unittest.main()
